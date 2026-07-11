import logging
import os
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Type,
    Union,
    cast,
)

from dbgpt.core import MessageConverter, ModelMetadata, ModelOutput, ModelRequest
from dbgpt.core.awel.flow import (
    TAGS_ORDER_HIGH,
    ResourceCategory,
    auto_register_resource,
)
from dbgpt.model.proxy.base import (
    AsyncGenerateStreamFunction,
    GenerateStreamFunction,
    ProxyLLMClient,
    ProxyTokenizer,
    TiktokenProxyTokenizer,
    register_proxy_model_adapter,
)
from dbgpt.model.proxy.llms.chatgpt import OpenAICompatibleDeployModelParameters
from dbgpt.model.proxy.llms.proxy_model import ProxyModel, parse_model_request
from dbgpt.util.i18n_utils import _

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic, ProxiesTypes

logger = logging.getLogger(__name__)


@auto_register_resource(
    label=_("Claude Proxy LLM"),
    category=ResourceCategory.LLM_CLIENT,
    tags={"order": TAGS_ORDER_HIGH},
    description=_("Claude Proxy LLM"),
    documentation_url="https://docs.anthropic.com/en/api/getting-started",
    show_in_ui=False,
)
@dataclass
class ClaudeDeployModelParameters(OpenAICompatibleDeployModelParameters):
    """Deploy model parameters for Claude."""

    provider: str = "proxy/claude"

    api_base: Optional[str] = field(
        default="${env:ANTHROPIC_BASE_URL:-https://api.anthropic.com}",
        metadata={
            "help": _("The base url of the claude API."),
        },
    )

    api_key: Optional[str] = field(
        default="${env:ANTHROPIC_API_KEY}",
        metadata={
            "help": _("The API key of the claude API."),
            "tags": "privacy",
        },
    )


async def claude_generate_stream(
    model: ProxyModel,
    tokenizer: Any,
    params: Dict[str, Any],
    device: str,
    context_len=2048,
) -> AsyncIterator[ModelOutput]:
    client: ClaudeLLMClient = cast(ClaudeLLMClient, model.proxy_llm_client)
    stream = _request_stream_enabled(params)
    request = parse_model_request(params, client.default_model, stream=stream)
    if not stream:
        yield await client.generate(request)
        return
    async for r in client.generate_stream(request):
        yield r


def _request_stream_enabled(params: Dict[str, Any]) -> bool:
    context = params.get("context")
    if isinstance(context, dict) and "stream" in context:
        return bool(context.get("stream"))
    stream = params.get("stream")
    if stream is not None:
        return bool(stream)
    return True


class ClaudeLLMClient(ProxyLLMClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        proxies: Optional["ProxiesTypes"] = None,
        timeout: Optional[int] = 240,
        model_alias: Optional[str] = "claude-3-5-sonnet-20241022",
        context_length: Optional[int] = 8192,
        client: Optional["AsyncAnthropic"] = None,
        claude_kwargs: Optional[Dict[str, Any]] = None,
        proxy_tokenizer: Optional[ProxyTokenizer] = None,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "Could not import python package: anthropic "
                "Please install anthropic by command `pip install anthropic"
            ) from exc
        if not model:
            model = "claude-3-5-sonnet-20241022"
        self._client = client
        self._model = model
        self._api_key = self._resolve_env_vars(api_key)
        self._api_base = api_base or os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        )
        self._api_base = self._resolve_env_vars(self._api_base)
        self._proxies = proxies
        self._timeout = timeout
        self._claude_kwargs = claude_kwargs or {}
        self._model_alias = model_alias
        self._proxy_tokenizer = proxy_tokenizer

        super().__init__(
            model_names=[model_alias],
            context_length=context_length,
            proxy_tokenizer=proxy_tokenizer,
        )

    @classmethod
    def new_client(
        cls,
        model_params: ClaudeDeployModelParameters,
        default_executor: Optional[Executor] = None,
    ) -> "ClaudeLLMClient":
        return cls(
            api_key=model_params.api_key,
            api_base=model_params.api_base,
            model=model_params.real_provider_model_name,
            proxies=model_params.http_proxy,
            model_alias=model_params.real_provider_model_name,
            context_length=max(model_params.context_length or 8192, 8192),
        )

    @classmethod
    def param_class(cls) -> Type[ClaudeDeployModelParameters]:
        """Get the model parameters class."""
        return ClaudeDeployModelParameters

    @classmethod
    def generate_stream_function(
        cls,
    ) -> Optional[Union[GenerateStreamFunction, AsyncGenerateStreamFunction]]:
        """Get generate stream function.

        Returns:
            Optional[Union[GenerateStreamFunction, AsyncGenerateStreamFunction]]:
                generate stream function
        """
        return claude_generate_stream

    @property
    def client(self) -> "AsyncAnthropic":
        from anthropic import AsyncAnthropic

        if self._client is None:
            kwargs = dict(
                api_key=self._api_key,
                base_url=self._api_base,
                timeout=self._timeout,
            )
            if self._proxies:
                kwargs["proxies"] = self._proxies
            try:
                self._client = AsyncAnthropic(**kwargs)
            except TypeError as exc:
                if self._proxies and "proxies" in str(exc):
                    kwargs.pop("proxies", None)
                    self._client = AsyncAnthropic(**kwargs)
                else:
                    raise
        return self._client

    @property
    def proxy_tokenizer(self) -> ProxyTokenizer:
        if not self._proxy_tokenizer:
            self._proxy_tokenizer = ClaudeProxyTokenizer(self.client)
        return self._proxy_tokenizer

    @property
    def default_model(self) -> str:
        """Default model name"""
        model = self._model
        if not model:
            model = "claude-3-5-sonnet-20241022"
        return model

    def _build_request(
        self, request: ModelRequest, stream: Optional[bool] = False
    ) -> Dict[str, Any]:
        payload = {"stream": stream}
        model = request.model or self.default_model
        payload["model"] = model
        # Apply claude kwargs
        for k, v in self._claude_kwargs.items():
            payload[k] = v
        if request.temperature:
            payload["temperature"] = request.temperature
        if request.max_new_tokens:
            payload["max_tokens"] = request.max_new_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.top_p:
            payload["top_p"] = request.top_p
        return payload

    async def generate(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> ModelOutput:
        request = self.local_covert_message(request, message_converter)
        messages, system_messages = request.split_messages()
        messages = _inline_system_messages(messages, system_messages)
        payload = self._build_request(request)
        logger.info(
            f"Send request to claude, payload: {payload}\n\n messages:\n{messages}"
        )
        try:
            if "max_tokens" not in payload:
                max_tokens = 1024
            else:
                max_tokens = payload["max_tokens"]
                del payload["max_tokens"]
            response = await self.client.messages.create(
                max_tokens=max_tokens,
                messages=messages,
                **payload,
            )
            usage = None
            finish_reason = response.stop_reason
            if response.usage:
                usage = _anthropic_usage(response.usage)
            response_content = response.content
            if not response_content:
                raise ValueError("Response content is empty")
            return ModelOutput(
                text=response_content[0].text,
                error_code=0,
                finish_reason=finish_reason,
                usage=usage,
            )
        except Exception as e:
            return ModelOutput(
                text=f"**Claude Generate Error, Please CheckErrorInfo.**: {e}",
                error_code=1,
            )

    async def generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> AsyncIterator[ModelOutput]:
        request = self.local_covert_message(request, message_converter)
        messages, system_messages = request.split_messages()
        messages = _inline_system_messages(messages, system_messages)
        payload = self._build_request(request, stream=True)
        logger.info(
            f"Send request to claude, payload: {payload}\n\n messages:\n{messages}"
        )
        try:
            if "max_tokens" not in payload:
                max_tokens = 1024
            else:
                max_tokens = payload["max_tokens"]
                del payload["max_tokens"]
            if "stream" in payload:
                del payload["stream"]
            full_text = ""
            async with self.client.messages.stream(
                max_tokens=max_tokens,
                messages=messages,
                **payload,
            ) as stream:
                async for text in stream.text_stream:
                    full_text += text
                    yield ModelOutput(text=full_text, error_code=0)
                final_message = await stream.get_final_message()
                yield ModelOutput(
                    text=full_text,
                    error_code=0,
                    usage=_anthropic_usage(final_message.usage),
                )
        except Exception as e:
            yield ModelOutput(
                text=f"**Claude Generate Stream Error, Please CheckErrorInfo.**: {e}",
                error_code=1,
            )

    async def models(self) -> List[ModelMetadata]:
        model_metadata = ModelMetadata(
            model=self._model_alias,
            context_length=await self.get_context_length(),
        )
        return [model_metadata]

    async def get_context_length(self) -> int:
        """Get the context length of the model.

        Returns:
            int: The context length.
        # TODO: This is a temporary solution. We should have a better way to get the
            context length.
            eg. get real context length from the openai api.
        """
        return self.context_length


def _anthropic_usage(raw_usage: Any) -> Dict[str, int]:
    prompt_tokens = int(getattr(raw_usage, "input_tokens", 0) or 0)
    completion_tokens = int(getattr(raw_usage, "output_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _inline_system_messages(
    messages: List[Dict[str, Any]], system_messages: List[str]
) -> List[Dict[str, Any]]:
    if not system_messages:
        return messages
    system_text = "\n\n".join(message for message in system_messages if message)
    if not system_text:
        return messages
    if not messages:
        return [
            {
                "role": "user",
                "content": _format_inlined_system(system_text, ""),
            }
        ]

    inlined = [dict(message) for message in messages]
    first_message = inlined[0]
    if first_message.get("role") == "user":
        first_content = first_message.get("content") or ""
        first_message["content"] = _format_inlined_system(system_text, first_content)
        return inlined
    return [
        {
            "role": "user",
            "content": _format_inlined_system(system_text, ""),
        },
        *inlined,
    ]


def _format_inlined_system(system_text: str, user_text: str) -> str:
    if user_text:
        return (
            "System instructions (follow silently; do not summarize or restate):\n"
            f"{system_text}\n\n"
            "User request:\n"
            f"{user_text}"
        )
    return (
        "System instructions (follow silently; do not summarize or restate):\n"
        f"{system_text}"
    )


class ClaudeProxyTokenizer(ProxyTokenizer):
    def __init__(self, client: "AsyncAnthropic", concurrency_limit: int = 10):
        self.client = client
        self.concurrency_limit = concurrency_limit
        self._tiktoken_tokenizer = TiktokenProxyTokenizer()

    def count_token(self, model_name: str, prompts: List[str]) -> List[int]:
        # Use tiktoken to count token in local environment
        return self._tiktoken_tokenizer.count_token(model_name, prompts)

    def support_async(self) -> bool:
        return True

    async def count_token_async(self, model_name: str, prompts: List[str]) -> List[int]:
        """Count token of given messages.

        This is relying on the claude beta API, which is not available for some users.
        """
        from dbgpt.util.chat_util import run_async_tasks

        tasks = []
        model_name = model_name or "claude-3-5-sonnet-20241022"
        for prompt in prompts:
            request = ModelRequest(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            tasks.append(
                self.client.beta.messages.count_tokens(
                    model=model_name,
                    messages=request.messages,
                )
            )
        try:
            results = await run_async_tasks(tasks, self.concurrency_limit)
        except Exception:
            logger.warning(
                "Claude beta token counting failed; falling back to local tokenizer",
                exc_info=True,
            )
            return self.count_token(model_name, prompts)
        return [_token_count_value(result) for result in results]


def _token_count_value(result: Any) -> int:
    if isinstance(result, int):
        return result
    if isinstance(result, dict):
        value = result.get("input_tokens")
        if value is None:
            value = result.get("tokens")
        return int(value or 0)
    value = getattr(result, "input_tokens", None)
    if value is not None:
        return int(value)
    return int(result)


register_proxy_model_adapter(
    ClaudeLLMClient,
    supported_models=[
        ModelMetadata(
            model=[
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-20241022",
                "claude-3-5-haiku-latest",
            ],
            context_length=200 * 1024,
            max_output_length=8 * 1024,
            description="Claude 3.5 by Anthropic",
            link="https://docs.anthropic.com/en/docs/about-claude/models#model-names",
            function_calling=True,
        ),
        ModelMetadata(
            model=[
                "claude-3-opus-20240229",
                "claude-3-opus-latest",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            context_length=200 * 1024,
            max_output_length=4 * 1024,
            description="Claude 3 by Anthropic",
            link="https://docs.anthropic.com/en/docs/about-claude/models#model-names",
            function_calling=True,
        ),
    ],
)
