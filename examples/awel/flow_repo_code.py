"""AWEL: repository code assistant backed by repo_rg.

Uses the generic ripgrep-backed repo_rg tool to gather source context.

Usage:
    curl -X POST http://localhost:5670/api/v1/awel/trigger/examples/repo_code \
      -H "Content-Type: application/json" \
      -d '{"model": "deepseek-chat", "user_input": "How does RepositoryIndex work?", "search_paths": ["src"]}'
"""

import logging
from typing import List, Optional

from dbgpt._private.pydantic import BaseModel, Field
from dbgpt.core import ModelMessage, ModelRequest
from dbgpt.core.awel import DAG, HttpTrigger, MapOperator
from dbgpt.model.operators import LLMOperator

logger = logging.getLogger(__name__)


class TriggerReqBody(BaseModel):
    model: str = Field(default="deepseek-chat", description="Model name")
    user_input: str = Field(..., description="User question about mounted repository code")
    search_paths: Optional[List[str]] = Field(
        default=None,
        description="Optional repository-relative paths to include in repo_rg search",
    )
    ignore_paths: Optional[List[str]] = Field(
        default=None,
        description="Optional repository-relative paths or glob patterns to exclude from repo_rg search",
    )
    file_globs: Optional[List[str]] = Field(
        default=None,
        description="Optional include globs passed to repo_rg, e.g. ['*.go', '*.md']",
    )


class RepoCodeOperator(MapOperator[TriggerReqBody, ModelRequest]):
    """Gather repository context with repo_rg, then build an LLM request."""

    SYSTEM_PROMPT = """You are a repository code assistant.

Rules:
1. Answer questions based on actual source context retrieved from the mounted repository.
2. Quote file paths and line numbers when using retrieved evidence.
3. If retrieval returns no relevant files, say so before using general knowledge.
4. Keep answers concrete and implementation-oriented."""

    async def map(self, input_value: TriggerReqBody) -> ModelRequest:
        question = input_value.user_input.strip()
        context = self._gather_context(
            question,
            search_paths=input_value.search_paths,
            ignore_paths=input_value.ignore_paths,
            file_globs=input_value.file_globs,
        )

        sys_msg = self.SYSTEM_PROMPT + "\n\n## Retrieved Repository Context\n" + context

        messages = [
            ModelMessage.build_system_message(sys_msg),
            ModelMessage.build_human_message(question),
        ]
        return ModelRequest.build_request(input_value.model, messages)

    def _gather_context(
        self,
        question: str,
        search_paths: Optional[List[str]] = None,
        ignore_paths: Optional[List[str]] = None,
        file_globs: Optional[List[str]] = None,
    ) -> str:
        from dbgpt_ext.datasource.tool_repo_rg import repo_rg

        parts = []
        for keyword in self._extract_keywords(question):
            try:
                result = repo_rg(
                    query=keyword,
                    paths=search_paths,
                    ignore_paths=ignore_paths,
                    file_globs=file_globs,
                    max_results=8,
                )
                parts.append(f"### repo_rg results for '{keyword}'\n{result}\n")
            except Exception as exc:
                logger.warning("repo_rg keyword %s failed: %s", keyword, exc)

        combined = "\n".join(parts)
        if len(combined) > 15000:
            combined = combined[:15000] + "\n\n...(truncated)"
        return combined if combined else "(No relevant repository context found)"

    @staticmethod
    def _extract_keywords(question: str) -> list:
        import re

        candidates = []
        candidates += re.findall(r"[A-Z][a-zA-Z0-9]+", question)
        candidates += re.findall(r"[a-z]+_[a-z_]+", question)
        candidates += re.findall(r"[a-zA-Z0-9]+\.[a-zA-Z0-9.]+", question)

        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question)
        candidates.extend([word for word in words if len(word) >= 4])

        seen = set()
        result = []
        for item in candidates:
            if item in seen or item.isdigit():
                continue
            seen.add(item)
            result.append(item)
        return result[:5]


with DAG("dbgpt_awel_repo_code_assistant") as dag:
    trigger = HttpTrigger(
        "/examples/repo_code",
        methods="POST",
        request_body=TriggerReqBody,
    )
    code_op = RepoCodeOperator()
    llm_task = LLMOperator(task_name="llm_task")
    output = MapOperator(lambda out: out.to_dict())
    trigger >> code_op >> llm_task >> output
