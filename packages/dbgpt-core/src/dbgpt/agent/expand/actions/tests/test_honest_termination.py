import pytest

from dbgpt.agent import AgentMessage
from dbgpt.agent.expand.actions.react_action import ReActAction, Terminate
from dbgpt.agent.expand.react_agent import ReActAgent
from dbgpt.agent.resource.tool.pack import ToolPack
from dbgpt.agent.util.react_parser import ReActStep


@pytest.mark.asyncio
async def test_plain_model_answer_does_not_terminate_successfully():
    agent = ReActAgent.model_construct()

    result = await agent.act(
        AgentMessage(
            content=(
                "This is a substantive plain answer that is deliberately longer "
                "than fifty characters and contains no ReAct action."
            )
        ),
        sender=None,
    )

    assert result.is_exe_success is False
    assert result.terminate is False
    assert "format" in result.content.lower()


@pytest.mark.asyncio
async def test_step_without_action_does_not_terminate_successfully():
    result = await ReActAction()._do_run(
        "A thought without an action is not execution evidence.",
        ReActStep(thought="I should inspect the datasource first."),
    )

    assert result.is_exe_success is False
    assert result.terminate is False
    assert "action" in result.content.lower()


@pytest.mark.asyncio
async def test_explicit_terminate_action_uses_declared_output_contract():
    action = ReActAction()
    action.init_resource(ToolPack([Terminate()]))

    result = await action._do_run(
        "Thought: evidence is complete\n"
        "Action: terminate\n"
        'Action Input: {"output": "verified final answer"}',
        ReActStep(
            thought="evidence is complete",
            action="terminate",
            action_input='{"output": "verified final answer"}',
        ),
        need_vis_render=False,
    )

    assert result.is_exe_success is True
    assert result.terminate is True
    assert result.content == "verified final answer"
