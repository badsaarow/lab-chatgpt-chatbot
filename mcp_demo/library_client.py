"""Interactive MCP library chatbot.

Starts the local library MCP server over stdio, exposes its tools to an LLM,
and answers questions entered in the terminal.

Run with: python mcp_demo/library_client.py
Commands: /reset (clear conversation), /stats (token usage), /quit (exit)
"""
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from pydantic import AnyUrl
from mcp.client.stdio import stdio_client

# Start the server beside this file, independent of the current working directory.
SERVER = Path(__file__).resolve().parent / "library_server.py"
SERVER_PARAMS = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
MODEL = "gpt-6-luna"
MAX_TURNS = 3
MAX_TOOL_CALLS = 5
SYSTEM = (
    "당신은 친절한 도서관 안내 챗봇입니다. 질문에 답할 때 필요한 경우 제공된 MCP 도구를 "
    "직접 호출하세요. 도서 검색이나 회원 대출 현황은 도구 결과에 근거해 답하고, "
    "도구 결과에 없는 정보는 지어내지 말고 확인이 필요하다고 말하세요."
)


def _text(result) -> str:
    """Extract readable text from an MCP result."""
    parts = []
    for item in getattr(result, "content", []) or []:
        parts.append(getattr(item, "text", str(item)))
    return "\n".join(parts) if parts else str(result)


def _tool_specs(tools) -> list[dict]:
    """Convert MCP tool definitions to the OpenAI function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools
    ]


async def chat(
    session: ClientSession, llm, history: list[list], usage_log: list[int],
    message: str, system_prompt: str,
) -> str:
    """Answer one user message, invoking MCP tools when the model requests them."""
    batch = [HumanMessage(content=message)]
    messages = [SystemMessage(content=system_prompt)]
    for turn in history[-MAX_TURNS + 1 :]:
        messages.extend(turn)
    messages.extend(batch)

    for _ in range(MAX_TOOL_CALLS + 1):
        response = await asyncio.to_thread(llm.invoke, messages)
        usage = getattr(response, "usage_metadata", None) or {}
        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            metadata = getattr(response, "response_metadata", None) or {}
            total_tokens = metadata.get("token_usage", {}).get("total_tokens")
        if total_tokens is not None:
            usage_log.append(total_tokens)

        batch.append(response)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", []) or []
        if not tool_calls:
            history.append(batch)
            return response.content if isinstance(response.content, str) else str(response.content)

        if len([item for item in batch if isinstance(item, ToolMessage)]) >= MAX_TOOL_CALLS:
            history.append(batch)
            return "도구 호출 횟수 제한에 도달했습니다. 질문을 나누어 다시 시도해 주세요."

        for tool_call in tool_calls:
            name = tool_call["name"]
            args = tool_call.get("args", {})
            call_id = tool_call["id"]
            try:
                result = await session.call_tool(name, args)
                content = _text(result)
                if getattr(result, "isError", False):
                    content = f"도구 오류: {content}"
            except Exception as exc:
                content = f"도구 호출 오류: {exc}"
            tool_message = ToolMessage(content=content, tool_call_id=call_id, name=name)
            batch.append(tool_message)
            messages.append(tool_message)

    history.append(batch)
    return "답변을 완성하지 못했습니다. 다시 질문해 주세요."


def reset(history: list[list], usage_log: list[int]) -> None:
    history.clear()
    usage_log.clear()
    print("(대화 초기화)")


def show_stats(usage_log: list[int]) -> None:
    if not usage_log:
        print("(아직 메시지 없음)")
        return

    price = 0.30 / 1_000_000  # input/output blended estimate
    total = sum(usage_log)
    per_call = total / len(usage_log)
    print(f"  모델 호출 수: {len(usage_log)}")
    print(f"  누적 토큰: {total}")
    print(f"  누적 비용: ${total * price:.5f}")
    print(f"  평균 호출당: {per_call:.1f} tokens")


async def main() -> None:
    load_dotenv()
    llm = ChatOpenAI(model=MODEL, reasoning_effort="none", temperature=0.3)
    history: list[list] = []
    usage_log: list[int] = []

    print(f"==> MCP 서버 연결: {SERVER.name}")
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            tool_result = await session.list_tools()
            tools = tool_result.tools
            llm_with_tools = llm.bind_tools(_tool_specs(tools))

            # Load the library resource once so the model can answer basic questions
            # such as opening hours without an additional user-facing tool call.
            resource_result = await session.read_resource(AnyUrl("config://library"))
            resource_text = "\n".join(
                item.text for item in resource_result.contents if getattr(item, "text", None)
            )
            system_content = SYSTEM
            if resource_text:
                system_content += f"\n\n도서관 안내 정보:\n{resource_text}"

            print(f"연결됨 — 서버: {info.serverInfo.name}")
            print("명령어: /reset 초기화 · /stats 통계 · /quit 종료")
            print("질문을 입력하세요.\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue
                    if user_input == "/reset":
                        reset(history, usage_log)
                        continue
                    if user_input == "/stats":
                        show_stats(usage_log)
                        continue
                    if user_input == "/quit":
                        print("(대화 종료)")
                        break

                    print("Bot: ", end="", flush=True)
                    try:
                        answer = await chat(
                            session, llm_with_tools, history, usage_log, user_input, system_content
                        )
                        print(answer)
                    except Exception as exc:
                        print(f"오류 발생: {exc}")
                    print()
                except KeyboardInterrupt:
                    print("\n(Ctrl+C로 종료)")
                    break


if __name__ == "__main__":
    asyncio.run(main())
