"""실습 6 — Streamlit + RAG + MCP 도서관.

문서 RAG와 도서관 MCP 챗봇을 제공한다. RAG는 업로드한 문서로만 답하고,
MCP 모드는 로컬 도서관 MCP 서버의 검색 및 대출 도구를 사용한다.

실행:
    streamlit run web/streamlit_rag.py
    # 컨테이너에서 8501 포트가 자동 포워딩된다.
"""
import asyncio
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

load_dotenv()

st.set_page_config(page_title="RAG · 도서관 MCP 챗봇", page_icon="📚")
st.title("문서 RAG · 도서관 MCP 챗봇 📚")

PROMPT = ChatPromptTemplate.from_template(
    """아래 '문맥'만 근거로 한국어로 답하세요.
문맥에 답이 없으면 반드시 "문서에서 찾을 수 없습니다"라고만 답하세요.

[문맥]
{context}

[질문]
{question}
"""
)

LIBRARY_SERVER = Path(__file__).resolve().parents[1] / "mcp_demo" / "library_server.py"
LIBRARY_MODEL = "gpt-6-luna"
MAX_TOOL_CALLS = 5
LIBRARY_SYSTEM = (
    "당신은 친절한 도서관 안내 챗봇입니다. 도서 검색이나 대출 현황 질문에는 반드시 "
    "제공된 MCP 도구를 호출하고, 답은 도구 결과에 근거해 한국어로 하세요. "
    "도구 결과에 없는 정보는 지어내지 말고 확인이 필요하다고 말하세요."
)


def _tool_specs(tools) -> list[dict]:
    """MCP 도구 정의를 LangChain의 function calling 형식으로 변환한다."""
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


def _mcp_text(result) -> str:
    parts = [getattr(item, "text", str(item)) for item in getattr(result, "content", []) or []]
    return "\n".join(parts) if parts else str(result)


async def _answer_with_library_mcp(question: str, history: list[tuple[str, str]]) -> str:
    """MCP 도서관 서버를 실행하고, 모델이 필요한 도구를 호출해 질문에 답한다."""
    llm = ChatOpenAI(model=LIBRARY_MODEL, reasoning_effort="none", temperature=0)
    server_params = StdioServerParameters(command=sys.executable, args=[str(LIBRARY_SERVER)])
    messages = [SystemMessage(content=LIBRARY_SYSTEM)]
    for user_text, assistant_text in history[-5:]:
        messages.extend([
            HumanMessage(content=user_text),
            AIMessage(content=assistant_text),
        ])
    messages.append(HumanMessage(content=question))

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_result = await session.list_tools()
            llm_with_tools = llm.bind_tools(_tool_specs(tool_result.tools))

            # 서버 안내 리소스는 매 질문마다 읽어 도구 대화의 근거로 제공한다.
            resource_result = await session.read_resource(AnyUrl("config://library"))
            resource_text = "\n".join(
                item.text for item in resource_result.contents if getattr(item, "text", None)
            )
            if resource_text:
                messages[0] = SystemMessage(content=f"{LIBRARY_SYSTEM}\n\n도서관 안내 정보:\n{resource_text}")

            tool_calls_used = 0
            for _ in range(MAX_TOOL_CALLS + 1):
                response = await asyncio.to_thread(llm_with_tools.invoke, messages)
                messages.append(response)
                tool_calls = getattr(response, "tool_calls", []) or []
                if not tool_calls:
                    return response.content if isinstance(response.content, str) else str(response.content)

                for tool_call in tool_calls:
                    if tool_calls_used >= MAX_TOOL_CALLS:
                        return "도구 호출 횟수 제한에 도달했습니다. 질문을 나누어 다시 시도해 주세요."
                    tool_calls_used += 1
                    try:
                        result = await session.call_tool(tool_call["name"], tool_call.get("args", {}))
                        content = _mcp_text(result)
                        if getattr(result, "isError", False):
                            content = f"도구 오류: {content}"
                    except Exception as exc:
                        content = f"도구 호출 오류: {exc}"
                    messages.append(ToolMessage(
                        content=content,
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    ))

    return "답변을 완성하지 못했습니다. 다시 질문해 주세요."


@st.cache_resource(show_spinner="문서 인덱싱 중...")
def build_chain(file_bytes: bytes, suffix: str):
    """업로드 파일 → 청크 → FAISS → RAG 체인. (캐시: 같은 파일은 1회만 인덱싱)"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyMuPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)

    vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)

    def fmt(ds):
        return "\n\n".join(d.page_content for d in ds)

    return (
        {"context": retriever | fmt, "question": RunnablePassthrough()}
        | PROMPT | llm | StrOutputParser()
    )


mode = st.radio("챗봇", ["문서 RAG", "도서관 MCP"], horizontal=True)

if mode == "문서 RAG":
    uploaded = st.file_uploader("문서 업로드 (.md / .txt / .pdf)", type=["md", "txt", "pdf"])
    if not uploaded:
        st.info("문서를 업로드하면 그 내용으로만 답합니다. 샘플: `data/sample_docs/hr_policy.md`")
    else:
        chain = build_chain(uploaded.getvalue(), Path(uploaded.name).suffix.lower())
        st.success(f"'{uploaded.name}' 인덱싱 완료 — 질문해 보세요.")
        question = st.chat_input("문서 내용을 물어보세요")
        if question:
            st.chat_message("user").write(question)
            with st.chat_message("assistant"):
                st.write(chain.invoke(question))
else:
    st.caption("MCP 도서관 서버의 검색 도구를 사용합니다. 예: 해리포터 책 있어?")
    history = st.session_state.setdefault("library_history", [])
    for user_text, assistant_text in history:
        st.chat_message("user").write(user_text)
        st.chat_message("assistant").write(assistant_text)

    question = st.chat_input("도서관에 책이 있는지 물어보세요")
    if question:
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            with st.spinner("도서관 MCP 서버에 검색 중..."):
                try:
                    answer = asyncio.run(_answer_with_library_mcp(question, history))
                except Exception as exc:
                    answer = f"도서관 검색 중 오류가 발생했습니다: {exc}"
            st.write(answer)
        history.append((question, answer))
