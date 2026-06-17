"""실습 5 Task 2 — Gradio 챗봇 (비교용).

Streamlit 챗봇(`web/streamlit_app.py`)과 대비해
같은 챗봇을 Gradio 로 만들어 본다. 내부는 동일하게 LangChain 메시지를 쓴다.

실행:
    python web/gradio_app.py
    # 컨테이너에서 7860 포트가 자동 포워딩된다. share=True 로 외부 URL 도 생성.
"""
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

SYSTEM = "당신은 친절한 AI 어시스턴트입니다. 모르면 '확인 필요'라고만 답하세요."
MAX_CHARS = 500  # 실습 5 Task 3 — 입력 길이 가드
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def respond(message, history):
    """history 는 [{'role': 'user'/'assistant', 'content': ...}] (Gradio messages 형식)."""
    if len(message) > MAX_CHARS:
        yield f"⚠️ 입력은 {MAX_CHARS}자 이내로 부탁드립니다."
        return

    msgs = [SystemMessage(content=SYSTEM)]
    for h in history:
        if h["role"] == "user":
            msgs.append(HumanMessage(content=h["content"]))
        else:
            msgs.append(AIMessage(content=h["content"]))
    msgs.append(HumanMessage(content=message))

    partial = ""
    try:
        for chunk in llm.stream(msgs):  # 스트리밍
            partial += chunk.content or ""
            yield partial
    except Exception:
        # 실습 5 Task 3 — 오류 시 Stack trace 노출 금지
        yield "죄송합니다. 일시적인 오류가 발생했어요. 잠시 후 다시 시도해 주세요."


demo = gr.ChatInterface(
    respond,
    title="나만의 ChatGPT 💬 (Gradio)",
    description="Streamlit 버전과 비교해 보세요 — 어느 쪽이 우리 회사에 맞나요?",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7880, share=True)
