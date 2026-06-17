"""간단한 Streamlit 채팅 UI (최소 구현).

실행:
    streamlit run web/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="간단 채팅", page_icon="💬")
st.title("나만의 ChatGPT (간단)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

msg = st.chat_input("메시지를 입력하세요")
if msg:
    st.session_state.messages.append({"role": "user", "content": msg})
    st.chat_message("user").write(msg)

    # 간단한 에코 응답 (실제 LLM 연동은 제거)
    reply = "응답: " + msg
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
