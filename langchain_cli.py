#!/usr/bin/env python3
"""
명령행에서 LangChain 예제를 실행하는 간단한 스크립트.

사용법:
  python langchain_cli.py            # 내장 예제 실행
  python langchain_cli.py --file msgs.json  # JSON 파일(리스트 형태)로 메시지 불러오기

메시지 포맷: [{"role":"user"|"assistant", "content":"..."}, ...]
"""
import argparse
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

MODEL = "gpt-4o-mini"
SYSTEM = "당신은 친절한 AI 어시스턴트입니다. 모르면 '확인 필요'라고만 답하세요."


def to_langchain_msgs(messages):
    msgs = [SystemMessage(content=SYSTEM)]
    for m in messages:
        role = m.get("role")
        text = m.get("content", "")
        cls = HumanMessage if role == "user" else AIMessage
        msgs.append(cls(content=text))
    return msgs


def load_messages_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run LangChain demo from CLI")
    parser.add_argument("--file", "-f", help="JSON file containing messages list")
    args = parser.parse_args()

    if args.file:
        messages = load_messages_from_file(args.file)
    else:
        messages = []

    llm = ChatOpenAI(model=MODEL, temperature=0.3)

    print("LangChain CLI — 입력하세요. /reset 으로 초기화, /quit 으로 종료합니다.")
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print()
                break

            if not user_input:
                continue

            if user_input == "/reset":
                messages = []
                print("대화가 초기화되었습니다.")
                continue

            if user_input == "/quit":
                print("종료합니다.")
                break

            # 사용자 메시지 추가 및 호출
            messages.append({"role": "user", "content": user_input})
            msgs = to_langchain_msgs(messages)
            try:
                resp = llm.invoke(msgs)
                assistant_text = resp.content
            except Exception as e:
                assistant_text = f"[ERROR] {e}"

            print("Assistant:", assistant_text)
            messages.append({"role": "assistant", "content": assistant_text})
    except KeyboardInterrupt:
        print("\n종료합니다.")


if __name__ == "__main__":
    main()
