#!/usr/bin/env python3
"""
멀티턴 채팅 CLI — LangChain 기반 대화 메모리 챗봇
Usage: python chat_cli.py
명령어: /reset (초기화), /quit (종료)
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 초기화
load_dotenv()
MODEL = "gpt-6-luna"   # 호출마다 reasoning_effort="none" 을 함께 넣는다 (교재 5.3)
MAX_TURNS = 3          # 모델에 전달할 최대 대화 턴 수
llm = ChatOpenAI(model=MODEL, reasoning_effort="none", temperature=0.3)

# 상태
history = []
usage_log = []
SYSTEM = "당신은 친절한 AI 어시스턴트입니다. 모르면 '확인 필요'라고만 답하세요."


def reset():
    """대화 히스토리 초기화."""
    global history, usage_log
    history.clear()
    usage_log.clear()
    print("(대화 초기화)")


def chat(message: str, temperature: float = 0.3) -> str:
    """사용자 메시지에 대한 챗봇 응답 생성."""
    global history, usage_log

    # 새 턴을 추가하기 전에 오래된 턴을 제거해 최근 MAX_TURNS 턴만 유지한다.
    if len(history) >= MAX_TURNS * 2:
        del history[:2]

    history.append({"role": "user", "content": message})

    # 노트북의 변환 루프와 같이 dict history를 LangChain 메시지로 바꾼다.
    msgs = [SystemMessage(content=SYSTEM)]
    for item in history:
        cls = HumanMessage if item["role"] == "user" else AIMessage
        msgs.append(cls(content=item["content"]))

    resp = llm.invoke(msgs, temperature=temperature)
    answer = resp.content
    history.append({"role": "assistant", "content": answer})
    usage = getattr(resp, "usage_metadata", None) or {}
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        # LangChain/OpenAI 조합에 따라 원본 usage 정보가 여기에 실릴 수 있다.
        metadata = getattr(resp, "response_metadata", None) or {}
        total_tokens = metadata.get("token_usage", {}).get("total_tokens")
    if total_tokens is not None:
        usage_log.append(total_tokens)

    return answer


def show_stats():
    """토큰 및 비용 통계 출력."""
    if not usage_log:
        print("(아직 메시지 없음)")
        return
    
    PRICE = 0.30 / 1_000_000  # 입출력 혼합 평균
    total = sum(usage_log)
    per_turn = total / len(usage_log)
    
    print(f"  턴 수: {len(usage_log)}")
    print(f"  누적 토큰: {total}")
    print(f"  누적 비용: ${total * PRICE:.5f}")
    print(f"  평균 턴당: {per_turn:.1f} tokens")
    print(f"  100턴 추정: ${per_turn * 100 * PRICE:.4f}")


def main():
    """메인 CLI 루프."""
    print("=" * 60)
    print("멀티턴 AI 챗봇 CLI")
    print("=" * 60)
    print("명령어:")
    print("  /reset   - 대화 초기화")
    print("  /stats   - 토큰 및 비용 통계")
    print("  /persona - 페르소나 변경 (구현 예정)")
    print("  /quit    - 종료")
    print("=" * 60)
    print()
    
    reset()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input == "/reset":
                reset()
                continue
            
            if user_input == "/stats":
                show_stats()
                continue
            
            if user_input == "/quit":
                print("(대화 종료)")
                break
            
            print("Bot: ", end="", flush=True)
            response = chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n(Ctrl+C로 종료)")
            break
        except Exception as e:
            print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
