from chain_bot_lib import get_llm, load_vectorstore, basic_chain

llm = get_llm()
if llm is None:
    print("API 키를 설정 후 다시 실행해주세요.")
    exit()

vs = load_vectorstore()


# 메인 로직
while True:
    human_input = input("[질문: q(종료)] ")
    if human_input == 'q':
        exit()

    # 벡터DB 구현 함수
    vs_data = basic_chain(llm, vs, human_input)

    print(f"[AI답변] {vs_data}")