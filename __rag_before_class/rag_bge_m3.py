from old_rag.rag_bge_m3_lib import *

try:
    llm = get_llm()
    retriever = build_rag_components()
except:
    print("LLM, VectorDB 호출에 실패하였습니다")
    exit()


while True:
    human_message = input("[질문(q:종료)] ")
    if human_message == "q":
        break

    # 이렇게는 실제 사용이 어렵다.
    ai_message = basic_rag_chain(retriever, llm, human_message)
    # LCEL 방식 전처리 포함 구현
    # ai_message = runnable_lambda(retriever, llm, human_message)

    print(f"[AI] {ai_message}")   
