from rag_bge_m3_class import RagBgeM3, get_llm, build_rag_components, runnable_lambda

rag = RagBgeM3()
llm = rag.get_llm()
retriever = rag.build_rag_components()
answer = rag.runnable_lambda(retriever, llm, "경영책임자의 의무는?")

print(answer)