"""
=======================================================
[STEP 08] RAG 체인 완성 — 검색 + 생성 파이프라인 (BGE-M3)
=======================================================
강의 슬라이드: Part 09 · Slide 42 (rag_chain.py)

■ 이 파일에서 배우는 것
  - LCEL로 RAG 전체 파이프라인을 한 번에 조립하기
  - BGE-M3 임베딩(로컬) + Gemini LLM 조합
  - Retriever → Prompt → LLM → Parser 흐름 이해
  - 출처(metadata)까지 함께 반환하는 고급 패턴
  - 스트리밍 RAG 답변 생성

■ 설치
  pip install langchain langchain-core langchain-google-genai
  pip install langchain-chroma chromadb
  pip install langchain-huggingface sentence-transformers
  pip install FlagEmbedding   ← BGE-M3 의존성

■ API 키 설정 (Gemini LLM 전용 — 임베딩은 로컬 BGE-M3 사용)
  export GOOGLE_API_KEY=AIza...      # Mac/Linux
  set GOOGLE_API_KEY=AIza...         # Windows
  → https://aistudio.google.com/apikey 에서 무료 발급 가능

■ 실행 방법
  python step08_rag_chain_bge-m3.py
  ※ chroma_db_bge_m3 가 미리 생성되어 있어야 합니다.
     (step04_vectorstore_bge-m3.py 또는 step07_basic_chain_bge-m3.py 먼저 실행)
=======================================================
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

DB_PATH = "./chroma_db_bge_m3"


def get_embeddings():
    """BAAI/bge-m3 로컬 임베딩 — API 키 불필요"""
    from langchain_huggingface import HuggingFaceEmbeddings
    print("  임베딩: BAAI/bge-m3 (로컬, API 키 불필요)")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_rag_components():
    """기존 chroma_db_bge_m3에서 Retriever를 준비합니다"""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"'{DB_PATH}' 디렉토리가 없습니다.\n"
            "  → step04_vectorstore_bge-m3.py 또는 step07_basic_chain_bge-m3.py 를 먼저 실행하세요."
        )

    vectorstore = Chroma(
        collection_name="jungdae_jaehai",
        embedding_function=get_embeddings(),
        persist_directory=DB_PATH,
    )
    count = vectorstore._collection.count()
    print(f"  VectorStore: {count}개 chunk 로드 완료 (중대재해처벌법 매뉴얼)")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    return retriever


def get_llm():
    """Gemini LLM을 초기화합니다"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY 환경 변수를 설정해주세요.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,  # RAG는 창의성보다 정확성이 중요 → 0 권장
    )


# ══════════════════════════════════════════════════════
# [실습 1] 기본 RAG 체인 (5줄!)
# ══════════════════════════════════════════════════════


def basic_rag_chain(retriever, llm):
    """슬라이드 42에 나온 LCEL RAG 체인 패턴 (BGE-M3 + Gemini LLM 버전)"""

    print("=" * 50)
    print("[실습 1] 기본 RAG 체인")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 중대재해처벌법 전문 어시스턴트입니다.\n"
                "아래 컨텍스트만을 근거로 답하고, 출처(source, page)를 함께 적으세요.\n"
                "컨텍스트에 답이 없으면 '문서에서 찾을 수 없습니다'라고 답하세요.\n"
                "한국어로, 친근하게 답합니다.",
            ),
            ("human", "### 컨텍스트\n{context}\n\n### 질문\n{question}"),
        ]
    )

    def format_docs(docs: list) -> str:
        """Document 리스트 → 출처 포함 텍스트 블록"""
        return "\n\n".join(
            f"[출처: {d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )

    # 🌟 RAG 체인 조립 (LCEL 파이프!)
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),  # 질문을 그대로 통과
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "경영책임자가 지켜야 할 안전보건 의무는 무엇인가요?",
        "중대산업재해 발생 시 처벌 수위는 어떻게 되나요?",
    ]

    for q in questions:
        print(f"\n  ❓ 질문: {q}")
        answer = rag_chain.invoke(q)
        print(f"  🤖 Gemini: {answer}")
    print()


# ══════════════════════════════════════════════════════
# [실습 2] 출처까지 반환하는 고급 RAG 체인
# ══════════════════════════════════════════════════════


def rag_with_sources(retriever, llm):
    """답변과 함께 사용된 문서의 출처 목록도 반환하는 패턴"""

    print("=" * 50)
    print("[실습 2] 출처 포함 RAG 체인")

    from langchain_core.runnables import RunnableParallel

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "주어진 컨텍스트만으로 질문에 답하세요. "
                "없으면 '문서에서 찾을 수 없습니다'라고 하세요.",
            ),
            ("human", "컨텍스트:\n{context}\n\n질문: {question}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    answer_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 답변 + 출처를 병렬로 반환
    rag_with_sources = RunnableParallel(
        {
            "answer": answer_chain,
            "sources": retriever,  # 검색된 Document 리스트도 함께 반환
        }
    )

    query = "중대재해처벌법 위반 시 법인이 받는 처벌은?"
    result = rag_with_sources.invoke(query)

    print(f"  ❓ 질문: {query}")
    print(f"\n  🤖 Gemini 답변:\n  {result['answer']}")
    print(f"\n  📚 참조 문서:")
    for doc in result["sources"]:
        print(
            f"     - p.{doc.metadata.get('page', '?')}: {doc.page_content[:50]}..."
        )
    print()


# ══════════════════════════════════════════════════════
# [실습 3] 스트리밍 RAG
# ══════════════════════════════════════════════════════


def streaming_rag(retriever, llm):
    """Gemini RAG 답변을 실시간 스트리밍하는 예시"""

    print("=" * 50)
    print("[실습 3] 스트리밍 RAG 답변")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "컨텍스트 기반으로 친절하게 답변하세요. 없으면 모른다고 하세요.",
            ),
            ("human", "컨텍스트:\n{context}\n\n질문: {question}"),
        ]
    )

    def format_docs(docs):
        return "\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    query = "안전보건관리체계 구축 의무에 대해 설명해줘"
    print(f"  ❓ 질문: {query}")
    print(f"  🤖 Gemini (실시간): ", end="", flush=True)

    for token in chain.stream(query):
        print(token, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    print("\n🚀 STEP 08 — RAG 체인 완성 실습 (BGE-M3 임베딩 + Gemini LLM)\n")

    print("[컴포넌트 준비 중...]")
    try:
        retriever = build_rag_components()
        llm = get_llm()
    except (EnvironmentError, FileNotFoundError) as e:
        print(f"⚠️  {e}")
        exit()

    print()

    print("━" * 50)
    print("  [실습 1] 기본 RAG 체인")
    print("━" * 50)
    basic_rag_chain(retriever, llm)

    print("━" * 50)
    print("  [실습 2] 출처 포함 RAG 체인")
    print("━" * 50)
    rag_with_sources(retriever, llm)

    print("━" * 50)
    print("  [실습 3] 스트리밍 RAG 답변")
    print("━" * 50)
    streaming_rag(retriever, llm)

    print("━" * 50)
    print("✅ STEP 08 완료! 다음은 step09_agent.py 를 실행하세요.")
