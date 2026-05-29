"""
=======================================================
[STEP 07] LCEL 기본 체인 — 파이프(|)로 컴포넌트 연결하기 (HuggingFace)
=======================================================
강의 슬라이드: Part 09 · Slide 41 (basic_chain.py)

■ 이 파일에서 배우는 것
  - LCEL(LangChain Expression Language)의 | 파이프 문법
  - Prompt → Gemini LLM → OutputParser 3단계 기본 체인
  - invoke / stream / batch 호출 방법
  - RunnableLambda로 커스텀 함수를 체인에 연결하기

■ 설치
  pip install langchain langchain-core langchain-google-genai langchain-chroma
  pip install langchain-huggingface sentence-transformers

■ API 키 설정 (Gemini LLM 전용 — 임베딩은 로컬 HuggingFace 모델 사용)
  export GOOGLE_API_KEY=AIza...      # Mac/Linux
  set GOOGLE_API_KEY=AIza...         # Windows
  → https://aistudio.google.com/apikey 에서 무료 발급 가능

■ 실행 방법
  python step07_basic_chain_huggingface.py
  ※ chroma_db_huggingface가 있으면 재사용, 없으면 source/manual.pdf에서 생성합니다.
  ※ 임베딩: paraphrase-multilingual-MiniLM-L12-v2 (로컬, API 키 불필요)
  ※ LLM: Gemini (API 키 필요)
=======================================================
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

PDF_PATH = "source/manual.pdf"
DB_PATH = "./chroma_db_huggingface"


def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  GOOGLE_API_KEY 환경 변수가 없습니다.")
        print("   https://aistudio.google.com/apikey 에서 무료 발급 후 설정하세요.\n")
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )


def get_embeddings():
    """HuggingFace 로컬 임베딩 — API 키 불필요, 다국어 지원"""
    from langchain_huggingface import HuggingFaceEmbeddings
    print("  임베딩: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (로컬, API 키 불필요)")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore():
    emb = get_embeddings()

    if os.path.exists(DB_PATH):
        print(f"  기존 '{DB_PATH}' 재사용")
        vs = Chroma(
            collection_name="jungdae_jaehai",
            embedding_function=emb,
            persist_directory=DB_PATH,
        )
        print(f"  → {vs._collection.count()}개 chunk 로드 완료\n")
        return vs

    print(f"  '{DB_PATH}' 없음 → '{PDF_PATH}'에서 새로 생성")
    exit()


def retrieve(vs, query: str, k: int = 3) -> str:
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(
        f"[p.{d.metadata.get('page','?')}] {d.page_content}" for d in docs
    )


# ══════════════════════════════════════════════════════
# [실습 1] 가장 단순한 LCEL 체인 — 실제 PDF 내용 요약
# ══════════════════════════════════════════════════════


def basic_chain(llm, vs):
    print("=" * 50)
    print("[실습 1] 기본 LCEL 체인 (prompt | llm | parser)")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 중대재해처벌법 전문가입니다. 주어진 내용을 한 문장으로 요약하세요."),
        ("human", "다음 내용을 요약해줘:\n\n{context}"),
    ])
    parser = StrOutputParser()
    chain = prompt | llm | parser

    context = retrieve(vs, "중대재해처벌법 개요 및 적용 대상", k=2)
    print(f"  [chroma_db 검색 결과]\n  {context[:150]}...\n")

    result = chain.invoke({"context": context})
    print(f"  [Gemini 요약 결과]")
    print(f"  {result}")
    print()


# ══════════════════════════════════════════════════════
# [실습 2] stream() — 실시간 스트리밍 출력
# ══════════════════════════════════════════════════════


def streaming(llm, vs):
    print("=" * 50)
    print("[실습 2] 스트리밍 출력 (.stream())")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 중대재해처벌법 전문가입니다. 질문에 근거 있게 답변하세요."),
        ("human", "참고 문서:\n{context}\n\n질문: {question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    question = "경영책임자가 지켜야 할 안전보건 의무 3가지를 알려줘"
    context = retrieve(vs, question, k=3)

    print(f"  질문: '{question}'")
    print(f"  [실시간 출력 →] ", end="", flush=True)
    for token in chain.stream({"context": context, "question": question}):
        print(token, end="", flush=True)
    print("\n")


# ══════════════════════════════════════════════════════
# [실습 3] batch() — 여러 질문 한꺼번에 처리
# ══════════════════════════════════════════════════════


def batch(llm, vs):
    print("=" * 50)
    print("[실습 3] 배치 처리 (.batch()) — 복수 질문 동시 처리")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 중대재해처벌법 전문가입니다. 참고 문서를 바탕으로 한 문장으로 답하세요."),
        ("human", "참고 문서:\n{context}\n\n질문: {question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    questions = [
        "중대산업재해의 정의는 무엇인가요?",
        "사망 사고 발생 시 경영책임자의 처벌 수위는?",
        "안전보건관리체계 점검 주기는 얼마나 되나요?",
    ]

    inputs = [
        {"context": retrieve(vs, q, k=2), "question": q}
        for q in questions
    ]
    results = chain.batch(inputs)

    print("  [질문별 Gemini 답변 (배치)]")
    for q, ans in zip(questions, results):
        print(f"\n  ❓ {q}")
        print(f"  🤖 {ans.strip()}")
    print()


# ══════════════════════════════════════════════════════
# [실습 4] RunnableLambda — 검색 + 전처리 + 후처리 체인
# ══════════════════════════════════════════════════════


def runnable_lambda(llm, vs):
    print("=" * 50)
    print("[실습 4] RunnableLambda — 검색·전처리·후처리를 체인에 연결")

    def preprocess(query: str) -> dict:
        """질문을 정제하고 chroma_db에서 관련 문서를 검색해 context 구성"""
        cleaned = query.strip().rstrip("?!.")
        context = retrieve(vs, cleaned, k=2)
        return {"context": context, "question": cleaned}

    def postprocess(text: str) -> str:
        """답변에 출처 안내 문구 추가"""
        return f"[중대재해처벌법 매뉴얼 기반 답변]\n{text.strip()}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "주어진 참고 문서만을 근거로 정확하게 답하세요."),
        ("human", "참고 문서:\n{context}\n\n질문: {question}"),
    ])

    chain = (
        RunnableLambda(preprocess)
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(postprocess)
    )

    query = "중대재해처벌법 위반 시 법인이 받는 처벌은?"
    print(f"  입력 질문: '{query}'")
    result = chain.invoke(query)
    print(f"\n  [최종 출력]")
    print(f"  {result}")
    print()


llm = get_llm()
if llm is None:
    print("API 키를 설정 후 다시 실행해주세요.")
    exit()

print("[VectorStore 준비 중...]")
vs = load_vectorstore()

basic_chain(llm, vs)
streaming(llm, vs)
batch(llm, vs)
runnable_lambda(llm, vs)
