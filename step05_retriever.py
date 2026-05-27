"""
=======================================================
[STEP 05] Retriever — 검색의 표준 인터페이스
=======================================================
강의 슬라이드: Part 07 · Slide 33 (as_retriever)

■ 이 파일에서 배우는 것
  - VectorStore → Retriever 변환 (as_retriever)
  - 3가지 검색 전략: similarity / MMR / score_threshold
  - retriever.invoke()로 관련 문서 가져오기

■ 설치
  pip install langchain langchain-chroma langchain-google-genai chromadb

■ API 키 설정 (Gemini)
  export GOOGLE_API_KEY=AIza...      # Mac/Linux
  set GOOGLE_API_KEY=AIza...         # Windows

■ 실행 방법
  python step05_retriever.py
  ※ chroma_db가 있으면 재사용, 없으면 source/manual.pdf에서 생성합니다.
=======================================================
"""

import os
import shutil
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "source/manual.pdf"
# DB_PATH = "./chroma_db_bge_m3"
DB_PATH = "./chroma_db_huggingface"

def get_doc_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    print("  임베딩(색인): gemini-embedding-2-preview / RETRIEVAL_DOCUMENT")
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        task_type="RETRIEVAL_DOCUMENT",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_query_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    print("  임베딩(검색): gemini-embedding-2-preview / RETRIEVAL_QUERY")
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        task_type="RETRIEVAL_QUERY",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def build_vectorstore():
    """chroma_db가 있으면 검색용(RETRIEVAL_QUERY)으로 로드,
    없으면 색인용(RETRIEVAL_DOCUMENT)으로 PDF를 임베딩해 생성"""

    if os.path.exists(DB_PATH):
        print(f"  기존 '{DB_PATH}' 재사용")
        vs = Chroma(
            collection_name="jungdae_jaehai",
            embedding_function=get_query_embeddings(),
            persist_directory=DB_PATH,
        )
        print(f"  → {vs._collection.count()}개 chunk 로드 완료\n")
        return vs

    print(f"  '{DB_PATH}' 없음 → '{PDF_PATH}'에서 새로 생성")
    if not os.path.exists(PDF_PATH):
        print(f"[오류] '{PDF_PATH}' 파일이 없습니다.")
        print("      → python create_manual_pdf.py 를 먼저 실행하세요!")
        return None

    pages = PyPDFLoader(PDF_PATH).load()
    docs = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(pages)

    vs = Chroma.from_documents(
        documents=docs,
        embedding=get_doc_embeddings(),
        collection_name="jungdae_jaehai",
        persist_directory=DB_PATH,
    )
    print(f"  → {len(docs)}개 chunk 생성 및 저장 완료\n")
    return vs


# ══════════════════════════════════════════════════════
# [실습 1] 기본 — similarity (top-k)
# ══════════════════════════════════════════════════════


def similarity_retriever(vectorstore):
    print("=" * 50)
    print("[실습 1] similarity Retriever — 가장 가까운 k개 반환")
    print("  ※ 관련 없는 질문도 무조건 k개를 반환합니다 (점수 무관)")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    for query in [
        "경영책임자의 안전보건 의무는 무엇인가요?",
        "오늘 저녁 날씨가 어떤가요?",
    ]:
        docs = retriever.invoke(query)
        print(f"\n  질문: '{query}'")
        print(f"  → {len(docs)}개 Document 반환 (항상 k=3 반환 — 필터 없음)")
        for i, doc in enumerate(docs):
            # print(f"    [{i+1}] p.{doc.metadata.get('page','?')}  {doc.page_content[:60]}...")
            print(f"    [{i+1}] p.{doc.metadata.get('page','?')}  {doc.page_content}...")
    print()


# ══════════════════════════════════════════════════════
# [실습 2] MMR — 유사하면서도 다양한 결과
# ══════════════════════════════════════════════════════


def mmr_retriever(vectorstore):
    """MMR: 질문과 유사하면서도 서로 중복되지 않는 결과를 선택합니다."""
    print("=" * 50)
    print("[실습 2] MMR Retriever — 유사성 + 다양성 균형")
    print("  ※ 관련 없는 질문도 k개 반환하지만, 최대한 다양한 chunk를 골라 줍니다")

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5,  # 0=다양성 최대, 1=유사도 최대
        },
    )

    for query in [
        "중대재해처벌법 처벌 규정",
        "오늘 저녁 날씨가 어떤가요?",
    ]:
        docs = retriever.invoke(query)
        print(f"\n  질문: '{query}'")
        print(f"  → {len(docs)}개 Document 반환 (중복 최소화, 필터 없음)")
        for i, doc in enumerate(docs):
            # print(f"    [{i+1}] p.{doc.metadata.get('page','?')}  {doc.page_content[:60]}...")
            print(f"    [{i+1}] p.{doc.metadata.get('page','?')}  {doc.page_content}...")
    print()
    print("  💡 similarity와 달리 결과 간 다양성을 확보하지만, 무관한 질문은 걸러내지 못합니다.")
    print()


# ══════════════════════════════════════════════════════
# [실습 3] Score Threshold — 점수 기준 필터
# ══════════════════════════════════════════════════════


def threshold_retriever(vectorstore):
    """score_threshold: 유사도가 기준값 이상인 결과만 반환합니다."""
    print("=" * 50)
    print("[실습 3] score_threshold Retriever — 낮은 점수 결과 제외")
    print("  ※ 임계값(0.5) 미만이면 빈 리스트 반환 → LLM에 '모른다' 유도 가능")

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5},
    )

    for query in [
        "중대산업재해 발생 요건이 어떻게 되나요?",
        "오늘 저녁 날씨가 어떤가요?",
    ]:
        docs = retriever.invoke(query)
        print(f"\n  질문: '{query}'")
        if docs:
            print(f"  → {len(docs)}개 반환 (임계값 이상)")
            for i, doc in enumerate(docs):
                # print(f"    [{i+1}] p.{doc.metadata.get('page','?')}  {doc.page_content[:60]}...")
                print(f"    [{i+1}] p.{doc.metadata.get('page','?')}  {doc.page_content}...")
        else:
            print(f"  → 0개 반환 ← 임계값(0.5) 미달 — 관련 문서 없음으로 처리")
            print(f"     → LLM에게 빈 컨텍스트가 전달되어 '찾을 수 없습니다' 답변 유도")
    print()
    print("  💡 3가지 비교 요약:")
    print("     similarity       : 무조건 k개 반환 (관련 없어도 반환)")
    print("     mmr              : 무조건 k개 반환 (단, 중복 최소화)")
    print("     score_threshold  : 유사도 미달 시 빈 리스트 → 가장 안전한 RAG용 전략")
    print()


if __name__ == "__main__":
    print("\n🔍 STEP 05 — Retriever 실습 — 중대재해처벌법 매뉴얼\n")

    print("[VectorStore 준비 중...]")
    vs = build_vectorstore()
    if vs is None:
        exit()

    similarity_retriever(vs)
    mmr_retriever(vs)
    threshold_retriever(vs)