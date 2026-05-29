"""
=======================================================
[STEP 04] VectorStore — 벡터를 저장하고 검색하기 (Chroma)
=======================================================
강의 슬라이드: Part 06 · Slide 29 (Chroma)

■ 이 파일에서 배우는 것
  - Chroma VectorStore에 Document 저장하기
  - similarity_search()로 유사 문서 검색하기
  - persist_directory로 디스크에 영속 저장하기
  - metadata 필터로 특정 페이지만 검색하기

■ 설치
  pip install langchain langchain-chroma langchain-huggingface chromadb
  pip install FlagEmbedding   ← BGE-M3 의존성

■ 실행 방법
  python step04_vectorstore_bge-m3.py
  ※ API 키 불필요 — 로컬 모델 자동 다운로드 (처음 실행 시 시간 소요)
=======================================================
"""

import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "source/manual.pdf"
DB_PATH = "./chroma_db_bge_m3"

def load_docs():
    if not os.path.exists(PDF_PATH):
        print(f"[오류] '{PDF_PATH}' 파일이 없습니다.")
        print("      → python create_manual_pdf.py 를 먼저 실행하세요!")
        return None
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    # 빈 chunk 제거 — 임베딩 API에 빈 텍스트가 전달되면 IndexError 발생
    docs = [d for d in docs if d.page_content.strip()]
    print(f"  → {len(docs)}개 유효 chunk 준비 완료")
    return docs


def get_embeddings():
    """BAAI/bge-m3 로컬 임베딩 — API 키 불필요, 색인/검색 모두 사용"""
    from langchain_huggingface import HuggingFaceEmbeddings
    print("  임베딩: BAAI/bge-m3 (로컬, API 키 불필요)")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ══════════════════════════════════════════════════════
# [실습 1] Chroma에 문서 저장하기
# ══════════════════════════════════════════════════════


def create_vectorstore():
    from langchain_chroma import Chroma

    print("=" * 50)
    print("[실습 1] VectorStore 생성 및 문서 저장")

    docs = load_docs()
    if docs is None:
        return None

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name="jungdae_jaehai",
        persist_directory=DB_PATH,
    )

    print(f"  → {len(docs)}개 chunk를 '{DB_PATH}'에 저장했습니다.")
    return vectorstore


# ══════════════════════════════════════════════════════
# [실습 2] 유사도 검색
# ══════════════════════════════════════════════════════

def similarity_search(vectorstore):
    print("=" * 50)
    print("[실습 2] 유사도 검색 (similarity_search)")

    query = "경영책임자가 지켜야 할 안전 의무는 무엇인가요?"
    results = vectorstore.similarity_search(query, k=3)

    print(f"  질문: '{query}'")
    print(f"  → {len(results)}개 결과 반환\n")
    for i, doc in enumerate(results):
        print(f"  [결과 {i+1}]")
        # print(f"    내용: {doc.page_content[:70]}...")
        print(f"    내용: {doc.page_content}...")
        print(f"    출처: {doc.metadata}")
    print()


# ══════════════════════════════════════════════════════
# [실습 3] 유사도 점수와 함께 검색
# ══════════════════════════════════════════════════════

def search_with_score(vectorstore):
    print("=" * 50)
    print("[실습 3] 점수 포함 검색 (similarity_search_with_score)")

    query = "중대재해 발생 시 처벌 수위는?"
    results = vectorstore.similarity_search_with_score(query, k=3)

    print(f"  질문: '{query}'\n")
    for doc, score in results:
        print(f"  점수: {score:.4f}  ← 낮을수록 유사 (Chroma는 거리 기준)")
        # print(f"  내용: {doc.page_content[:60]}...")
        print(f"  내용: {doc.page_content}...")
        print()


# ══════════════════════════════════════════════════════
# [실습 4] metadata 필터로 검색 범위 좁히기
# ══════════════════════════════════════════════════════

def search_with_filter(vectorstore):
    print("=" * 50)
    print("[실습 4] metadata 필터 검색")

    query = "재해 발생 요건"
    results = vectorstore.similarity_search(
        query,
        k=3,
        filter={"page": 0},  # 0번 페이지(첫 페이지) chunk만 검색
    )

    print(f"  질문: '{query}'  (page=0 chunk만 검색)")
    print(f"  → {len(results)}개 결과\n")
    for doc in results:
        print(f"  p.{doc.metadata.get('page', '?')}  출처: {doc.metadata.get('source', '')}")
        # print(f"  내용: {doc.page_content[:70]}...")
        print(f"  내용: {doc.page_content}...")
        print()

vs = create_vectorstore()
if vs is None:
    exit()

similarity_search(vs)
search_with_score(vs)
search_with_filter(vs)
