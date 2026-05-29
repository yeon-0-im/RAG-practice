"""
=======================================================================
 PDF Table-Aware RAG Pipeline
 PDF (표 포함) → PDFPlumberLoader → BGE-M3 Embedding → ChromaDB → 검색
=======================================================================

※ 사용법: PDF_PATH 변수에 다운로드한 PDF 경로를 지정하세요.
──────────────────────────────────────────────────────────────────────

[설치 명령어]
pip install langchain langchain-community langchain-huggingface
pip install pdfplumber chromadb sentence-transformers
"""

import os
import re
import shutil
import pdfplumber
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ══════════════════════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════════════════════
PDF_PATH    = "./source/manual2.pdf"   # 사용할 PDF 경로
CHROMA_DIR  = "./chroma_db_bge_m3_table"  # ChromaDB 저장 디렉토리
EMBED_MODEL = "BAAI/bge-m3"            # BGE-M3 임베딩 모델
DEVICE      = "cpu"                    # GPU 사용 시 "cuda"
CHUNK_SIZE  = 800                      # 텍스트 청크 크기
CHUNK_OVERLAP = 100                     # 청크 겹침 크기


# ══════════════════════════════════════════════════════════════════════
# 1. 표 포함 PDF 로더 (PDFPlumber 기반 커스텀 파서)
# ══════════════════════════════════════════════════════════════════════
def table_to_markdown(table: List[List], page_num: int, table_idx: int) -> str:
    """pdfplumber 테이블 → Markdown 포맷 변환"""
    if not table or not table[0]:
        return ""

    md = f"\n[TABLE {table_idx} - Page {page_num}]\n"

    for row_idx, row in enumerate(table):
        # None/빈 값 정리
        clean_row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
        md += "| " + " | ".join(clean_row) + " |\n"
        if row_idx == 0:
            md += "| " + " | ".join(["---"] * len(clean_row)) + " |\n"

    return md


def load_pdf_with_tables(pdf_path: str) -> Tuple[List[Document], List[Document]]:
    """
    PDFPlumberLoader + 표(table) 인식 파서
    
    Returns:
        text_docs  : 페이지별 텍스트 Document 리스트
        table_docs : 표별 독립 Document 리스트 (분할 없이 보존)
    """
    print(f"\n{'='*60}")
    print(f"  PDF 로딩: {pdf_path}")
    print(f"{'='*60}")

    text_docs  = []
    table_docs = []

    with pdfplumber.open(pdf_path) as pdf:
        total_tables = 0

        for page_num, page in enumerate(pdf.pages, start=1):
            # ── 텍스트 추출 ──────────────────────────────────────
            raw_text = page.extract_text() or ""

            # ── 표 추출 ──────────────────────────────────────────
            tables = page.extract_tables()

            page_table_texts = []
            for t_idx, table in enumerate(tables, start=1):
                if not table:
                    continue
                total_tables += 1
                md_table = table_to_markdown(table, page_num, t_idx)

                # 표 → 별도 Document (구조 그대로 보존)
                table_docs.append(Document(
                    page_content=md_table,
                    metadata={
                        "source":      pdf_path,
                        "page":        page_num,
                        "doc_type":    "table",
                        "table_index": t_idx,
                        "row_count":   len(table),
                        "col_count":   len(table[0]) if table else 0,
                    }
                ))
                page_table_texts.append(md_table)

            # ── 페이지 전체 텍스트 Document ───────────────────────
            full_content = raw_text
            if page_table_texts:
                full_content += "\n\n" + "\n".join(page_table_texts)

            if full_content.strip():
                text_docs.append(Document(
                    page_content=full_content,
                    metadata={
                        "source":   pdf_path,
                        "page":     page_num,
                        "doc_type": "text",
                    }
                ))

            text_preview = raw_text[:80].replace("\n", " ").strip()
            print(f"  Page {page_num:>2} │ 텍스트 {len(raw_text):>5}자 │ 표 {len(tables)}개")
            if text_preview:
                print(f"           📄 텍스트: {text_preview}{'...' if len(raw_text) > 80 else ''}")
            for t_idx, md_table in enumerate(page_table_texts, start=1):
                table_preview = md_table[:80].replace("\n", " ").strip()
                print(f"           📊 표 {t_idx}: {table_preview}...")

    print(f"\n  ✅ 로딩 완료: 텍스트 {len(text_docs)}페이지 / 표 {len(table_docs)}개")
    return text_docs, table_docs


# ══════════════════════════════════════════════════════════════════════
# 2. 텍스트 청킹
# ══════════════════════════════════════════════════════════════════════
def split_documents(text_docs: List[Document]) -> List[Document]:
    """
    페이지 텍스트를 청크로 분할.
    표 Document는 구조 보존을 위해 분할하지 않음.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(text_docs)

    # 청크 번호 메타데이터 추가
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"\n  ✅ 청킹 완료: {len(text_docs)}페이지 → {len(chunks)}개 청크")
    return chunks


# ══════════════════════════════════════════════════════════════════════
# 3. BGE-M3 임베딩 모델 초기화
# ══════════════════════════════════════════════════════════════════════
def get_bge_m3_embeddings() -> HuggingFaceEmbeddings:
    """
    BAAI/bge-m3 임베딩 모델 로드.
    - 100개 언어 지원 (한국어 포함)
    - Dense / Sparse / ColBERT 멀티벡터 지원
    - 최초 실행 시 모델 자동 다운로드 (~570MB)
    """
    print(f"\n  BGE-M3 모델 로딩 중: {EMBED_MODEL}")
    print(f"  Device: {DEVICE}")
    print(f"  (최초 실행 시 모델 다운로드 필요, 약 570MB)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={
            "device": DEVICE,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 8,
        },
    )

    print(f"  ✅ BGE-M3 임베딩 모델 로드 완료")
    return embeddings


# ══════════════════════════════════════════════════════════════════════
# 4. ChromaDB 벡터 저장소 구축
# ══════════════════════════════════════════════════════════════════════
def build_vectordb(
    all_docs: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_dir: str = CHROMA_DIR,
) -> Chroma:
    """
    Document 리스트 → BGE-M3 임베딩 → ChromaDB 저장
    """
    print(f"\n  ChromaDB 구축 중...")
    print(f"  저장 경로: {persist_dir}")
    print(f"  총 Document 수: {len(all_docs)}개")

    # 기존 DB 삭제 후 재생성
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="pdf_table_rag",
        collection_metadata={"hnsw:space": "cosine"},
    )

    count = vectordb._collection.count()
    print(f"  ✅ ChromaDB 구축 완료: {count}개 벡터 저장")
    return vectordb


# ══════════════════════════════════════════════════════════════════════
# 5. 기존 ChromaDB 불러오기
# ══════════════════════════════════════════════════════════════════════
def load_vectordb(
    embeddings: HuggingFaceEmbeddings,
    persist_dir: str = CHROMA_DIR,
) -> Chroma:
    """저장된 ChromaDB 로드 (재사용)"""
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="pdf_table_rag",
    )
    print(f"  ✅ 기존 ChromaDB 로드: {vectordb._collection.count()}개 벡터")
    return vectordb


# ══════════════════════════════════════════════════════════════════════
# 6. 유사도 검색
# ══════════════════════════════════════════════════════════════════════
def search(
    vectordb: Chroma,
    query: str,
    k: int = 3,
    filter_type: str = None,  # "table" or "page" or None (전체)
) -> List[Tuple[Document, float]]:
    """
    쿼리로 유사 Document 검색.
    
    Args:
        query       : 검색 쿼리 문자열
        k           : 반환할 결과 수
        filter_type : 검색 범위 필터 ("table"=표만, "text"=텍스트만, None=전체)

    Returns:
        (Document, similarity_score) 리스트
    """
    where_filter = {"doc_type": filter_type} if filter_type else None

    results = vectordb.similarity_search_with_relevance_scores(
        query=query,
        k=k,
        filter=where_filter,
    )
    return results


def print_search_results(query: str, results: List[Tuple[Document, float]]):
    """검색 결과를 보기 좋게 출력"""
    print(f"\n{'━'*65}")
    print(f"  🔍 검색 쿼리: \"{query}\"")
    print(f"{'━'*65}")

    for rank, (doc, score) in enumerate(results, 1):
        dtype = doc.metadata.get("doc_type", "unknown")
        page  = doc.metadata.get("page", "?")
        icon  = "📊" if dtype == "table" else "📄"

        print(f"\n  [{rank}위] {icon} 유형: {dtype:5s} │ 페이지: {page} │ 유사도: {score:.4f}")
        print(f"  {'─'*60}")

        # 내용 미리보기 (최대 300자)
        preview = doc.page_content[:300].replace("\n", " ")
        print(f"  {preview}...")

    print()


# ══════════════════════════════════════════════════════════════════════
# 7. 전체 파이프라인 실행
# ══════════════════════════════════════════════════════════════════════
def build_pipeline(pdf_path: str = PDF_PATH) -> Tuple[Chroma, HuggingFaceEmbeddings]:
    """PDF → VectorDB 전체 파이프라인"""

    print("\n" + "█"*65)
    print("  PDF Table-Aware RAG Pipeline")
    print("  PDF → PDFPlumber → BGE-M3 → ChromaDB")
    print("█"*65)

    # Step 1: PDF 로드 (표 포함)
    text_docs, table_docs = load_pdf_with_tables(pdf_path)

    # Step 2: 텍스트 청킹 (표는 분할 안 함)
    print("\n[Step 2] 텍스트 청킹")
    print(f"{'─'*40}")
    split_docs = split_documents(text_docs)

    # Step 3: 전체 Document 합치기
    all_docs = split_docs + table_docs
    print(f"\n  📦 최종 Document 구성:")
    print(f"     - 텍스트 청크: {len(split_docs)}개")
    print(f"     - 표 Document: {len(table_docs)}개")
    print(f"     - 합계:        {len(all_docs)}개")

    # Step 4: BGE-M3 임베딩 모델 로드
    print("\n[Step 3] BGE-M3 임베딩 모델 초기화")
    print(f"{'─'*40}")
    embeddings = get_bge_m3_embeddings()

    # Step 5: ChromaDB 구축
    print("\n[Step 4] ChromaDB 벡터 저장소 구축")
    print(f"{'─'*40}")
    vectordb = build_vectordb(all_docs, embeddings)

    return vectordb, embeddings


# ══════════════════════════════════════════════════════════════════════
# 8. 메인 실행
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 파이프라인 구축 ───────────────────────────────────────────────
    vectordb, embeddings = build_pipeline(PDF_PATH)

    # ── 검색 테스트 ───────────────────────────────────────────────────
    print("\n" + "█"*65)
    print("  검색 테스트 시작")
    print("█"*65)

    # 테스트 쿼리 목록
    test_queries = [
        # 텍스트 검색
        ("타겟시장 핵심 이슈의 미디어 엔터테인먼트의 활용사례에 대해서 알려줘.",           None),
        # 표 검색
        ("타겟시장 핵심 이슈의 미디어 엔터테인먼트의 활용사례에 대해서 알려줘.",   "table"),
    ]

    for query, filter_type in test_queries:
        filter_label = f"(필터: {filter_type})" if filter_type else "(전체 검색)"
        print(f"\n  {filter_label}")
        results = search(vectordb, query, k=3, filter_type=filter_type)
        print_search_results(query, results)

    # ── 인터랙티브 검색 ───────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  인터랙티브 검색 모드 (종료: 'quit' 입력)")
    print("═"*65)

    while True:
        try:
            query = input("\n  검색어 입력 > ").strip()
            if query.lower() in ("quit", "exit", "q"):
                print("  종료합니다.")
                break
            if not query:
                continue

            filter_input = input("  필터 (table/text/Enter=전체) > ").strip().lower()
            filter_type = filter_input if filter_input in ("table", "text") else None

            results = search(vectordb, query, k=3, filter_type=filter_type)
            print_search_results(query, results)

        except KeyboardInterrupt:
            print("\n  종료합니다.")
            break
