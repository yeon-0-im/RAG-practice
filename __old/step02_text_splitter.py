import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "source/manual.pdf"

def load_docs():
    if not os.path.exists(PDF_PATH):
        print(f"[오류] '{PDF_PATH}' 파일이 없습니다.")
        # print("      → python create_manual_pdf.py 를 먼저 실행하세요!")
        return None
    loader = PyPDFLoader(PDF_PATH)
    return loader.load()

def split_text():
    """PDF를 로드한 뒤 RecursiveCharacterTextSplitter로 기본 분할"""
    docs = load_docs()
    if docs is None:
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    print("=" * 50)
    print("[실습 1] RecursiveCharacterTextSplitter — 기본 분할")
    print(f"  파일: {PDF_PATH}")
    print(f"  → 원본 페이지 수 : {len(docs)}")
    print(f"  → 분할된 chunk 수: {len(chunks)}")
    print()

def experiment_parameters():
    """chunk_size와 chunk_overlap 값을 바꿔가며 결과 비교"""
    docs = load_docs()
    if docs is None:
        return

    full_text = "\n".join(doc.page_content for doc in docs)

    print("=" * 50)
    print("[실습 2] chunk_size / chunk_overlap 파라미터 실험")
    print(f"  (PDF 전체 텍스트 총 {len(full_text)}자)\n")

    for size, overlap in [(100, 10), (300, 50), (500, 100), (1000, 200)]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
        )
        chunks = splitter.split_text(full_text)
        print(f"  chunk_size={size:4d}, overlap={overlap:3d} → chunk 수: {len(chunks):3d}개")

    print()
    print("  💡 chunk가 크면 맥락은 풍부하지만 검색 정확도 ↓")
    print("  💡 chunk가 작으면 검색은 정확하지만 맥락이 끊길 수 있음")
    print()


def show_results():
    """분할된 각 chunk의 내용과 metadata를 상세 출력"""
    docs = load_docs()
    if docs is None:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    print("=" * 50)
    print("[실습 3] 분할 결과 — chunk 수 · 내용 · metadata 확인")
    print(f"  → 총 chunk 수: {len(split_docs)}\n")

    for i, doc in enumerate(split_docs):
        print(f"  ── chunk {i + 1} ──")
        print(f"  내용    : {doc.page_content[:100]}{'...' if len(doc.page_content) > 100 else ''}")
        print(f"  metadata: {doc.metadata}")
        print()


# 테스트
split_text()
experiment_parameters()
show_results()