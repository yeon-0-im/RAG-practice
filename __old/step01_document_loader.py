import os
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "source/manual.pdf"

def load_pdf():
    """PyPDFLoader로 중대재해처벌법 매뉴얼 PDF를 불러오는 예시"""

    if not os.path.exists(PDF_PATH):
        print(f"[오류] '{PDF_PATH}' 파일이 없습니다.")
        print("      → python create_manual_pdf.py 를 먼저 실행하세요!")
        return

    # 1) Loader 인스턴스 생성
    loader = PyPDFLoader(PDF_PATH)

    # 2) 모든 페이지 한꺼번에 로드 → List[Document]  (1페이지 = 1 Document)
    docs = loader.load()

    print("=" * 50)
    print("[실습 A] PyPDFLoader — 전체 페이지 로드")
    print(f"  파일: {PDF_PATH}")
    print(f"  → 총 페이지 수: {len(docs)}")
    print()

    # 3) 첫 번째 페이지 확인
    first = docs[0]
    print(f"  [1페이지 내용 앞 200자]")
    print(f"  {first.page_content[:200]}")
    print()
    # metadata에는 source(파일 경로)와 page(0-based 번호)가 자동 저장됨
    print(f"  metadata: {first.metadata}")
    print()

def iterate_pages():
    """각 페이지의 제목 키워드와 글자 수를 요약하는 예시"""

    if not os.path.exists(PDF_PATH):
        return

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print("=" * 50)
    print("[실습 B] 페이지별 요약")
    for doc in docs:
        page_num = doc.metadata.get("page", "?")
        content_preview = doc.page_content[:60].replace("\n", " ")
        print(f"  p.{page_num:02d}  ({len(doc.page_content):4d}자)  {content_preview}...")
    print()


def document_structure():
    """Document 객체의 page_content / metadata 구조를 직접 확인하는 예시"""

    if not os.path.exists(PDF_PATH):
        print(f"[오류] '{PDF_PATH}' 파일이 없습니다.")
        return

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print("=" * 50)
    print("[실습 C] Document 구조 직접 확인 (실제 PDF 데이터 — 전 페이지)")
    print(f"  총 페이지 수: {len(docs)}")
    print()
    for doc in docs:
        page_num = doc.metadata.get("page", "?")
        print(f"  ── 페이지 {page_num} ──")
        print(f"  page_content : {doc.page_content}")
        print(f"  metadata     : {doc.metadata}")
        print()


# 테스트
load_pdf()
iterate_pages()
document_structure()