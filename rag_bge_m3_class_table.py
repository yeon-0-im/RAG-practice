"""
rag_bge_m3_class.py

두 소스 파일을 하나로 통합한 LangChain RAG 클래스 파일입니다.

포함 기능
- PDF 문서 로드 및 chunk 분리
- BAAI/bge-m3 로컬 임베딩 생성
- Chroma VectorStore 생성 및 검색
- 기존 Chroma DB 기반 Retriever 생성
- Gemini LLM 초기화
- basic RAG chain 실행
- LCEL RunnableLambda 방식 RAG 실행

실행 예시
    python rag_bge_m3_class.py

사용 예시
    from rag_bge_m3_class import RagBgeM3, get_llm, build_rag_components, runnable_lambda

    rag = RagBgeM3()
    llm = rag.get_llm()
    retriever = rag.build_rag_components()
    answer = rag.runnable_lambda(retriever, llm, "경영책임자의 의무는?")
"""

import os
import shutil
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = "source/manual.pdf"
DB_PATH = "./chroma_db_bge_m3"
COLLECTION_NAME = "jungdae_jaehai"


class RagBgeM3:
    """BGE-M3 + Chroma + Gemini 기반 LangChain RAG 클래스"""

    def __init__(
        self,
        pdf_path: str = PDF_PATH,
        db_path: str = DB_PATH,
        collection_name: str = COLLECTION_NAME,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        search_k: int = 3,
        embedding_device: str = "cpu",
        llm_model: str = "gemini-2.5-flash",
        temperature: float = 0,
    ):
        load_dotenv()
        self.pdf_path = pdf_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_k = search_k
        self.embedding_device = embedding_device
        self.llm_model = llm_model
        self.temperature = temperature

    def load_docs(self):
        """PDF 문서를 로드하고 LangChain Document chunk로 분리합니다."""
        if not os.path.exists(self.pdf_path):
            print(f"[오류] '{self.pdf_path}' 파일이 없습니다.")
            print("      → python create_manual_pdf.py 를 먼저 실행하세요!")
            return None

        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        docs = splitter.split_documents(pages)
        docs = [d for d in docs if d.page_content.strip()]
        print(f"  → {len(docs)}개 유효 chunk 준비 완료")
        return docs

    def get_embeddings(self):
        """BAAI/bge-m3 로컬 임베딩 — API 키 불필요, 색인/검색 모두 사용"""
        from langchain_huggingface import HuggingFaceEmbeddings

        print("  임베딩: BAAI/bge-m3 (로컬, API 키 불필요)")
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": self.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def create_vectorstore(self):
        """LangChain Chroma VectorStore를 생성하고 PDF chunk를 저장합니다."""
        print("=" * 50)
        print("[실습 1] VectorStore 생성 및 문서 저장")

        docs = self.load_docs()
        if docs is None:
            return None

        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.get_embeddings(),
            collection_name=self.collection_name,
            persist_directory=self.db_path,
        )

        print(f"  → {len(docs)}개 chunk를 '{self.db_path}'에 저장했습니다.")
        return vectorstore

    def similarity_search(self, vectorstore, query: Optional[str] = None, k: int = 3):
        """VectorStore에서 유사도 검색을 실행합니다."""
        print("=" * 50)
        print("[실습 2] 유사도 검색 (similarity_search)")

        query = query or "경영책임자가 지켜야 할 안전 의무는 무엇인가요?"
        results = vectorstore.similarity_search(query, k=k)

        print(f"  질문: '{query}'")
        print(f"  → {len(results)}개 결과 반환\n")
        for i, doc in enumerate(results):
            print(f"  [결과 {i + 1}]")
            print(f"    내용: {doc.page_content}...")
            print(f"    출처: {doc.metadata}")
        print()
        return results

    def search_with_score(self, vectorstore, query: Optional[str] = None, k: int = 3):
        """VectorStore에서 유사도 점수와 함께 검색합니다."""
        print("=" * 50)
        print("[실습 3] 점수 포함 검색 (similarity_search_with_score)")

        query = query or "중대재해 발생 시 처벌 수위는?"
        results = vectorstore.similarity_search_with_score(query, k=k)

        print(f"  질문: '{query}'\n")
        for doc, score in results:
            print(f"  점수: {score:.4f}  ← 낮을수록 유사 (Chroma는 거리 기준)")
            print(f"  내용: {doc.page_content}...")
            print()
        return results

    def search_with_filter(
        self,
        vectorstore,
        query: Optional[str] = None,
        page: int = 0,
        k: int = 3,
    ):
        """metadata 필터로 검색 범위를 좁혀 검색합니다."""
        print("=" * 50)
        print("[실습 4] metadata 필터 검색")

        query = query or "재해 발생 요건"
        results = vectorstore.similarity_search(
            query,
            k=k,
            filter={"page": page},
        )

        print(f"  질문: '{query}'  (page={page} chunk만 검색)")
        print(f"  → {len(results)}개 결과\n")
        for doc in results:
            print(f"  p.{doc.metadata.get('page', '?')}  출처: {doc.metadata.get('source', '')}")
            print(f"  내용: {doc.page_content}...")
            print()
        return results

    def build_rag_components(self):
        """기존 chroma_db_bge_m3에서 Retriever를 준비합니다."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"'{self.db_path}' 디렉토리가 없습니다.\n"
                "  → create_vectorstore() 또는 기존 step04_vectorstore_bge-m3.py 를 먼저 실행하세요."
            )

        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.get_embeddings(),
            persist_directory=self.db_path,
        )
        count = vectorstore._collection.count()
        print(f"  VectorStore: {count}개 chunk 로드 완료 (중대재해처벌법 매뉴얼)")

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.search_k},
        )
        return retriever

    def get_llm(self):
        """Gemini LLM을 초기화합니다."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY 환경 변수를 설정해주세요.")

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=self.llm_model,
            google_api_key=api_key,
            temperature=self.temperature,
        )

    def basic_rag_chain(self, retriever, llm, human_message: str):
        """기본 LCEL RAG chain을 실행합니다."""
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

        rag_chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain.invoke(human_message)

    def runnable_lambda(self, retriever, llm, human_message: str):
        """LCEL RunnableLambda 방식으로 전처리/후처리를 포함한 RAG를 실행합니다."""

        def preprocess(query: str) -> dict:
            """질문을 정제하고 chroma_db에서 관련 문서를 검색해 context 구성"""
            cleaned = query.strip().rstrip("?!.")
            docs = retriever.invoke(cleaned)
            context = "\n\n".join(
                f"[p.{d.metadata.get('page', '?')}] {d.page_content}" for d in docs
            )
            return {"context": context, "question": cleaned}

        def postprocess(text: str) -> str:
            """답변에 출처 안내 문구 추가"""
            return f"[중대재해처벌법 매뉴얼 기반 답변]\n{text.strip()}"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "주어진 참고 문서만을 근거로 정확하게 답하세요."),
                ("human", "참고 문서:\n{context}\n\n질문: {question}"),
            ]
        )

        chain = (
            RunnableLambda(preprocess)
            | prompt
            | llm
            | StrOutputParser()
            | RunnableLambda(postprocess)
        )

        return chain.invoke(human_message)

    def run_cli(self):
        """질문을 입력받아 RAG 답변을 출력하는 CLI 실행 함수입니다."""
        try:
            llm = self.get_llm()
            retriever = self.build_rag_components()
        except Exception as exc:
            print("llm, vectorDB 호출에 실패하였습니다.")
            print(f"[상세 오류] {exc}")
            return

        while True:
            human_message = input("[질문(q:종료)]")
            if human_message == "q":
                return

            ai_message = self.runnable_lambda(retriever, llm, human_message)
            print(f"[AI] {ai_message}")


# -----------------------------------------------------------------------------
# 기존 함수명 호환용 래퍼
# 기존 소스에서 import 하던 함수명을 그대로 사용할 수 있게 유지합니다.
# -----------------------------------------------------------------------------
_default_rag = RagBgeM3()


def load_docs():
    return _default_rag.load_docs()


def get_embeddings():
    return _default_rag.get_embeddings()


def create_vectorstore():
    return _default_rag.create_vectorstore()


def similarity_search(vectorstore):
    return _default_rag.similarity_search(vectorstore)


def search_with_score(vectorstore):
    return _default_rag.search_with_score(vectorstore)


def search_with_filter(vectorstore):
    return _default_rag.search_with_filter(vectorstore)


def build_rag_components():
    return _default_rag.build_rag_components()


def get_llm():
    return _default_rag.get_llm()


def basic_rag_chain(retriever, llm, human_message):
    return _default_rag.basic_rag_chain(retriever, llm, human_message)


def runnable_lambda(retriever, llm, human_message):
    return _default_rag.runnable_lambda(retriever, llm, human_message)


if __name__ == "__main__":
    RagBgeM3().run_cli()
