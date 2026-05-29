# 라이브러리 불러오기
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

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

def basic_rag_chain(retriever, llm, human_message):
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

    answer = rag_chain.invoke(human_message)

    return answer


def runnable_lambda(retriever, llm, human_message):

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

    return chain.invoke(human_message)

