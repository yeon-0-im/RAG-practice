import os
import math
from langchain_huggingface import HuggingFaceEmbeddings

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """두 벡터 사이의 코사인 유사도를 계산합니다 (1에 가까울수록 유사)"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a**2 for a in vec_a))
    norm_b = math.sqrt(sum(b**2 for b in vec_b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def huggingface_embeddings():
    """
    HuggingFace의 무료 로컬 임베딩 모델 사용 예시
    인터넷 연결만 있으면 API 키 없이 사용 가능!
    pip install langchain-huggingface sentence-transformers
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("[건너뜀] langchain-huggingface 패키지가 없습니다.")
        print("         → pip install langchain-huggingface sentence-transformers\n")
        return

    print("=" * 50)
    print("[HuggingFace] 무료 임베딩 모델 (API 키 불필요)")
    print("  → 모델 다운로드 중... (처음 실행 시 시간이 걸립니다)")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    sentences = [
        "경영책임자는 안전보건관리체계를 구축해야 합니다.",
        "중대산업재해 발생 시 1년 이상 징역 또는 10억원 이하 벌금이 부과됩니다.",
        "주식 시장이 오늘 크게 올랐습니다.",
    ]
    query = "중대재해처벌법 처벌 기준"

    doc_vecs = embeddings.embed_documents(sentences)
    q_vec = embeddings.embed_query(query)

    print(f"  → 벡터 차원: {len(doc_vecs[0])}")
    print(f"\n  질문: '{query}'")
    print(f"  질문 벡터 앞 5개 값: {[round(v, 4) for v in q_vec[:5]]}")
    print()
    print("  [코사인 유사도 결과 — 문장별 상세]")
    print()
    for i, (sentence, vec) in enumerate(zip(sentences, doc_vecs)):
        sim = cosine_similarity(q_vec, vec)
        bar = "█" * int(sim * 30)
        print(f"  ── 문장 {i + 1} ──")
        print(f"  내용       : {sentence}")
        print(f"  벡터 앞 5개: {[round(v, 4) for v in vec[:5]]}")
        print(f"  유사도     : {sim:.4f}  {bar}")
        print()

    print("\n  💡 중대재해 관련 문장이 주식 문장보다 가까워야 합니다!")
    print()

# B: HuggingFace 임베딩 (무료, 로컬)
huggingface_embeddings()
