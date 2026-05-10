import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymilvus import AnnSearchRequest, RRFRanker, MilvusClient

from graph.rag.rag_config import fine_rowing_model, embedding_model
from graph.rag.state.rag_agent_state import RagAgentState

MAX_CANDIDATES = 10
MAX_FINAL = 10
TOP_K = 5
load_dotenv()
#MilvusClient配置
client = MilvusClient(uri=os.getenv("MILVUS_URI"),db_name=os.getenv("MILVUS_DB"))
COLLECTION_NAME = os.getenv("MILVUS_DB_COLLECTION_NAME")

def retrieve_from_milvus(state: RagAgentState):
    queries = [state["question"]]
    if state.get("rewritten_questions") is not None:
        queries.extend(state["rewritten_questions"])

    vectors = embedding_model.embed_documents(queries)

    vector_search = AnnSearchRequest(
        data=vectors,
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 16}},
        limit=MAX_CANDIDATES,
    )
    bm25_search = AnnSearchRequest(
        data=queries,
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=MAX_CANDIDATES,
    )

    fused = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[bm25_search, vector_search],
        ranker=RRFRanker(k=60),
        limit=MAX_FINAL,
        output_fields=["text", "source", "doc_type", "product_id", "product_name"],
    )

    ranked_context = fine_rowing_context(state["question"], fused)
    total_hits = sum(len(f) for f in fused)

    return {
        "context": ranked_context,
        "trace": [
            f"retrieve: {len(queries)}个查询混合检索 -> 命中{total_hits}条 -> 精排top-{TOP_K}"
        ],
    }


def fine_rowing_context(query: str, fused: list[list[dict]]) -> list[dict]:
    """对混合检索结果做精排,返回 top-K chunks (含score/text/source/doc_type/product_id/product_name)"""

    class RerankScores(BaseModel):
        scores: list[float] = Field(description="每个文档片段的相关性分数(0.0-1.0),顺序与输入一致")

    reranker = fine_rowing_model.with_structured_output(RerankScores)

    seen = set()
    chunks = []
    for hits in fused:
        for hit in hits:
            entity = hit["entity"]
            text = entity["text"]
            if text not in seen:
                seen.add(text)
                chunks.append({
                    "text": text,
                    "source": entity.get("source", ""),
                    "doc_type": entity.get("doc_type", ""),
                    "product_id": entity.get("product_id", 0),
                    "product_name": entity.get("product_name", ""),
                })

    if not chunks:
        return []

    prompt = f"""用户问题: {query}
     以下是检索到的商品信息/评价片段,请判断每个片段与用户问题的相关程度。
     返回每个文档的分数(0.0-1.0,越高越相关):
     {chunks}"""

    print("开始精排.....")
    try:
        result: RerankScores = reranker.invoke(prompt)
        scores = result.scores
    except Exception:
        scores = [1.0 - i * 0.1 for i in range(len(chunks))]

    scored_chunks = [
        {
            "score": scores[i],
            "text": chunks[i]["text"],
            "source": chunks[i]["source"],
            "doc_type": chunks[i]["doc_type"],
            "product_id": chunks[i]["product_id"],
            "product_name": chunks[i]["product_name"],
        }
        for i in range(min(len(scores), len(chunks)))
    ]
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return scored_chunks[:TOP_K]
