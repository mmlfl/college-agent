from pydantic import BaseModel, Field
from pymilvus import AnnSearchRequest, RRFRanker

from graph.rag.rag_config import fine_rowing_model, embedding_model
from graph.rag.state.rag_agent_state import RagAgentState
from graph.rag.rag_previous_work.rag_index_mivus import client, COLLECTION_NAME

MAX_CANDIDATES = 5
MAX_FINAL = 5


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
        output_fields=["text", "source"],
    )

    ranked_context = fine_rowing_context(state["question"], fused)
    total_hits = sum(len(f) for f in fused)

    return {
        "context": ranked_context,
        "trace": [
            f"retrieve: {len(queries)}个查询混合检索 -> 命中{total_hits}条 -> 精排top-3"
        ],
    }


def fine_rowing_context(query: str, fused: list[list[dict]]) -> list[dict]:
    """对混合检索结果做精排,返回 top-3 chunks (含score/text/source)"""

    class RerankScores(BaseModel):
        scores: list[float] = Field(description="每个文档片段的相关性分数(0.0-1.0),顺序与输入一致")

    reranker = fine_rowing_model.with_structured_output(RerankScores)

    seen = set()
    chunks = []
    for hits in fused:
        for hit in hits:
            text = hit['entity']['text']
            source = hit['entity']['source']
            if text not in seen:
                seen.add(text)
                chunks.append({"text": text, "source": source})

    if not chunks:
        return []

    prompt = f"""用户问题: {query}
     以下是检索到的文档片段,请判断每个片段与用户问题的相关程度。
     返回每个文档的分数(0.0-1.0,越高越相关):
     {chunks}"""

    print("开始精排.....")
    try:
        result: RerankScores = reranker.invoke(prompt)
        scores = result.scores
    except Exception:
        scores = [1.0 - i * 0.1 for i in range(len(chunks))]

    scored_chunks = [
        {"score": scores[i], "text": chunks[i]["text"], "source": chunks[i]["source"]}
        for i in range(min(len(scores), len(chunks)))
    ]
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return scored_chunks[:3]
