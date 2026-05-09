import os
from typing import List

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
from pymilvus.milvus_client.index import IndexParams

from graph.rag.rag_previous_work.rag_loadDocuments import get_chunks

load_dotenv()
# ================= 配置区 =================
URI = "http://192.168.56.10:19530"
DB_NAME = "lfl_college"
COLLECTION_NAME = "campus_details"
RESET_MODE = False
DIRECTORY_PATH = "/college_rag_data"

# ================= 模型配置 =================
APIKEY = os.getenv("DASHSCOPE_API_KEY")
BASEURL = os.getenv("DASHSCOPE_BASE_URL")
EMBEDDING_MODEL_NAME = "text-embedding-v2"
QUERY_REWRITE_MODEL_NAME = "qwen-turbo"
FINE_ROWING_MODEL_NAME = "qwen3-max-preview"

embedding_model = OpenAIEmbeddings(
    api_key=APIKEY,
    base_url=BASEURL,
    model=EMBEDDING_MODEL_NAME,
    check_embedding_ctx_length=False,
)

query_rewrite_model = ChatOpenAI(
    api_key=APIKEY,
    base_url=BASEURL,
    model=QUERY_REWRITE_MODEL_NAME,
)

fine_rowing_model = ChatOpenAI(
    api_key=APIKEY,
    base_url=BASEURL,
    model=FINE_ROWING_MODEL_NAME,
    max_retries=3,
)

client = MilvusClient(uri=URI, db_name=DB_NAME)


def prepare_milvus_env(reset=False):
    client_temp = MilvusClient(uri=URI)

    current_dbs = client_temp.list_databases()
    print(f"当前已有数据库: {current_dbs}")

    if DB_NAME not in current_dbs:
        client_temp.create_database(DB_NAME)
        print(f"创建数据库 '{DB_NAME}'")
    else:
        print(f"数据库 '{DB_NAME}' 已存在")

    client_temp.using_database(DB_NAME)
    # 切换到目标数据库
    if reset:
        if client_temp.has_collection(COLLECTION_NAME):
            print(f"清理】正在删除 collection: {COLLECTION_NAME}")
            client_temp.drop_collection(COLLECTION_NAME)
    client_temp.close()


def load_documents_in_milvus():
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1536)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    bm25_function = Function(
        name="bm25",
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    index_params = IndexParams()
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        }
    )
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={
            "nlist": 256
        }
    )
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print("加载文档数据...")
    chunks = get_chunks(DIRECTORY_PATH)
    texts = [doc.page_content for doc in chunks]
    sources = [doc.metadata.get("source", "unknown") for doc in chunks]
    vectors = embedding_model.embed_documents(texts)
    data = [
        {"text": t, "source": s, "vector": v}
        for t, s, v in zip(texts, sources, vectors)
    ]
    client.insert(
        collection_name=COLLECTION_NAME,
        data=data
    )
    # 刷到磁盘中
    client.flush(COLLECTION_NAME)
    # 刷到内存,使得attu显示
    client.load_collection(COLLECTION_NAME)


def search(query: str):
    if not query:
        return "未提供关键词"

    queries = rewrite_query(query)
    print(queries)
    all_results = []
    # 多查询语句用 set来进行查重
    seen = set()
    vectors = embedding_model.embed_documents(queries)
    print("向量检索请求构建....")
    vector_search = AnnSearchRequest(
        data=vectors,
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 16}},
        limit=10,
    )
    print("关键词检索请求构建....")
    bm25_search = AnnSearchRequest(
        data=queries,
        anns_field="sparse_vector",
        limit=10,
        param={"metric_type": "BM25"},
    )
    print("混合检索....")
    fused = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[vector_search, bm25_search],
        ranker=RRFRanker(k=60),
        limit=10,
        output_fields=["text", "source"],
    )

    return fine_rowing_context(query,fused)


def rewrite_query(query: str):
    if not query:
        return "询问为空"
    prompt = f"""你是搜索查询优化助手。请将以下用户问题重写为 3 个语义相近但表述不同的查询语句,用于在知识库中检索更多信息。

     要求:
     1. 第1条:更正式、完整的表述(补充隐含的主语/动词)
     2. 第2条:用同义词替换的表述
     3. 第3条:从不同角度/维度提问

     用户问题: {query}

     请直接返回3行文本,每行一个查询,不要序号、不要解释。
     """
    response = query_rewrite_model.invoke(prompt)
    variants = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
    # 保留原始查询 + LLM 生成的变体
    return [query] + variants[:3]
def fine_rowing_context(query: str, fused: List[List[dict]]) -> str:
    """对混合检索结果做精排,返回 top-k 最相关的上下文"""
    # 1.去重提取所有chunk数据
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
        return "未找到相关内容"
    # 构建精排prompt
    prompt = f"""用户问题: {query}
     以下是检索到的文档片段,请判断每个片段与用户问题的相关程度。
     返回每个文档的分数(0.0-1.0,越高越相关):
     {chunks}
     请按以下 JSON 格式返回,只返回 JSON,不要解释:
     {{"scores": [0.9, 0.3, 0.8, ...]}}
     """
    print("开始精排.....")
    response = fine_rowing_model.invoke(prompt)
    try:
        import json
        content = response.content.strip()
        # 处理可能的 markdown 代码块包裹
        if content.startswith("```"):
            content = content.strip("`").replace("json", "", 1).strip()
        scores_data = json.loads(content)
        scores = scores_data["scores"]
    except Exception:
        # 解析失败,直接按原始顺序返回
        scores = [1.0 - i * 0.1 for i in range(len(chunks))]

        # 4. 按分数排序,取 top-k
    scored_chunks = [
        {"score": scores[i], "text": chunks[i]["text"], "source": chunks[i]["source"]}
        for i in range(min(len(scores), len(chunks)))
    ]
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    top_chunks = scored_chunks[:3]  # 取前 3 条

    texts = [f"【来源: {c['source']}】(相关度: {c['score']:.2f})\n{c['text']}"
             for c in top_chunks]
    return "\n\n---\n\n".join(texts)


if __name__ == "__main__":
    prepare_milvus_env(reset=True)
    load_documents_in_milvus()
