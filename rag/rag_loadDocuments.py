import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, Function, FunctionType


def load_documents(directory: str):
    """遍历目录,加载所有支持的文档"""
    all_docs = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = file.lower().split(".")[-1]
            if ext == "txt" or ext == "md":
                loader = TextLoader(file_path, encoding="utf-8")
                all_docs.extend(loader.load())
            if ext == "pdf":
                loader = PyPDFLoader(file_path)
                all_docs.extend(loader.load())

    print(f"共加载 {len(all_docs)} 个文档")
    return all_docs


def split_documents(docs, chunk_size=300, chunk_overlap=30):
    """把文档切成小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"共切分成 {len(chunks)} 个片段")
    return chunks

def get_chunks(directory:str,chunk_size=300, chunk_overlap=30):
    """获取指定目录下的所有文档"""
    docs = load_documents(directory)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    return chunks

# def store_to_milvus(chunks, collection_name="college_rag"):
#     """把文档片段向量化并存储到milvus"""
#     load_dotenv()
#
#     # 1. 批量生成向量
#     print("正在批量生成向量...")
#     texts = [doc.page_content for doc in chunks]
#     vectors = embedding_model.embed_documents(texts)
#     print(f"已生成 {len(vectors)} 个向量, 维度: {len(vectors[0])}")
#
#     # 2. 连接 Milvus
#     print("正在连接 Milvus...")
#     connections.connect(host="192.168.56.10", port=19530)
#
#     # 3. 定义 schema (增加 sparse_vector 字段用于 BM25 关键词检索)
#     fields = [
#         FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#         FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
#         FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
#         FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=len(vectors[0])),
#         FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
#     ]
#
#     # BM25 Function: 自动将 text 分词并生成 sparse_vector
#     bm25_func = Function(
#         name="bm25_text_embedding",
#         function_type=FunctionType.BM25,
#         input_field_names=["text"],
#         output_field_names=["sparse_vector"],
#     )
#
#     schema = CollectionSchema(fields, description="college rag collection", functions=[bm25_func])
#
#     # 4. 删除旧 collection
#     print("正在创建/重建 collection...")
#     from pymilvus import utility
#     if utility.has_collection(collection_name):
#         utility.drop_collection(collection_name)
#
#     collection = Collection(collection_name, schema)
#
#     # 5. 插入数据(BM25 Function 会自动从 text 生成 sparse_vector,不需要手动提供)
#     print("正在插入数据...")
#     entities = [
#         {"text": doc.page_content, "source": doc.metadata.get("source", "unknown"), "vector": vec}
#         for doc, vec in zip(chunks, vectors)
#     ]
#     collection.insert(entities)
#
#     # 6. 创建索引(向量索引 + BM25 稀疏向量索引)
#     print("正在创建向量索引...")
#     index_params = {
#         "index_type": "IVF_FLAT",
#         "metric_type": "L2",
#         "params": {"nlist": 128},
#     }
#     collection.create_index("vector", index_params)
#
#     print("正在创建 BM25 索引...")
#     collection.create_index(
#         "sparse_vector",
#         {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"},
#     )
#
#     collection.load()
#
#     print(f"已存储 {len(chunks)} 个片段到 Milvus")
#     return collection
#
#
# if __name__ == "__main__":
#     docs = load_documents("D:/develop/code/python/langchain-ai-learning/college_rag_data")
#     chunks = split_documents(docs)
#     store_to_milvus_byMilvus(chunks)
