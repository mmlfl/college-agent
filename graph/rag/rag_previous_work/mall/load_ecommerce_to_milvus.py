"""将电商商品数据灌入 Milvus 向量库 — 商品描述 + 评价摘要作为 RAG 检索源"""

import os
import json

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient, DataType, Function, FunctionType
from pymilvus.milvus_client.index import IndexParams
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

load_dotenv()

# ================== MySQL 连接 ==================
DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
    f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}"
    f"/{os.getenv('MYSQL_DATABASE')}"
)
db_engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=db_engine)
Base = declarative_base()


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    category = Column(String(50))
    brand = Column(String(50))
    description = Column(Text)


class ProductReview(Base):
    __tablename__ = "product_reviews"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer)
    user_name = Column(String(50))
    rating = Column(Integer)
    content = Column(Text)


# ================== Milvus 连接 ==================
MILVUS_URI = os.getenv("MILVUS_URI", "http://192.168.56.10:19530")
DB_NAME = "lfl_mall"
COLLECTION_NAME = "product_review_data"
RESET_MODE = True

APIKEY = os.getenv("DASHSCOPE_API_KEY")
BASEURL = os.getenv("DASHSCOPE_BASE_URL")

embedding_model = OpenAIEmbeddings(
    api_key=APIKEY,
    base_url=BASEURL,
    model="text-embedding-v2",
    check_embedding_ctx_length=False,
)

client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)


def prepare_milvus_env(reset=False):
    client_temp = MilvusClient(uri=MILVUS_URI)
    current_dbs = client_temp.list_databases()
    print(f"当前数据库: {current_dbs}")

    if DB_NAME not in current_dbs:
        client_temp.create_database(DB_NAME)
        print(f"创建数据库 '{DB_NAME}'")

    client_temp.using_database(DB_NAME)
    if reset and client_temp.has_collection(COLLECTION_NAME):
        print(f"删除旧 collection: {COLLECTION_NAME}")
        client_temp.drop_collection(COLLECTION_NAME)
    client_temp.close()


def create_collection():
    schema = client.create_schema()
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="product_id", datatype=DataType.INT64)
    schema.add_field(field_name="product_name", datatype=DataType.VARCHAR, max_length=500)
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
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    index_params.add_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 256},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    print(f"创建 collection: {COLLECTION_NAME}")


def build_documents():
    session = Session()
    docs = []

    products = session.query(Product).all()
    for product in products:
        docs.append({
            "text": f"【{product.name}】商品描述：{product.description}",
            "source": f"product_{product.id}",
            "doc_type": "description",
            "product_id": product.id,
            "product_name": product.name,
        })

        reviews = session.query(ProductReview).filter(
            ProductReview.product_id == product.id
        ).all()
        for review in reviews:
            docs.append({
                "text": f"【{product.name}】用户评价({review.rating}星)：{review.content}",
                "source": f"review_{review.id}",
                "doc_type": "review",
                "product_id": product.id,
                "product_name": product.name,
            })

    session.close()
    return docs


def load_to_milvus(docs):
    texts = [d["text"] for d in docs]
    print(f"开始 embedding {len(texts)} 条文档...")

    # DashScope 限制单次 batch <= 25
    batch_size = 25
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = embedding_model.embed_documents(batch)
        all_vectors.extend(vectors)
        print(f"  batch {i // batch_size + 1}: {len(batch)} 条")

    data = [
        {
            "text": d["text"],
            "source": d["source"],
            "doc_type": d["doc_type"],
            "product_id": d["product_id"],
            "product_name": d["product_name"],
            "vector": v,
        }
        for d, v in zip(docs, all_vectors)
    ]

    client.insert(collection_name=COLLECTION_NAME, data=data)
    client.flush(COLLECTION_NAME)
    client.load_collection(COLLECTION_NAME)
    print(f"已加载 {len(data)} 条文档到 Milvus")


if __name__ == "__main__":
    prepare_milvus_env(reset=RESET_MODE)
    create_collection()
    docs = build_documents()
    load_to_milvus(docs)
    print("数据加载完成！")
