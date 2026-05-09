import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

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

grade_model = ChatOpenAI(
    api_key=APIKEY,
    base_url=BASEURL,
    model=QUERY_REWRITE_MODEL_NAME,
    temperature=0,
)

fine_rowing_model = ChatOpenAI(
    api_key=APIKEY,
    base_url=BASEURL,
    model=FINE_ROWING_MODEL_NAME,
    max_retries=3,
)