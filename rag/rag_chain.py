from langchain_core.tools import tool

from rag.rag_index_mivus import search

@tool
def rag_search(query:str):
    """用户搜索校园信息时,使用这个工具获取相关的背景信息"""
    results = search(query)
    return results
