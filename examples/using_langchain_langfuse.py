"""
LangChain + Langfuse で LLM リクエストをモニタリングする例。

実行前に以下をインストールしてください:
  pip install langchain-openai langfuse

環境変数:
  - OPENAI_API_KEY (または Azure の場合は AZURE_OPENAI_*)
  - Langfuse: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
  - (任意) LANGFUSE_BASE_URL 例: https://cloud.langfuse.com
"""

import os

from dotenv import load_dotenv

from nano_graphrag import GraphRAG, QueryParam

load_dotenv()

# GraphRAG で LangChain + Langfuse を有効にすると、
# すべての LLM 呼び出しが Langfuse にトレースされます
graph_func = GraphRAG(
    working_dir="./graphrag_bench_langfuse",
    using_azure_openai=os.environ.get("AZURE_OPENAI", "").lower() in ("1", "true", "yes"),
    using_langchain_langfuse=True,  # LangChain で実行し Langfuse でモニタリング
)

# テキストを 1 件だけ挿入してクエリ（トレース確認用）
graph_func.insert("Microsoft GraphRAG is a research project for knowledge extraction and question answering.")

print(graph_func.query("What is GraphRAG?", param=QueryParam(mode="local")))

# 短い処理後、Langfuse にイベントを送るため flush 推奨（アプリ終了時）
try:
    from langfuse import Langfuse
    Langfuse().flush()
except Exception:
    pass
