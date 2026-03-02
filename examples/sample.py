import os

from dotenv import load_dotenv

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._storage import Neo4jStorage, QdrantStorage
from nano_graphrag._parser.mm_pdf_parser import MultimodalPdfParser
from nano_graphrag.tracker import TokenTracker, TimeTracker


load_dotenv()

neo4j_config = {
    "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
    "neo4j_auth": (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "neo4j"),
    ),
}

# トークン数・時間を計測するトラッカー（任意）
token_tracker = TokenTracker()
time_tracker = TimeTracker()

graph_func = GraphRAG(
    working_dir="./graphrag_bench",
    using_azure_openai=True,
    vector_db_storage_cls=QdrantStorage,
    graph_storage_cls=Neo4jStorage,
    addon_params=neo4j_config,
    token_tracker=token_tracker,
    time_tracker=time_tracker,
)

graph_func.insert(
    "./files/public_document_ministry00005.pdf",
    parsers=[MultimodalPdfParser()],
)
print("[After insert]", token_tracker)
print("[After insert]", time_tracker)

# Perform global graphrag search
print(graph_func.query("What are the top themes in this story?"))
print("[After query]", token_tracker)
print("[After query]", time_tracker)

# Perform local graphrag search (I think is better and more scalable one)
print(
    graph_func.query(
        "What are the top themes in this story?", param=QueryParam(mode="local")
    )
)
print("[Final]", token_tracker)
print("[Final]", time_tracker)
