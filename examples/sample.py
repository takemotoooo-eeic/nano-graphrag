import os

from dotenv import load_dotenv

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._storage import Neo4jStorage, QdrantStorage


load_dotenv()

neo4j_config = {
    "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
    "neo4j_auth": (
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", "neo4j"),
    ),
}

graph_func = GraphRAG(
    working_dir="./test_pdf",
    using_azure_openai=True,
    vector_db_storage_cls=QdrantStorage,
    graph_storage_cls=Neo4jStorage,
    addon_params=neo4j_config,
)

graph_func.insert("./files/kouhou00000.pdf")

# Perform global graphrag search
print(graph_func.query("What are the top themes in this story?"))

# Perform local graphrag search (I think is better and more scalable one)
print(
    graph_func.query(
        "What are the top themes in this story?", param=QueryParam(mode="local")
    )
)
