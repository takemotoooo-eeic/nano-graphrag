from nano_graphrag import GraphRAG, QueryParam
from dotenv import load_dotenv

load_dotenv()

graph_func = GraphRAG(working_dir="./dickens", using_azure_openai=True)

with open("./book.txt") as f:
    graph_func.insert(f.read())

# Perform global graphrag search
print(graph_func.query("What are the top themes in this story?"))

# Perform local graphrag search (I think is better and more scalable one)
print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="local")))
