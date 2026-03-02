import os
import logging
import argparse
import json
from typing import Dict, List
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._storage import Neo4jStorage, QdrantStorage

# Load environment variables
load_dotenv()


def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source"""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions


def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    questions: Dict[str, List[dict]],
    sample: int,
    query_mode: str,
    use_neo4j_qdrant: bool,
):
    """Process a single corpus: index it with nano_graphrag and answer its questions."""
    logging.info(f"📚 Processing corpus: {corpus_name}")

    # Prepare output directory
    output_dir = os.path.join("./results/nano-graphrag", corpus_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")

    # Working directory per corpus (nano_graphrag stores index here)
    working_dir = os.path.join(base_dir, corpus_name)

    # Storage config (sample.py と同様: Neo4j + Qdrant またはデフォルト)
    graphrag_kwargs = {
        "working_dir": working_dir,
        "using_azure_openai": True,
        "using_langchain_langfuse": True,
    }
    if use_neo4j_qdrant:
        graphrag_kwargs["vector_db_storage_cls"] = QdrantStorage
        graphrag_kwargs["graph_storage_cls"] = Neo4jStorage
        graphrag_kwargs["addon_params"] = {
            "neo4j_url": os.environ.get("NEO4J_URL", "neo4j://localhost:7687"),
            "neo4j_auth": (
                os.environ.get("NEO4J_USER", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", "neo4j"),
            ),
        }
        logging.info("✅ Using Neo4j + Qdrant storage")
    else:
        logging.info("✅ Using default storage (NetworkX + NanoVectorDB)")

    # Initialize nano_graphrag GraphRAG (sample.py を参考)
    graph_func = GraphRAG(**graphrag_kwargs)

    # コーパスをインデクシング（文字列をそのまま insert）
    graph_func.insert(context)
    logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")

    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return

    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")

    # Query param (sample.py では local を推奨)
    param = QueryParam(mode=query_mode)

    # Process questions and generate answers
    results = []
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        try:
            # nano_graphrag の query は回答文字列を返す
            predicted_answer, context = graph_func.query(q["question"], param=param)
            # context は nano_graphrag では 1 回の query では取得しないため空リストで保存
            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": context,
                "evidence": q.get("evidence", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": predicted_answer,
                "ground_truth": q.get("answer", ""),
            })
        except Exception as e:
            logging.error(f"❌ Error processing question {q.get('id')}: {e}")
            results.append({
                "id": q["id"],
                "question": q.get("question", ""),
                "source": corpus_name,
                "error": str(e),
            })

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")


def main():
    # Dataset base: リポジトリの datasets/GraphRAG-Bechmark または --dataset_dir で上書き
    DEFAULT_DATASET_BASE = "./datasets/GraphRAG-Bechmark"
    SUBSET_PATHS = {
        "medical": {
            "corpus": os.path.join(DEFAULT_DATASET_BASE, "Corpus/medical.parquet"),
            "questions": os.path.join(DEFAULT_DATASET_BASE, "Questions/medical_questions.parquet"),
        },
        "novel": {
            "corpus": os.path.join(DEFAULT_DATASET_BASE, "Corpus/novel.parquet"),
            "questions": os.path.join(DEFAULT_DATASET_BASE, "Questions/novel_questions.parquet"),
        },
    }

    parser = argparse.ArgumentParser(
        description="GraphRAG (nano_graphrag): Index corpora and answer questions"
    )
    parser.add_argument(
        "--subset",
        required=True,
        choices=["medical", "novel"],
        help="Subset to process (medical or novel)",
    )
    parser.add_argument(
        "--base_dir",
        default="./examples/graphrag_workspace",
        help="Base working directory for nano_graphrag index",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Override dataset base directory (e.g. ./Datasets)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of questions to sample per corpus; also limits corpus to 1 doc if set",
    )
    parser.add_argument(
        "--query_mode",
        choices=["local", "global", "naive"],
        default="local",
        help="Query mode (local recommended for scalability)",
    )
    parser.add_argument(
        "--use_neo4j_qdrant",
        action="store_true",
        help="Use Neo4j + Qdrant storage (requires NEO4J_* and Qdrant). Otherwise use default file-based storage.",
    )

    args = parser.parse_args()

    # Override dataset paths if --dataset_dir is set
    if args.dataset_dir:
        SUBSET_PATHS["medical"]["corpus"] = os.path.join(args.dataset_dir, "Corpus", "medical.parquet")
        SUBSET_PATHS["medical"]["questions"] = os.path.join(args.dataset_dir, "Questions", "medical_questions.parquet")
        SUBSET_PATHS["novel"]["corpus"] = os.path.join(args.dataset_dir, "Corpus", "novel.parquet")
        SUBSET_PATHS["novel"]["questions"] = os.path.join(args.dataset_dir, "Questions", "novel_questions.parquet")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"graphrag_{args.subset}.log"),
        ],
    )

    logging.info(f"🚀 Starting nano_graphrag processing for subset: {args.subset}")

    if args.subset not in SUBSET_PATHS:
        logging.error(f"❌ Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return

    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    os.makedirs(args.base_dir, exist_ok=True)

    # Load corpus
    try:
        corpus_dataset = load_dataset("parquet", data_files=corpus_path, split="train")
        corpus_data = [
            {"corpus_name": item["corpus_name"], "context": item["context"]}
            for item in corpus_dataset
        ]
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return

    if args.sample:
        corpus_data = corpus_data[:1]

    # Load questions
    try:
        questions_dataset = load_dataset("parquet", data_files=questions_path, split="train")
        question_data = [
            {
                "id": item["id"],
                "source": item["source"],
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item["question_type"],
                "evidence": item["evidence"],
            }
            for item in questions_dataset
        ]
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded {len(question_data)} questions from {questions_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return

    # Process each corpus sequentially (nano_graphrag の insert/query は同期 API)
    for item in corpus_data:
        process_corpus(
            corpus_name=item["corpus_name"],
            context=item["context"][:1000],
            base_dir=args.base_dir,
            questions=grouped_questions,
            sample=args.sample,
            query_mode=args.query_mode,
            use_neo4j_qdrant=args.use_neo4j_qdrant,
        )


if __name__ == "__main__":
    main()
