import asyncio
import argparse
import json
import numpy as np
import os
from typing import Dict, List, Any
from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv


from .metrics.context_relevance import compute_context_relevance
from .metrics.evidence_recall import compute_evidence_recall

load_dotenv()

SEED = 42

# Azure OpenAI 用の環境変数（必須）
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

async def evaluate_dataset(
    dataset: Dataset,
    llm: Any,
    embeddings: Embeddings,
    max_concurrent: int = 1,
    detailed_output: bool = False
) -> Dict[str, Any]:
    """Evaluate context relevance and recall for a dataset"""
    results = {
        "context_relevancy": [],
        "evidence_recall": []
    }
    detailed_results = [] if detailed_output else None
    
    ids = dataset["id"]
    questions = dataset["question"]
    contexts_list = dataset["contexts"]
    evidences = dataset["evidences"]

    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(i):
        async with semaphore:
            sample_metrics = await evaluate_sample(
                question=questions[i],
                contexts=contexts_list[i],
                evidences=evidences[i],
                llm=llm,
                embeddings=embeddings
            )
            if detailed_output:
                return {
                    "id": ids[i],
                    "question": questions[i],
                    "contexts": contexts_list[i],
                    "evidences": evidences[i],
                    "metrics": sample_metrics
                }
            return sample_metrics

    tasks = [evaluate_with_semaphore(i) for i in range(total_samples)]
    sample_results = []
    completed = 0
    
    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            if detailed_output and detailed_results is not None:
                detailed_results.append(result)
                # metrics aggregation (guard types for linters)
                if isinstance(result, dict):
                    metrics_dict = result.get("metrics")
                    if isinstance(metrics_dict, dict):
                        for metric, score in metrics_dict.items():
                            if isinstance(score, (int, float)) and not np.isnan(score):
                                results[metric].append(score)
            else:
                sample_results.append(result)
                if isinstance(result, dict):
                    for metric, score in result.items():
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            results[metric].append(score)
            completed += 1
            print(f"✅ Completed sample {completed}/{total_samples} - {(completed/total_samples)*100:.1f}%")
        except Exception as e:
            print(f"❌ Sample failed: {e}")
            completed += 1
    
    avg_results = {
        "context_relevancy": np.nanmean(results["context_relevancy"]),
        "evidence_recall": np.nanmean(results["evidence_recall"])
    }
    
    if detailed_output:
        return {
            "average_scores": avg_results,
            "detailed": detailed_results
        }
    else:
        return avg_results


async def evaluate_sample(
    question: str,
    contexts: List[str],
    evidences: List[str],
    llm: Any,
    embeddings: Embeddings
) -> Dict[str, float]:
    """Evaluate retrieval metrics for a single sample"""
    # Evaluate both metrics in parallel
    relevance_task = compute_context_relevance(question, contexts, llm)
    recall_task = compute_evidence_recall(question, contexts, evidences, llm)
    
    # Wait for both tasks to complete
    relevance_score, recall_score = await asyncio.gather(relevance_task, recall_task)

    print(f"Relevance Score: {relevance_score}, Recall Score: {recall_score}")

    return {
        "context_relevancy": relevance_score,
        "evidence_recall": recall_score
    }

def _create_azure_llm(deployment_name: str) -> AzureChatOpenAI:
    """Azure OpenAI の LLM を固定で作成"""
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise ValueError(
            "Azure OpenAI の環境変数が未設定です: "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY を設定してください"
        )
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=deployment_name,
        temperature=0.0,
        max_retries=3,
        request_timeout=30,
        model_kwargs={
            "top_p": 1,
            "seed": SEED,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )


def _create_azure_embeddings(deployment_name: str) -> AzureOpenAIEmbeddings:
    """Azure OpenAI の Embedding を固定で作成"""
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise ValueError(
            "Azure OpenAI の環境変数が未設定です: "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY を設定してください"
        )
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=deployment_name,
    )


async def main(args: argparse.Namespace):
    """Main retrieval evaluation function（Azure LLM / Embedding 固定）"""
    chat_deployment = args.chat_deployment or AZURE_CHAT_DEPLOYMENT
    embedding_deployment = args.embedding_deployment or AZURE_EMBEDDING_DEPLOYMENT

    llm = _create_azure_llm(chat_deployment)
    embedding: Embeddings = _create_azure_embeddings(embedding_deployment)

    # Load evaluation data
    print(f"Loading evaluation data from {args.data_file}...")
    with open(args.data_file, 'r') as f:
        file_data = json.load(f)  # List of question items
    
    # Group data by question type
    grouped_data = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        if q_type not in grouped_data:
            grouped_data[q_type] = []
        grouped_data[q_type].append(item)
    
    all_results = {}
    
    # Evaluate each question type
    for question_type in list(grouped_data.keys()):
        print(f"\n{'='*50}")
        print(f"Evaluating question type: {question_type}")
        print(f"{'='*50}")
        
        # Prepare data from grouped items
        group_items = grouped_data[question_type]

        ids = [item['id'] for item in group_items]
        questions = [item['question'] for item in group_items]
        evidences = [item['evidence'] for item in group_items]
        contexts = [item['context'] for item in group_items]
        
        # Create dataset
        data = {
            "id": ids,
            "question": questions,
            "contexts": contexts,
            "evidences": evidences
        }
        dataset = Dataset.from_dict(data)
        
        # If sample
        if args.num_samples:
            dataset = dataset.select([i for i in list(range(args.num_samples))])
        
        # Perform evaluation
        results = await evaluate_dataset(
            dataset=dataset,
            llm=llm, 
            embeddings=embedding,
            detailed_output=args.detailed_output
        )
        
        all_results[question_type] = results
        print(f"\nResults for {question_type}:")
        if args.detailed_output:
            for metric, score in results["average_scores"].items():
                print(f"  {metric}: {score:.4f}")
        else:
            for metric, score in results.items():
                print(f"  {metric}: {score:.4f}")
            
    # Save final results
    if args.output_file:
        print(f"\nSaving results to {args.output_file}...")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

    print("\nEvaluation complete.")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Azure OpenAI は環境変数で指定。オプションでデプロイ名を上書き可能
    parser.add_argument(
        "--chat_deployment",
        type=str,
        default=None,
        help=f"Azure Chat デプロイ名（未指定時は環境変数 AZURE_OPENAI_CHAT_DEPLOYMENT または {AZURE_CHAT_DEPLOYMENT}）",
    )
    parser.add_argument(
        "--embedding_deployment",
        type=str,
        default=None,
        help=f"Azure Embedding デプロイ名（未指定時は環境変数 AZURE_OPENAI_EMBEDDING_DEPLOYMENT または {AZURE_EMBEDDING_DEPLOYMENT}）",
    )
    parser.add_argument(
        "--data_file", 
        type=str,
        required=True,
        help="Path to JSON file containing evaluation data"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str,
        default="retrieval_results.json",
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int,
        default=None,
        help="Number of samples per question type to evaluate (optional)"
    )

    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Whether to include detailed output"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function
    asyncio.run(main(args))