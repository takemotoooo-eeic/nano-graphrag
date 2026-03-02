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

from .metrics import compute_answer_correctness
from .metrics.rouge import compute_rouge_score
from .metrics.coverage import compute_coverage_score
from .metrics.faithfulness import compute_faithfulness_score

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
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings,
    max_concurrent: int = 3,  # Limit concurrent evaluations
    detailed_output: bool = False
) -> Dict[str, Any]:
    """Evaluate the metric scores on the entire dataset."""
    results = {metric: [] for metric in metrics}
    detailed_results = [] if detailed_output else None
    
    ids = dataset["id"]
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]
    
    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(i):
        async with semaphore:
            sample_metrics = await evaluate_sample(
                question=questions[i],
                answer=answers[i],
                contexts=contexts_list[i],
                ground_truth=ground_truths[i],
                metrics=metrics,
                llm=llm,
                embeddings=embeddings
            )
            if detailed_output:
                return {
                    "id": ids[i],
                    "question": questions[i],
                    "ground_truth": ground_truths[i],
                    "generated_answer": answers[i],
                    "contexts": contexts_list[i],
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
    
    avg_results = {metric: np.nanmean(scores) for metric, scores in results.items()}
    
    if detailed_output:
        return {
            "average_scores": avg_results,
            "detailed": detailed_results
        }
    else:
        return avg_results

async def evaluate_sample(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings
) -> Dict[str, float]:
    """Evaluate the metric scores for a single sample."""
    results = {}
    
    tasks = {}
    if "rouge_score" in metrics:
        tasks["rouge_score"] = compute_rouge_score(answer, ground_truth)
    
    if "answer_correctness" in metrics:
        tasks["answer_correctness"] = compute_answer_correctness(
            question, answer, ground_truth, llm, embeddings
        )
    
    if "coverage_score" in metrics:
        tasks["coverage_score"] = compute_coverage_score(
            question, ground_truth, answer, llm
        )
    
    if "faithfulness" in metrics:
        tasks["faithfulness"] = compute_faithfulness_score(
            question, answer, contexts, llm
        )
    
    task_results = await asyncio.gather(*tasks.values())
    
    for i, metric in enumerate(tasks.keys()):
        results[metric] = task_results[i]
    
    return results

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
    """Main evaluation function（Azure LLM / Embedding 固定）"""
    chat_deployment = args.chat_deployment or AZURE_CHAT_DEPLOYMENT
    embedding_deployment = args.embedding_deployment or AZURE_EMBEDDING_DEPLOYMENT

    llm = _create_azure_llm(chat_deployment)
    embedding: Embeddings = _create_azure_embeddings(embedding_deployment)

    # Load evaluation data
    print(f"Loading evaluation data from {args.data_file}...")
    with open(args.data_file, 'r') as f:
        file_data = json.load(f)  # Now a list of question items
    
    # Define the evaluation metrics for each question type
    metric_config = {
        'Fact Retrieval': ["rouge_score", "answer_correctness"],
        'Complex Reasoning': ["rouge_score", "answer_correctness"],
        'Contextual Summarize': ["answer_correctness", "coverage_score"],
        'Creative Generation': ["answer_correctness", "coverage_score", "faithfulness"]
    }
    
    # Group data by question type
    grouped_data = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        if q_type not in grouped_data:
            grouped_data[q_type] = []
        grouped_data[q_type].append(item)
    
    all_results = {}
    
    # Evaluate each found question type (only those in metric_config)
    for question_type in list(grouped_data.keys()):
        # Skip types not defined in metric_config
        if question_type not in metric_config:
            print(f"Skipping undefined question type: {question_type}")
            continue
            
        print(f"\n{'='*50}")
        print(f"Evaluating question type: {question_type}")
        print(f"{'='*50}")
        
        # Prepare data from grouped items
        group_items = grouped_data[question_type]
        ids = [item['id'] for item in group_items]
        questions = [item['question'] for item in group_items]
        ground_truths = [item['ground_truth'] for item in group_items]
        answers = [item['generated_answer'] for item in group_items]
        contexts = [item['context'] for item in group_items]
        
        # Create dataset
        data = {
            "id": ids,
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)

        # If sample
        if args.num_samples:
            dataset = dataset.select([i for i in list(range(args.num_samples))])

        # Perform evaluation
        results = await evaluate_dataset(
            dataset=dataset,
            metrics=metric_config[question_type],
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
    parser = argparse.ArgumentParser(
        description="Evaluate RAG performance using various metrics",
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
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use for evaluation"
    )

    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Whether to include detailed output"
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(args))