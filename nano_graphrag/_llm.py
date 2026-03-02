import json
import numpy as np
from typing import Optional, List, Any, Callable

import aioboto3
from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

global_openai_async_client = None
global_azure_openai_async_client = None
global_amazon_bedrock_async_client = None


def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        # Configure Azure OpenAI client using environment variables
        # These are typically loaded from a .env file in user code.
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        global_azure_openai_async_client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=azure_api_version,
        )
    return global_azure_openai_async_client


def get_amazon_bedrock_async_client_instance():
    global global_amazon_bedrock_async_client
    if global_amazon_bedrock_async_client is None:
        global_amazon_bedrock_async_client = aioboto3.Session()
    return global_amazon_bedrock_async_client


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
def _add_usage_to_tracker(tracker, response) -> None:
    """OpenAI/Azure の response.usage を tracker に加算する。"""
    if tracker is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    tt = getattr(usage, "total_tokens", None)
    if tt is None:
        tt = pt + ct
    tracker.add_usage({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt})


async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    tracker = kwargs.pop("tracker", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    _add_usage_to_tracker(tracker, response)

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": [{"text": prompt}]})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    inference_config = {
        "temperature": 0,
        "maxTokens": 4096 if "max_tokens" not in kwargs else kwargs["max_tokens"],
    }

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        if system_prompt:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
                system=[{"text": system_prompt}]
            )
        else:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
            )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response["output"]["message"]["content"][0]["text"], "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response["output"]["message"]["content"][0]["text"]


def create_amazon_bedrock_complete_function(model_id: str) -> Callable:
    """
    Factory function to dynamically create completion functions for Amazon Bedrock

    Args:
        model_id (str): Amazon Bedrock model identifier (e.g., "us.anthropic.claude-3-sonnet-20240229-v1:0")

    Returns:
        Callable: Generated completion function
    """
    async def bedrock_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Any] = [],
        **kwargs
    ) -> str:
        return await amazon_bedrock_complete_if_cache(
            model_id,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    # Set function name for easier debugging
    bedrock_complete.__name__ = f"{model_id}_complete"
    
    return bedrock_complete


async def best_llm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def cheap_llm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_embedding(texts: list[str]) -> np.ndarray:
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        embeddings = []
        for text in texts:
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": 1024,
                }
            )
            response = await bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0", body=body,
            )
            response_body = await response.get("body").read()
            embeddings.append(json.loads(response_body))
    return np.array([dp["embedding"] for dp in embeddings])


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(texts: list[str], **kwargs) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    tracker = kwargs.pop("tracker", None)
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    if tracker is not None and getattr(response, "usage", None) is not None:
        u = response.usage
        total = getattr(u, "total_tokens", None) or (getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0))
        if total is not None:
            tracker.add_usage({"prompt_tokens": total, "completion_tokens": 0, "total_tokens": total})
    return np.array([dp.embedding for dp in response.data])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    tracker = kwargs.pop("tracker", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )
    _add_usage_to_tracker(tracker, response)

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_best_llm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    deployment_name = os.getenv("AZURE_BEST_LLM_DEPLOYMENT", "gpt-4o")
    return await azure_openai_complete_if_cache(
        deployment_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_cheap_llm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    deployment_name = os.getenv("AZURE_CHEAP_LLM_DEPLOYMENT", "gpt-4o-mini")
    return await azure_openai_complete_if_cache(
        deployment_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=3072, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(texts: list[str], **kwargs) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    tracker = kwargs.pop("tracker", None)
    embedding_deployment = os.getenv(
        "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
    )
    response = await azure_openai_client.embeddings.create(
        model=embedding_deployment, input=texts, encoding_format="float"
    )
    if tracker is not None and getattr(response, "usage", None) is not None:
        u = response.usage
        total = getattr(u, "total_tokens", None) or (getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0))
        if total is not None:
            tracker.add_usage({"prompt_tokens": total, "completion_tokens": 0, "total_tokens": total})
    return np.array([dp.embedding for dp in response.data])
