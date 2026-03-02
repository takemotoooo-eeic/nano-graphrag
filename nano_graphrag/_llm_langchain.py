"""
LangChain + Langfuse を用いた LLM / Embedding のラッパー。

既存の nano_graphrag のインターフェース (async complete, async embedding)
を満たしつつ、LangChain で実行し Langfuse でトレースします。

利用には langchain-openai, langfuse のインストールが必要です:
  pip install langchain-openai langfuse
Azure 利用時は上記に加え環境変数で AZURE_OPENAI_* を設定してください。
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

import numpy as np

from ._utils import EmbeddingFunc, compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

# Optional imports
try:
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    from langchain_core.runnables import Runnable
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False

try:
    from openai import ContentFilterFinishReasonError
except ImportError:
    ContentFilterFinishReasonError = None  # type: ignore[misc, assignment]


def _check_deps():
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain を使用するには langchain-openai をインストールしてください: pip install langchain-openai"
        )
    if not _LANGFUSE_AVAILABLE:
        raise ImportError(
            "Langfuse でモニタリングするには langfuse をインストールしてください: pip install langfuse"
        )


def get_langfuse_handler():
    """Langfuse の CallbackHandler を返す。LangChain の invoke/ainvoke に config={"callbacks": [handler]} で渡す。"""
    _check_deps()
    return LangfuseCallbackHandler()


def _messages_from_prompt(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Any]] = None,
) -> List[BaseMessage]:
    """nano_graphrag 形式 (prompt, system_prompt, history_messages) を LangChain の messages に変換する。"""
    history_messages = history_messages or []
    msgs: List[BaseMessage] = []
    if system_prompt:
        msgs.append(SystemMessage(content=system_prompt))
    for m in history_messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            msgs.append(SystemMessage(content=content))
        elif role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            from langchain_core.messages import AIMessage
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=prompt))
    return msgs


def create_langchain_llm_complete(
    model_name: str = "gpt-4o",
    *,
    use_azure: bool = False,
    deployment_name: Optional[str] = None,
    langfuse_handler: Optional[Any] = None,
) -> Callable[..., Any]:
    """
    LangChain + Langfuse を使う LLM complete 関数を返す。

    返り値の関数は既存の nano_graphrag の best_model_func / cheap_model_func と
    同じシグネチャ: async (prompt, system_prompt=None, history_messages=[], **kwargs) -> str
    """

    _check_deps()

    if use_azure:
        deployment = deployment_name or os.getenv("AZURE_BEST_LLM_DEPLOYMENT", "gpt-4o")
        llm: Runnable = AzureChatOpenAI(
            deployment_name=deployment,
            model_name=model_name or deployment,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            temperature=0,
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
        )

    handler = langfuse_handler or get_langfuse_handler()
    config = {"callbacks": [handler]}

    async def _complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Any]] = None,
        **kwargs,
    ) -> str:
        hashing_kv: Optional[BaseKVStorage] = kwargs.pop("hashing_kv", None)
        tracker = kwargs.pop("tracker", None)
        max_tokens = kwargs.pop("max_tokens", None)
        response_format = kwargs.pop("response_format", None)

        messages = _messages_from_prompt(prompt, system_prompt, history_messages)

        if hashing_kv is not None:
            args_hash = compute_args_hash(model_name or "default", messages)
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                return cached["return"]

        bound = llm
        if max_tokens is not None:
            bound = bound.bind(max_tokens=max_tokens)
        if response_format is not None:
            bound = bound.bind(response_format=response_format)

        try:
            result = await bound.ainvoke(messages, config=config)
            content = result.content if hasattr(result, "content") else str(result)
            if tracker is not None and hasattr(result, "response_metadata"):
                meta = getattr(result, "response_metadata", {}) or {}
                usage = meta.get("usage", meta.get("token_usage"))
                if usage:
                    pt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                    ct = usage.get("completion_tokens", usage.get("output_tokens", 0))
                    tt = usage.get("total_tokens") or (pt + ct)
                    tracker.add_usage({"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt})
        except ValueError as e:
            # Azure のコンテンツフィルター等でレスポンスが返らない場合は空文字を返す
            if "content filter" in str(e).lower():
                content = ""
            else:
                raise
        except Exception as e:
            # OpenAI ContentFilterFinishReasonError 等のコンテンツフィルター系は空文字を返す
            if ContentFilterFinishReasonError is not None and isinstance(e, ContentFilterFinishReasonError):
                content = ""
            else:
                err_msg = str(e).lower()
                if "content filter" in err_msg or "contentfilter" in err_msg:
                    content = ""
                else:
                    raise

        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": content, "model": model_name or "default"}}
            )
            await hashing_kv.index_done_callback()

        return content

    return _complete


def create_langchain_llm_complete_azure(
    deployment_name: Optional[str] = None,
    model_name: Optional[str] = None,
    langfuse_handler: Optional[Any] = None,
) -> Callable[..., Any]:
    """Azure OpenAI 用の LangChain complete 関数を返す。"""
    return create_langchain_llm_complete(
        model_name or os.getenv("AZURE_BEST_LLM_DEPLOYMENT", "gpt-4o"),
        use_azure=True,
        deployment_name=deployment_name or os.getenv("AZURE_BEST_LLM_DEPLOYMENT", "gpt-4o"),
        langfuse_handler=langfuse_handler,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def langchain_openai_embedding(
    texts: List[str],
    *,
    langfuse_handler: Optional[Any] = None,
    tracker: Optional[Any] = None,
    **kwargs: Any,
) -> np.ndarray:
    """LangChain OpenAI Embeddings + Langfuse。tracker があれば usage を加算。"""
    _check_deps()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    handler = langfuse_handler or get_langfuse_handler()
    config = {"callbacks": [handler]}
    try:
        vecs = await embeddings.aembed_documents(texts, config=config)
    except TypeError:
        vecs = await embeddings.aembed_documents(texts)
    _add_embedding_usage_if_available(embeddings, texts, tracker)
    return np.array(vecs, dtype=np.float64)


def _add_embedding_usage_if_available(embeddings: Any, texts: List[str], tracker: Optional[Any]) -> None:
    """LangChain Embeddings のレスポンスから token usage が取れれば tracker に加算する。"""
    if tracker is None:
        return
    try:
        if hasattr(embeddings, "get_num_tokens_from_messages"):
            n = sum(embeddings.get_num_tokens_from_messages([t]) for t in texts)
            tracker.add_usage({"prompt_tokens": n, "completion_tokens": 0, "total_tokens": n})
    except Exception:
        pass


@wrap_embedding_func_with_attrs(embedding_dim=3072, max_token_size=8192)
async def langchain_azure_openai_embedding(
    texts: List[str],
    *,
    langfuse_handler: Optional[Any] = None,
    tracker: Optional[Any] = None,
    **kwargs: Any,
) -> np.ndarray:
    """LangChain Azure OpenAI Embeddings + Langfuse。tracker があれば usage を加算。"""
    _check_deps()
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=deployment,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )
    handler = langfuse_handler or get_langfuse_handler()
    config = {"callbacks": [handler]}
    try:
        vecs = await embeddings.aembed_documents(texts, config=config)
    except TypeError:
        vecs = await embeddings.aembed_documents(texts)
    _add_embedding_usage_if_available(embeddings, texts, tracker)
    return np.array(vecs, dtype=np.float64)


def get_langchain_langfuse_complete_factories(
    use_azure: bool = False,
    best_model_name: str = "gpt-4o",
    cheap_model_name: str = "gpt-4o-mini",
):
    """
    GraphRAG に渡す best_model_func / cheap_model_func を LangChain+Langfuse 版で返す。

    Returns:
        (best_model_func, cheap_model_func) のタプル。
        これらを GraphRAG(..., best_model_func=..., cheap_model_func=...) に渡す。
    """
    handler = get_langfuse_handler()
    best = create_langchain_llm_complete(
        best_model_name, use_azure=use_azure, langfuse_handler=handler
    )
    cheap = create_langchain_llm_complete(
        cheap_model_name, use_azure=use_azure, langfuse_handler=handler
    )
    return best, cheap


def get_langchain_langfuse_embedding(use_azure: bool = False) -> EmbeddingFunc:
    """
    GraphRAG に渡す embedding_func を LangChain+Langfuse 版で返す。
    EmbeddingFunc 互換（embedding_dim, max_token_size 付き）。tracker は **kwargs で渡される。
    """
    handler = get_langfuse_handler()
    if use_azure:
        async def _emb(texts: List[str], **kwargs: Any) -> np.ndarray:
            return await langchain_azure_openai_embedding(
                texts, langfuse_handler=handler, **kwargs
            )
        return EmbeddingFunc(embedding_dim=3072, max_token_size=8192, func=_emb)
    else:
        async def _emb(texts: List[str], **kwargs: Any) -> np.ndarray:
            return await langchain_openai_embedding(
                texts, langfuse_handler=handler, **kwargs
            )
        return EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=_emb)
