import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast



from ._llm import (
    amazon_bedrock_embedding,
    create_amazon_bedrock_complete_function,
    best_llm_complete,
    cheap_llm_complete,
    openai_embedding,
    azure_best_llm_complete,
    azure_openai_embedding,
    azure_cheap_llm_complete,
)
from ._op import (
    chunking_by_token_size,
    extract_entities,
    generate_community_report,
    get_chunks,
    local_query,
    global_query,
    naive_query,
)
from ._parser import parse_file
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
    TokenizerWrapper,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
    BaseParser,
)


def _wrap_embedding_with_tracker(embedding_func: EmbeddingFunc, tracker: Any) -> EmbeddingFunc:
    """embedding 呼び出し時に tracker を kwargs で渡すラッパー。"""
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs["tracker"] = tracker
        return await embedding_func(*args, **kwargs)
    return EmbeddingFunc(
        embedding_dim=embedding_func.embedding_dim,
        max_token_size=embedding_func.max_token_size,
        func=wrapped,
    )


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = False

    # text chunking
    tokenizer_type: str = "tiktoken"  # or 'huggingface'
    tiktoken_model_name: str = "gpt-4o"
    huggingface_model_name: str = "bert-base-uncased"  # default HF model
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            TokenizerWrapper,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 3072,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    query_better_than_threshold: float = 0.2

    # LLM
    using_azure_openai: bool = False
    using_amazon_bedrock: bool = False
    using_langchain_langfuse: bool = False  # True で LangChain + Langfuse でモニタリング
    best_model_id: str = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    cheap_model_id: str = "us.anthropic.claude-3-haiku-20240307-v1:0"
    best_model_func: callable = best_llm_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: callable = cheap_llm_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # entity extraction
    entity_extraction_func: callable = extract_entities

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    # optional trackers (TokenTracker / TimeTracker from nano_graphrag.tracker)
    token_tracker: Optional[Any] = None
    time_tracker: Optional[Any] = None

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        self.tokenizer_wrapper = TokenizerWrapper(
            tokenizer_type=self.tokenizer_type,
            model_name=self.tiktoken_model_name if self.tokenizer_type == "tiktoken" else self.huggingface_model_name
        )

        if self.using_azure_openai:
            # If there's no OpenAI API key, use Azure OpenAI
            if self.best_model_func == best_llm_complete:
                self.best_model_func = azure_best_llm_complete
            if self.cheap_model_func == cheap_llm_complete:
                self.cheap_model_func = azure_cheap_llm_complete
            if self.embedding_func == openai_embedding:
                self.embedding_func = azure_openai_embedding
            logger.info(
                "Switched the default openai funcs to Azure OpenAI if you didn't set any of it"
            )

        if self.using_amazon_bedrock:
            self.best_model_func = create_amazon_bedrock_complete_function(self.best_model_id)
            self.cheap_model_func = create_amazon_bedrock_complete_function(self.cheap_model_id)
            self.embedding_func = amazon_bedrock_embedding
            logger.info(
                "Switched the default openai funcs to Amazon Bedrock"
            )

        if self.using_langchain_langfuse:
            try:
                from ._llm_langchain import (
                    get_langchain_langfuse_complete_factories,
                    get_langchain_langfuse_embedding,
                )
                use_azure = self.using_azure_openai
                best_fn, cheap_fn = get_langchain_langfuse_complete_factories(use_azure=use_azure)
                if self.best_model_func in (best_llm_complete, azure_best_llm_complete):
                    self.best_model_func = best_fn
                if self.cheap_model_func in (cheap_llm_complete, azure_cheap_llm_complete):
                    self.cheap_model_func = cheap_fn
                if self.embedding_func in (openai_embedding, azure_openai_embedding):
                    self.embedding_func = get_langchain_langfuse_embedding(use_azure=use_azure)
                logger.info(
                    "Switched the default openai funcs to LangChain + Langfuse (LLM requests will be traced)"
                )
            except ImportError as e:
                logger.warning(
                    "using_langchain_langfuse=True but LangChain/Langfuse not available: %s. "
                    "Install with: pip install langchain-openai langfuse",
                    e,
                )

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        if self.token_tracker is not None:
            self.embedding_func = _wrap_embedding_with_tracker(
                self.embedding_func, self.token_tracker
            )

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        _best_kw = {"hashing_kv": self.llm_response_cache}
        _cheap_kw = {"hashing_kv": self.llm_response_cache}
        if self.token_tracker is not None:
            _best_kw["tracker"] = self.token_tracker
            _cheap_kw["tracker"] = self.token_tracker
        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, **_best_kw)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, **_cheap_kw)
        )



    def insert(
        self,
        string_or_strings,
        parsers: Optional[List[BaseParser]] = None,
    ):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings, parsers=parsers))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if self.time_tracker is not None:
            self.time_tracker.start("query")
        try:
            return await self._aquery_impl(query, param)
        finally:
            if self.time_tracker is not None:
                self.time_tracker.stop("query")

    async def _aquery_impl(self, query: str, param: QueryParam):
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self.tokenizer_wrapper,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    def _normalize_inputs_to_texts(
        self,
        string_or_strings,
        parsers: Optional[List[BaseParser]] = None,
    ) -> list[str]:
        """Normalize inputs (strings or paths) into a list of text contents.

        - If an element is a string and corresponds to an existing file path,
          it will be parsed via the file parsers (docx/pdf/xlsx/...) or read
          as plain text.
        - Otherwise the element is treated as already-parsed text.
        """

        if isinstance(string_or_strings, str):
            items = [string_or_strings]
        elif isinstance(string_or_strings, list):
            items = string_or_strings
        else:
            raise TypeError(
                "insert/ainsert only support `str` or `list[str]` as input."
            )

        texts: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise TypeError(
                    "insert/ainsert only support `str` or `list[str]` as input."
                )
            if os.path.exists(item):
                texts.append(parse_file(item, parsers=parsers))
            else:
                texts.append(item)
        return texts

    async def ainsert(
        self,
        string_or_strings,
        parsers: Optional[List[BaseParser]] = None,
    ):
        if self.time_tracker is not None:
            self.time_tracker.start("insert")
        await self._insert_start()
        try:
            texts = self._normalize_inputs_to_texts(string_or_strings, parsers=parsers)
            # ---------- new docs
            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in texts
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning(f"All docs are already in the storage")
                return
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            # ---------- chunking

            inserting_chunks = get_chunks(
                new_docs=new_docs,
                chunk_func=self.chunk_func,
                overlap_token_size=self.chunk_overlap_token_size,
                max_token_size=self.chunk_token_size,
                tokenizer_wrapper=self.tokenizer_wrapper,
            )

            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: don't support incremental update for communities now, so we have to drop all
            await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            maybe_new_kg = await self.entity_extraction_func(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                tokenizer_wrapper=self.tokenizer_wrapper,
                global_config=asdict(self),
                using_amazon_bedrock=self.using_amazon_bedrock,
            )
            if maybe_new_kg is None:
                logger.warning("No new entities found")
                return
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- update clusterings of graph
            logger.info("[Community Report]...")
            await self.chunk_entity_relation_graph.clustering(
                self.graph_cluster_algorithm
            )
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, self.tokenizer_wrapper, asdict(self)
            )

            # ---------- commit upsertings and indexing
            await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()
            if self.time_tracker is not None:
                self.time_tracker.stop("insert")

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
