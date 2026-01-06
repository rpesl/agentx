import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.faiss import FaissVectorStore
from socrag.endpointparser import EndpointParser
from llama_index.core import load_index_from_storage

logger = logging.getLogger("RAGRetriever")

class RAGRetriever:
    MAX_ENDPOINTS = 20

    def __init__(
        self,
        openapi_specs: List[str],
        top_k: int = 5,
        cache_dir: str = "data/rag_cache"
    ):
        self.openapi_specs = openapi_specs
        self.top_k = top_k
        self.cache_dir = Path(cache_dir)

        self.index: BaseIndex | None = None
        self._spec_cache = self._build_spec_cache()

        if Settings.embed_model is None:
            raise ValueError(
                "Settings.embed_model must be configured before using RAGRetriever"
            )

    def _build_spec_cache(self) -> Dict[int, Dict]:
        cache = {}
        for idx, spec_str in enumerate(self.openapi_specs):
            try:
                spec_obj = json.loads(spec_str)
                cache[idx] = {
                    "spec_str": spec_str,
                    "spec_obj": spec_obj,
                    "title": spec_obj.get("info", {}).get("title", "Unknown"),
                    "paths": spec_obj.get("paths", {}),
                    "components": spec_obj.get("components", {}),
                }
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in spec {idx}, skipping")

        return cache

    def _create_index(self) -> VectorStoreIndex:
        dimension = len(Settings.embed_model.get_text_embedding("test"))
        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        documents = [
            Document(text=spec, id_=str(idx))
            for idx, spec in enumerate(self.openapi_specs)
        ]

        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[EndpointParser()],
        )

    def _load_or_create_index(self, instance_id: str | int, domain_path: str) -> BaseIndex:
        if instance_id == "restbench":
            cache_path = self.cache_dir / "restbench" / domain_path.replace("/", "_")
        else:
            cache_path = self.cache_dir / f"instance_{instance_id}" / domain_path.replace("/", "_")

        if cache_path.exists():
            logger.info(f"Loading RAG index from cache: {cache_path}")
            try:
                vector_store = FaissVectorStore.from_persist_dir(str(cache_path))
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=str(cache_path),
                )
                return load_index_from_storage(storage_context)
            except Exception:
                pass

        index = self._create_index()
        cache_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(str(cache_path))
        return index

    def _extract_endpoint_info(
        self, node: NodeWithScore
    ) -> Tuple[int | None, str | None, str | None]:
        metadata = getattr(node.node, "metadata", None)
        if not metadata:
            return None, None, None

        method = metadata.get("verb")
        path = metadata.get("path")
        title = metadata.get("title")

        if not (method and path and title):
            return None, None, None

        for spec_idx, spec_data in self._spec_cache.items():
            if spec_data["title"] == title:
                return spec_idx, method, path.rstrip("/")

        return None, None, None

    def retrieve(self, query: str, instance_id: str | int, domain_path: str) -> List[str]:
        if self.index is None:
            self.index = self._load_or_create_index(instance_id, domain_path)

        retriever = self.index.as_retriever(
            similarity_top_k=self.MAX_ENDPOINTS * 2
        )

        nodes: List[NodeWithScore] = retriever.retrieve(query)

        all_endpoints: List[Tuple[int, str, str, float]] = []

        for node in nodes:
            spec_idx, method, path = self._extract_endpoint_info(node)
            if spec_idx is None:
                continue

            score = 1.0 / (1.0 + node.score)
            all_endpoints.append((spec_idx, method, path, score))

        if not all_endpoints:
            return []

        all_endpoints.sort(key=lambda x: x[3], reverse=True)
        all_endpoints = all_endpoints[: self.MAX_ENDPOINTS]

        spec_endpoints: Dict[int, List[Tuple[str, str, float]]] = defaultdict(list)
        for spec_idx, method, path, score in all_endpoints:
            spec_endpoints[spec_idx].append((method, path, score))

        spec_scores = {
            spec_idx: sum(score for _, _, score in eps) / len(eps)
            for spec_idx, eps in spec_endpoints.items()
        }

        sorted_specs = sorted(
            spec_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[: self.top_k]

        result_specs = []

        for spec_idx, _ in sorted_specs:
            filtered = self._create_filtered_spec(
                spec_idx,
                spec_endpoints[spec_idx]
            )
            if filtered:
                result_specs.append(filtered)

        if not result_specs:
            for spec_idx, _ in sorted_specs[:3]:
                result_specs.append(self._spec_cache[spec_idx]["spec_str"])

        return result_specs

    def _create_filtered_spec(
        self,
        spec_idx: int,
        matched_endpoints: List[Tuple[str, str, float]],
    ) -> str | None:
        spec_cache = self._spec_cache[spec_idx]
        spec_obj = spec_cache["spec_obj"]
        paths = spec_cache["paths"]

        filtered_paths = {}
        used_refs: Set[str] = set()

        for method, path, _ in matched_endpoints:
            if path not in paths:
                continue

            m = method.lower()
            if m not in paths[path]:
                continue

            filtered_paths.setdefault(path, {})[m] = paths[path][m]
            self._extract_refs(paths[path][m], used_refs)

        if not filtered_paths:
            return None

        filtered_components = {}
        for ctype, comps in spec_cache["components"].items():
            kept = {
                name: schema
                for name, schema in comps.items()
                if f"#/components/{ctype}/{name}" in used_refs
            }
            if kept:
                filtered_components[ctype] = kept

        minimized = {
            "openapi": spec_obj.get("openapi", "3.0.0"),
            "info": spec_obj.get("info", {}),
            "servers": spec_obj.get("servers", []),
            "paths": filtered_paths,
        }

        if filtered_components:
            minimized["components"] = filtered_components

        return json.dumps(minimized)

    def _extract_refs(self, data: Any, refs: Set[str]):
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "$ref" and isinstance(v, str):
                    refs.add(v)
                else:
                    self._extract_refs(v, refs)
        elif isinstance(data, list):
            for item in data:
                self._extract_refs(item, refs)

    def reset(self):
        self.index = None
