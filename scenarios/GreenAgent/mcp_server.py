import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from rag_retriever import RAGRetriever
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

mcp = FastMCP(name="SOCBench Tools", json_response=True)
BENCHMARK_ROOT = Path("scenarios/GreenAgent/benchmark")
RESTBENCH_ROOT = Path("scenarios/GreenAgent/benchmark/restbench/data/specs")

def _initialize_embedding_model():
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
    except Exception:
        raise

_initialize_embedding_model()

@mcp.tool()
def list_available_domains(instance_id: str | int = None) -> list[dict]:
    """
    List all available benchmark domains.

    Args:
        instance_id: Optional instance ID like 1, 2, etc. If not provided, lists all instances.

    Returns:
        List of dicts with domain info: {"path": "socbenchd_1/01-energy", "name": "Energy", "services": [...]}
    """
    domains = []

    if str(instance_id).lower() == "restbench":
        if RESTBENCH_ROOT.exists():
            for spec_file in sorted(RESTBENCH_ROOT.glob("*_oas.json")):
                service_name = spec_file.stem.replace("_oas", "").capitalize()
                domain_path = f"restbench/data/specs/{spec_file.name}"
                domains.append({
                    "path": domain_path,
                    "name": service_name
                })
        return domains

    for benchmark_dir in sorted(BENCHMARK_ROOT.iterdir()):
        if benchmark_dir.is_dir() and benchmark_dir.name.startswith("socbenchd"):
            try:
                dir_instance_id = int(benchmark_dir.name.split("_")[1]) if "_" in benchmark_dir.name else None
            except (IndexError, ValueError):
                continue
            if instance_id is not None and dir_instance_id != instance_id:
                continue

            for domain_dir in sorted(benchmark_dir.iterdir()):
                if domain_dir.is_dir():
                    domain_path = f"{benchmark_dir.name}/{domain_dir.name}"
                    domain_name = domain_dir.name.split("-")[-1].capitalize()
                    services = []
                    for service_dir in sorted(domain_dir.iterdir()):
                        if service_dir.is_dir():
                            openapi_file = service_dir / "openapi.json"
                            if openapi_file.exists():
                                with open(openapi_file, "r", encoding="utf-8") as f:
                                    spec = json.load(f)
                                    service_name = spec.get("info", {}).get("title", service_dir.name)
                                    services.append(service_name)

                    if services:
                        domains.append({
                            "path": domain_path,
                            "name": domain_name,
                            "services": services,
                            "description": f"{domain_name} domain with {len(services)} services",
                            "instance_id": dir_instance_id
                        })
    return domains


@mcp.tool()
def load_openapi_specs(domain_path: str) -> list[dict]:
    """
    Load OpenAPI specifications from a benchmark domain path.

    First call list_available_domains() to see what domains are available!

    Args:
        domain_path: Relative path like
                     - "socbenchd_1/01-energy"
                     - "restbench/data/specs/spotify_oas.json"

    Returns:
        List of OpenAPI specification dictionaries
    """
    openapis = []

    full_path = BENCHMARK_ROOT / domain_path

    if not full_path.exists():
        raise ValueError(
            f"Domain path not found: {domain_path}. "
            f"Use list_available_domains() to see available options."
        )
    if full_path.is_file() and full_path.suffix == ".json":
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
                openapis.append(spec)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to load RestBench spec {full_path}: {e}")
        return openapis

    for entry in sorted(full_path.iterdir()):
        if entry.is_dir():
            openapi_file = entry / "openapi.json"
            if openapi_file.exists():
                try:
                    with open(openapi_file, "r") as file:
                        spec = json.load(file)
                        openapis.append(spec)
                except json.JSONDecodeError as e:
                    print(f"Failed to load {openapi_file}: {e}")
                    continue

    if not openapis:
        raise ValueError(f"No OpenAPI specs found in {domain_path}")

    return openapis


@mcp.tool()
def retrieve_relevant_specs_with_rag(domain_path: str, query: str) -> list[dict]:
    """
    Load OpenAPI specs from domain and retrieve most relevant ones using RAG.

    Args:
        domain_path: Relative path like "socbenchd_1/01-energy"
        query: User query for semantic search

    Returns:
        List of most relevant OpenAPI specification dictionaries
    """
    all_specs = load_openapi_specs(domain_path)
    all_specs_json = [json.dumps(spec) for spec in all_specs]

    try:
        instance_id = int(domain_path.split("_")[1].split("/")[0])
    except (IndexError, ValueError):
        instance_id = "restbench"

    rag_retriever = RAGRetriever(
        openapi_specs=all_specs_json,
        top_k=5
    )
    relevant_specs_json = rag_retriever.retrieve(query, instance_id, domain_path)
    relevant_specs = [json.loads(spec) for spec in relevant_specs_json]
    return relevant_specs


if __name__ == "__main__":
    mcp.run(transport="sse")

