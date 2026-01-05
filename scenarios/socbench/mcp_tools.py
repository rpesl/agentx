import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from rag_retriever import RAGRetriever
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

mcp = FastMCP(
    name="SOCBench Tools",
    json_response=True
)
BENCHMARK_ROOT = Path("scenarios/socbench/benchmark")


def _setup_embedding_model():
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to setup embedding model: {e}")


_setup_embedding_model()


@mcp.tool()
def list_available_domains(instance_id: int = None) -> list[dict]:
    """
    List all available benchmark domains.

    Args:
        instance_id: Optional instance ID like 1, 2, etc. If not provided, lists all instances.

    Returns:
        List of dicts with domain info: {"path": "socbenchd_1/01-energy", "name": "Energy", "services": [...]}
    """
    domains = []

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
        domain_path: Relative path like "socbenchd_1/01-energy"

    Returns:
        List of OpenAPI specification dictionaries
    """
    full_path = BENCHMARK_ROOT / domain_path

    if not full_path.exists():
        raise ValueError(
            f"Domain path not found: {domain_path}. "
            f"Use list_available_domains() to see available options."
        )

    openapis = []

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
        instance_id = 1

    rag_retriever = RAGRetriever(
        openapi_specs=all_specs_json,
        top_k=5
    )

    relevant_specs_json = rag_retriever.retrieve(query, instance_id, domain_path)
    relevant_specs = [json.loads(spec) for spec in relevant_specs_json]
    return relevant_specs


def get_mcp_tools_for_openai(include_rag: bool = False) -> list[dict]:
    """
    Get MCP tools in OpenAI function calling format.

    Args:
        include_rag: If True, includes RAG tool for semantic search
    """
    base_tools = [
        {
            "type": "function",
            "function": {
                "name": "list_available_domains",
                "description": (
                    "List all available benchmark domains. "
                    "Use this FIRST to understand what domains and services are available. "
                    "Returns info about each domain including its path, name and services."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instance_id": {
                            "type": "integer",
                            "description": "Optional instance ID like 1, 2, etc. If not provided, lists all instances."
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "load_openapi_specs",
                "description": (
                    "Load OpenAPI specifications from a benchmark domain. "
                    "Use list_available_domains() first to find available domains. "
                    "This loads all service specs for a domain."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "domain_path": {
                            "type": "string",
                            "description": "Relative path like 'socbenchd_1/01-energy'"
                        }
                    },
                    "required": ["domain_path"]
                }
            }
        }
    ]
    rag_tool = {
        "type": "function",
        "function": {
            "name": "retrieve_relevant_specs_with_rag",
            "description": (
                "Use RAG (semantic search) to find the most relevant OpenAPI specifications for a query. "
                "Call this AFTER finding the right domain with list_available_domains(). "
                "Returns only the top-k most relevant specifications based on semantic similarity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain_path": {
                        "type": "string",
                        "description": "Domain path like 'socbenchd_1/01-energy'"
                    },
                    "query": {
                        "type": "string",
                        "description": "The user's query to match against"
                    },
                },
                "required": ["domain_path", "query"]
            }
        }
    }
    if include_rag:
        return base_tools + [rag_tool]
    else:
        return base_tools


def execute_mcp_tool(tool_name: str, arguments: dict):
    """
    Execute an MCP tool by name.
    """
    if tool_name == "list_available_domains":
        return list_available_domains(**arguments)
    elif tool_name == "load_openapi_specs":
        return load_openapi_specs(**arguments)
    elif tool_name == "retrieve_relevant_specs_with_rag":
        return retrieve_relevant_specs_with_rag(**arguments)
    else:
        raise ValueError(
            f"Unknown tool: {tool_name}. "
            f"Allowed tools: list_available_domains, load_openapi_specs, retrieve_relevant_specs_with_rag"
        )
