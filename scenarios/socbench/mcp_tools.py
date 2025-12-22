import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="SOCBench Tools",
    json_response=True
)
BENCHMARK_ROOT = Path("scenarios/socbench/benchmark")


@mcp.tool()
# TODO only domains of the right instance
def list_available_domains() -> list[dict]:
    """
    List all available benchmark domains.

    Returns:
        List of dicts with domain info: {"path": "socbenchd_1/01-energy", "name": "Energy", "services": [...]}

    Example:
    domains = list_available_domains()
    for domain in domains:
        print(f"{domain['path']}: {domain['name']}")
    """
    domains = []

    for benchmark_dir in sorted(BENCHMARK_ROOT.iterdir()):
        if benchmark_dir.is_dir() and benchmark_dir.name.startswith("socbenchd"):
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
                            "description": f"{domain_name} domain with {len(services)} services"
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

    Example:
    # 1. First discover domains
    domains = list_available_domains()

    # 2. Pick one and load specs
    specs = load_openapi_specs("socbenchd_1/01-energy")
    print(f"Loaded {len(specs)} specs")
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
                    with open(openapi_file, "r", encoding="utf-8") as file:
                        spec = json.load(file)
                        openapis.append(spec)
                except json.JSONDecodeError as e:
                    print(f"Failed to load {openapi_file}: {e}")
                    continue

    if not openapis:
        raise ValueError(f"No OpenAPI specs found in {domain_path}")

    return openapis


def get_mcp_tools_for_openai() -> list[dict]:
    """
    Get MCP tools in OpenAI function calling format.

    Includes:
    - list_available_domains(): Discover what domains exist
    - load_openapi_specs(): Load specs for a specific domain

    Returns:
        List of tool definitions for OpenAI API
    """
    return [
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
                    "properties": {}
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


def execute_mcp_tool(tool_name: str, arguments: dict):
    """
    Execute an MCP tool by name.

    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of arguments

    Returns:
        Tool execution result
    """
    if tool_name == "list_available_domains":
        print("Executing list_available_domains()")
        return list_available_domains()
    elif tool_name == "load_openapi_specs":
        print(f"Executing load_openapi_specs() with arguments: {arguments}")
        return load_openapi_specs(**arguments)
    else:
        raise ValueError(
            f"Unknown tool: {tool_name}."
            f"Allowed tools: list_available_domains, load_openapi_specs"
        )
