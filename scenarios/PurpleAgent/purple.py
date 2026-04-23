import argparse
import uvicorn
import asyncio
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill
)
from single_executor import SingleExecutor
from multi_executor import MultiExecutor

load_dotenv()


def create_agent_card(agent_url: str) -> AgentCard:
    skill = AgentSkill(
        id='generate_code',
        name='Generate Python Code',
        description=(
            "Generates Python code based on a query and MCP-provided tools."
            "Accepts JSON input with 'prompt' field containing the user's request."
        ),
        tags=['code generation', 'OpenAPI', 'Python'],
        examples=[
            """{
            "prompt": "Generate Python code to fetch the list of users from the API using the provided OpenAPI specs."
            }"""]
    )

    return AgentCard(
        name="code_generation",
        description='Generates Python code based on queries and MCP-provided tools.',
        url=agent_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A code generation agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--type", type=str, default="single", choices=["single", "multi", "parallel"])
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()

    agent_url = args.card_url or f'http://{args.host}:{args.port}/'

    if args.type == "multi":
        executor = MultiExecutor()
    else:
        executor = SingleExecutor()

    agent_card = create_agent_card(agent_url)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
    uvicorn_server = uvicorn.Server(uvicorn_config)
    await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
