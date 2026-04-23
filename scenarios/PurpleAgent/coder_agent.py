import uvicorn, asyncio, argparse
from dotenv import load_dotenv
from single_executor import SingleExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill


def make_agent_card(name: str, url: str) -> AgentCard:
    skill = AgentSkill(
        id=f"{name.lower()}_task",
        name=f"{name} Task",
        description=f"Handles code generation and auditing tasks for {name}.",
        tags=["code generation", "OpenAPI", "Python"],
        examples=[
            """{
            "prompt": "Generate Python code to fetch the list of users from the API using the provided OpenAPI specs."
            }"""
        ]
    )

    return AgentCard(
        name=name,
        description=f"{name} sub-agent for code generation and auditing.",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )


load_dotenv()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9019)
    args = parser.parse_args()

    executor = SingleExecutor()
    agent_card = make_agent_card("CoderAgent", f"http://{args.host}:{args.port}/")
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    server = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    config = uvicorn.Config(server.build(), host=args.host, port=args.port)
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    asyncio.run(main())
