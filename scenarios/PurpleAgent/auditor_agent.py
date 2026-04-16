import uvicorn, asyncio, argparse
from dotenv import load_dotenv
from auditor_executor import AuditorExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill


def make_agent_card(name: str, url: str) -> AgentCard:
    skill = AgentSkill(
        id=f"{name.lower()}_task",
        name=f"{name} Task",
        description=(
            "Audits generated Python code against real OpenAPI specs via MCP. "
            "Returns a structured JSON report with confirmed, wrong, and missing endpoints."
        ),
        tags=["audit", "OpenAPI", "Python", "code review"],
        examples=[
            """{
            "prompt": "Audit this code for the task: 'fetch energy readings'\n\nCODE:\nimport requests\nrequests.get('/energy/readings')"
            }"""
        ],
    )

    return AgentCard(
        name=name,
        description=f"{name} — MCP-backed API endpoint auditor.",
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
    parser.add_argument("--port", type=int, default=9020)
    args = parser.parse_args()

    executor = AuditorExecutor()
    agent_card = make_agent_card("AuditorAgent", f"http://{args.host}:{args.port}/")
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    server = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)

    config = uvicorn.Config(server.build(), host=args.host, port=args.port)
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    asyncio.run(main())