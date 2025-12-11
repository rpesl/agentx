import argparse
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)


def main():
    parser = argparse.ArgumentParser(description="Run the A2A code generation agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()

    root_agent = Agent(
        name="code_generation",
        model="moonshotai/Kimi-K2-Instruct",
        description="Generates code based on a random query.",
        instruction="You are a code generation agent. Generate code based on the given query.",
    )

    agent_card = AgentCard(
        name="code_generation",
        description='Generates code based on a random query.',
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
