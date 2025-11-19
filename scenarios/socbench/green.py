import argparse
import contextlib
import os, json
import random

import uvicorn
import asyncio
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

from scenarios.socbench.socbenchsc.src.socbenchsc.analysis import Analysis

load_dotenv()

from google import genai
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    Part,
    TextPart,
)
from a2a.utils import (
    new_agent_text_message
)

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider

from scenarios.socbench.models import DebateEval, debate_judge_agent_card

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debate_judge")

benchmark_root = "scenarios/socbench/benchmark"
domains = ["01-energy", "02-materials", "03-industrials", "04-consumer discretionary", "05-consumer staples",
           "06-health care", "07-financials", "08-information technology", "09-communication services",
           "10-utilities", "11-real estate"]


class DebateJudge(GreenAgent):
    def __init__(self, num_agents=2):
        self._required_roles = [f"PurpleAgent_{i}" for i in
                                range(num_agents)]  # TODO: Adapt for arbitrary number of agents
        self._required_config_keys = ["num_rounds"]
        self._client = genai.Client()
        self._tool_provider = ToolProvider()
        self.expected_endpoints = []

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        try:
            int(request.config["num_rounds"])
        except Exception as e:
            return False, f"Can't parse num_rounds: {e}"
        return True, "ok"

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting debate orchestration: {request}")

        try:
            code = await self.orchestrate_code_creation(request.participants,
                                                        request.config["num_rounds"],
                                                        updater)

            await updater.update_status(TaskState.working, new_agent_text_message(
                f"Code creation orchestration finished. Starting evaluation."))
            logger.info("Code creation orchestration finished. Evaluating debate.")
            code_eval: DebateEval = await self.judge_code(code)
            logger.info(f"Code Evaluation:\n{code_eval.model_dump_json()}")

            result = EvalResult(winner=code_eval.winner, detail=code_eval.model_dump())
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=code_eval.reason)),
                    Part(root=TextPart(text=result.model_dump_json())),
                ],
                name="Result",
            )
        finally:
            self._tool_provider.reset()

    async def orchestrate_code_creation(
            self,
            participants: dict[str, str],
            num_rounds: int,
            updater: TaskUpdater,
    ) -> dict[str, list[str]]:
        dictionary: dict[str, list[str]] = {role: [] for role in
                                            participants.keys()}  # TODO: Adapt for arbitrary number of agents

        async def turn(role: str, prompt: str) -> str:
            response = await self._tool_provider.talk_to_agent(prompt, str(participants[role]), new_conversation=False)
            logger.info(f"{role}: {response}")
            dictionary[role].append(response)
            await updater.update_status(TaskState.working, new_agent_text_message(f"{role}: {response}"))
            return response

        # TODO: Add benchmark cases
        # For each round
        # 1. Load openapi.json specifications
        # 2. Load query
        # 3. Each agent generates code to answer the query using the specifications
        # Collect results and return

        async def generate_code_for_query(role: str, openapis: list[dict], query_text: str) -> str:
            prompt = f"""
           You are a code generation agent. Your task is to generate Python code that answers the following query using the provided OpenAPI specifications.
           Query: {query_text}
           OpenAPI Specifications: {json.dumps(openapis)}
           Please provide the generated code. 
           """
            code = await turn(role, prompt)
            return code

        self.expected_endpoints = []
        for round_id in range(num_rounds):
            instance_id = random.randint(1, 5)  # or round_id % 5 + 1 for determinism
            domain = random.choice(domains)  # or domains[round_id % len(domains)] for determinism
            domain_path = os.path.join(benchmark_root, f"socbenchd_{instance_id}", domain)
            openapis = []
            service_names = []

            for entry in os.listdir(domain_path):
                service_path = os.path.join(domain_path, entry)
                if os.path.isdir(service_path):
                    openapi_file = os.path.join(service_path, "openapi.json")
                    if os.path.exists(openapi_file):
                        with open(openapi_file, "r") as file:
                            openapi_spec = json.load(file)
                            openapis.append(openapi_spec)
                            service_names.append(entry)

            await updater.update_status(TaskState.working, new_agent_text_message(
                f"[Round {round_id + 1}] loaded {len(openapis)} OpenAPI specifications."))

            # Load query
            query_file = os.path.join(domain_path, "query.json")
            with open(query_file, "r") as file:
                query_data = json.load(file)
                query_list = query_data["queries"]
                query = query_list[
                    random.randint(0, len(query_list) - 1)]  # or query_list[round_id % len(query_list)] for determinism
                query_text = query["query"]
                self.expected_endpoints.append(query["expected_endpoints"])

            for role in participants.keys():
                await generate_code_for_query(role, openapis, query_text)

        return dictionary

    async def judge_code(self, debate_text: dict[str, list[str]]) -> DebateEval:

        # TODO: For each round: Call static code analysis (socbenchsc)
        # Input: Code + Solution
        # Output: Called endpoints
        # Compute recall, i.e., number of correct endpoints in percent
        # return intermediate results + winner
        intermediate_results: dict[str, list[float]] = {role: [] for role in debate_text.keys()}
        for agent, generated_codes in debate_text.items():
            for code in generated_codes:
                currentRound = 0
                analysis = Analysis(code)
                retrievedEndpoints = analysis.perform_analysis()
                currentExpectedEndpoints = self.expected_endpoints[currentRound]
                true_positives = len(retrievedEndpoints.intersection(set(currentExpectedEndpoints)))
                recall = true_positives / len(currentExpectedEndpoints) if currentExpectedEndpoints else 0
                intermediate_results[agent].append(recall)
                currentRound += 1
        avg_results = {agent: sum(scores) / len(scores) if scores else 0.0 for agent, scores in
                       intermediate_results.items()}
        max_agent = max(avg_results, key=avg_results.get)

        return max_agent  # TODO: Replace with proper DebateEval


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A debate judge.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true",
                        help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url")
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        from agentbeats.cloudflare import quick_tunnel
        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")

    async with agent_url_cm as agent_url:
        agent = DebateJudge()
        executor = GreenExecutor(agent)
        agent_card = debate_judge_agent_card("DebateJudge", agent_url)

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


if __name__ == '__main__':
    asyncio.run(main())
