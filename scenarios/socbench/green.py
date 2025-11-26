import argparse
import contextlib
import os, json
import random
import uvicorn
import asyncio
import logging
from dotenv import load_dotenv
from openai import OpenAI
from socbenchsc.src.socbenchsc.analysis import Analysis
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import TaskUpdater
from a2a.types import (TaskState, Part, TextPart)
from a2a.utils import (new_agent_text_message)
from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider
from models import DebateEval, judge_agent_card, DebaterScore

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation_agent")

benchmark_root = "scenarios/socbench/benchmark"
domains = ["01-energy", "02-materials", "03-industrials", "04-consumer discretionary", "05-consumer staples",
           "06-health care", "07-financials", "08-information technology", "09-communication services",
           "10-utilities", "11-real estate"]


def run_openai_query(prompt: str) -> str:
    api_key = os.getenv("NEBIUS_API_KEY")
    client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=api_key)
    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct",
        messages=[
            {"role": "system", "content": "You are a code generation assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    generated_code = response.choices[0].message.content
    return generated_code


class DebateJudge(GreenAgent):
    def __init__(self, num_agents=2):
        self._required_roles = [f"PurpleAgent_{i}" for i in
                                range(num_agents)]
        self._required_config_keys = ["num_rounds"]
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
        logger.info(f"Starting code creation orchestration: {request}")

        try:
            code = await self.orchestrate_code_creation(request.participants,
                                                        request.config["num_rounds"],
                                                        updater)
            await updater.update_status(TaskState.working, new_agent_text_message(
                f"Code creation orchestration finished. Starting evaluation."))
            logger.info("Code creation orchestration finished. Evaluating code.")
            code_eval: DebateEval = await self.judge_code(code, updater)
            logger.info(f"Code Evaluation:\n{code_eval.model_dump_json()}")

            result = EvalResult(winner=code_eval.winner, detail=code_eval.model_dump())
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=result.model_dump_json())),
                ],
                name="Result",
            )
        finally:
            self._tool_provider.reset()

    @staticmethod
    def load_openapi_specs(domain_path: str) -> list[dict]:
        openapis = []
        for entry in os.listdir(domain_path):
            service_path = os.path.join(domain_path, entry)
            if os.path.isdir(service_path):
                openapi_file = os.path.join(service_path, "openapi.json")
                if os.path.exists(openapi_file):
                    with open(openapi_file, "r") as file:
                        openapis.append(json.load(file))
        return openapis

    @staticmethod
    def load_query(domain_path: str) -> tuple[str, list[str]]:
        query_file = os.path.join(domain_path, "queries.json")
        with open(query_file, "r") as file:
            query_data = json.load(file)
        query = random.choice(query_data["queries"])
        return query["query"], query["endpoints"]

    @staticmethod
    async def generate_code_for_query(role: str, openapis: list[dict], query_text: str, updater: TaskUpdater,
                                      dictionary: dict) -> str:
        prompt = f"""
        You are a code generation agent. Your task is to generate Python code that answers the following query using the provided OpenAPI specifications.
        Query: {query_text}
        OpenAPI Specifications: {json.dumps(openapis)}
        Please only provide the Python code without any explanations or additional text. Do not include markdown formatting. 
        """
        generated_code = run_openai_query(prompt)
        dictionary[role].append(generated_code)
        await updater.update_status(TaskState.working, new_agent_text_message(f"{role}: {generated_code}"))  # debugging
        return generated_code

    async def orchestrate_code_creation(
            self,
            participants: dict[str, str],
            num_rounds: int,
            updater: TaskUpdater,
    ) -> dict[str, list[str]]:

        dictionary = {role: [] for role in participants.keys()}
        self.expected_endpoints = []
        for round_id in range(num_rounds):
            instance_id = random.randint(1, 5)  # or round_id % 5 + 1 for determinism
            domain = random.choice(domains)  # or domains[round_id % len(domains)] for determinism
            domain_path = os.path.join(benchmark_root, f"socbenchd_{instance_id}", domain)

            openapis = self.load_openapi_specs(domain_path)
            await updater.update_status(TaskState.working, new_agent_text_message(
                f"[Round {round_id + 1}] loaded {len(openapis)} OpenAPI specifications."))

            query_text, endpoints = self.load_query(domain_path)
            self.expected_endpoints.append(endpoints)
            await updater.update_status(TaskState.working, new_agent_text_message(
                f"[Round {round_id + 1}] selected query: {query_text}"))

            for role in participants.keys():
                await self.generate_code_for_query(role, openapis, query_text, updater, dictionary)

        return dictionary

    @staticmethod
    def compute_recall(retrieved_endpoints: set[str], expected_endpoints: list[str]) -> float:
        if not expected_endpoints:
            return 0.0
        true_positives = len(retrieved_endpoints.intersection(set(expected_endpoints)))
        recall = true_positives / len(expected_endpoints)
        return recall

    async def judge_code(self, debate_text: dict[str, list[str]], updater=TaskUpdater) -> DebateEval:

        intermediate_results: dict[str, list[float]] = {role: [] for role in debate_text.keys()}

        for agent, generated_codes in debate_text.items():
            for round_index, code in enumerate(generated_codes):
                analysis = Analysis(code)
                retrieved_endpoints = analysis.perform_analysis()
                await updater.update_status(TaskState.working, new_agent_text_message(
                    f"Analysis for {agent} in round {round_index + 1}: Retrieved endpoints: {retrieved_endpoints}"))  # debugging
                recall = self.compute_recall(retrieved_endpoints, self.expected_endpoints[round_index])
                intermediate_results[agent].append(recall)

        avg_results = {agent: sum(scores) / len(scores) if scores else 0.0 for agent, scores in
                       intermediate_results.items()}

        max_agent = max(avg_results, key=avg_results.get)
        participants_recalls = {agent: DebaterScore(recall=avg_results[agent]) for agent in avg_results}
        debate_eval = DebateEval(
            participants=participants_recalls,
            winner="The winner is " + max_agent + " with an average recall of " + f"{avg_results[max_agent]:.2f}."
        )
        return debate_eval


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A code generation agent.")
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
        agent_card = judge_agent_card("DebateJudge", agent_url)
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
