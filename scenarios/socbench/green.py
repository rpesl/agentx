import argparse
import contextlib
import uvicorn
import asyncio
import logging
from dotenv import load_dotenv
from metric_evaluator import MetricEvaluator
from code_generator import CodeGenerator
from openapi_loader import OpenAPILoader
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
from models import CodeEval, judge_agent_card, CodeScore
from endpoint_normalizer import normalize_endpoint

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation_agent")

benchmark_root = "scenarios/socbench/benchmark"
domains = ["01-energy", "02-materials", "03-industrials", "04-consumer discretionary", "05-consumer staples",
           "06-health care", "07-financials", "08-information technology", "09-communication services",
           "10-utilities", "11-real estate"]


class CodeJudge(GreenAgent):
    def __init__(self, num_agents=2):
        self._required_roles = [f"PurpleAgent_{i}" for i in
                                range(num_agents)]
        self._required_config_keys = ["num_rounds"]
        self._tool_provider = ToolProvider()
        self.expected_endpoints = []
        self.openapi_loader = OpenAPILoader(benchmark_root)
        self.code_generator = CodeGenerator()

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
            code_eval: CodeEval = await self.judge_code(code, updater)
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

    async def orchestrate_code_creation(
            self,
            participants: dict[str, str],
            num_rounds: int,
            updater: TaskUpdater,
    ) -> dict[str, list[str]]:

        dictionary = {role: [] for role in participants.keys()}
        self.expected_endpoints = []
        logger.info(f"Orchestrating code creation for {num_rounds} rounds.")

        for round_id in range(num_rounds):
            domain_path = self.openapi_loader.get_random_domain_path(domains)
            openapis = self.openapi_loader.load_openapi_specs(domain_path)
            await updater.update_status(TaskState.working, new_agent_text_message(
                f"[Round {round_id + 1}] loaded {len(openapis)} OpenAPI specifications."))

            query_text, endpoints = self.openapi_loader.load_query(domain_path)
            self.expected_endpoints.append(endpoints)
            await updater.update_status(TaskState.working, new_agent_text_message(
                f"[Round {round_id + 1}] selected query: {query_text}"))

            for role in participants.keys():
                await self.code_generator.generate_code_for_query(role, openapis, query_text, updater, dictionary)

        return dictionary

    async def judge_code(self, code_text: dict[str, list[str]], updater=TaskUpdater) -> CodeEval:

        intermediate_results = {role: {"recall": [], "precision": [], "f1": []} for role in code_text.keys()}

        for agent, generated_codes in code_text.items():
            for round_index, code in enumerate(generated_codes):
                analysis = Analysis(code)
                retrieved_endpoints = analysis.perform_analysis()
                await updater.update_status(TaskState.working, new_agent_text_message(
                    f"Analysis for {agent} in round {round_index + 1}: Retrieved endpoints: {retrieved_endpoints}"))  # debugging
                normalized_retrieved = {normalize_endpoint(ep) for ep in retrieved_endpoints}
                normalized_expected = [normalize_endpoint(ep) for ep in self.expected_endpoints[round_index]]
                await updater.update_status(TaskState.working, new_agent_text_message(
                    f"Normalized for {agent} in round {round_index + 1}: Retrieved: {normalized_retrieved}, Expected: {normalized_expected}"))  # debugging

                recall = MetricEvaluator.compute_recall(normalized_retrieved, normalized_expected)
                precision = MetricEvaluator.compute_precision(normalized_retrieved, normalized_expected)
                f1 = MetricEvaluator.compute_f1(precision, recall)
                intermediate_results[agent]["recall"].append(recall)
                intermediate_results[agent]["precision"].append(precision)
                intermediate_results[agent]["f1"].append(f1)

        avg_results = {}
        for agent, metrics in intermediate_results.items():
            avg_results[agent] = {
                "recall": sum(metrics["recall"]) / len(metrics["recall"]) if metrics["recall"] else 0.0,
                "precision": sum(metrics["precision"]) / len(metrics["precision"]) if metrics["precision"] else 0.0,
                "f1": sum(metrics["f1"]) / len(metrics["f1"]) if metrics["f1"] else 0.0,
            }
        max_agent = max(avg_results, key=lambda a: avg_results[a]["f1"])
        participants_recalls = {agent: CodeScore(recall=avg_results[agent]["recall"],
                                                 precision=avg_results[agent]["precision"],
                                                 f1=avg_results[agent]["f1"]) for agent in avg_results.keys()}
        code_eval = CodeEval(
            participants=participants_recalls,
            winner="The winner is " + max_agent + " with an F1 score of " + f"{avg_results[max_agent]['f1']:.2f}."
        )
        return code_eval


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
        agent = CodeJudge()
        executor = GreenExecutor(agent)
        agent_card = judge_agent_card("CodeJudge", agent_url)
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
