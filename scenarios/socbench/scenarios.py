import json
from dataclasses import dataclass
from typing import Callable, Optional
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message
from socbenchsc.src.socbenchsc import Analysis
from endpoint_evaluator import normalize_endpoint


@dataclass
class ScenarioConfig:
    level: str
    max_attempts: int
    requires_confirmation: bool
    prompt_builder: Callable[..., str]
    expected_endpoints: Optional[list[str]] = None


class ScenarioRunner:
    def __init__(self, tool_provider):
        self._tool_provider = tool_provider

    @staticmethod
    async def log(
            updater: TaskUpdater,
            role: str,
            scenario: str,
            msg: str
    ) -> None:
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[{role}][{scenario}] {msg}")
        )

    @staticmethod
    def extract_endpoints(code: str) -> set[str]:
        try:
            analysis = Analysis(code)
            retrieved = analysis.perform_analysis()
        except (SyntaxError, ValueError, NotImplementedError):
            return set()

        return {normalize_endpoint(ep) for ep in retrieved}

    @staticmethod
    def confirmation_is_positive(text: str) -> bool:
        text = text.lower().strip()
        return text.startswith("yes") or text.startswith("**yes**") or text.startswith("'yes'")

    async def request_code(
            self,
            agent_url: str,
            prompt: str,
            instance_id: int,
            mode: str = "code"
    ) -> str:

        request_data = {
            "prompt": prompt,
            "instance_id": instance_id,
            "mode": mode
        }

        return await self._tool_provider.talk_to_agent(
            message=json.dumps(request_data),
            url=agent_url,
            new_conversation=True
        )

    async def request_confirmation(
            self,
            agent_url: str,
            query_text: str,
            extracted: set[str]
    ) -> str:
        prompt = (
            f"\nYou generated code for the query:\n{query_text}\n"
            f"The static analysis extracted the following endpoints from your code:\n"
            f"{list(extracted)}\n"
            f"Are these the correct endpoints you intended to use?\n"
            f"Please answer with 'Yes' or 'No' and briefly explain."
        )

        return await self._tool_provider.talk_to_agent(
            message=json.dumps({
                "mode": "confirm",
                "prompt": prompt
            }),
            url=agent_url,
            new_conversation=True
        )

    async def run_core(
            self,
            role: str,
            agent_url: str,
            updater: TaskUpdater,
            config: ScenarioConfig,
            query_text: str,
            instance_id: int,
            mode: str = "code"
    ) -> str:

        best_code = ""
        best_endpoint_count = -1

        await self.log(
            updater,
            role,
            config.level,
            f"Starting {mode.upper()} scenario ({config.max_attempts} attempts)"
        )

        for attempt in range(1, config.max_attempts + 1):

            await self.log(
                updater,
                role,
                config.level,
                f"Attempt {attempt}/{config.max_attempts}"
            )

            prompt = config.prompt_builder(
                query_text=query_text,
                expected_endpoints=config.expected_endpoints
            )

            code = await self.request_code(
                agent_url=agent_url,
                prompt=prompt,
                instance_id=instance_id,
                mode=mode
            )

            endpoints = self.extract_endpoints(code)

            await self.log(
                updater,
                role,
                config.level,
                f"Extracted endpoints: {endpoints}"
            )

            if len(endpoints) > best_endpoint_count:
                best_code = code
                best_endpoint_count = len(endpoints)

            if not config.requires_confirmation:
                return code

            confirmation = await self.request_confirmation(
                agent_url,
                query_text,
                endpoints
            )

            await self.log(
                updater,
                role,
                config.level,
                f"Confirmation: {confirmation}"
            )

            if self.confirmation_is_positive(confirmation):
                await self.log(
                    updater,
                    role,
                    config.level,
                    "Accepted"
                )
                return code

            await self.log(
                updater,
                role,
                config.level,
                "Rejected - retrying"
            )

        await self.log(
            updater,
            role,
            config.level,
            "Max attempts reached - returning best code"
        )

        return best_code

    @staticmethod
    def easy_prompt(*, expected_endpoints: list[str], query_text: str, **_) -> str:
        return f"""
        Generate Python code for this query:
        Query: {query_text}

        You have access to MCP tools to discover available domains and load their OpenAPI specs.
        Use the tools to find the right domain, load the specs and generate code.

        Expected endpoints to retrieve with your code:
        {expected_endpoints}
        """

    @staticmethod
    def task_prompt(*, query_text: str, **_) -> str:
        return f"""
        Generate Python code for this query:
        {query_text}

        You have access to MCP tools to discover available domains and load their OpenAPI specs.
        Use the tools to find the right domain, load the specs and generate code.
        """

    @staticmethod
    def _rag_easy_prompt(*, expected_endpoints: list[str], query_text: str, **_) -> str:
        return f"""
        Generate Python code using for this query:
        Query: {query_text}

        You should load OpenAPI specifications yourself using an EndpointParser.
        Use RAG to find the most relevant OpenAPI specs for this query.

        Expected endpoints to retrieve with your code:
        {expected_endpoints}
       """

    @staticmethod
    def _rag_task_prompt(*, query_text: str, **_) -> str:
        return f"""
        Generate Python code using for this query:
        {query_text}

        You should load OpenAPI specifications yourself using an EndpointParser.
        Use RAG to find the most relevant OpenAPI specs for this query.
        """

    async def run_easy(self, role, expected_endpoints, query_text, agent_url, updater, instance_id):
        config = ScenarioConfig(
            level="easy",
            max_attempts=3,
            requires_confirmation=True,
            expected_endpoints=expected_endpoints,
            prompt_builder=self.easy_prompt
        )
        return await self.run_core(
            role, agent_url, updater, config, query_text, instance_id, mode="code"
        )

    async def run_medium(self, role, query_text, agent_url, updater, instance_id):
        config = ScenarioConfig(
            level="medium",
            max_attempts=3,
            requires_confirmation=True,
            prompt_builder=self.task_prompt
        )
        return await self.run_core(
            role, agent_url, updater, config, query_text, instance_id, mode="code"
        )

    async def run_hard(self, role, query_text, agent_url, updater, instance_id):
        config = ScenarioConfig(
            level="hard",
            max_attempts=1,
            requires_confirmation=False,
            prompt_builder=self.task_prompt
        )
        return await self.run_core(
            role, agent_url, updater, config, query_text, instance_id, mode="code"
        )

    async def run_rag_easy(self, role, expected_endpoints, query_text, agent_url, updater, instance_id):

        config = ScenarioConfig(
            level="rag_easy",
            max_attempts=3,
            requires_confirmation=True,
            expected_endpoints=expected_endpoints,
            prompt_builder=self._rag_easy_prompt
        )
        return await self.run_core(role, agent_url, updater, config, query_text, instance_id, mode="rag")

    async def run_rag_medium(self, role, query_text, agent_url, updater, instance_id):
        config = ScenarioConfig(
            level="rag_medium",
            max_attempts=3,
            requires_confirmation=True,
            prompt_builder=self._rag_task_prompt
        )
        return await self.run_core(role, agent_url, updater, config, query_text, instance_id, mode="rag")

    async def run_rag_hard(self, role, query_text, agent_url, updater, instance_id):
        config = ScenarioConfig(
            level="rag_hard",
            max_attempts=1,
            requires_confirmation=False,
            prompt_builder=self._rag_task_prompt
        )
        return await self.run_core(
            role, agent_url, updater, config, query_text, instance_id, mode="rag")
