import json
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message
from endpoint_evaluator import normalize_endpoint
from socbenchsc.src.socbenchsc import Analysis


async def run_scenario_hard(
        self,
        role: str,
        scenario: str,
        openapis: list[dict],
        query_text: str,
        agent_url: str,
        updater: TaskUpdater
) -> str:
    generated_code = await request_code_from_agent(
        self,
        query_text,
        openapis,
        scenario,
        agent_url,
        new_conversation=True
    )

    await updater.update_status(
        TaskState.working,
        new_agent_text_message(
            f"[{role}][{scenario}] Generated code:\n{generated_code[:100]}..."
        )
    )

    return generated_code


async def run_scenario_medium(
        self,
        role: str,
        scenario: str,
        openapis: list[dict],
        query_text: str,
        agent_url: str,
        updater: TaskUpdater
) -> str:
    max_attempts = 3
    best_code = ""
    best_endpoint_count = -1

    def confirmation_is_positive(text: str) -> bool:
        text = text.lower()
        return text.startswith("yes") or text.startswith("**yes**")

    for attempt in range(1, max_attempts + 1):

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[{role}][{scenario}] Attempt {attempt}/{max_attempts}: Sending code generation request to {agent_url}"
            )
        )

        generated_code = await request_code_from_agent(
            self,
            query_text,
            openapis,
            scenario,
            agent_url,
            new_conversation=(attempt == 1)
        )
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[{role}][{scenario}] Attempt {attempt} - Generated code:\n"
                f"{generated_code[:100]}..."
            )
        )

        analysis = Analysis(generated_code)
        retrieved = analysis.perform_analysis()
        normalized_retrieved = {normalize_endpoint(ep) for ep in retrieved}

        endpoint_count = len(normalized_retrieved)

        if endpoint_count > best_endpoint_count:
            best_endpoint_count = endpoint_count
            best_code = generated_code

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[{role}][{scenario}] Attempt {attempt} - Extracted endpoints:\n"
                f"{normalized_retrieved}"
            )
        )
        confirmation_request = {
            "query": (
                f"You generated code for the query: '{query_text}'. "
                f"The static analysis extracted these endpoints: "
                f"{list(normalized_retrieved)}. "
                f"Are these the correct endpoints you intended to use? "
                f"Please answer 'Yes' or 'No' and briefly explain."
            ),
            "scenario": "confirmation"
        }

        confirmation_response = await self._tool_provider.talk_to_agent(
            message=json.dumps(confirmation_request),
            url=agent_url,
            new_conversation=False
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[{role}][{scenario}] Attempt {attempt} - Confirmation response:\n"
                f"{confirmation_response}"
            )
        )
        if confirmation_is_positive(confirmation_response):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[{role}][{scenario}] Confirmation accepted on attempt {attempt}"
                )
            )
            return generated_code

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[{role}][{scenario}] Confirmation rejected on attempt {attempt}"
            )
        )
    await updater.update_status(
        TaskState.working,
        new_agent_text_message(
            f"[{role}][{scenario}] Max attempts reached. Returning best generated code."
        )
    )

    return best_code


async def request_code_from_agent(
        self,
        query_text: str,
        openapis: list[dict],
        scenario: str,
        agent_url: str,
        new_conversation: bool
) -> str:
    request_data = {
        "query": query_text,
        "openapi_specs": openapis,
        "scenario": scenario
    }
    return await self._tool_provider.talk_to_agent(
        message=json.dumps(request_data),
        url=agent_url,
        new_conversation=new_conversation
    )
