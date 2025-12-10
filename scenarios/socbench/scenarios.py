from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

from endpoint_evaluator import normalize_endpoint
from socbenchsc.src.socbenchsc import Analysis


async def run_scenario_hard(self, role:str, scenario:str, openapis: list[dict], query_text: str, updater: TaskUpdater, dictionary: dict) -> str:
        generated_code = await self.code_generator.generate_code_for_query(
            role, scenario, openapis, query_text, updater, dictionary)
        return generated_code

async def run_scenario_medium(self, role:str, scenario:str, openapis: list[dict], query_text: str, updater: TaskUpdater, dictionary: dict) -> str:
    generated_code = await self.code_generator.generate_code_for_query(
        role, scenario, openapis, query_text, updater, dictionary)
    analysis = Analysis(generated_code)
    retrieved = analysis.perform_analysis()

    normalized_retrieved = {normalize_endpoint(ep) for ep in retrieved}
    await  updater.update_status(
        TaskState.working,
        new_agent_text_message(f"[{role}][{scenario}] Normalized retrieved endpoints: {normalized_retrieved}")
    )
    confirmation = await self.code_generator.confirm_endpoints(
        role, scenario, list(normalized_retrieved), updater)
    await updater.update_status(
        TaskState.working,
        new_agent_text_message(f"[{role}][{scenario}] Confirmation received: {confirmation}")
    )
    # TODO : Modify generated code based on confirmation if needed

    return generated_code

