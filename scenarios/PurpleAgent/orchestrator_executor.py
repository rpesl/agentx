import json, logging, os, sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)
from scenarios.PurpleAgent.base_executor import BaseExecutor
from scenarios.GreenAgent.endpoint_evaluator import normalize_endpoint
from scenarios.GreenAgent.socbenchsc.src.socbenchsc import Analysis
logger = logging.getLogger("OrchestratorExecutor")


class OrchestratorExecutor(BaseExecutor):
    """
    Improved multi-agent orchestration:
    Workflow:
    1. CoderAgent  → generates code (with full context forwarded)
    2. AuditorAgent → MCP-backed JSON audit report
    3. CoderAgent  → refinement pass using audit feedback
    4. Repeat up to MAX_REFINEMENT_ROUNDS if endpoint count keeps improving
    5. Return the code with the most extracted endpoints
    """

    MAX_REFINEMENT_ROUNDS: int = 3

    async def run_logic(self, task_description: str, context: dict) -> str:
        mode = context.get("mode", "code")

        if mode == "confirm":
            logger.info("Confirm mode – delegating directly to CoderAgent")
            return await self._ask_coder(task_description, context)

        return await self._orchestrate(task_description, context)

    async def _orchestrate(self, task_description: str, context: dict) -> str:
        coder_url   = os.getenv("CODER_AGENT_URL",   "http://127.0.0.1:9019")
        auditor_url = os.getenv("AUDITOR_AGENT_URL", "http://127.0.0.1:9020")

        best_code            = ""
        best_endpoint_count  = -1

        logger.info("Step 1 – Initial code generation")
        current_code = await self._ask_coder(task_description, context)
        if current_code is None:
            logger.warning("CoderAgent returned no code – treating as empty")
            current_code = ""
        current_code = self.clean_generated_code(current_code)

        ep_count = self._count_endpoints(current_code)
        logger.info(f"Initial endpoint count: {ep_count}")
        if ep_count > best_endpoint_count:
            best_code, best_endpoint_count = current_code, ep_count

        for round_no in range(1, self.MAX_REFINEMENT_ROUNDS + 1):
            logger.info(f"Round {round_no} - Auditing code")

            audit_context = {**context, "mode": "audit", "code": current_code}
            audit_report_raw = await self.call_other_purple_agent(
                agent_url=auditor_url,
                message=(
                    f"Audit this code for the task:\n{task_description}\n\n"
                    f"CODE:\n{current_code}"
                ),
                context=audit_context,
            )

            audit = self._parse_audit(audit_report_raw)
            logger.info(
                f"Audit summary: confirmed={len(audit['confirmed'])}, "
                f"wrong={len(audit['wrong'])}, missing={len(audit['missing'])}"
            )

            if not audit["wrong"] and not audit["missing"]:
                logger.info("Audit clean – stopping refinement loop")
                break

            logger.info(f"Round {round_no} – Refining code based on audit")
            refined_code = await self._refine_with_coder(
                coder_url, task_description, current_code, audit, context
            )
            refined_code = self.clean_generated_code(refined_code)

            new_ep_count = self._count_endpoints(refined_code)
            logger.info(f"Endpoint count after refinement: {new_ep_count}")

            if new_ep_count >= best_endpoint_count:
                best_code, best_endpoint_count = refined_code, new_ep_count

            if new_ep_count <= ep_count:
                logger.info("No endpoint improvement – stopping loop")
                break

            current_code = refined_code
            ep_count     = new_ep_count

        logger.info(f"Final best endpoint count: {best_endpoint_count}")
        return best_code


    async def _ask_coder(self, message: str, context: dict) -> str:
        coder_url = os.getenv("CODER_AGENT_URL", "http://127.0.0.1:9019")
        return await self.call_other_purple_agent(
            agent_url=coder_url,
            message=message,
            context=context,
        )


    async def _refine_with_coder(
        self,
        coder_url: str,
        task_description: str,
        original_code: str,
        audit: dict,
        context: dict,
    ) -> str:
        wrong_lines   = "\n".join(
            f"  - Used '{w['used']}' → should be '{w['correct']}': {w['reason']}"
            for w in audit["wrong"]
        ) or "  (none)"
        missing_lines = "\n".join(f"  - {m}" for m in audit["missing"]) or "  (none)"

        refinement_message = (
            f"Fix this Python code based on the auditor's findings.\n\n"
            f"TASK:\n{task_description}\n\n"
            f"ORIGINAL CODE:\n```python\n{original_code}\n```\n\n"
            f"AUDIT FINDINGS:\n"
            f"Wrong endpoints:\n{wrong_lines}\n"
            f"Missing endpoints:\n{missing_lines}\n\n"
            f"Instructions:\n"
            f"- Correct every wrong endpoint (path + method)\n"
            f"- Add every missing endpoint with a sensible call\n"
            f"- Keep all confirmed endpoints as-is\n"
            f"- Return ONLY the fixed Python code"
        )

        return await self.call_other_purple_agent(
            agent_url=coder_url,
            message=refinement_message,
            context={**context, "mode": context.get("mode", "code")},
        )


    @staticmethod
    def _parse_audit(raw: str) -> dict:
        template = {"confirmed": [], "wrong": [], "missing": [], "summary": ""}
        if not raw:
            return template
        try:
            data = json.loads(raw)
            return {**template, **data}
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse audit JSON – treating as no issues")
            return {**template, "summary": raw[:200]}



    @staticmethod
    def _count_endpoints(code: str) -> int:
        if not code:
            return 0
        try:
            analysis = Analysis(code)
            retrieved = analysis.perform_analysis()
            return len({normalize_endpoint(ep) for ep in retrieved})
        except (SyntaxError, ValueError, NotImplementedError):
            return 0