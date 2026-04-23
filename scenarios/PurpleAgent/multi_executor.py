import logging, json
from single_executor import SingleExecutor
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam, ChatCompletionToolMessageParam, ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
    Function as ToolCallFunction)

logger = logging.getLogger("MultiExecutor")


class MultiExecutor(SingleExecutor):
    """
    A sophisticated multi-evaluation flow that:
    1. Generates code (via SingleExecutor).
    2. Actively validates the code against MCP tools.
    3. Goes through a final correction loop.
    """

    async def run_logic(self, task_description: str, context: dict) -> str:
        """Overrides the base run_logic to implement a multi-step generation and validation process."""
        if context.get("mode") == "confirm":
            return await super().run_logic(task_description, context)

        logger.info("Planning API strategy...")
        plan = await self._plan_strategy(task_description, context)

        logger.info("Generating code based on plan...")
        task_with_plan = f"{task_description}\n\n USE THIS STRATEGY: {plan}"
        initial_code = await super().run_logic(task_with_plan, context)

        logger.info("Starting independent validation...")
        validation_results = await self._validate_code(initial_code, task_description, context)

        logger.info("Applying final corrections...")
        final_code = await self._final_refinement(initial_code, validation_results, task_description)

        return final_code

    async def _plan_strategy(self, task: str, context: dict) -> str:
        """Before generating code, we ask the model to plan out which domains and services it will interact with based on the task."""
        instance_id = context.get("instance_id", 1)
        domains = await self._call_mcp_tool("list_available_domains", {"instance_id": instance_id})

        plan_prompt = f"""
        Task: {task}
        Available Domains in Instance {instance_id}: {domains}

        Which specific domains and services are needed? 
        Briefly describe the workflow (e.g., 'First get user ID from Energy domain, then fetch status').
        Be concise. No code yet.
        """

        response = self.client.chat.completions.create(
            model="NousResearch/Hermes-4-405B",
            messages=[{"role": "user", "content": plan_prompt}]
        )
        return response.choices[0].message.content

    async def _validate_code(self, code: str, task: str, context: dict) -> str:
        """The auditor agent will use tools to fetch real specs and compare them against the generated code."""
        logger.info("Validating generated endpoints against real specs...")
        instance_id = context.get("instance_id", 1)
        use_rag = (context.get("mode") == "rag")

        tools = await self._list_mcp_tools(use_rag=use_rag)

        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": f"""
                        You are an independent API Auditor. You did NOT write the code. 
                        Your only job is to verify if the code uses the correct API endpoints for instance {instance_id}.

                        STEPS:
                        1. Identify all API endpoints used in the provided code.
                        2. Use your tools to fetch the REAL OpenAPI specs for these services.
                        3. Compare: Are the paths, methods, and parameters exactly as defined in the specs?
                        4. List all discrepancies clearly. If it's perfect, say "No issues found".
                        """.strip(),
        }

        user_msg: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": f"Please audit this code for the task: '{task}'\n\nCODE TO CHECK:\n{code}",
        }

        agent2_messages: list[ChatCompletionMessageParam] = [system_msg, user_msg]

        response = self.client.chat.completions.create(
            model="NousResearch/Hermes-4-405B",
            messages=agent2_messages,
            tools=tools,
            tool_choice="auto"
        )

        iteration = 0
        while response.choices[0].message.tool_calls and iteration < 3:
            iteration += 1
            assistant_message = response.choices[0].message

            tool_calls_param = [
                ChatCompletionMessageToolCallParam(
                    id=tc.id,
                    type="function",
                    function=ToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in (assistant_message.tool_calls or [])
            ]

            assistant_msg_obj: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": tool_calls_param,
            }
            agent2_messages.append(assistant_msg_obj)

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                logger.info(f"Auditor calling tool: {tool_name}")
                tool_result = await self._call_mcp_tool(tool_name, tool_args)

                tool_msg_obj: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result if isinstance(tool_result, str) else json.dumps(tool_result),
                }
                agent2_messages.append(tool_msg_obj)

            response = self.client.chat.completions.create(
                model="NousResearch/Hermes-4-405B",
                messages=agent2_messages,
                tools=tools
            )

        report = response.choices[0].message.content or "Audit complete."
        logger.info(f"Agent 2 Report:\n{report}")
        return report

    async def _final_refinement(self, initial_code: str, validation_report: str, task: str) -> str:
        """Based on the auditor's report, we ask the model to make final corrections to the code."""
        logger.info("Applying final refinements based on auditor feedback:")
        logger.info("Initial code:\n" + initial_code)
        logger.info("Validation report:\n" + validation_report)

        refine_prompt = f"""
        Original Task: {task}

        Initial Code:
        {initial_code}

        Validation Report from Auditor:
        {validation_report}

        Based on the report, fix the Python code. 
        Ensure all endpoints, methods, and parameters match the specifications perfectly.
        Return ONLY the final Python code.
        """

        response = self.client.chat.completions.create(
            model="NousResearch/Hermes-4-405B",
            messages=[{"role": "user", "content": refine_prompt}]
        )
        logger.info("Response from refinement step:\n" + response.choices[0].message.content)
        return self.clean_generated_code(response.choices[0].message.content)
