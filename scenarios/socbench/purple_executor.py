import json
import os
import re
import logging
from openai import OpenAI
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task, new_agent_text_message
from mcp_tools import get_mcp_tools_for_openai, execute_mcp_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PurpleExecutor")

class PurpleExecutor(AgentExecutor):

    def __init__(self, api_key_env: str = "NEBIUS_API_KEY"):
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")

        self.client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=self.api_key
        )

    @staticmethod
    def clean_generated_code(text: str) -> str:
        text = text.encode("ascii", "ignore").decode()
        code_blocks = re.findall(r"```(?:[\w+-]*)?\n(.*?)```", text, re.DOTALL)
        if code_blocks:
            filtered_blocks = [
                block.strip() for block in code_blocks
                if block.strip() and not block.strip().startswith("pip install")
            ]

            if filtered_blocks:
                cleaned = max(filtered_blocks, key=len)
                cleaned = re.sub(r"^#.*\n", "", cleaned)
                return cleaned
            return ""
        return text.strip()

    @staticmethod
    def clean_text(text: str) -> str:
        result = text.encode("ascii", "ignore").decode().strip()
        return result

    async def generate_code_with_rag_mcp(self, prompt: str, instance_id: int) -> str:
        system_message = f"""
        You are a code generation assistant specialized in creating Python code based on OpenAPI specifications.

        You have access to MCP (Model Context Protocol) tools INCLUDING RAG capabilities for loading OpenAPI specifications on-demand.

        You must ONLY use domains from instance '{instance_id}'.

        AVAILABLE MCP TOOLS:
        - list_available_domains(instance_id={instance_id}): List available domains from the specified instance
        - retrieve_relevant_specs_with_rag(domain_path, query): Use RAG to get ONLY the most relevant specs based on the query

        WORKFLOW FOR RAG:
        1. Use list_available_domains(instance_id={instance_id}) to see available domains
        2. Identify the most relevant domain based on the user's query. You must only choose one domain from the provided list.
        3. Use retrieve_relevant_specs_with_rag(domain_path, query) to only get relevant specs via semantic search
        4. Generate Python code using the 'requests' library based on the loaded specs
        5. Return ONLY the Python code, no explanations
        """

        tools = get_mcp_tools_for_openai(include_rag=True)

        return await self._generate_code_mcp_core(
            prompt=prompt,
            system_message=system_message,
            tools=tools,
            log_rag=True
        )

    async def generate_code_with_mcp(self, prompt: str, instance_id: int) -> str:
        system_message = f"""
        You are a code generation assistant specialized in creating Python code based on OpenAPI specifications.

        You have access to MCP (Model Context Protocol) tools to load OpenAPI specifications on-demand.

        You must ONLY use domains from instance '{instance_id}'.

        AVAILABLE MCP TOOLS:
        - list_available_domains(instance_id={instance_id}): List available domains from the specified instance
        - load_openapi_specs(domain_path): Load OpenAPI specs from a domain

        WORKFLOW:
        1. Use list_available_domains(instance_id={instance_id}) to see available domains in this instance
        2. Identify the most relevant domain based on the user's query. You must only choose one domain from the provided list.
        3. Use load_openapi_specs() with the full domain path to get the OpenAPI specs
        4. Generate Python code using the 'requests' library based on the loaded specs
        5. Return ONLY the Python code, no explanations
        """

        tools = get_mcp_tools_for_openai(include_rag=False)

        return await self._generate_code_mcp_core(
            prompt=prompt,
            system_message=system_message,
            tools=tools,
            log_rag=False
        )

    async def _generate_code_mcp_core(
            self,
            *,
            prompt: str,
            system_message: str,
            tools: list,
            log_rag: bool
    ) -> str:

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        logger.info(f"Available tools: {[t['function']['name'] for t in tools]}")

        try:
            logger.info("Calling LLM (1st call)...")
            response = self.client.chat.completions.create(
                model="moonshotai/Kimi-K2-Instruct",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            logger.info("LLM responded successfully")
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}", exc_info=True)
            raise

        iteration = 0
        max_iterations = 5

        while response.choices[0].message.tool_calls and iteration < max_iterations:
            iteration += 1
            logger.info(f"Tool iteration {iteration}/{max_iterations}")

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name

                try:
                    tool_args = json.loads(tool_call.function.arguments)
                    logger.info(f"Executing tool: {tool_name}")
                    logger.info(f"Tool args: {json.dumps(tool_args, indent=2)}")

                    tool_result = execute_mcp_tool(tool_name, tool_args)
                    result_str = json.dumps(tool_result)

                    if log_rag and tool_name == "retrieve_relevant_specs_with_rag":
                        logger.info("RAG retrieved Specs:")
                        if isinstance(tool_result, list):
                            for i, spec in enumerate(tool_result):
                                if isinstance(spec, dict):
                                    title = spec.get("info", {}).get("title", "Unknown")
                                    paths = list(spec.get("paths", {}).keys())
                                    logger.info(f"Spec {i + 1}: {title}")
                                    logger.info(f"  Sample paths: {paths}")

                except Exception as e:
                    logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
                    result_str = json.dumps({"error": str(e)})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str
                })

            try:
                logger.info(f"Calling LLM with tool results (iteration {iteration})...")
                response = self.client.chat.completions.create(
                    model="moonshotai/Kimi-K2-Instruct",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}", exc_info=True)
                raise

        final_response = response.choices[0].message.content
        if final_response is None:
            final_response = ""

        cleaned_code = self.clean_generated_code(final_response)
        return cleaned_code

    async def generate_text(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a reviewer agent. "
                        "Confirm or reject whether extracted API endpoints match the user's intent. "
                        "Respond with 'Yes' or 'No' followed by a brief explanation."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )

        result = self.clean_text(response.choices[0].message.content.strip())
        return result

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:

        request_text = context.get_user_input()

        try:
            request_data = json.loads(request_text)
            prompt = request_data.get("prompt", "")
            mode = request_data.get("mode", "code")
            instance_id = request_data.get("instance_id")
            logger.info(f"Prompt: {prompt}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {str(e)}")
            msg = context.message
            if msg:
                task = new_task(msg)
                await event_queue.enqueue_event(task)
                from a2a.server.tasks import TaskUpdater
                updater = TaskUpdater(event_queue, task.id, task.context_id)
                await updater.failed(
                    new_agent_text_message(
                        f"Invalid request format: {str(e)}",
                        context_id=context.context_id
                    )
                )
            return

        msg = context.message
        if msg:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
            logger.info(f"Task created: {task.id}")
        else:
            logger.error("No message context")
            return

        from a2a.server.tasks import TaskUpdater
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            if mode == "confirm":
                logger.info("Mode: CONFIRMATION")
                result = await self.generate_text(prompt)
            elif mode == "rag":
                logger.info("Mode: RAG CODE GENERATION")
                result = await self.generate_code_with_rag_mcp(prompt, instance_id)
            else:
                logger.info("Mode: CODE GENERATION")
                result = await self.generate_code_with_mcp(prompt, instance_id)

            logger.info(f"Result preview:\n"
                        f"{result[:1000]}..."
                        )

            await updater.complete(
                message=new_agent_text_message(result, context_id=context.context_id)
            )
            logger.info("Task completed successfully")

        except Exception as e:
            logger.error(f"Execution failed: {str(e)}", exc_info=True)

            await updater.failed(
                new_agent_text_message(
                    f"Error during execution: {str(e)}",
                    context_id=context.context_id
                )
            )

    async def cancel(self, request: RequestContext, event_queue: EventQueue):
        return None
