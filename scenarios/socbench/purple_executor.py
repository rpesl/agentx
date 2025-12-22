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
        logger.info("PurpleExecutor initialized")

    @staticmethod
    def clean_generated_code(text: str) -> str:
        logger.info(f"Cleaning code - input length: {len(text)} chars")
        text = text.encode("ascii", "ignore").decode()
        code_blocks = re.findall(r"```(?:[\w+-]*)?\n(.*?)```", text, re.DOTALL)
        logger.info(f"Found {len(code_blocks)} code blocks")

        if code_blocks:
            filtered_blocks = [
                block.strip() for block in code_blocks
                if block.strip() and not block.strip().startswith("pip install")
            ]
            logger.info(f"Filtered to {len(filtered_blocks)} valid blocks")

            if filtered_blocks:
                cleaned = max(filtered_blocks, key=len)
                logger.info(f"Selected largest block: {len(cleaned)} chars")
                cleaned = re.sub(r"^#.*\n", "", cleaned)
                logger.info(f"After removing comments: {len(cleaned)} chars")
                return cleaned
            return ""
        logger.info("No code blocks found, returning raw text")
        return text.strip()

    @staticmethod
    def clean_text(text: str) -> str:
        logger.info(f"Cleaning text - input: {len(text)} chars")
        result = text.encode("ascii", "ignore").decode().strip()
        logger.info(f"Cleaned text - output: {len(result)} chars")
        return result

    async def generate_code_with_mcp(self, prompt: str) -> str:
        """
        Generiert Code und lädt OpenAPI Specs automatisch via MCP.
        Der Agent entscheidet selbst, welche Domain er braucht!
        """
        logger.info("Starting code generation with MCP")
        logger.info(f"Prompt length: {len(prompt)} chars")
        logger.info(f"Prompt preview: {prompt[:200]}...")

        system_message = """You are a code generation assistant specialized in creating Python code based on OpenAPI specifications.

You have access to MCP (Model Context Protocol) tools to load OpenAPI specifications on-demand.

AVAILABLE MCP TOOLS:
- list_available_domains(): List available benchmark domains
- load_openapi_specs(): Load OpenAPI specs from the domain

WORKFLOW:
1. Use the list_available_domains tool to find relevant domains
2. Use the load_openapi_specs tool to get the OpenAPI specifications
3. Analyze the specs to understand available endpoints
4. Generate Python code using the 'requests' library
5. Return ONLY the Python code, no explanations
"""
        logger.info("System message configured")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        logger.info(f"Messages prepared: {len(messages)} messages")

        tools = get_mcp_tools_for_openai()
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
            has_tools = bool(response.choices[0].message.tool_calls)
            logger.info(f"Response has tool_calls: {has_tools}")

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
            logger.info(f"Added assistant message, total messages: {len(messages)}")

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                logger.info(f"Executing tool: {tool_name}")
                logger.info(f"Tool args: {json.dumps(tool_args, indent=2)}")

                try:
                    tool_result = execute_mcp_tool(tool_name, tool_args)

                    if isinstance(tool_result, list):
                        logger.info(f"Tool result: list with {len(tool_result)} items")
                        if tool_result and isinstance(tool_result[0], dict):
                            logger.info(f"First item keys: {list(tool_result[0].keys())}")
                    elif isinstance(tool_result, dict):
                        logger.info(f"Tool result: dict with keys: {list(tool_result.keys())}")
                    else:
                        logger.info(f"Tool result type: {type(tool_result)}")

                    result_str = json.dumps(tool_result)
                    logger.info(f"Serialized result: {len(result_str)} chars")

                except Exception as e:
                    logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
                    result_str = json.dumps({"error": str(e)})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str
                })
                logger.info(f"Added tool result, total messages: {len(messages)}")

            try:
                logger.info(f"Calling LLM with tool results (iteration {iteration})...")
                response = self.client.chat.completions.create(
                    model="moonshotai/Kimi-K2-Instruct",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )

                has_tool_calls = bool(response.choices[0].message.tool_calls)
                logger.info(f"LLM responded. Has more tool_calls: {has_tool_calls}")

                if has_tool_calls:
                    logger.info(f"LLM wants {len(response.choices[0].message.tool_calls)} more tool call(s)")
                else:
                    logger.info("LLM is done with tools - will generate final code")

            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}", exc_info=True)
                raise

        logger.info("Exiting tool loop - processing final response")
        final_response = response.choices[0].message.content
        logger.info(f"Final response length: {len(final_response)} chars")
        logger.info(f"Final response preview: {final_response[:300]}...")

        cleaned_code = self.clean_generated_code(final_response)
        logger.info(f"Code generation complete: {len(cleaned_code)} chars, {len(cleaned_code.split(chr(10)))} lines")
        logger.info(f"Code preview: {cleaned_code[:500]}...")

        return cleaned_code

    async def generate_text(self, prompt: str) -> str:
        logger.info("Starting confirmation generation")
        logger.info(f"Prompt length: {len(prompt)} chars")
        logger.info(f"Prompt: {prompt[:200]}...")

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
        logger.info(f"Confirmation generated: {len(result)} chars")
        logger.info(f"Result: {result}")
        return result

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("=" * 80)
        logger.info("PurpleExecutor.execute() called")
        logger.info("=" * 80)

        request_text = context.get_user_input()
        logger.info(f"Raw request text length: {len(request_text)} chars")
        logger.info(f"Raw request: {request_text}")

        try:
            request_data = json.loads(request_text)
            prompt = request_data.get("prompt", "")
            mode = request_data.get("mode", "code")

            logger.info(f"Parsed JSON successfully - Mode: {mode}")
            logger.info(f"Prompt length: {len(prompt)} chars")
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
        logger.info("TaskUpdater created")

        try:
            if mode == "confirm":
                logger.info("Mode: CONFIRMATION")
                result = await self.generate_text(prompt)
            else:
                logger.info("Mode: CODE GENERATION")
                result = await self.generate_code_with_mcp(prompt)

            logger.info("Generation complete")
            logger.info(f"Result preview: {result[:300]}...")

            await updater.complete(
                message=new_agent_text_message(result, context_id=context.context_id)
            )
            logger.info("Task completed successfully")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Execution failed: {str(e)}", exc_info=True)

            await updater.failed(
                new_agent_text_message(
                    f"Error during execution: {str(e)}",
                    context_id=context.context_id
                )
            )
            logger.info("=" * 80)

    async def cancel(self, request: RequestContext, event_queue: EventQueue):
        return None
