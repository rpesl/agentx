import json, os, re, logging
from abc import abstractmethod
from openai import OpenAI
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_task, new_agent_text_message
from mcp import ClientSession
from mcp.client.sse import sse_client
from agentbeats.tool_provider import ToolProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BaseExecutor")

class BaseExecutor(AgentExecutor):

    def __init__(self, api_key_env: str = "NEBIUS_API_KEY"):
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")
        self.mcp_url = "http://localhost:8000/sse"
        self.client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=self.api_key
        )
        self.mcp_session: ClientSession | None = None
        self._sse_context = None
        self._session_context = None
        self.mcp_initialized = False
        self._tool_provider = ToolProvider()

    async def call_other_purple_agent(self, agent_url: str, message: str, context: dict | None = None) -> str:
        full_message = message
        if context:
            full_message = f"{message}\n\n[Context: {json.dumps(context)}]"

        logger.info(f"Calling other agent at {agent_url}...")
        return await self._tool_provider.talk_to_agent(
            message=full_message,
            url=agent_url,
            new_conversation=True
        )

    async def _ensure_mcp_connected(self):
        if self.mcp_initialized:
            logging.info("MCP session already initialized")
            return

        try:
            logger.info(f"Connecting to shared MCP Server at {self.mcp_url}")

            self._sse_context = sse_client(self.mcp_url)
            streams = await self._sse_context.__aenter__()
            stdio, write = streams

            self._session_context = ClientSession(stdio, write)
            self.mcp_session = await self._session_context.__aenter__()

            await self.mcp_session.initialize()
            self.mcp_initialized = True
            logger.info("Successfully connected to shared MCP session")

        except Exception as e:
            logger.error(f"Failed to connect to central MCP: {e}")
            self.mcp_initialized = False
            raise

    async def _cleanup_mcp(self):
        if self._session_context is not None:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing MCP session: {e}")
            self._session_context = None
            self.mcp_session = None

        if self._sse_context is not None:
            try:
                await self._sse_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing SSE connection: {e}")
            self._sse_context = None

        self.mcp_initialized = False

    async def _call_mcp_tool(self, tool_name: str, arguments: dict):
        await self._ensure_mcp_connected()

        try:
            result = await self.mcp_session.call_tool(tool_name, arguments)
            if result.content:
                if hasattr(result.content[0], 'text'):
                    return result.content[0].text
                return str(result.content[0])
            return {}
        except Exception as e:
            logger.error(f"MCP Tool call failed: {str(e)}", exc_info=True)
            return {"error": str(e)}

    async def _list_mcp_tools(self, use_rag: bool = False) -> list[ChatCompletionToolParam]:
        await self._ensure_mcp_connected()

        tools_list = await self.mcp_session.list_tools()
        openai_tools: list[ChatCompletionToolParam] = []
        for tool in tools_list.tools:
            if not use_rag and tool.name == "retrieve_relevant_specs_with_rag":
                continue
            if use_rag and tool.name == "load_openapi_specs":
                continue
            openai_tool: ChatCompletionToolParam = {
                "type": "function",
                "function": FunctionDefinition(
                     name=tool.name,
                     description=tool.description or "",
                     parameters=tool.inputSchema
                )
            }
            openai_tools.append(openai_tool)

        logger.info(f"Available OpenAI MCP tools: {[tool['function']['name'] for tool in openai_tools]}")
        return openai_tools

    @staticmethod
    def extract_context(message: str) -> tuple[str, dict | None]:
        match = re.search(r'\[Context:\s*(\{.*?})]', message, re.DOTALL)

        if match:
            try:
                context = json.loads(match.group(1))
                clean_task = re.sub(r'\[Context:.*?]', '', message, flags=re.DOTALL).strip()
                return clean_task, context
            except json.JSONDecodeError:
                logger.warning("Found context marker but couldn't parse JSON")

        return message, None

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

    @abstractmethod
    async def run_logic(self, task_description: str, context: dict) -> str:
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request_text = context.get_user_input()
        task_description, json_context = self.extract_context(request_text)

        msg = context.message
        if not msg:
            logger.error("No message in RequestContext")
            return

        task = new_task(msg)
        await event_queue.enqueue_event(task)
        from a2a.server.tasks import TaskUpdater
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            logger.info(f"Starting execution for mode: {json_context.get('mode')}")
            result = await self.run_logic(task_description, json_context)
            logger.info("Result:" + result[:1000])

            await updater.complete(
                message=new_agent_text_message(result, context_id=context.context_id)
            )
            logger.info("Task completed successfully")
        except Exception as e:
            logger.error(f"Execution failed: {str(e)}", exc_info=True)
            await updater.failed(
                new_agent_text_message(f"Error: {str(e)}", context_id=context.context_id)
            )
        finally:
            await self._cleanup_mcp()

    async def cancel(self, request: RequestContext, event_queue: EventQueue):
        await self._cleanup_mcp()
