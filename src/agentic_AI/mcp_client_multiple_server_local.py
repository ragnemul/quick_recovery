import asyncio
from xml.sax import handler

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["/datos/PycharmProjects/GausianProcess/PhD/SLM/AgenticAI/langchain/src/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                # Make sure you start your weather server on port 8000
                "url": "http://localhost:8000/mcp",
                "transport": "http",
            }
        }
    )

    # Get tools
    tools = await client.get_tools()

    from langchain.agents.middleware import wrap_tool_call
    from langchain.messages import ToolMessage
    @wrap_tool_call
    def handle_tool_errors(request, handler):
        """Handle tool execution errors with custom messages."""
        try:
            return handler(request)
        except Exception as e:
            # Return a custom error message to the model
            return ToolMessage(
                content=f"Tool error: Please check your input and try again. ({str(e)}",
            tool_call_id=request.tool_call["id"]
        )

    model = ChatOllama(
                model="llama3.2:3b",           # hace bien el overwrite, considerando el prompting
                #model="llama3.1:latest",        # no hace bien el overwrite impuesto por las herramientas externas
                #model="qwen3:8b",              # hace bien el overwrite, considerando un promptint estricto
                #model="phi4:latest",
                #model="gpt-oss:20b",            # hace bien el overwrite, considerando el prompting
                validate_model_on_init=True,
                temperature=0,
                middleware=[handle_tool_errors],
                tools=[tool.name for tool in tools],  # Register tool names with the model
            )

    # Create and run the agent
    agent = create_agent(model, tools)

    math_response = await agent.ainvoke(
        {
            "messages":
            [
                {"role": "system","content": "You are a helpful assistant that performs mathematical calculations based only in external tools provided. Do not make any calculation out of provided by external tools. Do not evaluate the results. Do not correct the results. Do not verify answer. Just give the answer provided by the external tools. Inform if the result was provided by the external tools or by you."},
                {"role": "user", "content": "what's (3 + 5) x 12?"}
            ]
        }
    )

    # Optionally, print or handle the response
    for message in math_response["messages"]:
        print(type(message))  # Identifies the message type (e.g., <class 'langchain_core.messages.ai.AIMessage'>)
        print(message.content)  # Retrieves the content or result of the message


    weather_response = await agent.ainvoke(
        {
            "messages": [
                {"role": "system","content": "You are a helpful assistant that provides weather information based only in external tools provided. Do not make any weather information out of provided by external tools. Do not evaluate the results. Do not correct the results. Do not verify answer. Just give the answer provided by the external tools. Inform if the result was provided by the external tools or by you."},
                {"role": "user", "content": "what is the weather in Madrid?"}
            ]
        }
    )

    for message in weather_response["messages"]:
        print(type(message))  # Identifies the message type (e.g., <class 'langchain_core.messages.ai.AIMessage'>)
        print(message.content)  # Retrieves the content or result of the message



if __name__ == "__main__":
    asyncio.run(main())
