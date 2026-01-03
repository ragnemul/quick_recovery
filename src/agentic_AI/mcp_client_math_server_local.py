import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server_params = StdioServerParameters(
        command="python",
        # Make sure to update to the full absolute path to your math_server.py file
        args=["/datos/PycharmProjects/GausianProcess/PhD/SLM/AgenticAI/langchain/src/math_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            model = ChatOllama(
                model="llama3.2:3b",           # hace bien el overwrite, considerando el prompting
                #model="llama3.1:latest",        # no hace bien el overwrite impuesto por las herramientas externas
                #model="qwen3:8b",              # hace bien el overwrite, considerando un promptint estricto
                #model="phi4:latest",
                #model="gpt-oss:20b",            # hace bien el overwrite, considerando el prompting
                validate_model_on_init=True,
                temperature=0,
            )

            # Create and run the agent
            agent = create_agent(model, tools)

            agent_response = await agent.ainvoke(
                {
                    "messages":
                    [
                        {"role": "system","content": "You are a helpful assistant that performs mathematical calculations based only in external tools provided. Do not make any calculation out of provided by external tools. Do not evaluate the results. Do not correct the results. Do not verify answer. Just give the answer provided by the external tools. Inform if the result was provided by the external tools or by you."},
                        {"role": "user", "content": "what's (3 + 5) x 12?"}
                    ]
                }
            )

            print ("=== Reasoning Steps ===")
            reasoning_steps = [{"reasoning": b.content} for b in agent_response["messages"] if b.type == 'ai']
            print(" ".join(step["reasoning"] for step in reasoning_steps))

            print ("\n=== Full Response ===")
            # Optionally, print or handle the response
            for message in agent_response["messages"]:
                print(type(message))  # Identifies the message type (e.g., <class 'langchain_core.messages.ai.AIMessage'>)
                print(message.content)  # Retrieves the content or result of the message

            #print("Agent response:", agent_response)

if __name__ == "__main__":
    asyncio.run(main())
