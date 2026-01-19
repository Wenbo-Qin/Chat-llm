from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
import asyncio

from dotenv import load_dotenv

load_dotenv()

async def llm_agent_demo():
    # Create an MCP client that connects to both calculator and weather servers
    client = MultiServerMCPClient(
        {
            "calculator_service": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp_tool/calculate_server.py"],
            },
            "weather_service": {
                "transport": "stdio",
                "command": "python",
                "args": ["mcp_tool/weather_server.py"],
            }
        }
    )

    # Get the tools from the server
    tools = await client.get_tools()
    print("Available tools:", [tool.name for tool in tools])

    # Create an agent using the create_agent function
    agent = create_agent(
        "deepseek-chat", 
        tools,
        system_prompt=SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are an AI assistant tasked with math calculations work and answer questions. Your answers must be accurate and concise.If you don't find matched tool, you can answer it directly.",
                }
            ]
        )
    )
    # Example queries to test the agent
    queries = [
        "Calculate 15 plus 27 for me",
        "Can you multiple 123 and 3?",
        "How about 6 divide by 2?",
        "What's the current weather in Beijing?",
    ]

    for query_num, query in enumerate(queries):
        print(f"\n--- Query {query_num+1}: {query} ---")
        try:
            result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
            
            # Extract the AI's response from the result
            ai_message = None
            if 'messages' in result:
                # Find the last AI message in the conversation
                for msg in reversed(result['messages']):
                    if isinstance(msg, AIMessage):
                        ai_message = msg
                        break
            
            if ai_message:
                print(f"AI Response: {ai_message.content}")
            else:
                print("Could not find AI response in the result")

        except Exception as e:
            print(f"Error processing query '{query}': {e}")


if __name__ == "__main__":
    asyncio.run(llm_agent_demo())