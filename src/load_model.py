from dotenv import load_dotenv
import os
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pprint import pprint

load_dotenv() 

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


chatLLM = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.5-flash",
)

agent = create_agent(
    model=chatLLM
)

if __name__ == "__main__":
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "你是谁？"}]
    # response = chatLLM.invoke(messages)
    # pprint(response)
    from langchain.messages import AIMessage, HumanMessage

    for chunk in agent.stream({
        "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
    }, stream_mode="values"):
        # Each chunk contains the full state at that point
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            if isinstance(latest_message, HumanMessage):
                print(f"User: {latest_message.content}")
            elif isinstance(latest_message, AIMessage):
                print(f"Agent: {latest_message.content}")
        elif latest_message.tool_calls:
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")