import asyncio
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel
from openai import OpenAI, AsyncOpenAI
from agents.tracing import set_tracing_disabled

set_tracing_disabled(True)

model = OpenAIChatCompletionsModel(
    model="minimax-m2.7:cloud",
    openai_client=AsyncOpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    ),
)

history_agent = Agent(
    model=model,
    name="History Tutor",
    instructions="you answer history questions clearly and concisely"
)

async def main():
    query= "who built badshai masjid in lahore pakistan"
    result = await Runner.run(history_agent, query)
    print(result)

asyncio.run(main())