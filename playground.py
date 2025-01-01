from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
phi.api = os.getenv("PHI_API_KEY")

# web search agent
websearch_agent = Agent(
    name="Web Search Agent",
    role="Search the web for given query.",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview", api_key=groq_api_key),
    tools=[DuckDuckGo(fixed_max_results=5, news=True)],
    instructions=["You are a professional financial web search agent.", "Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# financial agent
financial_agent = Agent(
    name="Financial Research Agent",
    role="Research financial trends",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview", api_key=groq_api_key),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents=[financial_agent, websearch_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)