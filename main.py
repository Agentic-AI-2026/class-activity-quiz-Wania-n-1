import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import nest_asyncio

# Load environment variables
load_dotenv()
nest_asyncio.apply()

# Add Tools directory to path for MCP server access
TOOLS_DIR = Path(__file__).parent / "Tools"
sys.path.insert(0, str(TOOLS_DIR))

from graph import build_graph, PlannerExecutorState
from langchain_mcp_adapters.client import MultiServerMCPClient

# ─── LLM Configuration ─────────────────────────────────────────────────────────

def get_llm():
    """Initialize LLM based on .env configuration."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model = os.getenv("OLLAMA_MODEL", "llama2")
        print(f"🤖 Using Ollama ({model})")
        return ChatOllama(model=model, temperature=0)
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")
        print(f"🤖 Using Anthropic Claude")
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, api_key=api_key)
    
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env")
        print(f"🤖 Using Google Gemini")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, api_key=api_key)
    
    elif provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        print(f"🤖 Using Groq (llama-3.1-8b-instant)")
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

# ─── MCP Tools Setup ──────────────────────────────────────────────────────────

# ─── MCP Tools Setup (Optional) ───────────────────────────────────

async def get_mcp_tools(servers: list) -> tuple:
    """Load tools from MCP servers (optional - fails gracefully)."""
    try:
        mcp = MultiServerMCPClient({
            "data": {
                "command": sys.executable,
                "args": [str(TOOLS_DIR / "search_server.py")],
                "transport": "stdio",
            },
            "weather": {
                "command": sys.executable,
                "args": [str(TOOLS_DIR / "weather_server.py")],
                "transport": "stdio",
            },
        })
        
        tools = []
        print("📡 Attempting to connect to MCP servers...")
        for server in servers:
            try:
                server_tools = await asyncio.wait_for(mcp.get_tools(server_name=server), timeout=5)
                tools.extend(server_tools)
                print(f"  ✓ {server}")
            except asyncio.TimeoutError:
                print(f"  ⚠️  {server}: timeout (running agent without this tool)")
            except Exception as e:
                print(f"  ⚠️  {server}: {str(e)[:80]}")
        
        tools_map = {t.name: t for t in tools}
        if tools_map:
            print(f"✅ MCP tools loaded: {list(tools_map.keys())}\n")
        else:
            print("⚠️  No MCP tools available (agent will use LLM synthesis)\n")
        return tools, tools_map
    
    except Exception as e:
        print(f"⚠️  MCP connection failed: {str(e)[:100]}")
        print("⚠️  Running agent with LLM synthesis only\n")
        return [], {}

# ─── Main Agent Function ──────────────────────────────────────────────────────

async def run_planner_executor_agent(goal: str):
    """Run the LangGraph Planner-Executor agent."""
    print("=" * 70)
    print("🚀 PLANNER-EXECUTOR AGENT (LangGraph)")
    print("=" * 70)
    
    # Initialize LLM
    llm = get_llm()
    
    # Load MCP tools
    tools, tools_map = await get_mcp_tools(["data", "weather"])
    
    # Build and compile the graph
    print("\n📊 Building LangGraph workflow...")
    graph = build_graph(llm, tools_map)
    print("✅ Graph built!\n")
    
    # Run the agent
    initial_state = {
        "goal": goal,
        "plan": [],
        "current_step": 0,
        "results": [],
        "completed": False
    }
    
    print("▶️  Running agent...\n")
    final_state = graph.invoke(initial_state)
    
    # Print final results
    print("\n" + "=" * 70)
    print("📋 FINAL RESULTS")
    print("=" * 70)
    print(f"\nGoal: {final_state['goal']}\n")
    
    for result in final_state['results']:
        print(f"Step {result['step']}: {result['description']}")
        print(f"  Result: {result['result'][:300]}...\n")
    
    return final_state

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with the sample goal
    goal = "Fetch Q3 sales data and summarize it."
    
    # Run the agent
    final_state = asyncio.run(run_planner_executor_agent(goal))

