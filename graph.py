import json
import re
import asyncio
from typing import Annotated, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import TypedDict

# ─── State Definition ─────────────────────────────────────────────────────────

class PlannerExecutorState(TypedDict):
    """State for the Planner-Executor agent."""
    goal: str
    plan: list[dict]
    current_step: int
    results: list[dict]
    completed: bool

# ─── System Prompts ───────────────────────────────────────────────────────────

PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
Each step MUST follow this EXACT schema:
{"step": int, "description": str, "tool": str or null, "args": dict or null}

Available tools and their EXACT argument names:
- fetch_data_source(source: str) → source must be one of: sales, customers, expenses
- get_weather(city: str) → get real weather for a city
- search_web(query: str) → search the web for information

Use null for tool/args on synthesis or writing steps.
Return ONLY a valid JSON array. No markdown, no explanation."""

TOOL_ARG_MAP = {
    "fetch_data_source": "source",
    "get_weather": "city",
    "search_web": "query",
}

# ─── Helper Functions ─────────────────────────────────────────────────────────

def safe_args(tool_name: str, raw_args: dict) -> dict:
    """Remap hallucinated arg names to the correct parameter."""
    expected = TOOL_ARG_MAP.get(tool_name)
    if not expected or expected in raw_args:
        return raw_args
    first_val = next(iter(raw_args.values()), tool_name)
    print(f"  Remapped {raw_args} → {{'{expected}': '{first_val}'}}")
    return {expected: str(first_val)}

# ─── Graph Nodes ──────────────────────────────────────────────────────────────

# ─── Graph Nodes ──────────────────────────────────────────────────────────

def planner_node(state: PlannerExecutorState, llm, tools_map) -> PlannerExecutorState:
    """Generate a plan from the goal."""
    print(f"\n📋 PLANNER NODE")
    print(f"Goal: {state['goal']}\n")
    
    # Call LLM to generate plan
    plan_resp = llm.invoke([
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=state['goal'])
    ])
    
    # Extract and parse JSON
    raw_text = plan_resp.content if isinstance(plan_resp.content, str) else plan_resp.content[0].get("text", "")
    clean_json = re.sub(r"```json|```", "", raw_text).strip()
    plan = json.loads(clean_json)
    
    print(f"✅ Plan generated ({len(plan)} steps):")
    for s in plan:
        print(f"  Step {s['step']}: {s['description']} | tool={s.get('tool')}")
    print()
    
    return {
        **state,
        "plan": plan,
        "current_step": 0,
        "results": []
    }

def executor_node(state: PlannerExecutorState, llm, tools_map) -> PlannerExecutorState:
    """Execute one step at a time."""
    print(f"\n⚙️  EXECUTOR NODE")
    
    # Check if all steps are completed
    if state['current_step'] >= len(state['plan']):
        print("✅ All steps completed!\n")
        return {
            **state,
            "completed": True
        }
    
    step = state['plan'][state['current_step']]
    print(f"Executing Step {step['step']}: {step['description']}")
    
    tool_name = step.get("tool")
    
    if tool_name and tool_name in tools_map:
        # Tool step - handle async tools
        print(f"  Calling tool: {tool_name}")
        corrected_args = safe_args(tool_name, step.get("args") or {})
        
        # Check if tool has async support and call appropriately
        tool = tools_map[tool_name]
        try:
            # Try async invocation first
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(tool.ainvoke(corrected_args))
            loop.close()
            result_str = str(result)
        except (AttributeError, NotImplementedError, RuntimeError):
            # Fallback to sync invocation
            try:
                result = tool.invoke(corrected_args)
                result_str = str(result)
            except:
                # If both fail, use LLM synthesis
                print(f"  Tool failed, using LLM synthesis")
                response = llm.invoke([
                    HumanMessage(content=f"{step['description']}")
                ])
                result_str = response.content
    else:
        # Synthesis step - use LLM with prior results as context
        print(f"  Synthesis step (LLM)")
        context = "\n".join([f"Step {r['step']}: {r['result'][:200]}" for r in state['results']])
        response = llm.invoke([
            HumanMessage(content=f"{step['description']}\n\nContext:\n{context}")
        ])
        result_str = response.content
    
    print(f"  Result: {str(result_str)[:150]}\n")
    
    # Update state
    new_results = state['results'] + [{
        "step": step["step"],
        "description": step["description"],
        "result": result_str
    }]
    
    return {
        **state,
        "results": new_results,
        "current_step": state['current_step'] + 1,
        "completed": state['current_step'] + 1 >= len(state['plan'])
    }

def should_continue(state: PlannerExecutorState) -> str:
    """Routing function: continue executing or end."""
    if state['completed']:
        return "end"
    return "executor"

# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_graph(llm, tools_map):
    """Build the LangGraph workflow."""
    
    # Create graph
    workflow = StateGraph(PlannerExecutorState)
    
    # Create partially applied nodes (with llm and tools_map bound)
    def planner_node_bound(state):
        return planner_node(state, llm, tools_map)
    
    def executor_node_bound(state):
        return executor_node(state, llm, tools_map)
    
    # Add nodes
    workflow.add_node("planner", planner_node_bound)
    workflow.add_node("executor", executor_node_bound)
    
    # Add edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "executor": "executor",
            "end": END
        }
    )
    
    # Compile
    graph = workflow.compile()
    return graph
