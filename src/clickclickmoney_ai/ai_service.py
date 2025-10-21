from typing import TypedDict, Annotated, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

# 우리가 직접 만든 Tool들을 가져옵니다.
from .tools import all_tools

# 1. 도구 실행기(ToolExecutor) 준비
tool_executor = ToolExecutor(all_tools)

# 2. 모델(LLM)에 도구(Tools) 정보 바인딩
model = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
model_with_tools = model.bind_tools(all_tools)

# 3. LangGraph 상태 정의
class GraphState(TypedDict):
    messages: Annotated[list, Any]

# 4. 그래프 노드(Node) 정의

def call_model(state: GraphState):
    """LLM을 호출하여 응답 또는 Tool 사용 결정을 받습니다."""
    print("NODE: LLM 호출")
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    # 이전 대화 기록에 새로운 응답을 추가하여 반환합니다.
    return {"messages": messages + [response]}

def call_tool(state: GraphState):
    """LLM이 사용하기로 결정한 Tool을 실제로 실행합니다."""
    print("NODE: Tool 실행")
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}

    tool_outputs = []
    for tool_call in last_message.tool_calls:
        tool_invocation = ToolInvocation(tool=tool_call["name"], tool_input=tool_call["args"])
        output = tool_executor.invoke(tool_invocation)
        tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call['id']))
    
    # 이전 대화 기록에 도구 실행 결과를 추가하여 반환합니다.
    return {"messages": state["messages"] + tool_outputs}

# 5. 라우팅 로직 (Conditional Edge)
def should_continue(state: GraphState) -> str:
    """LLM의 응답에 따라 다음 노드를 결정합니다."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    else:
        return "end"

# 6. 그래프 생성 및 컴파일
workflow = StateGraph(GraphState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END}
)

# Tool 실행 후, 결과를 다시 agent(LLM)에게 전달하여 최종 답변 생성
# action 노드가 메시지를 누적하므로, action -> agent 엣지는 그대로 둡니다.
workflow.add_edge("action", "agent")

app_graph = workflow.compile()