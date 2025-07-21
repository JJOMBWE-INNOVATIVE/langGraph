from typing import Dict, TypedDict, List, Union
from langgraph.graph import StateGraph,START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatOpenAI(model="gpt-4o")


def process(state: AgentState) -> AgentState:
    """This node will solve the  request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content)) # extracts only the important content in the message
    print(f"\nAI: {response.content}")
    print("CURRENT STATE:", state["messages"])
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter something: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})

    # Print the AI's response
    #print(result["messages"])
    conversation_history = result["messages"]

    # Update the conversation history with the AI's response
    user_input = input("Enter something: ")


with open("conversation_history.txt", "w") as file:
    for message in conversation_history:
       if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
       elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of conversation.\n")

print("Conversation history saved to conversation_history.txt.")
           
  


