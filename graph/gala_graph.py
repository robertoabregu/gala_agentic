from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from agents.state import AgentState
from agents.contextualizer import contextualizer_node
from agents.router import router_node
from agents.chitchat import chitchat_node
from agents.query_rewriter import query_rewriter_node
from agents.retriever_node import retriever_node
from agents.answer import answer_node
from agents.bcra_agent import bcra_agent_node
from agents.benefits import benefits_node
from agents.branch_locator_tool import branch_locator_node
from agents.credit_card_statement import credit_card_statement_node
from agents.guardrail import guardrail_node
from agents.save_memory import save_memory_node


def build_graph(client, retriever, top_k, score_threshold, chat_model):
    graph = StateGraph(AgentState)

    llm = ChatOpenAI(
        model=chat_model,
        temperature=0,
    )

    def router_wrapper(state: AgentState):
        return router_node(
            state=state,
            llm=llm,
        )

    def contextualizer_wrapper(state: AgentState):
        return contextualizer_node(
            state=state,
            llm=llm,
        )

    def query_rewriter_wrapper(state: AgentState):
        return query_rewriter_node(
            state=state,
            llm=llm,
        )

    def retriever_wrapper(state: AgentState):
        return retriever_node(
            state=state,
            client=client,
            retriever=retriever,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    def answer_wrapper(state: AgentState):
        return answer_node(
            state=state,
            llm=llm,
        )

    def chitchat_wrapper(state: AgentState):
        return chitchat_node(state=state)

    def bcra_agent_wrapper(state: AgentState):
        return bcra_agent_node(
            state=state,
            llm=llm,
        )

    def branch_locator_wrapper(state: AgentState):
        return branch_locator_node(state=state)

    def benefits_wrapper(state: AgentState):
        return benefits_node(
            state=state,
            llm=llm,
        )

    def credit_card_statement_wrapper(state: AgentState):
        return credit_card_statement_node(
            state=state,
            llm=llm,
        )

    graph.add_node("contextualizer", contextualizer_wrapper)
    graph.add_node("router", router_wrapper)
    graph.add_node("chitchat_answer", chitchat_wrapper)
    graph.add_node("query_rewriter", query_rewriter_wrapper)
    graph.add_node("retriever", retriever_wrapper)
    graph.add_node("answer", answer_wrapper)
    graph.add_node("bcra_agent", bcra_agent_wrapper)
    graph.add_node("benefits", benefits_wrapper)
    graph.add_node("branch_locator", branch_locator_wrapper)
    graph.add_node("credit_card_statement", credit_card_statement_wrapper)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("contextualizer")
    graph.add_edge("contextualizer", "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "chitchat": "chitchat_answer",
            "rag": "query_rewriter",
            "loans_rag": "query_rewriter",
            "bcra_credit_status": "bcra_agent",
            "benefits": "benefits",
            "branch_locator": "branch_locator",
            "credit_card_statement": "credit_card_statement",
            "fallback": "guardrail",
            "sensitive": "guardrail",
        },
    )

    graph.add_edge("chitchat_answer", "guardrail")
    graph.add_edge("query_rewriter", "retriever")
    graph.add_edge("retriever", "answer")
    graph.add_edge("answer", "guardrail")
    graph.add_edge("bcra_agent", "guardrail")
    graph.add_edge("benefits", "guardrail")
    graph.add_edge("branch_locator", "guardrail")
    graph.add_edge("credit_card_statement", "guardrail")
    graph.add_edge("guardrail", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()
