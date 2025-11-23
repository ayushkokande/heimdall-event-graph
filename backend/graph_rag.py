# graph_rag.py
import os
import networkx as nx
from dotenv import load_dotenv

# --- OpenAI setup ------------------------------------------------------------

AI_AVAILABLE = False
client = None

try:
    from openai import OpenAI  # pip install openai

    load_dotenv()
    # Will read OPENAI_API_KEY from env
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    AI_AVAILABLE = True
except Exception:
    # No SDK / no key – fall back to heuristic answers
    client = None
    AI_AVAILABLE = False


def retrieve_subgraph_context(G: nx.DiGraph, query: str) -> str:
    """
    THE RETRIEVAL STEP:
    Instead of dumping the whole system, we find nodes relevant to the query
    and their immediate neighbors (1-hop).
    """
    relevant_nodes = set()

    # 1. Keyword Search (Simple Heuristic)
    query_lower = query.lower()
    for node, attrs in G.nodes(data=True):
        label = attrs.get("label", "").lower()
        if any(kw in label for kw in query_lower.split()):
            relevant_nodes.add(node)

    # 2. Always include any nodes marked as critical / hot
    for node, attrs in G.nodes(data=True):
        if attrs.get("status") == "critical" or attrs.get("color") == "#ff0044":
            relevant_nodes.add(node)

    # 3. Expand Context: 1-hop neighbours
    final_context_nodes = set(relevant_nodes)
    for node in relevant_nodes:
        if node in G:
            final_context_nodes.update(G.neighbors(node))
            final_context_nodes.update(G.predecessors(node))

    if not final_context_nodes:
        return "No specific system components found for this query."

    # 4. Serialize to text (topology + any logs attached to nodes)
    lines = []
    lines.append("RELEVANT SYSTEM TOPOLOGY & LOGS:\n")

    for node in final_context_nodes:
        attrs = G.nodes[node]
        label = attrs.get("label", node)
        group = attrs.get("group", "unknown")
        lines.append(f"- Node: {label} (Id: {node}, Type: {group})")

        # Attach agent logs if we have them
        if "logs" in attrs and attrs["logs"]:
            lines.append("  Agent logs:")
            logs_text = attrs["logs"]
            # logs_text is a single string with newlines, we indent it slightly
            for ln in str(logs_text).splitlines():
                lines.append(f"    {ln}")

        # Relationships inside this subgraph
        for neighbor in G.neighbors(node):
            if neighbor in final_context_nodes:
                relation = G.edges[node, neighbor].get("relation", "LINKS_TO")
                neighbor_label = G.nodes[neighbor].get("label", neighbor)
                lines.append(f"  -> [{relation}] -> {neighbor_label}")

        lines.append("")

    return "\n".join(lines)


def _heuristic_fallback(user_query: str, context: str) -> str:
    """
    Old deterministic behaviour, used when OpenAI is unavailable.
    """
    query_lower = user_query.lower()

    if "root cause" in query_lower or "why" in query_lower:
        return (
            "Based on the graph topology, the root cause is **Deployment v2.1**.\n\n"
            "Reasoning Trace:\n"
            "1. **Evt:Deploy** (Commit 8a2b) modified `InventoryService`.\n"
            "2. This immediately caused a **Deadlock** in `DB:Postgres`.\n"
            "3. The DB lock caused the **High Latency Alert** in `PaymentService`."
        )

    if "status" in query_lower or "health" in query_lower:
        return (
            "System is currently **DEGRADED**. Critical nodes: "
            "Alert:Latency, Err:Deadlock, DB:Postgres."
        )

    return f"I analyzed the graph. Relevant nodes:\n{context}"


def _fallback_answer(user_query: str, context: str) -> str:
    """
    Deterministic fallback used when no OpenAI key is available
    or the API call fails.
    """
    q = user_query.lower()

    if "root cause" in q or "why" in q:
        return (
            "Based on the graph topology, the likely root cause lies at the tail "
            "of the critical path (the last node where errors converge).\n\n"
            "Heuristic reasoning:\n"
            "1. Alerts and errors fan into a smaller set of service / DB nodes.\n"
            "2. Those nodes are where localized agents report the strongest signals "
            "(deploy proximity, DB contention, or unhandled exceptions).\n"
            "3. The node at the end of that chain is the best candidate to fix first.\n\n"
            "Context used:\n"
            f"{context}"
        )

    if "status" in q or "health" in q:
        return (
            "System health is DEGRADED in the region of the incident. "
            "Critical nodes show alerts/errors and downstream services inherit the impact.\n\n"
            "Context used:\n"
            f"{context}"
        )

    return f"I analyzed the graph with a heuristic. Context:\n{context}"


def query_graph_rag(G: nx.DiGraph, user_query: str) -> str:
    """
    Combine the user's question + retrieved graph context and send to OpenAI.
    Falls back to deterministic logic if OpenAI isn't available.
    """
    context = retrieve_subgraph_context(G, user_query)

    question = (
            user_query.strip()
            or "Explain what went wrong in this incident, the causal chain, and the most likely root cause and fix."
    )

    if AI_AVAILABLE and _client is not None:
        try:
            model_name = os.getenv("HEIMDALL_OPENAI_MODEL", "gpt-4.1-mini")

            resp = _client.chat.completions.create(
                model=model_name,
                temperature=0.3,
                max_tokens=700,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Heimdall, an expert Site Reliability Engineer. "
                            "You receive a subgraph of the system topology and some logs. "
                            "Explain incidents clearly, focusing on causality and root cause. "
                            "Do not invent services that aren't in the context."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}",
                    },
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:
            # fall through to heuristic answer if OpenAI has issues
            return f"(OpenAI error: {e})\n\n" + _heuristic_fallback(question, context)

    # No OpenAI client installed – keep deterministic behaviour
    return _heuristic_fallback(question, context)

