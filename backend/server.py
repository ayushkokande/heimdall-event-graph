from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import networkx as nx
from agents import get_dynamic_script
from graph_rag import query_graph_rag
from openai import OpenAI
import os


OPENAI_MODEL = os.getenv("HEIMDALL_OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

# --- CORS ---------------------------------------------------------------------

origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic models ----------------------------------------------------------


class GraphNode(BaseModel):
    id: str
    group: str
    label: str
    color: str
    status: Optional[str] = None
    details: Optional[str] = None


class GraphLink(BaseModel):
    source: str
    target: str
    relation: str
    critical: bool = False


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    links: List[GraphLink]


class UploadGraphNode(BaseModel):
    id: str
    group: str
    label: Optional[str] = None
    color: Optional[str] = None
    status: Optional[str] = None
    details: Optional[str] = None


class UploadGraphLink(BaseModel):
    source: str
    target: str
    relation: Optional[str] = "CALLS"
    critical: bool = False


class UploadGraphRequest(BaseModel):
    nodes: List[UploadGraphNode]
    links: List[UploadGraphLink]


class TraceStep(BaseModel):
    step: int
    node_id: str
    title: str
    description: str


class TraceResponse(BaseModel):
    steps: List[TraceStep]


class IncidentStatus(BaseModel):
    id: str
    title: str
    severity: str
    status: str          # IDLE | STARTED | RESOLVED
    started_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    description: Optional[str] = None


class AgentRun(BaseModel):
    agent_name: str
    summary: str
    score: float
    steps: List[str]


class NodeInvestigation(BaseModel):
    node_id: str
    suspicion_score: float
    agent_results: List[AgentRun]


class InvestigateRequest(BaseModel):
    focus_node_id: str
    radius: int = 2


class InvestigateResponse(BaseModel):
    critical_path: List[str]
    nodes: List[NodeInvestigation]


class ExplainRequest(BaseModel):
    question: Optional[str] = None
    focus_node_id: Optional[str] = None
    radius: int = 2


class ExplainResponse(BaseModel):
    answer: str


# --- In-memory state: incidents, graphs, traces -------------------------------

INCIDENTS: Dict[str, Dict[str, Any]] = {}
GRAPHS: Dict[str, nx.DiGraph] = {}
TRACES: Dict[str, List[TraceStep]] = {}
CRITICAL_PATHS: Dict[str, List[str]] = {}


def _color_for_group(group: str) -> str:
    return {
        "service": "#3b82f6",      # blue
        "db": "#f97316",           # orange (Postgres, MySQL, etc.)
        "cache": "#22c55e",        # green (Redis / Memcached)
        "dns": "#eab308",          # yellow
        "edge": "#0ea5e9",         # cyan (Cloudflare / API gateway)
        "event": "#a855f7",        # purple (deploys, feature flags)
        "error": "#ef4444",        # red
        "alert": "#f97373",        # light red
        "agent": "#f9fafb",        # white satellites
        "orchestrator": "#06b6d4", # teal master node
    }.get(group, "#9ca3af")


# --- Demo graphs (used if user skips upload) ----------------------------------


def build_checkout_graph() -> nx.DiGraph:
    g = nx.DiGraph()

    g.add_node(
        "Alert:Checkout_5xx",
        group="alert",
        label="Checkout 5xx SLO breach",
        color=_color_for_group("alert"),
        details="PagerDuty: spike in 5xx on /checkout across regions.",
    )
    g.add_node(
        "Error:Deadlock",
        group="error",
        label="Orders DB deadlock",
        color=_color_for_group("error"),
        details="Deadlock on checkout transaction touching inventory + orders tables.",
    )
    g.add_node(
        "Event:Deploy_v2.1",
        group="event",
        label="Payment service deploy v2.1",
        color=_color_for_group("event"),
        details="Canary of payment-service v2.1 rolled to 100% at 12:03 UTC.",
    )

    g.add_node(
        "Svc:Payment",
        group="service",
        label="Payment API (k8s)",
        color=_color_for_group("service"),
        details="Kubernetes deployment handling /checkout payment flow.",
    )
    g.add_node(
        "Svc:Inventory",
        group="service",
        label="Inventory API (k8s)",
        color=_color_for_group("service"),
        details="Reads available stock and reserves items for orders.",
    )
    g.add_node(
        "DB:Postgres",
        group="db",
        label="Orders DB (Postgres Asgard)",
        color=_color_for_group("db"),
        details="Primary OLTP Postgres for orders, payments, and inventory reservations.",
    )

    g.add_edge("Svc:Payment", "Svc:Inventory", relation="CALLS")
    g.add_edge("Svc:Payment", "DB:Postgres", relation="WRITES")

    g.add_edge(
        "Alert:Checkout_5xx",
        "Error:Deadlock",
        relation="CAUSED_BY",
        critical=True,
    )
    g.add_edge(
        "Error:Deadlock",
        "Event:Deploy_v2.1",
        relation="CORRELATED_WITH",
        critical=True,
    )
    g.add_edge(
        "Event:Deploy_v2.1",
        "Svc:Payment",
        relation="DEPLOYED_ON",
        critical=True,
    )

    return g



def build_inventory_graph() -> nx.DiGraph:
    g = nx.DiGraph()

    g.add_node(
        "Alert:Inventory_Latency",
        group="alert",
        label="Inventory latency breach",
        color=_color_for_group("alert"),
        details="p95 latency on /inventory/* > 5s for 10 minutes.",
    )
    g.add_node(
        "Error:Cache_Stampede",
        group="error",
        label="Redis cache stampede",
        color=_color_for_group("error"),
        details="Large burst of cache misses causes thundering herd on DB.",
    )
    g.add_node(
        "Cache:Inventory",
        group="cache",
        label="Inventory cache (Redis)",
        color=_color_for_group("cache"),
        details="Redis cluster acting as read-through cache for hot inventory keys.",
    )
    g.add_node(
        "Svc:Inventory",
        group="service",
        label="Inventory API (k8s)",
        color=_color_for_group("service"),
        details="Kubernetes deployment serving inventory reads/writes.",
    )
    g.add_node(
        "DB:Postgres",
        group="db",
        label="Orders DB (Postgres Asgard)",
        color=_color_for_group("db"),
        details="Same Postgres instance used by checkout flow.",
    )

    g.add_edge("Svc:Inventory", "Cache:Inventory", relation="READS")
    g.add_edge("Cache:Inventory", "DB:Postgres", relation="FALLBACK_READS")

    g.add_edge(
        "Alert:Inventory_Latency",
        "Error:Cache_Stampede",
        relation="CAUSED_BY",
        critical=True,
    )
    g.add_edge(
        "Error:Cache_Stampede",
        "Cache:Inventory",
        relation="CORRELATED_WITH",
        critical=True,
    )
    g.add_edge(
        "Cache:Inventory",
        "Svc:Inventory",
        relation="BACKPRESSURE_ON",
        critical=True,
    )

    return g


def build_auth_dns_graph() -> nx.DiGraph:
    g = nx.DiGraph()

    g.add_node(
        "Alert:Login_5xx",
        group="alert",
        label="Global login 5xx",
        color=_color_for_group("alert"),
        details="Login API 5xx from multiple regions within 2 minutes.",
    )
    g.add_node(
        "Error:Auth_Unreachable",
        group="error",
        label="Auth upstream unreachable",
        color=_color_for_group("error"),
        details="Edge tier cannot resolve or connect to regional auth clusters.",
    )
    g.add_node(
        "Edge:Global_Auth",
        group="edge",
        label="Global Auth Edge (Cloudflare)",
        color=_color_for_group("edge"),
        details="Cloudflare-style edge workers fronting auth.<region>.example.com.",
    )
    g.add_node(
        "DNS:Auth_Zone",
        group="dns",
        label="Auth DNS zone (Route53)",
        color=_color_for_group("dns"),
        details="Authoritative DNS zone for auth.<region>.example.com; recent CNAME change.",
    )

    g.add_edge("Edge:Global_Auth", "DNS:Auth_Zone", relation="RESOLVES_VIA")

    g.add_edge(
        "Alert:Login_5xx",
        "Error:Auth_Unreachable",
        relation="CAUSED_BY",
        critical=True,
    )
    g.add_edge(
        "Error:Auth_Unreachable",
        "Edge:Global_Auth",
        relation="SEEN_AT",
        critical=True,
    )
    g.add_edge(
        "Edge:Global_Auth",
        "DNS:Auth_Zone",
        relation="MISCONFIGURED_RECORD",
        critical=True,
    )

    return g


def init_data() -> None:
    # incidents
    INCIDENTS["checkout-5xx"] = {
        "id": "checkout-5xx",
        "title": "Checkout 5xx spike (Payment v2.1)",
        "severity": "SEV-2",
        "status": "IDLE",
        "started_at": None,
        "resolved_at": None,
        "description": "Deadlock after a bad deploy of the payment service.",
    }
    INCIDENTS["inventory-cache"] = {
        "id": "inventory-cache",
        "title": "Inventory cache stampede",
        "severity": "SEV-2",
        "status": "IDLE",
        "started_at": None,
        "resolved_at": None,
        "description": "Cache stampede causes DB overload and latency.",
    }
    INCIDENTS["auth-dns"] = {
        "id": "auth-dns",
        "title": "Global login outage (DNS misconfig)",
        "severity": "SEV-1",
        "status": "IDLE",
        "started_at": None,
        "resolved_at": None,
        "description": "Bad DNS change makes auth unreachable from the edge.",
    }

    # graphs
    GRAPHS["checkout-5xx"] = build_checkout_graph()
    GRAPHS["inventory-cache"] = build_inventory_graph()
    GRAPHS["auth-dns"] = build_auth_dns_graph()

    # traces (these encode the intended critical path)
    TRACES["checkout-5xx"] = [
        TraceStep(
            step=1,
            node_id="Alert:Checkout_5xx",
            title="Pager triggers",
            description="Checkout 5xx above SLO for 5 minutes.",
        ),
        TraceStep(
            step=2,
            node_id="Error:Deadlock",
            title="Error identified",
            description="DB deadlock detected on checkout transactions.",
        ),
        TraceStep(
            step=3,
            node_id="Event:Deploy_v2.1",
            title="Suspicious deploy",
            description="Deploy v2.1 happened shortly before deadlocks started.",
        ),
        TraceStep(
            step=4,
            node_id="Svc:Payment",
            title="Root cause & fix",
            description="Rollback v2.1 and re-add @Transactional to checkout().",
        ),
    ]

    TRACES["inventory-cache"] = [
        TraceStep(
            step=1,
            node_id="Alert:Inventory_Latency",
            title="Pager triggers",
            description="Inventory p95 latency breached.",
        ),
        TraceStep(
            step=2,
            node_id="Error:Cache_Stampede",
            title="Cache stampede detected",
            description="Burst of cache misses overloads DB.",
        ),
        TraceStep(
            step=3,
            node_id="Cache:Inventory",
            title="Cache policy issue",
            description="Over-aggressive cache invalidation.",
        ),
        TraceStep(
            step=4,
            node_id="Svc:Inventory",
            title="Mitigation",
            description="Introduce request coalescing + safer invalidation.",
        ),
    ]

    TRACES["auth-dns"] = [
        TraceStep(
            step=1,
            node_id="Alert:Login_5xx",
            title="Pager triggers",
            description="Login 5xx from multiple regions.",
        ),
        TraceStep(
            step=2,
            node_id="Error:Auth_Unreachable",
            title="Auth unreachable",
            description="Edge cannot reach auth targets.",
        ),
        TraceStep(
            step=3,
            node_id="Edge:Global_Auth",
            title="Edge layer",
            description="Auth edge nodes failing to resolve upstream.",
        ),
        TraceStep(
            step=4,
            node_id="DNS:Auth_Zone",
            title="Root cause",
            description="Bad DNS change in auth zone. Rollback.",
        ),
    ]

    for inc_id, steps in TRACES.items():
        CRITICAL_PATHS[inc_id] = [s.node_id for s in steps]


init_data()

# --- Agent helpers ------------------------------------------------------------

@dataclass
class NodeContext:
    node_id: str
    node_type: str
    attrs: Dict[str, Any]
    neighbors: List[str]


@dataclass
class AgentResult:
    agent_name: str
    node_id: str
    summary: str
    score: float        # 0–1 suspicion
    steps: List[str]    # reasoning bullets


def blast_radius_nodes(graph: nx.DiGraph, center: str, radius: int = 2) -> set[str]:
    if center not in graph:
        return set()

    visited = {center}
    frontier = {center}

    for _ in range(radius):
        next_frontier = set()
        for node in frontier:
            neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        if not next_frontier:
            break
        frontier = next_frontier

    return visited


class BaseAgent:
    name = "base"

    def run(self, ctx: NodeContext) -> AgentResult:
        raise NotImplementedError

class DBSpecialistAgent(BaseAgent):
    name = "DB Specialist"

    def run(self, ctx: NodeContext) -> AgentResult:
        label = ctx.attrs.get("label", ctx.node_id)
        steps = get_dynamic_script("DB Specialist", label)

        has_error_neighbor = any(
            nb.startswith("Error:") or nb.startswith("Alert:") for nb in ctx.neighbors
        )

        if ctx.node_type == "db" and has_error_neighbor:
            score = 0.9
        elif ctx.node_type == "db":
            score = 0.6
        else:
            score = 0.2

        summary = steps[-1] if steps else f"DB Specialist inspected {label}."
        return AgentResult(
            agent_name=self.name,
            node_id=ctx.node_id,
            summary=summary,
            score=score,
            steps=steps,
        )


class K8sSpecialistAgent(BaseAgent):
    name = "K8s Specialist"

    def run(self, ctx: NodeContext) -> AgentResult:
        label = ctx.attrs.get("label", ctx.node_id)
        steps = get_dynamic_script("K8s Specialist", label)

        has_error_neighbor = any(
            nb.startswith("Error:") or nb.startswith("Alert:") for nb in ctx.neighbors
        )

        if ctx.node_type in ("service", "edge") and has_error_neighbor:
            score = 0.85
        elif ctx.node_type in ("service", "edge"):
            score = 0.5
        else:
            score = 0.15

        summary = steps[-1] if steps else f"K8s Specialist tailed logs for {label}."
        return AgentResult(
            agent_name=self.name,
            node_id=ctx.node_id,
            summary=summary,
            score=score,
            steps=steps,
        )


class CodeDetectiveAgent(BaseAgent):
    name = "Code Detective"

    def run(self, ctx: NodeContext) -> AgentResult:
        label = ctx.attrs.get("label", ctx.node_id)
        steps = get_dynamic_script("Code Detective", label)

        has_deploy_neighbor = any(nb.startswith("Event:Deploy") for nb in ctx.neighbors)

        if ctx.node_type == "service" and has_deploy_neighbor:
            score = 0.95
        elif ctx.node_type == "service":
            score = 0.7
        elif ctx.node_type == "event":
            score = 0.5
        else:
            score = 0.2

        summary = steps[-1] if steps else f"Code Detective reviewed recent changes in {label}."
        return AgentResult(
            agent_name=self.name,
            node_id=ctx.node_id,
            summary=summary,
            score=score,
            steps=steps,
        )




class LogsAgent(BaseAgent):
    name = "logs"

    def run(self, ctx: NodeContext) -> AgentResult:
        neighbours_str = ", ".join(ctx.neighbors) or "none"
        steps = [
            f"Inspecting logs around node {ctx.node_id} (type={ctx.node_type}).",
            f"Neighbouring nodes in blast radius: {neighbours_str}.",
        ]

        # Incident-specific flavours based on node_id
        if ctx.node_id == "Alert:Inventory_Latency":
            steps += [
                "Pulling last 10 minutes of /inventory/* request logs.",
                "Latency jumps from ~250ms to 5–6s starting at 15:47 UTC.",
                "Time-correlated with spike in Redis cache misses on Inventory cache.",
            ]
            summary = (
                "Inventory alert is backed by a sharp latency spike that lines up "
                "with Redis cache pressure. Strong signal this is symptom of a cache issue."
            )
            score = 0.75

        elif ctx.node_id == "Alert:Login_5xx":
            steps += [
                "Streaming edge logs from Global Auth Edge.",
                "Majority of 5xx responses have upstream error 'auth upstream unreachable'.",
                "Unreachable errors begin immediately after a DNS change in auth zone.",
            ]
            summary = (
                "Login 5xx are clearly driven by upstream unreachability at the edge, "
                "likely tied to a DNS or routing change."
            )
            score = 0.8

        else:
            # Generic behaviour
            has_error_neighbor = any(
                nb.startswith("Error:") or nb.startswith("Alert:")
                for nb in ctx.neighbors
            )
            if ctx.node_type == "service" and has_error_neighbor:
                steps.append("Service sits adjacent to error/alert nodes in the graph.")
                steps.append("Logs near incident time are likely noisy and relevant.")
                summary = (
                    f"Service {ctx.node_id} is close to errors/alerts; "
                    "treat its logs as likely on the critical path."
                )
                score = 0.7
            else:
                steps.append("No strong correlation with local errors/alerts.")
                summary = (
                    f"No strong log signal for {ctx.node_id} from this neighbourhood."
                )
                score = 0.25

        return AgentResult(
            agent_name=self.name,
            node_id=ctx.node_id,
            summary=summary,
            score=score,
            steps=steps,
        )

class DeployAgent(BaseAgent):
    name = "deploy"

    def run(self, ctx: NodeContext) -> AgentResult:
        has_deploy_neighbor = any(nb.startswith("Event:Deploy") for nb in ctx.neighbors)

        steps = [
            f"Checking CI/CD events around node {ctx.node_id}.",
            f"Neighbouring nodes: {', '.join(ctx.neighbors) or 'none'}.",
        ]

        if has_deploy_neighbor:
            steps += [
                "Found recent deploy event in immediate neighbourhood.",
                "Comparing deploy timestamp to when alerts first fired.",
                "Deploy time lines up exactly with onset of failures.",
            ]
            summary = (
                f"{ctx.node_id} is directly connected to a fresh deploy; "
                "this change is strongly suspected as the introducer."
            )
            score = 0.9
        else:
            summary = f"No deploys in the immediate neighbourhood of {ctx.node_id}."
            score = 0.15

        return AgentResult(
            agent_name=self.name,
            node_id=ctx.node_id,
            summary=summary,
            score=score,
            steps=steps,
        )


class DBAgent(BaseAgent):
    name = "db"

    def run(self, ctx: NodeContext) -> AgentResult:
        steps = [
            f"DBAgent attached to {ctx.node_id} (type={ctx.node_type}).",
            f"Neighbours: {', '.join(ctx.neighbors) or 'none'}.",
        ]

        if ctx.node_type != "db":
            steps.append("Node is not a database; DB-specific signal is low.")
            summary = f"{ctx.node_id} is not a DB node; treat as low priority for DB root cause."
            score = 0.1
            return AgentResult(self.name, ctx.node_id, summary, score, steps)

        has_error_neighbor = any(
            nb.startswith("Error:") or nb.startswith("Alert:")
            for nb in ctx.neighbors
        )
        service_neighbors = [nb for nb in ctx.neighbors if nb.startswith("Svc:")]

        if ctx.node_id == "DB:Postgres" and "Error:Deadlock" in ctx.neighbors:
            steps += [
                "Inspecting pg_stat_activity for blocked queries.",
                "Lock graph shows ExclusiveLock held on orders table by checkout transaction.",
                "Multiple inventory + payment transactions blocked behind the same lock.",
            ]
            summary = (
                "Orders DB is the bottleneck: a deadlock introduced by the new "
                "payment transaction holds locks that block downstream requests."
            )
            score = 0.85

        elif ctx.node_id == "DB:Postgres" and "Error:Cache_Stampede" in ctx.neighbors:
            steps += [
                "Reviewing DB CPU and QPS after cache stampede.",
                "Read QPS and CPU jump 4x as Redis starts missing.",
                "Slow queries originate from Inventory API read paths.",
            ]
            summary = (
                "Orders DB is overloaded as a consequence of Redis cache stampede; "
                "it is a victim, not the original root cause."
            )
            score = 0.7

        elif has_error_neighbor and service_neighbors:
            steps.append(
                "DB neighbours include services and errors/alerts but pattern "
                "does not match explicit deadlock or overload."
            )
            summary = (
                f"{ctx.node_id} is a DB heavily exercised by services "
                f"{', '.join(service_neighbors)} and close to errors; "
                "candidate for performance issues."
            )
            score = 0.6
        elif service_neighbors:
            summary = (
                f"{ctx.node_id} is used by services "
                f"{', '.join(service_neighbors)}, but no direct errors nearby."
            )
            score = 0.35
        else:
            summary = (
                f"{ctx.node_id} has no service or error neighbours in this radius; "
                "low suspicion."
            )
            score = 0.2

        return AgentResult(
            agent_name=self.name,
            node_id=ctx.node_id,
            summary=summary,
            score=score,
            steps=steps,
        )


class BlastRadiusOrchestrator:
    """
    This is the 'master' agent: it sees a local subgraph and combines
    all per-node, per-agent signals into a ranked critical path candidate.
    """

    def __init__(self, graph: nx.DiGraph, agents: List[BaseAgent]):
        self.graph = graph
        self.agents = agents

    def build_ctx(self, node_id: str, nodes_in_radius: set[str]) -> NodeContext:
        attrs = self.graph.nodes[node_id]
        neighbors = list(
            (set(self.graph.successors(node_id)) | set(self.graph.predecessors(node_id)))
            & nodes_in_radius
        )
        return NodeContext(
            node_id=node_id,
            node_type=attrs.get("group", "unknown"),
            attrs=attrs,
            neighbors=neighbors,
        )

    def run_for_focus(self, focus_node_id: str, radius: int = 2) -> List[AgentResult]:
        node_ids = blast_radius_nodes(self.graph, focus_node_id, radius)
        results: List[AgentResult] = []

        for nid in node_ids:
            ctx = self.build_ctx(nid, node_ids)
            for agent in self.agents:
                results.append(agent.run(ctx))

        return results


def aggregate_agent_results(incident_id: str, focus_node_id: str, radius: int = 2):
    graph = GRAPHS.get(incident_id)
    if not graph or focus_node_id not in graph:
        return [], []

    # Master orchestrator with your three “persona” agents
    orchestrator = BlastRadiusOrchestrator(
        graph,
        agents=[
            DBSpecialistAgent(),
            K8sSpecialistAgent(),
            CodeDetectiveAgent(),
        ],
    )

    raw_results = orchestrator.run_for_focus(focus_node_id, radius)

    per_node: Dict[str, List[AgentResult]] = {}
    for r in raw_results:
        per_node.setdefault(r.node_id, []).append(r)

    investigations: List[NodeInvestigation] = []
    for nid, results in per_node.items():
        suspicion = max(r.score for r in results)
        agent_runs = [
            AgentRun(
                agent_name=r.agent_name,
                summary=r.summary,
                score=r.score,
                steps=r.steps,
            )
            for r in results
        ]
        investigations.append(
            NodeInvestigation(
                node_id=nid,
                suspicion_score=suspicion,
                agent_results=agent_runs,
            )
        )

        # Attach combined logs to the graph node itself so RAG can use them
        logs_lines: List[str] = []
        for r in results:
            logs_lines.append(f"[{r.agent_name}] {r.summary}")
            for st in r.steps or []:
                logs_lines.append(f"  {st}")
        graph.nodes[nid]["logs"] = "\n".join(logs_lines)

    # Rank nodes by suspicion – this becomes the “critical path candidate”
    investigations.sort(key=lambda ni: ni.suspicion_score, reverse=True)
    critical_path = [ni.node_id for ni in investigations]

    return investigations, critical_path


# --- Graph to response (with visual agents + orchestrator) --------------------

def graph_to_response(incident_id: str) -> GraphResponse:
    graph = GRAPHS.get(incident_id)
    if not graph:
        raise HTTPException(status_code=404, detail="Unknown incident")

    nodes: List[GraphNode] = []
    links: List[GraphLink] = []

    # core system nodes
    for nid, data in graph.nodes(data=True):
        nodes.append(
            GraphNode(
                id=nid,
                group=data.get("group", "unknown"),
                label=data.get("label", nid),
                color=data.get("color", _color_for_group(data.get("group", "unknown"))),
                status=data.get("status"),
                details=data.get("details"),
            )
        )

    for src, dst, data in graph.edges(data=True):
        links.append(
            GraphLink(
                source=src,
                target=dst,
                relation=data.get("relation", "CALLS"),
                critical=bool(data.get("critical", False)),
            )
        )

    # master orchestrator node
    orch_id = f"orch:{incident_id}"
    nodes.append(
        GraphNode(
            id=orch_id,
            group="orchestrator",
            label="Incident orchestrator",
            color=_color_for_group("orchestrator"),
            status=None,
            details=(
                "Master agent that receives reports from localized node agents "
                "and decides the cross-service critical path."
            ),
        )
    )

    # visual satellites: one tiny white agent node per system node, wired to orchestrator
    for nid, data in graph.nodes(data=True):
        agent_id = f"agent:{incident_id}:{nid}"
        nodes.append(
            GraphNode(
                id=agent_id,
                group="agent",
                label="",
                color=_color_for_group("agent"),
                status=None,
                details=f"Localized agent attached to {data.get('label', nid)}.",
            )
        )
        links.append(
            GraphLink(
                source=nid,
                target=agent_id,
                relation="LOCAL_AGENT",
                critical=False,
            )
        )
        links.append(
            GraphLink(
                source=agent_id,
                target=orch_id,
                relation="REPORTS_TO",
                critical=False,
            )
        )

    return GraphResponse(nodes=nodes, links=links)


# --- Routes -------------------------------------------------------------------


@app.get("/")
def root():
    return {"message": "Heimdall backend running"}


@app.get("/incidents", response_model=List[IncidentStatus])
def list_incidents():
    return [IncidentStatus(**meta) for meta in INCIDENTS.values()]


@app.get("/incident/{incident_id}", response_model=IncidentStatus)
def get_incident(incident_id: str):
    meta = INCIDENTS.get(incident_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown incident")
    return IncidentStatus(**meta)


@app.post("/incident/{incident_id}/introduce_error", response_model=IncidentStatus)
def introduce_error(incident_id: str):
    meta = INCIDENTS.get(incident_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown incident")
    meta["status"] = "STARTED"
    meta["started_at"] = datetime.utcnow()
    meta["resolved_at"] = None
    return IncidentStatus(**meta)


@app.post("/incident/{incident_id}/resolve", response_model=IncidentStatus)
def resolve_incident(incident_id: str):
    meta = INCIDENTS.get(incident_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown incident")
    meta["status"] = "RESOLVED"
    meta["resolved_at"] = datetime.utcnow()
    return IncidentStatus(**meta)


@app.get("/incident/{incident_id}/graph", response_model=GraphResponse)
def get_graph_for_incident(incident_id: str):
    return graph_to_response(incident_id)


@app.post("/incident/{incident_id}/graph/upload")
def upload_graph(incident_id: str, payload: UploadGraphRequest):
    """
    Replace the graph for this incident with an uploaded dependency graph.

    The JSON is the 'dependency relationship diagram' of your system:
      {
        "nodes": [ { "id": "...", "group": "service|db|cache|dns|edge|event|error|alert", ... } ],
        "links": [ { "source": "...", "target": "...", "relation": "CALLS" } ]
      }

    After upload, Heimdall automatically attaches localized agents and an orchestrator.
    """
    g = nx.DiGraph()

    for n in payload.nodes:
        label = n.label or n.id
        color = n.color or _color_for_group(n.group)
        g.add_node(
            n.id,
            group=n.group,
            label=label,
            color=color,
            status=n.status,
            details=n.details,
        )

    for e in payload.links:
        g.add_edge(
            e.source,
            e.target,
            relation=e.relation or "CALLS",
            critical=bool(e.critical),
        )

    GRAPHS[incident_id] = g
    return {
        "status": "ok",
        "node_count": g.number_of_nodes(),
        "edge_count": g.number_of_edges(),
    }


@app.get("/incident/{incident_id}/trace", response_model=TraceResponse)
def get_trace(incident_id: str):
    steps = TRACES.get(incident_id)
    if not steps:
        raise HTTPException(status_code=404, detail="Unknown incident")
    return TraceResponse(steps=steps)


@app.post(
    "/incident/{incident_id}/investigate",
    response_model=InvestigateResponse,
)
def investigate(incident_id: str, req: InvestigateRequest):
    investigations, critical_path = aggregate_agent_results(
        incident_id, req.focus_node_id, req.radius
    )
    return InvestigateResponse(critical_path=critical_path, nodes=investigations)

@app.post(
    "/incident/{incident_id}/explain",
    response_model=ExplainResponse,
)
def explain(incident_id: str, req: ExplainRequest):
    trace = TRACES.get(incident_id, [])
    meta = INCIDENTS.get(incident_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown incident")

    # Determine focus node
    focus = req.focus_node_id
    if not focus and trace:
        focus = trace[0].node_id

    # Blast radius
    radius = max(1, req.radius or 2)
    investigations, critical_path = (
        aggregate_agent_results(incident_id, focus, radius)
        if focus else ([], [])
    )

    # ------------------------------------------------------------
    # Build investigation text for LLM
    # ------------------------------------------------------------
    investigation_text = ""
    for ni in investigations:
        investigation_text += (
            f"\nNode: {ni.node_id}  (Suspicion: {ni.suspicion_score:.2f})\n"
        )
        for ar in ni.agent_results:
            investigation_text += f"  Agent: {ar.agent_name}\n"
            investigation_text += f"    Summary: {ar.summary}\n"
            for step in ar.steps:
                investigation_text += f"    - {step}\n"

    # Build trace text
    trace_text = "\n".join(
        f"Step {s.step}: {s.title} — {s.description}" for s in trace
    )

    # Question
    user_question = req.question.strip() if req.question else None
    if not user_question:
        user_question = "Give me a full root-cause analysis and reasoning."

    # ------------------------------------------------------------
    # Construct LLM prompt
    # ------------------------------------------------------------
    prompt = f"""
You are Heimdall, an expert SRE AI.
Your job is to analyze incident graphs, blast-radius investigations, agent logs, and suspicious nodes.

INCIDENT:
ID: {meta['id']}
Title: {meta['title']}
Severity: {meta['severity']}

====================================
TRACE TIMELINE
====================================
{trace_text}

====================================
CRITICAL PATH (Most Suspicious Nodes)
====================================
{chr(10).join(critical_path)}

====================================
BLAST-RADIUS AGENT INVESTIGATIONS
====================================
{investigation_text}

====================================
USER QUESTION
====================================
{user_question}

Respond with:
- concise but technical root cause
- step-by-step reasoning
- why agents flagged certain nodes
- how the issue propagated through dependencies
- actionable remediation
"""

    # ------------------------------------------------------------
    # OpenAI call
    # ------------------------------------------------------------
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are Heimdall, an SRE expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        answer = resp.choices[0].message.content

        return ExplainResponse(answer=answer)

    except Exception as e:
        print("LLM error:", e)
        return ExplainResponse(
            answer="LLM explanation failed. Check backend logs."
        )

