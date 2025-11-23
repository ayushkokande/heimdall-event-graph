# We keep the metadata static, but the LOGS will now be dynamic
def get_agent_meta(role):
    if role == "DB Specialist": return {"color": "#ff9800", "id_prefix": "DB"}
    if role == "K8s Specialist": return {"color": "#00bcd4", "id_prefix": "K8s"}
    if role == "Code Detective": return {"color": "#9c27b0", "id_prefix": "Code"}
    return {"color": "#ffffff", "id_prefix": "Res"}

def get_dynamic_script(role, node_label):
    """
    Instead of returning a static list, we generate a 'script' of 5-6 steps
    tailored to the specific node label.
    """

    # We simulate a "context" for the node
    node_context = {
        "label": node_label,
        "status": "CRITICAL",
        "metrics": "Latency > 5000ms" if "Alert" in node_label else "Deadlock Detected"
    }

    # Generate a sequence of thoughts
    # In a real app, you would generate these one by one in the async loop.
    # For the demo speed, we can pre-generate a few, or use a mix.

    # Option A: Fully AI Generated (Slower but impressive)
    # logs = []
    # history = ""
    # for _ in range(4):
    #    line = generate_agent_log(role, node_context, history)
    #    logs.append(line)
    #    history += line + "\n"
    # return logs

    # Option B: Hybrid (Fast & Reliable for Demo)
    # We use templates but inject the Real Node Name
    if role == "DB Specialist":
        return [
            f"🔎 [DB_AGENT] Connecting to {node_label}...",
            f"   -> Checking active query pool for {node_label}...",
            "⚠️ [DB_AGENT] WARN: ExclusiveLock found on PID 992.",
            "   -> Correlating transaction 4a with recent commits...",
            "✅ [DB_AGENT] FINDING: Deadlock confirmed. Recommendation: Kill PID."
        ]

    if role == "K8s Specialist":
        return [
            f"☸️ [K8S_AGENT] Authenticating with Cluster for {node_label}...",
            "   -> Context switched. tailing logs...",
            "   -> Grepping for 'Exception'...",
            f"❌ [K8S_AGENT] Found StackTrace in {node_label}.",
            "✅ [K8S_AGENT] FINDING: App crashed due to unhandled SQL exception."
        ]

    if role == "Code Detective":
        return [
            f"💻 [CODE_AGENT] Blaming git history for {node_label}...",
            "   -> Checkout commit 8a2b3c (HEAD).",
            "   -> Analyzing diff...",
            "   -> Detected removal of '@Transactional' annotation.",
            "✅ [CODE_AGENT] FINDING: Non-atomic DB write introduced in v2.1."
        ]

    return [f"🔎 [RESEARCHER] Scanning {node_label}..."]