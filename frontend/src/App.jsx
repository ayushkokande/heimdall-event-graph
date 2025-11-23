// src/App.jsx
import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import ForceGraph2D from "react-force-graph-2d";
import "./App.css";

const API_BASE = "http://localhost:8000";

function App() {
  const fgRef = useRef(null);

  const [showInitialUpload, setShowInitialUpload] = useState(true);
  const [incidents, setIncidents] = useState([]);
  const [activeIncidentId, setActiveIncidentId] = useState(null);
  const [incident, setIncident] = useState(null);

  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [trace, setTrace] = useState([]);
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const [selectedNodeId, setSelectedNodeId] = useState(null);

  const [investigation, setInvestigation] = useState(null);
  const [isInvestigating, setIsInvestigating] = useState(false);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [isExplaining, setIsExplaining] = useState(false);

  // dynamic, per-incident critical-path reveal
  const [revealedNodes, setRevealedNodes] = useState([]); // node ids
  const [revealedEdges, setRevealedEdges] = useState([]); // "src::dst" keys
   const [logStepIndex, setLogStepIndex] = useState(0);

  const debugLabel = "Heimdall UI loaded";

  const currentStep = trace[currentStepIdx] || null;
  const currentNodeIdFromTrace = currentStep?.node_id || null;
  const highlightNodeId = selectedNodeId || currentNodeIdFromTrace || null;


  const revealedNodeSet = useMemo(
    () => new Set(revealedNodes),
    [revealedNodes]
  );
  const revealedEdgeSet = useMemo(
    () => new Set(revealedEdges),
    [revealedEdges]
  );

    const activeNodeDetails = useMemo(() => {
      if (!graphData.nodes.length) return null;
      const id = selectedNodeId || currentNodeIdFromTrace;
      if (!id) return null;
      return graphData.nodes.find((n) => n.id === id) || null;
    }, [graphData.nodes, selectedNodeId, currentNodeIdFromTrace]);

     useEffect(() => {
        setLogStepIndex(0);
      }, [activeNodeDetails?.id]);

  // --- initial: load list of incidents --------------------------------------

  useEffect(() => {
    const loadIncidents = async () => {
      const res = await fetch(`${API_BASE}/incidents`);
      const json = await res.json();
      setIncidents(json || []);
      if (!activeIncidentId && json && json.length > 0) {
        setActiveIncidentId(json[0].id);
      }
    };
    loadIncidents().catch(console.error);
  }, [activeIncidentId]);

  // --- load per-incident data whenever activeIncidentId changes -------------

  useEffect(() => {
    if (!activeIncidentId) return;

    const fetchAll = async () => {
      const [gRes, tRes, iRes] = await Promise.all([
        fetch(`${API_BASE}/incident/${activeIncidentId}/graph`),
        fetch(`${API_BASE}/incident/${activeIncidentId}/trace`),
        fetch(`${API_BASE}/incident/${activeIncidentId}`),
      ]);
      const gJson = await gRes.json();
      const tJson = await tRes.json();
      const iJson = await iRes.json();

      setGraphData({
        nodes: gJson.nodes,
        links: gJson.links,
      });
      setTrace(tJson.steps || []);
      setIncident(iJson);
      setCurrentStepIdx(0);
      setIsPlaying(false);
      setSelectedNodeId(null);
      setInvestigation(null);
      setAnswer("");
      setRevealedNodes([]);
      setRevealedEdges([]);
            setLogStepIndex(0);
    };

    fetchAll().catch(console.error);
  }, [activeIncidentId]);

  // --- helpers ---------------------------------------------------------------

  const formatTime = (iso) => {
    if (!iso) return "";
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  // --- introduce error / resolve --------------------------------------------

  const introduceError = async () => {
    if (!activeIncidentId) return;
    if (!trace.length) return;

    try {
      const res = await fetch(
        `${API_BASE}/incident/${activeIncidentId}/introduce_error`,
        { method: "POST" }
      );
      const json = await res.json();
      setIncident(json);

      // start at first step, highlight first node only
      setCurrentStepIdx(0);
      setRevealedNodes([trace[0].node_id]);
      setRevealedEdges([]);
      setIsPlaying(true);
    } catch (e) {
      console.error("introduce_error error", e);
    }
  };

  const resolveIncident = async () => {
    if (!activeIncidentId) return;
    try {
      const res = await fetch(
        `${API_BASE}/incident/${activeIncidentId}/resolve`,
        { method: "POST" }
      );
      const json = await res.json();
      setIncident(json);
      setIsPlaying(false);
    } catch (e) {
      console.error("resolve error", e);
    }
  };

  // --- autoplay trace (and progressively reveal critical path) --------------

  useEffect(() => {
    if (!isPlaying || !trace.length) return;

    const atLast = currentStepIdx >= trace.length - 1;
    if (atLast) {
      setIsPlaying(false);
      return;
    }

    const id = setTimeout(() => {
      const nextIdx = currentStepIdx + 1;
      const pathNodes = trace.slice(0, nextIdx + 1).map((s) => s.node_id);

      // compute edges along this path
      const newEdges = [];
      for (let i = 0; i < pathNodes.length - 1; i++) {
        const a = pathNodes[i];
        const b = pathNodes[i + 1];

        const match = graphData.links.find((l) => {
          const src = typeof l.source === "string" ? l.source : l.source.id;
          const dst = typeof l.target === "string" ? l.target : l.target.id;
          return (
            (src === a && dst === b) || (src === b && dst === a)
          );
        });

        if (match) {
          const src = typeof match.source === "string" ? match.source : match.source.id;
          const dst = typeof match.target === "string" ? match.target : match.target.id;
          const key = `${src}::${dst}`;
          if (!newEdges.includes(key)) {
            newEdges.push(key);
          }
        }
      }

      setCurrentStepIdx(nextIdx);
      setRevealedNodes(pathNodes);
      setRevealedEdges(newEdges);
    }, 7000); // 2s "thinking" between hops

    return () => clearTimeout(id);
  }, [isPlaying, currentStepIdx, trace, graphData.links]);

  // --- auto-run localized agents for current node (master orchestrator) -----

    // --- auto-run localized agents for current focus (master orchestrator) ----
    useEffect(() => {
      if (!activeIncidentId) return;
      if (!incident || incident.status !== "STARTED") return;

      const focus = selectedNodeId || currentNodeIdFromTrace;
      if (!focus) return;

      let cancelled = false;

      const run = async () => {
        try {
          setIsInvestigating(true);
          const res = await fetch(
            `${API_BASE}/incident/${activeIncidentId}/investigate`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ focus_node_id: focus, radius: 2 })
            }
          );
          const json = await res.json();
          if (!cancelled) {
            setInvestigation(json);
          }
        } catch (e) {
          console.error("investigate error", e);
        } finally {
          if (!cancelled) setIsInvestigating(false);
        }
      };

      run();
      return () => {
        cancelled = true;
      };
    }, [activeIncidentId, incident, selectedNodeId, currentNodeIdFromTrace]);

  // progressively reveal agent steps for the active node
    useEffect(() => {
      if (!incident || incident.status !== "STARTED") return;
      if (!investigation || !activeNodeDetails) return;

      const nodeInv =
        investigation.nodes.find(
          (n) => n.node_id === activeNodeDetails.id
        ) || null;
      if (!nodeInv) return;

      const totalSteps = nodeInv.agent_results.reduce(
        (acc, ar) => acc + (ar.steps ? ar.steps.length : 0),
        0
      );

      if (logStepIndex >= totalSteps) return;

      const id = setTimeout(() => {
        setLogStepIndex((i) => Math.min(totalSteps, i + 1));
      }, 1200);

      return () => clearTimeout(id);
    }, [
      incident?.status,
      investigation,
      activeNodeDetails?.id,
      logStepIndex,
    ]);

  // --- explain endpoint (high-level LLM-style summary) ----------------------

  const askExplain = async () => {
    if (!activeIncidentId) return;

    try {
      setIsExplaining(true);
      const focusNodeId = selectedNodeId || currentNodeIdFromTrace;

      const payload = {
        question: question.trim() || null,
        focus_node_id: focusNodeId,
        radius: 2,
      };

      const res = await fetch(
        `${API_BASE}/incident/${activeIncidentId}/explain`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );
      const json = await res.json();
      setAnswer(json.answer || "");
    } catch (e) {
      console.error("explain error", e);
      setAnswer("Sorry, I couldn't generate an explanation.");
    } finally {
      setIsExplaining(false);
    }
  };

  // --- upload graph JSON for active incident --------------------------------

    const handleGraphUpload = async (event, opts = {}) => {
      if (!activeIncidentId) {
        alert("No active incident selected.");
        return;
      }
      const file = event.target.files[0];
      if (!file) return;

      try {
        const text = await file.text();
        const json = JSON.parse(text);

        if (!json.nodes || !json.links) {
          alert("Invalid graph JSON: missing nodes or links");
          return;
        }

        await fetch(
          `${API_BASE}/incident/${activeIncidentId}/graph/upload`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(json),
          }
        );

        const gRes = await fetch(
          `${API_BASE}/incident/${activeIncidentId}/graph`
        );
        const gJson = await gRes.json();
        setGraphData({
          nodes: gJson.nodes,
          links: gJson.links,
        });

        setTrace([]);
        setCurrentStepIdx(0);
        setSelectedNodeId(null);
        setInvestigation(null);
        setRevealedNodes([]);
        setRevealedEdges([]);
        setLogStepIndex(0);

        // if this upload came from the first-screen overlay, hide it
        if (opts.fromOverlay) {
          setShowInitialUpload(false);
        }
      } catch (e) {
        console.error("graph upload error", e);
        alert("Error uploading graph; see console.");
      } finally {
        event.target.value = null;
      }
    };

  // --- focus camera on the current highlighted node -------------------------

  useEffect(() => {
    if (!fgRef.current || !highlightNodeId) return;

    const node = graphData.nodes.find((n) => n.id === highlightNodeId);
    if (!node) return;

    fgRef.current.centerAt(node.x, node.y, 800);
    fgRef.current.zoom(4, 800);
  }, [highlightNodeId, graphData.nodes]);

  // --- node click ------------------------------------------------------------

  const handleNodeClick = (node) => {
    // ignore agent satellites in the panel; they are just visual
    if (node.group === "agent") return;
    setSelectedNodeId(node.id);
    setIsPlaying(false);
  };


  // --- drawing logic ---------------------------------------------------------

  const nodeCanvasObject = useCallback(
    (node, ctx, globalScale) => {
      const label = node.group === "agent" ? "" : (node.label || node.id);
      const fontSize = 18 / globalScale;

      const isOnPath = revealedNodeSet.has(node.id);
      const isCurrent = node.id === highlightNodeId;

      let r = 7;
      if (node.group === "alert" || node.group === "error") r = 12;
      if (node.group === "event") r = 10;
      if (node.group === "db" || node.group === "cache" || node.group === "dns")
        r = 10;
      if (node.group === "agent") r = 4;

      // glow for nodes on critical path
      if (isOnPath || isCurrent) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, r + 6, 0, 2 * Math.PI, false);
        ctx.fillStyle = isCurrent
          ? "rgba(250, 250, 250, 0.22)"
          : "rgba(248, 113, 113, 0.18)";
        ctx.fill();
      }

      // main circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
      ctx.fillStyle = node.color || "#9ca3af";
      ctx.fill();
      ctx.lineWidth = 1.5 / globalScale;
      ctx.strokeStyle = isCurrent ? "#ffffff" : "#020617";
      ctx.stroke();

      if (!label) return;

      // label
      ctx.font = `${fontSize}px system-ui, -apple-system, sans-serif`;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#e5e7eb";
      ctx.fillText(label, node.x + (r + 4) / globalScale, node.y);
    },
    [highlightNodeId, revealedNodeSet]
  );

  const nodePointerAreaPaint = (node, color, ctx) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, 13, 0, 2 * Math.PI, false);
    ctx.fill();
  };

  const linkColor = (link) => {
    const src = typeof link.source === "string" ? link.source : link.source.id;
    const dst = typeof link.target === "string" ? link.target : link.target.id;
    const key = `${src}::${dst}`;
    const keyRev = `${dst}::${src}`;

    if (revealedEdgeSet.has(key) || revealedEdgeSet.has(keyRev)) {
      return "#f97373"; // critical, but only once revealed
    }

    return "#4b5563"; // neutral
  };

  const linkWidth = (link) => {
    const src = typeof link.source === "string" ? link.source : link.source.id;
    const dst = typeof link.target === "string" ? link.target : link.target.id;
    const key = `${src}::${dst}`;
    const keyRev = `${dst}::${src}`;

    if (revealedEdgeSet.has(key) || revealedEdgeSet.has(keyRev)) {
      return 2.5;
    }
    return 1;
  };

  const linkParticles = (link) => {
    const src = typeof link.source === "string" ? link.source : link.source.id;
    const dst = typeof link.target === "string" ? link.target : link.target.id;
    const key = `${src}::${dst}`;
    const keyRev = `${dst}::${src}`;

    if (revealedEdgeSet.has(key) || revealedEdgeSet.has(keyRev)) {
      return 3;
    }
    return 0;
  };

  // --- render ----------------------------------------------------------------

  return (
    <div className="app-root">
      <div className="debug-label">{debugLabel}</div>

      {/* TOP HUD */}
      <div className="top-bar">
        {/* Incident panel */}
        <div className="top-panel incident-panel">
          <div className="incident-tabs">
            {incidents.map((inc) => (
              <button
                key={inc.id}
                className={
                  "incident-tab" +
                  (inc.id === activeIncidentId ? " active" : "")
                }
                onClick={() => setActiveIncidentId(inc.id)}
              >
                {inc.title}
              </button>
            ))}
          </div>

          {incident && (
            <div className="incident-meta-row">
              <div className="incident-meta-left">
                <span
                  className={
                    "status-pill status-" +
                    incident.status.toLowerCase()
                  }
                >
                  {incident.status}
                </span>
                <span className="severity-pill">
                  {incident.severity}
                </span>
                <span className="incident-id">{incident.id}</span>
                {incident.started_at && (
                  <span className="incident-time">
                    Started: {formatTime(incident.started_at)}
                  </span>
                )}
              </div>
              <div className="incident-buttons">
                <button
                  className="btn primary"
                  onClick={introduceError}
                >
                  Introduce error
                </button>
                <button className="btn" onClick={resolveIncident}>
                  Resolve
                </button>
              </div>
            </div>
          )}

          <div className="trace-controls">
            <button
              className="btn small"
              disabled={currentStepIdx === 0}
              onClick={() => {
                setIsPlaying(false);
                setCurrentStepIdx((i) => Math.max(0, i - 1));
              }}
            >
              ◀
            </button>
            <button
              className={
                "btn small" + (isPlaying ? " primary" : "")
              }
              disabled={!trace.length}
              onClick={() => {
                if (!trace.length) return;
                if (!isPlaying && currentStepIdx >= trace.length - 1) {
                  setCurrentStepIdx(0);
                  setRevealedNodes([trace[0].node_id]);
                  setRevealedEdges([]);
                }
                setIsPlaying((p) => !p);
              }}
            >
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button
              className="btn small"
              disabled={currentStepIdx >= trace.length - 1}
              onClick={() => {
                setIsPlaying(false);
                const nextIdx = Math.min(
                  trace.length - 1,
                  currentStepIdx + 1
                );
                setCurrentStepIdx(nextIdx);
                const pathNodes = trace
                  .slice(0, nextIdx + 1)
                  .map((s) => s.node_id);
                setRevealedNodes(pathNodes);
              }}
            >
              ▶
            </button>
            <span className="trace-step-label">
              Step {trace.length ? currentStepIdx + 1 : 0} /{" "}
              {trace.length}
            </span>
          </div>

          {currentStep && (
            <div className="trace-line">
              <span className="trace-node-id">
                Node in trace: {currentStep.node_id}
              </span>
              <span className="trace-title">{currentStep.title}</span>
              <span className="trace-desc">
                {currentStep.description}
              </span>
            </div>
          )}
        </div>

        {/* Node + agents panel */}
        <div className="top-panel node-panel">
          <div className="panel-title">Node & agents</div>
          {activeNodeDetails ? (
            <>
              <div className="node-line">
                <span className="pill">{activeNodeDetails.group}</span>
                <span className="node-title">
                  {activeNodeDetails.label}
                </span>
              </div>
              <div className="node-subid">
                {activeNodeDetails.id}
              </div>
              {activeNodeDetails.details && (
                <div className="node-details">
                  {activeNodeDetails.details}
                </div>
              )}

              <div className="agent-section-title">
                Node Insights – Agent Findings
              </div>
              {isInvestigating && (
                <div className="agent-think">
                  Agents are analysing this neighbourhood…
                </div>
              )}
              {investigation && (
                <div className="agent-list">
                  {(() => {
                    const nodeInv =
                      investigation.nodes.find(
                        (n) => n.node_id === activeNodeDetails.id
                      ) || null;
                    if (!nodeInv) {
                      return (
                        <div className="agent-empty">
                          No agent analysis yet for this node.
                        </div>
                      );
                    }

                    // how many total log lines are available
                    const totalSteps = nodeInv.agent_results.reduce(
                      (acc, ar) => acc + (ar.steps ? ar.steps.length : 0),
                      0
                    );

                    let remaining = logStepIndex;

                    return (
                      <>
                        <div className="agent-score">
                          Suspicion score:{" "}
                          {nodeInv.suspicion_score.toFixed(2)}
                        </div>

                        {logStepIndex < totalSteps && (
                          <div className="agent-think">
                            Agents are traversing this neighbourhood…
                          </div>
                        )}

                        {nodeInv.agent_results.map((ar) => {
                          const steps = ar.steps || [];
                          const visibleCount = Math.max(
                            0,
                            Math.min(steps.length, remaining)
                          );
                          remaining -= visibleCount;

                          return (
                            <div
                              key={ar.agent_name}
                              className="agent-entry"
                            >
                              <b>{ar.agent_name}</b>: {ar.summary}
                              {visibleCount > 0 && (
                                <ul>
                                  {steps.slice(0, visibleCount).map((s, idx) => (
                                    <li key={idx}>{s}</li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          );
                        })}
                      </>
                    );
                  })()}
                </div>
              )}
            </>
          ) : (
            <div className="node-empty">
              Click a node or start the incident trace to see analysis.
            </div>
          )}
        </div>

        {/* Explainer + upload panel */}
        <div className="top-panel explainer-panel">
          <div className="panel-title">Heimdall Insight Engine</div>
          <textarea
            className="explain-input"
            placeholder="Ask about this incident, or leave blank for a summary…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
          />
          <button
            className="btn primary full"
            onClick={askExplain}
            disabled={isExplaining}
          >
            {isExplaining ? "Thinking…" : "Explain incident"}
          </button>
          {answer && (
            <pre className="explain-output">{answer}</pre>
          )}
        </div>
      </div>

      {/* GRAPH */}
      <div className="graph-wrapper">
         <ForceGraph2D
                  ref={fgRef}
                  graphData={graphData}
                  nodeId="id"
                  linkSource="source"
                  linkTarget="target"
                  nodeCanvasObject={nodeCanvasObject}
                  nodePointerAreaPaint={nodePointerAreaPaint}
                  nodeLabel={(node) => {
                    const group = (node.group || "node").toUpperCase();
                    const label = node.label || node.id;
                    const details = node.details || "";
                    return `${group}: ${label}${details ? "\n" + details : ""}`;
                  }}
                  linkColor={linkColor}
                  linkWidth={linkWidth}
                  linkDirectionalParticles={linkParticles}
                  linkDirectionalParticleSpeed={0.01}
                  linkDirectionalParticleWidth={2}
                  linkDirectionalArrowLength={5}
                  linkDirectionalArrowRelPos={0.9}
                  linkCurvature={0.08}
                  backgroundColor="#020617"
                  onNodeClick={handleNodeClick}
                />
              </div>

              {showInitialUpload && (
                <div className="overlay">

                  <div className="overlay-inner">

                    <div className="overlay-card">
                      <div className="overlay-title">Heimdall —  Autonomous Incident Insight Engine</div>

                      <div className="overlay-subtitle">
                        Drop in a JSON dependency graph (<code>nodes</code> + <code>links</code>).
                        Heimdall will attach localized agents to every node and wire them
                        to a master orchestrator to trace incidents across the graph.
                      </div>

                      <div className="overlay-actions">
                        <input
                          type="file"
                          accept="application/json"
                          onChange={(e) => handleGraphUpload(e, { fromOverlay: true })}
                        />

                        <button className="btn" onClick={() => setShowInitialUpload(false)}>
                          Skip – use demo incidents
                        </button>
                      </div>

                      <div className="overlay-skip">
                        Expected format: <code>{"{"}"nodes":[...],"links":[...]{"}"}</code>
                      </div>
                    </div>
                  </div>

                </div>
              )}
    </div>
  );
}

export default App;
