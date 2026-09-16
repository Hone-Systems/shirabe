import { useEffect, useMemo, useState, useRef } from "react";
import { ArrowDownToLine, ArrowUpRight, RefreshCw, Search } from "lucide-react";

type Answer = {
  id: string;
  answer: "yes" | "no" | "unknown" | "not_applicable";
  rationale?: string;
  evidence: { source_id: string; date: string }[];
};
type Paper = {
  id: string;
  title: string;
  doi?: string;
  year: number;
  field: string;
  field_id: number;
  split: string;
  input_scope: string;
  researcher_kind?: string;
  previous_review?: {
    known_answers: number;
    coverage: number;
    score: number | null;
    followup_end: string;
  };
  observed_through?: string;
  followup_end: string;
  sampling_cohort?: string;
  aggregate: {
    score: number | null;
    coverage: number;
    lower_bound: number | null;
    upper_bound: number | null;
    known_answers: number;
    dimensions: Record<
      string,
      {
        yes: number;
        no: number;
        unknown: number;
        not_applicable: number;
        coverage: number | null;
        known_index: number | null;
      }
    >;
  };
  research: { answers: Answer[]; search_gaps: string[]; summary?: string };
  research_plan?: { queries?: string[] };
  source_manifest: {
    source_id: string;
    title?: string;
    finding?: string;
    scope?: string;
    date?: string;
    date_basis?: string;
    url: string;
    characters: number | null;
    method: string;
    truncated: boolean;
  }[];
  original_source: { url?: string; characters?: number };
  central_claims?: string[];
  identity_resolution?: { explanation?: string; training_eligible?: boolean };
  llm: {
    model: string;
    usage: { input_tokens: number | null; output_tokens: number | null };
  };
};
type Dataset = {
  planned_papers: number;
  progress: {
    id: string;
    title: string;
    stage: string;
    updated_at: string;
    source_count?: number;
    sources_read?: number;
    query?: string;
    note?: string;
    error?: string;
  }[];
  rubric_version: string;
  questions: { id: string; dimension: string; question: string }[];
  records: Paper[];
  summary: {
    papers: number;
    scored: number;
    answer_counts: Record<string, number>;
  };
  method: Record<string, string>;
};
const percent = (n: number | null) =>
  n === null ? "—" : `${(n * 100).toFixed(0)}%`;
const colors = {
  yes: "#8ee9cc",
  no: "#edb679",
  unknown: "#667975",
  not_applicable: "#303e3b",
};
export default function EvidenceExplorer() {
  const [data, setData] = useState<Dataset | null>(null),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(false);
  const [query, setQuery] = useState(""),
    [field, setField] = useState("all"),
    [split, setSplit] = useState("all"),
    [coverage, setCoverage] = useState("all"),
    [method, setMethod] = useState("all"),
    [selected, setSelected] = useState(""),
    [view, setView] = useState<"map" | "matrix">("matrix");
  async function refresh(quiet = false) {
    if (!quiet) setLoading(true);
    setError("");
    try {
      const r = await fetch("/api/evidence");
      if (!r.ok) throw new Error("Evidence dataset is unavailable.");
      setData(await r.json());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(true), 2500);
    return () => window.clearInterval(timer);
  }, []);
  const rows = useMemo(
    () =>
      data?.records.filter(
        (p) =>
          (field === "all" || String(p.field_id) === field) &&
          (split === "all" || p.split === split) &&
          (method === "all" ||
            (p.researcher_kind === "autonomous_agent" ? "agent" : "batch") ===
              method) &&
          (coverage === "all" ||
            (coverage === "scored"
              ? p.aggregate.score !== null
              : p.aggregate.score === null)) &&
          `${p.title} ${p.field}`.toLowerCase().includes(query.toLowerCase()),
      ) ?? [],
    [data, field, split, coverage, query, method],
  );
  const reviewed =
    data?.records.filter((p) => p.researcher_kind === "autonomous_agent")
      .length ?? 0;
  const jobs = data?.progress ?? [];
  const running = jobs.filter(
    (j) => !["complete", "error", "queued"].includes(j.stage),
  );
  const queued = jobs.filter((j) => j.stage === "queued").length;
  const failed = jobs.filter((j) => j.stage === "error");
  const stageNames: Record<string, string> = {
    reading_original: "Reading original paper",
    planning_searches: "Planning follow-up research",
    searching: "Searching for evidence",
    selecting_sources: "Choosing independent sources",
    reading_sources: "Reading full sources",
    answering_rubric: "Answering 20 rubric questions",
    validating_evidence: "Validating cited evidence",
  };
  const detailRef = useRef<HTMLElement>(null);
  const active = rows.find((p) => p.id === selected) ?? rows[0];
  useEffect(() => {
    detailRef.current?.scrollTo({ top: 0 });
  }, [active?.id]);
  function download() {
    const blob = new Blob(
      [JSON.stringify({ ...data, records: rows }, null, 2)],
      { type: "application/json" },
    );
    const u = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = u;
    a.download = "shirabe-evaluation-data.json";
    a.click();
    URL.revokeObjectURL(u);
  }
  return (
    <main id="workspace" className="evidence-workspace">
      <div className="evidence-heading">
        <div>
          <span className="micro">WORDING / OUTCOMES / RESEARCH PILOT</span>
          <h1>What happened next.</h1>
        </div>
        <div className="evidence-count">
          <strong>{data?.summary.papers ?? 0}</strong>
          <span>
            papers ·{" "}
            {data?.records.filter(
              (p) => p.researcher_kind === "autonomous_agent",
            ).length ?? 0}{" "}
            agent reviewed
          </span>
          <button
            className="icon"
            aria-label="Refresh evidence"
            onClick={() => void refresh()}
            disabled={loading}
          >
            <RefreshCw size={16} className={loading ? "spinning" : ""} />
          </button>
          <button
            className="icon"
            aria-label="Export filtered evidence"
            onClick={download}
            disabled={!data}
          >
            <ArrowDownToLine size={16} />
          </button>
        </div>
      </div>
      <section className="live-research" aria-label="Live research progress">
        <div className="live-research-heading">
          <span>
            <i className={running.length ? "live-dot active" : "live-dot"} />{" "}
            {running.length
              ? "GENERATING OUTCOMES"
              : queued
                ? "RESEARCH QUEUE"
                : reviewed
                  ? "REVIEWS COMPLETE"
                  : "RESEARCH QUEUE"}
          </span>
          <span role="status">
            {reviewed}/{data?.planned_papers ?? data?.summary.papers ?? 0} agent
            reviewed · {running.length} active · {queued} queued
            {failed.length ? ` · ${failed.length} need retry` : ""}
          </span>
        </div>
        {running.length > 0 && (
          <div className="live-jobs">
            {running.map((j) => (
              <div key={j.id} className="live-job" title={j.note || j.query}>
                <strong title={j.title}>{j.title}</strong>
                <span>
                  <i />
                  {stageNames[j.stage] ?? j.stage}
                  {j.stage === "reading_sources" && j.source_count !== undefined
                    ? ` · ${j.sources_read ?? 0}/${j.source_count ?? 0}`
                    : ""}
                </span>
              </div>
            ))}
          </div>
        )}
        {failed.length > 0 && (
          <details>
            <summary>{failed.length} retrieval or processing errors</summary>
            {failed.map((j) => (
              <p key={j.id}>
                {j.title}: {j.error}
              </p>
            ))}
          </details>
        )}
      </section>
      <div className="evidence-filters">
        <label className="evidence-search">
          <Search size={15} />
          <input
            aria-label="Search papers"
            placeholder="Find a paper"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <label>
          Field
          <select
            aria-label="Filter field"
            value={field}
            onChange={(e) => setField(e.target.value)}
          >
            <option value="all">All fields</option>
            {Array.from(
              new Map(
                data?.records.map((p) => [p.field_id, p.field]),
              ).entries(),
            ).map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Split
          <select
            aria-label="Filter split"
            value={split}
            onChange={(e) => setSplit(e.target.value)}
          >
            <option value="all">All splits</option>
            {["train", "validation", "calibration", "test", "challenge"].map(
              (s) => (
                <option key={s}>{s}</option>
              ),
            )}
          </select>
        </label>
        <label>
          Research
          <select
            aria-label="Filter research method"
            value={method}
            onChange={(e) => setMethod(e.target.value)}
          >
            <option value="all">All methods</option>
            <option value="agent">Agent reviewed</option>
            <option value="batch">Provisional batch</option>
          </select>
        </label>
        <label>
          Evidence
          <select
            aria-label="Filter coverage"
            value={coverage}
            onChange={(e) => setCoverage(e.target.value)}
          >
            <option value="all">All coverage</option>
            <option value="scored">Enough to score</option>
            <option value="sparse">Insufficient evidence</option>
          </select>
        </label>
      </div>
      {error && <p role="alert">{error}</p>}
      <div className="evidence-body">
        <section className="evidence-plots" aria-label="Evaluation overview">
          <div className="evidence-plot-controls">
            <span className="micro">{rows.length} PAPERS / 20 QUESTIONS</span>
            <div>
              <button
                aria-pressed={view === "matrix"}
                onClick={() => setView("matrix")}
              >
                Answers
              </button>
              <button
                aria-pressed={view === "map"}
                onClick={() => setView("map")}
              >
                Coverage map
              </button>
            </div>
          </div>
          <div className="evidence-legend">
            {Object.entries(colors).map(([key, color]) => (
              <span key={key}>
                <i style={{ background: color }} />
                {key.replace("_", " ")}
              </span>
            ))}
          </div>
          {rows.length === 0 ? (
            <div className="evidence-empty">
              {loading
                ? "Reading evidence…"
                : data?.records.length
                  ? "No papers match these filters."
                  : "Research is running. Completed papers will appear here automatically."}
            </div>
          ) : view === "matrix" ? (
            <div
              className="evidence-matrix"
              tabIndex={0}
              role="region"
              aria-label="Paper rubric answer matrix"
            >
              <div className="matrix-dimensions">
                <span>Paper</span>
                <span>Validation</span>
                <span>Uptake</span>
                <span>Utility</span>
                <span>Durability</span>
              </div>
              {rows.map((p) => (
                <button
                  key={p.id}
                  className={`matrix-row ${active?.id === p.id ? "selected" : ""}`}
                  onClick={() => setSelected(p.id)}
                  aria-label={`${p.title}; ${p.aggregate.known_answers} known answers; inspect paper`}
                >
                  <span className="matrix-title">
                    {p.title}
                    <small>
                      {p.year} · {p.split}
                    </small>
                  </span>
                  <span className="matrix-cells">
                    {data?.questions.map((q) => {
                      const a = p.research.answers.find((a) => a.id === q.id);
                      return (
                        <i
                          key={q.id}
                          style={{ background: colors[a?.answer ?? "unknown"] }}
                          title={`${q.question}: ${a?.answer ?? "unknown"}`}
                        />
                      );
                    })}
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <div className="evidence-map">
              <svg
                viewBox="0 0 660 450"
                role="img"
                aria-label="Coverage versus known-answer index; hollow points lack sufficient evidence for an overall score"
              >
                <path d="M60 25V380H630" fill="none" stroke="#667975" />
                {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                  <g key={v}>
                    <path d={`M60 ${380 - v * 340}H630`} stroke="#253b33" />
                    <text x="48" y={384 - v * 340} textAnchor="end">
                      {percent(v)}
                    </text>
                    <text x={60 + v * 570} y="403" textAnchor="middle">
                      {percent(v)}
                    </text>
                  </g>
                ))}
                <text x="350" y="440" textAnchor="middle">
                  Evidence coverage
                </text>
                <text
                  x="15"
                  y="215"
                  transform="rotate(-90 15 215)"
                  textAnchor="middle"
                >
                  Known-answer index
                </text>
                {rows.map((p, i) => {
                  const dims = Object.values(p.aggregate.dimensions).filter(
                    (d) => d.known_index !== null,
                  );
                  if (!dims.length) return null;
                  const value =
                    dims.reduce((s, d) => s + (d.known_index ?? 0), 0) /
                    dims.length;
                  return (
                    <circle
                      key={p.id}
                      cx={60 + p.aggregate.coverage * 570 + ((i % 3) - 1) * 3}
                      cy={380 - value * 340 + ((i % 5) - 2) * 2}
                      r={active?.id === p.id ? 8 : 5}
                      fill={p.aggregate.score === null ? "#0c1713" : "#8ee9cc"}
                      stroke={active?.id === p.id ? "#fff" : "#8ee9cc"}
                    >
                      <title>
                        {p.title}: coverage {percent(p.aggregate.coverage)},
                        known-answer index {percent(value)}; select paper in the
                        list below.
                      </title>
                    </circle>
                  );
                })}
              </svg>
              <div className="map-paper-list">
                {rows.map((p) => (
                  <button
                    key={p.id}
                    aria-pressed={active?.id === p.id}
                    onClick={() => setSelected(p.id)}
                  >
                    {p.title}
                  </button>
                ))}
              </div>
              <p>
                Hollow points have too little evidence for an overall score.
                Positions have a small visual offset to expose overlapping
                papers. Papers with no known answers have no index and appear
                only in the list, never as zero-score points.
              </p>
            </div>
          )}
        </section>
        <section
          ref={detailRef}
          className="evidence-paper"
          aria-label="Selected paper evidence"
          tabIndex={0}
        >
          {active && (
            <>
              <span className="micro">
                {active.split.toUpperCase()} / {active.year} /{" "}
                {active.input_scope.replaceAll("_", " ")} /{" "}
                {active.researcher_kind === "autonomous_agent"
                  ? "AGENT REVIEW"
                  : "PROVISIONAL BATCH"}
              </span>
              <h2>{active.title}</h2>
              {active.identity_resolution && (
                <p className="evidence-note" role="note">
                  Identity check: {active.identity_resolution.explanation}{" "}
                  {active.identity_resolution.training_eligible === false
                    ? "Excluded from training."
                    : ""}
                </p>
              )}
              <p className="evidence-note">
                Evidence through {active.followup_end}.{" "}
                {active.observed_through
                  ? "Observed outcomes through the research date."
                  : "Earlier batch used a four-year window; reassessment pending."}
              </p>
              <div className="paper-outcome">
                <div>
                  <strong>{percent(active.aggregate.score)}</strong>
                  <span>
                    {active.aggregate.score === null
                      ? active.aggregate.lower_bound === null
                        ? "Rubric not applicable"
                        : "Too little evidence"
                      : "Observed outcome index"}
                  </span>
                </div>
                <div>
                  <strong>{percent(active.aggregate.coverage)}</strong>
                  <span>Evidence coverage</span>
                </div>
              </div>
              {active.aggregate.lower_bound !== null && (
                <>
                  <div className="outcome-bounds">
                    <span
                      style={{
                        left: `${(active.aggregate.lower_bound ?? 0) * 100}%`,
                        width: `${((active.aggregate.upper_bound ?? 1) - (active.aggregate.lower_bound ?? 0)) * 100}%`,
                      }}
                    />
                  </div>
                  <p className="evidence-note">
                    Unknown-answer range {percent(active.aggregate.lower_bound)}
                    –{percent(active.aggregate.upper_bound)}. This is a rubric
                    index, not a predicted probability.
                  </p>
                </>
              )}
              <div className="paper-links">
                {(active.original_source.url || active.doi) && (
                  <a
                    href={active.original_source.url || active.doi}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Original paper <ArrowUpRight size={13} />
                  </a>
                )}
                <span>
                  {active.llm.usage.input_tokens === null
                    ? "Autonomous agent · sources read directly"
                    : `${active.llm.usage.input_tokens.toLocaleString()} input tokens · no clipping`}
                </span>
              </div>
              {["validation", "uptake", "utility", "durability"].map((d) => (
                <div className="rubric-dimension" key={d}>
                  <h3>
                    {d}
                    <span>
                      {percent(active.aggregate.dimensions[d].known_index)}
                    </span>
                  </h3>
                  {data?.questions
                    .filter((q) => q.dimension === d)
                    .map((q) => {
                      const a = active.research.answers.find(
                        (a) => a.id === q.id,
                      );
                      return (
                        <details key={q.id}>
                          <summary>
                            <i
                              style={{
                                background: colors[a?.answer ?? "unknown"],
                              }}
                            />
                            <span>{q.question}</span>
                            <b>{a?.answer.replace("_", " ") ?? "unknown"}</b>
                          </summary>
                          <div className="answer-sources">
                            {a?.rationale && <p>{a.rationale}</p>}
                            {a?.evidence.length ? (
                              a.evidence.map((e, i) => {
                                const source = active.source_manifest.find(
                                  (s) => s.source_id === e.source_id,
                                );
                                return source ? (
                                  <a
                                    key={i}
                                    href={source.url}
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    {e.date} · {new URL(source.url).hostname}{" "}
                                    <ArrowUpRight size={13} />
                                  </a>
                                ) : null;
                              })
                            ) : (
                              <p>
                                {a?.answer === "not_applicable"
                                  ? "Not applicable to this contribution."
                                  : "No validated supporting evidence in the follow-up window. This is not a failure label."}
                              </p>
                            )}
                          </div>
                        </details>
                      );
                    })}
                </div>
              ))}
              <details className="evidence-method">
                <summary>Research summary & search trail</summary>
                {!!active.central_claims?.length && (
                  <>
                    <p>Original central claims assessed:</p>
                    <ul>
                      {active.central_claims.map((claim, i) => (
                        <li key={i}>{claim}</li>
                      ))}
                    </ul>
                  </>
                )}
                {active.previous_review && (
                  <p>
                    Evidence answered: {active.previous_review.known_answers}/20
                    in the provisional batch → {active.aggregate.known_answers}
                    /20 in this agent review. The earlier batch ended at{" "}
                    {active.previous_review.followup_end}; this review includes
                    outcomes through {active.followup_end}.
                  </p>
                )}
                {active.research.summary && <p>{active.research.summary}</p>}
                <p>Cohort: {active.sampling_cohort ?? "initial pilot"}</p>
                <ol>
                  {active.research_plan?.queries?.map((q, i) => (
                    <li key={i}>{q}</li>
                  ))}
                </ol>
              </details>
              <details className="evidence-method">
                <summary>Research gaps & source manifest</summary>
                <ul>
                  {active.research.search_gaps.map((g, i) => (
                    <li key={i}>{g}</li>
                  ))}
                </ul>
                {active.source_manifest.map((s) => (
                  <div key={s.source_id}>
                    <p>
                      <a href={s.url} target="_blank" rel="noreferrer">
                        {s.source_id} · {s.title || new URL(s.url).hostname}
                      </a>{" "}
                      · {s.date_basis === "observed_as_of" ? "Observed " : ""}
                      {s.date}
                      {s.scope ? ` · ${s.scope.replaceAll("_", " ")}` : ""}
                      {s.characters !== null
                        ? ` · ${s.characters.toLocaleString()} characters`
                        : ""}
                    </p>
                    {s.finding && <p>{s.finding}</p>}
                  </div>
                ))}
              </details>
            </>
          )}
        </section>
      </div>
      <p className="evidence-footnote">
        Observed outcomes · unknown ≠ failure · landmark checks never train the
        model · Analyze shows an experimental outcome model
      </p>
    </main>
  );
}
