import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Check,
  ChevronDown,
  ChevronRight,
  FlaskConical,
  LoaderCircle,
  ScanText,
  Upload,
  X,
  ExternalLink,
  Info,
  SlidersHorizontal,
} from "lucide-react";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "./styles.css";
import type { Prediction, Report } from "./types";

const fmt = (n: number, digits = 3) => n.toFixed(digits);
const pct = (n: number, digits = 1) => `${(n * 100).toFixed(digits)}%`;
const number = (n: number) => n.toLocaleString("en-US");
const SAMPLE =
  "We examined whether the timing of feedback affects performance on a repeated learning task. Participants completed a series of trials under three feedback schedules, with assignment randomized before the first session. We measured accuracy and response time at baseline and after each training block. Immediate feedback was associated with higher accuracy during training, but the difference was smaller at the delayed assessment. The estimates were similar after accounting for baseline performance and the number of completed trials. However, the sample was limited to volunteers from a single institution, and the study did not assess transfer to other tasks. These results suggest that feedback timing may influence short-term learning, while its effect on retention remains uncertain. Further work with a larger and more varied sample is needed to estimate the conditions under which the observed differences persist.";

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(
      typeof data?.detail === "string"
        ? data.detail
        : `Request failed (${response.status}). Please try again.`,
    );
  return data;
}
function download(data: unknown, filename: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function SectionLabel({
  children,
  number: ordinal,
}: {
  children: React.ReactNode;
  number?: string;
}) {
  return (
    <div className="section-label">
      {ordinal && <span>{ordinal}</span>}
      {children}
    </div>
  );
}
function Empty({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="empty">
      <ScanText size={36} strokeWidth={1} />
      <h2>{title}</h2>
      <p>{children}</p>
    </div>
  );
}
function Chart({
  title,
  series,
  xLabel,
  yLabel,
  domain = [0, 1],
  xDomain = [0, 1],
  diagonal = false,
}: {
  title: string;
  series: { name: string; color: string; points: [number, number][] }[];
  xLabel: string;
  yLabel: string;
  domain?: [number, number];
  xDomain?: [number, number];
  diagonal?: boolean;
}) {
  const [active, setActive] = useState<string | null>(null);
  const x = (v: number) =>
    76 + ((v - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * 424;
  const y = (v: number) =>
    230 - ((v - domain[0]) / (domain[1] - domain[0] || 1)) * 190;
  return (
    <div className="chart">
      <svg
        viewBox="0 0 540 286"
        role="img"
        aria-label={`${title}. ${xLabel} on the horizontal axis; ${yLabel} on the vertical axis.`}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1="76"
              x2="500"
              y1={40 + t * 190}
              y2={40 + t * 190}
              className="gridline"
            />
            <text x="62" y={44 + t * 190} textAnchor="end">
              {fmt(domain[1] - t * (domain[1] - domain[0]), 2)}
            </text>
            <text x={76 + t * 424} y="250" textAnchor="middle">
              {xDomain[1] > 10
                ? number(Math.round(xDomain[0] + t * (xDomain[1] - xDomain[0])))
                : fmt(xDomain[0] + t * (xDomain[1] - xDomain[0]), 2)}
            </text>
          </g>
        ))}
        {diagonal && (
          <line
            x1={x(0)}
            y1={y(0)}
            x2={x(1)}
            y2={y(1)}
            stroke="var(--muted)"
            strokeDasharray="4 6"
          />
        )}
        {series.map((s, seriesIndex) => (
          <g key={s.name}>
            <polyline
              strokeDasharray={seriesIndex === 1 ? "6 4" : undefined}
              fill="none"
              stroke={s.color}
              strokeWidth="2.5"
              points={s.points.map(([a, b]) => `${x(a)},${y(b)}`).join(" ")}
            />
            {s.points.length <= 12 &&
              s.points.map(([a, b], i) => (
                <circle key={i} cx={x(a)} cy={y(b)} r="4" fill={s.color}>
                  <title>
                    {s.name}: {fmt(a)} / {fmt(b)}
                  </title>
                </circle>
              ))}
          </g>
        ))}
        <text x="275" y="277" textAnchor="middle">
          {xLabel}
        </text>
        <text x="12" y="136" textAnchor="middle" transform="rotate(-90 12 136)">
          {yLabel}
        </text>
      </svg>
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.name}>
            <i style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
        {diagonal && (
          <span>
            <i className="dash" />
            Reference
          </span>
        )}
      </div>
      <button
        className="text-button chart-data-toggle"
        onClick={() => setActive(active ? null : "data")}
        aria-expanded={!!active}
      >
        {active ? "Hide" : "Inspect"} chart data <ChevronDown size={13} />
      </button>
      {active && (
        <div
          tabIndex={0}
          role="region"
          aria-label={`${title} data`}
          className="table-scroll chart-data"
        >
          <table>
            <thead>
              <tr>
                <th>Series</th>
                <th>{xLabel}</th>
                <th>{yLabel}</th>
              </tr>
            </thead>
            <tbody>
              {series.flatMap((s) =>
                s.points.map(([a, b], i) => (
                  <tr key={`${s.name}-${i}`}>
                    <td>{s.name}</td>
                    <td>{fmt(a)}</td>
                    <td>{fmt(b)}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Analysis({ report }: { report: Report | null }) {
  const [text, setText] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState<"upload" | "predict" | null>(null);
  const [masked, setMasked] = useState(false);
  const [category, setCategory] = useState("all");
  const [allFeatures, setAllFeatures] = useState(false);
  const [drag, setDrag] = useState(false);
  const request = useRef<AbortController | null>(null);
  const words = (text.match(/[A-Za-z]+(?:[-’'][A-Za-z]+)*/g) || []).length;
  const valid = words >= 80 && words <= 800;
  useEffect(() => () => request.current?.abort(), []);
  function edit(value: string) {
    request.current?.abort();
    setText(value);
    setResult(null);
    setError("");
    setNotice("");
    setCategory("all");
    setAllFeatures(false);
    setBusy(null);
  }
  async function upload(file: File) {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setError("");
    setResult(null);
    setNotice("");
    setCategory("all");
    setAllFeatures(false);
    if (file.size > 10 * 1024 * 1024) {
      setError("Choose a PDF or text file smaller than 10 MB.");
      return;
    }
    if (!/\.(pdf|txt)$/i.test(file.name)) {
      setError("Choose a PDF or UTF-8 .txt file.");
      return;
    }
    setBusy("upload");
    const form = new FormData();
    form.append("file", file);
    try {
      const data = await api<{
        text: string;
        notice: string;
        filename: string;
      }>("/api/extract", {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
      setText(data.text);
      setNotice(`${data.filename} · ${data.notice}`);
    } catch (e) {
      if (!controller.signal.aborted) setError((e as Error).message);
    } finally {
      if (request.current === controller) setBusy(null);
    }
  }
  async function analyze(e: React.FormEvent) {
    e.preventDefault();
    if (!valid) {
      setError(
        `Use an English abstract of 80–800 words. This input has ${words}.`,
      );
      return;
    }
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setBusy("predict");
    setError("");
    setResult(null);
    try {
      setResult(
        await api<Prediction>("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
          signal: controller.signal,
        }),
      );
    } catch (e) {
      if (!controller.signal.aborted) setError((e as Error).message);
    } finally {
      if (request.current === controller) setBusy(null);
    }
  }
  const contributions = result
    ? [...result.features].sort(
        (a, b) => Math.abs(b.contribution) - Math.abs(a.contribution),
      )
    : [];
  const maxContribution = Math.max(
    ...contributions.map((f) => Math.abs(f.contribution)),
    0.1,
  );
  const categories = result
    ? [...new Set(result.tokens.map((t) => t.category))]
    : [];
  return (
    <>
      <div className="page-intro">
        <div>
          <SectionLabel>RESEARCH SIGNAL LABORATORY</SectionLabel>
          <h1>Read between the claims.</h1>
          <p>
            Explore what a paper’s writing style says about its citation
            attention.
          </p>
        </div>
        <div className="intro-index">
          <span>EXPERIMENT 001</span>
          <strong>Language → impact</strong>
          <span>ABSTRACT-LEVEL INFERENCE</span>
        </div>
      </div>
      <div className="analysis-grid">
        <section className="panel input-panel">
          <div className="panel-heading">
            <SectionLabel number="01">PAPER INPUT</SectionLabel>
            <span className="tag">ENGLISH ABSTRACT</span>
          </div>
          <form onSubmit={analyze}>
            <div
              className={`dropzone ${drag ? "dragging" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                if (!busy && e.dataTransfer.files[0])
                  void upload(e.dataTransfer.files[0]);
              }}
            >
              <Upload size={20} />
              <div>
                <label className="upload-label" htmlFor="paper-upload">
                  {busy === "upload" ? "Extracting paper…" : "Upload a paper"}
                  <input
                    id="paper-upload"
                    type="file"
                    accept=".pdf,.txt"
                    disabled={!!busy}
                    onChange={(e) => {
                      if (e.target.files?.[0]) void upload(e.target.files[0]);
                      e.target.value = "";
                    }}
                  />
                </label>
                <p>PDF or TXT · up to 10 MB · or drop it here</p>
              </div>
              <ArrowRight size={18} />
            </div>
            {notice && (
              <div className="notice">
                <Info size={16} />
                <span>{notice}</span>
              </div>
            )}
            <div className="field-heading">
              <label htmlFor="abstract">Abstract text</label>
              <button
                type="button"
                className="text-button"
                disabled={!!busy}
                onClick={() => {
                  edit(SAMPLE);
                  setNotice(
                    "Illustrative abstract, written for this demo. It is not a published paper or a labeled example.",
                  );
                }}
              >
                Load example <ArrowRight size={13} />
              </button>
            </div>
            <textarea
              id="abstract"
              value={text}
              aria-invalid={!!error}
              onChange={(e) => edit(e.target.value)}
              disabled={!!busy}
              placeholder="Paste the abstract here. We look at how the research is described: its claims, caveats, evidence, and structure."
              aria-describedby="input-help input-error"
              maxLength={20000}
            />
            <div className="input-meta" id="input-help">
              <span className={words > 800 ? "warning-text" : ""}>
                {number(words)} / 800 words
              </span>
              <span>Minimum 80 · English only</span>
            </div>
            <div id="input-error" aria-live="polite">
              {error && (
                <p className="error" role="alert">
                  {error}
                </p>
              )}
            </div>
            <div className="input-actions">
              <button
                type="submit"
                className="primary"
                disabled={!!busy || !text.trim()}
              >
                {busy ? (
                  <LoaderCircle className="spin" size={17} />
                ) : (
                  <Activity size={17} />
                )}{" "}
                {busy === "predict"
                  ? "Analyzing style…"
                  : busy === "upload"
                    ? "Extracting…"
                    : "Analyze writing style"}
                {!busy && <ArrowRight size={17} />}
              </button>
              <button
                type="button"
                className="icon-button"
                aria-label="Clear abstract"
                disabled={!text || !!busy}
                onClick={() => {
                  edit("");
                  setNotice("");
                }}
              >
                <X size={18} />
              </button>
            </div>
            <p className="privacy">
              <span className="status-dot" />
              Processed locally. Uploaded papers are not saved.
            </p>
          </form>
        </section>
        <section
          className="panel result-panel"
          aria-live="polite"
          aria-busy={busy === "predict"}
        >
          <div className="panel-heading">
            <SectionLabel number="02">SIGNAL ESTIMATE</SectionLabel>
            <span className="tag">
              {result ? "ANALYSIS COMPLETE" : "AWAITING INPUT"}
            </span>
          </div>
          {!result ? (
            <>
              <Empty
                title={
                  busy === "predict"
                    ? "Tracing the writing signal…"
                    : "A signal, with its evidence."
                }
              >
                Analyze an abstract to see its estimated probability, feature
                contributions, and token-level representation.
              </Empty>
              <div className="result-explainer">
                <span className="eyebrow">WHAT THIS ESTIMATES</span>
                <p>
                  Membership in the higher-cited group of a sampled field and
                  year. Citation attention is a proxy; it does not establish
                  scientific quality or success.
                </p>
              </div>
            </>
          ) : (
            <>
              <div className="score-row">
                <div>
                  <div className="score">
                    {(result.probability * 100).toFixed(1)}
                    <span>%</span>
                  </div>
                  <p>Estimated high-citation probability</p>
                </div>
                <div className="score-context">
                  <span>TRAINING PRIOR</span>
                  <strong>{pct(result.baseline)}</strong>
                  <span>
                    {result.probability - result.baseline >= 0 ? "+" : ""}
                    {((result.probability - result.baseline) * 100).toFixed(
                      1,
                    )}{" "}
                    pp from prior
                  </span>
                </div>
              </div>
              <div
                className="probability-bar"
                role="img"
                aria-label={`Estimated probability ${pct(result.probability)}; prior ${pct(result.baseline)}`}
              >
                <div style={{ width: pct(result.probability) }} />
                <span style={{ left: pct(result.baseline) }} />
              </div>
              <div className="scale-labels">
                <span>0%</span>
                <span>Sample-relative citation attention</span>
                <span>100%</span>
              </div>
              <p className="score-caveat">
                An experimental estimate for citation attention, not a
                probability that the research is true or will succeed.
              </p>
              {result.warnings.map((w) => (
                <p className="notice warning" key={w}>
                  <Info size={16} />
                  <span>{w}</span>
                </p>
              ))}
              <div className="contribution-header">
                <h2>What moved this estimate</h2>
                <span>LOG-ODDS CONTRIBUTION</span>
              </div>
              <div className="contribution-list">
                {contributions.slice(0, 6).map((f) => (
                  <div className="contribution" key={f.name}>
                    <span>{f.label}</span>
                    <div className="signed-track">
                      <i
                        className={
                          f.contribution >= 0 ? "positive" : "negative"
                        }
                        style={{
                          left:
                            f.contribution >= 0
                              ? "50%"
                              : `${50 - (50 * Math.abs(f.contribution)) / maxContribution}%`,
                          width: `${(50 * Math.abs(f.contribution)) / maxContribution}%`,
                        }}
                      />
                    </div>
                    <code
                      className={
                        f.contribution >= 0 ? "positive-text" : "warning-text"
                      }
                    >
                      {f.contribution >= 0 ? "+" : ""}
                      {fmt(f.contribution, 2)}
                    </code>
                  </div>
                ))}
              </div>
              <div className="result-footer">
                <span>Associations, not rewriting advice.</span>
                <button
                  className="text-button"
                  onClick={() => download(result, "shirabe-analysis.json")}
                >
                  <ArrowDownToLine size={14} />
                  Export analysis
                </button>
              </div>
            </>
          )}
        </section>
      </div>
      <section className="panel representation">
        <div className="panel-heading">
          <div>
            <SectionLabel number="03">INSIDE THE REPRESENTATION</SectionLabel>
            <h2>From language to a measurable signal.</h2>
          </div>
          <span className="tag">
            {report?.feature_count ?? "—"} STYLE FEATURES
          </span>
        </div>
        <div className="pipeline">
          {[
            { title: "Tokenize", body: "Words, numbers & punctuation" },
            {
              title: "Measure style",
              body: "Rhetoric, structure & function words",
            },
            { title: "Standardize", body: "Training mean · cap at ±5σ" },
            { title: "Estimate", body: "Linear weights · sigmoid calibration" },
          ].map((step, i) => (
            <React.Fragment key={step.title}>
              <div className="pipeline-step">
                <span>0{i + 1}</span>
                <strong>{step.title}</strong>
                <small>{step.body}</small>
              </div>
              {i < 3 && <ArrowRight className="pipeline-arrow" size={18} />}
            </React.Fragment>
          ))}
        </div>
        {result ? (
          <>
            <div className="token-toolbar">
              <div>
                <h3>Token inspection</h3>
                <span>
                  Dictionary matches describe language; they are not individual
                  token attributions.
                </span>
              </div>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={masked}
                  onChange={(e) => setMasked(e.target.checked)}
                />
                Mask content words
              </label>
            </div>
            <div className="filter-tabs" aria-label="Token categories">
              <button
                className={category === "all" ? "selected" : ""}
                aria-pressed={category === "all"}
                onClick={() => setCategory("all")}
              >
                All · {result.tokens.length}
              </button>
              {categories.map((c) => (
                <button
                  key={c}
                  className={category === c ? "selected" : ""}
                  aria-pressed={category === c}
                  onClick={() => setCategory(c)}
                >
                  {c.replace("_", " ")} ·{" "}
                  {result.tokens.filter((t) => t.category === c).length}
                </button>
              ))}
            </div>
            <div
              className="tokens"
              tabIndex={0}
              role="region"
              aria-label="Categorized abstract tokens"
            >
              {result.tokens.map((t, i) => (
                <span
                  key={i}
                  className={`token ${t.category} ${category !== "all" && category !== t.category ? "dimmed" : ""}`}
                  title={`${t.categories.length ? t.categories.join(", ") : t.category} · characters ${t.start}–${t.end}`}
                >
                  {masked && t.category === "content" ? "[content]" : t.text}
                </span>
              ))}
            </div>
            <button
              className="text-button feature-toggle"
              onClick={() => setAllFeatures(!allFeatures)}
              aria-expanded={allFeatures}
            >
              <SlidersHorizontal size={15} />
              {allFeatures ? "Hide" : "Inspect"} all {result.features.length}{" "}
              features & exact model arithmetic <ChevronDown size={14} />
            </button>
            {allFeatures && (
              <>
                <p className="formula">
                  sigmoid({fmt(result.intercept)} + Σ feature contributions) =
                  sigmoid({fmt(result.logit)}) = {pct(result.probability)}.
                  Intercept is the calibrated logit at the training feature
                  means; it differs from the prevalence prior.
                </p>
                <div
                  tabIndex={0}
                  role="region"
                  aria-label="Input feature contributions"
                  className="table-scroll"
                >
                  <table>
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Raw value</th>
                        <th>Standardized</th>
                        <th>Weight</th>
                        <th>Contribution</th>
                      </tr>
                    </thead>
                    <tbody>
                      {contributions.map((f) => (
                        <tr key={f.name}>
                          <td>
                            {f.label}
                            {f.unusual && (
                              <span className="tiny-note"> · unusual</span>
                            )}
                          </td>
                          <td>{fmt(f.value, 4)}</td>
                          <td>{fmt(f.z)}</td>
                          <td>{fmt(f.coefficient)}</td>
                          <td>{fmt(f.contribution)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </>
        ) : (
          <p className="representation-note">
            <Info size={16} />
            After analysis, inspect actual token categories and every weighted
            feature. Topic-word identities never enter the style model.
          </p>
        )}
      </section>
      <div className="method-strip">
        <FlaskConical size={20} />
        <div>
          <strong>Built to test a hypothesis.</strong>
          <p>
            {report
              ? `${number(report.splits.reduce((s, v) => s + v.n, 0))} historical abstracts. Six fields. A held-out 2021 test set. See what the experiment supports—and where it falls short.`
              : "The training notebook reports held-out performance, baselines, and transfer across disciplines."}
          </p>
        </div>
        <a href="/training">
          Open training notebook <ArrowRight size={16} />
        </a>
      </div>
    </>
  );
}

function Training({ report: r }: { report: Report }) {
  const [field, setField] = useState("all");
  const [curve, setCurve] = useState<"learning" | "convergence">("learning");
  const fieldNames = [
    ...new Map(r.cohorts.map((c) => [c.field_id, c.field])).entries(),
  ];
  const names: Record<number, string> = {
    13: "Biochemistry",
    17: "Computer science",
    20: "Economics",
    27: "Medicine",
    31: "Physics",
    33: "Social sciences",
  };
  return (
    <>
      <div className="page-intro">
        <div>
          <SectionLabel>TRAINING NOTEBOOK / RUN 001</SectionLabel>
          <h1>The experiment, exposed.</h1>
          <p>Real data, measured performance, and the limits of the signal.</p>
        </div>
        <button
          className="secondary"
          onClick={() => download(r, "shirabe-training-report.json")}
        >
          <ArrowDownToLine size={16} />
          Export run
        </button>
      </div>
      <div className="run-status">
        <span>
          <Check size={15} />
          Training completed
        </span>
        <code>{r.model_id}</code>
        <span>
          {new Date(r.trained_at).toLocaleDateString("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric",
            timeZone: "UTC",
          })}{" "}
          · {r.compute.device} · {fmt(r.compute.duration_seconds, 1)}s · $
          {r.compute.cloud_cost_usd} cloud spend
        </span>
      </div>
      <div className="stat-grid">
        <div>
          <span>TEST ROC AUC</span>
          <strong>{fmt(r.test.roc_auc)}</strong>
          <small>
            95% CI {r.test.auc_ci95?.map((n) => fmt(n)).join("–")} · chance
            0.500
          </small>
        </div>
        <div>
          <span>AVERAGE PRECISION</span>
          <strong>{fmt(r.test.average_precision)}</strong>
          <small>Test prevalence {pct(r.test.prevalence)}</small>
        </div>
        <div>
          <span>BRIER SCORE ↓</span>
          <strong>{fmt(r.test.brier)}</strong>
          <small>Prior baseline {fmt(r.comparisons[1].brier)}</small>
        </div>
        <div>
          <span>HELD-OUT PAPERS</span>
          <strong>{number(r.test.n)}</strong>
          <small>2021 · never used to fit or tune</small>
        </div>
      </div>
      <section className="finding">
        <Info size={20} />
        <div>
          <h2>A measurable signal. Subject independence is unproven.</h2>
          <p>
            Style outperforms length alone ({fmt(r.comparisons[2].roc_auc)}{" "}
            AUC), but remains associated with discipline. A separate classifier
            can identify the field from these features with{" "}
            {pct(r.discipline_probe.accuracy)} accuracy; the majority baseline
            is {pct(r.discipline_probe.majority_baseline)}. These are
            retrospective citation associations.
          </p>
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <SectionLabel number="01">EXPERIMENT DESIGN</SectionLabel>
            <h2>Time separates learning from evaluation.</h2>
          </div>
          <span className="tag">SEED {r.seed}</span>
        </div>
        <div className="split-grid">
          {r.splits.map((s, i) => (
            <div key={s.name} className={`split split-${i}`}>
              <span>{s.name.toUpperCase()}</span>
              <strong>{s.years}</strong>
              <div>{number(s.n)} papers</div>
              <small>{pct(s.positive_rate)} positive</small>
              <div className="split-line" />
            </div>
          ))}
        </div>
        <p className="panel-note">
          <strong>Target:</strong> {r.target}. Tied citation counts stay
          together, so the positive rate can exceed 25%. The model sees only
          English abstract style; the outcome uses citations measured at
          collection.
        </p>
      </section>
      <div className="chart-grid">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <SectionLabel number="02">DISCRIMINATION</SectionLabel>
              <h2>Separating citation outcomes</h2>
            </div>
            <span className="tag">2021 TEST</span>
          </div>
          <Chart
            title="ROC curve"
            xLabel="False positive rate"
            yLabel="True positive rate"
            diagonal
            series={[
              {
                name: `Style · AUC ${fmt(r.test.roc_auc)}`,
                color: "var(--accent)",
                points: r.roc.map((v) => [v.fpr, v.tpr]),
              },
            ]}
          />
        </section>
        <section className="panel">
          <div className="panel-heading">
            <div>
              <SectionLabel number="03">CALIBRATION</SectionLabel>
              <h2>Probability versus observation</h2>
            </div>
            <span className="tag">10 EQUAL-COUNT BINS</span>
          </div>
          <Chart
            title="Probability calibration"
            xLabel="Mean predicted probability"
            yLabel="Observed positive fraction"
            diagonal
            series={[
              {
                name: "Held-out observations",
                color: "var(--accent)",
                points: r.reliability.map((v) => [v.predicted, v.observed]),
              },
            ]}
          />
          <p className="chart-note">
            Sigmoid fitted on 2020 only. Test log loss: {fmt(r.test.log_loss)}.
            Calibration is approximate, not a guarantee for new papers.
          </p>
        </section>
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <SectionLabel number="04">BASELINE COMPARISONS</SectionLabel>
            <h2>What else explains the signal?</h2>
          </div>
          <span className="tag">SAME TEST PAPERS</span>
        </div>
        <div
          tabIndex={0}
          role="region"
          aria-label="Model baseline comparisons"
          className="table-scroll"
        >
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>ROC AUC ↑</th>
                <th>Avg. precision ↑</th>
                <th>Brier ↓</th>
                <th>Log loss ↓</th>
              </tr>
            </thead>
            <tbody>
              {r.comparisons.map((m, i) => (
                <tr className={i === 0 ? "deployed" : ""} key={m.name}>
                  <td>
                    <strong>{m.name}</strong>
                    <small>{m.role}</small>
                  </td>
                  <td>
                    <div className="auc-cell">
                      <span>{fmt(m.roc_auc)}</span>
                      <i style={{ width: `${(m.roc_auc - 0.4) * 150}px` }} />
                    </div>
                  </td>
                  <td>{fmt(m.average_precision)}</td>
                  <td>{fmt(m.brier)}</td>
                  <td>{fmt(m.log_loss)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="panel-note">
          The linear model was chosen in advance for exact explanations; the
          boosted comparison scores higher. Shuffled-label negative control:
          mean AUC {fmt(r.negative_control.mean_auc)} over{" "}
          {r.negative_control.runs} runs (range{" "}
          {fmt(r.negative_control.min_auc)}–{fmt(r.negative_control.max_auc)}).
        </p>
      </section>
      <div className="chart-grid">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <SectionLabel number="05">TRAINING PROCESS</SectionLabel>
              <h2>Learning from more papers</h2>
            </div>
          </div>
          <div className="segmented">
            <button
              className={curve === "learning" ? "selected" : ""}
              aria-pressed={curve === "learning"}
              onClick={() => setCurve("learning")}
            >
              Sample size
            </button>
            <button
              className={curve === "convergence" ? "selected" : ""}
              aria-pressed={curve === "convergence"}
              onClick={() => setCurve("convergence")}
            >
              Optimization
            </button>
          </div>
          {curve === "learning" ? (
            <Chart
              title="Learning curve"
              xLabel="Training papers"
              yLabel="ROC AUC"
              domain={[0.5, 0.85]}
              xDomain={[0, r.splits[0].n]}
              series={[
                {
                  name: "Train",
                  color: "var(--amber)",
                  points: r.learning_curve.map((v) => [v.n, v.train_auc]),
                },
                {
                  name: "Validation · 2019",
                  color: "var(--accent)",
                  points: r.learning_curve.map((v) => [v.n, v.validation_auc]),
                },
              ]}
            />
          ) : (
            <Chart
              title="Optimization convergence"
              xLabel="Solver iteration budget"
              yLabel="Log loss"
              domain={[0.45, 0.65]}
              xDomain={[0, 128]}
              series={[
                {
                  name: "Train",
                  color: "var(--amber)",
                  points: r.convergence.map((v) => [
                    v.iteration_budget,
                    v.train_loss,
                  ]),
                },
                {
                  name: "Validation · 2019",
                  color: "var(--accent)",
                  points: r.convergence.map((v) => [
                    v.iteration_budget,
                    v.validation_loss,
                  ]),
                },
              ]}
            />
          )}
          <p className="chart-note">
            L2 logistic regression · C = {r.selected_c} ·{" "}
            {r.optimizer_iterations} solver iterations. Curves are measured
            refits, not simulated epochs.
          </p>
        </section>
        <section className="panel">
          <div className="panel-heading">
            <div>
              <SectionLabel number="06">CROSS-FIELD TRANSFER</SectionLabel>
              <h2>Test a discipline never seen</h2>
            </div>
          </div>
          <div className="transfer-list">
            {r.field_transfer.map((f) => (
              <div key={f.field_id}>
                <div>
                  <strong>{names[f.field_id]}</strong>
                  <span>
                    {fmt(f.roc_auc)}{" "}
                    <small>
                      [{f.auc_ci95?.map((v) => fmt(v, 2)).join(", ")}]
                    </small>
                  </span>
                </div>
                <div className="transfer-bar">
                  <span style={{ width: `${f.roc_auc * 100}%` }} />
                  <i style={{ left: "50%" }} />
                </div>
              </div>
            ))}
          </div>
          <p className="panel-note">
            Each field is excluded from training, tuning, and calibration, then
            evaluated on its 2021 papers. Brackets show 95% bootstrap intervals.
            Dashed marker: chance.
          </p>
        </section>
      </div>
      <section className="panel">
        <div className="panel-heading">
          <div>
            <SectionLabel number="07">DATA & PROVENANCE</SectionLabel>
            <h2>
              {number(r.splits.reduce((a, b) => a + b.n, 0))} abstracts. Every
              cohort accounted for.
            </h2>
          </div>
          <label className="select-label">
            Field
            <select
              aria-label="Field"
              value={field}
              onChange={(e) => setField(e.target.value)}
            >
              <option value="all">All disciplines</option>
              {fieldNames.map(([id, name]) => (
                <option value={id} key={id}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div
          tabIndex={0}
          role="region"
          aria-label="Field and year cohorts"
          className="table-scroll cohort-table"
        >
          <table>
            <thead>
              <tr>
                <th>Field</th>
                <th>Year</th>
                <th>Papers</th>
                <th>Citation cutoff ≥</th>
                <th>Positive rate</th>
              </tr>
            </thead>
            <tbody>
              {r.cohorts
                .filter((c) => field === "all" || String(c.field_id) === field)
                .map((c) => (
                  <tr key={`${c.field_id}-${c.year}`}>
                    <td>{names[c.field_id]}</td>
                    <td>{c.year}</td>
                    <td>{number(c.n)}</td>
                    <td>{number(c.threshold)}</td>
                    <td>{pct(c.prevalence)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <div className="provenance-footer">
          <p>
            OpenAlex · seeded random samples · 400 requested per field/year
            <br />
            English articles with abstracts; 80–800 words; retractions excluded;
            IDs, DOI, title and abstract deduplicated.
          </p>
          <a href="/api/provenance" target="_blank" rel="noreferrer">
            View collection manifest <ExternalLink size={14} />
          </a>
        </div>
      </section>
      <details className="panel details">
        <summary>
          <SectionLabel number="08">
            MODEL WEIGHTS & REPRODUCIBILITY
          </SectionLabel>
          <ChevronDown size={18} />
        </summary>
        <p>
          Portable JSON weights. No GPU, external inference service, or topic
          vocabulary. Standardize using the training split, cap values at ±5σ,
          apply the weighted logit, then the 2020 sigmoid calibration.
        </p>
        <div
          tabIndex={0}
          role="region"
          aria-label="Model coefficient table"
          className="table-scroll"
        >
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Calibrated coefficient</th>
                <th>Training mean</th>
                <th>Training scale</th>
              </tr>
            </thead>
            <tbody>
              {[...r.features]
                .sort(
                  (a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient),
                )
                .map((f) => (
                  <tr key={f.name}>
                    <td>{f.label}</td>
                    <td>{fmt(f.coefficient, 4)}</td>
                    <td>{fmt(f.mean, 4)}</td>
                    <td>{fmt(f.scale, 4)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <p className="hash">Dataset SHA-256: {r.dataset_sha256}</p>
        <p>
          Python {r.compute.python} · scikit-learn {r.compute.sklearn} · seed{" "}
          {r.seed}
        </p>
      </details>
      <section className="limitations">
        <SectionLabel number="09">BOUNDARIES OF THE EXPERIMENT</SectionLabel>
        <h2>What this model cannot establish.</h2>
        <ol>
          {r.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ol>
        <p>
          Research context:{" "}
          <a
            href="https://www.nature.com/articles/s44271-025-00293-8"
            target="_blank"
            rel="noreferrer"
          >
            Promotional language and scientific attention{" "}
            <ExternalLink size={12} />
          </a>{" "}
          ·{" "}
          <a
            href="https://scikit-learn.org/stable/modules/calibration.html"
            target="_blank"
            rel="noreferrer"
          >
            Probability calibration <ExternalLink size={12} />
          </a>
        </p>
      </section>
    </>
  );
}

function App() {
  const [page, setPage] = useState(
    location.pathname === "/training" ? "training" : "analysis",
  );
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [modelReady, setModelReady] = useState(false);
  async function load() {
    setLoading(true);
    setError("");
    try {
      const [run, health] = await Promise.all([
        api<Report>("/api/report"),
        api<{ status: string; model_id: string }>("/api/health"),
      ]);
      setReport(run);
      setModelReady(
        health.status === "ready" && health.model_id === run.model_id,
      );
      if (health.model_id !== run.model_id)
        throw new Error(
          "The server has an older model loaded. Restart it to load the latest training run.",
        );
    } catch (e) {
      setModelReady(false);
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    void load();
    const listener = () =>
      setPage(location.pathname === "/training" ? "training" : "analysis");
    window.addEventListener("popstate", listener);
    return () => window.removeEventListener("popstate", listener);
  }, []);
  function navigate(e: React.MouseEvent<HTMLAnchorElement>, next: string) {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    history.pushState({}, "", next === "training" ? "/training" : "/");
    setPage(next);
    window.scrollTo(0, 0);
    requestAnimationFrame(() => {
      const heading = document.querySelector<HTMLElement>(
        next === "training"
          ? "main > .page-intro h1"
          : "main > div:not([hidden]) h1",
      );
      heading?.setAttribute("tabindex", "-1");
      heading?.focus({ preventScroll: true });
    });
  }
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header>
        <div className="header-inner">
          <a
            className="brand"
            href="/"
            onClick={(e) => navigate(e, "analysis")}
            aria-label="Shirabe home"
          >
            <span className="brand-symbol" aria-hidden="true">
              <i />
              <i />
              <i />
              <i />
            </span>
            <strong>
              shirabe<span>調べ</span>
            </strong>
          </a>
          <span className="header-descriptor">RESEARCH SIGNAL LABORATORY</span>
          <div className="header-status">
            <span className={`status-dot ${!modelReady ? "offline" : ""}`} />
            {modelReady
              ? "MODEL READY"
              : loading
                ? "CONNECTING"
                : "MODEL UNAVAILABLE"}
          </div>
        </div>
      </header>
      <div className="nav-wrap">
        <nav aria-label="Main navigation">
          <a
            href="/"
            className={page === "analysis" ? "active" : ""}
            aria-current={page === "analysis" ? "page" : undefined}
            onClick={(e) => navigate(e, "analysis")}
          >
            <ScanText size={17} />
            Paper analysis
          </a>
          <a
            href="/training"
            className={page === "training" ? "active" : ""}
            aria-current={page === "training" ? "page" : undefined}
            onClick={(e) => navigate(e, "training")}
          >
            <FlaskConical size={17} />
            Training notebook
          </a>
          <span className="nav-version">
            STYLE MODEL / v0.1 <ChevronRight size={12} />
          </span>
        </nav>
      </div>
      <main id="main">
        {error && (
          <div className="service-error" role="alert">
            <Info size={18} />
            <span>{error}</span>
            <button className="secondary" onClick={() => void load()}>
              Retry connection
            </button>
          </div>
        )}
        <div hidden={page !== "analysis"}>
          <Analysis report={report} />
        </div>
        {page === "training" &&
          (report ? (
            <Training report={report} />
          ) : (
            <Empty
              title={
                loading
                  ? "Loading training run…"
                  : "Training report unavailable"
              }
            >
              {loading
                ? "Reading measured results from the local model."
                : "Check the API connection and retry above."}
            </Empty>
          ))}
      </main>
      <footer>
        <div>
          <span className="brand-small">shirabe</span>
          <span>An open experiment in the language of research.</span>
          <a
            href="https://github.com/Hone-Systems/shirabe"
            target="_blank"
            rel="noreferrer"
          >
            Source · AGPL-3.0
          </a>
        </div>
        <span>
          LOCAL INFERENCE <span className="footer-separator">/</span> CITATION
          ATTENTION ≠ SCIENTIFIC TRUTH
        </span>
      </footer>
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
