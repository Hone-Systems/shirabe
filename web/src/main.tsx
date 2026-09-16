import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowUpRight,
  ArrowRight,
  Scan,
  X,
  Pause,
  Play,
  FileText,
  Plus,
  ArrowDownToLine,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  CircleHelp,
  RotateCcw,
  LoaderCircle,
} from "lucide-react";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "./styles.css";
import type { Prediction, Report } from "./types";

const SAMPLE =
  "We examined whether the timing of feedback affects performance on a repeated learning task. Participants completed a series of trials under three feedback schedules, with assignment randomized before the first session. We measured accuracy and response time at baseline and after each training block. Immediate feedback was associated with higher accuracy during training, but the difference was smaller at the delayed assessment. The estimates were similar after accounting for baseline performance and the number of completed trials. However, the sample was limited to volunteers from a single institution, and the study did not assess transfer to other tasks. These results suggest that feedback timing may influence short-term learning, while its effect on retention remains uncertain. Further work with a larger and more varied sample is needed to estimate the conditions under which the observed differences persist.";
const f = (v: number, d = 3) => v.toFixed(d);
const pct = (v: number) => `${f(v * 100, 1)}%`;
const num = (v: number) => v.toLocaleString("en-US");
const fieldNames: Record<number, string> = {
  13: "Biochemistry",
  17: "Computer science",
  20: "Economics",
  27: "Medicine",
  31: "Physics",
  33: "Social sciences",
};
async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const r = await fetch(url, options);
  const d = await r.json().catch(() => null);
  if (!r.ok)
    throw new Error(
      typeof d?.detail === "string"
        ? d.detail
        : `Connection failed (${r.status}). Try again.`,
    );
  return d;
}
function save(data: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function Label({ children }: { children: React.ReactNode }) {
  return <span className="micro">{children}</span>;
}
function Modal({
  title,
  children,
  onClose,
}: {
  title: string;
  children: React.ReactNode;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = ref.current;
    const previous = document.activeElement as HTMLElement | null;
    el?.showModal();
    el?.querySelector<HTMLTextAreaElement>("textarea")?.focus();
    return () => {
      el?.close();
      previous?.focus();
    };
  }, []);
  return (
    <dialog ref={ref} className="modal" aria-label={title} onCancel={onClose}>
      <div className="modal-head">
        <h2>{title}</h2>
        <button className="icon" onClick={onClose} aria-label="Close dialog">
          <X size={20} />
        </button>
      </div>
      <div className="modal-body">{children}</div>
    </dialog>
  );
}
const point = (angle: number, r: number) => ({
  x: 320 + Math.cos(angle) * r,
  y: 320 + Math.sin(angle) * r,
});
function FeatureField({
  report,
  result,
  busy,
  paused,
  selected,
  onSelect,
}: {
  report: Report | null;
  result: Prediction | null;
  busy: boolean;
  paused: boolean;
  selected: number;
  onSelect: (n: number) => void;
}) {
  const features = result
    ? result.features.map((v) => ({
        name: v.name,
        label: v.label,
        value: v.contribution,
      }))
    : (report?.features.map((v) => ({
        name: v.name,
        label: v.label,
        value: v.coefficient,
      })) ?? []);
  const maximum = Math.max(0.01, ...features.map((v) => Math.abs(v.value)));
  const active = features[selected];
  return (
    <div
      className={`feature-field ${paused ? "paused" : ""} ${busy ? "scanning" : ""}`}
    >
      <svg
        className="constellation"
        viewBox="0 0 640 640"
        role="img"
        aria-label={
          result
            ? "Radial plot of actual signed feature contributions to this estimate"
            : "Radial plot of the trained model coefficients; awaiting an abstract"
        }
      >
        <defs>
          <radialGradient id="field-glow">
            <stop offset="0" stopColor="#65dbc2" stopOpacity=".08" />
            <stop offset=".8" stopColor="#65dbc2" stopOpacity="0" />
          </radialGradient>
        </defs>
        <circle cx="320" cy="320" r="310" fill="url(#field-glow)" />
        <g className="guide-ring">
          <circle cx="320" cy="320" r="278" />
          <circle cx="320" cy="320" r="226" />
          {Array.from({ length: 120 }, (_, i) => {
            const a = (i / 120) * Math.PI * 2;
            const p = point(a, 278),
              q = point(a, i % 10 === 0 ? 287 : 281);
            return <line key={i} x1={p.x} y1={p.y} x2={q.x} y2={q.y} />;
          })}
        </g>
        <g className="orbit-dashes">
          <circle
            cx="320"
            cy="320"
            r="249"
            fill="none"
            stroke="var(--mint)"
            strokeOpacity=".28"
            strokeDasharray="2 30 70 300"
          />
        </g>
        <path
          d="M320 21V44M320 596V619M21 320H44M596 320H619"
          stroke="var(--muted)"
          strokeWidth="1"
        />
        <text className="plot-label" x="320" y="14" textAnchor="middle">
          {features.length} / LEARNED CHANNELS
        </text>
        <text className="plot-label" x="320" y="630" textAnchor="middle">
          {result ? "SIGNED CONTRIBUTIONS" : "TRAINED COEFFICIENTS"}
        </text>
        {features.map((feature, i) => {
          const angle = -Math.PI / 2 + (i / features.length) * Math.PI * 2;
          const start = point(angle, 155),
            node = point(angle, 222),
            end = point(angle, 229 + (Math.abs(feature.value) / maximum) * 39);
          const mid = point(angle + 0.09, 189);
          const color = feature.value >= 0 ? "var(--mint)" : "var(--amber)";
          const d = `M${node.x} ${node.y} Q${mid.x} ${mid.y} ${start.x} ${start.y}`;
          return (
            <g
              key={feature.name}
              className={`feature-ray ${selected === i ? "selected" : ""}`}
              onMouseEnter={() => onSelect(i)}
            >
              <path
                d={d}
                fill="none"
                stroke={color}
                opacity={selected === i ? 0.85 : 0.12}
              />
              <path
                className="signal-particle"
                d={d}
                fill="none"
                stroke={color}
                strokeWidth="2"
                strokeDasharray="2 140"
                style={{
                  animationDelay: `${-i * 0.17}s`,
                  animationDuration: `${3 + (i % 5) * 0.4}s`,
                }}
              />
              <line
                x1={node.x}
                y1={node.y}
                x2={end.x}
                y2={end.y}
                stroke={color}
                strokeWidth={selected === i ? 4 : 2}
                opacity={selected === i ? 1 : 0.65}
              />
              <circle
                cx={node.x}
                cy={node.y}
                r={selected === i ? 4 : 2}
                fill={color}
              />
              <title>
                {feature.label}: {f(feature.value)}
              </title>
            </g>
          );
        })}
        <circle
          className="inner-orbit"
          cx="320"
          cy="320"
          r="145"
          fill="none"
          stroke="var(--line)"
          strokeDasharray="1 6"
        />
        <circle cx="320" cy="320" r="130" fill="var(--bg)" stroke="#253b39" />
        <circle
          cx="320"
          cy="320"
          r="130"
          fill="none"
          stroke="var(--mint)"
          strokeWidth="2"
          pathLength="100"
          strokeDasharray={`${result ? result.probability * 100 : 0} 100`}
          transform="rotate(-90 320 320)"
          className="score-arc"
        />
      </svg>
      <div className="core" aria-live="polite">
        <Label>
          {busy
            ? "READING SIGNAL"
            : result
              ? "CITATION ATTENTION"
              : "AWAITING PAPER"}
        </Label>
        <div className="score" key={result?.probability ?? "empty"}>
          {result ? f(result.probability * 100, 1) : "—"}
          {result && <span>%</span>}
        </div>
        <span className="core-caption">
          {busy
            ? "Reading token context"
            : result
              ? "Estimated probability"
              : "A signal in the language"}
        </span>
        {result && <span className="prior">prior {pct(result.baseline)}</span>}
      </div>
      <div className="feature-readout">
        <button
          className="icon"
          aria-label="Previous feature"
          disabled={!features.length}
          onClick={() =>
            onSelect((selected + features.length - 1) % features.length)
          }
        >
          <ChevronLeft size={15} />
        </button>
        <div aria-live="polite" aria-atomic="true">
          <span>{active?.label ?? "Waiting for model"}</span>
          <code className={active && active.value < 0 ? "amber" : "mint"}>
            {active ? `${active.value >= 0 ? "+" : ""}${f(active.value)}` : "—"}
          </code>
        </div>
        <button
          className="icon"
          aria-label="Next feature"
          disabled={!features.length}
          onClick={() => onSelect((selected + 1) % features.length)}
        >
          <ChevronRight size={15} />
        </button>
      </div>
    </div>
  );
}
function MiniCurve({
  points,
  label,
  diagonal = false,
}: {
  points: [number, number][];
  label: string;
  diagonal?: boolean;
}) {
  return (
    <svg
      className="mini-curve"
      viewBox="0 0 200 100"
      role="img"
      aria-label={label}
    >
      <path d="M0 25H200M0 50H200M0 75H200" stroke="var(--line)" fill="none" />
      {diagonal && (
        <path d="M0 100L200 0" stroke="var(--muted)" strokeDasharray="3 5" />
      )}
      <polyline
        points={points.map(([x, y]) => `${x * 200},${100 - y * 100}`).join(" ")}
        fill="none"
        stroke="var(--mint)"
        strokeWidth="2"
      />
    </svg>
  );
}
function TrainingField({ report: r }: { report: Report }) {
  return (
    <div className="training-field">
      <div className="training-title">
        <Label>UNSEEN PAPERS / 2021</Label>
        <strong>{f(r.test.roc_auc)}</strong>
        <span>
          ROC AUC <i /> 95% CI {r.test.auc_ci95?.map((v) => f(v)).join("–")}
        </span>
      </div>
      <svg
        viewBox="0 0 620 440"
        className="roc-field"
        role="img"
        aria-label="Held-out ROC curve. False positive rate on x, true positive rate on y."
      >
        <defs>
          <linearGradient id="roc-fill" x1="0" y1="0" x2="0" y2="1">
            <stop stopColor="#79e4c9" stopOpacity=".16" />
            <stop offset="1" stopColor="#79e4c9" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <path
              d={`M70 ${370 - v * 300}H570M${70 + v * 500} 70V370`}
              stroke="var(--line)"
              strokeDasharray="2 6"
            />
            <text x="54" y={375 - v * 300} textAnchor="end">
              {f(v, 2)}
            </text>
            <text x={70 + v * 500} y="399" textAnchor="middle">
              {f(v, 2)}
            </text>
          </g>
        ))}
        <path d="M70 370L570 70" stroke="var(--muted)" strokeDasharray="5 9" />
        <polygon
          points={`70,370 ${r.roc.map((p) => `${70 + p.fpr * 500},${370 - p.tpr * 300}`).join(" ")} 570,370`}
          fill="url(#roc-fill)"
        />
        <polyline
          className="roc-trace"
          pathLength="1"
          points={r.roc
            .map((p) => `${70 + p.fpr * 500},${370 - p.tpr * 300}`)
            .join(" ")}
          fill="none"
          stroke="var(--mint)"
          strokeWidth="3"
        />
        <text x="320" y="433" textAnchor="middle">
          FALSE POSITIVE RATE
        </text>
        <text x="15" y="220" transform="rotate(-90 15 220)" textAnchor="middle">
          TRUE POSITIVE RATE
        </text>
      </svg>
      <div className="plot-legend">
        <span>
          <i />
          BERT-Mini
        </span>
        <span>
          <i className="dashed" />
          Chance · 0.500
        </span>
      </div>
    </div>
  );
}
function FullReport({ r }: { r: Report }) {
  const [field, setField] = useState("all");
  return (
    <div className="full-report">
      <p>
        {r.target}. English abstracts only. Citations measure attention, not
        scientific validity.
      </p>
      <div className="report-stats">
        {r.splits.map((s) => (
          <div key={s.name}>
            <Label>
              {s.name} · {s.years}
            </Label>
            <strong>{num(s.n)}</strong>
          </div>
        ))}
      </div>
      <h3>Model comparison</h3>
      <div
        className="table-wrap"
        tabIndex={0}
        role="region"
        aria-label="Model comparisons"
      >
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>AUC</th>
              <th>AP</th>
              <th>Brier</th>
              <th>Log loss</th>
            </tr>
          </thead>
          <tbody>
            {r.comparisons.map((v) => (
              <tr key={v.name}>
                <td>
                  {v.name}
                  <small>{v.role}</small>
                </td>
                <td>{f(v.roc_auc)}</td>
                <td>{f(v.average_precision)}</td>
                <td>{f(v.brier)}</td>
                <td>{f(v.log_loss)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3>Learning & optimization</h3>
      {r.epoch_history && (
        <>
          <p>
            BERT-Mini · 4 layers · 4 heads · 256 hidden dimensions ·{" "}
            {num(r.transformer?.parameter_count ?? 0)} parameters. Selected
            epoch {r.best_epoch} by validation loss.
          </p>
          <div
            className="table-wrap"
            tabIndex={0}
            role="region"
            aria-label="Transformer training history"
          >
            <table>
              <thead>
                <tr>
                  <th>Epoch</th>
                  <th>Train loss</th>
                  <th>Validation loss</th>
                  <th>Validation AUC</th>
                </tr>
              </thead>
              <tbody>
                {r.epoch_history.map((v) => (
                  <tr key={v.epoch}>
                    <td>{v.epoch}</td>
                    <td>{f(v.train_loss)}</td>
                    <td>{f(v.validation_loss)}</td>
                    <td>{f(v.validation_auc)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {r.learning_curve && (
        <>
          <div
            className="table-wrap"
            tabIndex={0}
            role="region"
            aria-label="Learning history"
          >
            <table>
              <thead>
                <tr>
                  <th>Training papers</th>
                  <th>Training AUC</th>
                  <th>Validation AUC</th>
                </tr>
              </thead>
              <tbody>
                {r.learning_curve?.map((v) => (
                  <tr key={v.n}>
                    <td>{num(v.n)}</td>
                    <td>{f(v.train_auc)}</td>
                    <td>{f(v.validation_auc)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            C = {r.selected_c} · {r.optimizer_iterations} optimizer iterations ·{" "}
            {f(r.compute.duration_seconds, 1)}s on CPU · $0 cloud spend.
          </p>
        </>
      )}
      <details>
        <summary>Optimization trace & calibration bins</summary>
        {r.convergence && (
          <>
            <div
              className="table-wrap"
              tabIndex={0}
              role="region"
              aria-label="Optimization history"
            >
              <table>
                <thead>
                  <tr>
                    <th>Iteration budget</th>
                    <th>Used</th>
                    <th>Train loss</th>
                    <th>Validation loss</th>
                  </tr>
                </thead>
                <tbody>
                  {r.convergence?.map((v) => (
                    <tr key={v.iteration_budget}>
                      <td>{v.iteration_budget}</td>
                      <td>{v.iterations_used}</td>
                      <td>{f(v.train_loss)}</td>
                      <td>{f(v.validation_loss)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
        <div
          className="table-wrap"
          tabIndex={0}
          role="region"
          aria-label="Calibration bins"
        >
          <table>
            <thead>
              <tr>
                <th>Predicted</th>
                <th>Observed</th>
                <th>Papers</th>
              </tr>
            </thead>
            <tbody>
              {r.reliability.map((v, i) => (
                <tr key={i}>
                  <td>{pct(v.predicted)}</td>
                  <td>{pct(v.observed)}</td>
                  <td>{v.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <h3>Cross-field transfer</h3>
      <div
        className="table-wrap"
        tabIndex={0}
        role="region"
        aria-label="Cross-field results"
      >
        <table>
          <thead>
            <tr>
              <th>Excluded field</th>
              <th>Test AUC</th>
              <th>95% interval</th>
            </tr>
          </thead>
          <tbody>
            {r.field_transfer.map((v) => (
              <tr key={v.field_id}>
                <td>{v.field}</td>
                <td>{f(v.roc_auc)}</td>
                <td>{v.auc_ci95?.map((x) => f(x)).join("–")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        Discipline remains detectable: {pct(r.discipline_probe.accuracy)}{" "}
        accuracy vs {pct(r.discipline_probe.majority_baseline)} baseline.
        {r.shuffled_token_test
          ? `Shuffled-token AUC: ${f(r.shuffled_token_test.roc_auc)}.`
          : `Shuffled-label mean AUC: ${f(r.negative_control?.mean_auc ?? 0)}.`}
      </p>
      <h3>Cohorts</h3>
      <label className="field-select">
        Field{" "}
        <select
          aria-label="Field"
          value={field}
          onChange={(e) => setField(e.target.value)}
        >
          <option value="all">All fields</option>
          {Object.entries(fieldNames).map(([id, name]) => (
            <option key={id} value={id}>
              {name}
            </option>
          ))}
        </select>
      </label>
      <div
        className="table-wrap"
        tabIndex={0}
        role="region"
        aria-label="Cohort data"
      >
        <table>
          <thead>
            <tr>
              <th>Field</th>
              <th>Year</th>
              <th>Papers</th>
              <th>Citation cutoff</th>
            </tr>
          </thead>
          <tbody>
            {r.cohorts
              .filter((c) => field === "all" || String(c.field_id) === field)
              .map((c) => (
                <tr key={`${c.field_id}-${c.year}`}>
                  <td>{fieldNames[c.field_id]}</td>
                  <td>{c.year}</td>
                  <td>{c.n}</td>
                  <td>{c.threshold}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <details>
        <summary>All {r.feature_count} head weights</summary>
        <div
          className="table-wrap"
          tabIndex={0}
          role="region"
          aria-label="Model weights"
        >
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Weight</th>
                <th>Mean</th>
                <th>Scale</th>
              </tr>
            </thead>
            <tbody>
              {r.features.map((v) => (
                <tr key={v.name}>
                  <td>{v.label}</td>
                  <td>{f(v.coefficient)}</td>
                  <td>{f(v.mean)}</td>
                  <td>{f(v.scale)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <h3>Limits</h3>
      <ul>
        {r.limitations.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
      <p className="hash">
        {r.model_id}
        <br />
        Dataset SHA-256: {r.dataset_sha256}
      </p>
      <a href="/api/provenance" target="_blank" rel="noreferrer">
        Collection manifest <ArrowUpRight size={14} />
      </a>
    </div>
  );
}
function App() {
  const [mode, setMode] = useState(
    location.pathname === "/training" ? "training" : "analysis",
  );
  const [report, setReport] = useState<Report | null>(null);
  const [ready, setReady] = useState(false);
  const [connectionError, setConnectionError] = useState("");
  const [text, setText] = useState("");
  const [source, setSource] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState<"extract" | "predict" | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [dialog, setDialog] = useState<
    "editor" | "features" | "report" | "about" | null
  >(null);
  const [paused, setPaused] = useState(false);
  const [selected, setSelected] = useState(0);
  const [masked, setMasked] = useState(false);
  const [drag, setDrag] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const request = useRef<AbortController | null>(null);
  const words = (text.match(/[A-Za-z]+(?:[-’'][A-Za-z]+)*/g) || []).length;
  async function connect() {
    setConnectionError("");
    try {
      const [r, h] = await Promise.all([
        api<Report>("/api/report"),
        api<{ status: string; model_id: string }>("/api/health"),
      ]);
      setReport(r);
      if (h.model_id !== r.model_id)
        throw new Error("Restart the server to load the latest model.");
      setReady(h.status === "ready");
    } catch (e) {
      setReady(false);
      setConnectionError((e as Error).message);
    }
  }
  useEffect(() => {
    void connect();
    const onPop = () =>
      setMode(location.pathname === "/training" ? "training" : "analysis");
    window.addEventListener("popstate", onPop);
    return () => {
      window.removeEventListener("popstate", onPop);
      request.current?.abort();
    };
  }, []);
  function navigate(next: string) {
    setMode(next);
    history.pushState({}, "", next === "training" ? "/training" : "/");
  }
  function edit(value: string) {
    request.current?.abort();
    setText(value);
    setResult(null);
    setError("");
    setNotice("");
    setSource("Pasted abstract");
    setSelected(0);
    setBusy(null);
  }
  async function analyze(value = text) {
    const count = (value.match(/[A-Za-z]+(?:[-’'][A-Za-z]+)*/g) || []).length;
    if (count < 80 || count > 800) {
      setError(`Use 80–800 English words. Current count: ${count}.`);
      return;
    }
    request.current?.abort();
    const control = new AbortController();
    request.current = control;
    setBusy("predict");
    setError("");
    setResult(null);
    setDialog(null);
    try {
      const p = await api<Prediction>("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: value }),
        signal: control.signal,
      });
      setResult(p);
      setSelected(
        p.features.reduce(
          (best, v, i, all) =>
            Math.abs(v.contribution) > Math.abs(all[best].contribution)
              ? i
              : best,
          0,
        ),
      );
    } catch (e) {
      if (!control.signal.aborted) setError((e as Error).message);
    } finally {
      if (request.current === control) setBusy(null);
    }
  }
  async function upload(file: File) {
    setError("");
    setNotice("");
    setResult(null);
    setSelected(0);
    if (file.size > 10 * 1024 * 1024) {
      setError("Maximum file size: 10 MB.");
      return;
    }
    if (!/\.(pdf|txt)$/i.test(file.name)) {
      setError("Choose a PDF or UTF-8 TXT file.");
      return;
    }
    request.current?.abort();
    const control = new AbortController();
    request.current = control;
    setBusy("extract");
    const form = new FormData();
    form.append("file", file);
    try {
      const d = await api<{ text: string; filename: string; notice: string }>(
        "/api/extract",
        { method: "POST", body: form, signal: control.signal },
      );
      setText(d.text);
      setSource(d.filename);
      setNotice(d.notice);
      setDialog("editor");
    } catch (e) {
      if (!control.signal.aborted) setError((e as Error).message);
    } finally {
      if (request.current === control) setBusy(null);
    }
  }
  const contributions = result
    ? [...result.features]
        .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
        .slice(0, 5)
    : [];
  const maxContribution = Math.max(
    0.01,
    ...contributions.map((v) => Math.abs(v.contribution)),
  );
  const total = report?.splits.reduce((a, b) => a + b.n, 0) ?? 0;
  return (
    <div className={`instrument mode-${mode} ${paused ? "motion-off" : ""}`}>
      <a href="#workspace" className="skip-link">
        Skip to instrument
      </a>
      <header className="topbar">
        <a
          href="/"
          className="brand"
          aria-label="Shirabe home"
          onClick={(e) => {
            if (!e.metaKey && !e.ctrlKey) {
              e.preventDefault();
              navigate("analysis");
            }
          }}
        >
          <span className="brand-glyph">
            <i />
            <i />
            <i />
            <i />
          </span>
          shirabe<span className="jp">調べ</span>
        </a>
        <nav aria-label="Workspace">
          <button
            onClick={() => navigate("analysis")}
            aria-pressed={mode === "analysis"}
          >
            Analyze
          </button>
          <button
            onClick={() => navigate("training")}
            aria-pressed={mode === "training"}
          >
            Training
          </button>
        </nav>
        <div className="top-right">
          <span className={`connection ${ready ? "online" : ""}`}>
            <i />
            {ready ? "LOCAL MODEL" : "CONNECTING"}
          </span>
          <button
            className="icon"
            onClick={() => setPaused(!paused)}
            aria-label={paused ? "Resume animation" : "Pause animation"}
            aria-pressed={paused}
          >
            {paused ? <Play size={16} /> : <Pause size={16} />}
          </button>
          <button
            className="icon"
            onClick={() => setDialog("about")}
            aria-label="About this experiment"
          >
            <CircleHelp size={17} />
          </button>
        </div>
      </header>
      <main id="workspace" className="workspace">
        <div className="screen-heading">
          <div>
            <Label>
              {source === "Illustrative example" && mode === "analysis"
                ? "SYNTHETIC EXAMPLE / LIVE MODEL"
                : "LANGUAGE / IMPACT / EXPERIMENT 001"}
            </Label>
            <h1>
              {mode === "analysis"
                ? "Every paper has a signal."
                : "The signal holds up. Within limits."}
            </h1>
          </div>
          <div className="run-id">
            <span>{report?.model_id ?? "ESTABLISHING CONNECTION"}</span>
            <span>
              CPU INFERENCE <i /> NO CLOUD
            </span>
          </div>
        </div>
        {connectionError && (
          <div className="connection-error" role="alert">
            {connectionError}
            <button onClick={() => void connect()}>Reconnect</button>
          </div>
        )}
        <div className="stage-grid">
          <aside
            className="left-rail"
            aria-label={
              mode === "analysis" ? "Paper source" : "Training cohorts"
            }
          >
            {mode === "analysis" ? (
              <>
                <div className="rail-title">
                  <Label>01 / SOURCE</Label>
                  <span className="tiny-index">EN</span>
                </div>
                <div
                  className={`paper-slot ${drag ? "dragging" : ""}`}
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
                  <div className="paper-corners" />
                  <div className="document-symbol">
                    <FileText size={34} strokeWidth={1} />
                    {busy === "extract" && <div className="scanline" />}
                  </div>
                  <h2>{text ? source : "A paper. A possibility."}</h2>
                  {text ? (
                    <button
                      className="paper-preview"
                      onClick={() => setDialog("editor")}
                      aria-label="Edit abstract"
                    >
                      {text}
                    </button>
                  ) : (
                    <p>Drop your PDF or TXT</p>
                  )}
                  <button
                    className="upload-action"
                    onClick={() => fileInput.current?.click()}
                    disabled={!!busy}
                  >
                    {busy === "extract" ? (
                      <LoaderCircle size={15} className="spin" />
                    ) : (
                      <Plus size={15} />
                    )}{" "}
                    {text ? "Replace paper" : "Choose paper"}
                  </button>
                  <input
                    ref={fileInput}
                    id="paper-upload"
                    type="file"
                    accept=".pdf,.txt"
                    aria-label="Upload paper"
                    className="file-input"
                    tabIndex={-1}
                    onChange={(e) => {
                      if (e.target.files?.[0]) void upload(e.target.files[0]);
                      e.target.value = "";
                    }}
                    disabled={!!busy}
                  />
                </div>
                <div className="source-tools">
                  <button onClick={() => setDialog("editor")}>
                    {text ? <FileText size={13} /> : <Plus size={13} />}{" "}
                    {text ? "Edit abstract" : "Paste abstract"}
                  </button>
                  <span>{text ? `${words} words` : "10 MB max"}</span>
                </div>
                <button
                  className="primary analyze"
                  disabled={!text || !!busy || !ready}
                  onClick={() => void analyze()}
                >
                  {busy === "predict" ? (
                    <LoaderCircle size={16} className="spin" />
                  ) : (
                    <Scan size={16} />
                  )}{" "}
                  {busy === "predict" ? "Reading signal…" : "Analyze paper"}
                  <ArrowRight size={16} />
                </button>
                <button
                  className="example"
                  disabled={!!busy || !ready}
                  onClick={() => {
                    setText(SAMPLE);
                    setSource("Illustrative example");
                    setNotice("Synthetic demonstration abstract.");
                    setSelected(0);
                    void analyze(SAMPLE);
                  }}
                >
                  Try an example <ArrowUpRight size={13} />
                </button>
                {source === "Illustrative example" && (
                  <span className="example-note">
                    Synthetic abstract · demo only
                  </span>
                )}
                {error && (
                  <div className="inline-error" role="alert">
                    {error}
                  </div>
                )}
                <div className="source-bottom">
                  <span>
                    <i />
                    PRIVATE BY DEFAULT
                  </span>
                  <button
                    className="icon"
                    aria-label="Clear abstract"
                    disabled={!text || !!busy}
                    onClick={() => {
                      edit("");
                      setSource("");
                    }}
                  >
                    <RotateCcw size={14} />
                  </button>
                </div>
              </>
            ) : (
              <>
                <div className="rail-title">
                  <Label>01 / DATASET</Label>
                  <span className="tiny-index">06 FIELDS</span>
                </div>
                <div className="dataset-count">
                  {report ? num(total) : "—"}
                  <span>historical abstracts</span>
                </div>
                <div className="timeline">
                  {report?.splits.map((s, i) => (
                    <div key={s.name}>
                      <i className={`split-${i}`} />
                      <div>
                        <span>{s.name}</span>
                        <strong>{s.years}</strong>
                      </div>
                      <code>{num(s.n)}</code>
                    </div>
                  ))}
                </div>
                <div className="rail-title transfer-label">
                  <Label>UNSEEN-FIELD AUC</Label>
                </div>
                <div className="transfer-bars">
                  {report?.field_transfer.map((v) => (
                    <div key={v.field_id}>
                      <span>{fieldNames[v.field_id]}</span>
                      <code>{f(v.roc_auc)}</code>
                      <div>
                        <i style={{ width: `${v.roc_auc * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </aside>
          <section
            className="visual-stage"
            aria-label={
              mode === "analysis"
                ? "Interactive feature visualization"
                : "Test performance visualization"
            }
          >
            <div className="visual-heading">
              <Label>
                {mode === "analysis"
                  ? "02 / FEATURE FIELD"
                  : "02 / HELD-OUT PERFORMANCE"}
              </Label>
              <span className="live-label">
                <i />
                {busy
                  ? "PROCESSING"
                  : mode === "training"
                    ? "RECORDED RUN"
                    : result
                      ? "SIGNAL RESOLVED"
                      : "MODEL LOADED"}
              </span>
            </div>
            {mode === "analysis" ? (
              <FeatureField
                report={report}
                result={result}
                busy={!!busy}
                paused={paused}
                selected={selected}
                onSelect={setSelected}
              />
            ) : report ? (
              <TrainingField report={report} />
            ) : (
              <div className="loading-stage">Loading experiment…</div>
            )}
            <div className="stage-caption">
              <span>
                {mode === "analysis" ? (
                  <>
                    <i className="positive-dot" />
                    Positive <i className="negative-dot" />
                    Negative
                  </>
                ) : (
                  <>{report ? num(report.test.n) : "—"} unseen papers · 2021</>
                )}
              </span>
              <button
                className="stage-inspect"
                disabled={!report}
                onClick={() =>
                  setDialog(
                    mode === "analysis" && result ? "features" : "report",
                  )
                }
              >
                {mode === "analysis" && result ? "Full trace" : "Run details"}
                <ArrowUpRight size={13} />
              </button>
            </div>
          </section>
          <aside
            className="right-rail"
            aria-label={
              mode === "analysis" ? "Signal evidence" : "Calibration evidence"
            }
          >
            <div className="rail-title">
              <Label>
                03 / {mode === "analysis" ? "EVIDENCE" : "RELIABILITY"}
              </Label>
              <button
                className="icon"
                onClick={() =>
                  setDialog(
                    mode === "analysis" && result ? "features" : "report",
                  )
                }
                aria-label={
                  mode === "analysis" && result
                    ? "Inspect all features"
                    : "Open full report"
                }
              >
                <Maximize2 size={14} />
              </button>
            </div>
            {mode === "analysis" ? (
              <>
                <div className="evidence-stat">
                  <Label>HELD-OUT ROC AUC</Label>
                  <strong>
                    {report ? f(report.test.roc_auc) : "—"}
                    <span>/ 1.000</span>
                  </strong>
                  <MiniCurve
                    label="Recorded test ROC curve"
                    diagonal
                    points={report?.roc.map((v) => [v.fpr, v.tpr]) ?? []}
                  />
                </div>
                <div className="rail-divider" />
                <div className="rail-title">
                  <Label>
                    {result ? "TOP CONTRIBUTIONS" : "BERT-MINI / ARCHITECTURE"}
                  </Label>
                </div>
                {result ? (
                  <div className="contribution-bars">
                    {result.transformer && (
                      <Label>4 layers × 4 heads · peak attention</Label>
                    )}
                    {result.transformer && (
                      <div
                        className="attention-grid"
                        role="img"
                        aria-label="Four layers by four heads: each cell shows maximum CLS attention weight, not causal importance"
                      >
                        {result.transformer.cls_attention.flatMap((layer, li) =>
                          layer.map((head, hi) => (
                            <i
                              key={`${li}-${hi}`}
                              title={`Layer ${li + 1}, head ${hi + 1}: peak CLS attention ${pct(Math.max(...head))}`}
                              style={{ opacity: 0.2 + 0.8 * Math.max(...head) }}
                            />
                          )),
                        )}
                      </div>
                    )}
                    {contributions.slice(0, 3).map((v) => (
                      <div key={v.name}>
                        <span>{v.label}</span>
                        <code className={v.contribution < 0 ? "amber" : "mint"}>
                          {v.contribution >= 0 ? "+" : ""}
                          {f(v.contribution, 2)}
                        </code>
                        <div>
                          <i
                            className={v.contribution < 0 ? "negative" : ""}
                            style={{
                              width: `${(Math.abs(v.contribution) / maxContribution) * 100}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="feature-families">
                    {[
                      ["Transformer layers", 4],
                      ["Heads per layer", 4],
                      ["Hidden dimensions", 256],
                    ].map(([name, count]) => (
                      <div key={name}>
                        <span>{name}</span>
                        <code>{count}</code>
                        <div className="family-ticks">
                          {Array.from(
                            { length: Math.min(count as number, 32) },
                            (_, i) => (
                              <i key={i} />
                            ),
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                <div className="rail-bottom">
                  {result ? (
                    <>
                      <button
                        className="subtle"
                        onClick={() => setDialog("features")}
                      >
                        Inspect arithmetic <ArrowUpRight size={14} />
                      </button>
                      <button
                        className="subtle"
                        onClick={() => save(result, "shirabe-analysis.json")}
                      >
                        Export analysis <ArrowDownToLine size={14} />
                      </button>
                    </>
                  ) : (
                    <span>
                      Words become context.
                      <br />
                      Features become a signal.
                    </span>
                  )}
                </div>
              </>
            ) : report ? (
              <>
                <div className="evidence-stat">
                  <Label>BRIER SCORE ↓</Label>
                  <strong>{f(report.test.brier)}</strong>
                  <span className="stat-note">
                    prior{" "}
                    {f(
                      report.comparisons.find(
                        (v) => v.role === "prior baseline",
                      )?.brier ?? 0,
                    )}
                  </span>
                  <MiniCurve
                    label="Calibration: predicted versus observed probability"
                    diagonal
                    points={report.reliability.map((v) => [
                      v.predicted,
                      v.observed,
                    ])}
                  />
                  <div className="mini-label">
                    CALIBRATION / PREDICTED → OBSERVED
                  </div>
                </div>
                <div className="rail-divider" />
                <div className="small-stat">
                  <Label>AVERAGE PRECISION</Label>
                  <strong>{f(report.test.average_precision)}</strong>
                </div>
                <div className="small-stat">
                  <Label>DISCIPLINE DETECTABLE</Label>
                  <strong>{pct(report.discipline_probe.accuracy)}</strong>
                  <span>Subject independence unproven</span>
                </div>
                <div className="rail-bottom">
                  <button
                    className="subtle"
                    onClick={() => setDialog("report")}
                  >
                    Full experiment <ArrowUpRight size={14} />
                  </button>
                  <button
                    className="subtle"
                    onClick={() => save(report, "shirabe-training-report.json")}
                  >
                    Export run <ArrowDownToLine size={14} />
                  </button>
                </div>
              </>
            ) : null}
          </aside>
        </div>
        <section
          className="bottom-strip"
          aria-label={
            mode === "analysis" ? "Token representation" : "Model comparison"
          }
        >
          <div className="strip-label">
            <Label>
              {mode === "analysis"
                ? "04 / TOKEN STREAM"
                : "04 / MODEL COMPARISON"}
            </Label>
            {mode === "analysis" ? (
              <label className="mask-control">
                <input
                  type="checkbox"
                  checked={masked}
                  disabled={!result}
                  onChange={(e) => setMasked(e.target.checked)}
                />
                Mask tokens
              </label>
            ) : (
              <span>TEST ROC AUC ↑</span>
            )}
          </div>
          {mode === "analysis" ? (
            <div
              className="token-stream"
              tabIndex={0}
              role="region"
              aria-label="WordPiece tokens with CLS attention"
            >
              {result ? (
                result.tokens.map((t, i) => (
                  <span
                    title={`CLS attention ${((t.attention ?? 0) * 100).toFixed(2)}% · characters ${t.start}–${t.end}`}
                    style={{
                      backgroundColor: `rgba(117, 239, 198, ${Math.min(0.24, (t.attention ?? 0) * 12)})`,
                    }}
                    className={`token ${t.category}`}
                    key={i}
                  >
                    {masked && t.category === "content" ? "[content]" : t.text}
                  </span>
                ))
              ) : (
                <div className="token-standby">
                  <span>WAITING FOR INPUT</span>
                  <div>
                    {Array.from({ length: 34 }, (_, i) => (
                      <i
                        key={i}
                        style={{
                          width: `${14 + ((i * 17) % 64)}px`,
                          animationDelay: `${i * 0.05}s`,
                        }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div
              className="baseline-strip"
              tabIndex={0}
              role="region"
              aria-label="Model AUC comparison"
            >
              {report?.comparisons.slice(0, 6).map((v) => (
                <div
                  key={v.name}
                  className={v.role === "deployed" ? "deployed" : ""}
                >
                  <span>
                    {v.role === "deployed"
                      ? "BERT-Mini"
                      : v.name.startsWith("Style")
                        ? "Style"
                        : v.role === "prior baseline"
                          ? "Prior"
                          : v.name.startsWith("Length")
                            ? "Length"
                            : v.name.startsWith("Rhetoric")
                              ? "Rhetoric"
                              : "TF-IDF"}
                  </span>
                  <strong>{f(v.roc_auc)}</strong>
                  <div>
                    <i style={{ width: `${v.roc_auc * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
      <footer className="statusbar">
        <button onClick={() => setDialog("about")}>
          <span className="status-square" />
          Citation attention ≠ scientific truth <ArrowUpRight size={12} />
        </button>
        {result?.warnings.length ? (
          <button
            className="warning-link"
            onClick={() => setDialog("features")}
          >
            {result.warnings.length} input{" "}
            {result.warnings.length === 1 ? "note" : "notes"}
          </button>
        ) : (
          <span className="footer-meta">ENGLISH ABSTRACTS · 80–800 WORDS</span>
        )}
        <a
          href="https://github.com/Hone-Systems/shirabe"
          target="_blank"
          rel="noreferrer"
        >
          SOURCE <ArrowUpRight size={12} />
        </a>
      </footer>
      {dialog === "editor" && (
        <Modal title="Paper input" onClose={() => setDialog(null)}>
          <div className="editor-top">
            <span>{source || "Paste an English abstract"}</span>
            <span>{words} / 800 words</span>
            <button
              className="icon"
              aria-label="Clear abstract"
              disabled={!text}
              onClick={() => {
                edit("");
                setSource("");
              }}
            >
              <RotateCcw size={15} />
            </button>
          </div>
          {notice && <p className="notice">{notice}</p>}
          <label className="editor-label" htmlFor="abstract">
            Abstract text
          </label>
          <textarea
            id="abstract"
            value={text}
            onChange={(e) => edit(e.target.value)}
            maxLength={20000}
            placeholder="Paste the abstract. 80–800 English words."
            aria-describedby="editor-error"
            aria-invalid={!!error}
            autoFocus
          />
          <p
            id="editor-error"
            className={error ? "inline-error" : "editor-hint"}
            role={error ? "alert" : undefined}
          >
            {error ||
              "Review the extracted text before analysis. Remove headings, authors, and footnotes."}
          </p>
          <button
            className="primary"
            disabled={!text || !!busy || !ready}
            onClick={() => void analyze()}
          >
            <Scan size={16} />
            Analyze paper
            <ArrowRight size={16} />
          </button>
        </Modal>
      )}
      {dialog === "features" && result && (
        <Modal title="Inside the estimate" onClose={() => setDialog(null)}>
          <div className="detail-score">
            <strong>{pct(result.probability)}</strong>
            <span>Estimated high-citation probability</span>
          </div>
          {result.warnings.map((w) => (
            <p className="notice" key={w}>
              {w}
            </p>
          ))}
          <p className="formula">
            sigmoid({f(result.intercept)} + Σ contributions) = sigmoid(
            {f(result.logit)}) = {pct(result.probability)}
          </p>
          <p>
            Exact log-odds contributions. Associations, not rewriting advice.
            The intercept is the calibrated head bias at zero pooled activation.
            Train-relative z values are diagnostics; the head uses raw
            activations. Attention is not causal attribution.
          </p>
          {result.transformer && (
            <details>
              <summary>Attention values · layers, heads and tokens</summary>
              <p>
                Peak CLS attention per head, including special tokens. Token
                values below are final-layer CLS attention averaged across
                heads. Attention is not causal importance.
              </p>
              <div
                className="table-wrap"
                tabIndex={0}
                role="region"
                aria-label="Attention by layer and head"
              >
                <table>
                  <thead>
                    <tr>
                      <th>Layer</th>
                      <th>Head</th>
                      <th>Peak attention</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.transformer.cls_attention.flatMap((layer, li) =>
                      layer.map((head, hi) => (
                        <tr key={`${li}-${hi}`}>
                          <td>{li + 1}</td>
                          <td>{hi + 1}</td>
                          <td>{pct(Math.max(...head))}</td>
                        </tr>
                      )),
                    )}
                  </tbody>
                </table>
              </div>
              <div
                className="table-wrap"
                tabIndex={0}
                role="region"
                aria-label="Token attention values"
              >
                <table>
                  <thead>
                    <tr>
                      <th>WordPiece</th>
                      <th>Character offset</th>
                      <th>CLS attention</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.tokens.map((t, i) => (
                      <tr key={i}>
                        <td>{t.text}</td>
                        <td>
                          {t.start}–{t.end}
                        </td>
                        <td>{((t.attention ?? 0) * 100).toFixed(3)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
          <div
            className="table-wrap"
            tabIndex={0}
            role="region"
            aria-label="All feature contributions"
          >
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Raw</th>
                  <th>Train-relative z</th>
                  <th>Weight</th>
                  <th>Contribution</th>
                </tr>
              </thead>
              <tbody>
                {[...result.features]
                  .sort(
                    (a, b) =>
                      Math.abs(b.contribution) - Math.abs(a.contribution),
                  )
                  .map((v) => (
                    <tr key={v.name}>
                      <td>
                        {v.label}
                        {v.unusual ? " · unusual" : ""}
                      </td>
                      <td>{f(v.value)}</td>
                      <td>{f(v.z)}</td>
                      <td>{f(v.coefficient)}</td>
                      <td>{f(v.contribution)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <button
            className="subtle"
            onClick={() => save(result, "shirabe-analysis.json")}
          >
            Export analysis <ArrowDownToLine size={15} />
          </button>
        </Modal>
      )}
      {dialog === "report" && report && (
        <Modal title="Experiment record" onClose={() => setDialog(null)}>
          <FullReport r={report} />
          <button
            className="subtle"
            onClick={() => save(report, "shirabe-training-report.json")}
          >
            Export run <ArrowDownToLine size={15} />
          </button>
        </Modal>
      )}
      {dialog === "about" && (
        <Modal title="A signal. Not a verdict." onClose={() => setDialog(null)}>
          <p>
            Shirabe estimates whether an English abstract belongs to the
            higher-cited group of its sampled field and year. The dataset
            contains {num(total)} historic abstracts;{" "}
            {num(report?.splits[0].n ?? 0)} were used for training. The rest
            were held out for selection, calibration and testing.
          </p>
          <p>
            The target is citation attention—not correctness, replication, or
            practical success. Results are retrospective; subject independence
            remains unproven.
          </p>
          <p>
            The field visualizes 256 learned classifier-head channels. Ray
            lengths show calibrated head weights before analysis and signed
            log-odds contributions afterward. The head contributions plus
            intercept reconstruct the score. Moving particles are illustrative.
          </p>
          <p>
            BERT-Mini reads WordPieces through four transformer layers, with
            four attention heads each. Token brightness shows final-layer CLS
            attention averaged over heads; attention is not causal importance.
            The model reads both topic and style. Uploads are processed locally
            and not retained; review extracted abstracts before scoring.
          </p>
          <button className="subtle" onClick={() => setDialog("report")}>
            Read the experiment <ArrowUpRight size={14} />
          </button>
        </Modal>
      )}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
