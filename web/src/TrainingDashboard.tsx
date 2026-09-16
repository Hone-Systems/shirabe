import { useEffect, useRef, useState } from "react";
import { ArrowDownToLine, ArrowUpRight, RefreshCw } from "lucide-react";
import "./training.css";
type Epoch = {
  epoch: number;
  train_loss: number;
  validation_loss: number;
  train_brier: number;
  validation_brier: number;
  seconds: number;
};
type Interval = {
  papers: number;
  mean: number | null;
  lower: number | null;
  upper: number | null;
};
type Metric = {
  papers?: {
    id: string;
    title: string;
    probabilities: (number | null)[];
    known_labels: Record<string, string>;
  }[];
  scorable_papers?: number;
  auc?: number | null;
  auc_heads?: number;
  known_items: number;
  log_loss: number | null;
  brier: number | null;
};
type Run = {
  id: string;
  history: Epoch[];
  steps: { step: number; loss: number; gradient_norm: number }[];
  chunks: number;
  best_epoch: number | null;
  metrics: Record<string, Metric>;
  constant_positive_baseline?: Record<string, { brier: number | null }>;
};
type Report = {
  artifact_round?: string;
  learning_curve_history?: Report["experiments"];
  external_evaluation?: {
    runs: {
      id: string;
      seed: number;
      metrics: Metric;
      comparisons: Record<string, Interval>;
    }[];
    controls: Record<string, Metric>;
    scorable_papers: number;
    note: string;
  };
  pending_dataset?: { papers: number; yes: number; no: number };
  status: string;
  stage: string;
  architecture?: { parameters?: number };
  config?: { epochs: number; learning_rate: number; max_chunk_tokens: number };
  dataset?: {
    splits: Record<string, { papers: number; yes: number; no: number }>;
    questions: {
      id: string;
      question: string;
      yes?: number;
      no?: number;
      unknown?: number;
      not_applicable?: number;
    }[];
  };
  inputs?: { scope: string }[];
  expansion?: { planned: number; reviewed: number };
  runs: Run[];
  promotion?: { reason: string };
  experiments?: {
    id: string;
    seed: number;
    fraction: number;
    training_papers: number;
    metrics: Record<string, Metric>;
    fresh_test_comparisons?: Record<string, Interval>;
  }[];
  controls?: Record<string, Record<string, Metric>>;
  conclusion?: { interpretation: string };
  leakage_probe?: {
    validation_accuracy: number;
    majority_accuracy: number;
    papers: number;
  };
  limitations?: string[];
  rl_readiness?: { status: string; reason: string };
};
const f = (n: number | null | undefined, d = 3) =>
  n == null ? "—" : n.toFixed(d);
const labels: Record<string, string> = {
  wording: "Wording",
  original: "Original text",
  topic: "Topic control",
};
function Lines({
  title,
  series,
  xLabel = "epoch",
  selected,
}: {
  title: string;
  series: {
    name: string;
    color: string;
    points: [number, number][];
    dashed?: boolean;
  }[];
  xLabel?: string;
  selected?: number;
}) {
  const points = series
    .flatMap((s) => s.points)
    .filter((p) => Number.isFinite(p[1]));
  const maxX = Math.max(1, ...points.map((p) => p[0]));
  const maxY = Math.max(0.001, ...points.map((p) => p[1])) * 1.1;
  return (
    <>
      <div className="train-legend">
        {series.map((s) => (
          <span key={s.name}>
            <i style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
      </div>
      {!points.length ? (
        <div className="train-empty">Waiting for measured epochs</div>
      ) : (
        <svg viewBox="0 0 440 180" role="img" aria-label={title}>
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <path d={`M42 ${150 - t * 120}H426`} stroke="var(--line)" />
              <text x="35" y={154 - t * 120} textAnchor="end">
                {f(maxY * t, 2)}
              </text>
            </g>
          ))}
          {selected != null && (
            <path
              d={`M${42 + (selected / maxX) * 380} 20V150`}
              stroke="var(--muted)"
              strokeDasharray="3 5"
            />
          )}
          {series.map((s) => (
            <g key={s.name}>
              <polyline
                fill="none"
                stroke={s.color}
                strokeWidth="2"
                strokeDasharray={s.dashed ? "5 4" : undefined}
                points={s.points
                  .map(
                    ([x, y]) =>
                      `${42 + (x / maxX) * 380},${150 - (y / maxY) * 120}`,
                  )
                  .join(" ")}
              />
              {s.points
                .filter(
                  (_, i) => s.points.length < 30 || i === s.points.length - 1,
                )
                .map(([x, y], i) => (
                  <circle
                    key={i}
                    cx={42 + (x / maxX) * 380}
                    cy={150 - (y / maxY) * 120}
                    r="3"
                    fill={s.color}
                  >
                    <title>
                      {s.name}, {xLabel} {x}: {f(y, 5)}
                    </title>
                  </circle>
                ))}
            </g>
          ))}
          <text x="42" y="170">
            0
          </text>
          <text x="230" y="170" textAnchor="middle">
            {xLabel}
          </text>
          <text x="424" y="170" textAnchor="end">
            {maxX}
          </text>
        </svg>
      )}
    </>
  );
}
export default function TrainingDashboard({
  onBaseline,
}: {
  onBaseline: () => void;
}) {
  const [data, setData] = useState<Report | null>(null),
    [error, setError] = useState(""),
    [runId, setRunId] = useState("wording"),
    [epoch, setEpoch] = useState<number | null>(null),
    [split, setSplit] = useState("validation"),
    [paperId, setPaperId] = useState("challenge-attention");
  const pending = useRef(false);
  const [refreshing, setRefreshing] = useState(false);
  async function refresh() {
    if (pending.current) return;
    pending.current = true;
    setRefreshing(true);
    try {
      const r = await fetch("/api/outcome-training");
      if (!r.ok) throw new Error("Training telemetry is unavailable");
      setData(await r.json());
      setError("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      pending.current = false;
      setRefreshing(false);
    }
  }
  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), 2000);
    return () => clearInterval(t);
  }, []);
  const run = data?.runs.find((r) => r.id === runId),
    history = run?.history ?? [],
    row = history.find((r) => r.epoch === epoch) ?? history.at(-1),
    selectedEpoch = row?.epoch;
  const qs = data?.dataset?.questions ?? [],
    train = data?.dataset?.splits.train;
  const active =
    !!data && !["complete", "error", "not_started"].includes(data.status);
  const evaluationName = data?.external_evaluation
    ? "External cancer cohort"
    : data?.artifact_round === "round-3"
      ? "Previously inspected holdout"
      : "Fresh test";
  const evaluationExperiments = data?.external_evaluation
    ? data.external_evaluation.runs.map((e) => ({
        ...e,
        fraction: 1,
        evaluation: e.metrics,
        comparisons: e.comparisons,
      }))
    : (data?.experiments ?? []).map((e) => ({
        ...e,
        evaluation: e.metrics.fresh_test,
        comparisons: e.fresh_test_comparisons,
      }));
  const evaluationControls =
    data?.external_evaluation?.controls ??
    Object.fromEntries(
      Object.entries(data?.controls ?? {}).map(([k, v]) => [k, v.fresh_test]),
    );
  const challengePapers = run?.metrics.challenge?.papers ?? [];
  const selectedPaper =
    challengePapers.find((p) => p.id === paperId) ?? challengePapers[0];
  function download() {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(
      new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    );
    a.download = "shirabe-outcome-training.json";
    a.click();
    URL.revokeObjectURL(a.href);
  }
  return (
    <main id="workspace" className="training-dashboard">
      <div className="training-heading">
        <div>
          <span className="micro">OUTCOME RUBRIC / TRANSFORMER EXPERIMENT</span>
          <h1>Inside the fit.</h1>
        </div>
        <div className="train-actions">
          <button onClick={onBaseline}>
            Citation baseline <ArrowUpRight size={14} />
          </button>
          <button
            onClick={() => void refresh()}
            disabled={refreshing}
            aria-label={refreshing ? "Refreshing training" : "Refresh training"}
          >
            <RefreshCw size={16} />
          </button>
          <button
            onClick={download}
            disabled={!data}
            aria-label="Export training report"
          >
            <ArrowDownToLine size={17} />
          </button>
        </div>
      </div>
      {error && (
        <p role="alert" className="train-warning">
          {error}
        </p>
      )}
      <div className="training-status" role="status">
        <span className={active ? "live-status" : ""}>
          {data?.stage ?? "Connecting to training run"}
        </span>
        <span>
          {data?.status === "complete"
            ? train?.no === 0
              ? "DIAGNOSTIC · NO NEGATIVE TRAINING LABELS"
              : "DIAGNOSTIC · NOT PROMOTED"
            : data?.status?.toUpperCase()}
        </span>
      </div>
      {data?.pending_dataset && (
        <p className="training-footnote">
          Expanded dataset being prepared: {data.pending_dataset.papers}{" "}
          training papers · {data.pending_dataset.yes} yes /{" "}
          {data.pending_dataset.no} no. Curves below belong to the previous fit.
        </p>
      )}
      <div className="training-metrics">
        <div>
          <strong>
            {data?.architecture?.parameters
              ? (data.architecture.parameters / 1e6).toFixed(2) + "M"
              : "BERT mini"}
          </strong>
          <span>4 layers · 4 heads · 20 outputs</span>
        </div>
        <div>
          <strong>
            {train?.papers ?? "—"} <small>papers</small>
          </strong>
          <span>
            {train?.yes ?? 0} yes / {train?.no ?? 0} no in training
          </span>
        </div>
        <div>
          <strong>{f(row?.train_loss)}</strong>
          <span>Training log loss · epoch {selectedEpoch ?? "—"}</span>
        </div>
        <div>
          <strong>{f(row?.validation_loss)}</strong>
          <span>Validation log loss</span>
        </div>
        <div>
          <strong>
            {run?.best_epoch ?? "—"}{" "}
            <small>/ {data?.config?.epochs ?? 8}</small>
          </strong>
          <span>Selected epoch · validation loss</span>
        </div>
      </div>
      <div className="training-controls">
        <div className="training-variants" aria-label="Input representation">
          {["wording", "original", "topic"].map((id) => (
            <button
              key={id}
              aria-pressed={runId === id}
              onClick={() => {
                setRunId(id);
                setEpoch(null);
              }}
            >
              {labels[id]}
            </button>
          ))}
        </div>
        <label className="epoch-scrubber">
          Epoch{" "}
          <input
            aria-label="Inspect epoch"
            type="range"
            min="1"
            max={Math.max(1, history.length)}
            value={selectedEpoch ?? 1}
            disabled={!history.length}
            onChange={(e) => setEpoch(Number(e.target.value))}
          />
          <span>{selectedEpoch ?? "—"}</span>
        </label>
        <button onClick={() => setEpoch(null)} disabled={epoch === null}>
          Latest
        </button>
      </div>
      <div className="training-charts">
        <section className="training-chart">
          <h2>
            Loss trajectory <span>lower is better</span>
          </h2>
          <Lines
            title="Training and validation log loss by epoch"
            selected={selectedEpoch}
            series={[
              {
                name: "Train",
                color: "var(--mint)",
                points: history.map((e) => [e.epoch, e.train_loss]),
              },
              {
                name: "Validation",
                color: "var(--amber)",
                dashed: true,
                points: history.map((e) => [e.epoch, e.validation_loss]),
              },
            ]}
          />
        </section>
        <section className="training-chart">
          <h2>
            Prediction error <span>Brier score</span>
          </h2>
          <Lines
            title="Training and validation Brier scores by epoch"
            selected={selectedEpoch}
            series={[
              {
                name: "Train",
                color: "var(--mint)",
                points: history.map((e) => [e.epoch, e.train_brier]),
              },
              {
                name: "Validation",
                color: "var(--amber)",
                dashed: true,
                points: history.map((e) => [e.epoch, e.validation_brier]),
              },
              {
                name: "Always yes · validation",
                color: "var(--muted)",
                dashed: true,
                points:
                  run?.constant_positive_baseline?.validation?.brier != null
                    ? history.map((e) => [
                        e.epoch,
                        run.constant_positive_baseline!.validation.brier!,
                      ])
                    : [],
              },
            ]}
          />
        </section>
        <section className="training-chart">
          <h2>
            Gradient strength <span>before clipping</span>
          </h2>
          <Lines
            title="Gradient norm by optimization step"
            xLabel="step"
            series={[
              {
                name: "Gradient norm",
                color: "var(--mint)",
                points: (run?.steps ?? []).map((e) => [
                  e.step,
                  e.gradient_norm,
                ]),
              },
            ]}
          />
        </section>
        <section className="training-chart">
          <h2>
            What can be learned? <span>20 question heads</span>
          </h2>
          <div
            className="question-bars"
            aria-label="Training labels per question"
          >
            {qs.map((q, i) => (
              <div key={q.id} title={q.question}>
                <span>
                  {String(i + 1).padStart(2, "0")} {q.id.replaceAll("_", " ")}
                </span>
                <div
                  role="img"
                  aria-label={`${q.question}: ${q.yes ?? 0} yes, ${q.no ?? 0} no, ${(q.unknown ?? 0) + (q.not_applicable ?? 0)} masked`}
                >
                  <i
                    style={{
                      width: `${(100 * (q.yes ?? 0)) / Math.max(1, train?.papers ?? 1)}%`,
                    }}
                  />
                  <b
                    style={{
                      width: `${(100 * (q.no ?? 0)) / Math.max(1, train?.papers ?? 1)}%`,
                    }}
                  />
                </div>
                <em>
                  {q.yes ?? 0}/{q.no ?? 0}
                </em>
              </div>
            ))}
          </div>
          <p className="chart-note">Yes / no · grey targets are masked</p>
        </section>
        <section className="training-chart">
          <h2>
            Observed targets <span>by split</span>
          </h2>
          <div className="split-bars">
            {Object.entries(data?.dataset?.splits ?? {}).map(([s, v]) => (
              <div key={s}>
                <span>{s}</span>
                <div>
                  <i
                    style={{
                      width: `${(100 * v.yes) / Math.max(1, v.papers * 20)}%`,
                    }}
                  />
                  <b
                    style={{
                      width: `${(100 * v.no) / Math.max(1, v.papers * 20)}%`,
                    }}
                  />
                </div>
                <small>
                  {v.yes} / {v.no}
                </small>
              </div>
            ))}
          </div>
          <div className="train-legend">
            <span>
              <i style={{ background: "var(--mint)" }} />
              Yes
            </span>
            <span>
              <i style={{ background: "var(--amber)" }} />
              No
            </span>
            <span>Grey: masked</span>
          </div>
          <p className="chart-note">
            Challenge papers are evaluation only. Papers with no known targets
            do not contribute to this fit.
          </p>
          <h3>Epoch compute time</h3>
          <svg
            viewBox="0 0 440 80"
            role="img"
            aria-label="Duration of each epoch in seconds"
          >
            {history.map((e, i) => (
              <g key={e.epoch}>
                <rect
                  x={20 + (i * 400) / Math.max(1, history.length)}
                  y={
                    60 -
                    (e.seconds /
                      Math.max(1, ...history.map((e) => e.seconds))) *
                      45
                  }
                  width={Math.max(2, 320 / Math.max(1, history.length))}
                  height={
                    (e.seconds /
                      Math.max(1, ...history.map((e) => e.seconds))) *
                    45
                  }
                  fill="var(--mint)"
                  opacity={selectedEpoch === e.epoch ? 1 : 0.45}
                >
                  <title>
                    Epoch {e.epoch}: {f(e.seconds, 1)} seconds
                  </title>
                </rect>
                <text x={24 + (i * 400) / Math.max(1, history.length)} y="76">
                  {e.epoch}
                </text>
              </g>
            ))}
          </svg>
        </section>
        <section className="training-chart">
          <h2>Representation comparison</h2>
          <label className="comparison-select">
            Evaluation{" "}
            <select
              aria-label="Evaluation split"
              value={split}
              onChange={(e) => setSplit(e.target.value)}
            >
              {[
                ...(data?.external_evaluation ? ["external"] : []),
                ...(data?.expansion
                  ? [
                      "fresh_test",
                      "fresh_validation",
                      "validation",
                      "calibration",
                      "test",
                      "challenge",
                    ]
                  : ["validation", "calibration", "test", "challenge"]),
              ].map((s) => (
                <option key={s} value={s}>
                  {s === "fresh_test" && data?.artifact_round === "round-3"
                    ? "previously inspected holdout"
                    : s}
                </option>
              ))}
            </select>
          </label>
          <div className="comparison-bars">
            {(data?.runs ?? []).map((r) => {
              const v = r.metrics[split]?.brier;
              return (
                <div key={r.id}>
                  <span>{labels[r.id]}</span>
                  <div>
                    <i style={{ width: `${100 * (v ?? 0)}%` }} />
                  </div>
                  <strong>{f(v)}</strong>
                </div>
              );
            })}
            <div>
              <span>Always yes</span>
              <div>
                <i
                  style={{
                    width: `${100 * (run?.constant_positive_baseline?.[split]?.brier ?? 0)}%`,
                    background: "var(--muted)",
                  }}
                />
              </div>
              <strong>
                {f(run?.constant_positive_baseline?.[split]?.brier)}
              </strong>
            </div>
          </div>
          <p className="chart-note">Brier error · 0–1 · lower is better</p>
          <div className="comparison-summary">
            <div>
              <strong>{run?.metrics[split]?.known_items ?? "—"}</strong>
              <span>Scorable held-out items</span>
            </div>
            <div>
              <strong>{f(run?.metrics[split]?.auc)}</strong>
              <span>
                {run?.metrics[split]?.auc != null
                  ? `Macro AUC · ${run.metrics[split].auc_heads} mixed heads`
                  : "AUC not established"}
              </span>
            </div>
          </div>
          <p className="chart-note">
            {train?.no === 0
              ? "A falling loss on positive-only labels cannot establish success/failure discrimination."
              : "Paper-weighted error. External papers were excluded from training and model selection; prior holdout results are development evidence."}
          </p>
        </section>
      </div>
      {!!data?.experiments?.length && (
        <div className="training-charts expansion-charts">
          <section className="training-chart">
            <h2>
              Does more data help? <span>validation Brier</span>
            </h2>
            <Lines
              title="Validation error as training paper count increases"
              xLabel="training papers"
              series={["wording", "original", "topic"].map((id, i) => ({
                name: labels[id],
                color: ["var(--mint)", "var(--amber)", "var(--muted)"][i],
                points: (data.learning_curve_history ?? data.experiments ?? [])
                  .filter(
                    (e) =>
                      e.id === id &&
                      e.seed === 42 &&
                      e.metrics.validation?.brier != null,
                  )
                  .map(
                    (e) =>
                      [e.training_papers, e.metrics.validation.brier!] as [
                        number,
                        number,
                      ],
                  ),
              }))}
            />
            <p className="chart-note">
              Nested subsets · same validation cohort · seed 42 · initial
              8-epoch fits
            </p>
          </section>
          <section className="training-chart">
            <h2>
              {evaluationName}, repeated seeds{" "}
              <span>Brier · lower is better</span>
            </h2>
            <div className="comparison-bars">
              {evaluationExperiments
                .filter((e) => e.id === runId && e.fraction === 1)
                .map((e) => (
                  <div key={e.seed}>
                    <span>Seed {e.seed}</span>
                    <div>
                      <i
                        style={{
                          width: `${100 * (e.evaluation?.brier ?? 0)}%`,
                        }}
                      />
                    </div>
                    <strong>{f(e.evaluation?.brier)}</strong>
                  </div>
                ))}
              {["prevalence", "always_yes", "rhetorical", "metadata"].map(
                (c) => (
                  <div key={c}>
                    <span>{c}</span>
                    <div>
                      <i
                        style={{
                          width: `${100 * (evaluationControls[c]?.brier ?? 0)}%`,
                          background: "var(--amber)",
                        }}
                      />
                    </div>
                    <strong>{f(evaluationControls[c]?.brier)}</strong>
                  </div>
                ),
              )}
            </div>
            <p className="chart-note">
              Controls fit on training papers only.{" "}
              {data.external_evaluation
                ? `${data.external_evaluation.scorable_papers} external papers; no fitting or checkpoint selection on this cohort.`
                : data.artifact_round === "round-3"
                  ? "This holdout was inspected in the previous round; it is development evidence."
                  : "Fresh test opened after all scheduled fits."}
            </p>
          </section>
          <section className="training-chart">
            <h2>
              How certain is the gain? <span>paired 95% interval</span>
            </h2>
            <div className="interval-list">
              {evaluationExperiments
                .filter((e) => e.id === runId && e.fraction === 1)
                .map((e) => {
                  const c = e.comparisons?.prevalence;
                  return (
                    <div key={e.seed}>
                      <span>Seed {e.seed}</span>
                      <strong>{f(c?.mean)}</strong>
                      <span>
                        [{f(c?.lower)}, {f(c?.upper)}]
                      </span>
                    </div>
                  );
                })}
            </div>
            <p className="chart-note">
              Positive means lower Brier than the prevalence control. An
              interval crossing zero leaves the improvement uncertain.
              Resampling unit: whole paper.
            </p>
            {data.leakage_probe && (
              <p className="chart-note">
                Cohort recoverable from wording:{" "}
                {(100 * data.leakage_probe.validation_accuracy).toFixed(0)}% vs{" "}
                {(100 * data.leakage_probe.majority_accuracy).toFixed(0)}%
                majority control ({data.leakage_probe.papers} validation
                papers). Topic independence remains unproven.
              </p>
            )}
          </section>
        </div>
      )}
      {data?.conclusion && (
        <p className="training-footnote">{data.conclusion.interpretation}</p>
      )}
      {data?.rl_readiness && (
        <p className="training-footnote">
          <strong>RL · {data.rl_readiness.status.replaceAll("_", " ")}</strong>{" "}
          — {data.rl_readiness.reason}
        </p>
      )}
      {!!challengePapers.length && (
        <details className="training-details training-data">
          <summary>
            Inspect held-out landmark predictions, including Attention
          </summary>
          <label className="comparison-select">
            Paper{" "}
            <select
              aria-label="Challenge paper"
              value={selectedPaper?.id}
              onChange={(e) => setPaperId(e.target.value)}
            >
              {challengePapers.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title}
                </option>
              ))}
            </select>
          </label>
          <p>
            These papers never enter training or checkpoint selection. Model
            outputs below are experimental per-question probabilities; observed
            answers come from source-backed research. They are separate from the
            older citation score on Analyze.
          </p>
          <div
            className="training-table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Challenge predictions"
          >
            <table>
              <thead>
                <tr>
                  <th scope="col">Outcome question</th>
                  <th scope="col">Model</th>
                  <th scope="col">Observed</th>
                </tr>
              </thead>
              <tbody>
                {qs.map((q, i) => (
                  <tr key={q.id}>
                    <th scope="row">{q.question}</th>
                    <td>
                      {selectedPaper?.probabilities[i] == null
                        ? "Unsupported"
                        : `${(100 * selectedPaper.probabilities[i]!).toFixed(1)}%`}
                    </td>
                    <td>
                      {selectedPaper?.known_labels[q.id] ?? "Unknown / N/A"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      <details className="training-details training-data">
        <summary>Inspect exact measurements & full questions</summary>
        <div
          className="training-table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Epoch measurements"
        >
          <table>
            <caption>{labels[runId]} · measured epochs</caption>
            <thead>
              <tr>
                {[
                  "Epoch",
                  "Train loss",
                  "Validation loss",
                  "Train Brier",
                  "Validation Brier",
                  "Seconds",
                ].map((h) => (
                  <th key={h} scope="col">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map((e) => (
                <tr key={e.epoch}>
                  <th scope="row">{e.epoch}</th>
                  {[
                    e.train_loss,
                    e.validation_loss,
                    e.train_brier,
                    e.validation_brier,
                    e.seconds,
                  ].map((n, i) => (
                    <td key={i}>{f(n, 5)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <details>
          <summary>Optimizer steps</summary>
          <div
            className="training-table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Optimizer measurements"
          >
            <table>
              <thead>
                <tr>
                  <th scope="col">Step</th>
                  <th scope="col">Loss</th>
                  <th scope="col">Gradient norm</th>
                </tr>
              </thead>
              <tbody>
                {run?.steps.map((e) => (
                  <tr key={e.step}>
                    <th scope="row">{e.step}</th>
                    <td>{f(e.loss, 5)}</td>
                    <td>{f(e.gradient_norm, 5)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
        <dl>
          {qs.map((q) => (
            <div key={q.id}>
              <dt>{q.question}</dt>
              <dd>
                {q.yes ?? 0} yes · {q.no ?? 0} no · {q.unknown ?? 0} unknown ·{" "}
                {q.not_applicable ?? 0} not applicable
              </dd>
            </div>
          ))}
        </dl>
      </details>
      <details className="training-details">
        <summary>Run settings, source coverage & limitations</summary>
        <p>
          Learning rate {data?.config?.learning_rate} · all input tokens in{" "}
          {data?.config?.max_chunk_tokens}-token chunks · {run?.chunks ?? 0}{" "}
          training and validation chunks ·{" "}
          {data?.inputs?.filter((i) => i.scope.includes("abstract")).length ??
            0}{" "}
          abstract inputs.
        </p>
        {data?.limitations?.map((t) => (
          <p key={t}>{t}</p>
        ))}
        {!!data?.experiments?.length && (
          <div
            className="training-table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Expansion experiments"
          >
            <table>
              <caption>All scheduled fits</caption>
              <thead>
                <tr>
                  <th>Input</th>
                  <th>Seed</th>
                  <th>Papers</th>
                  <th>Validation Brier</th>
                  <th>
                    {data.artifact_round === "round-3"
                      ? "Prior holdout"
                      : "Fresh test"}{" "}
                    Brier
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.experiments.map((e, i) => (
                  <tr key={i}>
                    <th scope="row">{labels[e.id]}</th>
                    <td>{e.seed}</td>
                    <td>{e.training_papers}</td>
                    <td>{f(e.metrics.validation?.brier, 5)}</td>
                    <td>{f(e.metrics.fresh_test?.brier, 5)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <a href="/api/outcome-training" target="_blank" rel="noreferrer">
          Full measured run data ↗
        </a>
      </details>
      <p className="training-footnote">
        {data?.promotion?.reason ??
          "Preparing the outcome-label experiment. Measurements appear as each epoch finishes."}
      </p>
    </main>
  );
}
