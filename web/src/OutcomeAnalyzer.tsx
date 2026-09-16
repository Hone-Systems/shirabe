import { useEffect, useRef, useState } from "react";
import { ArrowRight, RefreshCw, Upload } from "lucide-react";
import "./outcome.css";
type Question = {
  id: string;
  dimension: string;
  question: string;
  supported: boolean;
};
type Metadata = {
  model_id: string;
  parameters: number;
  training_papers: number;
  layers: { weights: number[][] }[];
  output_weights: number[][];
  questions: Question[];
  diagram_note: string;
};
type Output = {
  id: string;
  dimension: string;
  question: string;
  probability: number | null;
  training_yes: number;
  training_no: number;
};
type Prediction = {
  score: number | null;
  dimensions: {
    id: string;
    value: number | null;
    supported: number;
    total: number;
  }[];
  outputs: Output[];
  tokens: number;
  chunks: number;
  masked_fraction: number;
  layer_activations: number[][];
  head_entropy: number[][];
  model_id: string;
};
type Progress = {
  stage: string;
  model_id?: string;
  completed?: number;
  chunks?: number;
  tokens?: number;
  masked_fraction?: number;
  layer_activations?: number[][];
  head_entropy?: number[][];
  chunk_summaries?: {
    chunk: number;
    tokens: number;
    mean_supported_probability: number;
  }[];
};
const stages: Record<string, string> = {
  queued: "Waiting for local model",
  masking: "Masking topic phrases",
  encoding: "Tokenizing every chunk",
  inference: "Transformer inference",
  aggregating: "Combining chunk predictions",
};
async function streamPrediction(
  text: string,
  signal: AbortSignal,
  update: (p: Progress) => void,
): Promise<Prediction> {
  const response = await fetch("/api/outcome-predict/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal,
  });
  if (!response.ok || !response.body)
    throw new Error("Unable to start analysis. Please retry.");
  const reader = response.body.getReader(),
    decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let newline: number;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "error") throw new Error(event.detail);
        if (event.type === "progress") update(event);
        if (event.type === "result") return event.result;
      }
      if (done)
        throw new Error(
          "Analysis connection ended before completion. Please retry.",
        );
    }
  } finally {
    reader.releaseLock();
  }
}
const pct = (n: number | null | undefined) =>
  n == null ? "—" : `${(n * 100).toFixed(0)}`;
const label = (s: string) => s.replaceAll("_", " ");
async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const r = await fetch(url, options);
  const b = await r.json();
  if (!r.ok)
    throw new Error(
      typeof b.detail === "string" ? b.detail : "Request failed. Please retry.",
    );
  return b;
}
function Network({
  meta,
  result,
  layer,
  select,
  progress,
  working,
  animate,
}: {
  meta: Metadata | null;
  result: Prediction | null;
  layer: number;
  select: (n: number) => void;
  progress: Progress | null;
  working: boolean;
  animate: boolean;
}) {
  const x = (c: number) => 44 + c * 108,
    y = (r: number) => 70 + r * 40;
  const maxW = Math.max(
    0.000001,
    ...(meta?.layers.flatMap((l) => l.weights.flat().map(Math.abs)) ?? []),
  );
  const activations = result?.layer_activations ?? progress?.layer_activations;
  const maxA = Math.max(0.000001, ...(activations?.flat().map(Math.abs) ?? []));
  return (
    <div className={`oa-network ${working && animate ? "oa-working" : ""}`}>
      <p className="oa-diagram-caption">
        Grouped projection weights · 32 dimensions per node
      </p>
      <div
        className="oa-diagram-scroll"
        tabIndex={0}
        role="region"
        aria-label="Scrollable transformer diagram"
      >
        <svg
          viewBox="0 0 630 410"
          role="img"
          aria-labelledby="oa-network-title oa-network-desc"
        >
          <title id="oa-network-title">
            Four-layer transformer and twenty outcome heads
          </title>
          <desc id="oa-network-desc">
            Each node groups 32 hidden dimensions. Solid mint edges show
            positive mean weights; dashed amber edges show negative weights.
            Node fill shows measured activation magnitude after analysis.
            Buttons below select layers for exact measurements.
          </desc>
          {meta?.layers.map((l, c) =>
            l.weights.flatMap((row, to) =>
              row.map((w, from) => (
                <line
                  key={`${c}-${to}-${from}`}
                  x1={x(c)}
                  y1={y(from)}
                  x2={x(c + 1)}
                  y2={y(to)}
                  stroke={w >= 0 ? "#89efd0" : "#e7b984"}
                  strokeDasharray={w < 0 ? "2 4" : undefined}
                  strokeWidth={0.4 + (Math.abs(w) / maxW) * 1.6}
                  opacity={
                    (layer === c + 1 ? 0.22 : 0.07) +
                    (Math.abs(w) / maxW) * (layer === c + 1 ? 0.65 : 0.23)
                  }
                />
              )),
            ),
          )}
          {meta?.output_weights.flatMap((row, o) =>
            row.map((w, g) => (
              <line
                key={`o-${o}-${g}`}
                x1={x(4)}
                y1={y(g)}
                x2="596"
                y2={53 + o * 16}
                stroke={w >= 0 ? "#89efd0" : "#e7b984"}
                strokeDasharray={w < 0 ? "2 4" : undefined}
                opacity={meta.questions[o]?.supported ? 0.12 : 0.03}
                strokeWidth=".6"
              />
            )),
          )}
          {Array.from({ length: 5 }, (_, c) => (
            <g
              key={c}
              className="oa-stage-nodes"
              style={{ animationDelay: `${c * 0.3}s` }}
            >
              <text
                x={x(c)}
                y="22"
                textAnchor="middle"
                className="oa-svg-label"
              >
                {c ? `LAYER ${c}` : "EMBED"}
              </text>
              {Array.from({ length: 8 }, (_, r) => {
                const a = activations?.[c]?.[r];
                return (
                  <g key={r}>
                    <circle
                      cx={x(c)}
                      cy={y(r)}
                      r={layer === c ? 11 : 8}
                      fill="#101819"
                      stroke={
                        meta ? (layer === c ? "#89efd0" : "#738b85") : "#344b46"
                      }
                      strokeWidth={layer === c ? 1.5 : 1}
                    />
                    {a != null && (
                      <circle
                        cx={x(c)}
                        cy={y(r)}
                        r={3 + (Math.abs(a) / maxA) * 5}
                        fill={a >= 0 ? "#89efd0" : "#e7b984"}
                        opacity={0.3 + (Math.abs(a) / maxA) * 0.7}
                      />
                    )}
                  </g>
                );
              })}
              <text
                x={x(c)}
                y="384"
                textAnchor="middle"
                className="oa-svg-label"
              >
                {c ? "256 HIDDEN" : "TOKENS"}
              </text>
            </g>
          ))}
          <text x="596" y="22" textAnchor="middle" className="oa-svg-label">
            RUBRIC
          </text>
          {Array.from({ length: 20 }, (_, i) => (
            <rect
              key={i}
              x="591"
              y={49 + i * 16}
              width="10"
              height="8"
              rx="1"
              fill={
                result?.outputs[i]?.probability != null ? "#89efd0" : "#101819"
              }
              fillOpacity={result?.outputs[i]?.probability ?? 1}
              stroke={meta?.questions[i]?.supported ? "#89efd0" : "#52645f"}
            />
          ))}
          <text x="596" y="384" textAnchor="middle" className="oa-svg-label">
            20 HEADS
          </text>
        </svg>
      </div>
      <div className="oa-layer-controls" aria-label="Inspect network layer">
        {[0, 1, 2, 3, 4].map((i) => (
          <button key={i} aria-pressed={layer === i} onClick={() => select(i)}>
            {i ? `Layer ${i}` : "Embedding"}
          </button>
        ))}
      </div>
      <div className="oa-legend">
        <span>━ Positive weight</span>
        <span>┄ Negative weight</span>
        <span>● Activation magnitude</span>
      </div>
    </div>
  );
}
export function OutcomeAnalyzer({
  onCitationBaseline,
}: {
  onCitationBaseline: () => void;
}) {
  const [meta, setMeta] = useState<Metadata | null>(null),
    [modelError, setModelError] = useState("");
  const [text, setText] = useState(""),
    [source, setSource] = useState("Original English paper"),
    [result, setResult] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState<"extract" | "example" | "predict" | null>(
      null,
    ),
    [error, setError] = useState(""),
    [notice, setNotice] = useState("");
  const [progress, setProgress] = useState<Progress | null>(null);
  const [animate, setAnimate] = useState(true);
  const [needsReview, setNeedsReview] = useState(false);
  const [reviewed, setReviewed] = useState(true),
    [layer, setLayer] = useState(1);
  const upload = useRef<HTMLInputElement>(null),
    active = useRef<AbortController | null>(null),
    mounted = useRef(true),
    metaPending = useRef(false),
    metaReady = useRef(false);
  async function loadModel() {
    if (metaPending.current) return;
    metaPending.current = true;
    try {
      const m = await api<Metadata>("/api/outcome-model");
      if (mounted.current) {
        setMeta(m);
        setModelError("");
        metaReady.current = true;
      }
    } catch (e) {
      if (mounted.current) setModelError((e as Error).message);
    } finally {
      metaPending.current = false;
    }
  }
  useEffect(() => {
    mounted.current = true;
    void loadModel();
    const timer = setInterval(() => {
      if (!metaReady.current) void loadModel();
    }, 5000);
    return () => {
      clearInterval(timer);
      mounted.current = false;
      active.current?.abort();
    };
  }, []);
  async function action(kind: "extract" | "example" | "predict", file?: File) {
    if (file && file.size > 10 * 1024 * 1024) {
      setError("Choose a PDF or text file smaller than 10 MB.");
      return;
    }
    active.current?.abort();
    const control = new AbortController();
    active.current = control;
    setBusy(kind);
    setProgress(null);
    setError("");
    setResult(null);
    setNotice("");
    try {
      if (kind === "predict") {
        const prediction = await streamPrediction(
          text,
          control.signal,
          (event) => {
            if (!control.signal.aborted)
              setProgress((previous) => ({ ...previous, ...event }));
          },
        );
        if (prediction.model_id !== meta?.model_id) {
          const current = await api<Metadata>("/api/outcome-model", {
            signal: control.signal,
          });
          if (current.model_id !== prediction.model_id)
            throw new Error(
              "The model changed during analysis. Please analyze again to keep the displayed weights and score aligned.",
            );
          setMeta(current);
        }
        setResult(prediction);
      } else if (kind === "example") {
        const d = await api<{ text: string; title: string }>(
          "/api/outcome-example?paper=attention",
          { signal: control.signal },
        );
        setText(d.text);
        setSource(d.title);
        setReviewed(false);
        setNeedsReview(true);
        setNotice(
          "Held-out example. Review the original paper text, then analyze.",
        );
      } else if (file) {
        const form = new FormData();
        form.append("file", file);
        const d = await api<{ text: string; filename: string; notice: string }>(
          "/api/extract?full_text=true",
          { method: "POST", body: form, signal: control.signal },
        );
        setText(d.text);
        setSource(d.filename);
        setReviewed(false);
        setNeedsReview(true);
        setNotice(d.notice);
      }
    } catch (e) {
      if (!control.signal.aborted) {
        setError((e as Error).message);
        setProgress(null);
      }
    } finally {
      if (active.current === control) setBusy(null);
    }
  }
  const values = (result?.layer_activations ?? progress?.layer_activations)?.[
      layer
    ],
    weights = meta?.layers[layer - 1]?.weights;
  return (
    <main
      id="workspace"
      className={`outcome-analyzer ${progress ? "oa-has-progress" : ""}`}
      aria-label="Outcome model analysis"
    >
      <header className="oa-heading">
        <div>
          <span className="oa-eyebrow">WORDING → RESEARCH OUTCOMES</span>
          <h1>Read between the claims.</h1>
        </div>
        <div className="oa-heading-actions">
          <span className="oa-experimental">Unvalidated experiment</span>
          <button onClick={onCitationBaseline}>
            Citation baseline <ArrowRight size={14} />
          </button>
        </div>
      </header>
      <div className="oa-layout">
        <section className="oa-input">
          <div className="oa-section-title">
            <h2>01 / Paper</h2>
            <span>English full text</span>
          </div>
          <div className="oa-upload-actions">
            <button disabled={!!busy} onClick={() => upload.current?.click()}>
              <Upload size={14} />
              Upload PDF / TXT
            </button>
            <button disabled={!!busy} onClick={() => void action("example")}>
              Try Attention
            </button>
          </div>
          <input
            ref={upload}
            className="oa-file"
            type="file"
            accept=".pdf,.txt,text/plain,application/pdf"
            aria-label="Upload original paper"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void action("extract", f);
              e.target.value = "";
            }}
          />
          <label className="oa-source" htmlFor="outcome-paper-text">
            {source}
          </label>
          <textarea
            id="outcome-paper-text"
            value={text}
            disabled={!!busy}
            onChange={(e) => {
              setText(e.target.value);
              setSource("Edited paper text");
              setNotice("");
              setReviewed(true);
              setNeedsReview(false);
              setResult(null);
              setProgress(null);
              setError("");
            }}
            placeholder="Paste the original paper or upload its full text…"
            spellCheck={false}
            aria-describedby="oa-input-note"
          />
          <div className="oa-text-meta">
            <span>{text.length.toLocaleString()} characters</span>
            <span>
              {result ? "All chunks processed" : "Processes every chunk"}
            </span>
          </div>
          <p id="oa-input-note">
            Topic masking uses a cloud LLM. The outcome transformer runs
            locally.
          </p>
          {notice && <p className="oa-notice">{notice}</p>}
          {needsReview && (
            <label className="oa-review">
              <input
                type="checkbox"
                checked={reviewed}
                onChange={(e) => setReviewed(e.target.checked)}
              />
              I reviewed the extracted paper text
            </label>
          )}
          <button
            className="oa-analyze"
            disabled={!!busy || !meta || text.trim().length < 100 || !reviewed}
            onClick={() => void action("predict")}
          >
            {busy === "predict"
              ? "Masking topics & analyzing…"
              : busy === "extract"
                ? "Extracting full text…"
                : busy === "example"
                  ? "Loading original paper…"
                  : "Analyze wording"}
            {busy ? (
              <RefreshCw
                className={animate ? "oa-spin" : undefined}
                size={16}
              />
            ) : (
              <ArrowRight size={16} />
            )}
          </button>
          <div className="oa-feedback" aria-live="polite">
            {busy === "predict" &&
              `${stages[progress?.stage ?? "queued"]}${progress?.completed ? ` · ${progress.completed} / ${progress.chunks} chunks` : " · every supplied chunk will be processed."}`}
            {error && <span role="alert">{error}</span>}
          </div>
        </section>
        <section className="oa-model">
          <div className="oa-section-title">
            <h2>02 / Transformer</h2>
            <span>
              {meta
                ? `${(meta.parameters / 1e6).toFixed(2)}M parameters`
                : modelError
                  ? "Waiting for checkpoint"
                  : "Loading weights…"}
            </span>
          </div>
          {modelError && (
            <div className="oa-model-error" role="status">
              {modelError}
              <button onClick={() => void loadModel()}>Retry model</button>
            </div>
          )}
          <Network
            meta={meta}
            result={result}
            layer={layer}
            select={setLayer}
            progress={progress?.model_id === meta?.model_id ? progress : null}
            working={busy === "predict"}
            animate={animate}
          />
          {(busy === "predict" || progress?.layer_activations) && (
            <section className="oa-live" aria-label="Inference measurements">
              <div className="oa-live-heading">
                <span role="status">
                  {busy === "predict"
                    ? stages[progress?.stage ?? "queued"]
                    : "Measured inference"}
                  {progress?.completed
                    ? ` · ${progress.completed} / ${progress.chunks} chunks`
                    : ""}
                </span>
                {busy === "predict" && (
                  <button onClick={() => setAnimate(!animate)}>
                    {animate ? "Pause motion" : "Resume motion"}
                  </button>
                )}
              </div>
              <div
                className={`oa-progress-track ${busy === "predict" && !progress?.completed && animate ? "oa-indeterminate" : ""}`}
              >
                <span
                  style={{
                    width:
                      progress?.completed && progress.chunks
                        ? `${(100 * progress.completed) / progress.chunks}%`
                        : "0%",
                  }}
                />
              </div>
              {!progress?.layer_activations && (
                <p className="oa-live-note">
                  Waiting animation · model measurements appear after the first
                  chunk.
                </p>
              )}
              {progress?.layer_activations && (
                <>
                  <div className="oa-live-grids">
                    <div>
                      <span className="oa-live-label">
                        Signed means · rows Embed→L4 / columns G1–G8
                      </span>
                      <div
                        className="oa-activation-grid"
                        role="img"
                        aria-label="Cumulative grouped activation magnitudes, five layers by eight groups"
                      >
                        {progress.layer_activations.flatMap((row, i) => {
                          const max = Math.max(
                            ...progress.layer_activations!.flat().map(Math.abs),
                            0.000001,
                          );
                          return row.map((v, j) => (
                            <span
                              key={`${i}-${j}`}
                              title={`${i ? `Layer ${i}` : "Embedding"}, group ${j + 1}: ${v.toFixed(4)}`}
                              style={{
                                background: v >= 0 ? "#89efd0" : "#e7b984",
                                opacity: 0.15 + (0.85 * Math.abs(v)) / max,
                              }}
                            />
                          ));
                        })}
                      </div>
                    </div>
                    <div>
                      <span className="oa-live-label">
                        Attention entropy · rows L1–L4 / columns H1–H4
                      </span>
                      <div
                        className="oa-entropy-grid"
                        role="img"
                        aria-label="Measured normalized attention entropy; higher means more dispersed attention"
                      >
                        {progress.head_entropy?.flatMap((row, i) =>
                          row.map((v, j) => (
                            <span
                              key={`${i}-${j}`}
                              title={`Layer ${i + 1}, head ${j + 1}: ${v.toFixed(4)}`}
                              style={{
                                background: `rgba(137,239,208,${0.08 + v * 0.4})`,
                              }}
                            >
                              {v.toFixed(2)}
                            </span>
                          )),
                        )}
                      </div>
                    </div>
                  </div>
                  <div
                    className="oa-chunk-trace"
                    role="img"
                    aria-label="Per-chunk mean supported-head probability, from zero to one. This is not the final rubric index."
                  >
                    {progress.chunk_summaries?.map((c) => (
                      <span
                        key={c.chunk}
                        title={`Chunk ${c.chunk}: ${(c.mean_supported_probability * 100).toFixed(2)}%, ${c.tokens} tokens`}
                        style={{
                          height: `${Math.max(1, c.mean_supported_probability * 100)}%`,
                        }}
                      />
                    ))}
                  </div>
                  <p className="oa-live-note">
                    Chunk predictions · 0–100% mean across supported heads;
                    final index averages dimensions.
                  </p>
                </>
              )}
            </section>
          )}
          <div className="oa-telemetry">
            <div>
              <span>Training papers</span>
              <strong>{meta?.training_papers ?? "—"}</strong>
            </div>
            <div>
              <span>Input tokens</span>
              <strong>
                {(result?.tokens ?? progress?.tokens)?.toLocaleString() ?? "—"}
              </strong>
            </div>
            <div>
              <span>Chunks</span>
              <strong>
                {result?.chunks ??
                  (progress?.chunks
                    ? `${progress.completed ?? 0} / ${progress.chunks}`
                    : "—")}
              </strong>
            </div>
            <div>
              <span>Topic text masked</span>
              <strong>
                {(result?.masked_fraction ?? progress?.masked_fraction) != null
                  ? `${pct(result?.masked_fraction ?? progress?.masked_fraction)}%`
                  : "—"}
              </strong>
            </div>
          </div>
          <details className="oa-inspector">
            <summary>
              {layer ? `Layer ${layer}` : "Embedding"} · inspect measurements
            </summary>
            <p>
              {meta?.diagram_note ??
                "Weights will appear when the outcome model is available."}
            </p>
            {values ? (
              <>
                <table>
                  <caption>Grouped mean activations</caption>
                  <thead>
                    <tr>
                      <th>Group</th>
                      <th>Hidden dimensions</th>
                      <th>Mean activation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {values.map((v, i) => (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        <td>
                          {i * 32 + 1}–{(i + 1) * 32}
                        </td>
                        <td>{v.toFixed(5)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {layer > 0 && (
                  <div className="oa-entropy">
                    <span>Attention head entropy · normalized</span>
                    {(result?.head_entropy ?? progress?.head_entropy)?.[
                      layer - 1
                    ]?.map((v, i) => (
                      <div key={i}>
                        <span>Head {i + 1}</span>
                        <meter
                          min="0"
                          max="1"
                          value={v}
                          aria-label={`Layer ${layer}, attention head ${i + 1} normalized entropy`}
                        />
                        <span>{v.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <p>
                Analyze a paper to inspect its measured activations and
                attention-head entropy.
              </p>
            )}
            {!!progress?.chunk_summaries?.length && (
              <details>
                <summary>Exact chunk measurements</summary>
                <table>
                  <caption>Mean supported-head probability per chunk</caption>
                  <thead>
                    <tr>
                      <th>Chunk</th>
                      <th>Tokens</th>
                      <th>Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {progress.chunk_summaries.map((c) => (
                      <tr key={c.chunk}>
                        <td>{c.chunk}</td>
                        <td>{c.tokens}</td>
                        <td>
                          {(100 * c.mean_supported_probability).toFixed(3)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
            )}
            {weights && (
              <details>
                <summary>Grouped projection weights</summary>
                <div
                  className="oa-weight-table"
                  tabIndex={0}
                  role="region"
                  aria-label="Grouped projection weights"
                >
                  <table>
                    <caption>
                      Signed mean weights · destination × source group
                    </caption>
                    <thead>
                      <tr>
                        <th>To / from</th>
                        {Array.from({ length: 8 }, (_, i) => (
                          <th key={i}>{i + 1}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {weights.map((row, i) => (
                        <tr key={i}>
                          <th>{i + 1}</th>
                          {row.map((v, j) => (
                            <td key={j}>{v.toFixed(4)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
          </details>
        </section>
        <section className="oa-output">
          <div className="oa-section-title">
            <h2>03 / Outcomes</h2>
            <span>Experimental</span>
          </div>
          <div className="oa-score" aria-live="polite">
            <span>{pct(result?.score)}</span>
            <small>/ 100</small>
          </div>
          <h3>Predicted rubric index</h3>
          <p className="oa-score-note">
            An equal average of four dimension scores. Not a calibrated
            probability of success.
          </p>
          <div className="oa-dimensions">
            {(
              result?.dimensions ??
              ["validation", "uptake", "utility", "durability"].map((id) => ({
                id,
                value: null,
                supported: 0,
                total: 5,
              }))
            ).map((d) => (
              <div key={d.id}>
                <div>
                  <span>{label(d.id)}</span>
                  <strong>
                    {d.value == null ? "—" : `${pct(d.value)}/100`}
                  </strong>
                </div>
                <div className="oa-track">
                  <span style={{ width: `${(d.value ?? 0) * 100}%` }} />
                </div>
                <small>
                  {result
                    ? `${d.supported} / ${d.total} heads supported`
                    : busy === "predict"
                      ? "Analyzing…"
                      : "Awaiting paper"}
                </small>
              </div>
            ))}
          </div>
          <details className="oa-questions">
            <summary>Inspect all 20 questions</summary>
            {(
              result?.outputs ??
              meta?.questions.map((q) => ({
                ...q,
                probability: null,
                training_yes: 0,
                training_no: 0,
              })) ??
              []
            ).map((q) => (
              <article key={q.id}>
                <div>
                  <span>{label(q.dimension)}</span>
                  <strong>
                    {q.probability == null ? "—" : `${pct(q.probability)}/100`}
                  </strong>
                </div>
                <p>{q.question}</p>
                {result && (
                  <small>
                    {q.training_yes} yes / {q.training_no} no in training
                    {q.probability == null ? " · unsupported head" : ""}
                  </small>
                )}
              </article>
            ))}
          </details>
          <p className="oa-output-note">
            Inspect training labels and held-out tests before interpreting a
            score.
          </p>
        </section>
      </div>
    </main>
  );
}
