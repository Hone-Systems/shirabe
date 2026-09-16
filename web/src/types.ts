export type Metric = {
  n: number;
  positives: number;
  prevalence: number;
  roc_auc: number;
  average_precision: number;
  brier: number;
  log_loss: number;
  auc_ci95?: number[];
};
export type Feature = {
  name: string;
  label: string;
  value: number;
  z: number;
  contribution: number;
  coefficient: number;
  unusual: boolean;
};
export type Prediction = {
  transformer?: {
    cls_attention: number[][][];
    layer_cls: number[][];
    sequence_length: number;
    full_sequence_length: number;
  };
  model_id: string;
  probability: number;
  baseline: number;
  logit: number;
  intercept: number;
  word_count: number;
  features: Feature[];
  tokens: {
    text: string;
    start: number;
    end: number;
    attention?: number;
    category: string;
    categories: string[];
  }[];
  warnings: string[];
  seen_in_training: boolean;
};
export type Report = {
  model_id: string;
  trained_at: string;
  status: string;
  architecture: string;
  feature_count: number;
  dataset_sha256: string;
  seed: number;
  target: string;
  splits: { name: string; years: string; n: number; positive_rate: number }[];
  test: Metric;
  uncalibrated_test: Metric;
  comparisons: (Metric & { name: string; role: string })[];
  cohorts: {
    field: string;
    field_id: number;
    year: number;
    n: number;
    threshold: number;
    positives: number;
    prevalence: number;
  }[];
  field_results: (Metric & { field: string; field_id: number })[];
  field_transfer: (Metric & { field: string; field_id: number })[];
  discipline_probe: {
    accuracy: number;
    majority_baseline: number;
    description: string;
  };
  transformer?: {
    layers: number;
    heads: number;
    hidden_size: number;
    parameter_count: number;
  };
  epoch_history?: {
    epoch: number;
    train_loss: number;
    validation_loss: number;
    validation_auc: number;
  }[];
  best_epoch?: number;
  shuffled_token_test?: Metric;
  negative_control?: {
    runs: number;
    mean_auc: number;
    min_auc: number;
    max_auc: number;
  };
  learning_curve?: { n: number; train_auc: number; validation_auc: number }[];
  convergence?: {
    iteration_budget: number;
    iterations_used: number;
    train_loss: number;
    validation_loss: number;
  }[];
  roc: { fpr: number; tpr: number }[];
  reliability: { predicted: number; observed: number; n: number }[];
  features: {
    name: string;
    label: string;
    coefficient: number;
    mean: number;
    scale: number;
  }[];
  selected_c: number;
  candidates: { c: number; validation_log_loss: number }[];
  optimizer_iterations: number;
  compute: {
    device: string;
    cloud_cost_usd: number;
    duration_seconds: number;
    python: string;
    sklearn: string;
  };
  limitations: string[];
};
