import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Link,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

// ─────────────────────────────────────────────────────────────────────────────
// All verified benchmark results · seed=42 · 2026-05-18
// ─────────────────────────────────────────────────────────────────────────────

type ModelResult = {
  model: string;
  r2?: number;
  rmse?: number;
  acc?: number;
  f1?: number;
  auroc?: number;
  nrmse?: number;
  rel_l2?: number;
  runtime: number;
  pass: boolean;
};

type Benchmark = {
  key: string;
  name: string;
  citation: string;
  url: string;
  shape: string;
  n: string;
  capability: string;
  tier: 0 | 1 | 2 | 3 | 4;
  primaryMetric: string;
  threshold: string;
  thresholdNote?: string;
  results: ModelResult[];
};

const DATA: Benchmark[] = [
  // ── Scalar Regression ────────────────────────────────────────────────────────
  {
    key: "tabular.california_housing",
    name: "California Housing",
    citation: "Pace & Barry (1997) Statistics & Probability Letters",
    url: "https://doi.org/10.1016/S0167-7160(97)00010-0",
    shape: "8 → 1",
    n: "20,640",
    capability: "Scalar Regression",
    tier: 1,
    primaryMetric: "R²",
    threshold: "R² ≥ 0.75",
    results: [
      { model: "sklearn.random_forest", r2: 0.8062, rmse: 0.504, runtime: 2.95, pass: true },
      { model: "sklearn.gradient_boosting", r2: 0.7756, rmse: 0.542, runtime: 3.23, pass: true },
      { model: "sklearn.mlp", r2: 0.4016, rmse: 0.886, runtime: 68.6, pass: false },
      { model: "xgboost.xgbregressor", r2: 0.8469, rmse: 0.448, runtime: 1.28, pass: true },
      { model: "pytorch.mlp", r2: 0.7993, rmse: 0.513, runtime: 25.0, pass: true },
      { model: "pytorch.residual_mlp", r2: 0.8063, rmse: 0.504, runtime: 57.4, pass: true },
    ],
  },
  {
    key: "tabular.concrete_strength",
    name: "Concrete Compressive Strength",
    citation: "Yeh (1998) Cement and Concrete Research",
    url: "https://doi.org/10.1016/S0008-8846(98)00165-3",
    shape: "8 → 1",
    n: "1,030",
    capability: "Scalar Regression",
    tier: 1,
    primaryMetric: "R²",
    threshold: "R² ≥ 0.80",
    results: [
      { model: "sklearn.random_forest", r2: 0.8819, rmse: 5.52, runtime: 0.18, pass: true },
      { model: "sklearn.gradient_boosting", r2: 0.8829, rmse: 5.49, runtime: 0.10, pass: true },
      { model: "sklearn.mlp", r2: 0.7881, rmse: 7.39, runtime: 16.1, pass: false },
      { model: "xgboost.xgbregressor", r2: 0.9274, rmse: 4.32, runtime: 1.05, pass: true },
    ],
  },
  {
    key: "tabular.energy_efficiency",
    name: "Building Energy Efficiency",
    citation: "Tsanas & Xifara (2012) Energy & Buildings",
    url: "https://doi.org/10.1016/j.enbuild.2012.03.003",
    shape: "8 → 1",
    n: "768",
    capability: "Scalar Regression",
    tier: 1,
    primaryMetric: "R²",
    threshold: "R² ≥ 0.90",
    results: [
      { model: "sklearn.gradient_boosting", r2: 0.9904, rmse: 0.979, runtime: 0.05, pass: true },
      { model: "sklearn.mlp", r2: 0.6406, rmse: 6.00, runtime: 6.02, pass: false },
      { model: "xgboost.xgbregressor", r2: 0.9962, rmse: 0.619, runtime: 0.95, pass: true },
      { model: "pytorch.mlp", r2: 0.9729, rmse: 1.65, runtime: 1.03, pass: true },
      { model: "pytorch.residual_mlp", r2: 0.9933, rmse: 0.820, runtime: 2.53, pass: true },
    ],
  },
  {
    key: "tabular.airfoil_noise",
    name: "NASA Airfoil Self-Noise",
    citation: "Brooks, Pope & Marcolini (1989) NASA Report TM 100514",
    url: "https://archive.ics.uci.edu/dataset/291/airfoil+self+noise",
    shape: "5 → 1",
    n: "1,503",
    capability: "Scalar Regression",
    tier: 1,
    primaryMetric: "R²",
    threshold: "R² ≥ 0.85",
    thresholdNote: "sklearn.mlp failed (unscaled features)",
    results: [
      { model: "sklearn.random_forest", r2: 0.9347, rmse: 1.81, runtime: 0.18, pass: true },
      { model: "sklearn.gradient_boosting", r2: 0.8366, rmse: 2.86, runtime: 0.07, pass: false },
      { model: "xgboost.xgbregressor", r2: 0.9571, rmse: 1.47, runtime: 1.20, pass: true },
    ],
  },
  {
    key: "tabular.yacht_dynamics",
    name: "Yacht Hydrodynamics",
    citation: "Gerritsma, Onnink & Versluis (1981) CFD — SHIP DESIGN",
    url: "https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics",
    shape: "6 → 1",
    n: "308",
    capability: "Scalar Regression",
    tier: 1,
    primaryMetric: "R²",
    threshold: "R² ≥ 0.95",
    results: [
      { model: "sklearn.gradient_boosting", r2: 0.9975, rmse: 0.610, runtime: 0.03, pass: true },
      { model: "xgboost.xgbregressor", r2: 0.9981, rmse: 0.527, runtime: 0.91, pass: true },
      { model: "pytorch.mlp", r2: 0.9775, rmse: 1.83, runtime: 0.62, pass: true },
      { model: "pytorch.residual_mlp", r2: 0.9834, rmse: 1.57, runtime: 0.94, pass: true },
    ],
  },
  {
    key: "tabular.superconductor",
    name: "Superconductor Critical Temperature",
    citation: "Hamidieh (2018) Computational Materials Science",
    url: "https://doi.org/10.1016/j.commatsci.2018.07.052",
    shape: "81 → 1",
    n: "21,263",
    capability: "Scalar Regression",
    tier: 1,
    primaryMetric: "R²",
    threshold: "R² ≥ 0.90",
    thresholdNote: "Key materials-science surrogate benchmark",
    results: [
      { model: "sklearn.random_forest", r2: 0.9298, rmse: 8.99, runtime: 0, pass: true },
      { model: "sklearn.gradient_boosting", r2: 0.8678, rmse: 12.33, runtime: 0, pass: false },
      { model: "sklearn.mlp", r2: 0.6984, rmse: 18.63, runtime: 0, pass: false },
      { model: "xgboost.xgbregressor", r2: 0.9295, rmse: 9.01, runtime: 0, pass: true },
      { model: "pytorch.mlp", r2: 0.8912, rmse: 11.19, runtime: 0, pass: false },
      { model: "pytorch.residual_mlp", r2: 0.9035, rmse: 10.54, runtime: 0, pass: true },
    ],
  },
  {
    key: "tabular.diabetes",
    name: "Diabetes Progression",
    citation: "Efron, Hastie, Johnstone & Tibshirani (2004) Annals of Statistics",
    url: "https://doi.org/10.1214/009053604000000067",
    shape: "10 → 1",
    n: "442",
    capability: "Scalar Regression",
    tier: 1,
    primaryMetric: "R²",
    threshold: "R² ≥ 0.40",
    thresholdNote: "Inherently noisy; low R² expected",
    results: [
      { model: "sklearn.random_forest", r2: 0.4633, rmse: 54.5, runtime: 2.41, pass: true },
      { model: "sklearn.gradient_boosting", r2: 0.4241, rmse: 56.4, runtime: 0.07, pass: true },
      { model: "sklearn.mlp", r2: 0.4808, rmse: 53.6, runtime: 39.4, pass: true },
      { model: "xgboost.xgbregressor", r2: 0.3951, rmse: 57.8, runtime: 0.98, pass: false },
      { model: "pytorch.mlp", r2: 0.5280, rmse: 50.8, runtime: 0.61, pass: true },
    ],
  },
  // ── Multi-output Regression ─────────────────────────────────────────────────
  {
    key: "multioutput.scm20d",
    name: "SCM20d Supply Chain (20 targets)",
    citation: "Spyromitros-Xioufis et al. (2016) Machine Learning",
    url: "https://doi.org/10.1007/s10994-016-5546-z",
    shape: "61 → 20",
    n: "8,966",
    capability: "Multi-output Regression",
    tier: 1,
    primaryMetric: "R²",
    threshold: "avg R² ≥ 0.60",
    results: [
      { model: "sklearn.random_forest", r2: 0.8771, rmse: 94.2, runtime: 8.42, pass: true },
      { model: "sklearn.gradient_boosting", r2: 0.7320, rmse: 139.5, runtime: 0, pass: true },
      { model: "sklearn.mlp", r2: 0.6734, rmse: 153.9, runtime: 0, pass: true },
      { model: "xgboost.xgbregressor", r2: 0.8874, rmse: 90.4, runtime: 0, pass: true },
      { model: "pytorch.mlp", r2: 0.8187, rmse: 114.3, runtime: 13.5, pass: true },
      { model: "pytorch.residual_mlp", r2: 0.8876, rmse: 89.8, runtime: 25.0, pass: true },
    ],
  },
  {
    key: "synthetic.multioutput_2d",
    name: "Synthetic Multi-output (8 → 2)",
    citation: "SURGE inline fixture — Ax + ε",
    url: "",
    shape: "8 → 2",
    n: "600",
    capability: "Multi-output Regression",
    tier: 0,
    primaryMetric: "R²",
    threshold: "R² ≥ 0.80",
    results: [
      { model: "sklearn.random_forest", r2: 0.8640, runtime: 0.13, pass: true },
      { model: "sklearn.mlp", r2: 0.9852, runtime: 0.41, pass: true },
      { model: "xgboost.xgbregressor", r2: 0.9412, runtime: 1.51, pass: true },
      { model: "pytorch.mlp", r2: 0.9907, rmse: 0.129, runtime: 0.69, pass: true },
      { model: "pytorch.residual_mlp", r2: 0.9877, rmse: 0.149, runtime: 1.68, pass: true },
    ],
  },
  // ── Standard Classification ─────────────────────────────────────────────────
  {
    key: "tabular.breast_cancer",
    name: "Wisconsin Breast Cancer (WDBC)",
    citation: "Mangasarian & Wolberg (1990) University of Wisconsin",
    url: "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    shape: "30 → 2",
    n: "569",
    capability: "Tabular Classification",
    tier: 1,
    primaryMetric: "Accuracy",
    threshold: "Acc ≥ 0.95",
    results: [
      { model: "sklearn.random_forest_clf", acc: 0.9561, f1: 0.9526, auroc: 0.9939, runtime: 0.11, pass: true },
      { model: "sklearn.logistic_reg", acc: 0.9825, f1: 0.9812, auroc: 0.9954, runtime: 0.01, pass: true },
      { model: "xgboost.xgbclassifier", acc: 0.9561, f1: 0.9521, auroc: 0.9927, runtime: 0.45, pass: true },
      { model: "pytorch.mlp_classifier", acc: 0.9561, f1: 0.9535, auroc: 0.9934, runtime: 0.56, pass: true },
    ],
  },
  {
    key: "tabular.digits",
    name: "Optical Recognition of Handwritten Digits",
    citation: "Alpaydin & Kaynak (1998) UCI",
    url: "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
    shape: "64 → 10",
    n: "1,797",
    capability: "Tabular Classification",
    tier: 1,
    primaryMetric: "Accuracy",
    threshold: "Acc ≥ 0.97",
    results: [
      { model: "sklearn.random_forest_clf", acc: 0.9639, f1: 0.9634, auroc: 0.9992, runtime: 0.19, pass: false },
      { model: "sklearn.gradient_boosting_clf", acc: 0.9528, f1: 0.9521, auroc: 0.9989, runtime: 5.22, pass: false },
      { model: "sklearn.logistic_reg", acc: 0.9722, f1: 0.9719, auroc: 0.9991, runtime: 0.49, pass: true },
      { model: "xgboost.xgbclassifier", acc: 0.9639, f1: 0.9633, auroc: 0.9988, runtime: 5.43, pass: false },
      { model: "pytorch.mlp_classifier", acc: 0.9833, f1: 0.9830, auroc: 0.9994, runtime: 2.24, pass: true },
    ],
  },
  {
    key: "tabular.iris",
    name: "Iris (3-class)",
    citation: "Fisher (1936) Annals of Eugenics",
    url: "https://doi.org/10.1111/j.1469-1809.1936.tb02137.x",
    shape: "4 → 3",
    n: "150",
    capability: "Tabular Classification",
    tier: 1,
    primaryMetric: "Accuracy",
    threshold: "Acc ≥ 0.95",
    results: [
      { model: "sklearn.random_forest_clf", acc: 0.9211, f1: 0.9230, auroc: 0.9892, runtime: 1.96, pass: false },
      { model: "sklearn.gradient_boosting_clf", acc: 0.9737, f1: 0.9743, auroc: 0.9908, runtime: 0.23, pass: true },
      { model: "sklearn.logistic_reg", acc: 0.9474, f1: 0.9487, auroc: 0.9969, runtime: 0.01, pass: false },
      { model: "pytorch.mlp_classifier", acc: 0.9333, f1: 0.9333, auroc: 0.9950, runtime: 0.19, pass: false },
    ],
  },
  {
    key: "tabular.wine",
    name: "UCI Wine (3-class)",
    citation: "Aeberhard, Coomans & De Vel (1992) UCI",
    url: "https://archive.ics.uci.edu/dataset/109/wine",
    shape: "13 → 3",
    n: "178",
    capability: "Tabular Classification",
    tier: 1,
    primaryMetric: "Accuracy",
    threshold: "Acc ≥ 0.90",
    results: [
      { model: "sklearn.random_forest_clf", acc: 1.0, f1: 1.0, auroc: 1.0, runtime: 1.90, pass: true },
      { model: "sklearn.logistic_reg", acc: 1.0, f1: 1.0, auroc: 1.0, runtime: 0.00, pass: true },
      { model: "xgboost.xgbclassifier", acc: 1.0, f1: 1.0, auroc: 1.0, runtime: 0.62, pass: true },
      { model: "pytorch.mlp_classifier", acc: 0.9722, f1: 0.9710, auroc: 0.9989, runtime: 0.27, pass: true },
    ],
  },
  // ── Scientific Classification ───────────────────────────────────────────────
  {
    key: "classification.covertype",
    name: "Forest Covertype (7-class)",
    citation: "Blackard & Dean (1999) Computers & Electronics in Agriculture",
    url: "https://archive.ics.uci.edu/dataset/31/covertype",
    shape: "54 → 7",
    n: "20k (subsample of 581k)",
    capability: "Scientific Classification",
    tier: 1,
    primaryMetric: "Accuracy",
    threshold: "Acc ≥ 0.85",
    thresholdNote: "Threshold achievable with full 581k; 20k subsample harder",
    results: [
      { model: "sklearn.random_forest_clf", acc: 0.8320, f1: 0.6996, runtime: 0.47, pass: false },
      { model: "sklearn.gradient_boosting_clf", acc: 0.7592, f1: 0.6244, runtime: 25.0, pass: false },
      { model: "sklearn.logistic_reg", acc: 0.7110, f1: 0.4424, runtime: 2.20, pass: false },
      { model: "xgboost.xgbclassifier", acc: 0.8303, f1: 0.7504, runtime: 4.84, pass: false },
      { model: "pytorch.mlp_classifier", acc: 0.8093, f1: 0.6982, auroc: 0.9676, runtime: 20.0, pass: false },
    ],
  },
  {
    key: "classification.plasma_stability",
    name: "Electrical Grid Stability (plasma proxy)",
    citation: "Arzamasov, Bohm & Jochem (2018) IEEE PMAPS",
    url: "https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data",
    shape: "12 → 2",
    n: "10,000",
    capability: "Scientific Classification",
    tier: 2,
    primaryMetric: "Accuracy",
    threshold: "Acc ≥ 0.92",
    results: [
      { model: "sklearn.random_forest_clf", acc: 0.9245, f1: 0.9170, auroc: 0.9808, runtime: 0, pass: true },
      { model: "sklearn.gradient_boosting_clf", acc: 0.9360, f1: 0.9297, auroc: 0.9841, runtime: 0, pass: true },
      { model: "sklearn.logistic_reg", acc: 0.8200, f1: 0.8009, runtime: 0, pass: false },
      { model: "xgboost.xgbclassifier", acc: 0.9520, f1: 0.9476, auroc: 0.9926, runtime: 0, pass: true },
      { model: "pytorch.mlp_classifier", acc: 0.9775, f1: 0.9756, auroc: 0.9980, runtime: 0, pass: true },
    ],
  },
  {
    key: "classification.flow_regime",
    name: "CFD Flow Regime (4-class)",
    citation: "SURGE inline fixture — Mach / Re / AoA regime labeling",
    url: "",
    shape: "3 → 4",
    n: "800",
    capability: "Scientific Classification",
    tier: 0,
    primaryMetric: "Accuracy",
    threshold: "Acc ≥ 0.85",
    results: [
      { model: "sklearn.random_forest_clf", acc: 0.9437, f1: 0.9069, auroc: 0.9625, runtime: 0.11, pass: true },
      { model: "sklearn.gradient_boosting_clf", acc: 0.9437, f1: 0.9087, runtime: 0.38, pass: true },
      { model: "sklearn.logistic_reg", acc: 0.8938, f1: 0.8012, auroc: 0.9316, runtime: 0.01, pass: true },
      { model: "xgboost.xgbclassifier", acc: 0.9375, f1: 0.8936, auroc: 0.9621, runtime: 4.10, pass: true },
      { model: "pytorch.mlp_classifier", acc: 0.9313, f1: 0.8872, auroc: 0.9417, runtime: 1.90, pass: true },
    ],
  },
  // ── Time Series / Forecasting ────────────────────────────────────────────────
  {
    key: "sequence.lorenz63",
    name: "Lorenz-63 Chaotic Forecasting",
    citation: "Lorenz (1963) J. Atmospheric Sciences",
    url: "https://doi.org/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2",
    shape: "3×20 → 3×20 (rollout)",
    n: "1200 trajectories",
    capability: "Time Series / Forecasting",
    tier: 0,
    primaryMetric: "NRMSE",
    threshold: "NRMSE ≤ 0.10",
    results: [
      { model: "sklearn.random_forest", nrmse: 0.0256, runtime: 2.30, pass: true },
      { model: "sklearn.gradient_boosting", nrmse: 0.0274, runtime: 100.2, pass: true },
      { model: "xgboost.xgbregressor", nrmse: 0.0235, runtime: 86.9, pass: true },
      { model: "pytorch.mlp", r2: 0.9935, nrmse: 0.0369, runtime: 1.79, pass: true },
      { model: "pytorch.residual_mlp", r2: 0.9989, nrmse: 0.0141, runtime: 3.73, pass: true },
      { model: "pytorch.cnn1d", r2: 0.9984, nrmse: 0.0179, runtime: 166.1, pass: true },
      { model: "pytorch.lstm", r2: 0.9985, nrmse: 0.0171, runtime: 119.5, pass: true },
      { model: "pytorch.gru", r2: 0.9995, nrmse: 0.0097, runtime: 90.4, pass: true },
    ],
  },
  // ── 1D PDE Operator Learning ──────────────────────────────────────────────────
  {
    key: "pde.burgers_1d",
    name: "Viscous Burgers 1D (Inline FD Solver)",
    citation: "Li et al. — FNO paper (2021) ICLR; solver: Burgers (1948)",
    url: "https://arxiv.org/abs/2010.08895",
    shape: "64 → 64 (field)",
    n: "1024 simulations",
    capability: "1D PDE Operator",
    tier: 1,
    primaryMetric: "NRMSE / Rel-L2",
    threshold: "NRMSE ≤ 0.10",
    results: [
      { model: "sklearn.random_forest", rel_l2: 0.6137, runtime: 1.21, pass: false },
      { model: "sklearn.gradient_boosting", rel_l2: 0.0530, runtime: 71.5, pass: true },
      { model: "xgboost.xgbregressor", rel_l2: 0.0534, runtime: 47.7, pass: true },
      { model: "pytorch.mlp", r2: 0.9149, nrmse: 0.2912, runtime: 1.98, pass: false },
      { model: "pytorch.residual_mlp", r2: 0.9868, nrmse: 0.1146, runtime: 3.41, pass: false },
      { model: "pytorch.cnn1d", r2: 0.9984, nrmse: 0.0419, runtime: 144.7, pass: true },
      { model: "pytorch.gru", r2: 0.9586, nrmse: 0.2028, runtime: 69.9, pass: false },
      { model: "pytorch.fno1d", r2: 0.9997, nrmse: 0.0180, runtime: 202.4, pass: true },
      { model: "pytorch.deeponet", r2: 0.8844, nrmse: 0.3405, runtime: 3.40, pass: false },
    ],
  },
  // ── Vision ────────────────────────────────────────────────────────────────────
  {
    key: "vision.mnist",
    name: "MNIST Handwritten Digits",
    citation: "LeCun, Bottou, Bengio & Haffner (1998) Proc. IEEE",
    url: "http://yann.lecun.com/exdb/mnist/",
    shape: "28×28 → 10",
    n: "70,000",
    capability: "Vision",
    tier: 2,
    primaryMetric: "Accuracy",
    threshold: "Acc ≥ 0.99",
    thresholdNote: "Published LeNet-5: 99.05%; 98.9% with default epochs",
    results: [
      { model: "pytorch.lenet5", acc: 0.9889, f1: 0.9888, auroc: 0.9999, runtime: 431, pass: false },
    ],
  },
  // ── Scientific Domain ─────────────────────────────────────────────────────────
  {
    key: "fusion.m3dc1_sample",
    name: "M3DC1 Tokamak Equilibrium Surrogate",
    citation: "M3DC1 Group — NSTX-U experimental data, PPPL",
    url: "https://m3dc1.pppl.gov",
    shape: "13 → 1",
    n: "~500",
    capability: "Scientific Domain",
    tier: 2,
    primaryMetric: "R²",
    threshold: "R² ≥ 0.85",
    results: [
      { model: "sklearn.random_forest", r2: 0.8092, rmse: 0.830, runtime: 0.41, pass: false },
      { model: "sklearn.gradient_boosting", r2: 0.8950, rmse: 0.616, runtime: 0.62, pass: true },
      { model: "sklearn.mlp", r2: 0.9938, rmse: 0.150, runtime: 11.1, pass: true },
      { model: "xgboost.xgbregressor", r2: 0.9131, rmse: 0.561, runtime: 1.03, pass: true },
    ],
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Capability ordering
// ─────────────────────────────────────────────────────────────────────────────

const CAPABILITIES = [
  "Scalar Regression",
  "Multi-output Regression",
  "Tabular Classification",
  "Scientific Classification",
  "Time Series / Forecasting",
  "1D PDE Operator",
  "Vision",
  "Scientific Domain",
] as const;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function best(results: ModelResult[], key: keyof ModelResult): number | undefined {
  const vals = results.map((r) => r[key] as number | undefined).filter((v) => v !== undefined) as number[];
  if (!vals.length) return undefined;
  const lower = ["rmse", "nrmse", "rel_l2", "runtime"].includes(key as string);
  return lower ? Math.min(...vals) : Math.max(...vals);
}

function cell(
  val: number | undefined,
  bestVal: number | undefined,
  decimals = 4,
  pct = false
): string {
  if (val === undefined) return "—";
  const formatted = pct ? `${(val * 100).toFixed(2)}%` : val.toFixed(decimals);
  return val === bestVal ? `${formatted} ★` : formatted;
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark table
// ─────────────────────────────────────────────────────────────────────────────

function BmTable({ bm }: { bm: Benchmark }) {
  const isClf = bm.results.some((r) => r.acc !== undefined);
  const hasFl2 = bm.results.some((r) => r.rel_l2 !== undefined);
  const hasNrmse = bm.results.some((r) => r.nrmse !== undefined);
  const hasAuroc = bm.results.some((r) => r.auroc !== undefined);
  const hasR2 = bm.results.some((r) => r.r2 !== undefined);
  const hasRuntime = bm.results.some((r) => r.runtime > 0);

  const headers: string[] = ["Model", "Pass"];
  if (isClf) {
    headers.push("Accuracy", "F1 (macro)");
    if (hasAuroc) headers.push("AUROC");
  } else {
    if (hasR2) headers.push("R²");
    if (hasNrmse) headers.push("NRMSE ↓");
    if (hasFl2) headers.push("Rel-L2 ↓");
    if (!hasNrmse && !hasFl2) headers.push("RMSE ↓");
  }
  if (hasRuntime) headers.push("Time (s)");

  const bAcc = best(bm.results, "acc");
  const bF1 = best(bm.results, "f1");
  const bAuroc = best(bm.results, "auroc");
  const bR2 = best(bm.results, "r2");
  const bNrmse = best(bm.results, "nrmse");
  const bFl2 = best(bm.results, "rel_l2");
  const bRmse = best(bm.results, "rmse");
  const bRuntime = best(bm.results, "runtime");

  const rows = bm.results.map((r) => {
    const row: string[] = [r.model, r.pass ? "PASS" : "FAIL"];
    if (isClf) {
      row.push(cell(r.acc, bAcc, 4, true), cell(r.f1, bF1));
      if (hasAuroc) row.push(cell(r.auroc, bAuroc));
    } else {
      if (hasR2) row.push(cell(r.r2, bR2));
      if (hasNrmse) row.push(cell(r.nrmse, bNrmse));
      if (hasFl2) row.push(cell(r.rel_l2, bFl2));
      if (!hasNrmse && !hasFl2) row.push(cell(r.rmse, bRmse, 3));
    }
    if (hasRuntime) row.push(r.runtime > 0 ? cell(r.runtime, bRuntime, 1) + "s" : "—");
    return row;
  });

  const tones = bm.results.map((r) => (r.pass ? undefined : ("warning" as const)));

  return (
    <Stack gap={6}>
      <Row gap={8} style={{ alignItems: "center", flexWrap: "wrap" }}>
        <Text weight="medium">{bm.name}</Text>
        <Pill
          size="small"
          tone={bm.tier === 0 ? "success" : bm.tier === 1 ? "info" : "neutral"}
        >
          Tier {bm.tier}
        </Pill>
        <Text tone="secondary" size="small">
          {bm.shape} · n={bm.n}
        </Text>
        <Text tone="secondary" size="small">threshold: {bm.threshold}</Text>
        {bm.url ? (
          <Link href={bm.url} size="small">{bm.citation}</Link>
        ) : (
          <Text tone="secondary" size="small">{bm.citation}</Text>
        )}
      </Row>
      {bm.thresholdNote && (
        <Text tone="secondary" size="small" style={{ fontStyle: "italic" }}>
          Note: {bm.thresholdNote}
        </Text>
      )}
      <Table headers={headers} rows={rows} rowTone={tones} />
      <Text tone="secondary" size="small">
        ★ = best in column · PASS/FAIL vs. literature threshold · warning row = below threshold
      </Text>
    </Stack>
  );
}

function CapabilityBlock({ cap }: { cap: string }) {
  const bms = DATA.filter((b) => b.capability === cap);
  if (!bms.length) return null;
  const totalPass = bms.reduce((s, b) => s + b.results.filter((r) => r.pass).length, 0);
  const totalRun = bms.reduce((s, b) => s + b.results.length, 0);
  return (
    <Stack gap={14}>
      <Row gap={12} style={{ alignItems: "center" }}>
        <H2>{cap}</H2>
        <Text tone="secondary" size="small">
          {bms.length} benchmark{bms.length > 1 ? "s" : ""} · {totalPass}/{totalRun} model runs passed
        </Text>
      </Row>
      {bms.map((bm, i) => (
        <Stack key={bm.key} gap={6}>
          <BmTable bm={bm} />
          {i < bms.length - 1 && <Divider />}
        </Stack>
      ))}
    </Stack>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary charts
// ─────────────────────────────────────────────────────────────────────────────

function SummaryCharts() {
  const regressionBms = DATA.filter(
    (b) => b.results.some((r) => r.r2 !== undefined) &&
           !["Time Series / Forecasting", "1D PDE Operator"].includes(b.capability)
  );
  const r2Chart = regressionBms
    .map((b) => ({ label: b.key.split(".")[1].replace(/_/g, " "), value: best(b.results, "r2") ?? 0 }))
    .sort((a, c) => c.value - a.value);

  const classificationBms = DATA.filter((b) => b.results.some((r) => r.acc !== undefined));
  const accChart = classificationBms
    .map((b) => ({
      label: b.key.split(".")[1].replace(/_/g, " "),
      value: parseFloat(((best(b.results, "acc") ?? 0) * 100).toFixed(2)),
    }))
    .sort((a, c) => c.value - a.value);

  return (
    <Grid columns={2} gap={20}>
      <Stack gap={8}>
        <H3>Best R² per Regression Benchmark</H3>
        <BarChart data={r2Chart} xLabel="Benchmark" yLabel="Best R² (higher is better)" height={220} />
        <Text tone="secondary" size="small">Best model across all runs, seed=42, default hyperparameters.</Text>
      </Stack>
      <Stack gap={8}>
        <H3>Best Accuracy per Classification Benchmark (%)</H3>
        <BarChart data={accChart} xLabel="Benchmark" yLabel="Best Accuracy (%)" height={220} />
        <Text tone="secondary" size="small">Best model across all runs, seed=42, default hyperparameters.</Text>
      </Stack>
    </Grid>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Model leaderboard summary
// ─────────────────────────────────────────────────────────────────────────────

function ModelSummary() {
  return (
    <Stack gap={8}>
      <H2>Model Registry</H2>
      <Table
        headers={["Key", "Backend", "Architecture", "Best task"]}
        rows={[
          ["sklearn.random_forest", "scikit-learn", "Random Forest regressor", "Superconductor R²=0.930, Airfoil R²=0.935"],
          ["sklearn.gradient_boosting", "scikit-learn", "GBM (GBRT)", "Yacht R²=0.998, Energy R²=0.990"],
          ["sklearn.mlp", "scikit-learn", "MLP regressor", "M3DC1 R²=0.994 (best), Diabetes R²=0.481"],
          ["sklearn.random_forest_clf", "scikit-learn", "Random Forest classifier", "Flow regime 94.4%, Breast cancer 95.6%"],
          ["sklearn.gradient_boosting_clf", "scikit-learn", "GBM classifier", "Plasma stability 93.6%"],
          ["sklearn.logistic_reg", "scikit-learn", "Logistic Regression", "Wine 100%, Breast cancer 98.3%"],
          ["xgboost.xgbregressor", "XGBoost", "XGBoost regressor", "California R²=0.847, Yacht R²=0.998"],
          ["xgboost.xgbclassifier", "XGBoost", "XGBoost classifier", "Plasma 95.2%, Wine 100%"],
          ["pytorch.mlp", "PyTorch", "MLP (128-128-64)", "General tabular; fast neural baseline"],
          ["pytorch.residual_mlp", "PyTorch", "Residual MLP blocks", "SCM20d R²=0.888, Energy R²=0.993"],
          ["pytorch.mlp_classifier", "PyTorch", "MLP + CrossEntropy", "Plasma 97.8%, Digits 98.3%"],
          ["pytorch.cnn1d", "PyTorch", "Dilated 1D CNN", "Lorenz NRMSE=0.018, Burgers NRMSE=0.042"],
          ["pytorch.lstm", "PyTorch", "LSTM encoder-decoder", "Lorenz NRMSE=0.017"],
          ["pytorch.gru", "PyTorch", "GRU encoder-decoder", "Lorenz NRMSE=0.0097 (best overall)"],
          ["pytorch.fno1d", "PyTorch", "Fourier Neural Operator 1D", "Burgers NRMSE=0.018, R²=0.9997 (best)"],
          ["pytorch.deeponet", "PyTorch", "Deep Operator Network", "Burgers R²=0.884 (needs tuning)"],
          ["pytorch.lenet5", "PyTorch", "LeNet-5 (LeCun 1998)", "MNIST 98.9%"],
          ["pytorch.resnet20", "PyTorch", "ResNet-20 (He 2016)", "CIFAR-10 (pending)"],
          ["pytorch.resnet56", "PyTorch", "ResNet-56 (He 2016)", "CIFAR-10 deeper variant (pending)"],
        ]}
      />
    </Stack>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Pending benchmarks
// ─────────────────────────────────────────────────────────────────────────────

function PendingTable() {
  return (
    <Stack gap={8}>
      <H2>Pending / Higher-Tier Benchmarks</H2>
      <Table
        headers={["Key", "Dataset & Citation", "Capability", "Status"]}
        rows={[
          [
            "vision.cifar10",
            "CIFAR-10 · Krizhevsky (2009) · ResNet-20/56",
            "Vision",
            "Pending — GPU recommended; ResNet-20 published: 91.25%",
          ],
          [
            "pdebench.burgers_1d",
            "Takamoto et al. NeurIPS 2022 · HDF5 download · 1024-pt grid",
            "1D PDE (real data)",
            "Ready — requires: curl <darus_url> > Burgers_Rr1.0.hdf5",
          ],
          [
            "pdebench.darcy_2d",
            "Takamoto et al. NeurIPS 2022 · Darcy Flow 128×128",
            "2D PDE",
            "Ready — requires HDF5 download",
          ],
          [
            "pdebench.shallow_water_2d",
            "Takamoto et al. NeurIPS 2022 · Shallow Water 128×128",
            "2D PDE",
            "Ready — requires HDF5 download",
          ],
          [
            "thewell.gray_scott",
            "Ohana et al. NeurIPS 2024 · Gray-Scott 2D",
            "Reaction-Diffusion",
            "Ready — requires: pip install the-well",
          ],
          [
            "thewell.turbulence_2d",
            "Ohana et al. NeurIPS 2024 · 2D turbulence",
            "Turbulence",
            "Ready — requires the-well pkg",
          ],
          [
            "thewell.mhd",
            "Ohana et al. NeurIPS 2024 · 3D MHD",
            "MHD Plasma",
            "Ready — requires the-well pkg",
          ],
        ]}
      />
    </Stack>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Root canvas
// ─────────────────────────────────────────────────────────────────────────────

export default function SurgeBenchmarks() {
  const totalRuns = DATA.reduce((s, b) => s + b.results.length, 0);
  const totalPass = DATA.reduce((s, b) => s + b.results.filter((r) => r.pass).length, 0);
  const uniqueBenchmarks = DATA.length;

  return (
    <Stack gap={28} style={{ padding: "24px", maxWidth: "1100px" }}>
      {/* Header */}
      <Stack gap={4}>
        <H1>SURGE Benchmark Leaderboard</H1>
        <Text tone="secondary">
          Scientific surrogate model evaluation · all benchmarks verified · seed=42 · 2026-05-18
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value={String(uniqueBenchmarks)} label="Benchmarks" />
        <Stat value={String(totalRuns)} label="Model evaluations" />
        <Stat
          value={`${totalPass} / ${totalRuns}`}
          label="Passed threshold"
          tone={totalPass / totalRuns > 0.7 ? "success" : "warning"}
        />
        <Stat value="8" label="Capability domains" />
      </Grid>

      <Callout tone="info">
        <Text size="small">
          <strong>Key:</strong> ★ = best value in column · PASS/FAIL vs. literature threshold ·
          Tier 0 = inline (no download) · Tier 1 = internet on first run · Tier 2 = larger dataset / GPU-recommended ·
          All runs: default hyperparameters, 80/20 split, seed=42. HPO available via{" "}
          <code>--hpo --hpo-trials 20</code>.
        </Text>
      </Callout>

      <Divider />

      <SummaryCharts />

      <Divider />

      {CAPABILITIES.map((cap) => (
        <Stack key={cap} gap={0}>
          <CapabilityBlock cap={cap} />
          <Divider />
        </Stack>
      ))}

      <ModelSummary />

      <Divider />

      <PendingTable />

      <Text tone="secondary" size="small">
        SURGE · branch: model-bench · Run commands: python -m surge.benchmarks.run --leaderboard --tier 1 --mlflow --plot
      </Text>
    </Stack>
  );
}
