"""
SURGE Dataset Explorer — Streamlit app
=======================================
Interactive visualizations for every organized benchmark dataset.

Charts are interactive (hover, zoom, pan, legend toggle, 3-D rotate) via Plotly.
Each tabular/scientific dataset also gets a scatter explorer where you choose
the X / Y / color axes from dropdowns.

Run with:
    streamlit run scripts/dataset_explorer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Make surge importable from the repo root
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SURGE Dataset Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    # Vision
    "MNIST": {
        "category": "Vision",
        "n": "70,000",
        "shape": "28×28 px, grayscale",
        "target": "Digit class (0–9)",
        "source": "torchvision (LeCun et al. 1998)",
        "benchmark": "vision.mnist",
        "status": "cached",
    },
    "CIFAR-10": {
        "category": "Vision",
        "n": "60,000",
        "shape": "32×32 px, RGB",
        "target": "Object class (10 categories)",
        "source": "torchvision (Krizhevsky 2009)",
        "benchmark": "vision.cifar10",
        "status": "cached",
    },
    # PDE / Sequence
    "Burgers 1-D (PDE)": {
        "category": "PDE",
        "n": "1,024",
        "shape": "64-point field",
        "target": "Solution at t=T",
        "source": "Inline FD solver (generated, seed=42)",
        "benchmark": "pde.burgers_1d",
        "status": "cached",
    },
    "Lorenz-63 (Sequence)": {
        "category": "Sequence",
        "n": "1,200 trajectories",
        "shape": "3×20 → 3×20",
        "target": "Future 20 steps",
        "source": "Inline RK4 solver (generated, seed=42)",
        "benchmark": "sequence.lorenz63",
        "status": "cached",
    },
    # Plasma / Classification
    "C-Mod Density Limit": {
        "category": "Plasma (classification)",
        "n": "7,196 (balanced)",
        "shape": "6 features",
        "target": "Density limit disruption (binary)",
        "source": "MIT-PSFC open_density_limit_database",
        "benchmark": "plasma.cmod_density_limit",
        "status": "cached",
    },
    "Plasma Stability (UCI)": {
        "category": "Classification",
        "n": "10,000",
        "shape": "12 features",
        "target": "Stable / unstable (binary)",
        "source": "UCI #471 (Arzamasov et al. 2018)",
        "benchmark": "classification.plasma_stability",
        "status": "cached",
    },
    # Local tokamak datasets
    "NSTX-U Equilibria": {
        "category": "Tokamak (local)",
        "n": "9,441",
        "shape": "42 features",
        "target": "gamma_VDE, gamma_TOKAM",
        "source": "Local PKL — NSTX-U experiment data",
        "benchmark": "(not yet wired)",
        "status": "local",
    },
    "SMART Equilibria": {
        "category": "Tokamak (local)",
        "n": "4,000",
        "shape": "22 features",
        "target": "gamma, gamma_TOKAM",
        "source": "Local PKL — SMART experiment data",
        "benchmark": "(not yet wired)",
        "status": "local",
    },
    # Tabular (partially cached — loads on demand)
    "Concrete Strength (Tabular)": {
        "category": "Tabular regression",
        "n": "1,030",
        "shape": "8 features",
        "target": "Compressive strength (MPa)",
        "source": "OpenML #4353 (Yeh 1998)",
        "benchmark": "tabular.concrete_strength",
        "status": "cached",
    },
}

# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_mnist():
    from surge.benchmarks.leaderboard import _load_mnist
    return _load_mnist()


@st.cache_data(show_spinner="Loading dataset…")
def load_cifar10():
    from surge.benchmarks.leaderboard import _load_cifar10
    return _load_cifar10()


@st.cache_data(show_spinner="Loading dataset…")
def load_burgers():
    from surge.benchmarks.leaderboard import _load_burgers_1d
    return _load_burgers_1d()


@st.cache_data(show_spinner="Loading dataset…")
def load_lorenz():
    from surge.benchmarks.leaderboard import _load_lorenz63
    return _load_lorenz63()


@st.cache_data(show_spinner="Loading dataset…")
def load_cmod():
    from surge.benchmarks.leaderboard import _load_cmod_density_limit
    return _load_cmod_density_limit()


@st.cache_data(show_spinner="Loading dataset…")
def load_plasma_stability():
    from surge.benchmarks.leaderboard import _load_plasma_stability
    return _load_plasma_stability()


@st.cache_data(show_spinner="Loading dataset…")
def load_nstxu():
    import pandas as pd
    df = pd.read_pickle(_REPO / "data/datasets/NSTX-U/nstxu_run10k_equil_curated.pkl")
    feature_cols = [c for c in df.columns if c not in ("case_idx", "Converged", "Diverted",
                    "plasma_configuration", "GS_error", "gamma_VDE", "gamma_TOKAM")]
    X = df[feature_cols].values.astype(float)
    y = df[["gamma_VDE", "gamma_TOKAM"]].values.astype(float)
    return X, y, feature_cols, df


@st.cache_data(show_spinner="Loading dataset…")
def load_smart():
    import pandas as pd
    df = pd.read_pickle(_REPO / "data/datasets/SMART/smart_curated_shapes_gamma.pkl")
    target_cols = ["gamma", "gamma_TOKAM"]
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols].values.astype(float)
    y = df[target_cols].values.astype(float)
    return X, y, feature_cols, df


@st.cache_data(show_spinner="Loading dataset…")
def load_concrete():
    from surge.benchmarks.leaderboard import _load_concrete_strength
    return _load_concrete_strength()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    if status == "cached":
        return "🟢 cached"
    if status == "local":
        return "🔵 local"
    return "🟡 on-demand"


def _show_info(meta: dict):
    cols = st.columns(4)
    cols[0].metric("Samples (n)", meta["n"])
    cols[1].metric("Shape", meta["shape"])
    cols[2].metric("Target", meta["target"])
    cols[3].metric("Benchmark", meta["benchmark"])
    st.caption(f"Source: {meta['source']}")


def interactive_scatter_explorer(df, *, color_default: str | None = None, key: str = ""):
    """Dropdown-driven Plotly scatter: pick X, Y, and color axes live."""
    import plotly.express as px

    st.subheader("🔎 Interactive scatter explorer")
    st.caption("Choose any two columns to plot against each other. Hover for values, "
               "drag to zoom, double-click to reset.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for a scatter plot.")
        return

    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X axis", numeric_cols, index=0, key=f"{key}_x")
    y = c2.selectbox("Y axis", numeric_cols,
                     index=min(1, len(numeric_cols) - 1), key=f"{key}_y")
    color_opts = ["(none)"] + list(df.columns)
    default_idx = color_opts.index(color_default) if color_default in color_opts else 0
    color = c3.selectbox("Color by", color_opts, index=default_idx, key=f"{key}_color")

    sample = df.sample(min(len(df), 5000), random_state=0)
    fig = px.scatter(
        sample, x=x, y=y,
        color=None if color == "(none)" else color,
        opacity=0.65, render_mode="webgl",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig, use_container_width=True)


def plotly_correlation(df, title: str):
    import plotly.express as px

    corr = df.corr(numeric_only=True)
    n = corr.shape[0]
    fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=".2f" if n <= 14 else False, aspect="auto",
    )
    fig.update_layout(height=max(450, n * 34), title=title,
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plotly_feature_facets(df, feature_cols, *, color_col=None, title=""):
    import plotly.express as px

    melt = df.melt(
        id_vars=[color_col] if color_col else [],
        value_vars=feature_cols, var_name="feature", value_name="value",
    )
    fig = px.histogram(
        melt, x="value", color=color_col,
        facet_col="feature", facet_col_wrap=4,
        barmode="overlay", opacity=0.7, nbins=40,
    )
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font_size=10))
    n_rows = (len(feature_cols) + 3) // 4
    fig.update_layout(height=max(300, n_rows * 200), title=title,
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ── Visualization functions ───────────────────────────────────────────────────

def viz_vision(name: str, load_fn, cifar10: bool = False):
    import matplotlib.pyplot as plt
    import plotly.express as px

    _show_info(DATASETS[name])
    st.divider()

    X, y = load_fn()

    classes_cifar = ["plane", "car", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]
    class_names = classes_cifar if cifar10 else [str(i) for i in range(10)]
    counts = np.bincount(y)
    rng = np.random.default_rng(0)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Class distribution")
        fig = px.bar(x=class_names, y=counts,
                     labels={"x": "Class", "y": "Count"},
                     color=counts, color_continuous_scale="Blues")
        fig.update_layout(height=360, showlegend=False,
                          coloraxis_showscale=False,
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sample images — 5 per class")
        fig2, axes = plt.subplots(10, 5, figsize=(8, 16))
        for cls in range(10):
            idx = np.where(y == cls)[0]
            chosen = rng.choice(idx, size=5, replace=False)
            for col_i, img_idx in enumerate(chosen):
                ax = axes[cls, col_i]
                img = X[img_idx]
                if cifar10:
                    img = img.reshape(3, 32, 32).transpose(1, 2, 0)
                    ax.imshow(img.clip(0, 1))
                else:
                    ax.imshow(img.reshape(28, 28), cmap="gray")
                ax.axis("off")
                if col_i == 0:
                    ax.set_ylabel(class_names[cls], fontsize=7, rotation=0,
                                  labelpad=30, va="center")
        fig2.suptitle(f"{'CIFAR-10' if cifar10 else 'MNIST'} — sample images", fontsize=11)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("Pixel intensity distribution (sample of 5k images)")
    sample = X[rng.choice(len(X), size=5000, replace=False)].ravel()
    fig3 = px.histogram(x=sample, nbins=80, labels={"x": "Pixel value"})
    fig3.update_layout(height=300, bargap=0.02, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig3, use_container_width=True)


def viz_pde_burgers():
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    _show_info(DATASETS["Burgers 1-D (PDE)"])
    st.divider()

    X, y = load_burgers()              # X: (N, 64) initial, y: (N, 64) solution
    x_grid = np.linspace(0, 1, X.shape[1])
    rng = np.random.default_rng(0)

    n_samples = st.slider("Number of sample trajectories to overlay", 2, 20, 6)
    chosen = rng.choice(len(X), size=n_samples, replace=False)

    st.subheader("Sample initial conditions → solutions")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Initial conditions u₀(x)", "Solutions u_T(x)"))
    for idx in chosen:
        fig.add_trace(go.Scatter(x=x_grid, y=X[idx], mode="lines",
                                 name=f"#{idx}", legendgroup=f"{idx}",
                                 showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_grid, y=y[idx], mode="lines",
                                 name=f"#{idx}", legendgroup=f"{idx}",
                                 showlegend=False), row=1, col=2)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10),
                      legend_title="Sample")
    fig.update_xaxes(title_text="x")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All solution fields (heatmap — first 200 samples)")
    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.imshow(X[:200].T, aspect="auto", origin="lower",
                         color_continuous_scale="Blues",
                         labels={"x": "Sample index", "y": "x grid", "color": "u₀"})
        fig2.update_layout(height=380, title="Initial conditions u₀",
                           margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.imshow(y[:200].T, aspect="auto", origin="lower",
                         color_continuous_scale="Reds",
                         labels={"x": "Sample index", "y": "x grid", "color": "u_T"})
        fig3.update_layout(height=380, title="Solutions u_T",
                           margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig3, use_container_width=True)


def viz_lorenz():
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    _show_info(DATASETS["Lorenz-63 (Sequence)"])
    st.divider()

    X, y = load_lorenz()                       # (N, 60) each = 3 vars × 20 steps
    X3 = X.reshape(len(X), 3, 20)
    y3 = y.reshape(len(y), 3, 20)
    rng = np.random.default_rng(0)

    st.subheader("3-D Lorenz attractor (drag to rotate)")
    n_traj = st.slider("Number of trajectories", 1, 30, 8)
    chosen = rng.choice(len(X), size=n_traj, replace=False)
    fig3d = go.Figure()
    for idx in chosen:
        traj = np.concatenate([X3[idx], y3[idx]], axis=1)   # (3, 40)
        fig3d.add_trace(go.Scatter3d(
            x=traj[0], y=traj[1], z=traj[2], mode="lines",
            line=dict(width=3), name=f"#{idx}", showlegend=False,
        ))
    fig3d.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0),
                        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig3d, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Context window time series (input)")
        var_names = ["x", "y", "z"]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=var_names)
        for vi in range(3):
            for idx in chosen:
                fig.add_trace(go.Scatter(y=X3[idx, vi], mode="lines",
                                         showlegend=False, opacity=0.6),
                              row=vi + 1, col=1)
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10))
        fig.update_xaxes(title_text="Time step", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Phase portraits (all trajectories)")
        flat = X3.reshape(len(X), 3, -1)
        pts = flat.transpose(0, 2, 1).reshape(-1, 3)[::5]
        import pandas as pd
        dfp = pd.DataFrame(pts, columns=["x", "y", "z"])
        pair = st.selectbox("Projection", ["x vs y", "y vs z", "x vs z"])
        a, b = {"x vs y": ("x", "y"), "y vs z": ("y", "z"), "x vs z": ("x", "z")}[pair]
        figp = px.scatter(dfp, x=a, y=b, opacity=0.25, render_mode="webgl")
        figp.update_traces(marker=dict(size=3, color="steelblue"))
        figp.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(figp, use_container_width=True)


def viz_tabular(name: str, X: np.ndarray, y: np.ndarray,
                feature_names: list[str] | None, target_label: str):
    import plotly.express as px
    import pandas as pd

    _show_info(DATASETS[name])
    st.divider()

    feat_names = feature_names or [f"feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_names)
    df[target_label] = y[:, 0] if y.ndim == 2 else y

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Target distribution")
        fig = px.histogram(df, x=target_label, nbins=40)
        fig.update_layout(height=340, bargap=0.03, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Summary statistics")
        st.dataframe(df.describe().round(3), use_container_width=True, height=340)

    st.subheader("Feature distributions")
    plotly_feature_facets(df, feat_names, title=f"{name} — feature distributions")

    st.subheader("Correlation matrix (features + target)")
    plotly_correlation(df, "Pearson correlation")

    st.divider()
    interactive_scatter_explorer(df, color_default=target_label, key=name)


def viz_binary_classification(name: str, X: np.ndarray, y: np.ndarray,
                               feature_names: list[str]):
    import plotly.express as px
    import pandas as pd

    _show_info(DATASETS[name])
    st.divider()

    labels = (["Stable (0)", "Unstable (1)"] if "Stability" in name
              else ["Normal (0)", "Disruption (1)"])
    df = pd.DataFrame(X, columns=feature_names)
    df["class"] = [labels[int(v)] if int(v) < len(labels) else str(int(v)) for v in y]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Class balance")
        counts = df["class"].value_counts().reset_index()
        counts.columns = ["class", "count"]
        fig = px.bar(counts, x="class", y="count", color="class",
                     color_discrete_sequence=["steelblue", "tomato"])
        fig.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Summary statistics")
        st.dataframe(df[feature_names].describe().round(3),
                     use_container_width=True, height=340)

    st.subheader("Feature distributions by class")
    plotly_feature_facets(df, feature_names, color_col="class",
                          title=f"{name} — features by class")

    st.divider()
    interactive_scatter_explorer(df, color_default="class", key=name)


def viz_tokamak(name: str, X, y, feature_names, df_raw):
    import plotly.express as px
    import pandas as pd

    _show_info(DATASETS[name])
    st.divider()

    target_cols = (["gamma_VDE", "gamma_TOKAM"] if name == "NSTX-U Equilibria"
                   else ["gamma", "gamma_TOKAM"])

    st.subheader("Growth rate distributions")
    gcols = st.columns(len(target_cols))
    for gc, tc in zip(gcols, target_cols):
        with gc:
            fig = px.histogram(df_raw, x=tc, nbins=50,
                               color_discrete_sequence=["mediumpurple"])
            fig.update_layout(height=320, bargap=0.03, title=tc,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    physics = [c for c in ["Ip", "Bt", "R0", "a", "kappa", "delta", "q0", "q95",
                           "beta_pol", "beta_tor", "p0", "W", "V", "li"]
               if c in df_raw.columns]

    st.subheader("Physics parameter distributions")
    plotly_feature_facets(df_raw, physics, title=f"{name} — physics parameters")

    st.subheader("Correlation: parameters → growth rates")
    corr_cols = [c for c in df_raw.columns
                 if c in list(feature_names) + target_cols
                 and df_raw[c].dtype.kind in "fi"]
    plotly_correlation(df_raw[corr_cols], f"{name} — correlation matrix")

    st.divider()
    explorer_cols = physics + target_cols
    interactive_scatter_explorer(df_raw[explorer_cols].copy(),
                                 color_default=target_cols[0], key=name)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("⚡ SURGE Datasets")
st.sidebar.caption("Interactive exploration of benchmark datasets")
st.sidebar.divider()

selected = st.sidebar.selectbox(
    "Select dataset",
    list(DATASETS.keys()),
    format_func=lambda n: f"{DATASETS[n]['category']} — {n}",
)

st.sidebar.divider()
st.sidebar.markdown("**Status legend**")
st.sidebar.markdown("🟢 cached — loaded from `data/datasets/benchmarks/`")
st.sidebar.markdown("🔵 local — loaded from `data/datasets/` (proprietary)")
st.sidebar.markdown("🟡 on-demand — downloads on first use")

meta = DATASETS[selected]
st.sidebar.divider()
st.sidebar.markdown(f"**Status:** {_status_badge(meta['status'])}")
st.sidebar.markdown(f"**Benchmark key:** `{meta['benchmark']}`")
st.sidebar.caption("All charts: hover for values · drag to zoom · double-click to reset")

# ── Main panel ────────────────────────────────────────────────────────────────

st.title(f"⚡ {selected}")

if selected == "MNIST":
    viz_vision("MNIST", load_mnist, cifar10=False)

elif selected == "CIFAR-10":
    viz_vision("CIFAR-10", load_cifar10, cifar10=True)

elif selected == "Burgers 1-D (PDE)":
    viz_pde_burgers()

elif selected == "Lorenz-63 (Sequence)":
    viz_lorenz()

elif selected == "C-Mod Density Limit":
    X, y = load_cmod()
    feature_names = ["density", "elongation", "minor_radius",
                     "plasma_current", "toroidal_B_field", "triangularity"]
    viz_binary_classification("C-Mod Density Limit", X, y, feature_names)

elif selected == "Plasma Stability (UCI)":
    X, y = load_plasma_stability()
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    viz_binary_classification("Plasma Stability (UCI)", X, y, feature_names)

elif selected == "NSTX-U Equilibria":
    X, y, feature_names, df_raw = load_nstxu()
    viz_tokamak("NSTX-U Equilibria", X, y, feature_names, df_raw)

elif selected == "SMART Equilibria":
    X, y, feature_names, df_raw = load_smart()
    viz_tokamak("SMART Equilibria", X, y, feature_names, df_raw)

elif selected == "Concrete Strength (Tabular)":
    X, y = load_concrete()
    feature_names = [
        "cement", "blast_furnace_slag", "fly_ash", "water",
        "superplasticizer", "coarse_aggregate", "fine_aggregate", "age",
    ]
    viz_tabular("Concrete Strength (Tabular)", X, y, feature_names,
                "Compressive strength (MPa)")

st.divider()
st.caption("SURGE — Surrogate Unified Research & Generalization Engine · "
           "data/datasets/benchmarks/ · Plotly interactive · 2026")
