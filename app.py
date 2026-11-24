from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Konfigurasi aplikasi dan fitur model
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Rekomendasi Laptop | CBF",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DEMO_DATA_PATH = BASE_DIR / "data" / "laptop_demo_sample.csv"

CATEGORICAL_FEATURES = ["Usage_Type"]
NUMERIC_FEATURES = [
    "RAM_GB",
    "Storage_GB",
    "Cores",
    "Threads",
    "Base_Clock",
    "Boost_Clock",
    "TDP",
    "CPU_Performance",
    "GPU_Performance",
    "Price_USD",
]
REQUIRED_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

DTYPES = {
    "Usage_Type": "string",
    "RAM_GB": "int16",
    "Storage_GB": "int16",
    "Cores": "int16",
    "Threads": "int16",
    "Base_Clock": "float32",
    "Boost_Clock": "float32",
    "TDP": "int16",
    "CPU_Performance": "int32",
    "GPU_Performance": "int32",
    "Price_USD": "float32",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1450px;}
      [data-testid="stMetric"] {background: rgba(128,128,128,.07); padding: 14px 16px; border-radius: 12px;}
      .hero {padding: 1.15rem 1.35rem; border-radius: 16px; background: linear-gradient(120deg, rgba(79,70,229,.13), rgba(14,165,233,.10)); border: 1px solid rgba(128,128,128,.20);}
      .muted {color: rgba(128,128,128,.95); font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_demo_data(path_string: str) -> pd.DataFrame:
    """Muat sampel demo yang disertakan bersama aplikasi."""
    return pd.read_csv(path_string, usecols=REQUIRED_COLUMNS, dtype=DTYPES)


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes) -> pd.DataFrame:
    """Muat hanya kolom yang digunakan model dari CSV yang diunggah pengguna."""
    return pd.read_csv(io.BytesIO(file_bytes), usecols=REQUIRED_COLUMNS, dtype=DTYPES)


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Pembersihan minimum agar fitur model numerik dan dapat diproses."""
    data = data.copy()
    data["Usage_Type"] = data["Usage_Type"].astype("string").str.strip()
    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data[np.isfinite(data[NUMERIC_FEATURES]).all(axis=1)]
    data = data[data["Price_USD"] > 0]
    return data.reset_index(drop=True)


def fit_preprocessor(data: pd.DataFrame) -> tuple[OneHotEncoder, StandardScaler]:
    """Fit One-Hot Encoding dan StandardScaler secara bertahap untuk hemat memori."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:  # kompatibilitas scikit-learn versi lama
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
    encoder.fit(data[CATEGORICAL_FEATURES].astype(str))

    scaler = StandardScaler()
    chunk_size = 100_000
    for start in range(0, len(data), chunk_size):
        chunk = data.iloc[start : start + chunk_size][NUMERIC_FEATURES]
        scaler.partial_fit(chunk.to_numpy(dtype=np.float64, copy=True))
    return encoder, scaler


def build_recommendations(
    data: pd.DataFrame,
    preferences: dict[str, Any],
    top_n: int,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
) -> tuple[pd.DataFrame, int]:
    """Filter kandidat, hitung Cosine Similarity, lalu ambil rekomendasi teratas."""
    candidate_mask = (
        (data["Usage_Type"] == preferences["Usage_Type"])
        & (data["Price_USD"] <= preferences["budget"])
        & (data["RAM_GB"] >= preferences["min_ram"])
        & (data["Storage_GB"] >= preferences["min_storage"])
    )
    candidates = data.loc[candidate_mask].copy()
    candidate_count = len(candidates)
    if candidates.empty:
        return pd.DataFrame(), candidate_count

    profile = {
        "Usage_Type": preferences["Usage_Type"],
        "RAM_GB": preferences["target_ram"],
        "Storage_GB": preferences["target_storage"],
        "Cores": preferences["target_cores"],
        "Threads": preferences["target_threads"],
        "Base_Clock": preferences["target_base_clock"],
        "Boost_Clock": preferences["target_boost_clock"],
        "TDP": preferences["target_tdp"],
        "CPU_Performance": preferences["target_cpu_perf"],
        "GPU_Performance": preferences["target_gpu_perf"],
        "Price_USD": preferences["target_price"],
    }
    profile_df = pd.DataFrame([profile])

    profile_cat = encoder.transform(profile_df[CATEGORICAL_FEATURES].astype(str))
    profile_num = scaler.transform(profile_df[NUMERIC_FEATURES].to_numpy(dtype=np.float64))
    profile_vector = np.concatenate([profile_cat, profile_num], axis=1).astype(np.float32).ravel()
    profile_norm = float(np.linalg.norm(profile_vector))

    # Penilaian per-batch mencegah pembuatan matriks besar sekaligus.
    score_parts: list[np.ndarray] = []
    chunk_size = 50_000
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates.iloc[start : start + chunk_size]
        cat = encoder.transform(chunk[CATEGORICAL_FEATURES].astype(str))
        num = scaler.transform(chunk[NUMERIC_FEATURES].to_numpy(dtype=np.float64))
        matrix = np.concatenate([cat, num], axis=1).astype(np.float32, copy=False)
        dot = matrix @ profile_vector
        denominator = np.linalg.norm(matrix, axis=1) * profile_norm
        scores = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator != 0)
        score_parts.append(scores.astype(np.float32))

    candidates["Similarity"] = np.concatenate(score_parts)
    display_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    result = (
        candidates.sort_values("Similarity", ascending=False)
        .drop_duplicates(subset=display_features)
        .head(top_n)
        .copy()
    )
    result.insert(0, "Peringkat", np.arange(1, len(result) + 1))
    return result, candidate_count


def fmt_usd(value: float) -> str:
    return f"${value:,.2f}"


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div style="font-size: .88rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; opacity: .8;">Sistem Rekomendasi • Content-Based Filtering</div>
      <h1 style="margin: .25rem 0 .35rem 0;">Temukan laptop sesuai kebutuhanmu</h1>
      <div class="muted">Atur jenis penggunaan, anggaran, dan spesifikasi. Sistem akan menyaring kandidat lalu mengurutkannya berdasarkan Cosine Similarity.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.header("📂 Dataset")
    uploaded_file = st.file_uploader(
        "Unggah dataset CSV lengkap (opsional)",
        type=["csv"],
        help="Jika tidak mengunggah file, aplikasi menggunakan sampel demo 50.000 baris yang disertakan.",
    )

try:
    with st.spinner("Memuat dataset..."):
        if uploaded_file is not None:
            df_raw = load_uploaded_data(uploaded_file.getvalue())
            data_source = f"CSV unggahan: {uploaded_file.name}"
            raw_signature = (data_source, getattr(uploaded_file, "file_id", ""), getattr(uploaded_file, "size", 0), len(df_raw))
        else:
            if not DEMO_DATA_PATH.exists():
                st.error("File sampel demo tidak ditemukan. Pastikan folder data/laptop_demo_sample.csv ikut di-deploy.")
                st.stop()
            df_raw = load_demo_data(str(DEMO_DATA_PATH))
            data_source = "Sampel demo 50.000 baris"
            raw_signature = (data_source, len(df_raw))
        if st.session_state.get("clean_data_signature") == raw_signature:
            df = st.session_state["clean_data_frame"]
        else:
            df = clean_dataset(df_raw)
            st.session_state["clean_data_signature"] = raw_signature
            st.session_state["clean_data_frame"] = df
except ValueError as exc:
    st.error(
        "Struktur CSV tidak sesuai. Pastikan file berasal dari dataset laptop dan memiliki kolom: "
        + ", ".join(REQUIRED_COLUMNS)
    )
    st.exception(exc)
    st.stop()
except Exception as exc:
    st.error("Dataset gagal dibaca. Periksa format CSV dan coba unggah kembali.")
    st.exception(exc)
    st.stop()

if df.empty:
    st.error("Dataset tidak memiliki baris valid setelah pembersihan.")
    st.stop()

usage_options = sorted(df["Usage_Type"].dropna().astype(str).unique().tolist())

with st.sidebar:
    st.divider()
    st.header("🎯 Preferensi laptop")
    selected_usage = st.selectbox(
        "Jenis penggunaan",
        options=usage_options,
        index=usage_options.index("Student") if "Student" in usage_options else 0,
    )
    min_price = max(0, int(np.floor(float(df["Price_USD"].min()) / 50) * 50))
    max_price = int(np.ceil(float(df["Price_USD"].quantile(0.99)) / 100) * 100)
    default_budget = min(2500, max_price)
    if default_budget < min_price:
        default_budget = min_price
    budget = st.slider(
        "Anggaran maksimal (USD)",
        min_value=min_price,
        max_value=max_price,
        value=default_budget,
        step=50,
        format="$%d",
    )
    ram_options = sorted(int(x) for x in df["RAM_GB"].dropna().unique())
    storage_options = sorted(int(x) for x in df["Storage_GB"].dropna().unique())
    ram_default = 16 if 16 in ram_options else ram_options[0]
    storage_default = 512 if 512 in storage_options else storage_options[0]
    min_ram = st.selectbox("RAM minimum (GB)", ram_options, index=ram_options.index(ram_default))
    min_storage = st.selectbox(
        "Penyimpanan minimum (GB)",
        storage_options,
        index=storage_options.index(storage_default),
    )

    with st.expander("Spesifikasi target untuk kemiripan", expanded=True):
        target_price = st.slider(
            "Harga yang diinginkan (USD)",
            min_value=min_price,
            max_value=max(min_price, budget),
            value=min(max(2200, min_price), budget),
            step=50,
            format="$%d",
        )
        target_ram = st.select_slider("Target RAM (GB)", options=ram_options, value=min_ram)
        target_storage = st.select_slider(
            "Target penyimpanan (GB)", options=storage_options, value=min_storage
        )
        target_cores = st.slider(
            "Target jumlah core", int(df["Cores"].min()), int(df["Cores"].max()),
            min(6, int(df["Cores"].max())),
        )
        target_threads = st.slider(
            "Target jumlah thread", int(df["Threads"].min()), int(df["Threads"].max()),
            min(12, int(df["Threads"].max())),
        )
        target_base_clock = st.slider(
            "Target base clock (GHz)", 1.0, 4.0, 2.5, 0.1
        )
        target_boost_clock = st.slider(
            "Target boost clock (GHz)", 2.0, 5.5, 4.0, 0.1
        )
        target_tdp = st.slider("Target TDP (W)", 15, 125, 45, 5)
        target_cpu_perf = st.slider("Target CPU performance", 400, 8800, 2500, 100)
        target_gpu_perf = st.slider("Target GPU performance", 900, 3800, 1500, 100)

    top_n = st.selectbox("Jumlah rekomendasi", [5, 10, 15, 20], index=1)

preferences = {
    "Usage_Type": selected_usage,
    "budget": budget,
    "min_ram": min_ram,
    "min_storage": min_storage,
    "target_price": target_price,
    "target_ram": target_ram,
    "target_storage": target_storage,
    "target_cores": target_cores,
    "target_threads": target_threads,
    "target_base_clock": target_base_clock,
    "target_boost_clock": target_boost_clock,
    "target_tdp": target_tdp,
    "target_cpu_perf": target_cpu_perf,
    "target_gpu_perf": target_gpu_perf,
}

# -----------------------------------------------------------------------------
# Ringkasan dataset dan hasil rekomendasi
# -----------------------------------------------------------------------------
st.caption(f"Sumber data: **{data_source}** · {len(df):,} baris valid · {len(REQUIRED_COLUMNS)} kolom yang digunakan model")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Data laptop", f"{len(df):,}")
col_b.metric("Jenis penggunaan", f"{df['Usage_Type'].nunique():,}")
col_c.metric("Rentang harga", f"{fmt_usd(float(df['Price_USD'].min()))} – {fmt_usd(float(df['Price_USD'].max()))}")
col_d.metric("Anggaran maksimal", fmt_usd(float(budget)))

preprocessor_signature = (raw_signature, len(df))
if st.session_state.get("preprocessor_signature") != preprocessor_signature:
    with st.spinner("Menyiapkan One-Hot Encoding dan standardisasi fitur..."):
        fitted_encoder, fitted_scaler = fit_preprocessor(df)
        st.session_state["preprocessor_signature"] = preprocessor_signature
        st.session_state["fitted_encoder"] = fitted_encoder
        st.session_state["fitted_scaler"] = fitted_scaler
else:
    fitted_encoder = st.session_state["fitted_encoder"]
    fitted_scaler = st.session_state["fitted_scaler"]

with st.spinner("Menghitung kemiripan spesifikasi dan menyusun rekomendasi..."):
    recommendations, candidate_count = build_recommendations(
        df, preferences, top_n, fitted_encoder, fitted_scaler
    )

st.subheader("Hasil rekomendasi")
if recommendations.empty:
    st.warning(
        "Tidak ada kandidat yang memenuhi semua filter. Coba naikkan anggaran atau turunkan batas minimum RAM/penyimpanan."
    )
else:
    best = recommendations.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Kandidat setelah filter", f"{candidate_count:,}")
    m2.metric("Skor kemiripan teratas", f"{float(best['Similarity']):.4f}")
    m3.metric("Harga rekomendasi teratas", fmt_usd(float(best["Price_USD"])))

    st.success(
        f"Rekomendasi dengan skor tertinggi memiliki RAM {int(best['RAM_GB'])} GB, "
        f"storage {int(best['Storage_GB'])} GB, {int(best['Cores'])} core / {int(best['Threads'])} thread, "
        f"dan harga {fmt_usd(float(best['Price_USD']))}."
    )

    display = recommendations[
        [
            "Peringkat", "RAM_GB", "Storage_GB", "Cores", "Threads",
            "CPU_Performance", "GPU_Performance", "Price_USD", "Similarity",
        ]
    ].rename(
        columns={
            "RAM_GB": "RAM (GB)",
            "Storage_GB": "Storage (GB)",
            "Cores": "Core",
            "Threads": "Thread",
            "CPU_Performance": "CPU Perf.",
            "GPU_Performance": "GPU Perf.",
            "Price_USD": "Harga (USD)",
            "Similarity": "Cosine Similarity",
        }
    )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Harga (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "Cosine Similarity": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    csv_output = display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Unduh hasil rekomendasi (CSV)",
        data=csv_output,
        file_name="hasil_rekomendasi_laptop.csv",
        mime="text/csv",
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        bar_df = display.copy()
        bar_df["Peringkat"] = bar_df["Peringkat"].astype(str)
        fig_bar = px.bar(
            bar_df,
            x="Cosine Similarity",
            y="Peringkat",
            orientation="h",
            text="Cosine Similarity",
            title="Skor kemiripan rekomendasi",
            labels={"Peringkat": "Urutan rekomendasi"},
        )
        fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_bar.update_layout(height=430, yaxis={"categoryorder": "array", "categoryarray": bar_df["Peringkat"].tolist()[::-1]})
        st.plotly_chart(fig_bar, use_container_width=True)
    with chart_col2:
        scatter_df = recommendations.copy()
        fig_scatter = px.scatter(
            scatter_df,
            x="Price_USD",
            y="Similarity",
            size="RAM_GB",
            color="Cores",
            hover_data=["RAM_GB", "Storage_GB", "Threads", "GPU_Performance"],
            title="Harga dan skor kemiripan",
            labels={
                "Price_USD": "Harga (USD)",
                "Similarity": "Cosine Similarity",
                "Cores": "Jumlah core",
                "RAM_GB": "RAM (GB)",
            },
        )
        fig_scatter.update_layout(height=430)
        st.plotly_chart(fig_scatter, use_container_width=True)

with st.expander("Lihat distribusi jenis penggunaan pada dataset"):
    usage_df = df["Usage_Type"].value_counts().rename_axis("Jenis penggunaan").reset_index(name="Jumlah")
    fig_usage = px.bar(
        usage_df,
        x="Jenis penggunaan",
        y="Jumlah",
        text="Jumlah",
        title="Distribusi data berdasarkan jenis penggunaan",
    )
    fig_usage.update_layout(height=350)
    st.plotly_chart(fig_usage, use_container_width=True)

st.divider()
st.subheader("Cara kerja sistem")
step1, step2, step3, step4 = st.columns(4)
step1.markdown("**1. Input preferensi**\n\nJenis penggunaan, anggaran, RAM, dan storage.")
step2.markdown("**2. Filter kandidat**\n\nKandidat harus sesuai jenis penggunaan dan batas spesifikasi/anggaran.")
step3.markdown("**3. Encoding fitur**\n\nOne-Hot Encoding untuk Usage Type dan StandardScaler untuk fitur numerik.")
step4.markdown("**4. Pemeringkatan**\n\nCosine Similarity mengurutkan kandidat yang paling dekat dengan profil.")

st.info(
    "**Catatan interpretasi:** skor Cosine Similarity menunjukkan kedekatan vektor spesifikasi, bukan probabilitas, "
    "akurasi, atau jaminan kualitas produk. Dataset asal memiliki pasangan merek/model laptop dan GPU yang tidak konsisten, "
    "sehingga aplikasi ini tidak menampilkan nama merek/model dan hasilnya diperlakukan sebagai demonstrasi rekomendasi spesifikasi. "
    "Pada mode demo, hasil dihitung dari sampel 50.000 baris; unggah CSV lengkap untuk analisis seluruh dataset."
)

st.caption("Proyek UTS Sistem Rekomendasi · Content-Based Filtering · One-Hot Encoding · Cosine Similarity")
