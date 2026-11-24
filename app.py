from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from recommender import load_artifacts, recommend

st.set_page_config(
    page_title="Rekomendasi Laptop Indonesia",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "laptop_indonesia_featured.csv"
MATRIX_PATH = DATA_DIR / "laptop_features_encoded.npz"
META_PATH = DATA_DIR / "feature_metadata.json"

st.markdown(
    """
    <style>
      .block-container {max-width: 1450px; padding-top: 1.6rem; padding-bottom: 2.5rem;}
      .hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(120,120,120,.22);
             border-radius: 16px; background: linear-gradient(115deg,
             rgba(59,130,246,.12), rgba(16,185,129,.09));}
      .subtle {color: rgba(128,128,128,.95);}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Memuat dataset dan model encoding...")
def cached_load(csv_path: str, matrix_path: str, meta_path: str):
    return load_artifacts(csv_path, matrix_path, meta_path)


st.markdown(
    """
    <div class="hero">
      <div style="font-size:.85rem; letter-spacing:.07em; text-transform:uppercase; font-weight:650; opacity:.8;">
        Sistem Rekomendasi • Content-Based Filtering
      </div>
      <h1 style="margin:.3rem 0;">Temukan laptop sesuai kebutuhanmu</h1>
      <div class="subtle">Pilih anggaran dan spesifikasi minimum. Sistem menyaring katalog, kemudian
      mengurutkan produk berdasarkan Cosine Similarity.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not all(path.exists() for path in [CSV_PATH, MATRIX_PATH, META_PATH]):
    st.error("File model tidak lengkap. Pastikan folder data/ berisi CSV, NPZ, dan JSON.")
    st.stop()

try:
    df, matrix, metadata, scaler, feature_to_index = cached_load(
        str(CSV_PATH), str(MATRIX_PATH), str(META_PATH)
    )
except Exception as exc:
    st.error("Dataset atau matriks encoding gagal divalidasi.")
    st.exception(exc)
    st.stop()

checked_date = "Tidak diketahui"
if "Checked_Date" in df.columns and df["Checked_Date"].notna().any():
    checked_date = str(df["Checked_Date"].dropna().astype(str).max())

with st.sidebar:
    st.header("🎯 Preferensi laptop")
    st.caption("Harga di katalog merupakan snapshot, bukan harga real-time.")

    min_price = int(df["Price_IDR"].min())
    max_price = int(df["Price_IDR"].max())
    target_budget = min(max_price, 15_000_000)
    budget_default = min(
        max_price,
        min_price + round((target_budget - min_price) / 500_000) * 500_000,
    )
    budget_max = st.slider(
        "Anggaran maksimal",
        min_value=min_price,
        max_value=max_price,
        value=budget_default,
        step=500_000,
        format="Rp%d",
    )

    ram_values = sorted(int(value) for value in df["RAM_GB"].dropna().unique())
    storage_values = sorted(int(value) for value in df["Storage_GB"].dropna().unique())
    screen_values = sorted(float(value) for value in df["Screen_Size"].dropna().unique())

    min_ram = st.selectbox(
        "RAM minimum", ram_values,
        index=ram_values.index(16) if 16 in ram_values else 0,
    )
    min_storage = st.selectbox(
        "Storage minimum (GB)", storage_values,
        index=storage_values.index(512) if 512 in storage_values else 0,
    )

    st.divider()
    st.subheader("Preferensi kemiripan")
    target_ram = st.selectbox(
        "Target RAM (GB)", ram_values,
        index=ram_values.index(min_ram) if min_ram in ram_values else 0,
    )
    target_storage = st.selectbox(
        "Target storage (GB)", storage_values,
        index=storage_values.index(min_storage) if min_storage in storage_values else 0,
    )
    target_screen = st.selectbox(
        "Ukuran layar target (inci)", screen_values,
        index=min(range(len(screen_values)), key=lambda i: abs(screen_values[i] - 14.0)),
        format_func=lambda value: f'{value:g}"',
    )

    categories = metadata["categorical_features"]
    brand_values = sorted(df["Brand"].dropna().astype(str).unique())
    processor_brand_values = sorted(df["Processor_Brand"].dropna().astype(str).unique())
    cpu_family_values = sorted(df["CPU_Family"].dropna().astype(str).unique())
    gpu_type_values = sorted(df["GPU_Type"].dropna().astype(str).unique())

    brand = st.selectbox("Merek pilihan (opsional)", ["Bebas"] + brand_values)
    processor_brand = st.selectbox(
        "Merek prosesor (opsional)", ["Bebas"] + processor_brand_values
    )
    cpu_family = st.selectbox("Keluarga prosesor (opsional)", ["Bebas"] + cpu_family_values)
    required_gpu_type = st.selectbox(
        "Jenis GPU wajib (opsional)", ["Bebas"] + gpu_type_values,
        help="Pilih Dedicated untuk membatasi hasil hanya ke laptop dengan GPU dedicated.",
    )

    top_n = st.slider("Jumlah rekomendasi", min_value=3, max_value=10, value=5)

profile = {
    "budget_max": budget_max,
    "min_ram_gb": min_ram,
    "min_storage_gb": min_storage,
    "target_ram_gb": target_ram,
    "target_storage_gb": target_storage,
    "target_screen_size": target_screen,
    "categories": {
        "Brand": None if brand == "Bebas" else brand,
        "Processor_Brand": None if processor_brand == "Bebas" else processor_brand,
        "CPU_Family": None if cpu_family == "Bebas" else cpu_family,
        "GPU_Type": None if required_gpu_type == "Bebas" else required_gpu_type,
    },
    "required_gpu_type": None if required_gpu_type == "Bebas" else required_gpu_type,
}

try:
    results, candidate_count, unique_configuration_count = recommend(
        df, matrix, metadata, scaler, feature_to_index, profile, top_n=top_n, deduplicate=True
    )
except Exception as exc:
    st.error("Terjadi kesalahan saat menghitung rekomendasi.")
    st.exception(exc)
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Produk dalam katalog", f"{len(df):,}")
m2.metric("Kandidat lolos filter", f"{candidate_count:,}")
m3.metric("Konfigurasi unik", f"{unique_configuration_count:,}")
m4.metric("Tanggal pemeriksaan harga", checked_date)

st.divider()
st.subheader("Hasil rekomendasi")
st.caption(
    "Produk dengan kombinasi merek dan spesifikasi yang sama digabungkan. "
    "Jika ada beberapa listing identik, sistem memilih listing termurah."
)

if results.empty:
    st.warning("Belum ada laptop yang memenuhi filter tersebut. Coba naikkan anggaran atau turunkan spesifikasi minimum.")
else:
    display_columns = [
        "Rank", "Product_Name", "Brand", "Processor", "RAM_GB", "Storage_GB",
        "GPU", "Screen_Size", "Price_IDR", "Similarity_Percent", "Product_URL"
    ]
    display_df = results[[column for column in display_columns if column in results.columns]].copy()
    display_df = display_df.rename(columns={
        "Rank": "Peringkat",
        "Product_Name": "Nama Laptop",
        "Brand": "Merek",
        "Processor": "Prosesor",
        "RAM_GB": "RAM (GB)",
        "Storage_GB": "Storage (GB)",
        "GPU": "GPU",
        "Screen_Size": "Layar (inci)",
        "Price_IDR": "Harga (Rp)",
        "Similarity_Percent": "Cosine Similarity (%)",
        "Product_URL": "Link Produk",
    })
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Harga (Rp)": st.column_config.NumberColumn(format="Rp %d"),
            "Cosine Similarity (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Layar (inci)": st.column_config.NumberColumn(format="%.1f"),
            "Link Produk": st.column_config.LinkColumn("Link Produk", display_text="Buka produk"),
        },
    )

    chart_data = results.sort_values("Similarity_Percent", ascending=True)
    fig = px.bar(
        chart_data,
        x="Similarity_Percent",
        y="Product_Name",
        orientation="h",
        text=chart_data["Similarity_Percent"].map(lambda value: f"{value:.2f}%"),
        labels={
            "Similarity_Percent": "Cosine Similarity (%)",
            "Product_Name": "Nama laptop",
        },
        title="Peringkat rekomendasi berdasarkan Cosine Similarity",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=max(350, 58 * len(chart_data)), margin=dict(l=10, r=55, t=60, b=35))
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "⬇️ Download hasil rekomendasi (CSV)",
        data=results.to_csv(index=False).encode("utf-8-sig"),
        file_name="hasil_rekomendasi_laptop.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.markdown("### Detail produk")
    for _, row in results.iterrows():
        with st.expander(f"{int(row['Rank'])}. {row['Product_Name']} — Rp{int(row['Price_IDR']):,}"):
            left, right = st.columns([2, 1])
            with left:
                st.write(f"**Prosesor:** {row['Processor']}")
                st.write(f"**RAM / Storage:** {int(row['RAM_GB'])} GB / {int(row['Storage_GB'])} GB")
                st.write(f"**GPU:** {row['GPU']}")
                st.write(f"**Layar:** {row['Screen_Size']:g} inci")
                st.write(f"**Penjual:** {row.get('Seller', 'Tidak tersedia')}")
            with right:
                st.metric("Cosine Similarity", f"{row['Similarity_Percent']:.2f}%")
                st.link_button("Lihat produk di toko", row["Product_URL"], use_container_width=True)

st.divider()
st.caption(
    "Catatan: skor Cosine Similarity menunjukkan kemiripan vektor fitur, bukan akurasi model "
    "atau probabilitas pembelian. Harga dan stok mengikuti snapshot katalog yang diperiksa pada "
    f"{checked_date}; verifikasi kembali di halaman toko sebelum membeli."
)
