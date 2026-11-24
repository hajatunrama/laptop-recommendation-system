from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

NUMERIC_REQUIRED = ["RAM_GB", "Storage_GB", "Screen_Size"]
IDENTITY_COLUMNS = ["Product_Name", "Product_URL", "Price_IDR"]
DEDUP_COLUMNS = [
    "Brand", "Processor", "RAM_GB", "Storage_GB",
    "Storage_Type", "GPU_Type", "GPU", "Screen_Size"
]


def load_csr_npz(path: str | Path) -> csr_matrix:
    """Load CSR components saved through numpy.savez_compressed."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"data", "indices", "indptr", "shape"}
        if not required.issubset(set(archive.files)):
            raise ValueError(
                "File NPZ tidak berisi komponen CSR yang diharapkan "
                "(data, indices, indptr, shape)."
            )
        shape = tuple(int(x) for x in archive["shape"])
        matrix = csr_matrix(
            (archive["data"], archive["indices"], archive["indptr"]),
            shape=shape,
        )
    return matrix


def load_artifacts(
    csv_path: str | Path,
    matrix_path: str | Path,
    metadata_path: str | Path,
) -> tuple[pd.DataFrame, csr_matrix, dict[str, Any], StandardScaler, dict[str, int]]:
    df = pd.read_csv(csv_path)
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    matrix = load_csr_npz(matrix_path)

    for column in IDENTITY_COLUMNS + NUMERIC_REQUIRED:
        if column not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan pada CSV: {column}")
    categorical = metadata.get("categorical_features", [])
    numeric = metadata.get("numeric_features", [])
    feature_names = metadata.get("feature_names", [])
    if numeric != NUMERIC_REQUIRED:
        raise ValueError(
            f"Fitur numerik metadata tidak sesuai. Ditemukan {numeric}, "
            f"diharapkan {NUMERIC_REQUIRED}."
        )
    missing_cat = [column for column in categorical if column not in df.columns]
    if missing_cat:
        raise ValueError(f"Kolom kategori tidak ditemukan: {missing_cat}")
    if matrix.shape != (len(df), len(feature_names)):
        raise ValueError(
            f"Dimensi matriks {matrix.shape} tidak cocok dengan CSV/metadata "
            f"({len(df)}, {len(feature_names)})."
        )

    feature_to_index = {name: i for i, name in enumerate(feature_names)}
    if len(feature_to_index) != len(feature_names):
        raise ValueError("Nama fitur duplikat pada metadata.")

    # Pastikan urutan baris CSV dan matriks encoding masih sejajar.
    for row_index, row in df.iterrows():
        for column in categorical:
            feature_name = f"categorical__{column}_{row[column]}"
            if feature_name not in feature_to_index:
                raise ValueError(
                    f"Kategori {column}={row[column]!r} pada baris {row_index} "
                    "tidak ditemukan di metadata."
                )
            if matrix[row_index, feature_to_index[feature_name]] != 1:
                raise ValueError(
                    f"Urutan kategori CSV dan matriks tidak cocok pada baris {row_index}."
                )

    # Scaler yang sama dengan proses encoding: fit pada seluruh data katalog.
    scaler = StandardScaler()
    numeric_values = df[numeric].astype(float)
    scaler.fit(numeric_values)
    transformed = scaler.transform(numeric_values)
    max_diffs: dict[str, float] = {}
    for j, column in enumerate(numeric):
        feature_name = f"numeric__{column}"
        if feature_name not in feature_to_index:
            raise ValueError(f"Fitur numerik {feature_name} tidak ada pada metadata.")
        encoded_values = matrix[:, feature_to_index[feature_name]].toarray().ravel()
        max_diffs[column] = float(np.max(np.abs(encoded_values - transformed[:, j])))
    if max(max_diffs.values(), default=0.0) > 1e-7:
        raise ValueError(
            "Nilai fitur numerik di matriks tidak cocok dengan StandardScaler "
            f"yang direkonstruksi. Selisih: {max_diffs}"
        )

    # Casting tipe data dan validasi dasar.
    df = df.copy()
    for column in ["RAM_GB", "Storage_GB", "Screen_Size", "Price_IDR"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=IDENTITY_COLUMNS + NUMERIC_REQUIRED).reset_index(drop=True)
    if len(df) != matrix.shape[0]:
        raise ValueError(
            "Ada baris tidak valid yang harus dihapus. Jangan menghapus baris "
            "setelah encoding; bersihkan CSV lalu buat ulang matriks encoding."
        )
    return df, matrix, metadata, scaler, feature_to_index


def build_profile_vector(
    profile: dict[str, Any],
    metadata: dict[str, Any],
    scaler: StandardScaler,
    feature_to_index: dict[str, int],
) -> np.ndarray:
    feature_names = metadata["feature_names"]
    query = np.zeros(len(feature_names), dtype=np.float64)

    categorical_preferences = profile.get("categories", {})
    allowed_categories = set(metadata["categorical_features"])
    unknown_columns = set(categorical_preferences) - allowed_categories
    if unknown_columns:
        raise ValueError(f"Fitur kategori tidak dikenal: {sorted(unknown_columns)}")

    for column, value in categorical_preferences.items():
        if value in (None, "", "Semua", "Bebas"):
            continue
        feature_name = f"categorical__{column}_{value}"
        if feature_name not in feature_to_index:
            raise ValueError(f"Pilihan {column}={value!r} tidak tersedia dalam model.")
        query[feature_to_index[feature_name]] = 1.0

    numeric_features = metadata["numeric_features"]
    numeric_profile = pd.DataFrame(
        [{
            "RAM_GB": profile["target_ram_gb"],
            "Storage_GB": profile["target_storage_gb"],
            "Screen_Size": profile["target_screen_size"],
        }],
        columns=numeric_features,
    )
    scaled = scaler.transform(numeric_profile.astype(float))[0]
    for position, column in enumerate(numeric_features):
        query[feature_to_index[f"numeric__{column}"]] = scaled[position]

    if np.linalg.norm(query) == 0:
        raise ValueError("Vektor profil kosong. Masukkan spesifikasi target.")
    return query


def recommend(
    df: pd.DataFrame,
    matrix: csr_matrix,
    metadata: dict[str, Any],
    scaler: StandardScaler,
    feature_to_index: dict[str, int],
    profile: dict[str, Any],
    top_n: int = 5,
    deduplicate: bool = True,
) -> tuple[pd.DataFrame, int, int]:
    """Filter kandidat, hitung cosine similarity, dan opsional deduplikasi konfigurasi."""
    mask = (
        (df["Price_IDR"] <= float(profile["budget_max"]))
        & (df["RAM_GB"] >= float(profile["min_ram_gb"]))
        & (df["Storage_GB"] >= float(profile["min_storage_gb"]))
    )
    required_gpu_type = profile.get("required_gpu_type")
    if required_gpu_type not in (None, "", "Semua", "Bebas"):
        mask &= df["GPU_Type"].astype(str).eq(str(required_gpu_type))

    candidate_indices = np.flatnonzero(mask.to_numpy())
    candidate_count = int(len(candidate_indices))
    if candidate_count == 0:
        return pd.DataFrame(), 0, 0

    query_vector = build_profile_vector(profile, metadata, scaler, feature_to_index)
    candidate_matrix = matrix[candidate_indices]
    scores = cosine_similarity(candidate_matrix, query_vector.reshape(1, -1)).ravel()

    results = df.iloc[candidate_indices].copy()
    results["Similarity_Score"] = scores
    results["Similarity_Percent"] = scores * 100.0
    results = results.sort_values(
        ["Similarity_Score", "Price_IDR", "Product_Name"],
        ascending=[False, True, True],
        kind="stable",
    )

    if deduplicate:
        usable_key = [column for column in DEDUP_COLUMNS if column in results.columns]
        # Normalisasi terbatas hanya untuk mendeteksi duplikasi spesifikasi,
        # tanpa mengubah data sumber atau kategori pada matriks model.
        def normalize_for_dedup(value: Any) -> str:
            text = " ".join(str(value).casefold().split())
            text = text.replace("grapichs", "graphics")
            return text

        duplicate_keys = pd.DataFrame(
            {column: results[column].map(normalize_for_dedup) for column in usable_key},
            index=results.index,
        )
        results = results.loc[~duplicate_keys.duplicated(keep="first")].copy()
    unique_config_count = int(len(results))

    results = results.head(int(top_n)).copy().reset_index(drop=True)
    results.insert(0, "Rank", np.arange(1, len(results) + 1))
    results["Similarity_Label"] = results["Similarity_Percent"].map(lambda x: f"{x:.2f}%")
    return results, candidate_count, unique_config_count


def build_scenarios() -> list[dict[str, Any]]:
    """Skenario demo untuk evaluasi perilaku sistem, bukan ground-truth akurasi."""
    return [
        {
            "name": "Mahasiswa",
            "profile": {
                "budget_max": 15_000_000,
                "min_ram_gb": 16,
                "min_storage_gb": 512,
                "target_ram_gb": 16,
                "target_storage_gb": 512,
                "target_screen_size": 14.0,
                "categories": {"Processor_Brand": "Intel", "CPU_Family": "Intel Core 5"},
            },
        },
        {
            "name": "Office",
            "profile": {
                "budget_max": 12_000_000,
                "min_ram_gb": 8,
                "min_storage_gb": 256,
                "target_ram_gb": 8,
                "target_storage_gb": 512,
                "target_screen_size": 14.0,
                "categories": {"GPU_Type": "Integrated"},
            },
        },
        {
            "name": "Gaming",
            "profile": {
                "budget_max": 25_000_000,
                "min_ram_gb": 16,
                "min_storage_gb": 512,
                "target_ram_gb": 16,
                "target_storage_gb": 512,
                "target_screen_size": 15.6,
                "categories": {"GPU_Type": "Dedicated"},
                "required_gpu_type": "Dedicated",
            },
        },
        {
            "name": "Anggaran Terbatas",
            "profile": {
                "budget_max": 10_000_000,
                "min_ram_gb": 8,
                "min_storage_gb": 256,
                "target_ram_gb": 8,
                "target_storage_gb": 512,
                "target_screen_size": 14.0,
                "categories": {},
            },
        },
    ]
