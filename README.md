# Sistem Rekomendasi Laptop Indonesia — Streamlit

Aplikasi demonstrasi Content-Based Filtering untuk katalog laptop Indonesia (100 produk). Aplikasi menggunakan One-Hot Encoding, StandardScaler, dan Cosine Similarity berdasarkan matriks fitur yang dibuat pada Google Colab.

## Struktur repository

```text
.
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml
└── data/
    ├── laptop_indonesia_featured.csv
    ├── laptop_features_encoded.npz
    └── feature_metadata.json
```

## Jalankan secara lokal

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Ekstrak ZIP ini.
2. Buat repository GitHub baru.
3. Upload **seluruh isi folder hasil ekstrak ke root repository**, termasuk folder `data/` dan `.streamlit/`.
4. Buka https://share.streamlit.io/ dan login menggunakan GitHub.
5. Pilih repository dan branch `main`.
6. Isi **Main file path** dengan `app.py`, lalu deploy.

## Cara kerja

1. Sistem memfilter kandidat berdasarkan anggaran maksimum, RAM minimum, storage minimum, dan jenis GPU jika pengguna menetapkannya sebagai syarat wajib.
2. Sistem membangun vektor profil dari kategori pilihan serta target RAM, storage, dan ukuran layar.
3. Sistem menghitung Cosine Similarity antara profil dan vektor fitur produk yang sudah diencoding.
4. Hasil diurutkan berdasarkan similarity tertinggi.
5. Listing dengan kombinasi merek dan spesifikasi yang sama dideduplikasi; bila ada beberapa listing identik, harga terendah diprioritaskan.

## Catatan metodologis

- `Price_IDR` digunakan sebagai filter anggaran dan **tidak** masuk ke vektor Cosine Similarity.
- Skor Cosine Similarity menunjukkan kemiripan vektor, bukan akurasi atau probabilitas pembelian.
- Harga adalah snapshot sesuai `Checked_Date`, bukan harga real-time. Periksa ulang harga, stok, dan ketersediaan pada halaman penjual.
- Dataset berisi 100 listing dari satu penjual; cakupannya belum mewakili semua laptop yang beredar di Indonesia.
