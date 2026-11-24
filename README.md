# Sistem Rekomendasi Laptop — Streamlit

Aplikasi interaktif untuk proyek UTS Sistem Rekomendasi menggunakan **Content-Based Filtering**, **One-Hot Encoding**, **StandardScaler**, dan **Cosine Similarity**.

## Fitur

- Memilih jenis penggunaan laptop.
- Menentukan anggaran maksimal, RAM minimum, dan penyimpanan minimum.
- Mengatur target spesifikasi untuk menghitung kedekatan profil.
- Menampilkan daftar rekomendasi dan skor cosine similarity.
- Visualisasi skor dan hubungan harga dengan skor kemiripan.
- Mengunduh hasil rekomendasi ke CSV.
- Dataset demo 50.000 baris disertakan; pengguna dapat mengunggah CSV lengkap melalui sidebar.

## Jalankan secara lokal

1. Pastikan Python 3.10 atau lebih baru terpasang.
2. Ekstrak folder proyek.
3. Buka terminal pada folder proyek, lalu jalankan:

   ```bash
   python -m venv .venv
   ```

4. Aktifkan environment:
   - Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
5. Instal dependensi dan jalankan aplikasi:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

6. Aplikasi akan terbuka di browser, biasanya pada `http://localhost:8501`.

## Deploy ke Streamlit Community Cloud

1. Buat repository baru di GitHub.
2. Unggah seluruh isi folder proyek ini ke repository (termasuk `data/laptop_demo_sample.csv` dan `.streamlit/config.toml`).
3. Buka https://share.streamlit.io/ dan masuk menggunakan GitHub.
4. Pilih **Create app**, lalu pilih repository, branch, dan file utama `app.py`.
5. Klik **Deploy**. Streamlit akan memasang pustaka berdasarkan `requirements.txt`.
6. Setelah aplikasi aktif, salin URL publik untuk dicantumkan dalam laporan UTS.

## Menggunakan dataset lengkap

Aplikasi otomatis menggunakan sampel demo 50.000 baris saat pertama dibuka. Untuk analisis dataset penuh:

1. Klik uploader **Unggah dataset CSV lengkap** di sidebar.
2. Pilih `global_laptop_dataset_1.2M.csv`.
3. Aplikasi hanya membaca kolom yang diperlukan model, bukan semua 18 kolom dataset.

Dataset penuh berukuran sekitar 114 MB. Pemrosesan file penuh memerlukan RAM lebih besar dan dapat lebih lambat pada hosting gratis. Untuk demo publik, sampel yang disertakan biasanya lebih ringan. Untuk hasil analisis final proyek, unggah dan jalankan dataset penuh pada mesin dengan RAM memadai.

## Kolom dataset yang dibutuhkan

`Usage_Type`, `RAM_GB`, `Storage_GB`, `Cores`, `Threads`, `Base_Clock`, `Boost_Clock`, `TDP`, `CPU_Performance`, `GPU_Performance`, `Price_USD`.

## Catatan metodologi dan keterbatasan

- Filter kandidat menggunakan jenis penggunaan, harga maksimal, RAM minimum, dan storage minimum.
- `Usage_Type` diubah dengan One-Hot Encoding.
- Fitur numerik distandardisasi menggunakan StandardScaler yang di-fit bertahap agar penggunaan memori lebih efisien.
- Cosine Similarity digunakan untuk mengurutkan kandidat berdasarkan kedekatan vektor dengan profil pengguna.
- Skor similarity bukan probabilitas atau metrik akurasi.
- Dataset asal memiliki inkonsistensi pada beberapa pasangan merek/model laptop dan GPU. Karena itu, aplikasi hanya menampilkan kecocokan spesifikasi numerik dan tidak mengklaim nama produk sebagai rekomendasi pasar yang tervalidasi.
- Mode awal menggunakan sampel acak 50.000 baris untuk demonstrasi. Unggah dataset lengkap untuk mereproduksi hasil pada keseluruhan CSV.
