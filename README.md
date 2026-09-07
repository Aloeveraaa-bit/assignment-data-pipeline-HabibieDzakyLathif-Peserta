# Data Preparation & Pipeline Project

Project ini membangun pipeline sederhana menggunakan Python dan library seperti pandas, numpy, dan sejenisnya untuk mengubah dataset otomotif mentah (dirty) menjadi dataset bersih (processed) yang
siap digunakan untuk analisis atau pemodelan.

---

## 1. Deskripsi Dataset

- **Nama file input**: `automobileEDA_dirty_training.csv`
- **Sumber dataset**: disediakan mentor melalui link `https://s.id/dataset-sesi-3`
- **Isi dataset**: data spesifikasi teknis dan harga 205 kendaraan (make, dimensi,
  mesin, bahan bakar, harga, dsb.) beserta kolom tambahan `transaction_date`.


## 2. Sumber Dataset

File 
`automobileEDA_dirty_training.csv` Dataset utama digunakan sebagai input pipeline 
`automobileEDA_processed.csv` Dataset output dari pipeline

## 3. Struktur Folder Project

```
data-pipeline-assignment/
├── data/
│   ├── raw/
│   │   └── automobileEDA_dirty_training.csv
│   └── processed/
│       └── automobileEDA_processed.csv
├── src/
│   └── pipeline.py
├── documentation/
│   └── data-flow-diagram.png
├── README.md
└── requirements.txt
```

## 4. Kondisi Awal Dataset

- **Ukuran awal**: 205 baris x 30 kolom
- **Kolom dengan missing values**:

  | Kolom | Jumlah Missing |
  |---|---|
  | transaction_date | 2 |
  | make | 2 |
  | num-of-doors | 2 |
  | stroke | 4 |
  | horsepower | 3 |
  | price | 3 |
  | horsepower-binned | 1 |
- **Jumlah baris terduplikasi**: 4 baris (identik persis dengan baris lain)
- **Kolom dengan tipe data belum sesuai**:
  - `num-of-cylinders` bertipe teks (`"four"`, `"six"`, dst.) padahal secara
    semantik adalah nilai numerik.
  - `transaction_date` bertipe teks bebas, belum berupa tipe tanggal (`datetime`).
- **Penulisan kategori yang belum konsisten**:
  - `make`: variasi kapitalisasi & spasi ekstra, contoh `alfa-romero` vs
    `ALFA-ROMERO`, serta `"dodge  "` dan `"porsche  "` dengan trailing spaces.
  - `body-style`: campuran kapitalisasi, contoh `sedan`, `Sedan`, `SEDAN`.
  - `drive-wheels`: campuran kapitalisasi, contoh `rwd`, `RWD`, `AWD`, `4wd`.
  - `fuel-system`: campuran kapitalisasi, contoh `mpfi`, `MPFI`, `Mpfi`.
- **Permasalahan lain**:
  - `transaction_date` ditulis dalam **4 format berbeda** secara acak dalam satu
    kolom yang sama: `YYYY-MM-DD`, `DD/MM/YYYY`, `MM-DD-YYYY`, dan `DD-Mon-YYYY`
    (contoh: `2025-01-01`, `14/01/2025`, `01-15-2025`, `04-Jan-2025`).

## 5. Cleaning yang Dilakukan

Seluruh missing values diimputasi (diisi) alih-alih menghapus baris data, dengan
tujuan mempertahankan sebanyak mungkin data yang tersedia. Duplikat baru dihapus
setelah proses imputasi selesai. Hal ini membantu agar beberapa baris hanya
berbeda pada nilai yang kosong (misalnya 2 baris dengan `make` kosong yang
sebenarnya identik di seluruh kolom lain); begitu nilai kosong tersebut diisi, baris tersebut baru dapat terdeteksi sebagai duplikat.

| Permasalahan | Kolom | Metode Cleaning | Alasan |
|---|---|---|---|
| Format tanggal tidak konsisten | `transaction_date` | Parsing multi-format lalu diseragamkan menjadi `datetime` (ISO) | Format ganda dalam satu kolom membuat data tidak bisa diurutkan/dianalisis berdasarkan waktu tanpa diseragamkan lebih dulu |
| Kapitalisasi & spasi tidak konsisten | `make`, `body-style`, `drive-wheels`, `fuel-system`, dll (kolom teks kategorikal) | `strip()` untuk menghapus spasi + `lower()` untuk menyeragamkan huruf kecil | Nilai yang secara semantik sama (misal `RWD` dan `rwd`) harus dianggap satu kategori, bukan kategori berbeda |
| Tipe data belum sesuai | `num-of-cylinders` | Mapping teks angka (`"four"` → `4`, dst.) ke tipe `Int64` | Kolom ini bersifat numerik (jumlah silinder) sehingga lebih tepat diproses sebagai angka, bukan string |
| Missing values pada `make` | `make` | Dihapus | Nama merek tidak dapat diestimasi dan tidak digunakan pada proyek ini |
| Missing/gagal-parsing pada `transaction_date` | `transaction_date` | Dihapus | Awalnya digunakan pendekatan berdasarkan 2 nilai sebelum dan sesudah tetapi berakhir dihapus agar tidak menyebabkan kesalahan pada output  |
| Missing values pada `price` | `price` | Diisi dengan **nilai rata-rata (mean)** | Jumlah missing kecil relatif terhadap ukuran dataset sehingga mean imputation tidak signifikan mengubah distribusi harga secara keseluruhan |
| Missing values pada `num-of-doors` | `num-of-doors` | Diisi dengan **modus per `body-style`** | Jumlah pintu punya korelasi kuat dengan tipe bodi (mis. sedan mayoritas 4 pintu), sehingga imputasi berbasis grup lebih akurat dibanding modus global |
| Missing values pada data numerik kontinu | `stroke`, `horsepower` | Diisi dengan **nilai rata-rata (mean)** kolom tersebut | Jumlah missing kecil dan kolom bersifat numerik kontinu, sehingga mean imputation adalah pendekatan sederhana yang wajar dan tidak mengubah distribusi data secara signifikan |
| Missing values pada `horsepower-binned` | `horsepower-binned` | Dihitung ulang berdasarkan nilai `horsepower` (binning low/medium/high) | Nilai ini seharusnya turunan langsung dari `horsepower`, sehingga lebih konsisten dihitung ulang daripada diisi sembarang kategori |
| Baris terduplikasi | seluruh kolom | `drop_duplicates()` (dijalankan setelah seluruh imputasi selesai) | Baris yang identik persis adalah redundan dan dapat membiaskan hasil analisis/model jika tidak dihapus |

**Jumlah data sebelum dan sesudah cleaning:**

| Metode | Sebelum | Sesudah |
|---|---|---|
| Jumlah baris | 205 | 201 |
| Jumlah missing values (total) | 17 | 0 |
| Jumlah baris duplikat | 4 | 0 |

> Catatan: dataset mentah di `data/raw/` **tidak diubah sama sekali**; seluruh
> proses cleaning bekerja pada `pipeline.py`.

## 6. Transformasi yang Dilakukan

Berdasarkan perbandingan dengan `automobile_processed.csv` (referensi dari mentor),
transformasi diperluas agar seluruh kolom numerik kontinu berada dalam skala
seragam (0–1) dan seluruh kolom kategorikal ter-encode dengan metode yang sesuai
karakteristiknya.

### 6.1 Normalisasi — Min-Max Scaling

- **Kolom yang dinormalisasi**: `symboling`, `curb-weight`, `engine-size`,
  `horsepower`, `peak-rpm`, `city-mpg`, `highway-mpg`, `price`, `num-of-doors`,
  `num-of-cylinders`
- **Metode**: `MinMaxScaler` dari scikit-learn, menghasilkan rentang nilai 0–1
  untuk seluruh kolom tersebut.
- **Alasan pemilihan kolom**: kolom `length` dan `width` pada dataset ini sudah
  berskala 0–1 sejak sumber data original, sedangkan kolom-kolom di atas masih
  dalam skala aslinya yang berbeda-beda (contoh: `price` puluhan ribu,
  `peak-rpm` ribuan, `symboling` -3 sampai 3). Menyeragamkan skala ini penting
  agar tidak ada kolom yang "mendominasi" secara numerik saat data dipakai
  untuk pemodelan nantinya.
- `num-of-doors` dan `num-of-cylinders` dikonversi ke numerik terlebih dahulu
  (`two`→0/`four`→1, dan `"four"`→4 dst.) sebelum ikut di-scale, karena secara
  semantik keduanya adalah nilai berjenjang/ordinal.

| Kolom | Sebelum (baris 1) | Sesudah (baris 1) |
|---|---|---|
| symboling | 3 | 1.0 |
| curb-weight | 2548 | 0.411 |
| engine-size | 130 | 0.260 |
| horsepower | 111.0 | 0.294 |
| peak-rpm | 5000.0 | 0.347 |
| city-mpg | 21 | 0.222 |
| highway-mpg | 27 | 0.289 |
| price | 13495.0 | 0.208 |
| num-of-doors | two (0) | 0.0 |
| num-of-cylinders | four (4) | 0.2 |

### 6.2 Encoding

Empat metode encoding diterapkan sesuai karakteristik masing-masing kolom kategorikal:

| Metode | Kolom | Alasan |
|---|---|---|
| **One-Hot Encoding** | `body-style`, `drive-wheels`, `aspiration`, `engine-type`, `engine-location`, `fuel-system` | Kategori nominal (tidak berurutan) dengan jumlah kategori sedikit-menengah sehingga dapat direpresentasikan dengan biner terpisah|
| **Ordinal Encoding** | `horsepower-binned` → `horsepower_ordinal` (low=0, medium=1, high=2) | Kategori ini memiliki urutan/tingkatan yang jelas sehingga representasi angka berjenjang lebih tepat dibanding one-hot |
| **Frequency Encoding** | `make` → `make_freq` (proporsi kemunculan tiap merek) | `make` memiliki banyak kategori unik (22 setelah cleaning); one-hot akan menghasilkan puluhan kolom baru yang jarang terisi, sedangkan frequency encoding lebih ringkas dan tetap informatif |

Kolom kategorikal asli (`body-style`, `drive-wheels`, `aspiration`, `engine-type`,
`engine-location`, `fuel-system`, `horsepower-binned`, `make`) dihapus dari
dataset akhir setelah di-encode agar tidak redundan dengan representasi barunya.

**Contoh One-Hot Encoding (`drive-wheels`):**

| drive-wheels (sebelum) | drive-wheels_4wd | drive-wheels_awd | drive-wheels_fwd | drive-wheels_rwd |
|---|---|---|---|---|
| rwd | 0 | 0 | 0 | 1 |
| fwd | 0 | 0 | 1 | 0 |
| 4wd | 1 | 0 | 0 | 0 |

**Contoh Ordinal Encoding (`horsepower-binned`):**

| horsepower-binned (sebelum) | horsepower_ordinal (sesudah) |
|---|---|
| low | 0 |
| medium | 1 |
| high | 2 |

**Contoh Frequency Encoding (`make`):**

| make (sebelum) | make_freq (sesudah) |
|---|---|
| alfa-romero | 0.0154 |
| audi | 0.0359 |

## 7. Cara Menginstal Dependency

```bash
pip install -r requirements.txt
```
# berisi pandas, numpy, dan scikit-learn sebagai basic library untuk pembentukan pipeline data

## 8. Cara Menjalankan Pipeline

Jalankan dari folder project:

data-pipeline-assignment/src/pipeline.py
```

Seluruh tahapan (load, inspect, clean, transform, save) akan berjalan otomatis
dalam satu kali eksekusi, dan hasil pemeriksaan/log akan tercetak di terminal.

## 9. Alur ETL

```
Raw CSV (data/raw/) → Load Data → Data Inspection → Data Cleaning
→ Data Transformation → Processed CSV (data/processed/)
```

- **Extract**: `load_data()` membaca `automobileEDA_dirty_training.csv` dari `data/raw/`
- **Transform**: `inspect_data()`, `clean_data()`, `transform_data()` memeriksa,
  membersihkan, dan mentransformasi data
- **Load**: `save_data()` menyimpan hasil akhir ke `data/processed/`

Diagram visual tersedia di documentation/data-flow-diagram.png.

## 10. Lokasi Processed Dataset

```
data/processed/automobileEDA_processed.csv
```

Ukuran akhir: **201 baris x 50 kolom** — hasil dari 30 kolom asli (dikurangi
9 kolom yang dihapus: 8 kolom kategorikal asli yang sudah di-encode —
`make`, `body-style`, `drive-wheels`, `aspiration`, `engine-type`,
`engine-location`, `fuel-system`, `horsepower-binned` — dan kolom
`transaction_date` yang dihapus dari output akhir karena bukan bagian dari
skema dataset original) ditambah 29 kolom baru hasil encoding (27 kolom
one-hot + `horsepower_ordinal` + `make_freq`).

> Kolom `transaction_date` awalnya tetap diperiksa dan berakhir dihapus agar tidak menimbulkan
> Inspection & Data Cleaning (format tanggal diseragamkan  namun dihapus sebelum tahap Load , missing value diisi)

