"""
pipeline.py
============
Data Preparation & Pipeline Project 

Alur:
Raw CSV -> Load Data -> Data Inspection -> Data Cleaning
-> Data Transformation -> Processed CSV

Dataset  : automobileEDA_dirty_training.csv (dataset otomotif)
Input    : data/raw/automobileEDA_dirty_training.csv
Output   : data/processed/automobileEDA_processed.csv

Cara menjalankan:
data-pipeline-assignment/src/pipeline.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# ---------------------------------------------------------------------------
# KONFIGURASI PATH
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "automobileEDA_dirty_training.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "automobileEDA_processed.csv")

"""Mencetak header section di terminal agar mudah dibaca."""
def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. EXTRACT - Function untuk membaca dataset
# ---------------------------------------------------------------------------
"""Membaca dataset mentah (raw) dari folder data/raw/ sebagai DataFrame."""
def load_data(path: str) -> pd.DataFrame:
    print_section("1. EXTRACT: MEMBACA DATASET")
    df = pd.read_csv(path)
    print(f"Dataset berhasil dimuat dari: {path}")
    print(f"Ukuran dataset (baris, kolom): {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. INSPECT - Function untuk memeriksa kondisi awal dataset
# ---------------------------------------------------------------------------
def inspect_data(df: pd.DataFrame) -> dict:
    """
    Melakukan pemeriksaan awal terhadap dataset:
    - 5 baris pertama
    - Jumlah baris & kolom
    - Nama kolom
    - Tipe data tiap kolom
    - Jumlah missing values per kolom
    - Jumlah duplicate records
    - Nilai unik pada kolom kategorikal yang relevan
    """
    print_section("2. DATA INSPECTION: KONDISI AWAL DATASET")

    print("\n-- 5 Baris Pertama --")
    print(df.head())

    print(f"\n-- Jumlah Baris dan Kolom --\n{df.shape[0]} baris, {df.shape[1]} kolom")

    print("\n-- Nama Kolom --")
    print(list(df.columns))

    print("\n-- Tipe Data Tiap Kolom --")
    print(df.dtypes)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\n-- Jumlah Missing Values per Kolom (yang > 0) --")
    print(missing if not missing.empty else "Tidak ada missing values.")

    n_dupes = df.duplicated().sum()
    print(f"\n-- Jumlah Duplicate Records -- \n{n_dupes} baris terduplikasi")

    categorical_cols = [
        "make", "aspiration", "num-of-doors", "body-style",
        "drive-wheels", "engine-location", "engine-type",
        "num-of-cylinders", "fuel-system", "horsepower-binned",
    ]
    print("\n-- Nilai Unik pada Kolom Kategorikal Relevan --")
    for col in categorical_cols:
        if col in df.columns:
            print(f"  {col}: {sorted(df[col].dropna().unique().tolist())}")

    report = {
        "shape": df.shape,
        "missing_values": missing.to_dict(),
        "duplicates": int(n_dupes),
    }
    return report


# ---------------------------------------------------------------------------
# 3. TRANSFORM (CLEANING) - Function untuk melakukan data cleaning
# ---------------------------------------------------------------------------
def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    """
    Kolom transaction_date memiliki 4 format penulisan tanggal berbeda:
      - YYYY-MM-DD        (ISO)          contoh: 2025-01-01
      - DD/MM/YYYY        (slash)        contoh: 14/01/2025
      - MM-DD-YYYY        (dash numerik) contoh: 01-15-2025
      - DD-Mon-YYYY       (nama bulan)   contoh: 04-Jan-2025
    Melakukan parsing, lalu menyeragamkannya menjadi datetime (ISO).
    """
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%b-%Y"]

    def parse_one(value):
        if pd.isna(value):
            return pd.NaT
        for fmt in formats:
            try:
                return pd.to_datetime(value, format=fmt)
            except (ValueError, TypeError):
                continue
        return pd.NaT  # tidak cocok format manapun

    return series.apply(parse_one)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Melakukan proses data cleaning:
    - Menyeragamkan format transaction_date
    - Menghapus spasi tidak perlu & menyeragamkan kapitalisasi kolom kategorikal
    - Memperbaiki tipe data num-of-cylinders (teks -> angka)
    - Menangani missing values sesuai kondisi tiap kolom
    - Menghapus duplicate records
    Dataset asli (parameter df) TIDAK diubah langsung; function bekerja pada copy.
    """
    print_section("3. DATA CLEANING")

    data = df.copy()
    report = {"changed_columns": []}

    rows_before = len(data)
    missing_before = data.isnull().sum().sum()
    dupes_before = data.duplicated().sum()

    # --- 3.1 Menyeragamkan format transaction_date ---
    data["transaction_date"] = _parse_mixed_dates(data["transaction_date"])
    report["changed_columns"].append("transaction_date (format tanggal diseragamkan -> datetime ISO)")

    # --- 3.2 Membersihkan spasi & menyeragamkan kapitalisasi kolom kategorikal (teks) ---
    text_cols = [
        "make", "aspiration", "num-of-doors", "body-style", "drive-wheels",
        "engine-location", "engine-type", "num-of-cylinders", "fuel-system",
        "horsepower-binned",
    ]
    for column in text_cols:
        if column in data.columns:
            before_unique = data[column].nunique(dropna=True)
            data[column] = data[column].astype("string").str.strip().str.lower()
            after_unique = data[column].nunique(dropna=True)
            if before_unique != after_unique:
                report["changed_columns"].append(
                    f"{column} (whitespace dihapus, kapitalisasi diseragamkan menjadi lowercase; "
                    f"{before_unique} -> {after_unique} kategori unik)"
                )

    # --- 3.3 Memperbaiki tipe data: num-of-cylinders dari teks ke angka ---
    word_to_num = {
        "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "eight": 8, "twelve": 12,
    }
    data["num-of-cylinders"] = data["num-of-cylinders"].map(word_to_num).astype("Int64")
    report["changed_columns"].append(
        "num-of-cylinders (tipe data diperbaiki dari teks/string menjadi numerik/Int64)"
    )

    # --- 3.4 Menangani missing values (SEMUA diimputasi, tidak ada baris dihapus) ---
    # Urutan penting: imputasi dilakukan SEBELUM menghapus duplikat, karena ada
    # baris yang hanya berbeda pada nilai yang kosong (contoh: 2 baris dengan
    # make kosong yang sebenarnya identik di seluruh kolom lain). Begitu nilai
    # kosong tersebut diisi dengan cara yang sama, baris tersebut baru dapat
    # terdeteksi sebagai duplikat sejati oleh drop_duplicates().

    # make (2 missing): diisi dengan MODUS (make yang paling sering muncul).
    # Nama merek tidak bisa dihitung/diestimasi secara matematis, sehingga
    # nilai yang paling representatif secara statistik adalah modusnya.
    if data["make"].isnull().sum() > 0:
        make_mode = data["make"].mode().iloc[0]
        data["make"] = data["make"].fillna(make_mode)

    # transaction_date (2 missing/gagal parsing): diisi dengan forward-fill
    # (nilai tanggal terdekat sebelumnya), karena data tampak tersusun
    # berurutan berdasarkan waktu transaksi.
    if data["transaction_date"].isnull().sum() > 0:
        data["transaction_date"] = data["transaction_date"].ffill().bfill()

    # price (3 missing): diisi dengan rata-rata (mean) harga.
    # Jumlah missing kecil relatif terhadap ukuran dataset, sehingga mean
    # imputation tidak signifikan mengubah distribusi harga secara keseluruhan.
    if data["price"].isnull().sum() > 0:
        data["price"] = data["price"].fillna(data["price"].mean())

    # num-of-doors (2 missing): diisi dengan modus per body-style
    # (mengikuti pola umum: sedan mayoritas four, hatchback mayoritas two)
    if data["num-of-doors"].isnull().sum() > 0:
        mode_map = (
            data.dropna(subset=["num-of-doors"])
            .groupby("body-style")["num-of-doors"]
            .agg(lambda x: x.mode().iloc[0])
        )
        data["num-of-doors"] = data.apply(
            lambda row: mode_map.get(row["body-style"], data["num-of-doors"].mode().iloc[0])
            if pd.isna(row["num-of-doors"]) else row["num-of-doors"],
            axis=1,
        )

    # stroke (4 missing) & horsepower (3 missing): numerik kontinu -> diisi dengan rata-rata (mean)
    for col in ["stroke", "horsepower"]:
        if data[col].isnull().sum() > 0:
            mean_val = data[col].mean()
            data[col] = data[col].fillna(mean_val)

    # horsepower-binned (1 missing): dihitung ulang berdasarkan nilai horsepower
    # agar konsisten (bukan diisi sembarang kategori)
    if data["horsepower-binned"].isnull().sum() > 0:
        bins = [0, 100, 175, data["horsepower"].max() + 1]
        labels = ["low", "medium", "high"]
        recalculated = pd.cut(data["horsepower"], bins=bins, labels=labels)
        data["horsepower-binned"] = data["horsepower-binned"].fillna(recalculated.astype("string"))

    missing_after_fill = data.isnull().sum().sum()

    # --- 3.5 Menghapus duplicate records (dilakukan SETELAH seluruh imputasi) ---
    dupes_found = data.duplicated().sum()
    data = data.drop_duplicates()

    rows_after = len(data)
    missing_after = data.isnull().sum().sum()

    # --- Ringkasan cleaning ---
    print(f"Jumlah data sebelum cleaning : {rows_before} baris")
    print(f"Jumlah data sesudah cleaning : {rows_after} baris")
    print(f"Jumlah missing values sebelum cleaning : {missing_before}")
    print(f"Jumlah missing values sesudah cleaning : {missing_after}")
    print(f"Jumlah baris duplikat yang dihapus      : {dupes_found}")
    print("\nKolom yang mengalami perubahan:")
    for c in report["changed_columns"]:
        print(f"  - {c}")

    report.update({
        "rows_before": rows_before,
        "rows_after": rows_after,
        "missing_before": int(missing_before),
        "missing_after": int(missing_after),
        "duplicates_removed": int(dupes_found),
    })
    return data, report


# ---------------------------------------------------------------------------
# 4. TRANSFORM - Function untuk melakukan data transformation
# ---------------------------------------------------------------------------
def transform_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Melakukan transformasi data menggunakan scikit-learn, mencakup:

    1. Normalisasi (Min-Max Scaling) pada seluruh kolom numerik kontinu yang
       masih dalam skala aslinya (belum 0-1): symboling, curb-weight,
       engine-size, horsepower, peak-rpm, city-mpg, highway-mpg, price.
       (length & width tidak disentuh karena sudah dalam skala 0-1 sejak
       sumber data original).
    2. Encoding kolom kategorikal:
       - One-Hot Encoding pada kolom nominal: body-style, drive-wheels,
         aspiration, engine-type, engine-location, fuel-system.
       - Ordinal Encoding pada horsepower-binned (low=0, medium=1, high=2)
         -> horsepower_ordinal, karena kategori ini punya urutan/tingkatan.
       - Frequency Encoding pada make -> make_freq, karena make memiliki
         banyak kategori (proporsi kemunculan lebih informatif daripada
         one-hot yang akan menghasilkan puluhan kolom).
       - num-of-doors dan num-of-cylinders dikonversi ke numerik lalu turut
         di-scale ke rentang 0-1 agar konsisten dengan kolom numerik lain.
    Kolom kategorikal asli yang sudah di-encode dihapus dari dataset akhir
    agar tidak redundan dengan hasil encoding-nya.
    """
    print_section("4. DATA TRANSFORMATION")

    data = df.copy()
    report = {"normalization": {}, "encoding": {}}

    # -----------------------------------------------------------------
    # 4.1 Encoding num-of-doors & num-of-cylinders menjadi numerik dulu
    #     (supaya bisa ikut di-Min-Max Scaling bersama kolom numerik lain)
    # -----------------------------------------------------------------
    door_map = {"two": 0, "four": 1}
    data["num-of-doors"] = data["num-of-doors"].map(door_map).astype(float)

    # num-of-cylinders sudah numerik (Int64) sejak tahap cleaning
    data["num-of-cylinders"] = data["num-of-cylinders"].astype(float)

    # -----------------------------------------------------------------
    # 4.2 Normalisasi Min-Max Scaling pada beberapa kolom numerik sekaligus
    # -----------------------------------------------------------------
    numeric_cols_to_scale = [
        "symboling", "curb-weight", "engine-size", "horsepower",
        "peak-rpm", "city-mpg", "highway-mpg", "price",
        "num-of-doors", "num-of-cylinders",
    ]
    before_sample = data[numeric_cols_to_scale].head(3).to_dict(orient="records")

    scaler = MinMaxScaler()
    data[numeric_cols_to_scale] = scaler.fit_transform(data[numeric_cols_to_scale])

    after_sample = data[numeric_cols_to_scale].head(3).round(3).to_dict(orient="records")

    print(f"Min-Max Scaling (sklearn) diterapkan pada kolom: {numeric_cols_to_scale}")
    print(f"  Contoh sebelum (3 baris pertama): {before_sample}")
    print(f"  Contoh sesudah (3 baris pertama): {after_sample}")

    report["normalization"] = {
        "columns": numeric_cols_to_scale,
        "method": "MinMaxScaler (scikit-learn)",
        "before_sample": before_sample,
        "after_sample": after_sample,
    }

    # -----------------------------------------------------------------
    # 4.3 One-Hot Encoding pada kolom kategorikal nominal
    # -----------------------------------------------------------------
    onehot_cols = [
        "body-style", "drive-wheels", "aspiration",
        "engine-type", "engine-location", "fuel-system",
    ]
    encoder = OneHotEncoder(sparse_output=False, dtype=int)
    encoded_array = encoder.fit_transform(data[onehot_cols])
    encoded_cols = encoder.get_feature_names_out(onehot_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=data.index)

    print(f"\nOne-Hot Encoding (sklearn) pada kolom: {onehot_cols}")
    print(f"  Total kolom baru dihasilkan: {len(encoded_cols)}")

    data = pd.concat([data.drop(columns=onehot_cols), encoded_df], axis=1)

    # -----------------------------------------------------------------
    # 4.4 Ordinal Encoding pada horsepower-binned -> horsepower_ordinal
    # -----------------------------------------------------------------
    ordinal_map = {"low": 0, "medium": 1, "high": 2}
    before_binned = data["horsepower-binned"].head(5).tolist()
    data["horsepower_ordinal"] = data["horsepower-binned"].map(ordinal_map)
    data = data.drop(columns=["horsepower-binned"])

    print(f"\nOrdinal Encoding pada kolom 'horsepower-binned' -> 'horsepower_ordinal'")
    print(f"  Mapping: {ordinal_map}")
    print(f"  Contoh sebelum: {before_binned}")
    print(f"  Contoh sesudah: {data['horsepower_ordinal'].head(5).tolist()}")

    # -----------------------------------------------------------------
    # 4.5 Frequency Encoding pada make -> make_freq
    # -----------------------------------------------------------------
    freq_map = data["make"].value_counts(normalize=True)
    before_make = data["make"].head(5).tolist()
    data["make_freq"] = data["make"].map(freq_map)
    data = data.drop(columns=["make"])

    print(f"\nFrequency Encoding pada kolom 'make' -> 'make_freq'")
    print(f"  Contoh sebelum: {before_make}")
    print(f"  Contoh sesudah: {data['make_freq'].head(5).round(4).tolist()}")

    report["encoding"] = {
        "one_hot_columns": onehot_cols,
        "one_hot_new_columns": list(encoded_cols),
        "ordinal_column": "horsepower-binned -> horsepower_ordinal",
        "frequency_column": "make -> make_freq",
    }

    # -----------------------------------------------------------------
    # 4.6 Menghapus kolom transaction_date dari processed dataset akhir
    # -----------------------------------------------------------------
    # transaction_date sudah dibersihkan & diperiksa pada tahap sebelumnya
    # (bagian dari Data Inspection & Data Cleaning), namun kolom ini bukan
    # bagian dari skema dataset original (automobileEDA.csv) yang menjadi
    # referensi mentor, sehingga tidak disertakan pada processed dataset
    # akhir agar skema kolom konsisten dengan automobile_processed.csv.
    if "transaction_date" in data.columns:
        data = data.drop(columns=["transaction_date"])
        print("\nKolom 'transaction_date' dihapus dari processed dataset akhir")
        print("(sudah diperiksa & dibersihkan di tahap sebelumnya, tapi bukan bagian skema referensi)")

    return data, report


# ---------------------------------------------------------------------------
# 5. LOAD - Function untuk menyimpan hasil
# ---------------------------------------------------------------------------
def save_data(df: pd.DataFrame, path: str) -> None:
    """Menyimpan processed dataset ke folder data/processed/."""
    print_section("5. LOAD: MENYIMPAN PROCESSED DATASET")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Processed dataset berhasil disimpan ke: {path}")
    print(f"Ukuran akhir dataset: {df.shape}")


# ---------------------------------------------------------------------------
# MAIN PIPELINE (end-to-end)
# ---------------------------------------------------------------------------
def run_pipeline():
    print_section("MEMULAI DATA PIPELINE: automobileEDA_dirty_training.csv")

    # EXTRACT
    raw_df = load_data(RAW_PATH)

    # INSPECT (bagian dari tahap Transform: memeriksa)
    inspect_data(raw_df)

    # CLEAN (bagian dari tahap Transform: membersihkan)
    cleaned_df, clean_report = clean_data(raw_df)

    # TRANSFORM (bagian dari tahap Transform: mentransformasi)
    final_df, transform_report = transform_data(cleaned_df)

    # LOAD
    save_data(final_df, PROCESSED_PATH)

    print_section("PIPELINE SELESAI DIJALANKAN")
    print("Ringkasan akhir:")
    print(f"  - Baris awal (raw)      : {clean_report['rows_before']}")
    print(f"  - Baris akhir (processed): {len(final_df)}")
    print(f"  - Duplikat dihapus       : {clean_report['duplicates_removed']}")
    print(f"  - Missing values awal    : {clean_report['missing_before']}")
    print(f"  - Missing values akhir   : {clean_report['missing_after']}")


if __name__ == "__main__":
    run_pipeline()
