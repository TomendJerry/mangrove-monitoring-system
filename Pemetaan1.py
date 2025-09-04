# ==========================================================
#                   PENGATURAN AWAL & LIBRARY
# ==========================================================
import warnings
warnings.filterwarnings("ignore")

import os
import pathlib
import random
import webbrowser
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import rasterio
import seaborn as sns
from PIL import Image
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
import ee
import geemap
from datetime import datetime, timedelta

# Variabel Global
acq_date = "Unknown"

# ==========================================================
#      BAGIAN 1: KONEKSI GEE & PERSIAPAN DATA (VERSI PENCARIAN DATA TERBARU)
# ==========================================================
print("==========================================================")
print("Memulai skrip, menghubungkan ke Google Earth Engine (GEE)...")
print("==========================================================")

try:
    # Ganti 'skripsi8' dengan ID Project GEE Anda jika diperlukan
    # ee.Authenticate() # Jalankan baris ini jika otentikasi diperlukan
    ee.Initialize(project='skripsi8')
    print("Koneksi ke Google Earth Engine berhasil.\n")
except Exception as e:
    print(f"KESALAHAN FATAL: Gagal terhubung ke Google Earth Engine.")
    print(f"Detail Error: {e}")
    sys.exit()

# Mendefinisikan area studi (Area of Interest - AOI)
geojson_polygon = {
    "type": "Polygon",
    "coordinates": [[
        [106.72612, -6.107597], [106.73363, -6.115625], [106.734531, -6.113917],
        [106.7329, -6.111824], [106.737321, -6.106444], [106.737535, -6.103968],
        [106.737235, -6.102388], [106.737492, -6.100552], [106.736634, -6.099783],
        [106.72612, -6.107597]
    ]]
}
polygon = ee.Geometry.Polygon(geojson_polygon["coordinates"])

# <<< LOGIKA PENCARIAN DATA TERBARU DIMULAI DI SINI >>>

print("Memulai pencarian otomatis untuk data citra terbaru...")
# Mencari mundur dalam interval 6 bulan (180 hari)
# Akan mencoba hingga 5 kali (mundur ~2.5 tahun) jika diperlukan
collection = None
start_date_str, end_date_str = "", ""

for attempt in range(5):
    end_date = datetime.now() - timedelta(days=180 * attempt)
    start_date = end_date - timedelta(days=180)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"\n[Pencarian ke-{attempt + 1}] Mencari data antara {start_date_str} dan {end_date_str}...")

    # Mencari koleksi citra Sentinel-2 pada periode waktu yang sedang diuji
    current_collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                          .filterBounds(polygon)
                          .filterDate(start_date_str, end_date_str)
                          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5)))
    
    collection_size = current_collection.size().getInfo()
    
    if collection_size > 0:
        print(f"--> SUKSES: Ditemukan {collection_size} citra berkualitas baik di periode ini.")
        collection = current_collection
        break # Hentikan loop karena data sudah ditemukan
    else:
        print(f"--> INFO: Tidak ada citra yang memenuhi kriteria di periode ini.")

if collection is None:
    print("\n[KESALAHAN FATAL] Tidak ditemukan data citra yang memenuhi kriteria setelah mencari selama ~2.5 tahun.")
    sys.exit()

# Mendapatkan tanggal-tanggal sumber dari koleksi yang berhasil ditemukan
print("\nMendeteksi tanggal akuisisi dari semua citra sumber...")
timestamps = collection.aggregate_array('system:time_start').getInfo()
source_dates = [datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d') for ts in timestamps]
print("Tanggal-tanggal citra yang akan digunakan untuk komposit:")
for date_str in sorted(source_dates):
    print(f"- {date_str}")

# Simpan daftar tanggal ke file
os.makedirs("static", exist_ok=True)
source_dates_path = "static/composite_source_dates.txt"
with open(source_dates_path, "w") as f:
    f.write(f"Citra Komposit dari periode {start_date_str} hingga {end_date_str} dibuat dari {len(source_dates)} citra berikut:\n")
    for date_str in sorted(source_dates):
        f.write(f"{date_str}\n")
print(f"[+] Daftar tanggal sumber berhasil disimpan ke: {source_dates_path}\n")

# Variabel start_date dan end_date akan digunakan juga oleh Bagian 3 secara otomatis
start_date, end_date = start_date_str, end_date_str # Update variabel global jika perlu di skrip lain
acq_date = f"Komposit {start_date} - {end_date}"

# <<< LOGIKA PENCARIAN DATA TERBARU SELESAI >>>


# Proses pembuatan komposit median tetap sama
image = collection.median()
print(f"Berhasil membuat citra komposit median dari periode terbaru yang ditemukan.")

date_file_path = "static/acquisition_date.txt"
with open(date_file_path, "w") as f:
    f.write(acq_date)
print(f"[+] Keterangan '{acq_date}' berhasil disimpan ke: {date_file_path}")

bands_to_download = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
image = image.select(bands_to_download)
download_dir = "Dataset/Pemetaan"
os.makedirs(download_dir, exist_ok=True)

print("\n--- Memulai Proses Unduh & Timpa (Replace) Citra Komposit ---")
for i, band_name in enumerate(bands_to_download, 1):
    single_band = image.select([band_name])
    output_path = os.path.join(download_dir, f"B1 ({i}).tif")
    print(f"[>]  Mengunduh & Menimpa {band_name} -> B1 ({i}).tif")
    geemap.ee_export_image(
        single_band, output_path, scale=10, region=polygon, file_per_band=False
    )
print("--- Selesai Proses Unduh & Timpa ---\n")


# ==========================================================
#                   BAGIAN 2: CROP CITRA
# ==========================================================
output_folder = "Dataset/Cropeed Bands/cropped"
os.makedirs(output_folder, exist_ok=True)
polygon_wgs84 = shape(geojson_polygon)
input_files = [f"Dataset/Pemetaan/B1 ({i}).tif" for i in range(1, 13)]
print("--- Memulai Proses Cropping Citra ---")
for idx, tif_path in enumerate(input_files, start=1):
    if not os.path.exists(tif_path):
        print(f"File tidak ditemukan: {tif_path}")
        continue
    with rasterio.open(tif_path) as src:
        gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[polygon_wgs84])
        gdf_proj = gdf.to_crs(src.crs)
        out_image, out_transform = mask(src, [mapping(gdf_proj.geometry[0])], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
            "transform": out_transform, "count": 1
        })
        output_path = os.path.join(output_folder, f"band_{idx}.tif")
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image[0], 1)
        print(f" Cropped: {os.path.basename(tif_path)} -> band_{idx}.tif")
print("--- Selesai Proses Cropping ---\n")


# =================================================================================
#      BAGIAN 3: PENGAMBILAN SAMPEL MENGGUNAKAN PROXY GROUND TRUTH (SUDAH DIPERBAIKI)
# =================================================================================
print("\n" + "=" * 70)
print("           MEMULAI PENGAMBILAN SAMPEL & EKSPOR PROXY GROUND TRUTH")
print("=" * 70)

try:
    # Membuat koleksi data Dynamic World sesuai rentang waktu yang ditemukan di Bagian 1
    dw_collection = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate(start_date, end_date).select('label')

    # Mendeteksi tanggal sumber dari Dynamic World
    dw_collection_size = dw_collection.size().getInfo()
    print(f"\nDitemukan {dw_collection_size} citra Dynamic World untuk periode {start_date} hingga {end_date}.")
    
    print("\nMendeteksi tanggal akuisisi dari semua citra sumber Dynamic World...")
    dw_timestamps = dw_collection.aggregate_array('system:time_start').getInfo()
    dw_source_dates = [datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d') for ts in dw_timestamps]
    unique_dw_dates = sorted(list(set(dw_source_dates)))
    
    print(f"Ditemukan {len(unique_dw_dates)} tanggal akuisisi unik untuk citra Dynamic World:")
    for date_str in unique_dw_dates:
        print(f"- {date_str}")
        
    # Menyimpan daftar tanggal unik ke file untuk dokumentasi
    dw_dates_path = "static/dynamic_world_source_dates.txt"
    with open(dw_dates_path, "w") as f:
        f.write(f"Komposit mode Dynamic World dari periode {start_date} hingga {end_date} dibuat dari {dw_collection_size} citra dengan tanggal-tanggal akuisisi unik berikut:\n")
        for date_str in unique_dw_dates:
            f.write(f"{date_str}\n")
    print(f"[+] Daftar tanggal sumber Dynamic World berhasil disimpan ke: {dw_dates_path}\n")

    # Proses pembuatan komposit mode dan remapping kelas tetap sama
    dw_class_image = dw_collection.mode()

    from_classes_dw = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    to_classes_dw   = [2, 1, 3, 1, 3, 3, 3, 3, 3] 
    ground_truth_remapped = dw_class_image.remap(from_classes_dw, to_classes_dw).rename('Class')

    # Lanjutan skrip untuk ekspor dan sampling
    dw_export_dir = "Dataset/DynamicWorld Export"
    os.makedirs(dw_export_dir, exist_ok=True)
    
    # --- EKSPOR 1: VERSI CROP SESUAI AOI (UNTUK SAMPLING) ---
    output_dw_remapped_cropped = os.path.join(dw_export_dir, "DynamicWorld_Remapped_AOI_Cropped.tif")
    print("\n--- Memulai Ekspor Peta Dynamic World Hasil Remapping (DI-CROP SESUAI AOI) ---")
    geemap.ee_export_image(
        ground_truth_remapped.clip(polygon), 
        filename=output_dw_remapped_cropped, 
        scale=10, 
        region=polygon
    )
    print(f"[OK] Peta hasil remapping yang di-crop berhasil diekspor ke: {output_dw_remapped_cropped}")

    # --- EKSPOR 2: VERSI TANPA CROP (DALAM BOUNDING BOX) ---
    output_dw_remapped_full = os.path.join(dw_export_dir, "DynamicWorld_Remapped_Full_BBox.tif")
    print("\n--- Memulai Ekspor Peta Dynamic World TANPA CROP (dalam Bounding Box) ---")
    export_region_bbox = polygon.bounds()
    geemap.ee_export_image(
        ground_truth_remapped,
        filename=output_dw_remapped_full, 
        scale=10, 
        region=export_region_bbox
    )
    print(f"[OK] Peta tanpa crop (full bbox) berhasil diekspor ke: {output_dw_remapped_full}")


    training_image = image.addBands(ground_truth_remapped)

    print("\nMelakukan stratified sampling untuk mendapatkan titik data latih yang seimbang...")
    samples = training_image.stratifiedSample(
        numPoints=3000, classBand='Class', region=polygon, scale=10, geometries=True,
        dropNulls=True
    )

    def add_coords(feature):
        coords = feature.geometry().coordinates()
        return feature.set({'latitude': coords.get(1), 'longitude': coords.get(0)})
    samples_with_coords = samples.map(add_coords)

    print("Mengunduh data sampel dari GEE ke DataFrame...")
    df = geemap.ee_to_df(samples_with_coords)
    print("[OK] Data sampel Proxy Ground Truth berhasil dibuat.")

    if 'geometry' in df.columns:
        df.drop('geometry', axis=1, inplace=True)

    class_map = {1: "Mangrove", 2: "Air", 3: "Non-Mangrove"}
    df['Class'] = df['Class'].map(class_map)
    df.rename(columns={'longitude': 'Longitude', 'latitude': 'Latitude'}, inplace=True)

    print("\nDistribusi kelas dari hasil sampling:")
    print(df['Class'].value_counts())
    print("-" * 30)

    # <<< KODE TAMBAHAN UNTUK EKSPOR CSV MENTAH >>>
    output_csv_mentah = "Dataset/sampel_mentah_dari_gee.csv"
    try:
        df.to_csv(output_csv_mentah, index=False)
        print(f"\n[OK] Data sampel mentah ({len(df)} titik) berhasil diekspor ke: {output_csv_mentah}")
    except Exception as e:
        print(f"\n[PERINGATAN] Gagal menyimpan data sampel mentah: {e}")
    # <<< AKHIR KODE TAMBAHAN >>>

except Exception as e:
    print(f"\n[KESALAHAN FATAL] Gagal saat mengambil sampel atau ekspor Peta Dynamic World: {e}")
    sys.exit()


# ==========================================================
#             BAGIAN 4: PERSIAPAN DATA UNTUK KLASIFIKASI
# ==========================================================
print("\n[*] Mempersiapkan data untuk klasifikasi...")

raw_features = [f"B{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, '8A', 9, 11, 12]]
df_clean_raw = df.dropna(subset=raw_features + ["Class"]).copy().reset_index(drop=True)

# --- CONTOH KODE EKSPOR (untuk ditambahkan) ---
output_path_clean_raw = "Dataset/sampel_bersih_mentah.csv"
try:
    df_clean_raw.to_csv(output_path_clean_raw, index=False)
    print(f"\n[OK] Data sampel bersih (mentah) berhasil diekspor ke: {output_path_clean_raw}")
except Exception as e:
    print(f"\n[PERINGATAN] Gagal menyimpan data sampel bersih (mentah): {e}")
# --- AKHIR CONTOH ---

X_raw = df_clean_raw[raw_features]
y = df_clean_raw["Class"]
print("[OK] X (fitur mentah) dan y (target) telah dipisahkan.")

def process_features(data):
    df_processed = data.copy()
    band_columns = [f"B{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, '8A', 9, 11, 12]]
    df_processed[band_columns] = df_processed[band_columns] / 10000.0
    
    df_processed["NDVI"] = (df_processed["B8"] - df_processed["B4"]) / ((df_processed["B8"] + df_processed["B4"]).replace(0, np.nan))
    df_processed["EVI"] = 2.5 * (df_processed["B8"] - df_processed["B4"]) / ((df_processed["B8"] + 6 * df_processed["B4"] - 7.5 * df_processed["B2"] + 1).replace(0, np.nan))
    df_processed["SAVI"] = 1.5 * (df_processed["B8"] - df_processed["B4"]) / ((df_processed["B8"] + df_processed["B4"] + 0.5).replace(0, np.nan))
    
    return df_processed

features = raw_features + ["NDVI", "EVI", "SAVI"]
total_titik = len(df)
titik_valid = len(df_clean_raw)
titik_gagal = total_titik - titik_valid

print(f"\nStatistik Titik Sampel:")
print(f" . Total titik sample dari GEE: {total_titik}")
print(f" . Titik valid (mentah)       : {titik_valid}")
print(f" . Titik dibuang (NaN value)  : {titik_gagal}")


# =======================================================================================
#               BAGIAN 5: EKSPERIMEN RASIO DATA LATIH & UJI
# =======================================================================================
ratios_to_test = { "70:30": 0.30, "80:20": 0.20, "90:10": 0.10 }
results_summary = []
best_cv_accuracy = 0.0
best_ratio_str = None
best_model = None
best_X_val, best_y_val = None, None

print("\n\n" + "="*70)
print("   MEMULAI EKSPERIMEN HIBRIDA: VALIDASI DULU, BARU CROSS-VALIDATION")
print("="*70)

for ratio_str, test_size_val in ratios_to_test.items():
    train_ratio = int(100 - (test_size_val * 100))
    test_ratio = int(test_size_val * 100)
    
    print(f"\n\n--- Menguji Rasio {train_ratio}:{test_ratio} ---")
    
    if len(y.unique()) < 2 or titik_valid < 10:
        print("   [SKIP] Tidak cukup data atau kelas untuk melakukan split dan training.")
        continue

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y, test_size=test_size_val, stratify=y, random_state=42
    )

    print("   - Melakukan scaling dan kalkulasi indeks pada set train & validasi...")
    X_train = process_features(X_train_raw)
    X_val = process_features(X_val_raw)
    
    X_train.dropna(inplace=True)
    y_train = y_train.loc[X_train.index]
    X_val.dropna(inplace=True)
    y_val = y_val.loc[X_val.index]

    if len(y_train) < 5 or len(y_val) < 1:
        print("   [SKIP] Tidak cukup data setelah dropna untuk training/validasi.")
        continue

    loop_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    loop_model.fit(X_train[features], y_train)
    
    y_val_pred_loop = loop_model.predict(X_val[features])

    accuracy_val = accuracy_score(y_val, y_val_pred_loop)
    print(f"   [Evaluasi pada Data Validasi Hold-out]")
    print(f"   - Akurasi Validasi       : {accuracy_val * 100:.4f}%")

    cv_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    cv_scores = cross_val_score(cv_model, X_train[features], y_train, cv=5, scoring='accuracy')
    mean_cv_accuracy = np.mean(cv_scores)
    std_cv_accuracy = np.std(cv_scores)
    print(f"   [Evaluasi CV pada Data Latih]")
    print(f"   - Rata-rata Akurasi CV: {mean_cv_accuracy * 100:.4f}% (+- {std_cv_accuracy * 100:.4f})")
    
    precision_val = precision_score(y_val, y_val_pred_loop, average='weighted', zero_division=0)
    recall_val = recall_score(y_val, y_val_pred_loop, average='weighted', zero_division=0)
    f1_val = f1_score(y_val, y_val_pred_loop, average='weighted', zero_division=0)

    results_summary.append({
        "Rasio (Train:Test)": f"{train_ratio}:{test_ratio}",
        "Rata-rata Akurasi CV (%)": mean_cv_accuracy * 100,
        "Akurasi Validasi (%)": accuracy_val * 100,
        "Presisi Validasi (%)": precision_val * 100,
        "Recall Validasi (%)": recall_val * 100,
        "F1-Score Validasi (%)": f1_val * 100
    })

    if mean_cv_accuracy > best_cv_accuracy:
        print(f"\n[!] RASIO BARU TERBAIK DITEMUKAN! (Berdasarkan Akurasi CV: {mean_cv_accuracy*100:.4f}%)")
        best_cv_accuracy = mean_cv_accuracy
        best_ratio_str = f"{train_ratio}:{test_ratio}"
        best_X_val, best_y_val = X_val.copy(), y_val.copy()
        best_model = loop_model

# ==============================================================================
#               BAGIAN 6: EVALUASI AKHIR MENGGUNAKAN MODEL TERBAIK
# ==============================================================================
if not results_summary:
    print("\n[KESALAHAN] Tidak ada hasil eksperimen yang berhasil dijalankan. Menghentikan skrip.")
    sys.exit()

print("\n\n" + "="*80)
print("                         HASIL AKHIR PERBANDINGAN SEMUA RASIO")
print("="*80)
results_df = pd.DataFrame(results_summary)
print(results_df.to_string(index=False, columns=[
    "Rasio (Train:Test)", "Akurasi Validasi (%)", "Presisi Validasi (%)", 
    "Recall Validasi (%)", "F1-Score Validasi (%)", "Rata-rata Akurasi CV (%)"
]))
print("-" * 80)
print(f"\n[!] Rasio terbaik berdasarkan rata-rata Cross-Validation adalah {best_ratio_str} dengan akurasi CV {best_cv_accuracy * 100:.4f}%.")
print("    Model dari rasio ini akan digunakan untuk prediksi akhir.")
print("="*80)

y_val_pred = best_model.predict(best_X_val[features])
best_classification_report = classification_report(best_y_val, y_val_pred, target_names=best_model.classes_, zero_division=0)
best_cm_val = confusion_matrix(best_y_val, y_val_pred, labels=best_model.classes_)

# --- TAMBAHKAN KODE BARU DI SINI ---
print("\n\n" + "="*70)
print(f"        LAPORAN KLASIFIKASI RINCI (MODEL DARI RASIO {best_ratio_str})")
print("="*70)
print(best_classification_report)
print("="*70 + "\n")
# --- AKHIR KODE BARU ---

# --- KODE BARU UNTUK MENAMPILKAN STRUKTUR X dan y ---
print("\n" + "="*70)
print("                   STRUKTUR DATA X (FITUR) DAN y (TARGET)")
print("="*70)
# Membuat data X dan y yang sudah diproses dari keseluruhan data bersih untuk ditampilkan
final_X = process_features(df_clean_raw[raw_features])
final_y = df_clean_raw["Class"]

print(f"Dimensi Matriks Fitur (X): {final_X.shape}")
print("Contoh 5 baris pertama dari data X (setelah scaling dan rekayasa fitur):")
print(final_X.head())
print("\n" + "-"*40)
print(f"Dimensi Vektor Target (y): {final_y.shape}")
print("Contoh 5 baris pertama dari data y:")
print(final_y.head())
print("="*70 + "\n")
# --- AKHIR KODE BARU ---

print("[i] Memproses seluruh data sampel untuk output akhir...")
df_clean_processed = process_features(df_clean_raw)
df_clean_processed.dropna(subset=features, inplace=True)
print("[OK] Seluruh data sampel telah diproses (scaled + indices).")

# --- KODE EKSPOR BARU UNTUK DATA FINAL SIAP MODEL ---
output_path_final_processed = "Dataset/data_final_untuk_model.csv"
try:
    # Menyimpan DataFrame yang berisi X (fitur yang sudah diproses) dan y (target)
    df_clean_processed.to_csv(output_path_final_processed, index=False)
    print(f"\n[OK] Data final gabungan (fitur X dan target y) yang siap untuk model berhasil diekspor ke: {output_path_final_processed}")
except Exception as e:
    print(f"\n[PERINGATAN] Gagal menyimpan data final untuk model: {e}")
# --- AKHIR KODE EKSPOR ---


print("[i] Melakukan prediksi pada KESELURUHAN data bersih menggunakan model terbaik...")
all_predictions = best_model.predict(df_clean_processed[features])
df_clean_processed["Predicted_Class"] = all_predictions
print(f"[OK] Kolom 'Predicted_Class' telah diisi untuk {len(df_clean_processed)} baris data bersih.")


# ==============================================================================
#                   BAGIAN 7-12: GENERASI OUTPUT
# ==============================================================================
output_df = df_clean_processed 
output_df.to_csv("static/klasifikasi_titik_dengan_model.csv", index=False)
output_df.to_csv("./Dataset/klasifikasi_titik_dengan_model.csv", index=False)
print("[OK] Hasil klasifikasi titik sampel disimpan ke 'static/klasifikasi_titik_dengan_model.csv'")

plt.figure(figsize=(6, 5))
sns.heatmap(best_cm_val, annot=True, fmt='d', xticklabels=best_model.classes_, yticklabels=best_model.classes_, cmap="Blues")
plt.title(f"Confusion Matrix (Validation Data - Rasio {best_ratio_str})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()
print("[OK] Confusion matrix disimpan ke 'static/confusion_matrix.png'")

# --- KODE BARU UNTUK GRAFIK FEATURE IMPORTANCE ---
print("[OK] Membuat grafik tingkat kepentingan fitur...")
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Tingkat Kepentingan Fitur (Feature Importance)')
plt.xlabel('Importance Score')
plt.ylabel('Fitur')
plt.tight_layout()
plt.savefig("static/grafik_feature_importance.png")
plt.close()
print("[OK] Grafik feature importance disimpan ke 'static/grafik_feature_importance.png'")
# --- AKHIR KODE BARU ---

pixel_area_ha = 0.01
label_order = ["Air", "Non-Mangrove", "Mangrove"]
color_order = ["blue", "orange", "green"]
area_per_class = pd.Series(y_val_pred).value_counts().reindex(label_order, fill_value=0) * pixel_area_ha

plt.figure(figsize=(8, 5))
bars = plt.bar(label_order, area_per_class.values, color=color_order)
plt.title(f"Estimasi Luas Tutupan Lahan (Data Validasi - Rasio {best_ratio_str})")
plt.ylabel("Luas (Hektar)")
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.2f}", ha='center')
plt.tight_layout()
plt.savefig("static/grafik_luas_tutupan_lahan_validasi.png")
plt.close()
print("[OK] Grafik luas (data validasi) disimpan ke 'static/grafik_luas_tutupan_lahan_validasi.png'")

print("\n[...] Memprediksi seluruh piksel pada citra untuk membuat peta tutupan lahan...")
band_paths = [f"Dataset/Cropeed Bands/cropped/band_{i}.tif" for i in range(1, 13)]
bands_data = [rasterio.open(p) for p in band_paths]
band_arrays = [b.read(1) for b in bands_data]

stacked_bands = np.stack(band_arrays, axis=-1)
h, w, num_bands = stacked_bands.shape
reshaped_bands = stacked_bands.reshape(-1, num_bands)
column_names_full_image = [f"B{i}" for i in [1, 2, 3, 4, 5, 6, 7, 8, '8A', 9, 11, 12]]
df_full_image_raw = pd.DataFrame(reshaped_bands, columns=column_names_full_image)

print("[...] Memproses fitur untuk keseluruhan citra (scaling dan kalkulasi indeks)...")
df_full_image_processed = process_features(df_full_image_raw)
print("[OK] Fitur keseluruhan citra berhasil diproses.")

features_2d = df_full_image_processed[features].values
valid_mask = ~np.isnan(features_2d).any(axis=1)
valid_features = features_2d[valid_mask]
predicted_labels = best_model.predict(valid_features)

print("[*] Menghitung dan membuat grafik luas total tutupan lahan untuk keseluruhan citra...")
area_total_per_class = pd.Series(predicted_labels).value_counts().reindex(label_order, fill_value=0) * pixel_area_ha
plt.figure(figsize=(8, 5))
bars_total = plt.bar(label_order, area_total_per_class.values, color=color_order)
plt.title(f"Luas Total Tutupan Lahan (Keseluruhan Citra - {acq_date})")
plt.ylabel("Luas (Hektar)")
for bar in bars_total:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.2f}", ha='center')
plt.tight_layout()
plt.savefig("static/grafik_luas_tutupan_lahan_total.png")
plt.close()
print("[OK] Grafik luas total keseluruhan citra berhasil disimpan ke 'static/grafik_luas_tutupan_lahan_total.png'")

label_array = np.full(features_2d.shape[0], fill_value="Unknown", dtype=object)
label_array[valid_mask] = predicted_labels
label_array = label_array.reshape(h, w)
rgb_map = {"Air": (0, 0, 255), "Non-Mangrove": (255, 165, 0), "Mangrove": (0, 128, 0), "Unknown": (0, 0, 0)}
rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
for label, color in rgb_map.items():
    mask = label_array == label
    rgb_image[mask] = color
Image.fromarray(rgb_image).save("static/klasifikasi_rgb_output.png")
print("[OK] Peta RGB hasil klasifikasi disimpan ke 'static/klasifikasi_rgb_output.png'")

m = folium.Map(location=[output_df["Latitude"].mean(), output_df["Longitude"].mean()], zoom_start=14)
color_map_folium = {"Air": "blue", "Non-Mangrove": "orange", "Mangrove": "green"}
df_map_sample = output_df.sample(n=min(3000, len(output_df)), random_state=42)
for _, row in df_map_sample.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]], radius=4, color=color_map_folium.get(row["Predicted_Class"], "gray"),
        fill=True, fill_opacity=0.6,
        popup=f"Tanggal: {acq_date}<br>Class: {row['Predicted_Class']}<br>NDVI: {row['NDVI']:.3f}<br>EVI: {row['EVI']:.3f}"
    ).add_to(m)
peta_path = "static/peta_interaktif_klasifikasi.html"
m.save(peta_path)
print(f"[OK] Peta interaktif disimpan ke '{peta_path}'")

print("\n[>>] Membuka hasil visualisasi:")
try:
    for filename in ["confusion_matrix.png", "grafik_feature_importance.png", "grafik_luas_tutupan_lahan_total.png", "klasifikasi_rgb_output.png"]:
        path = pathlib.Path(f"static/{filename}").resolve()
        print(f" - Membuka: {path}")
        Image.open(path).show()

    webbrowser.open_new_tab(pathlib.Path(peta_path).resolve().as_uri())
    print(f" - Membuka peta interaktif di browser.")
except Exception as e:
    print(f" Gagal membuka file secara otomatis. Silakan buka manual dari folder 'static'. Error: {e}")


# ==============================================================================
#                   BAGIAN 13: SIMPAN STATISTIK AKHIR
# ==============================================================================
with open("static/statistik_pemetaan.txt", "w", encoding="utf-8") as f:
    f.write(f"STATISTIK PEMETAAN TUTUPAN LAHAN (Citra: {acq_date})\n")
    f.write("="*60 + "\n\n")

    f.write("Statistik Titik Sampel Awal (Proxy Ground Truth):\n")
    f.write(f". Total titik sample dari GEE : {total_titik}\n")
    f.write(f". Titik valid setelah diolah  : {titik_valid}\n")
    f.write(f". Titik dibuang (NaN value)   : {titik_gagal}\n")
    f.write("\n" + "="*60 + "\n\n")

    f.write("Evaluasi Model Machine Learning:\n")
    f.write(f". Model Terbaik Dipilih dari Rasio: {best_ratio_str}\n")
    f.write(f". Rata-rata Akurasi 5-Fold Cross-Validation: {best_cv_accuracy * 100:.2f}%\n\n")
    f.write("--- Laporan Klasifikasi Rinci Model Terbaik (diuji pada data validasi) ---\n")
    f.write(best_classification_report)
    f.write("\n" + "="*60 + "\n\n")

    f.write("Estimasi Luas Tutupan Lahan (dari Data Validasi Model Terbaik):\n")
    f.write(f" (Berdasarkan {len(y_val_pred)} piksel validasi, 1 piksel = {pixel_area_ha} Ha)\n")
    f.write("-" * 60 + "\n")
    f.write(f". Mangrove      : {area_per_class.get('Mangrove', 0):.2f} Ha\n")
    f.write(f". Non-Mangrove  : {area_per_class.get('Non-Mangrove', 0):.2f} Ha\n")
    f.write(f". Air           : {area_per_class.get('Air', 0):.2f} Ha\n")
    f.write("-" * 60 + "\n")
    f.write(f". Total Area    : {area_per_class.sum():.2f} Ha\n")
    f.write("\n" + "="*60 + "\n\n")

    f.write("Estimasi Luas Tutupan Lahan (dari KESELURUHAN CITRA):\n")
    f.write(f" (Berdasarkan {len(predicted_labels)} piksel citra yang valid, 1 piksel = {pixel_area_ha} Ha)\n")
    f.write("-" * 60 + "\n")
    f.write(f". Mangrove      : {area_total_per_class.get('Mangrove', 0):.2f} Ha\n")
    f.write(f". Non-Mangrove  : {area_total_per_class.get('Non-Mangrove', 0):.2f} Ha\n")
    f.write(f". Air           : {area_total_per_class.get('Air', 0):.2f} Ha\n")
    f.write("-" * 60 + "\n")
    f.write(f". Total Area    : {area_total_per_class.sum():.2f} Ha\n")

print("\n[OK] File statistik 'static/statistik_pemetaan.txt' telah diperbarui.")
print("\n--- Proses Selesai ---")