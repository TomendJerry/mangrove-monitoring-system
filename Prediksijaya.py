# === INSTALL (Once only) ===
# pip install pandas numpy rasterio folium matplotlib scikit-learn lxml tqdm

# === IMPORT ===
import webbrowser
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import folium
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import random
import sys # Ditambahkan untuk sys.exit()

# Fungsi main() dari skrip sebelumnya untuk membuat file CSV awal
# Ini dijalankan hanya untuk memastikan file input ada.
def setup_initial_csv():
    os.makedirs("Dataset", exist_ok=True)
    input_path = "static/klasifikasi_titik_dengan_model.csv"
    output_path = "Dataset/klasifikasi_dengan_spesies_dbh.csv"
        
    if not os.path.exists(input_path):
        print(f"File input tidak ditemukan: {input_path}")
        print("   Pastikan Anda telah menjalankan skrip klasifikasi sebelumnya untuk menghasilkan file ini.")
        # Membuat file dummy jika tidak ada untuk mencegah error
        dummy_df = pd.DataFrame({
            'Latitude': [-6.1754, -6.1755],
            'Longitude': [106.8272, 106.8273],
            'Class': ['mangrove', 'mangrove']
        })
        dummy_df.to_csv(input_path, index=False)
        print("File dummy input telah dibuat.")

    df = pd.read_csv(input_path)
    mangrove_species_dbh = {
        "Avicennia marina": 14.3, "Rhizophora apiculata": 13.4, "Lumnitzera racemosa": 7.6,
        "Rhizophora stylosa": 14.7, "Aegiceras corniculatum": 13.1, "Ceriops tagal": 15.4,
    }
    species_list = list(mangrove_species_dbh.keys())

    def assign_species_and_dbh(row):
        if str(row['Class']).lower() == 'mangrove':
            species = random.choice(species_list)
            dbh = mangrove_species_dbh[species] + random.uniform(-1.5, 1.5)
            return pd.Series([species, round(dbh, 2)])
        else:
            return pd.Series([np.nan, np.nan])

    df[['Dominant_Species', 'dbh']] = df.apply(assign_species_and_dbh, axis=1)
    df.to_csv(output_path, index=False)
    print(f"File '{output_path}' berhasil dibuat/disimpan.")

# Jalankan fungsi setup di awal
setup_initial_csv()


# === SKRIP UTAMA DIMULAI DI SINI ===
warnings.filterwarnings("ignore")

# === 1. BACA CSV ===
df = pd.read_csv("Dataset/klasifikasi_dengan_spesies_dbh.csv")
# === KONVERSI KOORDINAT WGS84 -> CRS RASTER ===
try:
    with rasterio.open("Dataset/Cropeed Bands/cropped/band_4.tif") as src:
        raster_crs = src.crs  # CRS raster (biasanya UTM)
except rasterio.errors.RasterioIOError:
    print("\nKESALAHAN: Tidak dapat membuka file raster 'band_4.tif'.")
    print("   Pastikan skrip 'Pemetaan' telah berjalan dan menghasilkan file band yang telah di-crop.")
    sys.exit()


# Buat GeoDataFrame dari koordinat awal (WGS84)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), crs="EPSG:4326")

# Reproject ke CRS raster
gdf = gdf.to_crs(raster_crs)

# Tambahkan kolom koordinat baru dalam satuan raster CRS (meter/UTM)
df["X"] = gdf.geometry.x
df["Y"] = gdf.geometry.y
df = df.rename(columns={"Longtitude": "Longitude"}) if "Longtitude" in df.columns else df
df = df.dropna(subset=["Latitude", "Longitude", "dbh", "Dominant_Species"])
df["Class"] = df["Class"].str.lower()
df = df[df["Class"] == "mangrove"].copy().reset_index(drop=True)

print("\n================ DIAGNOSTIK DATA ================")
print(f"Jumlah titik mangrove setelah pemuatan awal: {len(df)}")
# ----------------------------------------------------

# === 2. EKSTRAK TANGGAL DARI FILE acquisition_date.txt ===
acq_date = "Unknown"
date_file_path = "static/acquisition_date.txt"

try:
    with open(date_file_path, 'r') as f:
        acq_date = f.read().strip()
    if not acq_date:
        acq_date = "2025-08-16" # Fallback date if file is empty
        print(f"Peringatan: File '{date_file_path}' kosong. Menggunakan tanggal fallback: {acq_date}")
except FileNotFoundError:
    print(f"Gagal membaca tanggal: File '{date_file_path}' tidak ditemukan.")
    print("   Pastikan skrip sebelumnya telah berjalan dan membuat file ini.")
    acq_date = "2025-08-16" # Fallback date if file not found
    print(f"   Menggunakan tanggal fallback: {acq_date}")
except Exception as e:
    print(f"Terjadi kesalahan saat membaca file '{date_file_path}': {e}")
    acq_date = "2025-08-16" # Fallback date on error

df["Acquisition_Date"] = acq_date
if not df.empty and acq_date != "Unknown":
    print(f"Tanggal Akuisisi Citra (dari file .txt): {df['Acquisition_Date'].iloc[0]}")
elif acq_date == "Unknown":
    print("Tanggal Akuisisi Citra: Tidak diketahui (file tidak ditemukan/kosong).")


# === 3. HITUNG NDVI & EVI ===
b2_path = "Dataset/Cropeed Bands/cropped/band_2.tif"
b4_path = "Dataset/Cropeed Bands/cropped/band_4.tif"
b8_path = "Dataset/Cropeed Bands/cropped/band_8.tif"

if not all(os.path.exists(p) for p in [b2_path, b4_path, b8_path]):
    print("\nKESALAHAN: File band TIF tidak ditemukan di 'Dataset/Cropeed Bands/cropped/'.")
    print("   Pastikan Anda telah menjalankan skrip pengunduhan dan cropping citra sebelumnya.")
    sys.exit()

b2 = rasterio.open(b2_path)
b4 = rasterio.open(b4_path)
b8 = rasterio.open(b8_path)

ndvi_list, evi_list = [], []
for x, y in tqdm(zip(df['X'], df['Y']), total=len(df), desc="Menghitung NDVI & EVI"):
    try:
        row, col = b4.index(x, y)
        nir = b8.read(1)[row, col].astype(float)/ 10000.0
        red = b4.read(1)[row, col].astype(float)/ 10000.0
        blue = b2.read(1)[row, col].astype(float)/ 10000.0
        
        if (nir + red) != 0:
            ndvi = (nir - red) / (nir + red)
        else:
            ndvi = np.nan

        denominator_evi = nir + 6 * red - 7.5 * blue + 1
        if denominator_evi != 0:
            evi = 2.5 * (nir - red) / denominator_evi
        else:
            evi = np.nan
            
    except IndexError: 
        ndvi, evi = np.nan, np.nan
        
    ndvi_list.append(ndvi)
    evi_list.append(evi)

df["NDVI"] = ndvi_list
df["EVI"] = evi_list

print(f"Jumlah data setelah perhitungan NDVI/EVI: {len(df)}")
print(f"Jumlah data dengan nilai NDVI yang valid: {df['NDVI'].notna().sum()}")
print(f"Jumlah data dengan nilai EVI yang valid: {df['EVI'].notna().sum()}")
# ----------------------------------------------------

# === 4. HITUNG AGB & AGC (KG/POHON) ===
def estimate_agb_agc(dbh_cm, species):
    if pd.isna(dbh_cm) or pd.isna(species):
        return np.nan, np.nan
    dbh = float(dbh_cm)
    species_lower = str(species).lower()
    
    rho_dict = {
        "rhizophora apiculata": 0.89, "bruguiera gymnorrhiza": 0.77,
        "avicennia marina": 0.60, "rhizophora stylosa": 0.94,
        "xylocarpus granatum": 0.63, "ceriops tagal": 0.95
    }
    rho = rho_dict.get(species_lower, np.mean(list(rho_dict.values())))

    if "avicennia" in species_lower:
        agb_kg = 0.1848 * (dbh ** 2.3524)
    elif "rhizophora" in species_lower:
        agb_kg = 0.0695 * (dbh ** 2.644) * rho
    elif "bruguiera" in species_lower:
        agb_kg = 0.0754 * (dbh ** 2.505) * rho
    elif "sonneratia" in species_lower:
        agb_kg = 0.153 * (dbh ** 2.348)
    else: # Rumus umum
        agb_kg = 0.251 * rho * (dbh ** 2.46)
        
    agc_kg = agb_kg * 0.47
    return agb_kg, agc_kg

df[["AGB_per_tree_kg", "AGC_per_tree_kg"]] = df.apply(lambda row: pd.Series(estimate_agb_agc(row["dbh"], row["Dominant_Species"])), axis=1)

# === 5. PERSIAPAN DATA LATIH & VALIDASI ===
# Kolom log_dbh tidak lagi digunakan untuk fitur AGB/AGC, tapi bisa disimpan untuk analisis lain
df["log_dbh"] = np.log1p(df["dbh"]) 
df["Species_Code"] = pd.factorize(df["Dominant_Species"])[0]
features = ["NDVI", "EVI", "Species_Code"]

df_model = df.dropna(subset=features + ["dbh", "AGB_per_tree_kg", "AGC_per_tree_kg"]).copy()

print(f"Jumlah data FINAL yang akan di-plot (df_model): {len(df_model)}")
print("===============================================\n")

if df_model.empty:
    print("\nKESALAHAN FATAL: Tidak ada data valid yang tersisa setelah filtering.")
    print("   Ini kemungkinan besar disebabkan karena koordinat titik (dari file .csv)")
    print("   berada di luar jangkauan citra raster (.tif), sehingga NDVI/EVI gagal dihitung (NaN).")
    sys.exit()

# === PERUBAHAN DIMULAI: Mendefinisikan fitur untuk model ===
# Fitur untuk model DBH (hanya berdasarkan data inderaja)
X_dbh = df_model[features]
y_dbh = df_model["dbh"]

# Fitur untuk model AGB & AGC (SEKARANG JUGA HANYA BERDASARKAN DATA INDERAJA)
# "log_dbh" telah dihapus dari sini untuk memperbaiki kebocoran data.
X_agb_agc = df_model[features] 
y_agb = df_model["AGB_per_tree_kg"]
y_agc = df_model["AGC_per_tree_kg"]
# === PERUBAHAN SELESAI ===

acq_date_label = df_model["Acquisition_Date"].iloc[0]

print("=========== DIAGNOSTIK KONTEN DATA ===========")
print(df_model[['dbh', 'AGB_per_tree_kg', 'NDVI', 'EVI']].describe())
print("\nDistribusi Spesies:")
print(df_model['Dominant_Species'].value_counts())
print("==============================================\n")
# ----------------------------------------------------

# === 6. FUNGSI PELATIHAN & EVALUASI (TIDAK ADA PERUBAHAN DI FUNGSI INI) ===
def check_overfitting(model, X, y, label):
    """Fungsi ini tetap sama, untuk memeriksa stabilitas model final."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    print(f"\nOverfitting Check for {label}:")
    print(f"   R² Train: {r2_train:.3f}, R² Test: {r2_test:.3f}, Gap: {abs(r2_train - r2_test):.3f}")

def evaluate_splits_with_kfold(X, y, label):
    split_ratios = [0.10, 0.20, 0.30] 
    results_summary = []
    n_folds = 5
    
    print(f"\n--- Mencari Split Terbaik untuk Model {label} via Nested K-Fold CV ---")

    for test_size in split_ratios:
        split_label = f"{int((1-test_size)*100)}:{int(test_size*100)}"
        X_train_outer, X_test_outer, y_train_outer, y_test_outer = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=42)
        fold_scores = {'r2': [], 'rmse': [], 'mae': []}
        
        desc = f"Evaluasi Split {split_label} (K-Fold CV)"
        for train_idx, val_idx in tqdm(kf.split(X_train_outer), total=n_folds, desc=desc):
            X_train_inner, X_val_inner = X_train_outer.iloc[train_idx], X_train_outer.iloc[val_idx]
            y_train_inner, y_val_inner = y_train_outer.iloc[train_idx], y_train_outer.iloc[val_idx]
            model.fit(X_train_inner, y_train_inner)
            preds = model.predict(X_val_inner)
            fold_scores['r2'].append(r2_score(y_val_inner, preds))
            fold_scores['rmse'].append(np.sqrt(mean_squared_error(y_val_inner, preds)))
            fold_scores['mae'].append(mean_absolute_error(y_val_inner, preds))
            
        avg_r2 = np.mean(fold_scores['r2'])
        avg_rmse = np.mean(fold_scores['rmse'])
        avg_mae = np.mean(fold_scores['mae'])
        
        results_summary.append({
            'Train:Test': split_label, 'Avg_R2': avg_r2, 'Avg_RMSE': avg_rmse,
            'Avg_MAE': avg_mae, 'test_size': test_size
        })

    results_df = pd.DataFrame(results_summary).drop('test_size', axis=1)
    print("\nHasil Evaluasi Rata-rata K-Fold untuk Setiap Split Data:")
    print(results_df.to_string(index=False))

    best_result = max(results_summary, key=lambda item: item['Avg_R2'])
    best_split_label = best_result['Train:Test']
    print(f"\nSplit terbaik dipilih: {best_split_label} (Avg R² K-Fold = {best_result['Avg_R2']:.3f})")

    best_test_size = best_result['test_size']
    X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(
        X, y, test_size=best_test_size, random_state=42
    )
    model_for_plot = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=42)
    model_for_plot.fit(X_train_best, y_train_best)
    preds_for_plot = model_for_plot.predict(X_test_best)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_best, preds_for_plot, alpha=0.6, color='green')
    plt.plot([min(y_test_best), max(y_test_best)], [min(y_test_best), max(y_test_best)], 'r--', linewidth=2)
    plt.title(f"Prediction vs Actual for {label}\n(Split: {best_split_label} | Date: {acq_date_label})")
    plt.xlabel(f"Actual {label} (Unseen Test Data)")
    plt.ylabel(f"Predicted {label}")
    plt.grid(True)
    plt.tight_layout()

    output_dir = "static/plots"
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"prediksi_{label.replace(' ', '_').replace('/', '_')}.png")
    plt.savefig(image_path)
    print(f"Grafik prediksi (test data) disimpan di: {image_path}")
    webbrowser.open(f"file:///{os.path.abspath(image_path)}")
    plt.close()

    print(f"Melatih model final {label} pada seluruh dataset ({len(X)} sampel)...")
    final_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=42).fit(X, y)
    
    check_overfitting(final_model, X, y, label)
    
    return final_model

# === 7. LATIH SEMUA MODEL & BUAT PREDIKSI ===
print("\n--- Memulai Pelatihan Model DBH ---")
model_dbh = evaluate_splits_with_kfold(X_dbh, y_dbh, "DBH (cm)")
df_model["Predicted_DBH_cm"] = model_dbh.predict(X_dbh)

print("\n--- Memulai Pelatihan Model AGB ---")
model_agb = evaluate_splits_with_kfold(X_agb_agc, y_agb, "AGB (kg/tree)")
df_model["Predicted_AGB_kg"] = model_agb.predict(X_agb_agc)

print("\n--- Memulai Pelatihan Model AGC ---")
model_agc = evaluate_splits_with_kfold(X_agb_agc, y_agc, "AGC (kg/tree)")
df_model["Predicted_AGC_kg"] = model_agc.predict(X_agb_agc)


# === 8. SIMPAN HASIL PREDIKSI LENGKAP ===
cols_to_round = [
    'AGB_per_tree_kg', 'AGC_per_tree_kg', 
    'Predicted_AGB_kg', 'Predicted_AGC_kg', 
    'Predicted_DBH_cm', 'log_dbh'
]

for col in cols_to_round:
    if col in df_model.columns:
        df_model[col] = df_model[col].round(2)

output_csv_path = "static/output_prediksi_agb_agc.csv"
df_model.to_csv(output_csv_path, index=False, sep=";", decimal=",")
print(f"\nCSV lengkap disimpan di: {output_csv_path}")


# === 9. KALKULASI TOTAL DAN ESTIMASI LUAS ===
num_samples = len(df_model)
estimated_area_ha = num_samples * (10 * 10) / 10000
estimated_area_km2 = estimated_area_ha / 100
tree_density_per_ha = num_samples / estimated_area_ha if estimated_area_ha > 0 else 0
total_agb_per_ha = df_model["Predicted_AGB_kg"].mean() * tree_density_per_ha
total_agc_per_ha = df_model["Predicted_AGC_kg"].mean() * tree_density_per_ha
total_agb_ton_per_ha = total_agb_per_ha / 1000
total_agc_ton_per_ha = total_agc_per_ha / 1000

print(f"\n--- Ringkasan Statistik ({acq_date_label}) ---")
print(f"Jumlah Titik Mangrove Valid : {num_samples}")
print(f"Estimasi Luas Area Mangrove: {estimated_area_ha:.2f} ha ({estimated_area_km2:.4f} km²)")
print(f"Asumsi Kepadatan Pohon   : {tree_density_per_ha:.0f} pohon/ha (berdasarkan resolusi piksel)")
print(f"Total Estimasi AGB (Kasar) : {total_agb_ton_per_ha:,.2f} ton/ha")
print(f"Total Estimasi AGC (Kasar) : {total_agc_ton_per_ha:,.2f} ton/ha")


# === 10. BUAT PETA INTERAKTIF FOLIUM ===
if not df_model.empty:
    map_center = [df_model["Latitude"].mean(), df_model["Longitude"].mean()]
    m = folium.Map(location=map_center, zoom_start=14, tiles="CartoDB positron")

    for _, row in df_model.iterrows():
        popup_html = (
            f"<b>Tanggal:</b> {row['Acquisition_Date']}<br>"
            f"<b>Spesies:</b> {row['Dominant_Species']}<br>"
            f"<b>DBH (Prediksi):</b> {row['Predicted_DBH_cm']:.2f} cm<br>"
            f"<b>NDVI:</b> {row['NDVI']:.3f}<br>"
            f"<b>EVI:</b> {row['EVI']:.3f}<br>"
            f"<b>Pred. AGB:</b> {row['Predicted_AGB_kg']:.2f} kg/pohon<br>"
            f"<b>Pred. AGC:</b> {row['Predicted_AGC_kg']:.2f} kg/pohon"
        )
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]], radius=4,
            color='darkgreen', fill=True, fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=350)
        ).add_to(m)

    map_path = "static/peta_prediksi_agb_agc_mangrove.html"
    m.save(map_path)
    print(f"\nPeta interaktif disimpan sebagai: {map_path}")
    webbrowser.open(f"file:///{os.path.abspath(map_path)}")


# === 11. FUNGSI UNTUK MEMBUAT KESIMPULAN (DENGAN MODIFIKASI) ===
# === PERUBAHAN DIMULAI: Menambahkan paragraf asumsi pada fungsi kesimpulan ===
def generate_conclusion(agb_ton_ha, density_ha, area_ha):
    """
    Menghasilkan teks kesimpulan berdasarkan nilai AGB, kepadatan, dan luas total.
    Ambang batas (threshold) didasarkan pada literatur umum ekosistem mangrove.
    """
    # Ambang batas untuk AGB (ton/ha)
    high_agb_threshold = 250
    moderate_agb_threshold = 100

    if agb_ton_ha >= high_agb_threshold:
        health_status = "SANGAT SEHAT & PRODUKTIF"
        interpretation = (
            f"Dengan estimasi AGB sebesar {agb_ton_ha:,.2f} ton/ha, "
            "area mangrove ini menunjukkan kondisi yang sangat sehat, matang, dan produktif. "
            "Struktur hutannya kemungkinan besar kompleks dan rapat, didominasi oleh pohon-pohon besar."
        )
        implications = (
            "Implikasi Positif:\n"
            "1. Penyerapan Karbon Maksimal: Kemampuan menyerap dan menyimpan CO2 sangat tinggi.\n"
            "2. Perlindungan Pesisir yang Kuat: Mampu memberikan perlindungan maksimal terhadap abrasi.\n"
            "3. Keanekaragaman Hayati Tinggi: Menjadi habitat ideal bagi beragam spesies."
        )
    elif agb_ton_ha >= moderate_agb_threshold:
        health_status = "CUKUP SEHAT"
        interpretation = (
            f"Dengan estimasi AGB sebesar {agb_ton_ha:,.2f} ton/ha, area mangrove ini berada dalam kondisi cukup sehat. "
            "Hutan ini mungkin merupakan hutan yang sedang dalam tahap pertumbuhan (regenerasi) atau mengalami "
            "tekanan lingkungan ringan."
        )
        implications = (
            "Implikasi:\n"
            "1. Penyerapan Karbon Baik: Masih memiliki peran penting dalam penyerapan karbon.\n"
            "2. Perlindungan Pesisir Moderat: Memberikan tingkat perlindungan yang cukup baik.\n"
            "3. Potensi Peningkatan: Ada ruang untuk meningkatkan kesehatan ekosistem melalui konservasi."
        )
    else:  # Jika AGB rendah
        health_status = "KURANG SEHAT / TERDEGRADASI"
        interpretation = (
            f"Estimasi AGB yang rendah ({agb_ton_ha:,.2f} ton/ha) mengindikasikan bahwa ekosistem mangrove ini "
            "kemungkinan berada dalam kondisi kurang sehat, terdegradasi, atau merupakan hutan yang sangat muda. "
            "Hasil ini sangat dipengaruhi oleh asumsi kepadatan pohon yang digunakan dalam analisis."
        )
        implications = (
            "Implikasi Negatif jika Kondisi Buruk:\n"
            "1. Penyerapan Karbon Rendah: Kemampuannya untuk menyerap CO2 dari atmosfer berkurang.\n"
            "2. Kerentanan Pesisir Meningkat: Fungsi perlindungan terhadap abrasi menjadi lemah.\n"
            "3. Keanekaragaman Hayati Menurun: Berdampak negatif pada perikanan dan masyarakat.\n"
            "4. Indikasi Masalah: Bisa disebabkan oleh penebangan liar, polusi, atau konversi lahan."
        )

    # Paragraf yang menjelaskan asumsi kunci, ditambahkan untuk semua kondisi
    assumption_text = (
        "**Asumsi Kunci & Keterbatasan:**\n"
        f"Penting untuk dicatat bahwa estimasi kepadatan pohon sebesar {density_ha:.0f} pohon/ha adalah sebuah **asumsi** "
        "yang didasarkan pada metodologi di mana setiap titik sampel dianggap mewakili satu pohon dalam area seluas 10x10 meter "
        "(sesuai resolusi piksel citra). Angka ini bukan hasil pengukuran sensus lapangan langsung.\n"
        "Oleh karena itu, nilai total AGB dan AGC per hektar yang ditampilkan merupakan **estimasi kasar** yang sangat "
        "bergantung pada asumsi kepadatan ini. Untuk hasil yang lebih akurat, validasi lapangan diperlukan."
    )
    
    ideal_width_m = 100 
    area_m2 = area_ha * 10000
    protected_coastline_km = (area_m2 / ideal_width_m) / 1000 if ideal_width_m > 0 else 0

    protection_summary = (
        f"Dengan luas {area_ha:.2f} ha, area ini berpotensi melindungi garis pantai sepanjang {protected_coastline_km:.2f} km "
        f"jika diasumsikan membentuk sabuk mangrove dengan lebar efektif ({ideal_width_m} m)."
    )

    conclusion_header = f"=== KESIMPULAN ANALISIS (Status: {health_status}) ==="
    full_conclusion = (
        f"{conclusion_header}\n\n"
        f"**Interpretasi Umum:**\n{interpretation}\n\n"
        f"{assumption_text}\n\n"
        f"**Implikasi:**\n{implications}\n\n"
        f"**Potensi Perlindungan Pesisir:**\n{protection_summary}"
    )
    return full_conclusion
# === PERUBAHAN SELESAI ===

# Buat teks kesimpulan
conclusion_text = generate_conclusion(total_agb_ton_per_ha, tree_density_per_ha, estimated_area_ha)
print("\n" + "="*80)
print(conclusion_text)
print("="*80)

# === 12. SIMPAN RINGKASAN & KESIMPULAN KE FILE ===
stats_path = "static/statistik_prediksi_lengkap.txt"
with open(stats_path, "w", encoding="utf-8") as f:
    f.write(f"--- Ringkasan Statistik Prediksi Mangrove ({acq_date_label}) ---\n\n")
    f.write(f"Jumlah Titik Sampel Mangrove Valid : {num_samples}\n")
    f.write(f"Estimasi Luas Total Area Mangrove  : {estimated_area_ha:.2f} ha ({estimated_area_km2:.4f} km²)\n")
    f.write(f"Asumsi Kepadatan Pohon           : {tree_density_per_ha:.0f} pohon/ha (berdasarkan resolusi piksel)\n\n")
    f.write(f"Total Estimasi Aboveground Biomass (AGB) [KASAR] : {total_agb_ton_per_ha:,.2f} ton/ha\n")
    f.write(f"Total Estimasi Aboveground Carbon (AGC) [KASAR]  : {total_agc_ton_per_ha:,.2f} ton/ha\n")
    f.write("\n" + "="*60 + "\n\n")
    f.write(conclusion_text)

print(f"\nRingkasan statistik dan kesimpulan lengkap disimpan di: {stats_path}")