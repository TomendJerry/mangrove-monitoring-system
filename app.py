# app.py (Versi Modifikasi)

from flask import Flask, render_template, url_for, jsonify, request
import pandas as pd
import os
import subprocess
import threading # Import library threading

app = Flask(__name__)

# Variabel global untuk melacak status proses
job_status = {
    'pemetaan': {'status': 'idle', 'message': 'Belum ada proses yang dijalankan.'},
    'prediksi': {'status': 'idle', 'message': 'Belum ada proses yang dijalankan.'}
}

# Fungsi untuk menjalankan skrip di thread terpisah
def run_script(script_name, job_key):
    """
    Fungsi ini akan dijalankan di thread terpisah agar tidak memblokir server.
    """
    try:
        # Hapus file output lama agar hasil baru bisa dibuat
        if job_key == 'pemetaan':
            if os.path.exists("static/klasifikasi_titik_dengan_model.csv"):
                os.remove("static/klasifikasi_titik_dengan_model.csv")
        elif job_key == 'prediksi':
             if os.path.exists("static/output_prediksi_agb_agc.csv"):
                os.remove("static/output_prediksi_agb_agc.csv")

        # Set status menjadi 'running'
        job_status[job_key]['status'] = 'running'
        job_status[job_key]['message'] = f"Menjalankan {script_name}..."
        print(f"🚀 Memulai eksekusi {script_name} untuk job '{job_key}'...")

        # Jalankan skrip menggunakan subprocess
        # Ganti "Pemetaan.py" dengan "Pemetaan1.py" jika nama file Anda itu
        subprocess.run(["python", script_name], check=True, capture_output=True, text=True, encoding='utf-8')
        
        # Jika berhasil, set status menjadi 'completed'
        job_status[job_key]['status'] = 'completed'
        job_status[job_key]['message'] = f"Proses {script_name} berhasil diselesaikan."
        print(f"✅ Eksekusi {script_name} untuk job '{job_key}' berhasil.")

    except subprocess.CalledProcessError as e:
        # Jika ada error, catat dan set status menjadi 'error'
        job_status[job_key]['status'] = 'error'
        job_status[job_key]['message'] = f"Error saat menjalankan {script_name}: {e.stderr}"
        print(f"❌ ERROR saat eksekusi {script_name} untuk job '{job_key}': {e.stderr}")
    except Exception as e:
        job_status[job_key]['status'] = 'error'
        job_status[job_key]['message'] = f"Terjadi kesalahan tak terduga: {e}"
        print(f"❌ ERROR tak terduga untuk job '{job_key}': {e}")


@app.route("/")
def home():
    return render_template("home.html")

# --- ROUTE UNTUK MENJALANKAN PROSES ---

@app.route("/run/pemetaan", methods=['POST'])
def run_pemetaan():
    if job_status['pemetaan']['status'] == 'running':
        return jsonify({'status': 'error', 'message': 'Proses pemetaan sudah berjalan.'}), 400
    
    # Ganti nama file di sini jika perlu (Pemetaan1.py)
    thread = threading.Thread(target=run_script, args=("Pemetaan1.py", "pemetaan"))
    thread.start()
    return jsonify({'status': 'success', 'message': 'Proses pemetaan dimulai.'})

@app.route("/run/prediksi", methods=['POST'])
def run_prediksi():
    if not os.path.exists("static/klasifikasi_titik_dengan_model.csv"):
         return jsonify({'status': 'error', 'message': 'Jalankan proses pemetaan terlebih dahulu.'}), 400

    if job_status['prediksi']['status'] == 'running':
        return jsonify({'status': 'error', 'message': 'Proses prediksi sudah berjalan.'}), 400

    thread = threading.Thread(target=run_script, args=("Prediksijaya.py", "prediksi"))
    thread.start()
    return jsonify({'status': 'success', 'message': 'Proses prediksi dimulai.'})


# --- ROUTE UNTUK MENGECEK STATUS ---

@app.route("/status/<job_key>")
def get_status(job_key):
    if job_key not in job_status:
        return jsonify({'status': 'error', 'message': 'Job tidak ditemukan'}), 404

    # Salin status saat ini untuk dikirim ke browser
    current_job_state = job_status[job_key].copy()

    # JIKA statusnya sudah selesai (completed atau error), RESET status di server 
    # agar tidak menyebabkan loop refresh.
    if current_job_state['status'] in ['completed', 'error']:
        job_status[job_key]['status'] = 'idle'
        job_status[job_key]['message'] = 'Siap untuk proses berikutnya.'

    # Kirim status SEBELUM di-reset ke browser
    return jsonify(current_job_state)

# --- ROUTE UNTUK MENAMPILKAN HALAMAN HASIL ---

@app.route("/pemetaan")
def pemetaan():
    # Cek apakah file output sudah ada
    if not os.path.exists("static/klasifikasi_titik_dengan_model.csv"):
        # Jika belum ada, tampilkan halaman dengan pesan untuk menjalankan proses
        return render_template("pemetaan.html", data_available=False)

    # Jika file ada, baca dan tampilkan seperti biasa
    df = pd.read_csv("static/klasifikasi_titik_dengan_model.csv")
    columns = df.columns.tolist()
    data = df.head(100).to_dict(orient="records")
    with open("static/statistik_pemetaan.txt", "r", encoding="utf-8") as f:
        statistik = f.read()
    
    return render_template("pemetaan.html",
                           data_available=True,
                           data=data,
                           columns=columns,
                           peta=url_for('static', filename='peta_interaktif_klasifikasi.html'),
                           statistik=statistik)

@app.route("/prediksi")
def prediksi():
    if not os.path.exists("static/output_prediksi_agb_agc.csv"):
        return render_template("prediksi.html", data_available=False)
    
    df = pd.read_csv("static/output_prediksi_agb_agc.csv")
    columns = df.columns.tolist()
    data = df.head(100).to_dict(orient="records")
    with open("static/statistik_prediksi_lengkap.txt", "r", encoding="utf-8") as f:
        statistik = f.read()

    return render_template("prediksi.html",
                           data_available=True,
                           data=data,
                           columns=columns,
                           peta=url_for('static', filename='peta_prediksi_agb_agc_mangrove.html'),
                           statistik=statistik)


if __name__ == "__main__":
    print("="*50)
    print("🌐 Menjalankan server Flask. Buka http://127.0.0.1:5000 di browser Anda.")
    print("   Skrip analisis kini dijalankan melalui tombol di halaman web.")
    print("="*50)
    # Hapus logika pemanggilan skrip dari sini
    app.run(debug=True)