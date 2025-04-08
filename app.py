import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Set page config sebagai perintah pertama
st.set_page_config(page_title="cAriKBLI", page_icon="<< 🔍 >>")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model dan tokenizer dari Hugging Face
tokenizer = AutoTokenizer.from_pretrained("ketut/IndoBERTkbli")
model = AutoModelForSequenceClassification.from_pretrained("ketut/IndoBERTkbli")
# tokenizer = AutoTokenizer.from_pretrained("ketut/dKBLI", trust_remote_code=True)
# model = AutoModelForSequenceClassification.from_pretrained("ketut/dKBLI")
model.to(device)

# Coba memuat label_encoder dari file .pth
try:
    # Gunakan weights_only=False untuk memuat objek non-tensor (dengan peringatan keamanan)
    # label_encoder = torch.load("label_encoder.pth", map_location=device, weights_only=False)
    label_encoder = joblib.load("label_encoder_base_p2_augmented.pkl")
    # Verifikasi apakah itu LabelEncoder
    if not isinstance(label_encoder, LabelEncoder):
        raise ValueError("File label_encoder.pth tidak berisi objek LabelEncoder yang valid.")
except FileNotFoundError:
    st.warning("File label_encoder.pth tidak ditemukan. Menggunakan contoh sementara.")
    kbli_codes = ["47771", "47772", "47773"]  # GANTI DENGAN DAFTAR KODE KBLI ASLI
    label_encoder = LabelEncoder()
    label_encoder.fit(kbli_codes)
except Exception as e:
    st.error(f"Gagal memuat label_encoder.pth: {e}")
    st.warning("Menggunakan contoh sementara karena error.")
    kbli_codes = ["47771", "47772", "47773"]  # GANTI DENGAN DAFTAR KODE KBLI ASLI
    label_encoder = LabelEncoder()
    label_encoder.fit(kbli_codes)

# Load file konsep_kbli.csv
try:
    kbli_df = pd.read_csv("konsep_kbli.csv")
    kbli_df["kode_kbli"] = kbli_df["kode_kbli"].astype(str)
except FileNotFoundError:
    st.error("File konsep_kbli.csv tidak ditemukan. Harap sediakan file tersebut.")
    kbli_df = pd.DataFrame(columns=["kode_kbli", "deskripsi"])

# Fungsi prediksi dengan persentase keyakinan
def predict_r201b(text_r201, text_r202, model, tokenizer, label_encoder, device):
    combined_text = f"{text_r201} {text_r202}"
    inputs = tokenizer(combined_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class] * 100
    st.write(f"Indeks prediksi: {predicted_class}")
    try:
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        return str(predicted_label), confidence
    except ValueError:
        return f"Error: Indeks {predicted_class} tidak ada di label_encoder", 0.0

# Antarmuka Streamlit
st.image("cariKBLI.png", width=120)
st.write("Masukkan Rincian 201 dan Rincian 202 untuk mendapatkan kode KBLI.")

# Form input
with st.form(key="kbli_form"):
    r201 = st.text_input("Rincian 201 - Tuliskan secara lengkap jenis kegiatan utama (meliputi proses dan output)", value="Menjual pisang goreng")
    r202 = st.text_input("Rincian 202 - Produk utama (barang atau jasa) yang dihasilkan/dijual", value="Pisang goreng")
    submit_button = st.form_submit_button(label="Cari Kode KBLI")

# Proses setelah tombol ditekan
if submit_button:
    if r201 and r202:
        with st.spinner("Memprediksi kode KBLI..."):
            start_time = time.time()
            prediction, confidence = predict_r201b(r201, r202, model, tokenizer, label_encoder, device)
            inference_time = time.time() - start_time
            st.success("Hasil Prediksi:")
            st.write(f"**Rincian 201:** {r201}")
            st.write(f"**Rincian 202:** {r202}")
            deskripsi = kbli_df[kbli_df["kode_kbli"] == prediction]["deskripsi"].values
            if len(prediction) == 4:
                prediction = '0' + prediction
            else:
                prediction = prediction
            st.write(f"**Kode KBLI:** {prediction}")
            if len(deskripsi) > 0:
                st.write(f"**Deskripsi:** {deskripsi[0]}")
            else:
                st.write("**Deskripsi:** Tidak ditemukan deskripsi untuk kode KBLI ini.")
            st.write(f"**Keyakinan Model:** {confidence:.2f}%")
            st.write(f"**Waktu Inferensi:** {inference_time:.6f} detik")
    else:
        st.warning("Harap isi kedua rincian sebelum mencari!")
