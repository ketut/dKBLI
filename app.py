import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import joblib
import numpy as np

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model dan tokenizer dari Hugging Face
tokenizer = AutoTokenizer.from_pretrained("ketut/dKBLI")
model = AutoModelForSequenceClassification.from_pretrained("ketut/dKBLI")
model.to(device)

# Coba memuat label_encoder (jika ada file lokal)
try:
    label_encoder = joblib.load("label_encoder.pkl")
    #st.write(f"Jumlah label dari label_encoder: {len(label_encoder.classes_)}")
except FileNotFoundError:
    # Jika label_encoder.pkl tidak ada, definisikan manual (sesuaikan dengan R201B)
    st.warning("File label_encoder.pkl tidak ditemukan. Menggunakan contoh sementara.")
    kbli_codes = ["47771", "47772", "47773"]  # GANTI DENGAN DAFTAR KODE KBLI ASLI
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(kbli_codes)

# Cek jumlah label dari model untuk verifikasi
#st.write(f"Jumlah label dari model: {model.config.num_labels}")

# Fungsi prediksi dengan persentase keyakinan
def predict_r201b(text_r201, text_r202, model, tokenizer, label_encoder, device):
    combined_text = f"{text_r201} {text_r202}"
    inputs = tokenizer(combined_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # Move to CPU and convert to numpy
    predicted_class = np.argmax(probabilities)  # Get the index of the highest probability
    confidence = probabilities[predicted_class] * 100  # Convert to percentage
    # Debugging
    st.write(f"Indeks prediksi: {predicted_class}")
    try:
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        return predicted_label, confidence
    except ValueError:
        return f"Error: Indeks {predicted_class} tidak ada di label_encoder", 0.0

# Antarmuka Streamlit
st.title("Pencari Kode KBLI - KBLI 2015")
st.write("Masukkan Rincian 201 dan Rincian 202 untuk mendapatkan kode KBLI.")

# Form input
with st.form(key="kbli_form"):
    r201 = st.text_input("Rincian 201 - Tuliskan secara lengkap jenis kegiatan utama (meliputi proses dan output)", value="Menjual Canang sari")
    r202 = st.text_input("Rincian 202 - Produk utama (barang atau jasa) yang dihasilkan/dijual", value="Canang sari")
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
            # st.write(f"DEBUG -- length of KBLI: {len(prediction)}")
            prediction = str(prediction)
            if len(prediction) == 4:
                prediction = '0'+prediction
            else:
                pass
            st.write(f"**Kode KBLI:** {prediction}")
            st.write(f"**Keyakinan Model:** {confidence:.2f}%")
            st.write(f"**Waktu Inferensi:** {inference_time:.6f} detik")
    else:
        st.warning("Harap isi kedua rincian sebelum mencari!")
