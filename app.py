import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

# Load model dan tokenizer dari Hugging Face
tokenizer = AutoTokenizer.from_pretrained("ketut/dKBLI")
model = AutoModelForSequenceClassification.from_pretrained("ketut/dKBLI")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Cek jumlah label dari model
num_labels = model.config.num_labels
st.write(f"Jumlah label dari model: {num_labels}")  # Debugging

# Fungsi prediksi
def predict_kbli(r201, r202, model, tokenizer, device):
    inputs = tokenizer(r201, r202, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # Debugging: Tampilkan nilai prediksi
    st.write(f"Nilai prediksi dari model: {prediction}")
    
    # Sesuaikan label_encoder dengan jumlah label model
    label_encoder = {i: f"Label {i}" for i in range(num_labels)}  # Default sementara
    # Ganti dengan label asli dari model ketut/dKBLI jika diketahui
    # Contoh: label_encoder = {0: "Tidak Cocok", 1: "Cocok", 2: "Lainnya"}
    
    try:
        return label_encoder[prediction]
    except KeyError:
        return f"Prediksi tidak valid: {prediction} (label tidak ditemukan)"

# Antarmuka Streamlit
st.title("Pencari KBLI")
st.write("Masukkan Rincian 201 dan Rincian 202.")

with st.form(key="kbli_form"):
    r201 = st.text_input("Rincian 201", value="Pedagang ayam potong")
    r202 = st.text_input("Rincian 202", value="Ayam potong")
    submit_button = st.form_submit_button(label="Cari KBLI")

if submit_button:
    if r201 and r202:
        with st.spinner("Memprediksi..."):
            start_time = time.time()
            prediction = predict_kbli(r201, r202, model, tokenizer, device)
            inference_time = time.time() - start_time
            st.success("Hasil Prediksi:")
            st.write(f"**Rincian 201:** {r201}")
            st.write(f"**Rincian 202:** {r202}")
            st.write(f"**Prediksi:** {prediction}")
            st.write(f"**Waktu Inferensi:** {inference_time:.6f} detik")
    else:
        st.warning("Harap isi kedua rincian!")
