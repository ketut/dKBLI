import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
from sklearn.preprocessing import LabelEncoder

# Load model dan tokenizer dari Hugging Face
tokenizer = AutoTokenizer.from_pretrained("ketut/dKBLI")
model = AutoModelForSequenceClassification.from_pretrained("ketut/dKBLI")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Cek jumlah label dari model
num_labels = model.config.num_labels
st.write(f"Jumlah label yang didukung model: {num_labels}")  # Debugging

# Inisialisasi label_encoder (sesuaikan dengan data pelatihan asli)
kbli_codes = ["47771", "47772", "47773"]  # GANTI DENGAN DAFTAR KODE KBLI ASLI
label_encoder = LabelEncoder()
label_encoder.fit(kbli_codes)

# Fungsi prediksi dengan logika KBLI
def predict_r201b(text_r201, text_r202, model, tokenizer, label_encoder, device):
    combined_text = f"{text_r201} {text_r202}"
    inputs = tokenizer(combined_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Debugging: Tampilkan predicted_class
    st.write(f"Indeks prediksi dari model: {predicted_class}")
    
    # Cek apakah predicted_class valid
    if predicted_class >= len(label_encoder.classes_):
        return f"Error: Indeks {predicted_class} di luar jangkauan label (max {len(label_encoder.classes_) - 1})"
    return label_encoder.inverse_transform([predicted_class])[0]

# Antarmuka Streamlit
st.title("Pencari Kode KBLI")
st.write("Masukkan Rincian 201 dan Rincian 202 untuk mendapatkan kode KBLI.")

with st.form(key="kbli_form"):
    r201 = st.text_input("Rincian 201", value="Menjual Canang sari")
    r202 = st.text_input("Rincian 202", value="Canang sari")
    submit_button = st.form_submit_button(label="Cari Kode KBLI")

if submit_button:
    if r201 and r202:
        with st.spinner("Memprediksi kode KBLI..."):
            start_time = time.time()
            prediction = predict_r201b(r201, r202, model, tokenizer, label_encoder, device)
            inference_time = time.time() - start_time
            st.success("Hasil Prediksi:")
            st.write(f"**Rincian 201:** {r201}")
            st.write(f"**Rincian 202:** {r202}")
            st.write(f"**Kode KBLI:** {prediction}")
            st.write(f"**Waktu Inferensi:** {inference_time:.6f} detik")
    else:
        st.warning("Harap isi kedua rincian sebelum mencari!")
