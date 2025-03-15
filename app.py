import streamlit as st
import time
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model dan tokenizer yang sudah disimpan
model_path = 'bert_indonesian_classifier_KBLI'
tokenizer_path = 'bert_indonesian_tokenizer_KBLI'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Setup device (GPU jika ada, kalau tidak pakai CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Misalkan label_encoder sudah didefinisikan (sesuaikan dengan kebutuhan)
# Contoh: label_encoder = {0: "Tidak Cocok", 1: "Cocok"}
label_encoder = {0: "Tidak Cocok", 1: "Cocok"}  # Ganti sesuai label aslimu

# Fungsi prediksi (disesuaikan dari kode asli)
def predict_r201b(r201, r202, model, tokenizer, label_encoder, device):
    # Tokenisasi input
    inputs = tokenizer(r201, r202, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Prediksi
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return label_encoder[prediction]

# Antarmuka Streamlit
st.title("Pencari KBLI")
st.write("Masukkan Rincian 201 dan Rincian 202 untuk memprediksi kecocokan KBLI.")

# Form input
with st.form(key="kbli_form"):
    r201 = st.text_input("Rincian 201", value="Pedagang ayam potong")
    r202 = st.text_input("Rincian 202", value="Ayam potong")
    submit_button = st.form_submit_button(label="Cari KBLI")

# Proses setelah tombol ditekan
if submit_button:
    if r201 and r202:
        with st.spinner("Memprediksi..."):
            # Hitung waktu inferensi
            start_time = time.time()
            prediction = predict_r201b(r201, r202, model, tokenizer, label_encoder, device)
            single_inference_time = time.time() - start_time
            
            # Tampilkan hasil
            st.success("Hasil Prediksi:")
            st.write(f"**Rincian 201:** {r201}")
            st.write(f"**Rincian 202:** {r202}")
            st.write(f"**Prediksi:** {prediction}")
            st.write(f"**Waktu Inferensi:** {single_inference_time:.6f} detik")
    else:
        st.warning("Harap isi kedua rincian sebelum mencari!")
