import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model dan tokenizer langsung dari Hugging Face
tokenizer = AutoTokenizer.from_pretrained("ketut/dKBLI")
model = AutoModelForSequenceClassification.from_pretrained("ketut/dKBLI")

# Setup device (GPU jika ada, kalau tidak CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fungsi prediksi
def predict_kbli(r201, r202, model, tokenizer, device):
    # Tokenisasi input
    inputs = tokenizer(r201, r202, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Prediksi
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # Misalkan label (sesuaikan dengan modelmu)
    label_encoder = {0: "Tidak Cocok", 1: "Cocok"}  # Ganti sesuai label asli
    return label_encoder[prediction]

# Antarmuka Streamlit
st.title("Pencari KBLI dengan Model Hugging Face")
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
            prediction = predict_kbli(r201, r202, model, tokenizer, device)
            inference_time = time.time() - start_time
            
            # Tampilkan hasil
            st.success("Hasil Prediksi:")
            st.write(f"**Rincian 201:** {r201}")
            st.write(f"**Rincian 202:** {r202}")
            st.write(f"**Prediksi:** {prediction}")
            st.write(f"**Waktu Inferensi:** {inference_time:.6f} detik")
    else:
        st.warning("Harap isi kedua rincian sebelum mencari!")
