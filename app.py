#tab2
import streamlit as st
import pandas as pd
#from shapely.geometry import Point
from shapely import wkt

# tab1
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


# Set page config
st.set_page_config(page_title="cAriKBLI", page_icon="<< 🔍 >>")

# tab
tab1, tab2 = st.tabs(["🔍 cAriKBLI", "🗺️ CariBanjar"])

# tab pak tut
with tab1:
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model dan tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ketut/IndoBERTkbli")
    model = AutoModelForSequenceClassification.from_pretrained("ketut/IndoBERTkbli")
    model.to(device)

    # Load label_encoder
    try:
        label_encoder = joblib.load("label_encoder_base_p2_augmented.pkl")
        if not isinstance(label_encoder, LabelEncoder):
            raise ValueError("File tidak berisi LabelEncoder.")
    except FileNotFoundError:
        st.warning("File label_encoder tidak ditemukan. Menggunakan kode dummy.")
        kbli_codes = ["47771", "47772", "47773"]
        label_encoder = LabelEncoder()
        label_encoder.fit(kbli_codes)
    except Exception as e:
        st.error(f"Error loading label_encoder: {e}")
        st.warning("Menggunakan kode dummy.")
        kbli_codes = ["47771", "47772", "47773"]
        label_encoder = LabelEncoder()
        label_encoder.fit(kbli_codes)

    # Load konsep_kbli.csv
    try:
        kbli_df = pd.read_csv("konsep_kbli.csv")
        kbli_df["kode_kbli"] = kbli_df["kode_kbli"].astype(str)
    except FileNotFoundError:
        st.error("File konsep_kbli.csv tidak ditemukan.")
        kbli_df = pd.DataFrame(columns=["kode_kbli", "deskripsi"])

    # Fungsi prediksi top-3
    def predict_r201b(text_r201, text_r202, model, tokenizer, label_encoder, device):
        combined_text = f"{text_r201} {text_r202}"
        inputs = tokenizer(combined_text, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        top3_preds = []
        for idx in top3_indices:
            try:
                label = label_encoder.inverse_transform([idx])[0]
            except ValueError:
                label = f"Error: Indeks {idx} tidak ada di label_encoder"
            confidence = probabilities[idx] * 100
            top3_preds.append((str(label), confidence))
        return top3_preds

    # Antarmuka Streamlit
    st.image("cariKBLI.png", width=120)
    st.write("Masukkan Rincian Kegiatan Utama dan Produk Utama untuk mendapatkan kode KBLI.")
    st.write("nb: Masih KBLI 2015, KBLI2020 coming soon ya")

    # Form input
    with st.form(key="kbli_form"):
        r201 = st.text_input("Tuliskan secara lengkap jenis kegiatan utama (meliputi proses dan output)", value="Menjual pisang goreng")
        r202 = st.text_input("Produk utama (barang atau jasa) yang dihasilkan/dijual", value="Pisang goreng")
        submit_button = st.form_submit_button(label="Cari Kode KBLI")

    # Proses prediksi
    if submit_button:
        if r201 and r202:
            with st.spinner("Memprediksi kode KBLI..."):
                start_time = time.time()
                top3_predictions = predict_r201b(r201, r202, model, tokenizer, label_encoder, device)
                inference_time = time.time() - start_time

                st.success("Hasil Prediksi:")
                st.write(f"**Kegiatan Utama:** {r201}")
                st.write(f"**Produk Utama:** {r202}")
                for i, (prediction, confidence) in enumerate(top3_predictions, start=1):
                    if len(prediction) == 4:
                        prediction_display = '0' + prediction
                    else:
                        prediction_display = prediction
                    st.markdown(f"### 🔹 Top-{i} Prediksi:")
                    st.write(f"**Kode KBLI:** {prediction_display}")
                    deskripsi = kbli_df[kbli_df["kode_kbli"] == prediction_display]["deskripsi"].values
                    if len(deskripsi) > 0:
                        st.write(f"**Deskripsi:** {deskripsi[0]}")
                    else:
                        st.write("**Deskripsi:** Tidak ditemukan untuk kode ini.")
                    st.write(f"**Keyakinan Model:** {confidence:.2f}%")
                st.write(f"**Waktu Inferensi:** {inference_time:.6f} detik")
        else:
            st.warning("Harap isi kedua rincian sebelum mencari!")

# tab jimm
with tab2:
    st.title("Banjar apa ini ya?")
    coord = st.text_input("Input koordinat, lalu press enter" , placeholder="-8.12345, 115.12345")

    # df berasal dari kdsls.geojson, di export as shapefile, trs diexport lagi as csv dengan geometry option pake as_wkt, dapetla multipolygon
    df = pd.read_csv(r"csv_kdsls25.csv")
    df['polygon'] = df['WKT'].apply(wkt.loads)
    st.markdown("Dataframe menggunakan data :orange[Kabupaten Badung semester I 2024]")

    # function search
    def search(coordinate):
        for iter in range(0,len(df)):
            iswithin = coordinate.within(df['polygon'][iter])  
            #print(iswithin)  
            if iswithin == True:
                #print("dalem"+ str(iswithin)+ str(iter))
                return [False,iter]
                #break
        #print("luar"+ str(iswithin)+ str(iter))
        return [True,0]

    # read coordinate
    if coord:
        #coordinate = "-8.586727348573866, 115.21251283915583"
        coorsplit = coord.split(",")
        coorfin = wkt.loads( f"Point({coorsplit[1]} {coorsplit[0]})" )

        #for i in range(0,len(df)):
        #    iswithin = coorfin.within(df['polygon'][i])  
        #    #print(iswithin)  
        #    if iswithin == True:
        #        break
        #    notfound = True
        #print(df['nmsls'][i])
        
        # search
        res = search(coordinate=coorfin)
        notfound = res[0]
        i = res[1]

        # show result
        if notfound == False:
            st.success(f"Ketemu itu adalah :orange[{df.iloc[i,]['nmsls']}], Desa {df.iloc[i,]['nmdesa']}, Kec. {df.iloc[i,]['nmkec']}, Kab. {df.iloc[i,]['nmkab']} [{df.iloc[i,]['idsls']}]")
            st.map(pd.DataFrame({'lat':[float(coorsplit[0])]  , 'lon': [float(coorsplit[1])]   }),zoom=15, size=17)
        else:
            st.warning("Maaf di luar Kabupaten Badung")
