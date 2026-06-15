import streamlit as st
import pandas as pd
import torch
import re
import emoji
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time


# =====================================
# KONFIGURASI HALAMAN
# =====================================

st.set_page_config(
    page_title="Shopee User Insight Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================
# CSS STYLING
# =====================================

st.markdown(
    """
<style>

[data-testid="stSidebar"]{
background:#1E293B;
}

[data-testid="stSidebar"] *{
color:white;
}

.metric-card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0 4px 10px rgba(0,0,0,0.1);
}

.big-font{
font-size:28px;
font-weight:700;
color:#EE4D2D;
}

</style>

""",
    unsafe_allow_html=True,
)

# =====================================
# LOAD MODEL
# =====================================

MODEL_NAME = "nabilaeky/shopee-sentiment-indobert"

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        subfolder="model_indobert"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        subfolder="model_indobert"
    )

    return tokenizer, model

tokenizer, model = load_model()

# =====================================
# PREPROCESSING
# =====================================


def clean_text(text):

    text = str(text)

    text = text.lower()

    text = emoji.replace_emoji(text, replace="")

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================
# PREDIKSI SENTIMEN
# =====================================
def predict_sentiment(text):

    text = clean_text(text)

    if text == "":
        return "Negatif", 0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()

    st.write("Probabilities:", probabilities.tolist())
    st.write("Prediction:", prediction)

    confidence = torch.max(probabilities).item()

    label_map = {
        0: "Negatif",
        1: "Positif"
    }

    hasil = label_map.get(prediction)

    return hasil, confidence

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🛒 Shopee Dashboard")

menu = st.sidebar.radio(
    "Menu",
    [
        "Executive Dashboard",
        "Analisis Manual",
        "Analisis Dataset",
    ],
)


# =====================================
# DASHBOARD
# =====================================

if menu == "Executive Dashboard":

    st.title("🛒 Shopee User Insight Dashboard")

    st.caption("AI Based Customer Sentiment Monitoring")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🤖 Model", "IndoBERT")

    col2.metric("📱 Sumber", "Google Play")

    col3.metric("⚡ Framework", "Streamlit")

    col4.metric("🧠 AI", "Aktif")

# =====================================
# ANALISIS MANUAL
# =====================================

elif menu == "Analisis Manual":

    st.title("✍️ Analisis Sentimen")

    review = st.text_area("Masukkan ulasan")

    if st.button("Analisis"):

        hasil, score = predict_sentiment(review)

        if hasil == "Positif":

            st.success(f"😊 {hasil}")

        elif hasil == "Negatif":

            st.error(f"☹️ {hasil}")

        else:

            st.warning(hasil)

        st.progress(score)

        st.write(f"Confidence : {score*100:.2f}%")
# =====================================
# ANALISIS DATASET
# =====================================

elif menu == "Analisis Dataset":

    st.title("📂 Analisis Dataset")

    uploaded_file = st.file_uploader(
        "Upload file CSV atau Excel",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        file_name = uploaded_file.name.lower()

        # CSV
        if file_name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        # Excel
        elif file_name.endswith((".xlsx", ".xls")):

            excel_file = pd.ExcelFile(uploaded_file)

            sheet = st.selectbox(
                "Pilih Sheet",
                excel_file.sheet_names
            )

            df = pd.read_excel(
                excel_file,
                sheet_name=sheet
            )

        else:

            st.error("Format file tidak didukung")
            st.stop()

        st.subheader("📋 Preview Dataset")
        st.dataframe(df.head())
        # =====================
        # CEK KOLOM REVIEW
        # =====================

        review_column = None

        possible_columns = ["review", "content", "review_text"]

        for col in possible_columns:
            if col in df.columns:
                review_column = col
                break

        if review_column is None:
            st.error("❌ Kolom review tidak ditemukan")

        else:

            # hapus data kosong
            df = df.dropna(subset=[review_column])

            df = df[
                df[review_column]
                .astype(str)
                .str.strip()
                != ""
            ]

            start_time = time.time()

            progress_bar = st.progress(0)
            status_text = st.empty()

            hasil_list = []
            confidence_list = []

            total_data = len(df)

            for i, text in enumerate(df[review_column]):

                hasil, score = predict_sentiment(text)

                hasil_list.append(hasil)
                confidence_list.append(
                    round(score * 100, 2)
                )

                progress = int(
                    ((i + 1) / total_data) * 100
                )

                elapsed = time.time() - start_time

                rata_rata = elapsed / (i + 1)

                sisa = rata_rata * (
                    total_data - (i + 1)
                )

                progress_bar.progress(progress)

                status_text.info(
                    f"⏳ Diproses {i+1}/{total_data} data | "
                    f"Progress {progress}% | "
                    f"Estimasi sisa {sisa:.1f} detik"
                )

            df["hasil"] = hasil_list
            df["confidence"] = confidence_list

            total_time = time.time() - start_time

            progress_bar.progress(100)

            status_text.success(
                f"✅ Analisis selesai dalam {total_time:.2f} detik"
            )

            speed = total_data / total_time

            st.metric(
                "⚡ Kecepatan Analisis",
                f"{speed:.2f} review/detik"
            )

            # =====================
            # TABEL HASIL
            # =====================

            st.subheader("📊 Hasil Analisis")

            st.dataframe(df)

        # =====================
        # TABEL HASIL
        # =====================

        st.subheader("📊 Hasil Analisis")

        st.dataframe(df)

            # =====================
            # METRIK
            # =====================

        total = len(df)
        positif = len(df[df["hasil"] == "Positif"])
        negatif = len(df[df["hasil"] == "Negatif"])
        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Total Review", total)
        col2.metric("😊 Positif", positif)
        col3.metric("☹️ Negatif", negatif)

            # =====================
            # VISUALISASI
            # =====================

        sentiment_count = (
    df[df["hasil"].isin(["Positif", "Negatif"])]
    ["hasil"]
    .value_counts()
    .reset_index()
)
        sentiment_count.columns = ["Sentiment", "Count"]

        st.subheader("📈 Visualisasi")

        col1, col2 = st.columns(2)

        with col1:

                fig_bar = px.bar(
                    sentiment_count,
                    x="Sentiment",
                    y="Count",
                    title="Distribusi Sentimen",
                )

                st.plotly_chart(fig_bar, use_container_width=True)

        with col2:

                fig_pie = px.pie(
                    sentiment_count,
                    names="Sentiment",
                    values="Count",
                    title="Persentase Sentimen",
                )

                st.plotly_chart(fig_pie, use_container_width=True)

            # =====================
            # WORD CLOUD
            # =====================

        st.subheader("☁️ Word Cloud")

        text = " ".join(df[review_column].astype(str))

        wc = WordCloud(width=1200, height=500, background_color="white").generate(
                text
            )

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.imshow(wc)

        ax.axis("off")

        st.pyplot(fig)

            # =====================
            # SEARCH REVIEW
            # =====================

        st.subheader("🔍 Cari Review")

        keyword = st.text_input("Masukkan kata kunci")

        if keyword:

                hasil_filter = df[
                    df[review_column]
                    .astype(str)
                    .str.contains(keyword, case=False, na=False)
                ]

                st.dataframe(hasil_filter)

            # =====================
            # DOWNLOAD
            # =====================

        csv = df.to_csv(index=False)

        st.download_button(
                label="📥 Download Hasil",
                data=csv,
                file_name="hasil_sentimen.csv",
                mime="text/csv",
            )
