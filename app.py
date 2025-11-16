import os
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import traceback

# ==============================
# 1. Konfigurasi Path & Loader
# ==============================
BASE_DIR = Path(__file__).parent.absolute()
MODEL_PATH = BASE_DIR / "models" / "golf_model.pkl"
DATA_PATH = BASE_DIR / "data" / "golf_dataset_long_format_with_text.csv"

@st.cache_resource
def load_model():
    """Load model dengan error handling"""
    try:
        if not MODEL_PATH.exists():
            st.error(f"❌ Model tidak ditemukan di: {MODEL_PATH}")
            st.stop()
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error saat loading model: {str(e)}")
        st.stop()

@st.cache_data
def load_data():
    """Load dataset dengan error handling"""
    try:
        if not DATA_PATH.exists():
            st.error(f"❌ Dataset tidak ditemukan di: {DATA_PATH}")
            st.stop()
        df = pd.read_csv(DATA_PATH)
        if df.empty:
            st.error("❌ Dataset kosong!")
            st.stop()
        return df
    except Exception as e:
        st.error(f"❌ Error saat loading data: {str(e)}")
        st.stop()

# ==============================
# 2. Inisialisasi Aplikasi
# ==============================
st.set_page_config(
    page_title="Prediksi Golf Player",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources
model = load_model()
df = load_data()

# ==============================
# 3. Header Aplikasi
# ==============================
st.title("⛳ Analisis Prediktif Perilaku Pemain Golf")
st.markdown("""
Aplikasi ini memprediksi probabilitas seorang pemain akan **bermain golf** pada hari tertentu 
berdasarkan faktor cuaca dan kondisi operasional lapangan.
""")

# Info dataset
with st.expander("ℹ️ Info Dataset"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", len(df))
    with col2:
        st.metric("Jumlah Fitur", len(df.columns))
    with col3:
        if "Play" in df.columns:
            play_rate = (df["Play"].sum() / len(df) * 100)
            st.metric("% Pemain Aktif", f"{play_rate:.1f}%")

st.sidebar.header("🎯 Input Kondisi Hari Ini")

# ==============================
# 4. Fungsi Helper
# ==============================
def get_options(col):
    """Ambil pilihan unik dari kolom dengan handling None"""
    try:
        options = df[col].dropna().unique().tolist()
        return sorted([str(x) for x in options])
    except KeyError:
        st.warning(f"⚠️ Kolom '{col}' tidak ditemukan dalam dataset")
        return ["N/A"]

def get_numeric_range(col):
    """Ambil min, max, dan median dari kolom numerik"""
    try:
        return (
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].median())
        )
    except KeyError:
        st.warning(f"⚠️ Kolom '{col}' tidak ditemukan")
        return (0.0, 100.0, 50.0)

# ==============================
# 5. Input Categorical
# ==============================
st.sidebar.subheader("📅 Informasi Waktu")
weekday = st.sidebar.selectbox("Hari", get_options("Weekday"), help="Pilih hari dalam seminggu")
holiday = st.sidebar.selectbox("Hari Libur", get_options("Holiday"), help="Apakah hari libur?")
month = st.sidebar.selectbox("Bulan", get_options("Month"), help="Pilih bulan")
season = st.sidebar.selectbox("Musim", get_options("Season"), help="Pilih musim")

st.sidebar.subheader("🌤️ Kondisi Cuaca")
outlook = st.sidebar.selectbox("Cuaca", get_options("Outlook"), help="Kondisi cuaca umum")
windy = st.sidebar.selectbox("Angin", get_options("Windy"), help="Apakah berangin?")

temp_min, temp_max, temp_med = get_numeric_range("Temperature")
temperature = st.sidebar.slider(
    "Suhu (°C)", 
    min_value=temp_min, 
    max_value=temp_max,
    value=temp_med,
    help="Suhu udara dalam derajat Celsius"
)

hum_min, hum_max, hum_med = get_numeric_range("Humidity")
humidity = st.sidebar.slider(
    "Kelembaban (%)", 
    min_value=hum_min, 
    max_value=hum_max,
    value=hum_med,
    help="Kelembaban udara dalam persen"
)

st.sidebar.subheader("🏌️ Kondisi Lapangan")
crowd_min, crowd_max, crowd_med = get_numeric_range("Crowdedness")
crowdedness = st.sidebar.slider(
    "Kepadatan Lapangan", 
    min_value=crowd_min, 
    max_value=crowd_max,
    value=crowd_med,
    help="Tingkat kepadatan lapangan golf"
)

st.sidebar.subheader("📧 Marketing")
email_campaign = st.sidebar.selectbox(
    "Kampanye Email", 
    get_options("EmailCampaign"),
    help="Apakah ada kampanye email aktif?"
)

st.sidebar.subheader("👤 Identitas Pemain")
default_id = str(df["ID"].iloc[0]) if "ID" in df.columns and not df.empty else "Player_001"
player_id = st.sidebar.text_input(
    "Player ID", 
    value=default_id,
    help="Masukkan ID pemain"
)

# ==============================
# 6. Bentuk Input DataFrame
# ==============================
input_dict = {
    "Weekday": weekday,
    "Holiday": holiday,
    "Month": month,
    "Season": season,
    "Temperature": float(temperature),
    "Humidity": float(humidity),
    "Windy": windy,
    "Outlook": outlook,
    "Crowdedness": float(crowdedness),
    "EmailCampaign": email_campaign,
    "ID": player_id,
}

input_df = pd.DataFrame([input_dict])

# ==============================
# 7. Tampilan Input
# ==============================
st.subheader("📋 Ringkasan Input")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Kondisi Waktu & Cuaca:**")
    st.dataframe(
        input_df[["Weekday", "Holiday", "Month", "Season", "Outlook", "Windy"]],
        use_container_width=True
    )

with col2:
    st.markdown("**Kondisi Numerik & Operasional:**")
    st.dataframe(
        input_df[["Temperature", "Humidity", "Crowdedness", "EmailCampaign", "ID"]],
        use_container_width=True
    )

# ==============================
# 8. Prediksi
# ==============================
st.markdown("---")
predict_btn = st.button("🎯 Prediksi Peluang Bermain", type="primary", use_container_width=True)

if predict_btn:
    try:
        with st.spinner("Melakukan prediksi..."):
            # Prediksi probabilitas
            prob = model.predict_proba(input_df)[0, 1]
            label = int(model.predict(input_df)[0])
            
            # Tampilan hasil
            st.subheader("📊 Hasil Prediksi")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Probabilitas Bermain", 
                    f"{prob:.1%}",
                    delta=f"{prob - 0.5:.1%}" if prob > 0.5 else f"{prob - 0.5:.1%}"
                )
            
            with col2:
                confidence = abs(prob - 0.5) * 200  # Confidence score
                st.metric("Confidence", f"{confidence:.1f}%")
            
            with col3:
                status = "BERMAIN" if label == 1 else "TIDAK BERMAIN"
                st.metric("Keputusan", status)
            
            # Progress bar
            st.progress(prob)
            
            # Interpretasi
            if label == 1:
                if prob >= 0.8:
                    st.success("✅ **Prediksi: SANGAT MUNGKIN BERMAIN**")
                    st.info("💡 Kondisi sangat mendukung untuk bermain golf!")
                elif prob >= 0.6:
                    st.success("✅ **Prediksi: KEMUNGKINAN BERMAIN**")
                    st.info("💡 Kondisi cukup baik untuk bermain golf.")
                else:
                    st.warning("⚠️ **Prediksi: MUNGKIN BERMAIN**")
                    st.info("💡 Kondisi marginal, pertimbangkan faktor lain.")
            else:
                if prob <= 0.2:
                    st.error("❌ **Prediksi: SANGAT TIDAK MUNGKIN BERMAIN**")
                    st.info("💡 Kondisi tidak mendukung untuk bermain golf.")
                elif prob <= 0.4:
                    st.warning("⚠️ **Prediksi: KEMUNGKINAN TIDAK BERMAIN**")
                    st.info("💡 Kondisi kurang ideal untuk bermain golf.")
                else:
                    st.warning("⚠️ **Prediksi: MUNGKIN TIDAK BERMAIN**")
                    st.info("💡 Kondisi marginal, keputusan sulit diprediksi.")
            
            
    except Exception as e:
        st.error(f"❌ Error saat melakukan prediksi: {str(e)}")
        with st.expander("Detail Error"):
            st.code(traceback.format_exc())

# ==============================
# 9. Footer & Info
# ==============================
st.markdown("---")
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    **Model:** Random Forest Classifier
    
    **Fitur yang digunakan:**
    - Faktor waktu: Hari, Bulan, Musim, Hari Libur
    - Faktor cuaca: Suhu, Kelembaban, Angin, Kondisi Cuaca
    - Faktor operasional: Kepadatan Lapangan, Kampanye Email
    - Identitas: Player ID
    
    **Preprocessing:**
    - Imputasi nilai kosong
    - One-hot encoding untuk variabel kategorikal
    - Feature scaling untuk variabel numerik
    
    **Catatan:** Model dilatih menggunakan data historis perilaku pemain golf.
    Akurasi prediksi dapat bervariasi tergantung kualitas data input.
    """)

st.caption("© 2024 Golf Prediction System | Powered by Streamlit & scikit-learn")