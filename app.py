import streamlit as st
import pandas as pd
from joblib import load
import os

# --- 1. Konfigurasi dan Pemuatan Model/Encoders ---

# Pemuatan Model dan Encoders (menggunakan cache untuk performa)
@st.cache_resource
def load_assets():
    """Memuat model dan label encoders yang disimpan menggunakan joblib."""
    # List nama file yang diperlukan
    asset_files = ['model_klasifikasi.pkl', 'le_platform.pkl', 'le_genre.pkl', 'le_publisher.pkl']
    
    # Cek keberadaan semua file
    for filename in asset_files:
        if not os.path.exists(filename):
            st.error(f"Error: File aset '{filename}' tidak ditemukan.")
            st.error("Pastikan semua file .pkl hasil dari Google Colab (setelah K=8) berada di folder yang sama dengan app.py.")
            st.stop()
            
    try:
        model = load('model_klasifikasi.pkl')
        le_platform = load('le_platform.pkl')
        le_genre = load('le_genre.pkl')
        le_publisher = load('le_publisher.pkl')
        return model, le_platform, le_genre, le_publisher
    except Exception as e:
        st.error(f"Error saat memuat model/encoders: {e}")
        st.error("Ini mungkin disebabkan oleh ketidakcocokan versi Python/library. Coba instal ulang scikit-learn dan joblib.")
        st.stop()


model, le_platform, le_genre, le_publisher = load_assets()

# --- 2. Pemetaan Klaster (K=8) ---
# Pemetaan ini diatur agar Klaster 0 tidak lagi menjadi Penjualan Tertinggi.
# Harap verifikasi urutan klaster 0-7 dengan output cluster_analysis Colab Anda.
CLUSTER_MAP = {
    # Penjualan Minimal (Klaster terdistribusi rendah)
    0: {'label': 'Klaster Penjualan Minimal (Niche)', 'icon': '💨', 'color': 'gray'}, 
    4: {'label': 'Klaster Penjualan Rendah II', 'icon': '🔻', 'color': 'red'},
    5: {'label': 'Klaster Penjualan Sangat Rendah', 'icon': '🐌', 'color': 'brown'},
    # Penjualan Menengah
    2: {'label': 'Klaster Penjualan Menengah Bawah', 'icon': '💼', 'color': 'teal'},
    3: {'label': 'Klaster Penjualan Menengah Atas I', 'icon': '📈', 'color': 'green'},
    6: {'label': 'Klaster Penjualan Menengah Atas II', 'icon': '💰', 'color': 'lime'},
    # Penjualan Tertinggi
    1: {'label': 'Klaster Penjualan Tinggi II (Gold Tier)', 'icon': '🥇', 'color': '#FFAC33'}, 
    7: {'label': 'Klaster Penjualan Tinggi I (Platinum Tier)', 'icon': '🏆', 'color': '#FFD700'},
}

# --- 3. Fungsi Prediksi ---
def predict_cluster(platform, genre, publisher, year):
    """Melakukan encoding input dan prediksi menggunakan model yang dimuat."""
    
    # 3.1. Encoding Input
    try:
        # Transformasi dilakukan pada nilai tunggal dan diambil elemen pertamanya
        platform_encoded = le_platform.transform([platform])[0]
        genre_encoded = le_genre.transform([genre])[0]
        publisher_encoded = le_publisher.transform([publisher])[0]
    except ValueError as e:
        # Ini terjadi jika input tidak ada dalam data training 500 baris (publisher yang sangat jarang)
        st.error(f"Gagal memprediksi: Input {e} tidak dikenali oleh model. Coba pilih Publisher/Genre/Platform lain.")
        return None, "Input tidak valid"

    # 3.2. Buat DataFrame untuk Input Model
    input_data = pd.DataFrame({
        'Platform_Encoded': [platform_encoded],
        'Genre_Encoded': [genre_encoded],
        'Publisher_Encoded': [publisher_encoded],
        'Year': [year]
    })

    # 3.3. Prediksi
    predicted_cluster = model.predict(input_data)[0]
    return predicted_cluster, None

# --- 4. Tampilan Aplikasi Streamlit ---

st.set_page_config(
    page_title="Prediksi Klaster Penjualan Game",
    page_icon="🎮",
    layout="wide"
)

st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .st-emotion-cache-18ni7ap { /* Class for button container */
        display: flex;
        justify-content: center;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
        background-color: #388E3C; /* Dark Green */
        color: white;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2E7D32; /* Slightly darker green on hover */
    }
    .cluster-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 3px solid; /* Border lebih tebal */
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


st.header("🎮 Prediktor Klaster Penjualan Game (PS3, PS4, X360, PC)")
st.caption("Aplikasi ini menggunakan Model Klasifikasi Random Forest yang dilatih pada 500 sampel data klaster K-Means dengan **K=8**.")

# Kolom untuk Input
col1, col2 = st.columns(2)

# Dapatkan semua kategori unik dari encoder
platforms = sorted(le_platform.classes_.tolist())
genres = sorted(le_genre.classes_.tolist())
publishers = sorted(le_publisher.classes_.tolist())

with col1:
    st.subheader("Detail Game")
    platform_input = st.selectbox("Pilih Platform", platforms)
    genre_input = st.selectbox("Pilih Genre", genres)
    
with col2:
    st.subheader("Detail Tambahan")
    publisher_input = st.selectbox("Pilih Publisher", publishers)
    
    # Batas tahun 1980 - 2020 sesuai dataset
    year_input = st.number_input("Tahun Rilis (1980 - 2020)", min_value=1980, max_value=2020, value=2010, step=1)
    
st.markdown("---")

# Tombol Prediksi di tengah
if st.button("PREDIKSI KLASTER PENJUALAN"):
    if year_input < 2005 and platform_input in ['PS4', 'X360']:
        st.warning(f"💡 Peringatan Logika: Tahun {year_input} tidak masuk akal untuk platform {platform_input}. Model mungkin memprediksi tinggi karena bobot platform dominan di data pelatihan.")
        
    with st.spinner('Model sedang memprediksi...'):
        
        # Panggil fungsi prediksi
        predicted_cluster, error = predict_cluster(platform_input, genre_input, publisher_input, year_input)

        if error:
            # Error ditangani di fungsi load_assets atau predict_cluster
            pass 
        else:
            # Ambil detail klaster dari peta
            cluster_info = CLUSTER_MAP.get(predicted_cluster, {'label': 'Klaster Tidak Dikenal', 'icon': '❓', 'color': 'gray'})
            
            label = cluster_info['label']
            icon = cluster_info['icon']
            color = cluster_info['color']
            
            # Tampilkan Hasil dengan styling
            st.markdown(f"""
            <div class="cluster-box" style="border-color: {color};">
                <p class="big-font" style="color: {color};">
                    {icon} Klaster Diprediksi: **{label}**
                </p>
                <p style="font-size: 18px;">
                    Game ini diprediksi berada di klaster <b>{predicted_cluster}</b>.
                    <br>
                    Coba input seperti **Platform: PC, Genre: Strategy, Publisher: Koei, Tahun: 2000** untuk melihat hasil klaster rendah.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()