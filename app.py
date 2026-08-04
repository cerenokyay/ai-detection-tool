import streamlit as st
import requests

# Sayfa Ayarları
st.set_page_config(page_title="AI Metin Dedektifi", page_icon="🕵️‍♀️", layout="centered")

# Başlık ve Açıklama
st.title("🕵️‍♀️ Yapay Zeka Metin Dedektifi")
st.markdown("""
Bu araç, metnin bir insan tarafından mı yoksa yapay zeka (ChatGPT, Claude vb.) tarafından mı yazıldığını 
**Perplexity (Şaşkınlık)** ve **Burstiness (Patlamasallık)** metriklerini makine öğrenmesi ile analiz ederek bulur.
""")

# Metin Giriş Alanı
text_input = st.text_area("Analiz edilecek metni buraya yapıştırın (En az 40 kelime, 3 cümle):", height=250)

# Analiz Butonu
if st.button("Metni Analiz Et", type="primary", use_container_width=True):
    if len(text_input.split()) < 40:
        st.warning("⚠️ Lütfen sağlıklı bir analiz için en az 40 kelimelik bir metin girin.")
    else:
        with st.spinner("ONNX Motoru ve Lojistik Regresyon çalışıyor..."):
            try:
                # FastAPI Backend'ine istek at
                response = requests.post("http://127.0.0.1:8000/analyze", json={"text": text_input})
                
                if response.status_code == 200:
                    data = response.json()
                    analysis = data["analysis"]
                    
                    st.divider()
                    
                    # Sonuç Başlığı
                    if analysis["is_ai_generated_prediction"]:
                        st.error(f"🤖 **Yapay Zeka Tespit Edildi!** (Olasılık: {analysis['ai_probability']})")
                    else:
                        st.success(f"✍️ **İnsan Tarafından Yazılmış!** (Yapay Zeka Olasılığı: {analysis['ai_probability']})")
                        
                    # Metrikleri Göster
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Perplexity (Şaşkınlık)", analysis["perplexity"], help="Yüksek = İnsan")
                    col2.metric("Burstiness (Dalgalanma)", analysis["burstiness"], help="Yüksek = İnsan")
                    col3.metric("İşlem Süresi", f"{data['process_time_ms']} ms", help="ONNX Motoru Yanıt Süresi")
                    
                    st.caption(f"Kullanılan Motor: {analysis['engine']} | Karar Algoritması: {analysis['metrics_metadata']['decision_model']}")
                    
                elif response.status_code == 422:
                    st.warning(f"⚠️ {response.json()['detail']}")
                else:
                    st.error("Sunucudan beklenmeyen bir hata döndü.")
            
            except requests.exceptions.ConnectionError:
                st.error("🚨 API'ye ulaşılamıyor! Lütfen terminalde 'uvicorn main:app --reload' komutunun çalıştığından emin olun.")