from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time
from typing import Dict, Any

from nlp_pipeline import NLPProcessor
from metrics import TextMetricsAnalyzer

app = FastAPI(
    title="AI Text Detection API",
    description="Metinlerin yapay zeka tarafından yazılıp yazılmadığını Perplexity ve Burstiness ile analiz eden yüksek performanslı ONNX tabanlı API.",
    version="1.0.0"
)

nlp_processor = None
metrics_analyzer = None

class TextRequest(BaseModel):
    # Kullanıcının en azından boş bir şey veya tek harf göndermesini Pydantic ile engelliyoruz
    text: str = Field(..., min_length=50, description="Analiz edilecek metin (en az 50 karakter olmalı)")

@app.on_event("startup")
async def load_models():
    global nlp_processor, metrics_analyzer
    print("API Başlatılıyor... Yapay zeka modelleri belleğe yükleniyor.")
    nlp_processor = NLPProcessor()
    metrics_analyzer = TextMetricsAnalyzer()
    print("Sistem istek almaya hazır!")

@app.get("/")
async def root():
    return {"message": "AI Text Detection API çalışıyor. Analiz için /analyze endpointine POST isteği gönderin."}

@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_text(request: TextRequest):
    start_time = time.time()
    
    try:
        # 1. Metni NLP pipeline'ından geçir
        nlp_result = nlp_processor.process_text(request.text)
        
        if "error" in nlp_result:
            raise HTTPException(status_code=400, detail="Metin işlenirken bir hata oluştu.")
            
        word_count = nlp_result["word_count"]
        sentence_count = nlp_result["sentence_count"]

        # --- YENİ EKLENEN PROFESYONEL DOĞRULAMA (VALIDATION) ADIMI ---
        
        # Kural 1: En az 40 kelime olmalı (İstatistiksel anlamlılık için)
        if word_count < 40:
            raise HTTPException(
                status_code=422, 
                detail=f"Metin çok kısa. Doğru bir yapay zeka tespiti yapabilmek için en az 40 kelime girmelisiniz. (Şu anki: {word_count} kelime)"
            )
            
        # Kural 2: En az 3 cümle olmalı (Burstiness - Standart Sapma hesabı için)
        if sentence_count < 3:
            raise HTTPException(
                status_code=422, 
                detail=f"Cümle sayısı yetersiz. Cümle uzunluklarındaki dalgalanmayı (Burstiness) ölçebilmemiz için metin en az 3 cümleden oluşmalıdır. (Şu anki: {sentence_count} cümle)"
            )
            
        # -------------------------------------------------------------

        # 2. Kuralları geçtiyse ONNX motoru ile analiz et
        analysis_result = metrics_analyzer.analyze(nlp_result)
        
        process_time_ms = round((time.time() - start_time) * 1000, 2)
        
        return {
            "status": "success",
            "process_time_ms": process_time_ms,
            "text_stats": {
                "word_count": word_count,
                "sentence_count": sentence_count
            },
            "analysis": analysis_result
        }
        
    except HTTPException:
        # FastAPI'nin kendi HTTP hatalarını doğrudan dışarı aktar
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu tarafında bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)