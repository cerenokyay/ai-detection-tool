import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import List, Dict, Any

class TextMetricsAnalyzer:
    def __init__(self, model_id: str = "gpt2"):
        """
        Analiz motorunu başlatır. Modeli ve tokenizer'ı yükler, 
        Apple Silicon (MPS) hızlandırmasını aktif eder.
        """
        print(f"Metrics Analyzer başlatılıyor... {model_id} modeli yükleniyor.")
        
        # Cihaz seçimi: M serisi Mac'ler için MPS, yoksa CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Kullanılan donanım (Device): {self.device.upper()}")
        
        # Tokenizer ve Model yükleme
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.model.eval() # Modeli değerlendirme (inference) moduna alıyoruz

    def calculate_perplexity(self, text: str) -> float:
        """
        Metnin Perplexity (Şaşkınlık) değerini hesaplar.
        Düşük değer = Yapay zeka olma ihtimali yüksek.
        Yüksek değer = İnsan olma ihtimali yüksek.
        """
        if not text.strip():
            return 0.0

        # Metni token'lara çevir ve ilgili cihaza (MPS) gönder
        encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Metnin maksimum token uzunluğunu aşmasını engelle
        max_length = self.model.config.n_positions
        if encodings.input_ids.size(1) > max_length:
            # Çok uzunsa kesiyoruz
            encodings.input_ids = encodings.input_ids[:, :max_length]
            encodings.attention_mask = encodings.attention_mask[:, :max_length]

        with torch.no_grad(): # Gradient hesaplamasını kapat (hızlandırır ve RAM tasarrufu sağlar)
            # Modeli kendi girdisiyle test et ve loss (kayıp) değerini al
            outputs = self.model(
                input_ids=encodings.input_ids, 
                attention_mask=encodings.attention_mask,
                labels=encodings.input_ids
            )
            loss = outputs.loss
            
            # Perplexity formülü: e^loss (Euler sayısı üzeri kayıp değeri)
            perplexity = torch.exp(loss).item()
            
        return perplexity

    def calculate_burstiness(self, sentences: List[str]) -> float:
        """
        Metnin Burstiness (Patlamasallık) değerini hesaplar.
        Cümle uzunluklarının standart sapmasını (varyansını) kullanır.
        """
        if not sentences or len(sentences) <= 1:
            return 0.0

        # Her bir cümlenin kelime sayısını bul
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        
        # Standart sapmayı hesapla (Cümle uzunluklarındaki dalgalanma)
        burstiness = float(np.std(sentence_lengths))
        return burstiness

    def analyze(self, nlp_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        NLP Pipeline'dan gelen verileri alır ve tüm metrikleri hesaplar.
        """
        text = nlp_result.get("cleaned_text", "")
        sentences = nlp_result.get("sentences", [])

        perplexity = self.calculate_perplexity(text)
        burstiness = self.calculate_burstiness(sentences)

        # Basit bir eşik (threshold) mantığı - Daha sonra makine öğrenmesi modeliyle değiştirilebilir
        is_ai_generated = perplexity < 60 and burstiness < 15

        return {
            "perplexity": round(perplexity, 2),
            "burstiness": round(burstiness, 2),
            "is_ai_generated_prediction": is_ai_generated,
            "device_used": self.device
        }

# Test Scripti
if __name__ == "__main__":
    from nlp_pipeline import NLPProcessor
    
    print("Modüller yükleniyor, lütfen bekleyin...\n")
    nlp = NLPProcessor()
    analyzer = TextMetricsAnalyzer()
    
    # Test için ChatGPT'nin yazabileceği tipik jenerik bir metin
    ai_text = "Artificial intelligence is a rapidly growing field of technology. It has the potential to change many aspects of our daily lives. Many companies are investing heavily in machine learning algorithms. The future of automation looks very promising."
    
    # Test için düzensiz, insansı bir metin
    human_text = "So, I was thinking about AI today. It's crazy! Right? I mean, we are literally teaching rocks to think by trapping lightning inside them, which sounds like bad sci-fi but here we are."
    
    print("\n--- AI METNİ TESTİ ---")
    ai_nlp = nlp.process_text(ai_text)
    ai_metrics = analyzer.analyze(ai_nlp)
    print(f"Perplexity: {ai_metrics['perplexity']} (Düşük beklenir)")
    print(f"Burstiness: {ai_metrics['burstiness']} (Düşük beklenir)")
    print(f"AI Kararı: {ai_metrics['is_ai_generated_prediction']}")
    
    print("\n--- İNSAN METNİ TESTİ ---")
    human_nlp = nlp.process_text(human_text)
    human_metrics = analyzer.analyze(human_nlp)
    print(f"Perplexity: {human_metrics['perplexity']} (Yüksek beklenir)")
    print(f"Burstiness: {human_metrics['burstiness']} (Yüksek beklenir)")
    print(f"AI Kararı: {human_metrics['is_ai_generated_prediction']}")