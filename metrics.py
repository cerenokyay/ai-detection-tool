import numpy as np
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
from typing import List, Dict, Any

class TextMetricsAnalyzer:
    def __init__(self, onnx_dir: str = "gpt2_onnx_model"):
        """
        Analiz motorunu HuggingFace Optimum ve ONNX Runtime ile başlatır.
        """
        print(f"Metrics Analyzer (Optimum/ONNX) başlatılıyor...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
            self.model = ORTModelForCausalLM.from_pretrained(onnx_dir)
            print("ONNX Inference Engine Başarıyla Yüklendi.")
        except Exception as e:
            raise RuntimeError(f"ONNX modeli yüklenemedi. Klasör eksik olabilir. Hata: {e}")

    def calculate_perplexity(self, text: str) -> float:
        if not text.strip(): return 0.0
        inputs = self.tokenizer(text, return_tensors="pt")
        
        if inputs["input_ids"].shape[1] > 1024:
             inputs["input_ids"] = inputs["input_ids"][:, :1024]
             inputs["attention_mask"] = inputs["attention_mask"][:, :1024]

        outputs = self.model(**inputs)
        logits = outputs.logits.detach().numpy()
        
        shift_logits = logits[0, :-1, :]
        shift_labels = inputs['input_ids'][0, 1:].numpy()
        
        max_logits = np.max(shift_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shift_logits - max_logits)
        log_probs = shift_logits - max_logits - np.log(np.sum(exp_logits, axis=-1, keepdims=True))
        
        target_log_probs = log_probs[np.arange(len(shift_labels)), shift_labels]
        loss = -np.mean(target_log_probs)
        return float(np.exp(loss))

    def calculate_burstiness(self, sentences: List[str]) -> float:
        if not sentences or len(sentences) <= 1: return 0.0
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        return float(np.std(sentence_lengths))

    # --- YENİ EKLENEN DİNAMİK ALGORİTMA ---
    def calculate_ai_probability(self, perplexity: float, burstiness: float, word_count: int, sentence_count: int) -> dict:
        """
        Metin uzunluğuna göre dinamik eşikler belirler ve % üzerinden AI olasılığı hesaplar.
        """
        # 1. Dinamik Eşikleri Belirle
        # Cümle sayısı azsa, beklenen burstiness düşüktür (Min: 5, Max: 20 olacak şekilde logaritmik/lineer artış)
        dynamic_burstiness_threshold = min(20.0, 5.0 + (sentence_count * 0.6))
        
        # Kelime sayısı azsa, metin dar kapsamlıdır ve perplexity doğal olarak düşüktür.
        # Kelime sayısı arttıkça beklenen perplexity eşiği yükselir.
        dynamic_perplexity_threshold = max(35.0, 75.0 - (1000 / (word_count + 1)))

        # 2. Skorlama Mantığı (0 = Kesin İnsan, 100 = Kesin Yapay Zeka)
        # Perplexity Skoru (%60 Ağırlık)
        if perplexity < (dynamic_perplexity_threshold * 0.5):
            p_score = 100.0
        elif perplexity > dynamic_perplexity_threshold:
            p_score = 0.0
        else:
            # Eşikler arasında lineer bir yüzde hesapla
            p_score = 100.0 - ((perplexity - (dynamic_perplexity_threshold * 0.5)) / (dynamic_perplexity_threshold * 0.5) * 100.0)

        # Burstiness Skoru (%40 Ağırlık)
        if burstiness < (dynamic_burstiness_threshold * 0.3):
            b_score = 100.0
        elif burstiness > dynamic_burstiness_threshold:
            b_score = 0.0
        else:
            b_score = 100.0 - ((burstiness - (dynamic_burstiness_threshold * 0.3)) / (dynamic_burstiness_threshold * 0.7) * 100.0)

        # Nihai Olasılığı Hesapla
        ai_probability = (p_score * 0.6) + (b_score * 0.4)

        return {
            "ai_probability_percentage": round(ai_probability, 2),
            "is_ai_generated": ai_probability > 50.0,
            "dynamic_thresholds": {
                "expected_min_perplexity_for_human": round(dynamic_perplexity_threshold, 2),
                "expected_min_burstiness_for_human": round(dynamic_burstiness_threshold, 2)
            }
        }

    def analyze(self, nlp_result: Dict[str, Any]) -> Dict[str, Any]:
        text = nlp_result.get("cleaned_text", "")
        sentences = nlp_result.get("sentences", [])
        word_count = nlp_result.get("word_count", 0)
        sentence_count = nlp_result.get("sentence_count", 0)

        perplexity = self.calculate_perplexity(text)
        burstiness = self.calculate_burstiness(sentences)

        # Yeni dinamik analiz motorunu çağırıyoruz
        decision_data = self.calculate_ai_probability(perplexity, burstiness, word_count, sentence_count)

        return {
            "perplexity": round(perplexity, 2),
            "burstiness": round(burstiness, 2),
            "ai_probability": f"%{decision_data['ai_probability_percentage']}",
            "is_ai_generated_prediction": decision_data['is_ai_generated'],
            "metrics_metadata": decision_data['dynamic_thresholds'],
            "engine": "ONNX Runtime (via Optimum) with Dynamic Thresholds"
        }