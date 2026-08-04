import numpy as np
import joblib
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
from typing import List, Dict, Any

class TextMetricsAnalyzer:
    def __init__(self, onnx_dir: str = "gpt2_onnx_model", clf_path: str = "ai_classifier.pkl"):
        """
        HuggingFace Optimum ve Scikit-learn Classifier ile motoru başlatır.
        """
        print("Metrics Analyzer (Optimum/ONNX + ML Classifier) başlatılıyor...")
        try:
            # 1. NLP ve ONNX Modelleri
            self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
            self.model = ORTModelForCausalLM.from_pretrained(onnx_dir)
            
            # 2. Makine Öğrenmesi Sınıflandırıcısını (Logistic Regression) Yükle
            self.classifier = joblib.load(clf_path)
            
            print("Tüm modeller başarıyla belleğe yüklendi.")
        except Exception as e:
            raise RuntimeError(f"Modeller yüklenirken hata oluştu. Klasörleri ve .pkl dosyasını kontrol edin. Hata: {e}")

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

    def analyze(self, nlp_result: Dict[str, Any]) -> Dict[str, Any]:
        text = nlp_result.get("cleaned_text", "")
        sentences = nlp_result.get("sentences", [])

        # 1. Metrikleri hesapla (Feature Extraction)
        perplexity = self.calculate_perplexity(text)
        burstiness = self.calculate_burstiness(sentences)

        # 2. ML Modeline Ver (Inference)
        # Sınıflandırıcıya iki boyutlu bir dizi (2D array) vermemiz gerekiyor
        features = np.array([[perplexity, burstiness]])
        
        # predict_proba, [İnsan Olma Olasılığı, AI Olma Olasılığı] şeklinde bir dizi döner
        probabilities = self.classifier.predict_proba(features)[0]
        ai_probability_percentage = probabilities[1] * 100 # İkinci index (1) AI sınıfıdır
        
        # Kesin karar (Eğer AI olasılığı %50'den büyükse True)
        is_ai = bool(self.classifier.predict(features)[0] == 1)

        return {
            "perplexity": round(perplexity, 2),
            "burstiness": round(burstiness, 2),
            "ai_probability": f"%{round(ai_probability_percentage, 2)}",
            "is_ai_generated_prediction": is_ai,
            "metrics_metadata": {
                "decision_model": "Logistic Regression Classifier"
            },
            "engine": "ONNX Runtime (via Optimum)"
        }