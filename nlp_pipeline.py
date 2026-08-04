import spacy
import re
from typing import List, Dict, Any

class NLPProcessor:
    def __init__(self):
        """
        NLP işlemcisini başlatır. Spacy dil modelini yükler.
        """
        print("NLP Pipeline başlatılıyor... Spacy en_core_web_sm modeli yükleniyor.")
        try:
            # İngilizce dil modelini yüklüyoruz
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "Spacy modeli bulunamadı. Lütfen 'python -m spacy download en_core_web_sm' komutunu çalıştırın."
            )

    def clean_text(self, text: str) -> str:
        """
        Ham metni analiz için temizler ve standartlaştırır.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Fazla boşlukları, tab'ları ve yeni satır karakterlerini tek boşluğa indirge
        text = re.sub(r'\s+', ' ', text)
        
        # 2. İsteğe bağlı: Linkleri kaldır (AI tespiti metnin yapısına odaklandığı için linkler gürültüdür)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Başındaki ve sonundaki boşlukları temizle
        return text.strip()

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Metni temizler, Spacy Doc nesnesine dönüştürür ve cümle istatistiklerini çıkarır.
        
        Returns:
            Dict: Temizlenmiş metin, cümle listesi, kelime listesi ve temel istatistikleri içeren sözlük.
        """
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return {"error": "Boş veya geçersiz metin."}

        # Spacy pipeline'ından geçir
        doc = self.nlp(cleaned_text)

        # Cümleleri çıkar (Burstiness analizi için kritik)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # Sadece harf ve rakamlardan oluşan tokenleri (kelimeleri) çıkar (Noktalama işaretlerini filtrele)
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]

        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "sentences": sentences,
            "sentence_count": len(sentences),
            "tokens": tokens,
            "word_count": len(tokens),
            "avg_words_per_sentence": len(tokens) / len(sentences) if len(sentences) > 0 else 0
        }

# Modülü test etmek için basit bir script
if __name__ == "__main__":
    processor = NLPProcessor()
    
    sample_text = """
    Artificial intelligence is rapidly evolving...   It's changing how we work! 
    Check out this link: https://example.com. 
    However, human creativity remains unique.
    """
    
    result = processor.process_text(sample_text)
    
    print("\n--- NLP İşlem Sonucu ---")
    print(f"Cümle Sayısı: {result['sentence_count']}")
    print(f"Kelime Sayısı: {result['word_count']}")
    print(f"Ortalama Cümle Uzunluğu: {result['avg_words_per_sentence']:.2f} kelime")
    print("\nTemizlenmiş Metin:")
    print(result['cleaned_text'])
    print("\nCümleler:")
    for i, sent in enumerate(result['sentences'], 1):
        print(f"{i}. {sent}")