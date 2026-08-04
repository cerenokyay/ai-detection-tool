import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

print("Sentetik veri seti oluşturuluyor...")
# Sınıf 0: İnsan (Yüksek Perplexity, Yüksek Burstiness)
# Sınıf 1: Yapay Zeka (Düşük Perplexity, Düşük Burstiness)

# İnsan verileri (Yaklaşık 1000 örnek)
human_perplexity = np.random.normal(60, 15, 1000) # Ortalama 60, sapma 15
human_burstiness = np.random.normal(12, 4, 1000)  # Ortalama 12, sapma 4
human_labels = np.zeros(1000)

# AI verileri (Yaklaşık 1000 örnek) - GPT-2 profiline uygun
ai_perplexity = np.random.normal(20, 8, 1000)     # Ortalama 20, sapma 8
ai_burstiness = np.random.normal(3, 1.5, 1000)    # Ortalama 3, sapma 1.5
ai_labels = np.ones(1000)

# Verileri birleştir
X = np.column_stack((
    np.concatenate([human_perplexity, ai_perplexity]),
    np.concatenate([human_burstiness, ai_burstiness])
))
y = np.concatenate([human_labels, ai_labels])

print("Lojistik Regresyon modeli eğitiliyor...")
# Modeli oluştur ve eğit
clf = LogisticRegression(random_state=42)
clf.fit(X, y)

# Modelin başarı oranını (Accuracy) ölç
accuracy = clf.score(X, y)
print(f"Model Eğitim Başarısı: %{accuracy * 100:.2f}")

# Modeli diske kaydet
model_path = "ai_classifier.pkl"
joblib.dump(clf, model_path)
print(f"Model başarıyla kaydedildi: {model_path}")