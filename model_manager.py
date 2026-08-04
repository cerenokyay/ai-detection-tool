import os
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

class ModelManager:
    def __init__(self, model_id: str = "gpt2"):
        """
        Optimum kütüphanesi ile ONNX dönüşüm yöneticisini başlatır.
        """
        self.model_id = model_id
        self.onnx_export_dir = "gpt2_onnx_model"

    def export_to_onnx(self):
        """
        HuggingFace Optimum kullanarak modeli ONNX'e çevirir.
        Bu yöntem PyTorch'un FakeTensor hatalarını tamamen aşar.
        """
        print(f"[{self.model_id}] modeli ONNX formatına dönüştürülüyor (Optimum kullanılıyor)...")
        
        # Tokenizer'ı kaydet
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.save_pretrained(self.onnx_export_dir)
        
        # Optimum ile modeli indir, ONNX'e çevir ve klasöre kaydet
        print("Model indiriliyor ve ONNX grafiği çıkarılıyor (1-2 dakika sürebilir)...")
        model = ORTModelForCausalLM.from_pretrained(self.model_id, export=True)
        model.save_pretrained(self.onnx_export_dir)
        
        print(f"\nBaşarılı! Model ve tokenizer şu klasöre kaydedildi: {self.onnx_export_dir}/")
        
        # Boyut kontrolü
        onnx_file_path = os.path.join(self.onnx_export_dir, "model.onnx")
        if os.path.exists(onnx_file_path):
            size_mb = os.path.getsize(onnx_file_path) / (1024 * 1024)
            print(f"ONNX Model Boyutu: {size_mb:.2f} MB")

if __name__ == "__main__":
    manager = ModelManager()
    manager.export_to_onnx()