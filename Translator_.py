import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class EnglishToKoreanTranslator:
    def __init__(self, model_name: str = "facebook/m2m100_418M"):
        print(f"[INFO] 번역 모델 로드 중... ({model_name})")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Translator device: {self.device}")
        self.model.to(self.device)

    def translate(self, text: str, max_len: int = 256) -> str:
        text = text.strip()
        if not text:
            return ""

        self.tokenizer.src_lang = "en"
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id("ko"),
            max_length=max_len,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        return self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]