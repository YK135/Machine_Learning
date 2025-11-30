import torch
from transformers import BartForConditionalGeneration, BartTokenizer

class EnglishSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        print(f"[INFO] BART 모델 로딩 중 ... ({model_name})")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Summarizer Device: {self.device}")

        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)

    """ 
    Main 코드 젤 중요 부분 
    """
    def _summarize_once(self, text, max_len, min_len):
        # 너무 짧은 텍스트는 요약하지 않음
        if len(text.split()) < 20:
            return text.strip()

        # ============= 입력 인코딩 =============
        inputs = self.tokenizer(
            text,                   # 요약하려는 기사의 원문의 문자열
            max_length=1024,        # bart는 1024 토큰까지 입력 받음 (이상으로 길어지면 잘라버림)
            truncation=True,        # 길면 자르기 (여기서 앞부분이 위주로 요약설정 됨)
            return_tensors="pt"     # 결과를 파이토치텐서 형태로 변형함
        ).to(self.device)           # cpu/gpu 맞춰서 텐서 

        # bart 요약
        try:
            summary_ids = self.model.generate(
                inputs["input_ids"],        # 인풋 아이디 텐서
                num_beams=4,                # 빔 서치 개수 (가장 확률이 높은 문장 후보를 4개 가지치지 하면서 탐색함)
                                            # >> 빔 개수가 많을수록 성능이 좋아지지만 속도는 확연히 느려짐
                max_length=max_len,         # 요약문 최대 길이
                min_length=min_len,         # 요약문 최소 길이
                early_stopping=True,        # 최소/최대 길이에 도달하면 멈춤 (불필요한 연산을 줄여줌)           
                no_repeat_ngram_size=3      # 같은 문자 반복을 막아줌 ex) my name is ~~ so my name is ~~ my name is ~~
            )                               # >> my name is 같이 반복되는걸 방지함

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

            # 빈 요약 방지
            if not summary:
                print("[WARN] empty summary → fallback")
                return text[:300]

            return summary

        except Exception as e:
            print(f"[ERROR] summarization failed: {e}")
            return text[:300]   # fallback 요약

    # 계층구조 변형한 요약 함수
    def summarize(self, text, return_both=False):
        # Pass 1
        mid = self._summarize_once(text, max_len=120, min_len=40)

        # Pass1이 너무 짧으면 Pass2 불필요
        if len(mid.split()) < 10:
            if return_both:
                return mid, mid
            return mid

        # Pass 2
        final = self._summarize_once(mid, max_len=60, min_len=20)

        if return_both:
            return mid, final
        return final