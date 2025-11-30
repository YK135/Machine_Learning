from datasets import load_dataset
import requests
from bs4 import BeautifulSoup

def load_articles(n):
    print(f"[INFO] CNN/DailyMail 데이터셋에서 기사 {n}개 로드 중...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{n}]")
    return [item["article"] for item in dataset]

def extract_article_from_url(url: str) -> str:
    """
    뉴스 기사 URL에서 본문 텍스트를 추출,
    언론사마다 구조가 달라서 완벽하진 않음,
    대표적인 article/body 영역을 우선적으로 탐색
    """
    print(f"[INFO] URL에서 기사 본문 추출 중... ({url})")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] 요청 실패: {e}")
        return " "

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    """
    언론사별로 자주 쓰는 본문 영역 후보의 태그들을 리스트로 모아줌
    외에 클래스들은 자주 쓰이는 이름들
    """
    candidates = [
        soup.find("article"),
        soup.find("div", {"id": "dic_area"}),          # 네이버 뉴스
        soup.find("div", {"class": "news_body"}),
        soup.find("div", {"class": "article_body"}),
        soup.find("div", {"class": "art_txt"}),
    ]

    for c in candidates:
        if c:
            text = c.get_text(separator="\n")
            cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            return cleaned

    # 예외처리 (후보에서 못 찾으면 페이지 전체를 번역해줌)
    fallback = soup.get_text(separator="\n")
    cleaned = "\n".join(line.strip() for line in fallback.splitlines() if line.strip())
    return cleaned
