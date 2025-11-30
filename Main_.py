from Dataset_loader_ import load_articles, extract_article_from_url
from Summarizer_ import EnglishSummarizer
from Translator_ import EnglishToKoreanTranslator
from Evaluator_ import evaluate_summaries, plot_rouge_scores, save_result
from Utils_ import print_section
from rich import print

if __name__ == "__main__":
    print_section("뉴스 기사 요약 & 번역 시스템")

    # 모드 선택: URL 입력 or 엔터로 배치 모드
    user_input = input("기사 URL을 입력하세요 [엔터만 누르면 CNN/DailyMail 평가 모드]: ").strip()

    summarizer = EnglishSummarizer()
    translator = EnglishToKoreanTranslator()

    if user_input.startswith("http"):
        article = extract_article_from_url(user_input)
        if not article:
            print("[bold red][ERROR] URL에서 기사를 가져오지 못했습니다.[/bold red]")
        else:
            print_section("원문 일부\n")
            print(article[:500] + "\n...")

            # Multi-pass 요약: 최종 한 줄 영어 요약
            mid_en, final_en = summarizer.summarize(article, return_both=True)
            mid_ko = translator.translate(mid_en)

            print_section("요약 결과 (영어)\n")
            print(f"[1차 요약]: {mid_en}\n")
            print(f"[최종 한 줄 요약]: {final_en}\n")

            # 한국어 번역
            final_ko = translator.translate(final_en)
            print_section("한국어 번역\n")
            print(f"[1차 한국어 번역]: {mid_ko}\n")
            print(f"[한국어 번역]: {final_ko}\n")

    else:
        print_section("배치 평가 모드 (CNN/DailyMail 샘플)\n")
        n = 10  # ====== 평가용 기사 n개 불러오기 ======
        articles = load_articles(n)

        english_summaries, korean_summaries = [], []

        for i, article in enumerate(articles, start=1):
            print_section(f"기사 {i}")

            mid_en, final_en = summarizer.summarize(article, return_both=True)
            mid_ko = translator.translate(mid_en)
            final_ko = translator.translate(final_en)

            print(f"[1차 영어 요약]: {mid_en}\n")
            print(f"[1차 한국어 번역]: {mid_ko}\n")
            print(f"[최종 영어 한 줄 요약]: {final_en}\n")
            print(f"[한국어 번역]: {final_ko}\n")

            english_summaries.append(final_en)
            korean_summaries.append(final_ko)
            save_result(article, final_en, final_ko)

        rouge_scores = evaluate_summaries(articles, english_summaries)
        plot_rouge_scores(rouge_scores)

        print_section("평가 완료")