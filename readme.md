**#### 필요 라이브 설치 ####**

**pip install	transformers** 

			**datasets** 

			**sentencepiece** 

			**rouge-score** 

			**beautifulsoup4** 

			**requests** 

			**rich** 

			**matplotlib**



**#### 프로젝트 구조 ####**

**Project\_Beta**

**|**

**|ㅡㅡㅡ Dataset\_loader\_.py			# CNN / DilyMail 데이터 불러오기 및 입력 된 URL 본문 추출**

**|ㅡㅡㅡ Summarizer\_.py				# BART 모델 기반으로 영어 요약**

**|ㅡㅡㅡ Translator\_.py				# M2M 모델로 영어 -> 한국어로 번역**

**|ㅡㅡㅡ Evaluator\_.py				# ROUGE 평가 및 그래프 생성**

**|ㅡㅡㅡ Utils\_.py					# 출력 포맷 및 실생시간 측정 유틸 모음**

**|\_\_\_\_\_\_ Main\_.py					# 전체 실행**









