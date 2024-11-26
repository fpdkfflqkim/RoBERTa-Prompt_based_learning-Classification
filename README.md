# Context-Based Prompt Selection Methodology to Enhance Performance in Prompt-Based Learning
Paper

## Purpose
Prompt based learning 기법 중 하나인 PET(Pattern Exploiting Training)의 한계점을 개선(무작위로 앙상블 참여 PVP 선정)

## Methodology
기존 데이터셋과의 유사도를 기준으로 가장 가까운 PVP를 선정하여 PET 학습 수행 및 기존 PET와의 성능 비교

	Model : Roberta-large
	Dataset : Yelp polarity

## Result
1. 기존 방법론보다 평균 약 1.23%p 성능 향상
2. 학습 전 앙상블 참여 PVP를 사전에 선정. 효과적인 학습을 수행
3. 기존 데이터셋과 PVP(프롬프트)의 유사도를 기반으로 새로운 선정 기준 제시