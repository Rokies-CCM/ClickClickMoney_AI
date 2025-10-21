---
title: 라이브 운영 팁
version: 1.4
last_updated: 2025-10-19
locale: ko-KR
---

# 라이브 운영 팁

## 1) 핵심 목표
- p95 지연 **< 900ms**(SSE 첫 토막 기준)  
- **의미 캐시 적중률 ≥ 0.70**  
- **오류율(5xx) < 0.5%**

## 2) 지연(p95) 상승 시 점검 순서
1) **웹검색/재랭크 호출 비중** 확인  
   - `USE_TAVILY` 켜짐 여부, Upstage rerank 호출 빈도  
   - 긴 질문(≥120자)에서만 rerank가 동작하는지 확인
2) **모델 라우팅**  
   - 단문이 OpenAI로 과도 전환됐는지(복잡 키워드 false positive)  
   - `GROQ_MODEL_UPSCALE`이 70B 등 고비용으로 고정되지 않았는지
3) **네트워크/클라이언트**  
   - `httpx` 타임아웃/리밋(`CONNECT=5s`, `READ/WRITE=60s`)  
   - Keep-Alive/HTTP2 활성화 여부(이미 on)
4) **RAG**  
   - 인덱스 누락/빈 검색 경고(로그 warn-once) 발생 여부  
   - `FAISS_EF_SEARCH`(권장 160) 및 `TOP_K`(권장 4) 확인

## 3) 캐시
- **의미 캐시**(0.90 이상): 동일/유사 질의 재사용  
- **응답 캐시**: 질의+컨텍스트 키 기반 15분 TTL  
- 캐시 미스가 많은 경우: 질의 표면형 정규화(`normalize_question`) 재점검

## 4) 응답 길이 관리
- 장문 증가 시 `.env`의 `REPLY_MAX_TOKENS`를 **360**으로 상향  
- STOP 토큰(`출처:,참고:` 등)으로 근거 섹션 앞 과도 생성 방지  
- 프롬프트에서 “요약+실행 불릿 2–3개” 준수

## 5) 장애/오류 대응
- 전역 예외 응답: 메시지 비공개, **trace_id**만 노출  
- CORS/도메인: 운영 시 `CORS_ORIGINS`를 와일드카드(`*`)에서 **도메인**으로 제한  
- 업로드 오류: `INVALID_HEADERS / FILE_TOO_LARGE / INVALID_MONTH` 등 표준 코드 확인

## 6) 운영 체크리스트
- [ ] `/v1/metrics`의 `hit_rate`, `provider_counts`, 토큰 평균 모니터링  
- [ ] `/v1/bench/recent`에서 p50/p95/avg 변동 감시  
- [ ] 의미 캐시가 **질의 정규화**와 함께 정상 동작  
- [ ] 링크/수치 가드가 스트림에 적용(“[링크 생략]”, “[수치 생략]”)  
