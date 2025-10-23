# core/models.py
from __future__ import annotations

import os
import logging
from typing import AsyncGenerator, Optional, List, Dict, Any

# OpenAI SDK & 예외 호환 (버전 차이 대비)
try:
    from openai import AsyncOpenAI, BadRequestError
    try:
        from openai import APIStatusError, NotFoundError  # 최신 SDK
    except Exception:  # 구버전 호환
        class APIStatusError(Exception): ...
        class NotFoundError(Exception): ...
except Exception as _e:
    from openai import AsyncOpenAI  # type: ignore
    class BadRequestError(Exception): ...
    class APIStatusError(Exception): ...
    class NotFoundError(Exception): ...

from core.config import get_settings

log = logging.getLogger("ai-chatbot")
_settings = get_settings()


def _compose_messages(system: str, developer: str, user: str, context: Optional[str], chat_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    """
    system + developer + context를 하나의 system 메시지로 병합하여
    OpenAI/Groq 모두 호환되게 만듦.
    최종: [system], [user]
    """
    sys_parts: List[str] = []
    if system:
        sys_parts.append(system.strip())
    if developer:
        sys_parts.append(developer.strip())
    if context:
        sys_parts.append(f"[컨텍스트]\n{context.strip()}")

    sys_text = "\n\n".join([p for p in sys_parts if p])

    messages: List[Dict[str, str]] = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user})
    return messages

def _dyn_max_tokens(prompt: str) -> Optional[int]:
    """
    REPLY_MAX_TOKENS(기본) + 프롬프트 길이 기반 확대.
    - >1200자: max(기본, 360)
    - >600자 : max(기본, 320)
    - 그 외: 기본
    """
    base = getattr(_settings, "REPLY_MAX_TOKENS", 0) or 0
    if base <= 0:
        return None
    plen = len(prompt or "")
    if plen > 1200:
        return max(base, 360)
    if plen > 600:
        return max(base, 320)
    return base


class OpenAICompatibleClient:
    """
    OpenAI 호환 Chat Completions API.
    - developer role 비호환 방지(system 병합)
    - Groq 오류 시 1차: Groq 내부 폴백 모델 → 2차: OpenAI 최종 폴백
      (본 프로젝트는 GPT 단일 경로라 사용되지 않지만 안전장치로 유지)
    """
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise RuntimeError(f"API key missing for base_url={base_url}")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.base_url = base_url  # 감지용

    def _stop_tokens(self) -> Optional[List[str]]:
        stops = getattr(_settings, "STOP_TOKENS", None)
        if isinstance(stops, (list, tuple)):
            arr = [s for s in stops if isinstance(s, str) and s.strip()]
            return arr or None
        # 혹시 문자열로 들어온 환경에서의 호환(예: "a,b,c")
        raw = os.getenv("STOP_TOKENS", "").strip()
        if not raw:
            return None
        arr = [s.strip() for s in raw.split(",") if s.strip()]
        return arr or None

    def _is_groq(self) -> bool:
        try:
            return "groq" in (self.base_url or "").lower()
        except Exception:
            return False

    def _groq_fallback_model(self) -> str:
        # 우선순위: GROQ_FALLBACK_MODEL > GROQ_MODEL > 기본값
        return (
            os.getenv("GROQ_FALLBACK_MODEL")
            or os.getenv("GROQ_MODEL")
            or "llama-3.1-8b-instant"
        )

    async def _create_with_fallback(self, **kwargs) -> Any:
        """
        1) 1차 시도: 현재 클라이언트(모델)로 생성
        2) Groq이고 실패(400/404/기타 API 오류)면 Groq 폴백 모델 1회 재시도
        3) 그래도 실패 시 OpenAI 최종 폴백
        """
        try:
            return await self.client.chat.completions.create(**kwargs)
        except (BadRequestError, NotFoundError, APIStatusError) as e:
            if self._is_groq():
                fb = self._groq_fallback_model()
                log.warning(f"[groq] primary '{kwargs.get('model')}' failed: {e}. fallback -> {fb}")
                try:
                    kwargs["model"] = fb
                    return await self.client.chat.completions.create(**kwargs)
                except (BadRequestError, NotFoundError, APIStatusError) as e2:
                    log.warning(f"[groq] fallback '{fb}' failed: {e2}. final -> openai")
        except Exception as e:
            # 기타 예외도 안전 폴백
            if self._is_groq():
                fb = self._groq_fallback_model()
                log.warning(f"[groq] primary '{kwargs.get('model')}' failed({type(e).__name__}): {e}. fallback -> {fb}")
                try:
                    kwargs["model"] = fb
                    return await self.client.chat.completions.create(**kwargs)
                except Exception as e2:
                    log.warning(f"[groq] fallback '{fb}' failed({type(e2).__name__}): {e2}. final -> openai")

        # 최종 폴백(OpenAI)
        oai_key = _settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        oai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        oai_model = (os.getenv("OAI_MODEL_DEFAULT")  # ✅ 추가 지원
                     or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        oai = AsyncOpenAI(api_key=oai_key, base_url=oai_base)
        kwargs["model"] = oai_model
        log.warning(f"[fallback] using openai model '{kwargs['model']}'")
        return await oai.chat.completions.create(**kwargs)

    async def stream(self, system: str, developer: str, user: str, context: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[str, None]:
        messages = _compose_messages(system, developer, user, context, chat_history)
        full_prompt = (system or "") + (developer or "") + (context or "") + (user or "")
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": True,
        }
        mt = _dyn_max_tokens(full_prompt)
        if mt is not None:
            kwargs["max_tokens"] = mt
        stop = self._stop_tokens()
        if stop:
            kwargs["stop"] = stop

        resp = await self._create_with_fallback(**kwargs)
        async for chunk in resp:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate(self, system: str, developer: str, user: str, context: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        messages = _compose_messages(system, developer, user, context, chat_history)
        full_prompt = (system or "") + (developer or "") + (context or "") + (user or "")
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": False,
        }
        mt = _dyn_max_tokens(full_prompt)
        if mt is not None:
            kwargs["max_tokens"] = mt
        stop = self._stop_tokens()
        if stop:
            kwargs["stop"] = stop

        resp = await self._create_with_fallback(**kwargs)
        return (resp.choices[0].message.content or "").strip()


def build_groq_client() -> OpenAICompatibleClient:
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = _settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    return OpenAICompatibleClient(api_key=api_key, base_url=base_url, model=model)


def build_openai_client() -> OpenAICompatibleClient:
    # OAI_MODEL_DEFAULT 우선 지원, 없으면 기존 OPENAI_MODEL 사용
    model = (os.getenv("OAI_MODEL_DEFAULT")
             or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = _settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    return OpenAICompatibleClient(api_key=api_key, base_url=base_url, model=model)
