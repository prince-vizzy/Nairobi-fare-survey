"""
Agent layer for the Nairobi Transit Assistant.
Uses google-genai SDK with Gemini chat sessions.
Chat sessions cache the system prompt context, avoiding re-sending it every turn.
"""

import time
import logging
from google import genai
from google.genai import types

log = logging.getLogger(__name__)

MODELS = [
    "gemini-2.5-flash-lite",  # fastest — primary for all transit queries
    "gemini-2.0-flash-lite",  # proven fallback
    "gemini-2.0-flash",       # slightly slower, more capable
    "gemini-2.5-flash",       # last resort — most capable but overkill for directions
]


class AgentOrchestrator:

    def __init__(self, api_key: str, system_prompt: str, few_shot_history: list | None = None):
        self.client           = genai.Client(api_key=api_key)
        self.system_prompt    = system_prompt
        self.few_shot_history = few_shot_history or []
        self._sessions: dict[str, tuple[str, object]] = {}

    def _make_session(self, model: str):
        return self.client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.2,        # focused, consistent tone
                max_output_tokens=280,  # 3-5 sentences — keeps responses punchy and fast
            ),
            history=self.few_shot_history,
        )

    def _session(self, session_id: str, model: str):
        if session_id not in self._sessions or self._sessions[session_id][0] != model:
            self._sessions[session_id] = (model, self._make_session(model))
        return self._sessions[session_id][1]

    def process(self, session_id: str, message: str, **_) -> dict:
        for model in MODELS:
            for attempt in range(3):
                try:
                    reply = self._session(session_id, model).send_message(message)
                    return {"reply": reply.text, "model": model, "agent": "gemini"}
                except Exception as exc:
                    err = str(exc)
                    if "503" in err or "UNAVAILABLE" in err:
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            continue
                        break
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            continue
                        break
                    raise

        raise RuntimeError("All Gemini models unavailable. Try again in a moment.")
