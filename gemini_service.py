import re
import json
import logging
from typing import List, Dict, Any, Generator, Tuple, Optional
from google import genai
from google.genai import types
from google.genai import errors
import config

logger = logging.getLogger("omnimind.gemini")

BASE_SYSTEM_INSTRUCTION = """You are OmniMind AI, an advanced, intelligent, general-purpose AI assistant.

Your responsibility is to assist users across virtually all domains, including education, programming, artificial intelligence, data science, mathematics, science, technology, career development, business, creativity, and everyday problem-solving.

You must understand the user's intention and provide relevant, accurate, helpful, and logically structured answers.

You are not restricted to any particular subject.

Understand contextual references and maintain conversation continuity.

When users ask follow-up questions, use the relevant previous conversation history.

When explaining difficult concepts, use step-by-step explanations and practical examples.

When generating code, provide complete, logically correct, and properly formatted implementations whenever requested.

When solving mathematical problems, show formulas, calculations, and intermediate steps when appropriate.

When discussing data science, provide technically meaningful explanations and practical Python implementations.

Respect the user's requested language, response length, formatting, and level of technical detail.

Do not unnecessarily repeat explanations.

Never invent facts, sources, statistics, API responses, or real-time information.

Clearly communicate uncertainty when information cannot be verified.

For current or time-sensitive questions, do not pretend to have live internet access unless an actual search or retrieval tool is integrated.

For health, legal, and financial topics, provide responsible general information and explain relevant limitations.

Do not refuse ordinary questions simply because they are outside a particular category.

Your goal is to provide an intelligent, reliable, versatile, and user-friendly AI experience.

FORMATTING & INTERACTION GUIDELINES:
1. Mathematical Expressions: Format mathematical equations, variables, and formulas using standard LaTeX math syntax: inline math with $...$ (e.g. $E = mc^2$ or $\\mu = \\frac{1}{N}\\sum_{i=1}^{N} x_i$) and block math with $$...$$.
2. Code Blocks: Always specify the programming language in fenced code blocks (e.g. ```python, ```javascript, ```sql). When requested to write code, provide complete, runnable code rather than truncated fragments.
3. Suggested Follow-Up Questions: At the very end of your response, provide 2 to 3 intelligent, concise, relevant follow-up questions that the user might want to explore next. Format them strictly as a JSON list wrapped inside the tag <followups>["Question 1?", "Question 2?", "Question 3?"]</followups>.
4. MANDATORY PROFESSIONAL & FORMAL COMMUNICATION PROTOCOL:
OmniMind AI operates as an academic and professional assistant for MSc Data Science and higher education.
- If the user asks in non-formal words, informal slang, street vernacular, texting shorthand (e.g., "yo", "bro", "bruh", "sup", "gimme", "wazzup", "dude", "lol", "idk", "wat is dis", "send me fast"), broken or ungrammatical slang words:
  DO NOT fulfill the answer using casual conversational tone.
  INSTEAD, politely ask and guide the user to ask in a formal or professional way.
  Format your reply strictly as follows:
  ### 📋 Communication Standard: Professional Phrasing Required

  Please present your query in a formal and professional manner so I can provide you with a high-caliber academic response.

  **Suggested Professional Rephrasing:**
  *"[Insert polished, professional, formal academic version of the user's intent]"*

  *Please feel free to ask using the formal phrasing above.*

  Include the suggested professional question inside the `<followups>["..."]</followups>` tag so the user can proceed with a single click.
- When the query is presented in clear, polite, formal, and professional language, provide a comprehensive, rigorous, and logically structured response.
"""

class GeminiService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.primary_model = model or config.GEMINI_MODEL
        self._client: Optional[genai.Client] = None
        if self.api_key:
            self._init_client()

    def _init_client(self):
        try:
            self._client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Client: {e}")
            self._client = None

    def update_api_key(self, api_key: str):
        self.api_key = api_key
        self._init_client()

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 0 and self._client is not None)

    def build_system_instruction(self, style: Optional[str] = None, language: Optional[str] = None, custom_instructions: Optional[str] = None) -> str:
        prompt = BASE_SYSTEM_INSTRUCTION
        modifiers = []

        if style:
            s = style.lower()
            if s == "beginner":
                modifiers.append("EXPLANATION STYLE: Beginner-friendly. Use clear intuitive analogies, simple language, and avoid unexplained jargon.")
            elif s in ("technical", "advanced"):
                modifiers.append("EXPLANATION STYLE: Advanced technical. Assume domain expertise, use rigorous technical terminology, and cover architectural considerations.")
            elif s == "step-by-step":
                modifiers.append("EXPLANATION STYLE: Step-by-step. Break the answer down into distinct numbered sequential steps with clear rationale.")
            elif s == "concise":
                modifiers.append("EXPLANATION STYLE: Concise. Keep response brief, high-density, directly answering the prompt without unnecessary preamble.")
            elif s == "detailed":
                modifiers.append("EXPLANATION STYLE: Highly detailed. Provide comprehensive coverage, practical examples, background, edge cases, and best practices.")
            elif s == "code-only":
                modifiers.append("EXPLANATION STYLE: Code-first / Code-only. Provide direct, fully commented, production-ready code with minimal surrounding prose.")
            elif s == "example-based":
                modifiers.append("EXPLANATION STYLE: Example-driven. Anchor every concept with concrete, practical, real-world examples.")
            elif s == "table-based":
                modifiers.append("EXPLANATION STYLE: Structured tabular. Utilize markdown comparison tables wherever comparing concepts, pros/cons, or features.")

        if language and language.strip().lower() not in ("english", "auto", ""):
            modifiers.append(f"LANGUAGE DIRECTIVE: Respond fluently and accurately in {language.strip()}.")

        if custom_instructions and custom_instructions.strip():
            modifiers.append(f"USER SPECIFIC INSTRUCTION: {custom_instructions.strip()}")

        if modifiers:
            prompt += "\n\nACTIVE CUSTOMIZATION DIRECTIVES:\n" + "\n".join(f"- {m}" for m in modifiers)

        return prompt

    def _prepare_contents(self, history: List[Dict[str, str]], current_message: str) -> List[types.Content]:
        contents: List[types.Content] = []
        for msg in history:
            role = msg.get("role")
            content_text = msg.get("content", "")
            if not content_text:
                continue
            gemini_role = "user" if role == "user" else "model"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content_text)]
                )
            )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=current_message)]
            )
        )
        return contents

    def _extract_followups(self, text: str) -> Tuple[str, List[str]]:
        pattern = r"<followups>(.*?)</followups>"
        match = re.search(pattern, text, re.DOTALL)
        followups = []
        clean_text = text
        if match:
            raw_json = match.group(1).strip()
            clean_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, list):
                    followups = [str(q).strip() for q in parsed if q]
            except Exception:
                pass
        return clean_text, followups

    def generate_chat_response(
        self,
        history: List[Dict[str, str]],
        message: str,
        style: Optional[str] = None,
        language: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        preferred_model: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured. Please set GEMINI_API_KEY in .env or the settings panel.")

        system_instruction = self.build_system_instruction(style, language, custom_instructions)
        contents = self._prepare_contents(history, message)

        models_to_try = [preferred_model or self.primary_model]
        for m in config.FALLBACK_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

        last_error = None
        for model_name in models_to_try:
            try:
                gen_config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                )
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config
                )
                raw_text = response.text or ""
                clean_text, followups = self._extract_followups(raw_text)
                return {
                    "content": clean_text,
                    "suggested_followups": followups,
                    "model": model_name
                }
            except (errors.ServerError, errors.ClientError) as e:
                logger.warning(f"Model {model_name} failed with error: {e}. Trying fallback if available.")
                last_error = e
                continue
            except Exception as e:
                last_error = e
                break

        raise last_error or RuntimeError("Failed to generate response with all available models.")

    def stream_chat_response(
        self,
        history: List[Dict[str, str]],
        message: str,
        style: Optional[str] = None,
        language: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        preferred_model: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        if not self.is_configured():
            yield {"type": "error", "error": "Gemini API key is not configured. Please set GEMINI_API_KEY in backend/.env"}
            return

        system_instruction = self.build_system_instruction(style, language, custom_instructions)
        contents = self._prepare_contents(history, message)

        models_to_try = [preferred_model or self.primary_model]
        for m in config.FALLBACK_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

        for model_name in models_to_try:
            has_yielded = False
            full_text = ""
            in_followup_tag = False
            buffered_tag = ""
            try:
                gen_config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                )
                stream = self._client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=gen_config
                )

                for chunk in stream:
                    if chunk.text:
                        chunk_text = chunk.text
                        full_text += chunk_text

                        # Prevent streaming raw <followups> block to UI
                        if "<followups>" in chunk_text or "<followup" in chunk_text:
                            in_followup_tag = True

                        if not in_followup_tag:
                            yield {"type": "token", "content": chunk_text}
                            has_yielded = True
                        else:
                            buffered_tag += chunk_text

                clean_text, followups = self._extract_followups(full_text)
                yield {
                    "type": "done",
                    "content": clean_text,
                    "suggested_followups": followups,
                    "model": model_name
                }
                return
            except (errors.ServerError, errors.ClientError) as e:
                logger.warning(f"Model {model_name} streaming error: {str(e)[:120]}. Attempting fallback...")
                if has_yielded:
                    yield {"type": "error", "error": "Stream interrupted due to model rate limit."}
                    return
                continue
            except Exception as e:
                logger.error(f"Unexpected streaming error on {model_name}: {e}")
                if has_yielded:
                    yield {"type": "error", "error": f"Generation interrupted: {str(e)}"}
                    return
                continue

        yield {
            "type": "error",
            "error": "Gemini API daily request quota reached for current free tier. Please enter your personal Google AI Studio API key in Settings (cog icon) to continue unlimited conversations."
        }

    def formalize_prompt(self, text: str) -> str:
        """Convert informal / casual draft text into formal, professional academic phrasing."""
        if not self.is_configured():
            return text
        instruction = "You are a professional academic editor. Rewrite the following user draft into a single clear, formal, polite, and grammatically impeccable English question suitable for an MSc Data Science assistant. Return ONLY the rewritten question without any explanation, quotes, or conversational filler."
        
        models_to_try = [self.primary_model] + config.FALLBACK_MODELS
        for model_name in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Content(role="user", parts=[types.Part.from_text(text=f"Draft: {text}")])
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=instruction,
                        temperature=0.3
                    )
                )
                return (response.text or text).strip().strip('"').strip("'")
            except Exception as e:
                logger.warning(f"Formalize attempt with {model_name} failed: {e}")
                continue
        return text

# Global singleton service
gemini_service = GeminiService()
