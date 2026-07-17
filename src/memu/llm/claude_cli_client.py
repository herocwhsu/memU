from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The CLI has no max_tokens control; kept for interface parity with the other
# clients so callers can pass it without special-casing this backend.
_DEFAULT_TIMEOUT_SECONDS = 300

# Claude Code's default system prompt steers answers toward markdown-formatted
# coding assistance, which breaks callers that put strict output-format specs
# in the user prompt (memU's extraction expects bare XML). Always replace it;
# use this neutral prompt when the caller doesn't supply one.
_NEUTRAL_SYSTEM_PROMPT = (
    "You are a text-processing engine. Follow the output format requested in "
    "the user's message exactly. Output only the requested format, with no "
    "preamble, commentary, or markdown fences."
)


class ClaudeCLIClient:
    """Claude LLM client backed by the Claude Code CLI (``claude --print``).

    Uses a Claude subscription session (``CLAUDE_CODE_OAUTH_TOKEN``) instead of
    an Anthropic API key, for hosts where unattended jobs already authenticate
    through Claude Code. Mirrors the surface of
    :class:`memu.llm.anthropic_client.AnthropicClient` so it can be wrapped by
    :class:`memu.llm.wrapper.LLMClientWrapper`. Text-only: vision and
    transcription raise.

    ``api_key`` doubles as the OAuth token; when empty, the token is read from
    ``CLAUDE_CODE_OAUTH_TOKEN`` in the environment or ``~/.claude-code-oauth-token``.
    The CLI binary is resolved from ``CLAUDE_CLI_PATH`` or ``PATH``.
    """

    def __init__(
        self,
        *,
        chat_model: str,
        api_key: str = "",
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    ):
        self.chat_model = chat_model
        self._token = api_key or ""
        self.timeout_seconds = timeout_seconds
        self._cli = os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude")
        if not self._cli:
            msg = "claude CLI not found: set CLAUDE_CLI_PATH or add 'claude' to PATH"
            raise ValueError(msg)
        # Neutral cwd so the CLI never picks up a project's CLAUDE.md context.
        self._cwd = tempfile.gettempdir()

    def _resolve_token(self) -> str:
        if self._token:
            return self._token
        env_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
        if env_token:
            return env_token
        token_file = Path.home() / ".claude-code-oauth-token"
        if token_file.is_file():
            return token_file.read_text().strip()
        msg = "No Claude subscription token: set api_key, CLAUDE_CODE_OAUTH_TOKEN, or ~/.claude-code-oauth-token"
        raise ValueError(msg)

    async def _run(self, prompt: str, system_prompt: str | None) -> tuple[str, dict[str, Any]]:
        argv = [
            self._cli,
            "--print",
            "--model",
            self.chat_model,
            "--system-prompt",
            system_prompt if system_prompt is not None else _NEUTRAL_SYSTEM_PROMPT,
        ]
        env = {**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": self._resolve_token()}
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self._cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=self.timeout_seconds
            )
        except TimeoutError:
            proc.kill()
            msg = f"claude CLI timed out after {self.timeout_seconds}s"
            raise RuntimeError(msg) from None
        raw: dict[str, Any] = {
            "returncode": proc.returncode,
            "stderr": stderr.decode(errors="replace")[-2000:],
        }
        if proc.returncode != 0:
            msg = f"claude CLI failed (exit {proc.returncode}): {raw['stderr'][:500]}"
            raise RuntimeError(msg)
        text = stdout.decode(errors="replace").strip()
        logger.debug("claude CLI response: %s", text[:500])
        return text, raw

    async def chat(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,  # noqa: ARG002 — CLI does not expose max_tokens
        system_prompt: str | None = None,
        temperature: float = 0.2,  # noqa: ARG002 — CLI does not expose temperature
    ) -> tuple[str, dict[str, Any]]:
        """Generic chat completion via ``claude --print``."""
        return await self._run(prompt, system_prompt)

    async def summarize(
        self,
        text: str,
        *,
        max_tokens: int | None = None,  # noqa: ARG002 — CLI does not expose max_tokens
        system_prompt: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        system = system_prompt or "Summarize the text in one short paragraph."
        return await self._run(text, system)

    async def vision(
        self,
        prompt: str,
        image_path: str,
        *,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, Any]:
        msg = "claude_cli backend is text-only; use the 'anthropic' backend for vision."
        raise NotImplementedError(msg)

    async def transcribe(
        self,
        audio_path: str,
        *,
        prompt: str | None = None,
        language: str | None = None,
        response_format: str = "text",
    ) -> tuple[str, None]:
        msg = "claude_cli backend does not provide audio transcription."
        raise NotImplementedError(msg)
