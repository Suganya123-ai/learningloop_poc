"""
Ollama LLM client for ITOps agent decision-making.

Calls a locally-running Ollama server (default: http://localhost:11434).
Builds a structured prompt from the observation, extracts one valid action
from the response, and returns None on any error so the caller can degrade
to the Q-table without crashing.

v2 changes:
  - MTTR (avg resolution steps) shown per action in evidence rows
  - Conclusive evidence (≥5 attempts, ≥80% success) injected into system
    prompt at highest authority, not only the user turn
  - Mandatory prohibition language replaces soft recommendation
"""
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Optional

from agent.environment import ACTIONS
from telemetry.langfuse_client import trace_llm_call

logger = logging.getLogger("itops.llm")

_SYSTEM_PROMPT = """\
You are an ITOps agent responsible for resolving IT system incidents.
You will be given the current state of a system and must select exactly
ONE action from the allowed list.

Allowed actions:
- no_op            : do nothing
- restart_service  : restart the affected service process
- scale_up         : add more compute or replica capacity
- scale_down       : reduce over-provisioned capacity
- rollback         : revert to the last known-good deployment
- investigate      : gather diagnostic data before acting
- clear_disk       : remove logs or temp files to free disk space
- escalate         : hand off to a senior engineer

RULE: If the EVIDENCE block below shows one action resolved ≥ 80% of
identical incidents with ≥ 5 real attempts, you MUST select that action.
This is measured data from the live environment — it overrides your prior
knowledge. Selecting any other action is incorrect.

Output format — two lines only:
<one-line reason>
ACTION: <action_name>
"""

_SYSTEM_PROMPT_WITH_OVERRIDE = """\
You are an ITOps agent responsible for resolving IT system incidents.
You will be given the current state of a system and must select exactly
ONE action from the allowed list.

Allowed actions:
- no_op            : do nothing
- restart_service  : restart the affected service process
- scale_up         : add more compute or replica capacity
- scale_down       : reduce over-provisioned capacity
- rollback         : revert to the last known-good deployment
- investigate      : gather diagnostic data before acting
- clear_disk       : remove logs or temp files to free disk space
- escalate         : hand off to a senior engineer

╔══════════════════════════════════════════════════════════════════╗
║  MANDATORY OVERRIDE — REAL SESSION DATA                         ║
║  Action: {action:<17}  Success: {successes}/{attempts} = {rate:.0%}       ║
║  Avg resolution: {mttr}                                   ║
║  You MUST select '{action}'. Do NOT choose any other action.    ║
╚══════════════════════════════════════════════════════════════════╝

Output format — two lines only:
<one-line reason>
ACTION: <action_name>
"""

_USER_TEMPLATE = """\
Incident:
  Alert      : {alert}
  Service    : {service_status}
  Load       : {load}
  CPU usage  : {cpu_pct}%
  Memory     : {mem_pct}%
  Prev action: {prev_action}
  Scenario   : {scenario_name}
  Context    : {scenario_desc}

{evidence_block}Select the best action.
"""

_EVIDENCE_BLOCK = """\
══ EVIDENCE FROM {n_total} PAST INCIDENTS ══
{rows}
{recommendation}
════════════════════════════════════════════

"""

_EVIDENCE_ROW = "  {action:<17}: {successes}/{attempts} resolved ({rate:.0%})  avg {mttr}{marker}"
_NO_EVIDENCE  = "  (No past evidence yet — use domain knowledge)\n\n"

# Threshold for moving evidence into system prompt (highest model priority)
_STRONG_ATTEMPTS = 5
_STRONG_RATE     = 0.80


class OllamaClient:
    """
    Thin HTTP client for Ollama's /api/chat endpoint.

    Parameters
    ----------
    model       : Ollama model tag, e.g. "adept-it:latest", "qwen2.5:7b-instruct"
    base_url    : Ollama server URL (default: http://localhost:11434)
    timeout_s   : per-request timeout in seconds
    temperature : sampling temperature (0.0 = deterministic, 1.0 = creative)
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: int = 30,
        temperature: float = 0.1,
    ):
        self.model = model
        self._url = f"{base_url}/api/chat"
        self._timeout = timeout_s
        self._temperature = temperature
        self._call_count = 0
        self._fallback_count = 0
        self._override_injected = 0
        # Pending LangFuse trace fields (set after decide(), flushed by record_outcome)
        self._lf_obs = self._lf_sys = self._lf_prompt = self._lf_response = self._lf_action = None

    # ------------------------------------------------------------------
    def decide(self, observation: dict, qtable_hint: Optional[dict] = None) -> Optional[str]:
        """
        Ask the LLM which action to take.

        qtable_hint carries real evidence from the RL success tracker:
          {"evidence": [{action, attempts, successes, rate, avg_mttr}, ...], "n_updates": int}

        When evidence is strong, it is injected into the system prompt
        (highest model authority) in addition to the user turn.
        """
        system_prompt, evidence_block = self._build_prompts(qtable_hint)

        prompt = _USER_TEMPLATE.format(
            alert          = observation.get("alert", "UNKNOWN"),
            service_status = observation.get("service_status", "UNKNOWN"),
            load           = observation.get("load", "UNKNOWN"),
            cpu_pct        = observation.get("cpu_pct", 0),
            mem_pct        = observation.get("mem_pct", 0),
            prev_action    = observation.get("state_key", ("", "", "", "no_op"))[-1],
            scenario_name  = observation.get("scenario_name", ""),
            scenario_desc  = observation.get("scenario_desc", ""),
            evidence_block = evidence_block,
        )

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": self._temperature},
        }).encode()

        try:
            req = urllib.request.Request(
                self._url,
                data    = payload,
                headers = {"Content-Type": "application/json"},
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())

            raw_text = body["message"]["content"]
            action   = self._extract_action(raw_text)
            self._call_count += 1

            if action:
                logger.info(
                    "LLM decided: %s  (model=%s, ep=%s, step=%s)",
                    action, self.model,
                    observation.get("episode"), observation.get("step"),
                )
                logger.debug("LLM raw response: %s", raw_text.strip())
                # ── LangFuse trace (non-blocking, fails silently) ──────
                self._lf_obs      = observation
                self._lf_sys      = system_prompt
                self._lf_prompt   = prompt
                self._lf_response = raw_text
                self._lf_action   = action
                return action

            logger.warning("LLM response had no valid action — falling back. raw=%r", raw_text)
            self._fallback_count += 1
            return None

        except urllib.error.URLError as exc:
            logger.warning("Ollama unreachable (%s) — falling back to Q-table.", exc)
            self._fallback_count += 1
            return None
        except (KeyError, json.JSONDecodeError) as exc:
            logger.warning("Ollama response parse error (%s) — falling back.", exc)
            self._fallback_count += 1
            return None

    # ------------------------------------------------------------------
    def _build_prompts(self, qtable_hint: Optional[dict]) -> tuple[str, str]:
        """
        Returns (system_prompt, evidence_block_for_user_turn).

        When the best evidenced action is conclusive, it is embedded in the
        system prompt (highest model authority) via a mandatory override box.
        The user-turn evidence block is always included for transparency.
        """
        if not qtable_hint:
            return _SYSTEM_PROMPT, _NO_EVIDENCE

        evidence = qtable_hint.get("evidence", [])
        n_total  = qtable_hint.get("n_updates", 0)
        tried    = [e for e in evidence if e["attempts"] > 0]

        if not tried:
            return _SYSTEM_PROMPT, _NO_EVIDENCE

        best = tried[0]
        is_strong = (
            best["attempts"] >= _STRONG_ATTEMPTS
            and best["rate"]  >= _STRONG_RATE
        )

        # ── System prompt ──────────────────────────────────────────────
        if is_strong:
            mttr_str = (
                f"{best['avg_mttr']:.1f} steps" if best["avg_mttr"] else "N/A"
            )
            system_prompt = _SYSTEM_PROMPT_WITH_OVERRIDE.format(
                action    = best["action"],
                successes = best["successes"],
                attempts  = best["attempts"],
                rate      = best["rate"],
                mttr      = mttr_str.ljust(10),
            )
            self._override_injected += 1
        else:
            system_prompt = _SYSTEM_PROMPT

        # ── User-turn evidence block ───────────────────────────────────
        best_rate = tried[0]["rate"]
        rows = []
        for e in tried:
            marker = " ← BEST" if e["rate"] == best_rate and e["rate"] > 0 else ""
            mttr_str = (
                f"{e['avg_mttr']:.1f} steps" if e["avg_mttr"] else "N/A        "
            )
            rows.append(_EVIDENCE_ROW.format(
                action    = e["action"],
                successes = e["successes"],
                attempts  = e["attempts"],
                rate      = e["rate"],
                mttr      = mttr_str,
                marker    = marker,
            ))

        if is_strong:
            rec = (
                f"MANDATORY: You MUST select '{best['action']}'. "
                f"Real data: {best['successes']}/{best['attempts']} resolved "
                f"({best['rate']:.0%}). Selecting any other action is wrong."
            )
        elif best["attempts"] >= 2 and best["rate"] >= 0.60:
            rec = (
                f"STRONG RECOMMENDATION: Use '{best['action']}' "
                f"({best['successes']}/{best['attempts']} = {best['rate']:.0%} resolved)."
            )
        else:
            rec = "Insufficient evidence — use domain knowledge."

        evidence_block = _EVIDENCE_BLOCK.format(
            n_total        = n_total,
            rows           = "\n".join(rows),
            recommendation = rec,
        )
        return system_prompt, evidence_block

    # ------------------------------------------------------------------
    def record_outcome(self, optimal_action: str, resolved: bool, reward: float):
        """Call this after env.step() so LangFuse gets the outcome alongside the prompt."""
        if self._lf_action is None:
            return
        obs = self._lf_obs or {}
        try:
            trace_llm_call(
                loop           = obs.get("episode", 0),
                step           = obs.get("step", 0),
                scenario       = obs.get("scenario_name", ""),
                alert_type     = obs.get("alert", "UNKNOWN"),
                system_prompt  = self._lf_sys or "",
                user_prompt    = self._lf_prompt or "",
                response       = self._lf_response or "",
                action_chosen  = self._lf_action,
                optimal_action = optimal_action,
                resolved       = resolved,
                reward         = reward,
                model          = self.model,
            )
        finally:
            self._lf_obs = self._lf_sys = self._lf_prompt = self._lf_response = self._lf_action = None

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            url = self._url.replace("/api/chat", "/api/tags")
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            return False

    def stats(self) -> dict:
        return {
            "model":             self.model,
            "llm_calls":         self._call_count,
            "fallbacks":         self._fallback_count,
            "override_injected": self._override_injected,
            "llm_rate": round(
                self._call_count / max(1, self._call_count + self._fallback_count), 3
            ),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_action(text: str) -> Optional[str]:
        match = re.search(r"ACTION\s*:\s*(\w+)", text, re.IGNORECASE)
        if match:
            candidate = match.group(1).lower().strip()
            if candidate in ACTIONS:
                return candidate

        for line in reversed(text.strip().splitlines()[-3:]):
            token = line.strip().lower()
            if token in ACTIONS:
                return token

        for action in ACTIONS:
            if re.search(rf"\b{action}\b", text, re.IGNORECASE):
                return action

        return None
