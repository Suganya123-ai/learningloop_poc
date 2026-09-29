"""
LangFuse v4 tracing client — traces every LLM call and CEO Gate decision.
Visible at http://localhost:5700  (project: learningloop_agenticai)
"""
import os
import logging

logger = logging.getLogger("itops.langfuse")

LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST",       "http://localhost:5700")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY",  "pk-lf-42023f91-b0d9-4b8e-b8d9-6628d4cb8600")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY",  "sk-lf-5af0cf5b-7e08-4935-9b89-d64f3fdcf48d")

_lf = None

def _get_client():
    global _lf
    if _lf is not None:
        return _lf
    try:
        from langfuse import Langfuse
        _lf = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        _lf.auth_check()
        logger.info("LangFuse connected at %s", LANGFUSE_HOST)
    except Exception as e:
        logger.warning("LangFuse unavailable — tracing disabled: %s", e)
        _lf = None
    return _lf


def trace_llm_call(
    loop, step, scenario, alert_type,
    system_prompt, user_prompt, response,
    action_chosen, optimal_action,
    resolved, reward, model,
):
    lf = _get_client()
    if lf is None:
        return
    try:
        trace_id = lf.create_trace_id()
        with lf.start_as_current_observation(
            type="generation",
            trace_id=trace_id,
            name="llm-decision",
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            output=response,
            metadata={
                "loop": loop, "step": step,
                "scenario": scenario, "alert_type": alert_type,
                "action_chosen": action_chosen, "optimal_action": optimal_action,
            },
        ):
            lf.score_current_trace(
                name="accuracy",
                value=1.0 if action_chosen == optimal_action else 0.0,
                comment=f"chosen={action_chosen}  optimal={optimal_action}",
            )
            lf.score_current_trace(
                name="resolved",
                value=1.0 if resolved else 0.0,
            )
            lf.score_current_trace(name="reward", value=reward)
    except Exception as e:
        logger.debug("LangFuse trace_llm_call failed: %s", e)


def trace_ceo_override(
    loop, step, scenario, alert_type, state_key,
    action, optimal_action, trust_score,
    n_attempts, n_success, resolved, reward,
):
    lf = _get_client()
    if lf is None:
        return
    try:
        trace_id = lf.create_trace_id()
        with lf.start_as_current_observation(
            type="span",
            trace_id=trace_id,
            name="ceo-gate-override",
            input={"state_key": state_key, "action": action},
            output={"resolved": resolved, "reward": reward},
            metadata={
                "loop": loop, "step": step,
                "scenario": scenario, "alert_type": alert_type,
                "trust_score": trust_score,
                "n_attempts": n_attempts, "n_success": n_success,
                "optimal_action": optimal_action,
            },
        ):
            lf.score_current_trace(name="trust_score",  value=trust_score)
            lf.score_current_trace(name="resolved",     value=1.0 if resolved else 0.0)
            lf.score_current_trace(
                name="accuracy",
                value=1.0 if action == optimal_action else 0.0,
                comment=f"chosen={action}  optimal={optimal_action}",
            )
            lf.score_current_trace(name="reward", value=reward)
    except Exception as e:
        logger.debug("LangFuse trace_ceo_override failed: %s", e)


def flush():
    lf = _get_client()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass
