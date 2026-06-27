# -*- coding: utf-8 -*-
"""
Structured agent-activity logging — the "agents in action" feed.

Every agent (orchestrator, classifier, adversary, the LLM, the report writer)
narrates what it is doing through `say(agent, msg)`. Each call prints ONE line
with a machine-parseable prefix:

    [AGENT:<key>] <message>

The line stays human-readable in the raw console log, and the Streamlit
dashboard parses the `[AGENT:<key>]` prefix to route the message into a live,
per-agent activity panel (see unified_app.py). Keeping the convention in one
place means the producer (the pipeline) and the consumer (the UI) never drift.

This module has no third-party dependencies and prints with flush=True so the
feed appears line-by-line while a run streams over a subprocess pipe.
"""

# Canonical agent keys. The UI maps these to icons/colours; keep them in sync.
ORCHESTRATOR = "orchestrator"
CLASSIFIER   = "classifier"
ADVERSARY    = "adversary"
LLM          = "llm"
REPORT       = "report"

VALID_AGENTS = {ORCHESTRATOR, CLASSIFIER, ADVERSARY, LLM, REPORT}


def say(agent: str, msg: str) -> None:
    """Emit a single structured activity line for `agent`.

    `agent` should be one of the canonical keys above; an unknown key is still
    printed (the UI shows it under a generic icon) so callers never crash.

    Hardened against console encoding: the messages use a few Unicode glyphs
    (λ, →, ≥, …). On a non-UTF-8 console (Windows cp1252) that would raise
    UnicodeEncodeError; we fall back to an ASCII-safe rendering so logging can
    never take down a run.
    """
    line = f"[AGENT:{agent}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


# Thin role-named wrappers so call sites read naturally.
def orchestrator(msg: str) -> None: say(ORCHESTRATOR, msg)
def classifier(msg: str) -> None:   say(CLASSIFIER, msg)
def adversary(msg: str) -> None:    say(ADVERSARY, msg)
def llm(msg: str) -> None:          say(LLM, msg)
def report(msg: str) -> None:       say(REPORT, msg)
