---
trigger: always_on
---

## Code Quality & Error Handling Standards

**Core Principle: Fail Fast & Trust Contracts.**

1.  **Eliminate Defensive Noise**:

    - **NEVER** use `try...except Exception: pass` or return `None` silently. Let execution fail on logical errors.
    - **TRUST** type hints. If a function returns `pd.DataFrame`, do not check `if df is not None`.
    - **REJECT** nesting. Use **Guard Clauses** (`if not valid: return`) to keep the "Happy Path" at the root indentation level.

2.  **EAFP vs LBYL**:
    - **Use EAFP** (Try/Except) **ONLY** for volatile I/O operations (Files, Network, APIs).
    - **Use LBYL** (If condition) for predictable logic where mapped keys or attributes _should_ exist by design.
