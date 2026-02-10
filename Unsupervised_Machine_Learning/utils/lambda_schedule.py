import re
import math
from typing import Callable, Dict, Any

LambdaFn = Callable[[float, int, int], float]  # (progress, step, epoch) -> lambda


def _clamp01(x: float) -> float:
    """Clamp x into [0, 1]."""
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def parse_lambda_spec(spec: Any) -> LambdaFn:
    """
    Parse a lambda schedule specification and return a callable.

    Parameters
    ----------
    spec:
        Supported forms:
        - float / int:
            Constant value (always returns this number).
        - str:
            Function-like strings, e.g.
              - "const(1e-2)"
              - "step(1e-2,0,at=0.3)"                  # progress < at -> a else b
              - "linear(1e-2,0,start=0.0,end=0.5,warmup=0.1)"
              - "cosine(1e-2,0,start=0.0,end=1.0,warmup=0.0)"
        - dict:
            Optional structured form for future extensions, e.g.
              {"type":"linear","a":1e-2,"b":0.0,"start":0.0,"end":0.5,"warmup":0.1}

    Returns
    -------
    LambdaFn:
        A function (progress, step, epoch) -> lambda (float).

    Reproducibility / semantics notes
    --------------------------------
    - This module is deterministic: no RNG usage.
    - `progress` is assumed to be a normalized training progress in [0,1].
      If the caller provides values outside [0,1], this module clamps internally.
    - `step` and `epoch` are accepted for convenience / future use, but the current
      built-in schedules depend only on `progress` (except they are passed through).
    """
    if isinstance(spec, (int, float)):
        val = float(spec)
        return lambda progress, step, epoch: val

    if isinstance(spec, dict):
        # Dict form is optional. Keeping it allows explicit configs without parsing strings.
        t = str(spec.get("type", "const")).lower()
        if t == "const":
            v = float(spec["value"])
            return lambda p, s, e: v
        if t == "linear":
            a = float(spec["a"])
            b = float(spec["b"])
            start = float(spec.get("start", 0.0))
            end = float(spec.get("end", 1.0))
            warmup = float(spec.get("warmup", 0.0))
            return _linear_fn(a, b, start=start, end=end, warmup=warmup)
        raise ValueError(f"Unknown lambda dict type: {t}")

    if not isinstance(spec, str):
        raise TypeError(f"lambda spec must be float/int/str/dict, got {type(spec)}")

    s = spec.strip().lower()

    # const(x)
    m = re.fullmatch(r"const\(([^)]+)\)", s)
    if m:
        v = float(m.group(1))
        return lambda progress, step, epoch: v

    # step(a,b,at=0.3): progress<at -> a else b
    m = re.fullmatch(r"step\(([^,]+),([^,]+),\s*at=([^)]+)\)", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        at = float(m.group(3))
        return lambda p, step, epoch: a if p < at else b

    # linear(a,b, start=..., end=..., warmup=...)
    m = re.fullmatch(r"linear\(([^,]+),([^,]+)(?:,(.*))?\)", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        kwargs = _parse_kwargs(m.group(3))
        return _linear_fn(a, b, **kwargs)

    # cosine(a,b, start=..., end=..., warmup=...)
    m = re.fullmatch(r"cosine\(([^,]+),([^,]+)(?:,(.*))?\)", s)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        kwargs = _parse_kwargs(m.group(3))
        return _cosine_fn(a, b, **kwargs)

    raise ValueError(f"Invalid lambda spec: {spec}")


def _parse_kwargs(argstr: str | None) -> Dict[str, float]:
    """
    Parse keyword arguments part of a schedule string.

    Example
    -------
      "start=0.0,end=0.5,warmup=0.1" -> {"start":0.0, "end":0.5, "warmup":0.1}
    """
    out: Dict[str, float] = {}
    if not argstr:
        return out

    parts = [p.strip() for p in argstr.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Invalid kwarg: {p}")
        k, v = p.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def _linear_fn(
    a: float,
    b: float,
    start: float = 0.0,
    end: float = 1.0,
    warmup: float = 0.0
) -> LambdaFn:
    """
    Linear interpolation schedule from a -> b over progress in [start, end].

    Behavior
    --------
    - If progress <= start: returns a
    - If progress >= end:   returns b
    - Else: linear interpolation between a and b
    - Warmup option:
        If warmup > 0, then for progress < start + warmup, returns a (flat),
        and interpolation begins after that point.

    Notes
    -----
    - progress is clamped into [0,1] internally.
    """
    start = float(start)
    end = float(end)
    warmup = float(warmup)
    if end <= start:
        raise ValueError("linear: end must be > start")

    def fn(p: float, step: int, epoch: int) -> float:
        p = _clamp01(float(p))

        # Warmup: keep lambda fixed at 'a' for [start, start+warmup)
        if warmup > 0.0 and p < start + warmup:
            return a

        if p <= start:
            return a
        if p >= end:
            return b

        t = (p - start) / (end - start)
        return a + (b - a) * t

    return fn


def _cosine_fn(
    a: float,
    b: float,
    start: float = 0.0,
    end: float = 1.0,
    warmup: float = 0.0
) -> LambdaFn:
    """
    Cosine decay schedule from a -> b over progress in [start, end].

    Behavior
    --------
    - If progress <= start: returns a
    - If progress >= end:   returns b
    - Else: cosine interpolation (smooth decay) from a to b
    - Warmup option:
        If warmup > 0, then for progress < start + warmup, returns a (flat),
        and decay begins after that point.

    Notes
    -----
    - progress is clamped into [0,1] internally.
    """
    start = float(start)
    end = float(end)
    warmup = float(warmup)
    if end <= start:
        raise ValueError("cosine: end must be > start")

    def fn(p: float, step: int, epoch: int) -> float:
        p = _clamp01(float(p))

        if warmup > 0.0 and p < start + warmup:
            return a
        if p <= start:
            return a
        if p >= end:
            return b

        t = (p - start) / (end - start)  # 0..1
        # Cosine decay from a -> b
        w = 0.5 * (1.0 + math.cos(math.pi * t))
        return b + (a - b) * w

    return fn
