"""Typed fail-closed error primitives for HoloSphere (zero bare assert statements)."""

from __future__ import annotations


class HoloSphereError(ValueError):
    """Fail-closed invariant, shape, or algebraic contract violation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


HoloSphereValueError = HoloSphereError


def require_ok(cond: bool, message: str, *, code: str = "HOLOSPHERE_INVARIANT") -> None:
    """Assertion-free guard safe under python3 -O."""
    if cond is not True:
        raise HoloSphereError(code, message)
