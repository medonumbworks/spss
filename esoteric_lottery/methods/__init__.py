"""Esoteric number-generation methods. Each module documents its algorithm and sources."""
from .chaldean import ChaldeanMethod
from .gematria import GematriaMethod
from .iching import IChingMethod
from .lenormand import LenormandMethod
from .meihua import MeihuaMethod
from .pythagorean import PythagoreanMethod
from .tarot import TarotMethod
from .vedic_astrology import VedicAstrologyMethod
from .western_astrology import WesternAstrologyMethod

REGISTRY = {cls.key: cls for cls in (
    MeihuaMethod, IChingMethod, WesternAstrologyMethod, VedicAstrologyMethod, PythagoreanMethod,
    ChaldeanMethod, GematriaMethod, TarotMethod, LenormandMethod,
)}


def build_methods(cfg: dict) -> list:
    seed = int(cfg["seed"])
    enabled = cfg.get("methods", {})
    out = []
    for key, cls in REGISTRY.items():
        params = enabled.get(key, {}) or {}
        if params.get("enabled", True):
            out.append(cls(params, seed))
    return out
