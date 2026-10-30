"""Pokrece selftest svih modula redom; pada na prvoj gresci."""
import importlib
import sys

MODULES = ["core", "numerology", "kamea", "iching", "cards", "astro", "calendars", "history", "probability", "score"]
for name in MODULES:
    m = importlib.import_module(name)
    print(f"===== {name} =====")
    m.selftest()
print("SVI SELFTESTOVI OK")
