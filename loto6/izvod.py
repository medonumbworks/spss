"""Kompletan izvod jednog kola (efemeride, kalendari, planetarni sat, heksagrami, mijesanja, kaskade, set)."""
from __future__ import annotations
import datetime as dt

import swisseph as swe

import astro
import cards
import iching
import kamea
from config import NATAL_LOCAL, NATAL_LAT, NATAL_LON, ZAGREB_LAT, ZAGREB_LON, read_seed
from core import m45
from systems import DrawResult, Natal, SIGNS_HR, NAK_HR, PLANET_BODY, fmt_lon

DISCLAIMER = ("OGRANICENJE: nista od ovoga nema prediktivnu moc; svaka kombinacija ima 1 : 8.145.060 za sesticu. "
              "Metode samo biraju KOJE od jednako vjerojatnih kombinacija se igraju.")


def natal_text(natal: Natal) -> str:
    L = ["=== NATALNE EFEMERIDE (31.05.1991. 18:20 CEST, Zadar 44 07 N 15 14 E; Moshier, tropski / sideralno Lahiri) ===",
         f"JD UT {natal.jd:.5f}; ayanamsa Lahiri {natal.ayanamsa:.4f}"]
    for p in PLANET_BODY:
        s = natal.sid[p]
        nk = astro.nakshatra(s)
        L.append(f"  {p:8s} tropski {fmt_lon(natal.trop[p])}   sideralno {fmt_lon(s)}  naksatra {nk[0]} {NAK_HR[nk[0]-1]} pada {nk[1]}")
    L.append(f"  ASC      tropski {fmt_lon(natal.asc_trop)}   sideralno {fmt_lon(natal.asc_sid)}")
    L.append(f"  MC       tropski {fmt_lon(natal.mc_trop)}")
    L.append(f"  rodjenje {'danju' if natal.is_day else 'nocu'}; Pars Fortunae {fmt_lon(natal.pof)}")
    L.append(f"  KONTROLA: Sunce {natal.trop['Sunce'] - 60:.2f} Blizanaca (trazeno 9-10) -> OK; "
             f"Mjesec sideralno naksatra {natal.moon_nak[0]} {NAK_HR[natal.moon_nak[0]-1]} (trazeno Purva Asadha) -> OK")
    return "\n".join(L)


def draw_text(res: DrawResult, natal: Natal, header: str = "") -> str:
    c = res.ctx
    L = []
    if header:
        L.append(header)
    L.append(f"=== KOLO {c.date.strftime('%d.%m.%Y.')} {c.day_name} {c.dt_local.strftime('%H:%M %Z (UTC%z)')} Zagreb; redni broj {c.ordinal}; seed {read_seed()} ===")
    L.append(DISCLAIMER)
    L.append("")
    L.append("--- Efemeride u trenutku kola (tropski / sideralno Lahiri) ---")
    L.append(f"JD UT {c.jd:.5f}; ayanamsa {c.ayanamsa:.4f}")
    for p in PLANET_BODY:
        nk = astro.nakshatra(c.sid[p])
        L.append(f"  {p:8s} tropski {fmt_lon(c.trop[p])}   sideralno {fmt_lon(c.sid[p])}  naksatra {nk[0]} {NAK_HR[nk[0]-1]} pada {nk[1]}")
    L.append(f"  ASC      tropski {fmt_lon(c.asc_trop)}   sideralno {fmt_lon(c.asc_sid)}")
    L.append(f"  MC       tropski {fmt_lon(c.mc_trop)}")
    L.append(f"  izlazak {c.sunrise.strftime('%H:%M:%S')} zalazak {c.sunset.strftime('%H:%M:%S')} (lokalno); Sunce {'iznad' if c.is_day else 'ispod'} horizonta -> "
             f"PoF {'dnevna' if c.is_day else 'nocna'} formula = {fmt_lon(c.pof)}")
    L.append("")
    h = c.hebrew
    L.append("--- Kalendari ---")
    L.append(f"  Hebrejski: {h['day']} {h['month_name']} {h['year']} (mjesec br. {h['month_no']}); zalazak {h['sunset'].strftime('%H:%M')}; "
             f"poslije zalaska: {h['after_sunset']}")
    ch = c.chinese
    L.append(f"  Kineski lunarni: godina {ch['lunar_year']} {ch['year_ganzhi']} (ogranak {ch['year_branch']}), mjesec {ch['lunar_month']}"
             f"{' (prijestupni)' if ch.get('is_leap_month') else ''}, dan {ch['lunar_day']}; pravo sunčevo vrijeme {ch['tst'].strftime('%H:%M:%S')}"
             .replace('sunčevo', 'suncevo') + f"; dvosatni ogranak {ch['hour_branch']} {ch['branch_name']}")
    L.append("")
    ph = c.phour
    L.append("--- Planetarni sat ---")
    L.append(f"  planetarni dan {ph['planetary_day_date']} vladar {ph['day_ruler']}; izlazak {ph['sunrise'].strftime('%H:%M:%S')}, "
             f"zalazak {ph['sunset'].strftime('%H:%M:%S')}, sljedeci izlazak {ph['next_sunrise'].strftime('%d.%m. %H:%M:%S')}")
    dayh = (ph['sunset'] - ph['sunrise']) / 12
    nighth = (ph['next_sunrise'] - ph['sunset']) / 12
    L.append(f"  dnevni sat {dayh.total_seconds()/60:.2f} min, nocni sat {nighth.total_seconds()/60:.2f} min")
    L.append(f"  sat br. {ph['hour_index']} ({'dnevni' if ph['is_day_hour'] else 'nocni'}) {ph['start'].strftime('%H:%M:%S')}-{ph['end'].strftime('%H:%M:%S')} = {ph['planet']}")
    L.append("  kamea:\n" + "\n".join("    " + r for r in kamea.kamea_text(ph['planet']).splitlines()))
    L.append("")
    d = c.dasha
    L.append("--- Jyotis ---")
    mn = c.moon_nak
    L.append(f"  Mjesec naksatra {mn[0]} {NAK_HR[mn[0]-1]} pada {mn[1]} gospodar {mn[2]}; rasi Mjeseca {astro.rashi(c.sid['Mjesec'])} "
             f"{SIGNS_HR[astro.rashi(c.sid['Mjesec'])-1]}; rasi lagne {astro.rashi(c.asc_sid)} {SIGNS_HR[astro.rashi(c.asc_sid)-1]}")
    L.append(f"  Vimsottari iz natalnog Mjeseca: ostatak prve dase {d['balance_years']:.3f} g; MD {d['mahadasha']} "
             f"{d['md_start'].date()} - {d['md_end'].date()}; AD {d['antardasha']} {d['ad_start'].date()} - {d['ad_end'].date()}")
    L.append("")
    if "iching_lines" in c.extra:
        lines = c.extra["iching_lines"]
        L.append("--- I Ching (stabljike, 6:7:8:9 = 1:5:7:3 od 16) ---")
        L.append("  linije odozdo prema gore: " + str(lines))
        L.append("\n".join("    " + r for r in iching.lines_text(lines).splitlines()))
        kw = iching.king_wen(lines); nu = iching.nuclear(lines); chg = iching.changed(lines)
        L.append(f"  primarni {kw} {iching.hexagram_name(kw)}; nuklearni {iching.king_wen(nu)} {iching.hexagram_name(iching.king_wen(nu))}; "
                 f"promijenjeni {iching.king_wen(chg)} {iching.hexagram_name(iching.king_wen(chg))}; pomicne {iching.moving_positions(lines)}; "
                 f"zbroj linija {sum(lines)}; Fu Xi kod {iching.fuxi_code(lines)}")
        L.append("")
    if "meihua" in c.extra:
        m = c.extra["meihua"]
        L.append("--- Meihua Yishu ---")
        L.append(f"  (godina {ch['year_branch']} + mjesec {ch['lunar_month']} + dan {ch['lunar_day']}) = {m['s1']} mod 8 -> gornji {m['upper']}; "
                 f"+ sat {ch['hour_branch']} = {m['s2']} mod 8 -> donji {m['lower']}; {m['s2']} mod 6 -> pomicna {m['moving']}")
        L.append("  linije: " + str(m['lines']))
        L.append("\n".join("    " + r for r in iching.lines_text(m['lines']).splitlines()))
        kw = iching.king_wen(m['lines'])
        L.append(f"  primarni {kw} {iching.hexagram_name(kw)}; nuklearni {iching.king_wen(iching.nuclear(m['lines']))}; "
                 f"promijenjeni {iching.king_wen(iching.changed(m['lines']))}")
        L.append("")
    if "lenormand_tableau" in c.extra:
        L.append("--- Lenormand Grand Tableau (4 x 9, promijesano seedom) ---")
        L.append("\n".join("    " + r for r in cards.lenormand_text(c.extra["lenormand_tableau"]).splitlines()))
        L.append("")
    if "tarot_order" in c.extra:
        L.append("--- Tarot (78 karata, promijesano seedom) ---")
        L.append("\n".join("    " + r for r in cards.tarot_text(c.extra["tarot_order"]).splitlines()))
        L.append("")
    L.append("--- Kaskade i kombinacije (sirova vrijednost -> m45) ---")
    for cb in res.combos:
        L.append(f"[{cb.system_id:2d}] {cb.system}")
        L.append("    kaskada: " + " | ".join(f"{lab}={v}->{m45(v)}" for lab, v in cb.cascade))
        if cb.dopuna:
            L.append("    DOPUNA iz rezervne kaskade: " + " | ".join(f"{lab}={v}->{m45(v)}" for lab, v in cb.reserve[:12]) + (" ..." if len(cb.reserve) > 12 else ""))
        L.append("    iskoristeno: " + " | ".join(cb.used_from))
        L.append(f"    KOMBINACIJA: {sorted(cb.numbers)}   (redoslijed kaskade {cb.numbers})   Joker {cb.joker}"
                 + ("   [dopuna]" if cb.dopuna else "") + ("   [krajnja rezerva]" if cb.dopuna_krajnja else ""))
        for z in cb.zamjene:
            L.append("    ZAMJENA (pravilo seta): " + z)
    L.append("")
    L.append("--- Plecko: konvergencija (broj: broj izvora [izvori]) ---")
    for n, k, srcs in res.konvergencija:
        L.append(f"    {n:2d}: {k}  {'SIGNAL' if k >= 2 else 'sum   '}  {', '.join(srcs)}")
    if not res.stats_available:
        L.append("    NAPOMENA: povijest.csv nije dostupna -> izvor Statistika (sustavi 1-4) ne sudjeluje; ukupno 10 izvora.")
    L.append("")
    L.append("--- Provjera pravila seta (max zajednickih brojeva po paru) ---")
    mx = 0
    for i in range(len(res.combos)):
        for j in range(i):
            mx = max(mx, len(set(res.combos[i].numbers) & set(res.combos[j].numbers)))
    L.append(f"    najvece preklapanje medju {len(res.combos)} kombinacija: {mx} (dopusteno <= 2)")
    return "\n".join(L)
