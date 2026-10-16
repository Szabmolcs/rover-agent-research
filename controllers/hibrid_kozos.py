#!/usr/bin/env python3
"""A hibrid mod kozos resze: a kezzel irt akadalykerules "makroja".

UGYANAZT hasznalja a tanito kornyezet (rl_env.py) es az eles vezerlo
(rl_controller.py), igy a halo tanitaskor pontosan ugyanazokat a
helyzeteket latja az akadalykerules UTAN, mint az eles futasban.
"""

from __future__ import annotations

from typing import Any

import controllers.baseline_line_follower as _baseline
from controllers.baseline_line_follower import (
    AKADALY_KUSZOB_BELEPES_M,
    Allapot,
    FutasStatisztika,
    akadaly_elol,
    egy_lepes_akadaly,
)
from controllers.rl_obs import Memoria, vonalon_van

MAKRO_MAX_ITERACIO = 150
KIFUTAS_MAX_LEPES = 0
KIFUTAS_KUSZOB_M = 0.45


def _szabadabb_oldal_elojele_javitott(observe_valasz: dict[str, Any]) -> int:
    """Javitott iranyvalasztas: a TELJES lidar-legyezot nezi, felosztva bal (0,1,2) es jobb (3,4,5)
    felre. Igy ha az akadaly kicsit a menetirany JOBB oldalan van (3-as szektor), balra kerulunk ki,
    es forditva. (Az eredeti csak a 0,1 vs 4,5 szektorokat hasonlitotta, ezert egyenlo eseten mindig
    jobbra fordult - akar az akadaly fele is.)"""
    sz = observe_valasz.get("lidar_szektor_min") or []
    if len(sz) < 6:
        return 1
    return 1 if min(sz[3:6]) >= min(sz[0:3]) else -1


# A baseline egy_lepes_akadaly() a modul-szintu szabadabb_oldal_elojele()-t hivja; azt cseréljuk,
# a baseline_line_follower.py fajlt NEM kell modositani.
_baseline.szabadabb_oldal_elojele = _szabadabb_oldal_elojele_javitott


class FelvevoKliens:
    """Atlatszo burkolo: tovabbitja a parancsokat, es megjegyzi a turn/move-okat,
    hogy a halo memoriaja a kezzel irt akadalykerules lepeseit is lassa."""

    def __init__(self, kliens: Any) -> None:
        self.kliens = kliens
        self.felvett: list[tuple[float, float]] = []  # (fordulas_fok, elorehaladas_m)
        self._fuggo_szog = 0.0

    def kuld(self, parancs: dict[str, Any]) -> dict[str, Any]:
        valasz = self.kliens.kuld(parancs)
        nev = parancs.get("command")
        if nev == "turn":
            self._fuggo_szog += float(parancs.get("angle_deg", 0.0))
        elif nev == "move":
            self.felvett.append((self._fuggo_szog, float(parancs.get("distance_m", 0.0))))
            self._fuggo_szog = 0.0
        return valasz

    def ures_felvett(self) -> list[tuple[float, float]]:
        r = self.felvett
        self.felvett = []
        return r

    def close(self) -> None:
        self.kliens.close()


def vonalon(o: dict[str, Any]) -> bool:
    return vonalon_van(
        o["sensor_left"]["intensity"], o["sensor_center"]["intensity"], o["sensor_right"]["intensity"]
    )


def akadaly_kell_kerulni(o: dict[str, Any]) -> bool:
    return akadaly_elol(o, AKADALY_KUSZOB_BELEPES_M)


def akadaly_makro(
    kliens: FelvevoKliens,
    memoria: Memoria,
    stat: FutasStatisztika,
    max_utkozes: int,
) -> tuple[int, dict[str, Any]]:
    """Lefuttatja a kezzel irt akadalykerulest, amig az kesz nem lesz (vagy zsakutca /
    utkozes-hatar). Visszaad: (megtett lepesek szama, az utolso observe)."""
    utolso_irany = [1]
    akadaly_lepesek = [0]
    lepesek = 0
    kliens.ures_felvett()
    for _ in range(MAKRO_MAX_ITERACIO):
        allapot = egy_lepes_akadaly(kliens, stat, utolso_irany, akadaly_lepesek, None, lepesek)
        for szog, tav in kliens.ures_felvett():
            memoria.frissit(szog, tav, False)
            lepesek += 1
        o = kliens.kuld({"command": "observe"})
        if vonalon(o):
            memoria.nullaz()
        if allapot is not Allapot.AKADALY:
            break
        if int(o.get("collision_count", 0) or 0) >= max_utkozes:
            break
    # Biztonsagi ráhagyas: amig az akadaly meg a rover mellett/elott van (barmelyik lidar-szektor
    # a KOZELBEN latja), egyenesen megyunk tovabb - igy a halo nem fordul vissza az akadalyba.
    for _ in range(KIFUTAS_MAX_LEPES):
        sz = o.get("lidar_szektor_min") or []
        if len(sz) < 6 or min(sz) >= KIFUTAS_KUSZOB_M or vonalon(o):
            break
        if min(sz[1:5]) < 0.3:  # az akadaly nem oldalt, hanem elol van - egyenesen menni veszelyes
            break
        kliens.kuld({"command": "move", "distance_m": 0.08, "max_speed": 0.20})
        for szog, tav in kliens.ures_felvett():
            memoria.frissit(szog, tav, False)
            lepesek += 1
        o = kliens.kuld({"command": "observe"})
        if vonalon(o):
            memoria.nullaz()
        if int(o.get("collision_count", 0) or 0) >= max_utkozes:
            break
    stat.akadaly_kerulesek_szama += 1
    return lepesek, o


# --- oldalso vedelem: a halo ne sodorja a rovert egy oldalt (+-30..60 fok) levo akadalynak ----
OLDALSO_KUSZOB_M = 0.3  # az akadaly felszinetol mert tavolsag, ami alatt elharitunk
OLDALSO_SZEKTOROK = (1, 4)
ELHARITAS_FORDULAT_FOK = 45.0


def elharitas_szukseges(o: dict[str, Any]) -> int:
    """0 = nincs teendo; +1 = jobbra kell elharitani (akadaly balra); -1 = balra (akadaly jobbra)."""
    sz = o.get("lidar_szektor_min") or []
    if len(sz) < 6:
        return 0
    bal, jobb = sz[OLDALSO_SZEKTOROK[0]], sz[OLDALSO_SZEKTOROK[1]]
    if min(bal, jobb) >= OLDALSO_KUSZOB_M:
        return 0
    return 1 if bal <= jobb else -1


def elharitas(kliens: FelvevoKliens, memoria: Memoria, irany: int) -> tuple[int, dict[str, Any]]:
    """Egy kis kitero az oldalt levo akadaly elol. Visszaad: (lepesek, uj observe)."""
    kliens.ures_felvett()
    kliens.kuld({"command": "turn", "angle_deg": irany * ELHARITAS_FORDULAT_FOK, "max_angular_speed": 20.0})
    kliens.kuld({"command": "move", "distance_m": 0.08, "max_speed": 0.20})
    for szog, tav in kliens.ures_felvett():
        memoria.frissit(szog, tav, False)
    o = kliens.kuld({"command": "observe"})
    if vonalon(o):
        memoria.nullaz()
    return 1, o
