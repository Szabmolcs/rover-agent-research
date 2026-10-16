#!/usr/bin/env python3
"""Task success (teljesitett kor) es koridő szamitasa lepesenkenti naplobol.

Az M09-M11 meresek a vonalvesztest, parancsszamot, utkozest es palyaelhagyast
mertek, de a kiiras M09-es pontja szerinti "koridő" es a task success (sikerult-e
a rovernek korbeernie a palyat) nem volt kozvetlenul merve. Ez a szkript ezt
potolja - utolag is, barmelyik korabbi lepesnaplora.

Modszer: a lepesenkenti naploban rogzitett privilegizalt pozicio (position)
alapjan a palya kozeppontja (0, 0) koruli szoget kicsomagoljuk (unwrap), es a
kumulalt szogelfordulast elosztjuk 2*pi-vel. Ez a "kor-arany": 1.0 = pontosan
egy teljes kor. A pozicio KIZAROLAG kiertekelesre hasznalhato, a vezerlesre nem
(lasd docs/protocol.md) - a szkript ezert kulon, utolagos elemzo eszkoz.

Hasznalat:
    python3 controllers/kor_metrika.py                      # az alapertelmezett naplo
    python3 controllers/kor_metrika.py --naplo-fajl <fajl>  # masik naplo
    python3 controllers/kor_metrika.py --utolso 30          # csak az utolso N futas
    python3 controllers/kor_metrika.py --json               # gepi feldolgozashoz
    python3 controllers/kor_metrika.py --szcenario experiments/scenarios/<palya>.json
                                                            # a palya meretei a szcenariofajlbol

Palya-meretek: alapertelmezetten a train palya meretei (adapter/backend.py). Mas
meretu palyan (pl. validacios/holdout) a --szcenario kapcsoloval a szcenariofajl
`track` blokkjabol (straight_length_m, turn_radius_m, line_width_m) olvassuk be
oket - kulonben a "vonalon toltott arany" es a hatekonysag ervenytelen lenne.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

GYOKER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GYOKER))

from adapter.backend import (  # noqa: E402
    PALYA_EGYENES_HOSSZ_M,
    PALYA_IV_SUGAR_M,
    PALYA_VONAL_SZELESSEG_M,
)

ALAPERTELMEZETT_NAPLO = GYOKER / "logs" / "kiserlet_naplo.jsonl"
# Ennyi vonalon toltott arany alatt egy "teljes kor" gyanus: a kor-arany csak a (0, 0)
# kozeppont koruli szog, ezert egy a palya belsejeben, a vonalat sosem erinto kor is
# teljes korre szamitana (lasd tests/kor_metrika_test.py).
GYANUS_KOR_VONALON_ARANY = 0.2


def alapertelmezett_palya() -> dict:
    """A train palya merete (adapter/backend.py allando ertekei)."""
    return {
        "straight_length_m": PALYA_EGYENES_HOSSZ_M,
        "turn_radius_m": PALYA_IV_SUGAR_M,
        "line_width_m": PALYA_VONAL_SZELESSEG_M,
    }


def palya_szcenariobol(szcenario_fajl: Path) -> dict:
    """A palya meretei a szcenariofajl `track` blokkjabol (docs/scenario-schema.md)."""
    track = json.loads(szcenario_fajl.read_text(encoding="utf-8"))["track"]
    return {
        "straight_length_m": float(track["straight_length_m"]),
        "turn_radius_m": float(track["turn_radius_m"]),
        "line_width_m": float(track["line_width_m"]),
    }


def palya_kerulet(palya: dict) -> float:
    return 2 * palya["straight_length_m"] + 2 * math.pi * palya["turn_radius_m"]


def tavolsag_a_kozepvonaltol(x: float, z: float, palya: dict) -> float:
    """A stadion alaku palya kozepvonalatol mert tavolsag (m) a megadott meretekkel.

    Ugyanaz a szamitas, mint az adapter.backend.tavolsag_a_kozepvonaltol (es a Unity
    TrackController.TavolsagAKozepvonaltol), de a palya meretei parameterek, nem
    beegetett allandok.
    """
    fel_hossz = palya["straight_length_m"] / 2.0
    r = palya["turn_radius_m"]

    def egyenes(allando_x: float) -> float:
        zc = max(-fel_hossz, min(fel_hossz, z))
        return math.hypot(x - allando_x, z - zc)

    def felkor(kozep_z: float, irany: float) -> float:
        dz = z - kozep_z
        if dz * irany < 0:
            return math.inf
        return abs(math.hypot(x, dz) - r)

    return min(egyenes(r), egyenes(-r), felkor(-fel_hossz, -1.0), felkor(fel_hossz, 1.0))


def pozicio(sor: dict) -> dict | None:
    """A privilegizalt pozicio mind a ket naploformatumbol (M10 es M11)."""
    diag = sor.get("privilegizalt_diagnosztika") or {}
    return diag.get("position") or sor.get("position")


def utkozes_szamlalo(sor: dict) -> int | None:
    diag = sor.get("privilegizalt_diagnosztika") or {}
    ertek = diag.get("collision_count", sor.get("collision_count"))
    return int(ertek) if isinstance(ertek, (int, float)) else None


def futas_metrikai(sorok: list[dict], palya: dict | None = None) -> dict:
    palya = palya or alapertelmezett_palya()
    poziciok = [p for sor in sorok if (p := pozicio(sor))]
    if len(poziciok) < 2:
        return {
            "lepesek": len(sorok),
            "kor_arany": 0.0,
            "teljes_kor": False,
            "gyanus_kor": False,
            "megtett_ut_m": 0.0,
            "hatekonysag": 0.0,
            "vonalon_arany": 0.0,
            "utkozesek": 0,
        }

    kumulalt_szog = 0.0
    for elozo, kovetkezo in zip(poziciok, poziciok[1:]):
        kulonbseg = math.atan2(kovetkezo["z"], kovetkezo["x"]) - math.atan2(elozo["z"], elozo["x"])
        # kicsomagolas: a -pi..pi ugrasokat visszavezetjuk a valodi elfordulasra
        while kulonbseg > math.pi:
            kulonbseg -= 2 * math.pi
        while kulonbseg < -math.pi:
            kulonbseg += 2 * math.pi
        kumulalt_szog += kulonbseg
    kor_arany = abs(kumulalt_szog) / (2 * math.pi)

    megtett_ut = sum(
        math.hypot(b["x"] - a["x"], b["z"] - a["z"]) for a, b in zip(poziciok, poziciok[1:])
    )
    fel_szelesseg = palya["line_width_m"] / 2.0
    vonalon = sum(
        1 for p in poziciok if tavolsag_a_kozepvonaltol(p["x"], p["z"], palya) <= fel_szelesseg
    )

    szamlalok = [c for sor in sorok if (c := utkozes_szamlalo(sor)) is not None]
    utkozesek = (max(szamlalok) - min(szamlalok)) if szamlalok else 0

    vonalon_arany = vonalon / len(poziciok)
    teljes_kor = kor_arany >= 1.0
    return {
        "lepesek": len(sorok),
        "kor_arany": round(kor_arany, 4),
        "teljes_kor": teljes_kor,
        # a teljes kor kriteriuma szukseges, de nem elegseges: ha a rover alig volt a
        # vonalon, a "teljes kor" valoszinuleg nem a palya bejarasa volt
        "gyanus_kor": teljes_kor and vonalon_arany < GYANUS_KOR_VONALON_ARANY,
        "megtett_ut_m": round(megtett_ut, 2),
        "hatekonysag": (
            round(kor_arany * palya_kerulet(palya) / megtett_ut, 4) if megtett_ut else 0.0
        ),
        "vonalon_arany": round(vonalon_arany, 4),
        "utkozesek": utkozesek,
    }


def futasokra_bont(naplo_fajl: Path) -> dict[str, list[dict]]:
    futasok: dict[str, list[dict]] = {}
    for sor in naplo_fajl.read_text(encoding="utf-8").splitlines():
        if not sor.strip():
            continue
        bejegyzes = json.loads(sor)
        futasok.setdefault(bejegyzes.get("run_id", "ismeretlen"), []).append(bejegyzes)
    return futasok


def kiir(eredmenyek: list[dict], naplo_fajl: Path, palya: dict | None = None) -> None:
    palya = palya or alapertelmezett_palya()
    print(f"Kor-metrika ({len(eredmenyek)} futas, {naplo_fajl.name}):")
    print(f"  Palya kerulete:       {palya_kerulet(palya):.1f} m")
    korok = [e["kor_arany"] for e in eredmenyek]
    utak = [e["megtett_ut_m"] for e in eredmenyek]
    vonalon = [e["vonalon_arany"] for e in eredmenyek]
    hatekonysag = [e["hatekonysag"] for e in eredmenyek]
    utkozesek = [e["utkozesek"] for e in eredmenyek]
    teljes = sum(1 for e in eredmenyek if e["teljes_kor"])

    def sor(nev: str, ertekek: list[float], tizedes: int = 3) -> str:
        atlag = statistics.mean(ertekek)
        szoras = statistics.stdev(ertekek) if len(ertekek) > 1 else 0.0
        return (
            f"  {nev:<24}atlag={atlag:.{tizedes}f}, szoras={szoras:.{tizedes}f}, "
            f"min={min(ertekek):.{tizedes}f}, max={max(ertekek):.{tizedes}f}"
        )

    print(sor("Kor-arany:", korok))
    print(sor("Megtett ut (m):", utak, 1))
    print(sor("Hatekonysag:", hatekonysag))
    print(sor("Vonalon toltott arany:", vonalon))
    print(sor("Utkozesek:", [float(u) for u in utkozesek], 1))
    print(f"  TASK SUCCESS (teljes kor): {teljes}/{len(eredmenyek)} futas")
    print(
        f"  Utkozesmentes futas:       "
        f"{sum(1 for e in eredmenyek if e['utkozesek'] == 0)}/{len(eredmenyek)}"
    )
    gyanus = sum(1 for e in eredmenyek if e["gyanus_kor"])
    if gyanus:
        print(
            f"  FIGYELEM: {gyanus} 'teljes kor' vonalon toltott aranya < "
            f"{GYANUS_KOR_VONALON_ARANY:.0%} - valoszinuleg nem a palya bejarasa."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Task success es koridő a lepesnaplobol")
    parser.add_argument("--naplo-fajl", default=str(ALAPERTELMEZETT_NAPLO))
    parser.add_argument("--utolso", type=int, default=None, help="Csak az utolso N futas.")
    parser.add_argument("--json", action="store_true", help="Gepi olvashato kimenet.")
    parser.add_argument(
        "--szcenario",
        default=None,
        help="Szcenariofajl, amelybol a palya meretei jonnek (alapertelmezett: train palya).",
    )
    args = parser.parse_args()

    naplo_fajl = Path(args.naplo_fajl)
    if not naplo_fajl.is_file():
        print(f"Nincs ilyen naplofajl: {naplo_fajl}", file=sys.stderr)
        return 2

    palya = alapertelmezett_palya()
    if args.szcenario:
        szcenario_fajl = Path(args.szcenario)
        if not szcenario_fajl.is_file():
            print(f"Nincs ilyen szcenariofajl: {szcenario_fajl}", file=sys.stderr)
            return 2
        palya = palya_szcenariobol(szcenario_fajl)

    futasok = futasokra_bont(naplo_fajl)
    azonositok = list(futasok)
    if args.utolso:
        azonositok = azonositok[-args.utolso :]

    eredmenyek = []
    for run_id in azonositok:
        metrika = futas_metrikai(futasok[run_id], palya)
        metrika["run_id"] = run_id
        eredmenyek.append(metrika)

    if not eredmenyek:
        print("A naplo nem tartalmaz futast.", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(eredmenyek, ensure_ascii=False, indent=2))
    else:
        kiir(eredmenyek, naplo_fajl, palya)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
