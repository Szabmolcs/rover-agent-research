#!/usr/bin/env python3
"""M10: lépésenkénti napló elemzése az M09-ben dokumentált kétmodális
akadálykerülési jelenség (feltételezett oszcilláció) diagnosztizálásához.

Bemenet: lépésenkénti napló: az M10-es `logs/m10_lepes_naplo.jsonl` (felső szintű
`position`) vagy az M11 óta a `common/kiserlet_naplo.py` formátuma, ahol a
`position` és a `collision_count` a `privilegizalt_diagnosztika` alatt van
(lásd docs/m10-plan.md, 3. munkacsomag; docs/m11-plan.md). Mindkét formátum
olvasható. Ha egy futás egyetlen bejegyzésében sincs pozíció, a futás NEM
"rendben", hanem "NINCS ADAT" jelölést kap - ellenkező esetben egy formátumváltás
miatt vak elemző minden futásra hamis "rendben"-t mondana.

Módszertan (heurisztikus, NEM végleges hiba-taxonómia):
Minden futáson (`run_id`) belül megkeressük az AKADÁLY állapotba lépés
eseményeit (ahol `allapot_utana == "AKADALY"` és `allapot_elotte !=
"AKADALY"`), és minden ilyen eseménynél feljegyezzük a rover
`position` mezőjét (kizárólag ebben a diagnosztikai szkriptben
használjuk - a kontroller maga nem támaszkodik rá).

Ha két egymást követő AKADÁLY-belépés között a pozíció XZ-síkbeli
távolsága egy küszöb alatt marad (alapértelmezetten 0.3 m), az arra
utal, hogy a rover NEM haladt érdemben előre a két akadálytalálkozás
között - ez a jelenség, amit az M09-terv "oszcillációként" ír le
(ismétlődő, ugyanazon a helyen történő akadálytalálkozás), szemben
azzal, amikor a rover egyszerűen egymás után több, különböző
akadállyal találkozik a pályán haladva.

FONTOS KORLÁT: ez a szkript kizárólag azt méri, hogy a pozíció alig
változott két AKADÁLY-belépés között - nem bizonyítja, hogy a rover
ugyanazt az akadályt kerülte-e ismételten, és nem helyettesíti a
videós/vizuális ellenőrzést. Első, gyors triázs-eszköznek szánjuk.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

ALAPERTELMEZETT_NAPLO_FAJL = (
    Path(__file__).resolve().parent.parent / "logs" / "m10_lepes_naplo.jsonl"
)
ALAPERTELMEZETT_OSZCILLACIO_KUSZOB_M = 0.3


def sorok_beolvasasa(fajl: Path) -> list[dict[str, Any]]:
    sorok = []
    with fajl.open(encoding="utf-8") as f:
        for sor in f:
            sor = sor.strip()
            if sor:
                sorok.append(json.loads(sor))
    return sorok


def futasokra_csoportositva(
    bejegyzesek: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    futasok: dict[str, list[dict[str, Any]]] = {}
    for bejegyzes in bejegyzesek:
        futasok.setdefault(bejegyzes["run_id"], []).append(bejegyzes)
    return futasok


def pozicio(bejegyzes: dict[str, Any]) -> dict[str, float] | None:
    """A privilegizalt pozicio mind a ket naploformatumbol (M11: diagnosztika alatt, M10: felso szint)."""
    diag = bejegyzes.get("privilegizalt_diagnosztika") or {}
    return diag.get("position") or bejegyzes.get("position")


def utkozes_szamlalo(bejegyzes: dict[str, Any]) -> Any:
    diag = bejegyzes.get("privilegizalt_diagnosztika") or {}
    ertek = diag.get("collision_count")
    return ertek if ertek is not None else bejegyzes.get("collision_count")


def xz_tavolsag(poz1: dict[str, float] | None, poz2: dict[str, float] | None) -> float | None:
    if not poz1 or not poz2:
        return None
    dx = poz1.get("x", 0.0) - poz2.get("x", 0.0)
    dz = poz1.get("z", 0.0) - poz2.get("z", 0.0)
    return math.sqrt(dx * dx + dz * dz)


def akadaly_belepesek(futas_bejegyzesei: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Az AKADALY allapotba valo belepesek (nem a bennmaradas lepesei)."""
    belepesek = []
    for bejegyzes in futas_bejegyzesei:
        if (
            bejegyzes.get("allapot_utana") == "AKADALY"
            and bejegyzes.get("allapot_elotte") != "AKADALY"
        ):
            belepesek.append(bejegyzes)
    return belepesek


def futas_elemzese(
    run_id: str, futas_bejegyzesei: list[dict[str, Any]], kuszob_m: float
) -> dict[str, Any]:
    belepesek = akadaly_belepesek(futas_bejegyzesei)
    gyanus_parok = []
    for elozo, kovetkezo in zip(belepesek, belepesek[1:]):
        tav = xz_tavolsag(pozicio(elozo), pozicio(kovetkezo))
        if tav is not None and tav < kuszob_m:
            gyanus_parok.append(
                {
                    "elozo_lepes": elozo["lepes_szam"],
                    "kovetkezo_lepes": kovetkezo["lepes_szam"],
                    "tavolsag_m": round(tav, 4),
                    "position": pozicio(kovetkezo),
                }
            )

    utolso = futas_bejegyzesei[-1] if futas_bejegyzesei else {}
    van_pozicio = any(pozicio(b) for b in futas_bejegyzesei)
    return {
        "run_id": run_id,
        "akadaly_belepesek_szama": len(belepesek),
        "gyanus_oszcillacios_parok_szama": len(gyanus_parok),
        "gyanus_parok": gyanus_parok,
        "vegso_collision_count": utkozes_szamlalo(utolso),
        # False: a naplo egyetlen sorabol sem olvashato pozicio -> az elemzes nem ertelmezheto
        "van_pozicio": van_pozicio,
    }


def elemez(naplo_fajl: Path, kuszob_m: float) -> list[dict[str, Any]]:
    bejegyzesek = sorok_beolvasasa(naplo_fajl)
    futasok = futasokra_csoportositva(bejegyzesek)
    eredmenyek = [
        futas_elemzese(run_id, futas_bejegyzesei, kuszob_m)
        for run_id, futas_bejegyzesei in futasok.items()
    ]
    eredmenyek.sort(key=lambda e: e["gyanus_oszcillacios_parok_szama"], reverse=True)
    return eredmenyek


def main() -> int:
    parser = argparse.ArgumentParser(
        description="M10: oszcillacio-gyanus akadalykerulesek keresese a lepesnaploban."
    )
    parser.add_argument("--naplo", default=str(ALAPERTELMEZETT_NAPLO_FAJL))
    parser.add_argument(
        "--kuszob-m",
        type=float,
        default=ALAPERTELMEZETT_OSZCILLACIO_KUSZOB_M,
        help="Ket AKADALY-belepes kozotti minimalis XZ-tavolsag (m), ami alatt gyanusnak jelezzuk.",
    )
    args = parser.parse_args()

    naplo_fajl = Path(args.naplo)
    if not naplo_fajl.exists():
        print(f"Nem talalhato naplofajl: {naplo_fajl}")
        print("Futtasd eloszor a baseline_line_follower.py-t a lepesnaplozassal.")
        return 1

    eredmenyek = elemez(naplo_fajl, args.kuszob_m)
    if not eredmenyek:
        print("A naplofajl ures vagy nem tartalmaz ertelmezheto bejegyzest.")
        return 0

    print(f"{len(eredmenyek)} futas elemezve (kuszob: {args.kuszob_m} m)\n")
    for eredmeny in eredmenyek:
        if not eredmeny["van_pozicio"]:
            jelzo = "NINCS ADAT (nincs pozicio a naploban)"
        elif eredmeny["gyanus_oszcillacios_parok_szama"] > 0:
            jelzo = "!! GYANUS !!"
        else:
            jelzo = "rendben"
        print(
            f"run_id={eredmeny['run_id'][:8]}... "
            f"akadaly_belepesek={eredmeny['akadaly_belepesek_szama']} "
            f"gyanus_parok={eredmeny['gyanus_oszcillacios_parok_szama']} "
            f"[{jelzo}]"
        )
        for par in eredmeny["gyanus_parok"]:
            print(
                f"    lepes {par['elozo_lepes']} -> {par['kovetkezo_lepes']}: "
                f"{par['tavolsag_m']} m (poz: {par['position']})"
            )

    adat_nelkul = sum(1 for e in eredmenyek if not e["van_pozicio"])
    if adat_nelkul:
        print(
            f"\nFIGYELEM: {adat_nelkul}/{len(eredmenyek)} futasnal nincs pozicio a naploban, "
            "ezeket nem lehetett elemezni."
        )

    osszes_gyanus = sum(e["gyanus_oszcillacios_parok_szama"] for e in eredmenyek)
    erintett_futasok = sum(1 for e in eredmenyek if e["gyanus_oszcillacios_parok_szama"] > 0)
    print(
        f"\nOsszegzes: {erintett_futasok}/{len(eredmenyek)} futasban van "
        f"legalabb egy gyanus (helyben-ismetlodo) akadalytalalkozas, "
        f"osszesen {osszes_gyanus} gyanus par."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
