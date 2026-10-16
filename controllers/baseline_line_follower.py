#!/usr/bin/env python3
"""Hagyományos (AI nélküli) vonalkövető baseline kontroller - M09/M10/M10.5/M11.

Állapotgép (VONALON / KERESÉS / AKADÁLY) + P-szabályozó a bal/jobb
szenzor intenzitáskülönbségére. A vezérlési döntések kizárólag az
observe válasz sensor_left/center/right és lidar_szektor_min
mezőire támaszkodnak, nem használják a position/speed/collision_*
privilegizált szimulátor-mezőket (ezek csak a lépésenkénti
diagnosztikai naplóba kerülnek).

M11: a lépésenkénti naplózás mostantól a common.kiserlet_naplo
egységes modulját használja (KiserletNaplozo), ami minden jelenlegi
és jövőbeli kontroller (baseline, agent-alapú, tanult) számára közös
séma szerint ír JSONL naplót a logs/kiserlet_naplo.jsonl fájlba.

Lásd docs/m09-plan.md a tervezési döntésekért.
"""

from __future__ import annotations

import argparse
import json
import socket
import math
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

_PROJEKT_GYOKER = str(Path(__file__).resolve().parent.parent)
if _PROJEKT_GYOKER not in sys.path:
    sys.path.insert(0, _PROJEKT_GYOKER)

from common.kiserlet_naplo import KiserletMetaadat, KiserletNaplozo


ALAPERTELMEZETT_HOST = "127.0.0.1"
ALAPERTELMEZETT_PORT = 8765
MAXIMALIS_FRAME_MERET = 16 * 1024
NAPLO_FAJL = Path(__file__).resolve().parent.parent / "logs" / "m09_runs.jsonl"
KISERLET_NAPLO_FAJL = Path(__file__).resolve().parent.parent / "logs" / "kiserlet_naplo.jsonl"
CONTROLLER_NEV = "baseline_line_follower"
BACKEND_NEV = "unity_sim"

MOVE_LEPES_M = 0.08
MOVE_SEBESSEG = 0.20
TURN_MIN_FOK = 1.0
TURN_MAX_FOK = 5.0
TURN_SEBESSEG = 20.0
P_EROSITES = 8.0
HOLTSAV = 0.02

# M10.6: a KERESES mostantol nem helyben forog, hanem korivet ir le (turn + move),
# mert akadalykerules utan a rover fel meterre is kerulhet a vonaltol - helyben
# forgassal egy ilyen tavoli vonal matematikailag megtalalhatatlan.
# A korív sugara: MOVE_LEPES_M / radian(KERESES_FORDULAT_FOK) ~ 1.15 m, atmeroje
# ~2.3 m; a MAX_LEPES egy teljes kort fed le (360 / 4 = 90 lepes).
KERESES_FORDULAT_FOK = 4.0
# Holdout utan: a kereses elso fazisa helyben lengo kereses (+15, -15, +35, -35,
# vissza 0 fokra, 5 fokos lepesekben), mert a vonal egy lepessel korabban meg a
# kozepso szenzor alatt volt. Csak ha ez sem talalja, jon az ivben korzo kereses.
KERESES_LENGES_FOK = [5.0] * 3 + [-5.0] * 6 + [5.0] * 10 + [-5.0] * 14 + [5.0] * 7
KERESES_MAX_LEPES = 90 + len(KERESES_LENGES_FOK)
# M10.6b: taguló spiral. A fix sugarú korív (r = MOVE_LEPES_M / radian(szog))
# csak akkor talalja meg a vonalat, ha az az atmerojen belul van - a meresek
# szerint a sikertelen futasok mind 1.4-1.5 m-re alltak meg a vonaltol, epp a
# 2.3 m atmeroju kor hatarán kivul. Ezert a fordulat szoget lepesenkent
# csokkentjuk (a sugarat noveljuk), amig el nem erjuk a minimumot: igy a
# nyomvonal taguló spiral, ami barmilyen tavoli vonalat metsz.
# M10.6b meres: a taguló spiral (8 fok -> 2 fok, 120 lepes) rontott - a rover
# nem esett ki, de 9.6 m-nyi kanyargas kozben eltavolodott a vonaltol es
# folyamatosan utkozott (163 utkozes/futas, a lepesek 9%-aban volt a vonalon).
# Ezert a spiral kikapcsolva (csokkenes = 0), a keresés fix, szűk korív marad.
KERESES_SPIRAL_CSOKKENES = 0.0
KERESES_FORDULAT_MIN_FOK = 2.0

# M13-elo: a rover sarkai ~1.07 m-re vannak a kozepponttol, a helyben
# fordulas ezen a sugaron sopor. 0.5 m-es belepesnel a kiteres fordulasa mar
# beleert az akadalyba, ezert korabban kezdunk kiterni.
AKADALY_KUSZOB_BELEPES_M = 1.0
AKADALY_KUSZOB_KILEPES_M = 1.3
AKADALY_FORDULAT_FOK = 15.0
# M10.6: az akadaly akkor van tenylegesen megkerulve, ha nem csak elottunk, hanem
# az elkerules oldalan (mellettunk) is szabad az ut. Enelkul a rover mar akkor
# "tisztanak" latta a helyzetet, amikor az akadaly meg mellette volt, es a
# VISSZATALALAS visszafordulasa egyenesen bele vitte.
AKADALY_KUSZOB_OLDAL_M = 1.0
ZSAKUTCA_AKADALY_MAX_LEPES = 20
ELOLSO_SZEKTOROK = (2, 3)
BAL_SZEKTOROK = (0, 1)
JOBB_SZEKTOROK = (4, 5)

# M10.6b: hatarozottabb visszafordulas a vonal fele. A korabbi 5 fok / 15 lepes
# osszesen 75 fokot fordult, mikozben 1.2 m-t haladt elore - ez nem eleg ahhoz,
# hogy a rover visszakanyarodjon a vonalra, csak tovabb tavolodott tole.
VISSZATALALAS_FORDULAT_FOK = 5.0
VISSZATALALAS_MAX_LEPES = 15
# M13-elo: iranytartas. Az AKADALY-ban kiadott fordulatokat osszegezzuk (a vonal
# iranyahoz kepest), es a VISSZATALALAS ezt forgatja vissza, majd BEFOGO_FOK-os
# szogben tart a vonal fele. A fix 5 fok x 15 lepes nem forgatta vissza a
# kiterest, a rover tovabb tavolodott es elhagyta a palyat.
VISSZATALALAS_BEFOGO_FOK = 30.0
VISSZATALALAS_MAX_LEPES_IRANYTARTO = 40


class Allapot(Enum):
    VONALON = "VONALON"
    KERESES = "KERESES"
    AKADALY = "AKADALY"
    VISSZATALALAS = "VISSZATALALAS"


@dataclass
class FutasStatisztika:
    lepesek_szama: int = 0
    parancsok_szama: int = 0
    vonalvesztesek_szama: int = 0
    akadaly_kerulesek_szama: int = 0
    zsakutcak_szama: int = 0
    palyaelhagyas: bool = False
    utkozott: bool = False
    utkozesek_szama: int = 0
    kezdet: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def szenzor_mezok(observe: dict[str, Any]) -> dict[str, Any]:
    return {
        "sensor_left": observe.get("sensor_left"),
        "sensor_center": observe.get("sensor_center"),
        "sensor_right": observe.get("sensor_right"),
        "lidar_szektor_min": observe.get("lidar_szektor_min"),
        # A kerules dontese (akadaly_elol/akadaly_oldalt) a nyers sugarakbol szamol, ezert
        # ezeket is naplozni kell, kulonben a kerulesi dontesek a naplobol nem rekonstrualhatok.
        "lidar_nyers": observe.get("lidar_nyers"),
        "lidar_nyers_ervenyes": observe.get("lidar_nyers_ervenyes"),
    }


def privilegizalt_diagnosztika_mezok(observe: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": observe.get("position"),
        "collision_occurred": observe.get("collision_occurred"),
        "collision_count": observe.get("collision_count"),
    }


class GatewayKliens:
    def __init__(self, host: str, port: int) -> None:
        self.socket = socket.create_connection((host, port), timeout=5)
        self.socket.settimeout(None)

    def kuld(self, parancs: dict[str, Any]) -> dict[str, Any]:
        parancs = {"request_id": str(uuid4()), **parancs}
        payload = json.dumps(parancs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if not 1 <= len(payload) <= MAXIMALIS_FRAME_MERET:
            raise ValueError("A kimeno JSON tul nagy vagy ures.")

        self.socket.sendall(len(payload).to_bytes(4, "big", signed=False))
        self.socket.sendall(payload)

        hossz_prefix = self._pontosan_fogad(4)
        valasz_hossz = int.from_bytes(hossz_prefix, "big", signed=False)
        valasz_payload = self._pontosan_fogad(valasz_hossz)
        return json.loads(valasz_payload.decode("utf-8"))

    def _pontosan_fogad(self, hossz: int) -> bytes:
        reszek: list[bytes] = []
        hatralevo = hossz
        while hatralevo:
            adat = self.socket.recv(hatralevo)
            if not adat:
                raise ConnectionError("A kapcsolat varatlanul megszakadt.")
            reszek.append(adat)
            hatralevo -= len(adat)
        return b"".join(reszek)

    def close(self) -> None:
        self.socket.close()


def hibajel_szamitasa(observe_valasz: dict[str, Any]) -> float:
    bal = observe_valasz["sensor_left"]["intensity"]
    jobb = observe_valasz["sensor_right"]["intensity"]
    return jobb - bal


def mindharom_nem_feher(observe_valasz: dict[str, Any]) -> bool:
    return not (
        observe_valasz["sensor_left"]["white"]
        or observe_valasz["sensor_center"]["white"]
        or observe_valasz["sensor_right"]["white"]
    )


# A rover tenyleges meretei (a jelenet es a prefab alapjan): a kerekek kulso
# ele +-0.85 m-re van a kozepvonaltol, a rover eleje 0.65 m-re a kozepponttol,
# a LiDAR a kozepvonalon, 0.264 m-rel a kozeppont elott.
ROVER_FEL_SZELESSEG_M = 0.85
ROVER_ELEJE_M = 0.65
LIDAR_ELORE_M = 0.264
LIDAR_LATOMEZO_FOK = 180.0
KORRIDOR_RAHAGYAS_M = 0.10


def akadaly_elol(observe_valasz: dict[str, Any], kuszob: float) -> bool:
    """Van-e akadaly a rover szelessegenek savjaban, az elejetol kuszob m-en belul?

    Korabban csak a +-30 fokos elulso szektorokat nezte a LiDAR-tol merve, ami
    0.5 m-en +-0.29 m-es savot fed le, mikozben a rover +-0.85 m szeles. Az
    oldalt, a kerek vonalaban allo akadalyokat igy nem latta (kulso audit: az
    M10-es utkozesek 76%-anal az elulso szektor szabadot mutatott). Mostantol a
    nyers sugarakbol szamolja, mi esik a rover savjaba.
    """
    nyers = observe_valasz.get("lidar_nyers")
    maszk = observe_valasz.get("lidar_nyers_ervenyes")
    if nyers and maszk and len(nyers) == len(maszk) and len(nyers) >= 2:
        n = len(nyers)
        fel = ROVER_FEL_SZELESSEG_M + KORRIDOR_RAHAGYAS_M
        eleje = ROVER_ELEJE_M - LIDAR_ELORE_M
        for i, (r, talalt) in enumerate(zip(nyers, maszk)):
            if not talalt:
                continue
            szog = math.radians(-LIDAR_LATOMEZO_FOK / 2 + i * LIDAR_LATOMEZO_FOK / (n - 1))
            elore, oldalra = r * math.cos(szog), r * math.sin(szog)
            if abs(oldalra) <= fel and elore > 0 and elore - eleje < kuszob:
                return True
        return False
    # Tartalek: nyers jel nelkul a regi, szektoros dontes.
    szektorok = observe_valasz.get("lidar_szektor_min") or []
    if len(szektorok) <= max(ELOLSO_SZEKTOROK):
        return False
    return any(szektorok[i] < kuszob for i in ELOLSO_SZEKTOROK)


OLDAL_RAHAGYAS_M = 0.30
# A LiDAR 180 fokos, a rover hatulja ~0.91 m-rel mogotte van: ami a hatso fel
# melle kerul, azt mar nem latjuk. Ennyit megyunk meg egyenesen visszafordulas elott.
VAKZONA_M = 1.0


def akadaly_oldalt(observe_valasz: dict[str, Any], elkerulesi_irany: int, kuszob: float) -> bool:
    """Ott van-e meg az akadaly az elkerules oldalan, a rover valodi szelessegevel?

    Jobbra kerulesnel (+1) az akadaly a bal oldalon (negativ szog) marad. Nyers
    LiDAR-ral: akadaly, ha egy talalat ezen az oldalon a rover felszelessege +
    rahagyason belul van, es nincs messze a rover eleje elott. Nyers jel nelkul
    a regi szektoros dontes (kuszob) a tartalek.
    """
    nyers = observe_valasz.get("lidar_nyers")
    ervenyes = observe_valasz.get("lidar_nyers_ervenyes")
    if nyers and ervenyes and len(nyers) == len(ervenyes) and len(nyers) > 1:
        n = len(nyers)
        hatar = ROVER_FEL_SZELESSEG_M + OLDAL_RAHAGYAS_M
        for i, (r, ok) in enumerate(zip(nyers, ervenyes)):
            if not ok:
                continue
            szog = math.radians(-LIDAR_LATOMEZO_FOK / 2 + i * LIDAR_LATOMEZO_FOK / (n - 1))
            elore = r * math.cos(szog) + LIDAR_ELORE_M
            oldalra = r * math.sin(szog)
            figyelt = oldalra < 0 if elkerulesi_irany > 0 else oldalra > 0
            if figyelt and abs(oldalra) <= hatar and elore <= ROVER_ELEJE_M + OLDAL_RAHAGYAS_M:
                return True
        return False
    szektorok = observe_valasz.get("lidar_szektor_min") or []
    if len(szektorok) <= max(JOBB_SZEKTOROK):
        return False
    figyelt_szektorok = BAL_SZEKTOROK if elkerulesi_irany > 0 else JOBB_SZEKTOROK
    return any(szektorok[i] < kuszob for i in figyelt_szektorok)


def szabadabb_oldal_elojele(observe_valasz: dict[str, Any]) -> int:
    szektorok = observe_valasz.get("lidar_szektor_min") or []
    if len(szektorok) <= max(JOBB_SZEKTOROK):
        return 1
    bal_min = min(szektorok[i] for i in BAL_SZEKTOROK)
    jobb_min = min(szektorok[i] for i in JOBB_SZEKTOROK)
    return 1 if jobb_min >= bal_min else -1


def egy_lepes_vonalon(
    kliens: GatewayKliens,
    stat: FutasStatisztika,
    utolso_elojel: list[int],
    naplo: KiserletNaplozo | None,
    lepes_szam: int,
) -> Allapot:
    observe = kliens.kuld({"command": "observe"})
    stat.parancsok_szama += 1
    kiadott_parancsok: list[dict[str, Any]] = []

    if akadaly_elol(observe, AKADALY_KUSZOB_BELEPES_M):
        stat.akadaly_kerulesek_szama += 1
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe),
                kiadott_parancsok,
                Allapot.VONALON.value,
                Allapot.AKADALY.value,
                privilegizalt_diagnosztika_mezok(observe),
            )
        return Allapot.AKADALY

    if mindharom_nem_feher(observe):
        stat.vonalvesztesek_szama += 1
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe),
                kiadott_parancsok,
                Allapot.VONALON.value,
                Allapot.KERESES.value,
                privilegizalt_diagnosztika_mezok(observe),
            )
        return Allapot.KERESES

    hiba = hibajel_szamitasa(observe)
    if abs(hiba) > HOLTSAV:
        utolso_elojel[0] = 1 if hiba > 0 else -1
        korrekcio_fok = max(TURN_MIN_FOK, min(TURN_MAX_FOK, abs(hiba) * P_EROSITES))
        szog = korrekcio_fok if hiba > 0 else -korrekcio_fok
        turn_parancs = {"command": "turn", "angle_deg": szog, "max_angular_speed": TURN_SEBESSEG}
        kliens.kuld(turn_parancs)
        stat.parancsok_szama += 1
        kiadott_parancsok.append(turn_parancs)

    move_parancs = {"command": "move", "distance_m": MOVE_LEPES_M, "max_speed": MOVE_SEBESSEG}
    kliens.kuld(move_parancs)
    stat.parancsok_szama += 1
    kiadott_parancsok.append(move_parancs)
    if naplo is not None:
        naplo.rogzit(
            lepes_szam,
            szenzor_mezok(observe),
            kiadott_parancsok,
            Allapot.VONALON.value,
            Allapot.VONALON.value,
            privilegizalt_diagnosztika_mezok(observe),
        )
    return Allapot.VONALON


def _irany_nullazasa(fordulat_osszeg: list[float] | None, allapot: Allapot) -> Allapot:
    if fordulat_osszeg is not None:
        fordulat_osszeg[0] = 0.0
    return allapot


def egy_lepes_akadaly(
    kliens: GatewayKliens,
    stat: FutasStatisztika,
    utolso_elkerulesi_irany: list[int],
    akadaly_lepesek: list[int],
    naplo: KiserletNaplozo | None,
    lepes_szam: int,
    fordulat_osszeg: list[float] | None = None,
) -> Allapot:
    observe = kliens.kuld({"command": "observe"})
    stat.parancsok_szama += 1

    elol_zart = akadaly_elol(observe, AKADALY_KUSZOB_KILEPES_M)
    oldalt_zart = akadaly_oldalt(observe, utolso_elkerulesi_irany[0], AKADALY_KUSZOB_OLDAL_M)

    if not elol_zart and not oldalt_zart:
        akadaly_lepesek[0] = 0
        # Vak zona: a hatso fel mellett levo akadalyt mar nem latjuk, ezert
        # meg VAKZONA_M-et egyenesen megyunk, mielott a vonal fele fordulnank.
        vakzona_parancsok: list[dict[str, Any]] = []
        kovetkezo = Allapot.VISSZATALALAS
        megtett = 0.0
        while megtett + 1e-9 < VAKZONA_M:
            ellenorzes = kliens.kuld({"command": "observe"})
            stat.parancsok_szama += 1
            if akadaly_elol(ellenorzes, AKADALY_KUSZOB_BELEPES_M):
                kovetkezo = Allapot.AKADALY
                break
            move_parancs = {
                "command": "move",
                "distance_m": MOVE_LEPES_M,
                "max_speed": MOVE_SEBESSEG,
            }
            kliens.kuld(move_parancs)
            stat.parancsok_szama += 1
            vakzona_parancsok.append(move_parancs)
            megtett += MOVE_LEPES_M
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe),
                vakzona_parancsok,
                Allapot.AKADALY.value,
                kovetkezo.value,
                privilegizalt_diagnosztika_mezok(observe),
            )
        return kovetkezo

    if not elol_zart:
        # Elhaladasi fazis: elottunk mar szabad, de az akadaly meg mellettunk van.
        # Nem fordulunk tovabb (az tulforgatas lenne), csak egyenesen elhaladunk,
        # amig az akadaly a hatunk moge nem kerul.
        move_parancs = {"command": "move", "distance_m": MOVE_LEPES_M, "max_speed": MOVE_SEBESSEG}
        kliens.kuld(move_parancs)
        stat.parancsok_szama += 1
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe),
                [move_parancs],
                Allapot.AKADALY.value,
                Allapot.AKADALY.value,
                privilegizalt_diagnosztika_mezok(observe),
            )
        return Allapot.AKADALY

    # Csak azok a lepesek szamitanak zsakutcanak, amikor az UT ELOTTUNK zart -
    # az elhaladasi fazis normalis mukodes, nem elakadas.
    akadaly_lepesek[0] += 1
    if akadaly_lepesek[0] >= ZSAKUTCA_AKADALY_MAX_LEPES:
        akadaly_lepesek[0] = 0
        stat.zsakutcak_szama += 1
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe),
                [],
                Allapot.AKADALY.value,
                Allapot.KERESES.value,
                privilegizalt_diagnosztika_mezok(observe),
            )
        return _irany_nullazasa(fordulat_osszeg, Allapot.KERESES)

    irany = szabadabb_oldal_elojele(observe)
    utolso_elkerulesi_irany[0] = irany
    if fordulat_osszeg is not None:
        fordulat_osszeg[0] += irany * AKADALY_FORDULAT_FOK
    turn_parancs = {
        "command": "turn",
        "angle_deg": irany * AKADALY_FORDULAT_FOK,
        "max_angular_speed": TURN_SEBESSEG,
    }
    kliens.kuld(turn_parancs)
    stat.parancsok_szama += 1
    kiadott_parancsok: list[dict[str, Any]] = [turn_parancs]

    move_parancs = {"command": "move", "distance_m": MOVE_LEPES_M, "max_speed": MOVE_SEBESSEG}
    kliens.kuld(move_parancs)
    stat.parancsok_szama += 1
    kiadott_parancsok.append(move_parancs)

    if naplo is not None:
        naplo.rogzit(
            lepes_szam,
            szenzor_mezok(observe),
            kiadott_parancsok,
            Allapot.AKADALY.value,
            Allapot.AKADALY.value,
            privilegizalt_diagnosztika_mezok(observe),
        )
    return Allapot.AKADALY


def egy_lepes_visszatalalas(
    kliens: GatewayKliens,
    stat: FutasStatisztika,
    utolso_elkerulesi_irany: list[int],
    visszatalalas_lepesek: list[int],
    naplo: KiserletNaplozo | None,
    lepes_szam: int,
    fordulat_osszeg: list[float] | None = None,
) -> Allapot:
    observe = kliens.kuld({"command": "observe"})
    stat.parancsok_szama += 1
    kiadott_parancsok: list[dict[str, Any]] = []

    # M10.6: ha a visszafordulas kozben ujra akadaly kerult elenk, ne erolteseuk
    # a vonalat - vissza az elkerulesbe. Enelkul a rover az akadaly oldalan
    # korbe-korbe csuszott, ismetelten nekiutkozve.
    if akadaly_elol(observe, AKADALY_KUSZOB_BELEPES_M):
        visszatalalas_lepesek[0] = 0
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe),
                [],
                Allapot.VISSZATALALAS.value,
                Allapot.AKADALY.value,
                privilegizalt_diagnosztika_mezok(observe),
            )
        return Allapot.AKADALY

    if not mindharom_nem_feher(observe):
        visszatalalas_lepesek[0] = 0
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe),
                kiadott_parancsok,
                Allapot.VISSZATALALAS.value,
                Allapot.VONALON.value,
                privilegizalt_diagnosztika_mezok(observe),
            )
        return _irany_nullazasa(fordulat_osszeg, Allapot.VONALON)

    # Visszavonva (2026-10-08): a netto fordulat elojele azt mutatja, merre nez a rover,
    # nem azt, melyik oldalon a vonal; ujrakiteres utan atbillent, es a rover kifele
    # indult (validacios elotti proba). Az utolso kiteresi irany szamit.
    irany_vissza = -utolso_elkerulesi_irany[0]
    if fordulat_osszeg is not None:
        hiany = irany_vissza * VISSZATALALAS_BEFOGO_FOK - fordulat_osszeg[0]
        szog = max(-AKADALY_FORDULAT_FOK, min(AKADALY_FORDULAT_FOK, hiany))
    else:
        szog = irany_vissza * VISSZATALALAS_FORDULAT_FOK
    if abs(szog) >= TURN_MIN_FOK:
        turn_parancs = {"command": "turn", "angle_deg": szog, "max_angular_speed": TURN_SEBESSEG}
        kliens.kuld(turn_parancs)
        stat.parancsok_szama += 1
        kiadott_parancsok.append(turn_parancs)
        if fordulat_osszeg is not None:
            fordulat_osszeg[0] += szog

    move_parancs = {"command": "move", "distance_m": MOVE_LEPES_M, "max_speed": MOVE_SEBESSEG}
    kliens.kuld(move_parancs)
    stat.parancsok_szama += 1
    kiadott_parancsok.append(move_parancs)
    visszatalalas_lepesek[0] += 1

    observe2 = kliens.kuld({"command": "observe"})
    stat.parancsok_szama += 1

    if not mindharom_nem_feher(observe2):
        visszatalalas_lepesek[0] = 0
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe2),
                kiadott_parancsok,
                Allapot.VISSZATALALAS.value,
                Allapot.VONALON.value,
                privilegizalt_diagnosztika_mezok(observe2),
            )
        return _irany_nullazasa(fordulat_osszeg, Allapot.VONALON)

    max_lepes = (
        VISSZATALALAS_MAX_LEPES_IRANYTARTO
        if fordulat_osszeg is not None
        else VISSZATALALAS_MAX_LEPES
    )
    if visszatalalas_lepesek[0] >= max_lepes:
        visszatalalas_lepesek[0] = 0
        stat.vonalvesztesek_szama += 1
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe2),
                kiadott_parancsok,
                Allapot.VISSZATALALAS.value,
                Allapot.KERESES.value,
                privilegizalt_diagnosztika_mezok(observe2),
            )
        return _irany_nullazasa(fordulat_osszeg, Allapot.KERESES)

    if naplo is not None:
        naplo.rogzit(
            lepes_szam,
            szenzor_mezok(observe2),
            kiadott_parancsok,
            Allapot.VISSZATALALAS.value,
            Allapot.VISSZATALALAS.value,
            privilegizalt_diagnosztika_mezok(observe2),
        )
    return Allapot.VISSZATALALAS


def egy_lepes_kereses(
    kliens: GatewayKliens,
    stat: FutasStatisztika,
    utolso_elojel: list[int],
    kereses_lepesek: list[int],
    naplo: KiserletNaplozo | None,
    lepes_szam: int,
) -> Allapot:
    irany = utolso_elojel[0] or 1
    if kereses_lepesek[0] < len(KERESES_LENGES_FOK):
        lenges = {
            "command": "turn",
            "angle_deg": irany * KERESES_LENGES_FOK[kereses_lepesek[0]],
            "max_angular_speed": TURN_SEBESSEG,
        }
        kliens.kuld(lenges)
        stat.parancsok_szama += 1
        kereses_lepesek[0] += 1
        observe = kliens.kuld({"command": "observe"})
        stat.parancsok_szama += 1
        talalt = not mindharom_nem_feher(observe)
        if talalt:
            kereses_lepesek[0] = 0
        kovetkezo = Allapot.VONALON if talalt else Allapot.KERESES
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe),
                [lenges],
                Allapot.KERESES.value,
                kovetkezo.value,
                privilegizalt_diagnosztika_mezok(observe),
            )
        return kovetkezo
    # Taguló spiral: minel regebb ota keresunk, annal nagyobb ivet irunk le.
    szog = max(
        KERESES_FORDULAT_MIN_FOK,
        KERESES_FORDULAT_FOK - KERESES_SPIRAL_CSOKKENES * kereses_lepesek[0],
    )
    turn_parancs = {
        "command": "turn",
        "angle_deg": irany * szog,
        "max_angular_speed": TURN_SEBESSEG,
    }
    kliens.kuld(turn_parancs)
    stat.parancsok_szama += 1
    kiadott_parancsok: list[dict[str, Any]] = [turn_parancs]

    # M10.6: a forgas utan elore is haladunk, igy a rover korivet ir le a
    # helyben forgas helyett - csak igy talalhat meg egy tole tavolabb levo
    # vonalat (pl. akadalykerules utan).
    move_parancs = {"command": "move", "distance_m": MOVE_LEPES_M, "max_speed": MOVE_SEBESSEG}
    kliens.kuld(move_parancs)
    stat.parancsok_szama += 1
    kiadott_parancsok.append(move_parancs)
    kereses_lepesek[0] += 1

    observe = kliens.kuld({"command": "observe"})
    stat.parancsok_szama += 1

    if akadaly_elol(observe, AKADALY_KUSZOB_BELEPES_M):
        kereses_lepesek[0] = 0
        stat.akadaly_kerulesek_szama += 1
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe),
                kiadott_parancsok,
                Allapot.KERESES.value,
                Allapot.AKADALY.value,
                privilegizalt_diagnosztika_mezok(observe),
            )
        return Allapot.AKADALY

    if not mindharom_nem_feher(observe):
        kereses_lepesek[0] = 0
        if naplo is not None:
            naplo.rogzit(
                lepes_szam,
                szenzor_mezok(observe),
                kiadott_parancsok,
                Allapot.KERESES.value,
                Allapot.VONALON.value,
                privilegizalt_diagnosztika_mezok(observe),
            )
        return Allapot.VONALON

    if kereses_lepesek[0] >= KERESES_MAX_LEPES:
        stat.palyaelhagyas = True

    if naplo is not None:
        naplo.rogzit(
            lepes_szam,
            szenzor_mezok(observe),
            [turn_parancs],
            Allapot.KERESES.value,
            Allapot.KERESES.value,
            privilegizalt_diagnosztika_mezok(observe),
        )
    return Allapot.KERESES


def _szcenario_nev(reset_valasz: object) -> str | None:
    """A Unity a reset_position valaszanak uzeneteben kozli a betoltott szcenariot."""
    if not isinstance(reset_valasz, dict):
        return None
    uzenet = str(reset_valasz.get("message", ""))
    if "scenario=" not in uzenet:
        return None
    return (uzenet.split("scenario=", 1)[1].split() or [None])[0]


def _time_scale(reset_valasz: object) -> float | None:
    if not isinstance(reset_valasz, dict):
        return None
    uzenet = str(reset_valasz.get("message", ""))
    if "timescale=" not in uzenet:
        return None
    try:
        return float(uzenet.split("timescale=", 1)[1].split()[0])
    except (ValueError, IndexError):
        return None


def _parameterek() -> dict:
    """A kontroller osszes modulszintu konstansa, a meres megismetlesehez."""
    return {
        nev: ertek
        for nev, ertek in sorted(globals().items())
        if nev.isupper() and isinstance(ertek, (int, float, tuple, list))
    }


def _git_allapot() -> dict:
    gyoker = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=gyoker, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        piszkos = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)logs"],
            cwd=gyoker,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return {"git_commit": commit or None, "git_tiszta": piszkos == ""}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_tiszta": None}


def futtat(
    host: str,
    port: int,
    max_lepes: int,
    kiserlet_naplo_fajl: Path | None = KISERLET_NAPLO_FAJL,
    seed: int | None = None,
    elvart_szcenario: str | None = None,
) -> FutasStatisztika:
    kliens = GatewayKliens(host, port)
    stat = FutasStatisztika()
    allapot = Allapot.VONALON
    utolso_elojel = [1]
    kereses_lepesek = [0]
    utolso_elkerulesi_irany = [1]
    visszatalalas_lepesek = [0]
    akadaly_lepesek = [0]
    fordulat_osszeg = [0.0]
    run_id = str(uuid4())
    naplo = (
        KiserletNaplozo(
            kiserlet_naplo_fajl,
            run_id,
            KiserletMetaadat(controller=CONTROLLER_NEV, backend=BACKEND_NEV, seed=seed),
        )
        if kiserlet_naplo_fajl
        else None
    )

    try:
        # M10.6c: a futas elott biztosan tiszta allapotbol indulunk. A
        # reset_position csak IDLE-ben mukodik (lasd docs/protocol.md), ezert ha
        # az elozo futas ERROR-ral vegzodott, eloszor abbol kell kilepni -
        # kulonben a rover a pályán kivul ragad, minden move/turn elutasitasra
        # kerul, es a kovetkezo futasok mind azonos, ertelmetlen eredményt adnak.
        allapot_valasz = kliens.kuld({"command": "get_status"})
        if allapot_valasz.get("state") == "ERROR":
            kliens.kuld({"command": "reset_error"})
        kliens.kuld({"command": "stop"})
        reset_parancs: dict[str, Any] = {"command": "reset_position"}
        if seed is not None:
            reset_parancs["noise_seed"] = seed
        reset_valasz = kliens.kuld(reset_parancs)
        szcenario = _szcenario_nev(reset_valasz)
        if elvart_szcenario and szcenario != elvart_szcenario:
            raise RuntimeError(
                f"A Unityben a(z) '{szcenario}' szcenario van betoltve, "
                f"a mereshez '{elvart_szcenario}' kell."
            )
        if naplo:
            naplo.metaadat_rogzitese(
                seed=seed,
                scenario=szcenario,
                time_scale=_time_scale(reset_valasz),
                max_lepes=max_lepes,
                parameterek=_parameterek(),
                **_git_allapot(),
            )

        while stat.lepesek_szama < max_lepes and not stat.palyaelhagyas:
            if allapot is Allapot.VONALON:
                allapot = egy_lepes_vonalon(kliens, stat, utolso_elojel, naplo, stat.lepesek_szama)
            elif allapot is Allapot.AKADALY:
                allapot = egy_lepes_akadaly(
                    kliens,
                    stat,
                    utolso_elkerulesi_irany,
                    akadaly_lepesek,
                    naplo,
                    stat.lepesek_szama,
                    fordulat_osszeg=fordulat_osszeg,
                )
            elif allapot is Allapot.VISSZATALALAS:
                allapot = egy_lepes_visszatalalas(
                    kliens,
                    stat,
                    utolso_elkerulesi_irany,
                    visszatalalas_lepesek,
                    naplo,
                    stat.lepesek_szama,
                    fordulat_osszeg=fordulat_osszeg,
                )
                # M10 utan: sikertelen visszateres utan a kereses a vonal feloli oldalra induljon.
                if allapot is Allapot.KERESES:
                    utolso_elojel[0] = -utolso_elkerulesi_irany[0]
            else:
                allapot = egy_lepes_kereses(
                    kliens, stat, utolso_elojel, kereses_lepesek, naplo, stat.lepesek_szama
                )
            stat.lepesek_szama += 1

        vegso_observe = kliens.kuld({"command": "observe"})
        stat.parancsok_szama += 1
        stat.utkozott = bool(vegso_observe.get("collision_occurred", False))
        stat.utkozesek_szama = int(vegso_observe.get("collision_count", 0) or 0)
    finally:
        if naplo is not None:
            naplo.close()
        kliens.close()

    return stat


def naplo_iras(stat: FutasStatisztika) -> None:
    NAPLO_FAJL.parent.mkdir(parents=True, exist_ok=True)
    bejegyzes = {
        "kezdet": stat.kezdet,
        "vege": datetime.now(timezone.utc).isoformat(),
        "lepesek_szama": stat.lepesek_szama,
        "parancsok_szama": stat.parancsok_szama,
        "vonalvesztesek_szama": stat.vonalvesztesek_szama,
        "akadaly_kerulesek_szama": stat.akadaly_kerulesek_szama,
        "zsakutcak_szama": stat.zsakutcak_szama,
        "palyaelhagyas": stat.palyaelhagyas,
        "utkozott": stat.utkozott,
        "utkozesek_szama": stat.utkozesek_szama,
    }
    with NAPLO_FAJL.open("a", encoding="utf-8") as f:
        json.dump(bejegyzes, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"Naplozva: {NAPLO_FAJL}")


def main() -> int:
    parser = argparse.ArgumentParser(description="M09 vonalkoveto baseline kontroller")
    parser.add_argument("--host", default=ALAPERTELMEZETT_HOST)
    parser.add_argument("--port", type=int, default=ALAPERTELMEZETT_PORT)
    parser.add_argument(
        "--max-lepes",
        type=int,
        default=500,
        help="Biztonsagi felso korlat a ciklusok szamara.",
    )
    parser.add_argument(
        "--kiserlet-naplo",
        default=str(KISERLET_NAPLO_FAJL),
        help=(
            "M11: egyseges kiserlet-naplo (JSONL) utvonala. "
            "Ures string (--kiserlet-naplo '') eseten kikapcsolja."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A hasznalt szcenario seed-je, kizarolag naplozasi celra.",
    )
    parser.add_argument(
        "--elvart-szcenario",
        default=None,
        help="Ha meg van adva, a futas csak akkor indul el, ha a Unity ezt a szcenariot toltotte be.",
    )
    args = parser.parse_args()
    kiserlet_naplo_fajl = Path(args.kiserlet_naplo) if args.kiserlet_naplo else None

    print(f"Kapcsolodas: {args.host}:{args.port} ...")
    try:
        stat = futtat(
            args.host,
            args.port,
            args.max_lepes,
            kiserlet_naplo_fajl,
            args.seed,
            args.elvart_szcenario,
        )
    except (ConnectionError, OSError, RuntimeError) as hiba:
        print(f"Hiba: {hiba}", file=sys.stderr)
        return 1

    print(
        f"Futas vege: {stat.lepesek_szama} lepes, "
        f"{stat.parancsok_szama} parancs, "
        f"{stat.vonalvesztesek_szama} vonalveszes, "
        f"{stat.akadaly_kerulesek_szama} akadalykerules, "
        f"zsakutcak={stat.zsakutcak_szama}, "
        f"utkozesek={stat.utkozesek_szama}, "
        f"palyaelhagyas={stat.palyaelhagyas}"
    )
    naplo_iras(stat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
