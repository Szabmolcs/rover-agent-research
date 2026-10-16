#!/usr/bin/env python3
"""Unit tesztek a baseline vonalkövető állapotgépéhez - M10.

Ezt a fájlt Claude írta (lásd AI_USAGE.md), Dalma review-ja mellett.
Ellentétben a `protocol_fuzz_test.py`-jal, ez a teszt NEM igényel
futó Unity szervert: egy StubKliens-t injektál a
`gateway/client.py`-hoz hasonló interfésszel, ami előre megadott
`observe` válaszokat ad vissza. Ez azt teszteli, hogy a
`controllers/baseline_line_follower.py` állapotgépe (VONALON /
AKADÁLY / VISSZATALALAS / KERESÉS) a specifikációnak megfelelően
vált állapotot - nem azt, hogy a Unity-szimuláció fizikailag helyesen
viselkedik (azt a Play Mode-os manuális/videós tesztek és a
jövőbeli, Unityt igénylő integrációs tesztek fedik le).

Futtatás:
    python3 tests/baseline_line_follower_test.py
    python3 -m pytest -v tests/baseline_line_follower_test.py
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from typing import Any

GYOKER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GYOKER / "controllers"))

from baseline_line_follower import (  # noqa: E402
    LIDAR_ELORE_M,
    LIDAR_LATOMEZO_FOK,
    MOVE_LEPES_M,
    ROVER_ELEJE_M,
    VAKZONA_M,
    Allapot,
    FutasStatisztika,
    VISSZATALALAS_MAX_LEPES,
    ZSAKUTCA_AKADALY_MAX_LEPES,
    egy_lepes_akadaly,
    egy_lepes_kereses,
    egy_lepes_visszatalalas,
    egy_lepes_vonalon,
)


class StubGatewayKliens:
    """Előre megadott observe-válaszokat visszaadó, hálózat nélküli stub.

    A move/turn parancsokra egyszerű "completed" választ ad, ezeket nem
    kell előre megadni - csak az observe válaszok sorrendje számít.
    """

    def __init__(self, observe_valaszok: list[dict[str, Any]]) -> None:
        self._observe_valaszok = list(observe_valaszok)
        self.kuldott_parancsok: list[dict[str, Any]] = []

    def kuld(self, parancs: dict[str, Any]) -> dict[str, Any]:
        self.kuldott_parancsok.append(parancs)
        if parancs["command"] == "observe":
            if not self._observe_valaszok:
                raise AssertionError("Nem volt tobb elokeszitett observe valasz.")
            return self._observe_valaszok.pop(0)
        return {"status": "completed"}


LIDAR_SUGARSZAM = 36
LIDAR_NINCS_AKADALY_M = 5.0


def lidar_nyers(akadaly_tavolsag_m: float) -> tuple[list[float], list[bool]]:
    """Nyers LiDAR-sugarak es talalati maszk egy, a rover elott allo FAL-hoz.

    A Unity valos observe valasza `lidar_nyers` es `lidar_nyers_ervenyes` mezot is ad, es a
    kerules (akadaly_elol / akadaly_oldalt) ezekbol dont - a szektoros `lidar_szektor_min`
    csak tartalek. Ezert a csonk is ad nyers sugarakat, hogy az allapotgep-tesztek a
    valodi dontesi agat futtassak. A tavolsag ugyanazt jelenti, mint a korabbi szektoros
    csonkban: a rover elejetol az akadalyig mert elore irany tavolsag (a fal a rover elott
    `akadaly_tavolsag_m` meterre van). Ha ez a "nincs akadaly" ertek, egy sugar sem talal.
    """
    n = LIDAR_SUGARSZAM
    if akadaly_tavolsag_m >= LIDAR_NINCS_AKADALY_M:
        return [10.0] * n, [False] * n
    eleje = ROVER_ELEJE_M - LIDAR_ELORE_M
    elore = akadaly_tavolsag_m + eleje
    nyers = []
    for i in range(n):
        szog = math.radians(-LIDAR_LATOMEZO_FOK / 2 + i * LIDAR_LATOMEZO_FOK / (n - 1))
        nyers.append(elore / math.cos(szog) if abs(math.cos(szog)) > 1e-6 else 10.0)
    return nyers, [r <= 10.0 for r in nyers]


def observe_valasz(
    white: bool = False,
    akadaly_tavolsag_m: float = 5.0,
    intensity_bal: float = 0.0,
    intensity_jobb: float = 0.0,
) -> dict[str, Any]:
    nyers, ervenyes = lidar_nyers(akadaly_tavolsag_m)
    return {
        "sensor_left": {"white": white, "intensity": intensity_bal},
        "sensor_center": {"white": white, "intensity": 0.0},
        "sensor_right": {"white": white, "intensity": intensity_jobb},
        "lidar_szektor_min": [akadaly_tavolsag_m] * 6,
        "lidar_nyers": nyers,
        "lidar_nyers_ervenyes": ervenyes,
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "collision_occurred": False,
        "collision_count": 0,
    }


class AllapotgepAtmenetekTest(unittest.TestCase):
    def test_vonalon_akadaly_eszleleskor_atvalt_akadalyra(self) -> None:
        stat = FutasStatisztika()
        kliens = StubGatewayKliens([observe_valasz(white=True, akadaly_tavolsag_m=0.3)])
        uj_allapot = egy_lepes_vonalon(kliens, stat, [1], None, 0)
        self.assertIs(uj_allapot, Allapot.AKADALY)
        self.assertEqual(stat.akadaly_kerulesek_szama, 1)

    def test_vonalon_vonalvesztesnel_atvalt_keresesre(self) -> None:
        stat = FutasStatisztika()
        kliens = StubGatewayKliens([observe_valasz(white=False, akadaly_tavolsag_m=5.0)])
        uj_allapot = egy_lepes_vonalon(kliens, stat, [1], None, 0)
        self.assertIs(uj_allapot, Allapot.KERESES)
        self.assertEqual(stat.vonalvesztesek_szama, 1)

    def test_akadaly_kozeli_akadalynal_marad_akadalyban_es_iranyt_jegyez(self) -> None:
        stat = FutasStatisztika()
        irany = [0]
        kliens = StubGatewayKliens([observe_valasz(akadaly_tavolsag_m=0.3)])
        uj_allapot = egy_lepes_akadaly(kliens, stat, irany, [0], None, 0)
        self.assertIs(uj_allapot, Allapot.AKADALY)
        self.assertIn(irany[0], (1, -1))

    def test_akadaly_elhagyasakor_visszatalalasra_valt_nem_kozvetlenul_vonalonra(self) -> None:
        stat = FutasStatisztika()
        # Az akadaly eltunese utan a rover meg VAKZONA_M-et egyenesen megy (minden
        # lepes elott observe), es csak utana fordul vissza a vonal fele.
        vakzona_lepes = math.ceil(VAKZONA_M / MOVE_LEPES_M - 1e-9)
        valaszok = [observe_valasz(akadaly_tavolsag_m=5.0) for _ in range(1 + vakzona_lepes)]
        kliens = StubGatewayKliens(valaszok)
        uj_allapot = egy_lepes_akadaly(kliens, stat, [1], [0], None, 0)
        self.assertIs(uj_allapot, Allapot.VISSZATALALAS)
        mozgasok = [p["command"] for p in kliens.kuldott_parancsok if p["command"] != "observe"]
        self.assertEqual(mozgasok, ["move"] * vakzona_lepes)

    def test_vakzona_kozben_uj_akadaly_visszaad_akadalyba(self) -> None:
        stat = FutasStatisztika()
        kliens = StubGatewayKliens(
            [
                observe_valasz(akadaly_tavolsag_m=5.0),
                observe_valasz(akadaly_tavolsag_m=5.0),
                observe_valasz(akadaly_tavolsag_m=0.3),
            ]
        )
        uj_allapot = egy_lepes_akadaly(kliens, stat, [1], [0], None, 0)
        self.assertIs(uj_allapot, Allapot.AKADALY)
        mozgasok = [p["command"] for p in kliens.kuldott_parancsok if p["command"] != "observe"]
        self.assertEqual(mozgasok, ["move"])

    def test_akadaly_zsakutca_eszleles_eskalal_keresesre(self) -> None:
        stat = FutasStatisztika()
        # ZSAKUTCA_AKADALY_MAX_LEPES-szer egymas utan "meg mindig akadaly
        # elottunk" valaszt adunk - ez szimulalja, hogy a rover nem tud
        # kikerulni (pl. ket akadaly koze szorult).
        valaszok = [
            observe_valasz(akadaly_tavolsag_m=0.3) for _ in range(ZSAKUTCA_AKADALY_MAX_LEPES)
        ]
        kliens = StubGatewayKliens(valaszok)
        irany = [1]
        lepesek = [0]
        uj_allapot = Allapot.AKADALY
        for _ in range(ZSAKUTCA_AKADALY_MAX_LEPES):
            uj_allapot = egy_lepes_akadaly(kliens, stat, irany, lepesek, None, 0)
        self.assertIs(uj_allapot, Allapot.KERESES)
        self.assertEqual(stat.zsakutcak_szama, 1)
        self.assertEqual(lepesek[0], 0)

    def test_visszatalalas_azonnal_vonalon_ha_mar_feher_a_szenzor(self) -> None:
        stat = FutasStatisztika()
        kliens = StubGatewayKliens([observe_valasz(white=True)])
        uj_allapot = egy_lepes_visszatalalas(kliens, stat, [1], [0], None, 0)
        self.assertIs(uj_allapot, Allapot.VONALON)

    def test_visszatalalas_forgas_es_mozgas_utan_megtalalja_a_vonalat(self) -> None:
        stat = FutasStatisztika()
        kliens = StubGatewayKliens([observe_valasz(white=False), observe_valasz(white=True)])
        uj_allapot = egy_lepes_visszatalalas(kliens, stat, [1], [0], None, 0)
        self.assertIs(uj_allapot, Allapot.VONALON)

    def test_visszatalalas_max_lepes_utan_eszkalal_keresesre(self) -> None:
        stat = FutasStatisztika()
        visszatalalas_lepesek = [0]
        allapot = Allapot.VISSZATALALAS
        lepesszam = 0
        # +2 biztonsagi ratartas, hogy biztosan tuljussunk a korlaton.
        while lepesszam < VISSZATALALAS_MAX_LEPES + 2:
            kliens = StubGatewayKliens([observe_valasz(white=False), observe_valasz(white=False)])
            allapot = egy_lepes_visszatalalas(
                kliens, stat, [1], visszatalalas_lepesek, None, lepesszam
            )
            lepesszam += 1
            if allapot is not Allapot.VISSZATALALAS:
                break
        self.assertIs(allapot, Allapot.KERESES)
        self.assertLessEqual(lepesszam, VISSZATALALAS_MAX_LEPES + 1)

    def test_kereses_vonal_megtalalasakor_visszavalt_vonalonra(self) -> None:
        stat = FutasStatisztika()
        kliens = StubGatewayKliens([observe_valasz(white=True)])
        uj_allapot = egy_lepes_kereses(kliens, stat, [1], [0], None, 0)
        self.assertIs(uj_allapot, Allapot.VONALON)

    def test_kereses_max_lepes_utan_palyaelhagyast_jelez(self) -> None:
        stat = FutasStatisztika()
        kereses_lepesek = [0]
        from baseline_line_follower import KERESES_MAX_LEPES

        for lepes in range(KERESES_MAX_LEPES):
            kliens = StubGatewayKliens([observe_valasz(white=False)])
            egy_lepes_kereses(kliens, stat, [1], kereses_lepesek, None, lepes)
        self.assertTrue(stat.palyaelhagyas)


class IranytartasTest(unittest.TestCase):
    def test_visszatalalas_visszaforgatja_a_kiterest_majd_befog(self) -> None:
        stat = FutasStatisztika()
        osszeg = [45.0]  # az AKADALY 3 x 15 fokot fordult jobbra (+1)
        kliens = StubGatewayKliens([observe_valasz(), observe_valasz()])
        egy_lepes_visszatalalas(kliens, stat, [1], [0], None, 0, fordulat_osszeg=osszeg)
        fordulatok = [p["angle_deg"] for p in kliens.kuldott_parancsok if p["command"] == "turn"]
        self.assertEqual(fordulatok, [-15.0])
        self.assertAlmostEqual(osszeg[0], 30.0)

    def test_vonal_megtalalasakor_az_iranyosszeg_nullazodik(self) -> None:
        stat = FutasStatisztika()
        osszeg = [20.0]
        kliens = StubGatewayKliens([observe_valasz(white=True)])
        allapot = egy_lepes_visszatalalas(kliens, stat, [1], [0], None, 0, fordulat_osszeg=osszeg)
        self.assertIs(allapot, Allapot.VONALON)
        self.assertEqual(osszeg[0], 0.0)


class LengoKeresesTest(unittest.TestCase):
    def test_lenges_mindket_iranyba_es_nullara_er_vissza(self) -> None:
        from baseline_line_follower import KERESES_LENGES_FOK

        self.assertEqual(sum(KERESES_LENGES_FOK), 0.0)
        reszosszegek, osszeg = [], 0.0
        for d in KERESES_LENGES_FOK:
            osszeg += d
            reszosszegek.append(osszeg)
        self.assertEqual(max(reszosszegek), 35.0)
        self.assertEqual(min(reszosszegek), -35.0)

    def test_lenges_kozben_nincs_elorehaladas(self) -> None:
        stat = FutasStatisztika()
        kliens = StubGatewayKliens([observe_valasz(), observe_valasz()])
        egy_lepes_kereses(kliens, stat, [-1], [0], None, 0)
        egy_lepes_kereses(kliens, stat, [-1], [1], None, 1)
        mozgas = [p for p in kliens.kuldott_parancsok if p["command"] != "observe"]
        self.assertEqual([p["command"] for p in mozgas], ["turn", "turn"])
        self.assertEqual(mozgas[0]["angle_deg"], -5.0)


if __name__ == "__main__":
    unittest.main()
