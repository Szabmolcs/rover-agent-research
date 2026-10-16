"""Tesztek a controllers/analyze_step_log.py-hoz (M10 oszcillacio-triazs).

Regresszios teszt: az M11 ota a pozicio a privilegizalt_diagnosztika alatt van, a regi
elemzo a felso szintu mezot kereste, es minden futasra hamis "rendben"-t mondott.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_GYOKER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_GYOKER / "controllers"))

from analyze_step_log import futas_elemzese, pozicio, utkozes_szamlalo  # noqa: E402


def belepes(lepes: int, x: float, z: float, formatum: str, utkozes: int = 0) -> dict:
    sor = {"lepes_szam": lepes, "allapot_elotte": "VONALON", "allapot_utana": "AKADALY"}
    poz = {"x": x, "y": 0.0, "z": z}
    if formatum == "m11":
        sor["privilegizalt_diagnosztika"] = {"position": poz, "collision_count": utkozes}
    else:
        sor["position"] = poz
        sor["collision_count"] = utkozes
    return sor


class AnalyzeStepLogTest(unittest.TestCase):
    def test_mindket_formatum_olvashato(self) -> None:
        for formatum in ("m10", "m11"):
            sor = belepes(1, 1.0, 2.0, formatum, utkozes=3)
            self.assertEqual(pozicio(sor), {"x": 1.0, "y": 0.0, "z": 2.0})
            self.assertEqual(utkozes_szamlalo(sor), 3)

    def test_helyben_ismetlodo_belepes_gyanus_mindket_formatumban(self) -> None:
        for formatum in ("m10", "m11"):
            sorok = [
                belepes(1, 0.0, 0.0, formatum),
                {**belepes(5, 0.0, 0.0, formatum), "allapot_elotte": "VISSZATALALAS"},
            ]
            e = futas_elemzese("r", sorok, 0.3)
            self.assertTrue(e["van_pozicio"])
            self.assertEqual(e["gyanus_oszcillacios_parok_szama"], 1)

    def test_halado_rover_nem_gyanus(self) -> None:
        sorok = [belepes(1, 0.0, 0.0, "m11"), belepes(9, 0.0, 3.0, "m11")]
        e = futas_elemzese("r", sorok, 0.3)
        self.assertEqual(e["gyanus_oszcillacios_parok_szama"], 0)

    def test_pozicio_nelkuli_naplo_nincs_adat(self) -> None:
        sorok = [
            {"lepes_szam": 1, "allapot_elotte": "VONALON", "allapot_utana": "AKADALY"},
            {"lepes_szam": 5, "allapot_elotte": "VONALON", "allapot_utana": "AKADALY"},
        ]
        e = futas_elemzese("r", sorok, 0.3)
        self.assertFalse(e["van_pozicio"])


if __name__ == "__main__":
    unittest.main()
