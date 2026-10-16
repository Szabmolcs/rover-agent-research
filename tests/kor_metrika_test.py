"""Tesztek a controllers/kor_metrika.py-hoz: ez a projekt fo metrikaja (task success).

Minden kapudontes erre epul, ezert szintetikus palyabejarasokon ellenorizzuk: a teljes kor,
a fordito irany, a palya belsejeben futo kis kor (gyanus_kor), a palya meretei a
szcenariofajlbol, es az utkozesszamlalo mindket naploformatumban.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

_GYOKER = Path(__file__).resolve().parent.parent
for _utvonal in (str(_GYOKER), str(_GYOKER / "controllers")):
    if _utvonal not in sys.path:
        sys.path.insert(0, _utvonal)

from adapter.backend import tavolsag_a_kozepvonaltol as adapter_tavolsag  # noqa: E402
from kor_metrika import (  # noqa: E402
    alapertelmezett_palya,
    futas_metrikai,
    palya_kerulet,
    palya_szcenariobol,
    tavolsag_a_kozepvonaltol,
)


def kozepvonal_pontok(
    palya: dict, lepes_m: float = 0.1, irany: int = 1
) -> list[tuple[float, float]]:
    """Egy teljes kor a stadion kozepvonalan, a (r, 0) pontbol indulva."""
    r = palya["turn_radius_m"]
    fel = palya["straight_length_m"] / 2.0
    pontok: list[tuple[float, float]] = []
    z = 0.0
    while z < fel:
        pontok.append((r, z))
        z += lepes_m
    n_iv = max(8, int(math.pi * r / lepes_m))
    pontok += [
        (r * math.cos(math.pi * i / n_iv), fel + r * math.sin(math.pi * i / n_iv))
        for i in range(n_iv)
    ]
    z = fel
    while z > -fel:
        pontok.append((-r, z))
        z -= lepes_m
    pontok += [
        (-r * math.cos(math.pi * i / n_iv), -fel - r * math.sin(math.pi * i / n_iv))
        for i in range(n_iv)
    ]
    z = -fel
    while z < 0.0:
        pontok.append((r, z))
        z += lepes_m
    pontok.append((r, 0.0))
    if irany < 0:
        pontok = pontok[::-1]
    # kis tulfutas, hogy a lebegopontos kerekites ne tegye 0,99999-re a kor-aranyt
    return pontok + pontok[1:4]


def sorok(pontok: list[tuple[float, float]], utkozes: list[int] | None = None) -> list[dict]:
    """M11 formatumu naplosorok: a pozicio a privilegizalt_diagnosztika alatt van."""
    kimenet = []
    for i, (x, z) in enumerate(pontok):
        diag = {"position": {"x": x, "y": 0.0, "z": z}}
        if utkozes is not None:
            diag["collision_count"] = utkozes[i]
        kimenet.append({"lepes_szam": i, "privilegizalt_diagnosztika": diag})
    return kimenet


class KorMetrikaTest(unittest.TestCase):
    def test_tokeletes_kor_teljes_es_hatekony(self) -> None:
        palya = alapertelmezett_palya()
        m = futas_metrikai(sorok(kozepvonal_pontok(palya)))
        self.assertTrue(m["teljes_kor"])
        self.assertFalse(m["gyanus_kor"])
        self.assertGreaterEqual(m["kor_arany"], 1.0)
        self.assertEqual(m["vonalon_arany"], 1.0)
        self.assertAlmostEqual(m["hatekonysag"], 1.0, delta=0.05)
        self.assertEqual(m["utkozesek"], 0)

    def test_fordito_irany_is_teljes_kor(self) -> None:
        palya = alapertelmezett_palya()
        m = futas_metrikai(sorok(kozepvonal_pontok(palya, irany=-1)))
        self.assertTrue(m["teljes_kor"])

    def test_fel_kor_nem_teljes(self) -> None:
        palya = alapertelmezett_palya()
        pontok = kozepvonal_pontok(palya)
        m = futas_metrikai(sorok(pontok[: len(pontok) // 2]))
        self.assertFalse(m["teljes_kor"])
        self.assertLess(m["kor_arany"], 1.0)

    def test_kis_kor_a_palya_belsejeben_gyanus(self) -> None:
        # Az audit szintetikus probaja: 1 m sugaru kor, amely sosem erinti a vonalat.
        pontok = [
            (math.cos(t * 2 * math.pi / 100), math.sin(t * 2 * math.pi / 100)) for t in range(105)
        ]
        m = futas_metrikai(sorok(pontok))
        self.assertTrue(m["teljes_kor"])
        self.assertEqual(m["vonalon_arany"], 0.0)
        self.assertTrue(m["gyanus_kor"])

    def test_kevesebb_mint_ket_pozicio(self) -> None:
        m = futas_metrikai([{"lepes_szam": 0}])
        self.assertFalse(m["teljes_kor"])
        self.assertFalse(m["gyanus_kor"])

    def test_utkozes_mindket_naploformatumban(self) -> None:
        palya = alapertelmezett_palya()
        pontok = kozepvonal_pontok(palya)
        szamlalo = [0] * 10 + [2] * (len(pontok) - 10)
        m11 = futas_metrikai(sorok(pontok, szamlalo))
        self.assertEqual(m11["utkozesek"], 2)
        m10 = []
        for i, (x, z) in enumerate(pontok):
            m10.append(
                {"lepes_szam": i, "position": {"x": x, "z": z}, "collision_count": szamlalo[i]}
            )
        self.assertEqual(futas_metrikai(m10)["utkozesek"], 2)

    def test_palya_meretei_a_szcenariobol(self) -> None:
        szcenario = {
            "track": {
                "straight_length_m": 20.0,
                "turn_radius_m": 6.0,
                "line_width_m": 0.3,
                "background_color_rgb": [0, 0, 0],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            fajl = Path(tmp) / "sz.json"
            fajl.write_text(json.dumps(szcenario), encoding="utf-8")
            palya = palya_szcenariobol(fajl)
        self.assertEqual(palya["straight_length_m"], 20.0)
        self.assertAlmostEqual(palya_kerulet(palya), 40.0 + 2 * math.pi * 6.0)
        # a nagyobb palya sajat kozepvonalan a rover "vonalon" van, a beegetett meretekkel nem
        m = futas_metrikai(sorok(kozepvonal_pontok(palya)), palya)
        self.assertTrue(m["teljes_kor"])
        self.assertEqual(m["vonalon_arany"], 1.0)
        self.assertAlmostEqual(m["hatekonysag"], 1.0, delta=0.05)
        m_alap = futas_metrikai(sorok(kozepvonal_pontok(palya)))
        self.assertLess(m_alap["vonalon_arany"], 1.0)

    def test_tavolsag_egyezik_az_adapterevel(self) -> None:
        palya = alapertelmezett_palya()
        for x in (-6.0, -4.0, -1.5, 0.0, 2.0, 4.0, 4.3, 7.0):
            for z in (-12.0, -6.0, -2.0, 0.0, 3.0, 6.0, 9.0, 12.0):
                self.assertAlmostEqual(
                    tavolsag_a_kozepvonaltol(x, z, palya), adapter_tavolsag(x, z), places=9
                )


if __name__ == "__main__":
    unittest.main()
