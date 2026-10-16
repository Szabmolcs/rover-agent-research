"""A lepesnaplo es a metaadat teljessege (M11): amibol a kontroller dont, az a naploban van."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_GYOKER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_GYOKER))
sys.path.insert(0, str(_GYOKER / "controllers"))

import baseline_line_follower as bl  # noqa: E402


class NaploMezokTest(unittest.TestCase):
    def test_szenzor_mezok_tartalmazza_a_nyers_lidart(self) -> None:
        observe = {
            "lidar_szektor_min": [1.0] * 6,
            "lidar_nyers": [2.0, 3.0],
            "lidar_nyers_ervenyes": [True, False],
        }
        mezok = bl.szenzor_mezok(observe)
        self.assertEqual(mezok["lidar_nyers"], [2.0, 3.0])
        self.assertEqual(mezok["lidar_nyers_ervenyes"], [True, False])

    def test_parameterek_tartalmazza_a_lista_konstansokat(self) -> None:
        parameterek = bl._parameterek()
        self.assertIn("KERESES_LENGES_FOK", parameterek)
        for nev, ertek in vars(bl).items():
            if nev.isupper() and isinstance(ertek, (int, float, tuple, list)):
                self.assertIn(nev, parameterek)


if __name__ == "__main__":
    unittest.main()
