#!/usr/bin/env python3
"""M14: a policy rovid emlekezete (odometria) - a tanito kornyezet es az eles
controller UGYANEZT hasznalja, hogy a halo ugyanazt lassa mindket helyen.

Miert kell? Ha a rover elhagyja a vonalat (pl. akadaly kerulesehez), mind a
harom vonalszenzor 0-t mutat, es a halo a pillanatnyi szenzorokbol nem tudja,
merre van a vonal. Ez a modul azt jegyzi meg, MENNYIT fordult es haladt a
rover a legutobbi vonalon-pillanat ota - kizarolag a SAJAT kiadott parancsaibol
(odometria), a position/speed/collision_* privilegizalt mezok nelkul.

A nyilvantartas minden lepesben nullazodik, amikor valamelyik vonalszenzor
feher (a rover a vonalon van).
"""

from __future__ import annotations

import math

import numpy as np

MEMORIA_DIM = 4
FEHER_KUSZOB = 0.5


class Memoria:
    def __init__(self) -> None:
        self.nullaz()

    def nullaz(self) -> None:
        self.fordulas_fok = 0.0  # osszes kiadott fordulas a vonalrol valo leteres ota
        self.oldalirany_m = 0.0  # becsult jobbra-eltolodas a leteres pillanatanak iranyahoz kepest
        self.elore_m = 0.0  # becsult elorehaladas ugyanabban a keretben
        self.lepesek = 0

    def frissit(self, szog_fok: float, tav_m: float, vonalon: bool) -> None:
        """Egy lepes utan: a kiadott fordulas (szog_fok), a kiadott lepes (tav_m)
        es az, hogy a lepes utan a vonalon van-e a rover."""
        if vonalon:
            self.nullaz()
            return
        self.fordulas_fok += szog_fok
        rad = math.radians(self.fordulas_fok)
        self.elore_m += tav_m * math.cos(rad)
        self.oldalirany_m += tav_m * math.sin(rad)
        self.lepesek += 1

    def vektor(self) -> np.ndarray:
        return np.array(
            [
                float(np.clip(self.fordulas_fok / 90.0, -2.0, 2.0)),
                float(np.clip(self.oldalirany_m / 2.0, -2.0, 2.0)),
                float(np.clip(self.elore_m / 3.0, -2.0, 2.0)),
                float(np.clip(self.lepesek / 100.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )


def vonalon_van(szenzor_bal: float, szenzor_kozep: float, szenzor_jobb: float) -> bool:
    return max(szenzor_bal, szenzor_kozep, szenzor_jobb) >= FEHER_KUSZOB
