#!/usr/bin/env python3
"""M14: a betanitott policy kiertekelese veletlen kiserletezes (epsilon) NELKUL.

Minden lepesben a halo legjobb akciojat valasztja (argmax), igy a kapott
siker-arany a modell valodi teljesitmenyet mutatja.

Hasznalat (a projekt gyokereben):
    python3 controllers/kiertekeles.py
    python3 controllers/kiertekeles.py --epizodok 200
    python3 controllers/kiertekeles.py --szcenario experiments/scenarios/stadium-train-baseline-always-visible.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

_PROJEKT_GYOKER = str(Path(__file__).resolve().parent.parent)
if _PROJEKT_GYOKER not in sys.path:
    sys.path.insert(0, _PROJEKT_GYOKER)

from controllers.rl_env import MAX_ELTERES_M, RoverEnv

ALAPERTELMEZETT_MODELL_FAJL = Path(__file__).resolve().parent.parent / "models" / "rl_policy.keras"


def main() -> int:
    parser = argparse.ArgumentParser(description="M14: betanitott policy kiertekelese")
    parser.add_argument("--modell-fajl", default=str(ALAPERTELMEZETT_MODELL_FAJL))
    parser.add_argument("--epizodok", type=int, default=100)
    parser.add_argument("--szcenario", default=None)
    parser.add_argument("--max-lepes", type=int, default=900)
    parser.add_argument("--max-elteres", type=float, default=MAX_ELTERES_M)
    args = parser.parse_args()

    modell_fajl = Path(args.modell_fajl)
    if not modell_fajl.exists():
        print(f"HIBA: nem talalhato a modell fajl: {modell_fajl}")
        return 1

    modell = keras.models.load_model(modell_fajl)
    q_elore = tf.function(lambda x: modell(x, training=False))

    env = RoverEnv(
        szcenario_fajl=args.szcenario,
        max_lepes=args.max_lepes,
        max_elteres_m=args.max_elteres,
    )

    sikerek = 0
    gps_osszeg = 0
    utkozes_osszeg = 0
    lepes_osszeg_sikeres = 0

    for _ in range(args.epizodok):
        obs = env.reset()
        done = False
        lepesek = 0
        while not done:
            akcio = int(np.argmax(q_elore(obs[np.newaxis, :]).numpy()[0]))
            obs, _, done, info = env.step(akcio)
            lepesek += 1
        sikerek += int(info["done_ok"])
        gps_osszeg += info["elert_gps_pontok"]
        utkozes_osszeg += info["utkozesek"]
        if info["done_ok"]:
            lepes_osszeg_sikeres += lepesek

    n = args.epizodok
    print(f"Epizodok szama:           {n}")
    print(f"Sikeres korok:            {sikerek}/{n} = {sikerek / n:.1%}")
    print(f"Atlag elert GPS pont:     {gps_osszeg / n:.1f} / {env.waypoint_szam}")
    print(f"Atlag utkozes/epizod:     {utkozes_osszeg / n:.2f}")
    if sikerek:
        print(f"Atlag lepes (sikeres):    {lepes_osszeg_sikeres / sikerek:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())