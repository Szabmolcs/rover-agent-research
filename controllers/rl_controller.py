#!/usr/bin/env python3
"""HIBRID vezerlo: neuralis halo (vonalkovetes + visszatalalas) + kezzel irt akadalykerules.

Mukodes:
  * Amig NINCS akadaly a rover elott (LIDAR elso szektorai > AKADALY_KUSZOB_BELEPES_M),
    a betanitott TensorFlow halo donti el minden lepesben, mennyit forduljon.
    A halo bemenete kizarolag szenzoradat (3 vonalszenzor) + az elozo akcio +
    az odometria-memoria. A position / collision_* mezoket NEM hasznalja.
  * Ha akadaly kerul a rover ele, a baseline_line_follower.py kezzel irt
    akadalykerulese (egy_lepes_akadaly) veszi at az iranyitast, amig kikerultuk.
  * Utana ujra a halo vezet - ez talal vissza a vonalra.

Hasznalat (a projekt gyokereben, Unity-ben a Play gomb megnyomva):
    python3 controllers/rl_controller.py
    python3 controllers/rl_controller.py --max-lepes 1500 --modell-fajl models/hibrid.keras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJEKT_GYOKER = str(Path(__file__).resolve().parent.parent)
if _PROJEKT_GYOKER not in sys.path:
    sys.path.insert(0, _PROJEKT_GYOKER)

from controllers.baseline_line_follower import FutasStatisztika  # noqa: E402
from controllers.hibrid_kozos import (  # noqa: E402
    FelvevoKliens,
    akadaly_kell_kerulni,
    akadaly_makro,
    elharitas,
    elharitas_szukseges,
    vonalon,
)
from controllers.rl_obs import Memoria, vonalon_van  # noqa: E402

# Ugyanaz az akcioter, mint a tanitaskor (rl_env.py). NEM SZABAD ELTERNI.
ANGLES_FOK = [-20.0, -10.0, -4.0, 0.0, 4.0, 10.0, 20.0]
LEPESHOSSZ_M = 0.08
FORDULASI_SEBESSEG = 20.0
MOZGASI_SEBESSEG = 0.20


def observacio(o: dict[str, Any], elozo_akcio: int, memoria: Memoria) -> np.ndarray:
    """Ugyanaz a 14 elemu vektor, mint RoverEnv._observe()-ban (lidar nelkul)."""
    szenzorok = [
        o["sensor_left"]["intensity"],
        o["sensor_center"]["intensity"],
        o["sensor_right"]["intensity"],
    ]
    one_hot = [0.0] * len(ANGLES_FOK)
    one_hot[elozo_akcio] = 1.0
    return np.concatenate([np.array(szenzorok + one_hot, dtype=np.float32), memoria.vektor()])


def fut(
    kliens_nyers: Any, modell: Any, max_lepes: int, max_utkozes: int, naplo_gyakorisag: int = 10
) -> FutasStatisztika:
    import tensorflow as tf

    q_elore = tf.function(lambda x: modell(x, training=False))
    kliens = FelvevoKliens(kliens_nyers)
    stat = FutasStatisztika()
    memoria = Memoria()
    elozo_akcio = ANGLES_FOK.index(0.0)
    elozo_szog = None  # az utolso NN-lepes fordulasa

    # Tiszta kezdoallapot (ugyanaz, mint a baseline_line_follower.py-ban).
    if kliens.kuld({"command": "get_status"}).get("state") == "ERROR":
        kliens.kuld({"command": "reset_error"})
    kliens.kuld({"command": "stop"})
    kliens.kuld({"command": "reset_position"})
    kliens.ures_felvett()

    nn_lepes = 0
    akadaly_lepes = 0
    while stat.lepesek_szama < max_lepes:
        o = kliens.kuld({"command": "observe"})
        stat.utkozesek_szama = int(o.get("collision_count", 0) or 0)
        # Leesett-e a rover a palyarol? (csak leallitasra/diagnosztikara hasznaljuk, a halo nem lathatja)
        poz_y = (o.get("position") or {}).get("y", 0.0)
        if poz_y < -1.0:
            print(f"Leallitva: a rover leesett a palyarol (y={poz_y:.1f}) a(z) {stat.lepesek_szama}. lepesnel.")
            break
        if stat.lepesek_szama % naplo_gyakorisag == 0:
            szen = [o["sensor_left"]["intensity"], o["sensor_center"]["intensity"], o["sensor_right"]["intensity"]]
            print(
                f"lepes={stat.lepesek_szama:4d}  szenzor(B,K,J)={szen[0]:.2f},{szen[1]:.2f},{szen[2]:.2f}  "
                f"lidar_min={min(o.get('lidar_szektor_min') or [99]):.2f}  utkozes={stat.utkozesek_szama}",
                flush=True,
            )
        if stat.utkozesek_szama >= max_utkozes:
            print(f"Leallitva: {stat.utkozesek_szama} utkozes elerte a hatart ({max_utkozes}).")
            break

        # --- akadaly elol: a kezzel irt akadalykerules veszi at, amig kesz nem lesz ---
        if akadaly_kell_kerulni(o):
            lepesek, _ = akadaly_makro(kliens, memoria, stat, max_utkozes)
            akadaly_lepes += lepesek
            stat.lepesek_szama += max(lepesek, 1)
            elozo_akcio = ANGLES_FOK.index(0.0)
            elozo_szog = None
            continue

        # --- oldalt levo akadaly: kis kitero, hogy a halo ne sodorja bele a rovert ---
        irany = elharitas_szukseges(o)
        if irany:
            elharitas(kliens, memoria, irany)
            stat.lepesek_szama += 1
            akadaly_lepes += 1
            elozo_akcio = ANGLES_FOK.index(0.0)
            elozo_szog = None
            continue

        # --- neuralis halo lepese ---
        if elozo_szog is not None:
            memoria.frissit(elozo_szog, LEPESHOSSZ_M, vonalon(o))
        obs = observacio(o, elozo_akcio, memoria)
        akcio = int(np.argmax(q_elore(obs[np.newaxis, :]).numpy()[0]))
        szog = ANGLES_FOK[akcio]
        if abs(szog) >= 1.0:
            kliens.kuld({"command": "turn", "angle_deg": szog, "max_angular_speed": FORDULASI_SEBESSEG})
        kliens.kuld({"command": "move", "distance_m": LEPESHOSSZ_M, "max_speed": MOZGASI_SEBESSEG})
        kliens.ures_felvett()
        elozo_akcio = akcio
        elozo_szog = szog if abs(szog) >= 1.0 else 0.0
        nn_lepes += 1
        stat.lepesek_szama += 1

    print(
        f"Kesz: {stat.lepesek_szama} lepes (halo: {nn_lepes}, akadalykerules: {akadaly_lepes}), "
        f"akadalykerulesek: {stat.akadaly_kerulesek_szama}, utkozesek: {stat.utkozesek_szama}"
    )
    return stat


def main() -> int:
    parser = argparse.ArgumentParser(description="Hibrid vezerlo: halo + kezi akadalykerules")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--modell-fajl", default=str(Path(__file__).resolve().parent.parent / "models" / "hibrid.keras")
    )
    parser.add_argument("--max-lepes", type=int, default=1500)
    parser.add_argument("--max-utkozes", type=int, default=5)
    args = parser.parse_args()

    modell_fajl = Path(args.modell_fajl)
    if not modell_fajl.exists():
        print(f"HIBA: nem talalhato a modell fajl: {modell_fajl}")
        return 1

    from tensorflow import keras

    from adapter.backend import UnityTcpBackend

    print(f"Modell betoltese: {modell_fajl}")
    modell = keras.models.load_model(modell_fajl)
    print(f"Csatlakozas: {args.host}:{args.port}")
    backend = UnityTcpBackend(host=args.host, port=args.port)
    try:
        fut(backend, modell, args.max_lepes, args.max_utkozes)
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())