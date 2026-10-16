#!/usr/bin/env python3
"""Sebessegmeres: megmutatja, melyik resz lassu a tanitasban.

Hasznalat (a projekt gyokereben):
    python3 controllers/profil.py
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.rl_env import RoverEnv
from controllers.train_dqn import ReplayPuffer, epit_tanit_fuggveny, halo_epit, tanit_lepes


def main() -> None:
    env = RoverEnv()

    # 1) szimulalt lepes (TensorFlow nelkul, veletlen akciokkal)
    t0 = time.perf_counter()
    lepesek = 0
    for _ in range(5):
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(random.randrange(env.n_actions))
            lepesek += 1
    t_env = time.perf_counter() - t0
    print(f"1) Szimulalt lepes: {1000 * t_env / lepesek:.2f} ms/lepes "
          f"(5 epizod, atlag {lepesek / 5:.0f} lepes/epizod)")

    # 2) halo kiertekelese lepesenkent
    halo = halo_epit(env.obs_dim, env.n_actions)
    q_elore = tf.function(lambda x: halo(x, training=False))
    x = np.zeros((1, env.obs_dim), dtype=np.float32)
    q_elore(x)  # bemelegites
    t0 = time.perf_counter()
    for _ in range(200):
        q_elore(x).numpy()
    t_q = (time.perf_counter() - t0) / 200
    print(f"2) Halo kiertekeles: {1000 * t_q:.2f} ms/lepes")

    # 3) egy tanulasi lepes
    target = halo_epit(env.obs_dim, env.n_actions)
    puffer = ReplayPuffer()
    obs = env.reset()
    for _ in range(600):
        a = random.randrange(env.n_actions)
        nobs, r, d, _ = env.step(a)
        puffer.hozzaad(obs, a, r, nobs, d)
        obs = env.reset() if d else nobs
    tanit_lepes(halo, target, puffer, 64, 0.95)  # bemelegites
    t0 = time.perf_counter()
    for _ in range(20):
        tanit_lepes(halo, target, puffer, 64, 0.95)
    t_fit = (time.perf_counter() - t0) / 20
    print(f"3) Regi tanulasi lepes (.fit): {1000 * t_fit:.1f} ms")

    # 4) uj, lefordított tanulasi lepes
    tanit = epit_tanit_fuggveny(halo, target, env.n_actions)
    gamma_t = tf.constant(0.95, dtype=tf.float32)
    tanit(*puffer.mintavetel(64), gamma_t)  # bemelegites
    t0 = time.perf_counter()
    for _ in range(50):
        tanit(*puffer.mintavetel(64), gamma_t)
    t_uj = (time.perf_counter() - t0) / 50
    print(f"4) UJ tanulasi lepes (tf.function): {1000 * t_uj:.1f} ms (minden 8. lepesnel fut)")

    atlag_hossz = lepesek / 5
    becsles = atlag_hossz * (t_env / lepesek + t_q + t_uj / 8)
    print(f"\nBecsles: kb. {becsles:.1f} s/epizod, 20 epizod = {20 * becsles / 60:.1f} perc "
          f"(random akciokkal mert epizodhosszal)")


if __name__ == "__main__":
    main()