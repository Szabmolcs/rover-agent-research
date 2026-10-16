#!/usr/bin/env python3
"""M14: DQN tanitas a teljes vezerleshez (vonalkoveto + akadalykerulo +
GPS-pont alapu navigacio egyben), TensorFlow/Keras-szal, a gyors
MockBackend kornyezetben (lasd rl_env.py).

Hasznalat:
    python3 controllers/train_dqn.py --epizodok 2000
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

_PROJEKT_GYOKER = str(Path(__file__).resolve().parent.parent)
if _PROJEKT_GYOKER not in sys.path:
    sys.path.insert(0, _PROJEKT_GYOKER)

from controllers.rl_env import MAX_ELTERES_M, MAX_UTKOZES_ALAPERTELMEZETT, RoverEnv

ALAPERTELMEZETT_MODELL_FAJL = Path(__file__).resolve().parent.parent / "models" / "rl_policy.keras"


def halo_epit(obs_dim: int, n_actions: int) -> keras.Model:
    modell = keras.Sequential(
        [
            keras.layers.Input(shape=(obs_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(n_actions, activation="linear"),
        ]
    )
    modell.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return modell


class ReplayPuffer:
    def __init__(self, kapacitas: int = 50_000) -> None:
        self.puffer: deque = deque(maxlen=kapacitas)

    def hozzaad(self, obs, action, reward, next_obs, done) -> None:
        self.puffer.append((obs, action, reward, next_obs, done))

    def mintavetel(self, meret: int):
        batch = random.sample(self.puffer, meret)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(action, dtype=np.int32),
            np.array(reward, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.puffer)


def epsilon_greedy(q_ertekek: np.ndarray, epsilon: float, n_actions: int) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return int(np.argmax(q_ertekek))


def tanit_lepes(online_halo, target_halo, puffer, batch_meret, gamma) -> float:
    obs, action, reward, next_obs, done = puffer.mintavetel(batch_meret)

    kovetkezo_q = target_halo(next_obs, training=False).numpy()
    cel_q = reward + gamma * np.max(kovetkezo_q, axis=1) * (1.0 - done)

    aktualis_q = online_halo(obs, training=False).numpy()
    for i, a in enumerate(action):
        aktualis_q[i, a] = cel_q[i]

    tortenet = online_halo.fit(obs, aktualis_q, verbose=0, epochs=1)
    return float(tortenet.history["loss"][0])


def epit_tanit_fuggveny(online_halo, target_halo, n_actions: int, tanulasi_rata: float = 5e-4, tau: float = 0.01):
    """Lefordított (tf.function) tanulasi lepes: Double DQN + Huber-veszteseg +
    gradiens-vagas + a celhalo folyamatos (lagy) kovetese - ez sokkal stabilabb,
    mint a ritka teljes masolas es az MSE."""
    optimalizalo = keras.optimizers.Adam(learning_rate=tanulasi_rata, clipnorm=10.0)
    huber = keras.losses.Huber(delta=1.0)

    @tf.function
    def tanit(obs, action, reward, next_obs, done, gamma):
        # Double DQN: az online halo valasztja a legjobb kovetkezo akciot, a celhalo ertekeli
        legjobb_kov = tf.argmax(online_halo(next_obs, training=False), axis=1)
        kov_q_target = target_halo(next_obs, training=False)
        kov_q = tf.reduce_sum(kov_q_target * tf.one_hot(legjobb_kov, n_actions), axis=1)
        cel_q = reward + gamma * kov_q * (1.0 - done)
        with tf.GradientTape() as szalag:
            q = online_halo(obs, training=True)
            q_akcio = tf.reduce_sum(q * tf.one_hot(action, n_actions), axis=1)
            veszteseg = huber(cel_q, q_akcio)
        gradiensek = szalag.gradient(veszteseg, online_halo.trainable_variables)
        optimalizalo.apply_gradients(zip(gradiensek, online_halo.trainable_variables))
        # lagy celhalo-frissites
        for tv, ov in zip(target_halo.trainable_variables, online_halo.trainable_variables):
            tv.assign(tau * ov + (1.0 - tau) * tv)
        return veszteseg

    return tanit


def greedy_ertekeles(env, q_elore, n: int) -> tuple[float, float]:
    """n epizod veletlen kiserletezes (epsilon) NELKUL. Visszaadja: (sikeres arany, atlag GPS pont)."""
    sikeres = 0
    gps_osszeg = 0.0
    for _ in range(n):
        obs = env.reset()
        done = False
        while not done:
            akcio = int(np.argmax(q_elore(obs[np.newaxis, :]).numpy()[0]))
            obs, _, done, info = env.step(akcio)
        sikeres += int(info["done_ok"])
        gps_osszeg += info["elert_gps_pontok"]
    return sikeres / n, gps_osszeg / n


def main() -> int:
    parser = argparse.ArgumentParser(description="M14: DQN tanitas a teljes rover-vezerleshez")
    parser.add_argument("--epizodok", type=int, default=2000)
    parser.add_argument("--szcenario", default=None)
    parser.add_argument("--max-lepes", type=int, default=900)
    parser.add_argument("--batch-meret", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--target-frissites", type=int, default=0,
                        help="Hard celhalo-masolas N epizodonkent (0 = kikapcsolva, a lagy frissites eleg).")
    parser.add_argument("--jutalom-skala", type=float, default=0.05,
                        help="A jutalmakat ennyivel szorozza a tanulashoz (a nagy ertekek instabilak).")
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--naplo-gyakorisag", type=int, default=20)
    parser.add_argument("--modell-fajl", default=str(ALAPERTELMEZETT_MODELL_FAJL))
    parser.add_argument("--min-puffer", type=int, default=500)
    parser.add_argument("--tanitas-gyakorisag", type=int, default=8,
                        help="Hanyadik lepesenkent tanuljon a halo (nagyobb = gyorsabb futas).")
    parser.add_argument("--max-utkozes", type=int, default=MAX_UTKOZES_ALAPERTELMEZETT,
                        help="Ennyi utkozes utan veget er az epizod (akadalyos tanitasnal legyen nagy).")
    parser.add_argument("--max-elteres", type=float, default=MAX_ELTERES_M,
                        help="Ennyi meternel messzebb a kozepvonaltol veget er az epizod "
                        "(akadalyos tanitasnal 2.0 ajanlott, hogy legyen hely kikerulni).")
    parser.add_argument("--folytat", default=None,
                        help="Mar betanitott modell fajl, amirol a tanitas folytatodik "
                        "(pl. models/rl_policy.keras).")
    parser.add_argument("--kurzus", action="store_true",
                        help="Fokozatos tanitas: 0 akadallyal indul, es egyesevel adja hozza az akadalyokat, "
                        "amint az aktualis szint sikeres.")
    parser.add_argument("--kurzus-kuszob", type=float, default=0.7,
                        help="Ekkora siker-aranynal (utolso --kurzus-ablak epizod) lep a kovetkezo szintre.")
    parser.add_argument("--kurzus-ablak", type=int, default=50)
    parser.add_argument("--indulasi-zaj", type=float, default=0.0,
                        help="Veletlen indulasi eltérés (m): oldalirany +-ennyi, irany +-40x ennyi fok. "
                        "Pl. 0.15 -> nem egyetlen utvonalat tanul meg fejbol.")
    parser.add_argument("--helyreallitas", action="store_true",
                        help="Veletlen indulas a palya barmely pontjan, a vonaltol eltolva es elforgatva "
                        "(a visszatalalas tanitasa - hibrid modhoz).")
    parser.add_argument("--kezi-akadaly", type=int, default=0,
                        help="1 = hibrid mod: az akadalykerulest a kezzel irt logika vegzi, a halo csak a tobbit.")
    parser.add_argument("--lidar-bemenet", type=int, default=1,
                        help="1 = a halo megkapja a lidart (teljes tanult mod); "
                        "0 = nem (hibrid mod: az akadalykerules kezi logika).")
    parser.add_argument("--ertekeles-gyakorisag", type=int, default=10,
                        help="Hany epizodonkent probalja ki a halot veletlen kiserletezes nelkul "
                        "(a legjobb allapotot menti el).")
    parser.add_argument("--ertekeles-epizodok", type=int, default=5)
    parser.add_argument("--kurzus-greedy-kuszob", type=float, default=0.8,
                        help="Kurzusnal: ekkora kiserletezes-mentes siker kell a kovetkezo szinthez.")
    parser.add_argument("--kurzus-egymas", type=int, default=2,
                        help="Kurzusnal: ennyi egymas utani sikeres kiertekeles kell a szintlepeshez.")
    parser.add_argument("--cel-egymas", type=int, default=5,
                        help="Ennyi egymas utani sikeres (>=90%%) kiertekeles utan leall a tanitas "
                        "a teljes akadalyszinten.")
    parser.add_argument("--cel-siker-arany", type=float, default=0.89)
    parser.add_argument("--siker-ablak", type=int, default=100)
    args = parser.parse_args()

    env = RoverEnv(
        szcenario_fajl=args.szcenario,
        max_lepes=args.max_lepes,
        max_utkozes=args.max_utkozes,
        max_elteres_m=args.max_elteres,
        indulasi_zaj=args.indulasi_zaj,
        helyreallitas=args.helyreallitas,
        lidar_bemenet=bool(args.lidar_bemenet),
        kezi_akadaly=bool(args.kezi_akadaly),
    )
    online_halo = halo_epit(env.obs_dim, env.n_actions)
    if args.folytat:
        folytat_fajl = Path(args.folytat)
        if not folytat_fajl.exists():
            print(f"HIBA: nem talalhato a folytatando modell: {folytat_fajl}")
            return 1
        online_halo.set_weights(keras.models.load_model(folytat_fajl).get_weights())
        print(f"Folytatas a mentett modellrol: {folytat_fajl}", flush=True)
    target_halo = halo_epit(env.obs_dim, env.n_actions)
    target_halo.set_weights(online_halo.get_weights())

    # lefordított (gyors) elorecsatolas a lepesenkenti akcio-valasztashoz
    q_elore = tf.function(lambda x: online_halo(x, training=False))
    tanit = epit_tanit_fuggveny(online_halo, target_halo, env.n_actions, tau=args.tau)
    gamma_t = tf.constant(args.gamma, dtype=tf.float32)

    puffer = ReplayPuffer()
    epsilon = args.epsilon_start
    osszes_lepes = 0

    kor_aranyok: list[float] = []
    utkozesek_listaja: list[int] = []
    sikerek: list[bool] = []

    osszes_akadaly = len(env.akadalyok_sablon)
    if args.kurzus:
        env.aktiv_akadalyok = 0
    szint_kezdete = 0
    legjobb_pontszam = None
    legjobb_suly = None
    sikeres_egymas = 0  # kurzus: egymas utani sikeres kiertekelesek az aktualis szinten
    cel_egymas = 0  # teljes szinten egymas utani sikeres kiertekelesek
    modell_fajl = Path(args.modell_fajl)
    modell_fajl.parent.mkdir(parents=True, exist_ok=True)
    print("Tanitas indul...", flush=True)

    for epizod in range(1, args.epizodok + 1):
        obs = env.reset()
        done = False
        osszes_reward = 0.0

        while not done:
            q_ertekek = q_elore(obs[np.newaxis, :]).numpy()[0]
            action = epsilon_greedy(q_ertekek, epsilon, env.n_actions)

            next_obs, reward, done, info = env.step(action)
            puffer.hozzaad(obs, action, reward, next_obs, done)
            obs = next_obs
            osszes_reward += reward
            osszes_lepes += 1

            if len(puffer) >= args.min_puffer and osszes_lepes % args.tanitas_gyakorisag == 0:
                o_b, a_b, r_b, n_b, d_b = puffer.mintavetel(args.batch_meret)
                tanit(o_b, a_b, r_b * args.jutalom_skala, n_b, d_b, gamma_t)

        kor_aranyok.append(info["kor_arany"])
        utkozesek_listaja.append(info["utkozesek"])
        sikerek.append(info["done_ok"])

        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)

        if args.target_frissites and epizod % args.target_frissites == 0:
            target_halo.set_weights(online_halo.get_weights())

        if epizod % args.naplo_gyakorisag == 0:
            ablak = args.naplo_gyakorisag
            print(
                f"epizod={epizod:5d}  epsilon={epsilon:.3f}  "
                f"atlag_kor_arany={sum(kor_aranyok[-ablak:]) / ablak:.3f}  "
                f"atlag_utkozes={sum(utkozesek_listaja[-ablak:]) / ablak:.2f}  "
                f"gps_pontok={info['elert_gps_pontok']:3d}  "
                f"siker_arany={sum(sikerek[-ablak:]) / ablak:.2f}  "
                f"reward={osszes_reward:.1f}",
                flush=True,
            )

        if epizod % args.ertekeles_gyakorisag == 0:
            siker_e, gps_e = greedy_ertekeles(env, q_elore, args.ertekeles_epizodok)
            szint = env.aktiv_akadalyok
            pontszam = (siker_e, gps_e)
            if legjobb_pontszam is None or pontszam > legjobb_pontszam:
                legjobb_pontszam = pontszam
                legjobb_suly = online_halo.get_weights()
                online_halo.save(modell_fajl)
                print(f"  * uj legjobb ({szint}/{osszes_akadaly} akadaly): kiserletezes nelkuli siker="
                      f"{siker_e:.0%}, atlag GPS={gps_e:.1f}  [mentve]", flush=True)

            sikeres_egymas = sikeres_egymas + 1 if siker_e >= args.kurzus_greedy_kuszob else 0
            if args.kurzus and szint < osszes_akadaly:
                if sikeres_egymas >= args.kurzus_egymas:
                    env.aktiv_akadalyok += 1
                    szint_kezdete = len(sikerek)
                    epsilon = max(epsilon, 0.4)
                    legjobb_pontszam = None
                    legjobb_suly = None
                    sikeres_egymas = 0
                    print(f"  -> kovetkezo szint: {env.aktiv_akadalyok}/{osszes_akadaly} akadaly "
                          f"(epizod {epizod})", flush=True)
            else:
                cel_egymas = cel_egymas + 1 if siker_e >= 0.9 else 0
                if cel_egymas >= args.cel_egymas:
                    print(f"\nElerte a celt: {args.cel_egymas} egymas utani kiertekeles >=90% sikerrel "
                          f"a teljes akadalyszinten (epizod {epizod}).", flush=True)
                    break

        kurzus_kesz = (not args.kurzus) or env.aktiv_akadalyok >= osszes_akadaly
        if kurzus_kesz and len(sikerek) - (szint_kezdete if args.kurzus else 0) >= args.siker_ablak:
            gordulo_siker = sum(sikerek[-args.siker_ablak:]) / args.siker_ablak
            if gordulo_siker >= args.cel_siker_arany:
                print(
                    f"\nElerte a celt: az utolso {args.siker_ablak} epizod "
                    f"{gordulo_siker:.1%} sikerrel zarult (cel: {args.cel_siker_arany:.0%}). "
                    f"Tanitas leallitva {epizod}. epizodnal.",
                    flush=True,
                )
                break

    if legjobb_suly is not None:
        online_halo.set_weights(legjobb_suly)
    online_halo.save(modell_fajl)
    print(f"\nModell elmentve (a legjobb allapot): {modell_fajl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())