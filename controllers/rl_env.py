#!/usr/bin/env python3
"""M14: gyors, Unity nelkuli tanito kornyezet a teljes vezerlest tanulo
neuralis halohoz (lasd train_dqn.py).

A kor GPS-pontokra van osztva (WAYPOINT_SZAM db, egyenletesen elosztva).
A policy akkor kap NAGY jutalmat, ha eleri a kovetkezo GPS pontot, es KICSI
jutalmat minden lepesnel, amig a felfestett uton marad. Buntetes NINCS.

FONTOS: a policy bemenete KIZAROLAG szenzoradat (sensor_left/center/right +
lidar_szektor_min) + az elozo lepesben valasztott akcio one-hot kodja - SOHA
nem kapja meg a position-t vagy a GPS-pont indexet. A position-t csak a
jutalom kiszamitasahoz hasznaljuk a kornyezetben.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_PROJEKT_GYOKER = str(Path(__file__).resolve().parent.parent)
if _PROJEKT_GYOKER not in sys.path:
    sys.path.insert(0, _PROJEKT_GYOKER)

from controllers.rl_obs import MEMORIA_DIM, Memoria, vonalon_van
from controllers.baseline_line_follower import FutasStatisztika
from controllers.hibrid_kozos import (
    FelvevoKliens,
    akadaly_kell_kerulni,
    akadaly_makro,
    elharitas,
    elharitas_szukseges,
)
from adapter.backend import (
    LIDAR_MAX_HATOTAV_M,
    PALYA_IV_SUGAR_M,
    PALYA_VONAL_SZELESSEG_M,
    MockAkadaly,
    MockBackend,
    tavolsag_a_kozepvonaltol,
)

ANGLES_FOK = [-20.0, -10.0, -4.0, 0.0, 4.0, 10.0, 20.0]
LEPESHOSSZ_M = 0.08
FORDULASI_SEBESSEG = 20.0
MOZGASI_SEBESSEG = 0.20

MAX_LEPES_ALAPERTELMEZETT = 900  # a kor ~49 m = ~615 lepes, ezert kell ennel tobb
MAX_UTKOZES_ALAPERTELMEZETT = 8  # ennyi utkozes utan veget er az epizod (nem buntetes)
MAX_ELTERES_M = 1.5  # ennyire eltavolodva a kozepvonaltol veget er az epizod (nem buntetes)

WAYPOINT_SZAM = 24  # minden 15 fok egy "GPS pont"
CELPONT_JUTALOM = 20.0
UTVONAL_JUTALOM = 0.05
KOR_BEFEJEZES_JUTALOM = 30.0


PALYA_EGYENES_M = 12.0  # = adapter.backend.PALYA_EGYENES_HOSSZ_M
PALYA_SUGAR_M = 4.0  # = adapter.backend.PALYA_IV_SUGAR_M


def palya_pont(s_m: float) -> tuple[float, float, float, float, float]:
    """A stadion kozepvonalanak egy pontja a (4, -6) ponttol merve, a menetirany szerint
    (jobb egyenes felfele, felso iv, bal egyenes lefele, also iv).
    Visszaadja: (x, z, irany_fok, kifele_x, kifele_z). A kifele mutato egysegvektor a
    menetirany JOBB oldala."""
    fel = PALYA_EGYENES_M / 2.0
    r = PALYA_SUGAR_M
    iv = math.pi * r
    hossz = 2 * PALYA_EGYENES_M + 2 * iv
    s = s_m % hossz
    if s < PALYA_EGYENES_M:
        return r, -fel + s, 0.0, 1.0, 0.0
    s -= PALYA_EGYENES_M
    if s < iv:
        fi = s / r
        return r * math.cos(fi), fel + r * math.sin(fi), (-math.degrees(fi)) % 360.0, math.cos(fi), math.sin(fi)
    s -= iv
    if s < PALYA_EGYENES_M:
        return -r, fel - s, 180.0, -1.0, 0.0
    s -= PALYA_EGYENES_M
    fi = math.pi + s / r
    return r * math.cos(fi), -fel + r * math.sin(fi), (-math.degrees(fi)) % 360.0, math.cos(fi), math.sin(fi)


def _akadalyok_betoltese(szcenario_fajl: Path | None) -> list[MockAkadaly]:
    if szcenario_fajl is None or not szcenario_fajl.exists():
        return []

    dokumentum = json.loads(szcenario_fajl.read_text(encoding="utf-8"))
    akadalyok: list[MockAkadaly] = []
    for a in dokumentum.get("obstacles", []):
        poz = a["position_m"]
        meret = a["size_m"]
        sugar = max(meret["x"], meret["z"]) / 2.0
        akadalyok.append(MockAkadaly(x=poz["x"], z=poz["z"], sugar_m=sugar))
    return akadalyok


@dataclass
class LepesInfo:
    kor_arany: float
    utkozesek: int
    elert_gps_pontok: int
    done_ok: bool


class RoverEnv:
    """Gym-szeru interfesz: reset() -> obs, step(action_idx) -> (obs, reward, done, info)."""

    def __init__(
        self,
        szcenario_fajl: str | Path | None = None,
        max_lepes: int = MAX_LEPES_ALAPERTELMEZETT,
        max_utkozes: int = MAX_UTKOZES_ALAPERTELMEZETT,
        waypoint_szam: int = WAYPOINT_SZAM,
        max_elteres_m: float = MAX_ELTERES_M,
        indulasi_zaj: float = 0.0,
        helyreallitas: bool = False,
        lidar_bemenet: bool = True,
        kezi_akadaly: bool = False,
    ) -> None:
        # kezi_akadaly=True: ha akadaly kerul a rover ele, a kezzel irt akadalykerules (hibrid_kozos.py)
        # veszi at az iranyitast, es a halo csak az utana kovetkezo helyzeteket lathatja/tanulja
        self.kezi_akadaly = kezi_akadaly
        # helyreallitas: veletlen indulas a palya BARMELY pontjan, a vonaltol eltolva (+-2 m) es
        # elforgatva (+-75 fok) - igy tanul meg visszatalalni a vonalra (pl. akadalykerules utan)
        self.helyreallitas = helyreallitas
        # lidar_bemenet=False: a halo NEM kapja meg a lidart (a hibrid modban az akadalykerulest
        # a kezzel irt logika vegzi, a halo csak vonalat kovet es visszatalal)
        self.lidar_bemenet = lidar_bemenet
        self.max_elteres_m = max_elteres_m
        # indulasi_zaj > 0: a rover minden epizodban kicsit mashonnan/mas irannyal indul
        # (oldalirany: +-indulasi_zaj m, irany: +-40*indulasi_zaj fok), hogy a halo ne egyetlen
        # utvonalat tanuljon meg fejbol
        self.indulasi_zaj = indulasi_zaj
        self.akadalyok_sablon = _akadalyok_betoltese(
            Path(szcenario_fajl) if szcenario_fajl else None
        )
        # az akadalyok a palyan valo elhelyezkedes (indulastol szamitott szog) szerint rendezve:
        # igy a fokozatos tanitas (kurzus) mindig a legkorabbi akadallyal kezd
        self.akadalyok_sablon.sort(key=lambda a: math.atan2(a.z, a.x) % (2 * math.pi))
        self.aktiv_akadalyok = len(self.akadalyok_sablon)
        self.max_lepes = max_lepes
        self.max_utkozes = max_utkozes
        self.waypoint_szam = waypoint_szam
        self.celpont_szogek = [2 * math.pi * i / waypoint_szam for i in range(1, waypoint_szam + 1)]
        self.n_actions = len(ANGLES_FOK)
        self.obs_dim = 3 + (6 if lidar_bemenet else 0) + self.n_actions + MEMORIA_DIM
        self.reset()

    def reset(self) -> np.ndarray:
        zaj = self.indulasi_zaj
        dx = random.uniform(-zaj, zaj) if zaj > 0 else 0.0
        irany = random.uniform(-40.0 * zaj, 40.0 * zaj) if zaj > 0 else 0.0
        mem_mag = None
        if self.helyreallitas:
            px, pz, h, nx, nz = palya_pont(random.uniform(0.0, 2 * PALYA_EGYENES_M + 2 * math.pi * PALYA_SUGAR_M))
            if random.random() < 0.3:
                d, dh = 0.0, 0.0  # a vonalon indul
            else:
                d = random.uniform(-2.0, 2.0)  # oldalirany (+ = a menetirany jobb oldala)
                dh = random.uniform(-75.0, 75.0)  # elforgatas a vonal iranyahoz kepest
            self.backend = MockBackend(
                akadalyok=list(self.akadalyok_sablon[: self.aktiv_akadalyok]),
                x=px + nx * d,
                z=pz + nz * d,
                irany_fok=(h + dh) % 360.0,
            )
            if random.random() < 0.7:
                mem_mag = (dh, d)  # a hibrid futasban a memoria ismeri az eltolodast
        else:
            self.backend = MockBackend(
                akadalyok=list(self.akadalyok_sablon[: self.aktiv_akadalyok]),
                x=PALYA_IV_SUGAR_M + dx,
                irany_fok=irany % 360.0,
            )
        self.lepesszam = 0
        self.kumulalt_szog = 0.0
        self.utolso_pozicio = (self.backend.x, self.backend.z)
        self.utolso_akcio_idx = ANGLES_FOK.index(0.0)
        self.kovetkezo_celpont_idx = 1
        self.memoria = Memoria()
        if mem_mag is not None:
            self.memoria.fordulas_fok = mem_mag[0]
            self.memoria.oldalirany_m = mem_mag[1]
            self.memoria.lepesek = 10
        obs, _ = self._observe()
        return obs

    def _observe(self) -> tuple[np.ndarray, dict[str, Any]]:
        o = self.backend.kuld({"command": "observe"})
        szenzorok = [
            o["sensor_left"]["intensity"],
            o["sensor_center"]["intensity"],
            o["sensor_right"]["intensity"],
        ]
        lidar = [
            min(v, LIDAR_MAX_HATOTAV_M) / LIDAR_MAX_HATOTAV_M
            for v in o["lidar_szektor_min"]
        ]
        akcio_one_hot = [0.0] * self.n_actions
        akcio_one_hot[self.utolso_akcio_idx] = 1.0
        vektor = np.concatenate(
            [
                np.array(szenzorok + (lidar if self.lidar_bemenet else []) + akcio_one_hot, dtype=np.float32),
                self.memoria.vektor(),
            ]
        )
        return vektor, o

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        szog = ANGLES_FOK[action_idx]
        self.utolso_akcio_idx = action_idx

        if abs(szog) >= 1.0:
            self.backend.kuld(
                {"command": "turn", "angle_deg": szog, "max_angular_speed": FORDULASI_SEBESSEG}
            )
        self.backend.kuld(
            {"command": "move", "distance_m": LEPESHOSSZ_M, "max_speed": MOZGASI_SEBESSEG}
        )

        # a memoria a SAJAT kiadott parancsokbol (odometria) + a lepes utani szenzorokbol frissul
        o_elo = self.backend.kuld({"command": "observe"})
        self.memoria.frissit(
            szog if abs(szog) >= 1.0 else 0.0,
            LEPESHOSSZ_M,
            vonalon_van(
                o_elo["sensor_left"]["intensity"],
                o_elo["sensor_center"]["intensity"],
                o_elo["sensor_right"]["intensity"],
            ),
        )
        extra_lepes = 0
        if self.kezi_akadaly:
            if akadaly_kell_kerulni(o_elo):
                extra_lepes, _ = akadaly_makro(
                    FelvevoKliens(self.backend), self.memoria, FutasStatisztika(), self.max_utkozes
                )
                self.utolso_akcio_idx = ANGLES_FOK.index(0.0)
            elif elharitas_szukseges(o_elo):
                extra_lepes, _ = elharitas(FelvevoKliens(self.backend), self.memoria, elharitas_szukseges(o_elo))
                self.utolso_akcio_idx = ANGLES_FOK.index(0.0)
        obs, o = self._observe()
        x, z = o["position"]["x"], o["position"]["z"]

        elozo_szog = math.atan2(self.utolso_pozicio[1], self.utolso_pozicio[0])
        jelen_szog = math.atan2(z, x)
        delta = jelen_szog - elozo_szog
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        self.kumulalt_szog += delta
        self.utolso_pozicio = (x, z)

        tav_vonaltol = tavolsag_a_kozepvonaltol(x, z)
        utkozesszam = o["collision_count"]

        reward = 0.0

        if tav_vonaltol <= PALYA_VONAL_SZELESSEG_M / 2.0:
            reward += UTVONAL_JUTALOM

        self.lepesszam += 1 + extra_lepes
        done = False
        done_ok = False

        while (
            not done
            and self.kovetkezo_celpont_idx <= self.waypoint_szam
            and abs(self.kumulalt_szog) >= self.celpont_szogek[self.kovetkezo_celpont_idx - 1]
        ):
            reward += CELPONT_JUTALOM
            self.kovetkezo_celpont_idx += 1
            if self.kovetkezo_celpont_idx > self.waypoint_szam:
                reward += KOR_BEFEJEZES_JUTALOM
                done = True
                done_ok = True

        if not done:
            if utkozesszam >= self.max_utkozes:
                done = True
            elif tav_vonaltol > self.max_elteres_m:
                done = True
            elif self.lepesszam >= self.max_lepes:
                done = True

        info = {
            "kor_arany": abs(self.kumulalt_szog) / (2 * math.pi),
            "utkozesek": utkozesszam,
            "elert_gps_pontok": self.kovetkezo_celpont_idx - 1,
            "done_ok": done_ok,
        }
        return obs, reward, done, info
