# Tesztek - lefedettségi összefoglaló

Összesen 60 Python-teszt + 21 adapter-teszt + 10 Unity-teszt + 1 integrációs
ellenőrzés. A CI (`.github/workflows/ci.yml`) a Unity-t nem igénylő 54 tesztet,
az adapter-teszteket és a
referenciaepizód-ellenőrzést futtatja minden push-nál; a 6 fuzz teszt
élő Unity Play módot igényel, ezért csak helyben fut.

| Fájl | Tesztek | Mit fed le | Regressziós teszt korábbi hibára |
|---|---|---|---|
| `baseline_line_follower_test.py` | 15 | A vonalkövető állapotgép (VONALON / AKADALY / VISSZATALALAS / KERESES) minden átmenete, stub klienssel, Unity nélkül; a csonk a Unityhez hasonlóan `lidar_nyers` / `lidar_nyers_ervenyes` mezőt is ad, így a valódi (nyers sugaras) kerülési ág fut | M10: akadály elhagyása után VISSZATALALAS-ra vált, nem közvetlenül VONALON-ra; zsákutca-eszkaláció KERESES-re `ZSAKUTCA_AKADALY_MAX_LEPES` után; M09: keresés maximuma után pályaelhagyás jelzése; M10 kapu: vak zóna (egyenes haladás az akadály eltűnése után), iránytartás a visszatéréskor |
| `kerules_korridor_test.py` | 5 | Korridoros akadályészlelés a nyers LiDAR-ból a rover teljes szélességével | M13-előkészítés: a régi ±30°-os döntés nem látta a rover sávjába eső oldalsó akadályt |
| `kerules_oldal_test.py` | 7 | Oldalsó ellenőrzés elhaladáskor a valódi félszélességgel, tartalék szektoros döntés, a kitérés a sarkok söprési sugarán kívül kezdődik | M10 kapu: a régi 1,0 m-es oldalküszöb szabadnak látta a rover oldalától 25 cm-re levő akadályt |
| `adapter_test.py` | 21 | MCP-adapter: őrség, rejtett mezők, mock backend, nyers LiDAR és maszk, `get_status` szűrése | M13-előkészítés: a `get_status` pozíció- és ütközésszivárgása; a mock maszkja fordított volt |
| `kor_metrika_test.py` | 8 | Fő metrika: teljes kör mindkét irányban, fél kör, a pálya belsejében futó kis kör (`gyanus_kor`), pályaméret a szcenáriófájl `track` blokkjából, ütközésszámláló M10 és M11 naplóformátumban, távolságfüggvény egyezése az adapterével | Audit: a metrika beégetett pályaméretet használt és tesztje nem volt |
| `analyze_step_log_test.py` | 4 | M10 oszcilláció-elemző: pozíció olvasása mindkét naplóformátumból, "NINCS ADAT" pozíció nélküli naplóra | Audit: az M11 óta az elemző vak volt, új naplón minden futásra hamis "rendben"-t írt |
| `naplo_mezok_test.py` | 2 | A lépésnapló tartalmazza a `lidar_nyers` sugarakat; a metaadat paraméterei között ott vannak a lista típusú konstansok (`KERESES_LENGES_FOK`) | Audit: a kerülési döntések a naplóból nem voltak rekonstruálhatók, a v3 fő paramétere nem került a metaadatba |
| `kiserlet_naplo_test.py` | 5 | M11 egységes naplóséma: mezők, több lépés hozzáfűzése, seed nélküli működés, privilegizált diagnosztika alapértéke, két naplózó közös fájlba | - (új modul) |
| `replay_visualizer_test.py` | 4 | Futás betöltése run_id alapján, ütközések jelölése, hiányzó diagnosztika kezelése | M11: a "ragadós" `collision_occurred` mező nem jelölhet minden lépést ütközöttnek - a `collision_count` növekményét kell figyelni (a replay-eszköz fejlesztése közben talált hiba) |
| `scenario_seed_test.py` | 4 | Szcenárió-generálás determinisztikus seedelése; a bejegyzett szcenáriófájl egyezik a generátor kimenetével | M10 audit: a `stadium-train-baseline.json`-t mérés miatt tartósan módosították - ez a teszt azóta megakadályozza, hogy a generátor-hű fájl észrevétlenül megváltozzon |
| `protocol_fuzz_test.py` | 6 (Unity kell) | v1 protokoll: randomizált move/turn tartományok, hibás/csonka JSON, extra/duplikált mezők, idempotencia, MOVING állapotbeli második move (1300-as hibakód) | M05: a fuzz teszt által feltárt két eredeti hiba (szerver-lefagyás csonka JSON-nál, engedékeny validáció) |
| `scripts/referencia_epizod.py` | integrációs | Commitolt 500 lépéses referencia-napló metrikái egyeznek az `elvart.json`-nal, replay-kép elkészül | M11 elfogadási feltétel: referenciaepizód friss klónból |

Futtatás (Unity nélkül):

```bash
python3 tests/baseline_line_follower_test.py -v
python3 tests/kerules_korridor_test.py -v
python3 tests/kerules_oldal_test.py -v
python3 tests/kiserlet_naplo_test.py -v
python3 tests/replay_visualizer_test.py -v
python3 tests/scenario_seed_test.py -v
python3 scripts/referencia_epizod.py
```

## Unity-tesztek (`unity/Assets/Tests/`, Unity Test Runner)

| Fájl | Mód | Tesztek | Mit fed le | Regressziós teszt korábbi hibára |
|---|---|---|---|---|
| `unity/Assets/Tests/EditMode/TrackControllerGeometriaTeszt.cs` | Edit | 5 | `TrackController.TavolsagAKozepvonaltol`: egyenes, ív, pálya közepe, vonalszélesség-küszöb | M11: fantom-ív a stadion belsejében (a `TavolsagIvtol` teljes körhöz mért, nem félkörívhez; M07 óta lappangó hiba, e teszt találta) |
| `unity/Assets/Tests/PlayMode/TrackSceneTeszt.cs` | Play | 3 | `TrackScene` betöltése, kötelező komponensek; Lidar sugárszám és hatótáv; középső színszenzor és pálya-geometria konzisztenciája | M07 kalibráció: szenzor-ítélet = geometria-ítélet |
| `unity/Assets/Tests/PlayMode/MovementControllerPlayModeTests.cs` | Play | 2 | `MovementController` move és reset_position (korai mérföldkövekből) | - |

Futtatás: Unity → Window → General → Test Runner → EditMode / PlayMode →
Run All (Play mód ne fusson közben). Eredmény: 10/10 zöld
(`docs/screenshots/m11-unity-tests-editmode.png`,
`docs/screenshots/m11-unity-tests-playmode.png`). A Unity-tesztek nem
futnak a CI-ban (Unity-licenc kell hozzá) - ez M12+ nyitott pont.

Ami **nincs** lefedve automatizált teszttel: a `RoverGatewayServer`
ütközésdetektálása és protokoll-kezelése Unity-oldalról (a Python fuzz
teszt élő Unity ellen fedi), valamint a `ColorSensor` zaj/kimaradás
paraméterei (kézi kalibrációval dokumentálva: `docs/sensors.md`,
`docs/lidar.md`, `docs/m10-plan.md`).

---

# Rover Gateway v1 fuzz tesztek

A `protocol_fuzz_test.py` a futó Unity Play Mode szervert TCP-n, a v1
4 bájtos big-endian hossz-prefixes protokollon keresztül teszteli.

## Előfeltételek

1. Nyisd meg a `unity/` projektet Unityben.
2. Indítsd el a `NetworkControlScene` jelenetet Play Mode-ban.
3. Ellenőrizd, hogy a gateway a `127.0.0.1:8765` címen figyel.

## Futtatás

A tesztek csak a Python standard libraryt igénylik:

```bash
python3 tests/protocol_fuzz_test.py
```

Ha telepítve van a pytest, ugyanazok a tesztek azzal is futtathatók:

```bash
python3 -m pytest -v tests/protocol_fuzz_test.py
```

A szerver hiányakor a teljes tesztosztály skip státuszt kap. Egy sikertelen
teszt után érdemes újraindítani a Play Mode-ot, ha a rover `ERROR` állapotban
maradt.

## Beállítások

Környezeti változókkal felülírható:

- `ROVER_GATEWAY_HOST` (alapérték: `127.0.0.1`)
- `ROVER_GATEWAY_PORT` (alapérték: `8765`)
- `ROVER_GATEWAY_FUZZ_TIMEOUT` (alapérték: `2` másodperc)
- `ROVER_GATEWAY_FUZZ_CASES` (alapérték: `30` random eset/parancs)
- `ROVER_GATEWAY_FUZZ_SEED` (alapérték: `20260726`)

Gyors smoke futtatás:

```bash
ROVER_GATEWAY_FUZZ_CASES=5 python3 tests/protocol_fuzz_test.py
```

A seed kiírt/alapértelmezett értékével a randomizált esetek reprodukálhatók.
