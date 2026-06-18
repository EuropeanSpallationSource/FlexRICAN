# AHU_03 Control — Technical Documentation

**Project:** 10AHUs · **Unit:** AHU_03 (canvas group **"DeHumi_Water_El_afterheating"**)
**Source selection:** `10ahus_selection_AHU3.json`

| File | Platform | Role |
|------|----------|------|
| `AHU_03.st` | CoDeSys 3.5, IEC 61131-3 Structured Text | PLC control program (single `PROGRAM`) |
| `ahu_03_control.py` | Python 3.8+ | Soft-PLC control program for a virtual PC (alternative to the licensed CoDeSys runtime) |

Both files implement the same logic on a **200 ms** cycle. AHU_03 is AHU_02 plus an electric after-heater.

---

## 1. What is different from AHU_02

AHU_03 keeps everything from AHU_02 — sequence state machine, VFD fans with tracking, rotary heat-recovery wheel, modulating dampers, water heating coil with frost protection, cooling coil, supply low-limit, dehumidification, cascade temperature control — and adds an **electric after-heater coil** mounted downstream of the cooling coil (the group name reads *DeHumi · Water · El_afterheating*).

The air path is: **wheel → water heating coil → cooling coil → electric after-heater → supply fan.**

This adds:

1. **A third heating stage.** Heating capacity is sequenced cheapest-first: recovery wheel → water coil → electric coil.
2. **Electric reheat for dehumidification.** When drying the air, the electric after-heater (not the water coil) reheats the over-cooled supply air — exactly what an after-heater is for.
3. **Electric-coil safety**, which a water coil does not need:
   - **Airflow interlock** — the electric coil can only energize in `RUN` with proven supply airflow.
   - **Overheat thermostat** — a latched input that disables the coil and raises the common alarm until acknowledged.
   - **Fan cool-down overrun** — on shutdown the fans keep running (and dampers stay open) for a cool-down period to dissipate residual element heat before stopping.

Everything else matches the AHU_02 documentation.

---

## 2. Operating sequence

Identical states to AHU_01/02: `OFF → STARTUP → RUN → SHUTDOWN`, with `FROST`, `FIRE`, `FAULT` at priority **FIRE > FROST > FAULT**. The only change is in **SHUTDOWN**: the unit holds the fans at minimum speed with the outdoor/exhaust dampers open until both the purge time and the electric cool-down timer have elapsed, then goes to `OFF`. The electric coil is forced off the moment shutdown begins.

---

## 3. I/O list (additions over AHU_02)

### New digital input

| Signal | Meaning |
|--------|---------|
| **`DI_ElHeaterOverheat`** | Electric coil overheat thermostat (latched → coil disabled, alarm) |

### New digital output

| Signal | Meaning |
|--------|---------|
| **`DO_ElHeater_Enable`** | Electric coil safety contactor (energized only when the coil is actually heating) |

### New analog output

| Signal | Meaning |
|--------|---------|
| **`AO_ElHeater_Pct`** | Electric after-heater modulation 0…100 % (`AO_0_10V`) |

All other inputs and outputs are as documented for AHU_02 (fire, fan run/fault, filters, supply low-limit, reset, remote enable; inlet/supply/exhaust/room/return-water temperatures and humidities; fan, damper, wheel, water-valve, cool-valve outputs; pump and common-alarm).

---

## 4. Parameters (additions over AHU_02)

Marked **(default)** = commissioning value, not in the JSON.

| Group | Parameter | Value | Origin |
|-------|-----------|-------|--------|
| Electric coil | Output range | 0 / 100 % | `valveType = AO_0_10V` |
| | Cool-down overrun | 60 s | **(default)** |
| Heating split (with recovery) | wheel / water / electric bands | 0–40 / 40–75 / 75–100 % | **(default)** |
| Heating split (no recovery) | water / electric bands | 0–60 / 60–100 % | **(default)** |

All AHU_02 parameters (sequence, fans, wheel, water frost trip 7/10 °C, pump run-on 120 s, supply low-limit 5 °C, PI gains, dehumidification 50 %/5 %/60 %) are unchanged.

---

## 5. Three-stage heating sequence

In `RUN`, the cascade (room PI → supply setpoint 12…30 °C → supply PI) produces one sequencer demand of −100…+100 % exactly as before. The positive (heating) half is now split across three devices so the cheapest energy is used first:

```
 heating authority   0% ─────── 40% ─────── 75% ─────── 100%
  (recovery useful)   │ wheel 0→100 │ water 0→100 │ electric 0→100 │

 heating authority   0% ─────────────── 60% ─────────── 100%
  (no recovery)       │   water 0→100    │ electric 0→100 │
```

The recovery wheel is only used when the extract air is warmer than the outdoor air by more than 1 K. When recovery is not beneficial the wheel stays off and the band collapses to water → electric. The cooling half is unchanged from AHU_02 (wheel for free cooling, then cooling valve). The stage split points are **defaults** to tune.

---

## 6. Dehumidification with electric reheat

When the unit is running and room humidity exceeds setpoint + deadband (default 50 % + 5 %):

- The **cooling valve is forced to at least 60 %** to condense moisture on the cooling coil.
- The **electric after-heater reheats** the over-cooled air back to the supply-temperature setpoint (proportional reheat). The water coil stays closed during dehumidification.

This is the designed role of the after-heater in this unit ("El_afterheating").

---

## 7. Electric-coil safety

The electric coil is governed by three independent protections, applied after the sequencing logic:

- **Airflow interlock:** `AO_ElHeater_Pct` is forced to 0 unless the unit is in `RUN` **and** supply-fan run feedback (`DI_FanSupply_Run`) is present. No airflow ⇒ no electric heat.
- **Overheat:** `DI_ElHeaterOverheat` latches, forces the coil off, and raises the common alarm; it clears only after the thermostat resets and the operator presses reset.
- **Cool-down overrun:** a timer tracks the time since the coil last switched off. On shutdown the fans and dampers stay active until 60 s have elapsed, dissipating residual element heat before the unit stops.

`DO_ElHeater_Enable` (the safety contactor) is energized only while the coil is actually modulating above zero, giving a hard electrical interlock in addition to the analog signal.

The water-coil frost protection (return-water 7/10 °C, supply low-limit 5 °C or hardware stat, pump run-on 120 s) is unchanged from AHU_02 and still trips the protective `FROST` state.

---

## 8. Deploying & running

**ST (`AHU_03.st`):** in CoDeSys 3.5 add the `E_AHU_STATE` DUT (listed at the bottom of the file), add a PROGRAM `AHU_03` with the body, and map the I/O including the new `DI_ElHeaterOverheat`, `DO_ElHeater_Enable` and `AO_ElHeater_Pct` channels. Run it in a 200 ms cyclic task. No external libraries (inline PI + standard `TON`).

**Python (`ahu_03_control.py`):** run standalone with the built-in simulator:

```bash
python3 ahu_03_control.py
```

It prints state, room/supply temperatures, room humidity, fan speed, wheel/water/electric/cool positions, a DEHUM flag and alarm status. Structure mirrors the earlier units (`IOImage`, `AHUController.scan()`, `PlantSimulator`, `run()`); for field use replace `PlantSimulator` with a real I/O driver and keep `scan()` unchanged. Use `run(cycles=N, realtime=False)` for fast headless testing.

---

## 9. Verified behavior

Exercised with the built-in simulator and deterministic scans:

- **Maximum heating:** all three stages engage in order — wheel to 100 %, then water valve, then electric coil.
- **Moderate heating:** only the recovery wheel modulates; water and electric stay off (free energy first).
- **Dehumidification:** cooling opens to the dehumidify minimum and the **electric** coil reheats while the water coil stays closed.
- **Electric overheat:** the coil is forced off and the alarm latches until reset.
- **Airflow interlock:** loss of supply-fan proof drives the electric output to 0 immediately.
- **Cool-down overrun:** after a stop request the unit stays in `SHUTDOWN` with fans running and the electric coil off, then reaches `OFF` once the 60 s cool-down elapses.
- **Fire / frost:** all dampers close, electric and fans off, common alarm raised; frost still trips on return-water or supply low-limit.

Both programs make the same decisions in the same order, so the ST program is expected to reproduce this on the PLC. Re-validate on the real plant, especially the **(default)** parameters.

> **Note on defaults.** The heating-stage split points, the electric cool-down time, the recovery benefit threshold, the dehumidification setpoints, the supply low-limit value, the PI gains, the fan ramp and the proof-of-flow time are not specified in the JSON selection. The values used are reasonable starting points only and must be tuned to the actual AHU during commissioning.
