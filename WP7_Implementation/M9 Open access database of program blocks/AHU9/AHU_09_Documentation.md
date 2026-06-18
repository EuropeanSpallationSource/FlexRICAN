# AHU_09 Control — Technical Documentation

**Project:** 10AHUs · **Unit:** AHU_09 (canvas group **"RoofTop_Heatin_Cooling"**)
**Source selection:** `10ahus_selection_AHU9.json`

| File | Platform | Role |
|------|----------|------|
| `AHU_09.st` | CoDeSys 3.5, IEC 61131-3 Structured Text | PLC control program (single `PROGRAM`) |
| `ahu_09_control.py` | Python 3.8+ | Soft-PLC control program for a virtual PC (alternative to the licensed CoDeSys runtime) |

Both files implement the same logic on a **200 ms** cycle.

---

## 1. Unit type

AHU_09 is a **roof-top heating + cooling supply unit**. It is the AHU_08 rooftop heating unit **with a cooling coil added**, so it conditions the air in both directions. Its plant is:

- one **supply VFD fan** (run at constant volume),
- a **2-point fresh-air damper** with an **OPEN feedback** start interlock,
- a **modulating exhaust / relief damper**,
- a **water heating coil** (3-way valve) with **frost protection** and a circulation pump,
- a **cooling coil** (2-way valve),
- sensors: supply T/RH, exhaust(return) temperature, room temperature, return-water temperature.

Because both a heating coil and a cooling coil are present, temperature is held by a full **cascade with split-range heat/cool** (versus the heating-only control of AHU_08). The fresh-air damper interlock and water-coil frost protection are unchanged from AHU_08.

---

## 2. Operating sequence

`OFF → STARTUP → RUN → SHUTDOWN`, with protective `FROST`, `FIRE`, `FAULT` at priority **FIRE > FROST > FAULT**.

| State | Behavior |
|-------|----------|
| **OFF** | Fan stopped, fresh-air damper closed, exhaust damper closed, valves closed. |
| **STARTUP** | Command fresh-air damper open; wait 20 s **and** the OPEN feedback; start fan, wait 10 s; on proof of flow → RUN. |
| **RUN** | Fresh-air damper open, exhaust damper at relief, fan constant volume, coils under cascade control. |
| **SHUTDOWN** | Purge 0 s → OFF. |
| **FROST** | Fan off, both dampers closed, heating valve 100 %, cooling closed, pump on, alarm latched. |
| **FIRE** | `FANS_OFF_DAMPERS_CLOSE`: fan off, dampers closed, valves closed. |
| **FAULT** | Safe stop on latched fan, sensor or damper fault. |

---

## 3. I/O list

### Digital inputs

| Signal | Meaning |
|--------|---------|
| `DI_FireAlarm` | Fire signal → `FIRE` |
| `DI_FanSupply_Run` | Supply fan run feedback / proof of flow |
| `DI_FanSupply_Fault` | Supply fan hardware fault |
| `DI_FilterSupply_Dirty` | Inlet filter ΔP switch (warning) |
| `DI_FreshDamper_Open` | 2-point fresh-air damper OPEN feedback (start interlock) |
| `DI_Reset` | Operator acknowledge/reset |
| `DI_RemoteEnable` | External enable (BMS / schedule) |

### Analog inputs

| Signal | JSON sensor | Notes |
|--------|-------------|-------|
| `AI_T_Supply_C` / `AI_RH_Supply_pct` | `th_supply` | Supply-air control + monitoring |
| `AI_T_Exhaust_C` | `th_exhaust` | Return-air monitoring |
| `AI_T_Room_C` | `temp_room` | Cascade master |
| `AI_T_WaterReturn_C` | `temp_water_return` | Frost protection |

### Digital outputs

| Signal | Meaning |
|--------|---------|
| `DO_FanSupply_Run` | Supply fan enable |
| `DO_FreshDamper_Open` | 2-point fresh-air damper command |
| `DO_HeatPump_Run` | Heating-coil circulation pump |
| `DO_CommonAlarm` | Grouped alarm lamp / BMS bit |

### Analog outputs (0…100 %)

| Signal | Meaning |
|--------|---------|
| `AO_FanSupply_Pct` | Supply VFD speed (constant in RUN) |
| `AO_DamperExhaust_Pct` | Modulating exhaust / relief damper |
| `AO_HeatValve_Pct` | Heating 3-way valve (`AO_0_10V`) |
| **`AO_CoolValve_Pct`** | **Cooling valve (`AO_0_10V`)** |

---

## 4. Parameters

Values from the JSON unless marked **(default)**.

| Group | Parameter | Value | Origin |
|-------|-----------|-------|--------|
| Sequence | Supply / room SP | 18 / 22 °C | `supplyTempSpC`, `roomTempSpC` |
| | Supply limits | 12 / 30 °C | `min/maxSupplyTempC` |
| | Damper / fan / purge waits | 20 / 10 / 0 s | `startup*`, `shutdownPurgeS` |
| Supply fan | Min / max / volume | 20 / 100 / 60 % | `minSpeedPct`, `maxSpeedPct`, `speedSpPct` |
| | Ramp / proof | 10 %/s · 15 s | **(default)** |
| Exhaust damper | Open time | 120 s | `openTimeS` |
| Fresh-air damper | Open time / proof timeout | 20 s · 40 s | `openTimeS` · **(default)** |
| Heating/frost | Trip / release | 7 / 10 °C | `tReturnTripC`, `tReturnReleaseC` |
| | Pump run-on | 120 s | `pumpRunOnS` |
| Cooling coil | Min supply temp | 12 °C | `minSupplyTempC` |
| Return water | Alarm / warning | 5/60 · 10/55 °C | sensor limits |
| Room / supply PI | Kp / Ti | 1.5/600 · 12/180 | **(default)** |
| Sequencer | Deadband | 5 % | **(default)** |

---

## 5. Temperature control (cascade + split range)

Active only in `RUN`. The supply fan runs at its constant volume setpoint (60 %). Temperature is held by a cascade with split range, identical in form to AHU_01/AHU_07:

- the **room PI** turns the room error into a supply-air setpoint clamped to 12…30 °C;
- the **supply PI** turns the supply error into a single demand of −100…+100 %;
- a 5 % deadband splits the demand: positive opens the heating valve, negative opens the cooling valve, and the two never run together.

The supply-temperature low clamp (12 °C) also keeps the cooling coil from over-cooling the supply air.

---

## 6. Interlocks, frost & alarms

- **Fresh-air damper interlock:** the fan only starts once the damper proves open; a damper that fails to prove open within the proof timeout (40 s) latches a damper fault and stops the unit. In OFF/SHUTDOWN/FROST/FIRE/FAULT the damper is commanded closed.
- **Frost:** return-water ≤ 7 °C trips `FROST` (heating valve 100 %, cooling closed, dampers closed, pump on, latched); release at ≥ 10 °C with operator reset.
- **Heating pump:** runs on heating demand or frost, with a 120 s run-on after demand ends.

| Alarm | Type | Cause | Clears when |
|-------|------|-------|-------------|
| Frost | Latched | Return-water ≤ 7 °C | Water ≥ 10 °C **and** reset |
| Fan fault | Latched | Fault input or lost proof of flow | Fault gone **and** reset |
| Damper fault | Latched | Fresh-air damper fails to prove open | Reset (re-checked next scan) |
| Sensor fault | Latched | Supply/room/return-water out of range | Reset (re-checked next scan) |
| Return-water alarm | Live | Outside 5…60 °C | Value back in range |
| Filter dirty | Warning | Filter ΔP | ΔP normal |
| Fire | Live | Fire input | Input clears |

`DO_CommonAlarm` is the OR of the four latched faults, fire and the return-water alarm.

---

## 7. Deploying & running

**ST (`AHU_09.st`):** in CoDeSys 3.5 add the `E_AHU_STATE` DUT (listed at the bottom of the file), add a PROGRAM `AHU_09` with the body, map the I/O including `DI_FreshDamper_Open`, `DO_FreshDamper_Open`, `AO_DamperExhaust_Pct` and `AO_CoolValve_Pct`, and run it in a 200 ms cyclic task. No external libraries (inline PI + standard `TON`).

**Python (`ahu_09_control.py`):** run standalone with the built-in simulator:

```bash
python3 ahu_09_control.py
```

It prints state, room/supply temperatures, the active supply setpoint, fan speed, the fresh-damper command/feedback pair, valve positions and alarm status. Structure mirrors the other units (`IOImage`, `AHUController.scan()`, `PlantSimulator`, `run()`); for field use replace `PlantSimulator` with a real I/O driver and keep `scan()` unchanged. Use `run(cycles=N, realtime=False)` for fast headless testing.

---

## 8. Verified behavior

Exercised with the built-in simulator and deterministic scans:

- **Startup with interlock:** the fresh-air damper proves open, then the fan starts and the unit reaches RUN; in cold weather the heating valve opens.
- **Cooling:** with a warm space and warm supply air the cooling valve opens and the heating valve stays closed.
- **Damper fault:** a fresh-air damper that never proves open latches a fault and drives the unit to `FAULT`.
- **Frost:** return-water ≤ 7 °C trips `FROST` with valve 100 %, cooling closed, dampers closed and pump on, latched until warm and acknowledged.
- **Fire:** the fan stops and both dampers close, common alarm raised.
- **Fan fault:** loss of proof of flow latches a fault and drives the unit to `FAULT`.

Both programs make the same decisions in the same order, so the ST program is expected to reproduce this on the PLC. Re-validate on the real plant, especially the **(default)** parameters.

> **Note on defaults.** The fresh-air damper proof timeout, the PI gains, the fan ramp and the proof-of-flow time are not specified in the JSON selection. The values used are reasonable starting points only and must be tuned to the actual unit during commissioning. Confirm the exhaust/relief damper strategy (here held open in RUN) matches the rooftop's pressure-relief design.
