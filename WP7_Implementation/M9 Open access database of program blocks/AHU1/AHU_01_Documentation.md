# AHU_01 Control — Technical Documentation

**Project:** 10AHUs · **Unit:** AHU_01
**Source selection:** `10ahus_selection_AHU1.json`
**Deliverables documented here:**

| File | Platform | Role |
|------|----------|------|
| `AHU_01.st` | CoDeSys 3.5, IEC 61131-3 Structured Text | PLC control program (single `PROGRAM`) |
| `ahu_01_control.py` | Python 3.8+ | Soft-PLC control program for a virtual PC (functional alternative to the licensed CoDeSys runtime) |

Both files implement **the same control logic**, scan-for-scan, on a **200 ms** cycle. This document explains what the logic does, how it is structured, the full I/O list and parameters, and how to deploy and run each program.

---

## 1. Overview

AHU_01 is a supply/exhaust air handling unit with energy recovery (ERV), modulating outdoor-air and exhaust dampers, two variable-speed (VFD) fans, a hot-water heating coil with a 3-way valve and circulation pump, and a chilled-water cooling coil. Room and supply conditions are measured by temperature/humidity sensors; a return-water sensor protects the heating coil against frost.

The controller runs a **sequence state machine** that brings the unit through start-up, normal running, and shutdown, and forces it into protective states for **fire**, **frost**, and **equipment faults**. While running, it holds the room temperature with a **cascade control** strategy and sequences heating and cooling so the two never fight each other.

```
                 ┌──────────── FIRE  (highest priority) ───────────┐
                 │            FROST                                 │
                 │            FAULT                                 │
                 ▼                                                  │
   OFF ──enable──► STARTUP ──proof──► RUN ──disable──► SHUTDOWN ──► OFF
        ▲            │ (dampers 20 s, then fans 10 s)        (purge 0 s)
        └────────────┴──────────────────────────────────────────────┘
```

---

## 2. Operating sequence (state machine)

States, taken from the `AHU_SEQUENCE` block: `OFF, STARTUP, RUN, SHUTDOWN, FROST, FIRE, FAULT`.

| State | Entry condition | Behavior |
|-------|-----------------|----------|
| **OFF** | default / sequence finished | Everything stopped, dampers closed, valves closed. Waits for a run request. |
| **STARTUP** | `OFF` + remote enable + no fire/frost | Drive dampers open and wait `startupDamperWaitS` = **20 s**; then start fans and wait `startupFanWaitS` = **10 s**; when supply-fan run feedback is proven → `RUN`. |
| **RUN** | proof of flow reached | Dampers open, fans at base speed, temperature control active. |
| **SHUTDOWN** | remote enable removed | Optional purge `shutdownPurgeS` = **0 s**, then → `OFF`. |
| **FROST** | return-water ≤ trip while running | Protective: fans off, dampers closed, heating valve 100 %, pump on, alarm latched. |
| **FIRE** | fire alarm input | `fireBehavior = FANS_OFF_DAMPERS_CLOSE`: fans off, dampers closed, valves closed. |
| **FAULT** | latched fan or sensor fault | Unit stopped safely; clears only after the cause is gone and acknowledged. |

**Priority each scan:** `FIRE > FROST > FAULT > normal flow`. The three protective states are tested at the top of every cycle and override the normal sequence.

---

## 3. I/O list

The control logic works in **engineering units** (°C, %, %RH). Field scaling (e.g. Ni1000 ↔ °C, 0–10 V ↔ 0–100 %) is done in the I/O layer, not in the control logic.

### 3.1 Digital inputs

| Signal | Meaning |
|--------|---------|
| `DI_FireAlarm` | Fire signal active → forces `FIRE` |
| `DI_FanSupply_Run` | Supply fan run feedback (proof of flow) |
| `DI_FanSupply_Fault` | Supply fan hardware fault |
| `DI_FanExhaust_Run` | Exhaust fan run feedback |
| `DI_FanExhaust_Fault` | Exhaust fan hardware fault |
| `DI_FilterSupply_Dirty` | Supply/inlet filter ΔP switch |
| `DI_FilterExhaust_Dirty` | Exhaust filter ΔP switch |
| `DI_Reset` | Operator alarm acknowledge/reset |
| `DI_RemoteEnable` | External enable (BMS / schedule); default `TRUE` |

### 3.2 Analog inputs

| Signal | JSON topic / component | Notes |
|--------|------------------------|-------|
| `AI_T_Outdoor_C` | `temp_outdoor` | Ni1000, −20…80 °C |
| `AI_T_Supply_C` | `th_supply` (temperature) | Supply-air control variable |
| `AI_RH_Supply_pct` | `th_supply` (humidity) | Monitoring |
| `AI_T_Exhaust_C` | `th_exhaust` (temperature) | Monitoring |
| `AI_RH_Exhaust_pct` | `th_exhaust` (humidity) | Monitoring |
| `AI_T_Room_C` | `temp_room` | Cascade master variable |
| `AI_RH_Room_pct` | `rh_room` | Monitoring |
| `AI_T_WaterReturn_C` | `temp_water_return` | Frost protection + limit alarms |

### 3.3 Digital outputs

| Signal | Meaning |
|--------|---------|
| `DO_FanSupply_Run` | Supply fan enable |
| `DO_FanExhaust_Run` | Exhaust fan enable |
| `DO_HeatPump_Run` | Heating-coil circulation pump |
| `DO_CommonAlarm` | Grouped alarm lamp / BMS bit |

### 3.4 Analog outputs (0…100 %)

| Signal | Meaning |
|--------|---------|
| `AO_FanSupply_Pct` | Supply VFD speed reference |
| `AO_FanExhaust_Pct` | Exhaust VFD speed reference |
| `AO_DamperInlet_Pct` | Outdoor-air damper |
| `AO_DamperExhaust_Pct` | Exhaust-air damper |
| `AO_HeatValve_Pct` | Heating 3-way valve (`AO_0_10V`) |
| `AO_CoolValve_Pct` | Cooling valve (`AO_0_10V`) |

---

## 4. Parameters

All values are pulled from the JSON selection unless marked **(default)** — those are commissioning choices not present in the JSON and **should be tuned to the real plant**.

| Group | Parameter | Value | Origin |
|-------|-----------|-------|--------|
| Sequence | Supply setpoint | 18.0 °C | `supplyTempSpC` |
| | Room setpoint | 22.0 °C | `roomTempSpC` |
| | Supply limits | 12.0 / 30.0 °C | `min/maxSupplyTempC` |
| | Damper wait / fan wait / purge | 20 / 10 / 0 s | `startup*`, `shutdownPurgeS` |
| Fans | Min / max / base speed | 20 / 100 / 60 % | `minSpeedPct`, `maxSpeedPct`, `speedSpPct` |
| | Exhaust tracking offset | 0 % | `trackingOffsetPct` (FOLLOW_SUPPLY) |
| | Ramp rate | 10 %/s | **(default)** from `startMode = RAMP` |
| | Proof-of-flow timeout | 15 s | **(default)** |
| Dampers | Open time | 120 s | `openTimeS` |
| Heating/frost | Frost trip / release | 7 / 10 °C | `tReturnTripC`, `tReturnReleaseC` |
| | Pump run-on | 120 s | `pumpRunOnS` |
| Return water | Alarm low/high | 5 / 60 °C | sensor `alarm_low/high` |
| | Warning low/high | 10 / 55 °C | sensor `warning_low/high` |
| Sensor range | Plausibility | −20…80 °C | Ni1000 range |
| Room PI | Kp / Ti | 1.5 / 600 s | **(default)** |
| Supply PI | Kp / Ti | 12 / 180 s | **(default)** |
| Sequencer | Deadband | 5 % | **(default)** |

---

## 5. Temperature control (cascade + split range)

Control is active **only in `RUN`**. The controllers are reset (integral cleared, setpoint = base) in all other states so the unit restarts cleanly.

**Stage 1 — outer/room loop.** A PI controller compares room temperature to the room setpoint (22 °C). Its output is a **supply-air setpoint**, clamped to the commissioned limits 12…30 °C. A cold room raises the supply setpoint; a warm room lowers it.

**Stage 2 — inner/supply loop.** A second PI controller compares supply-air temperature to that computed setpoint. Its output is a single sequencer demand in the range **−100…+100 %**.

**Stage 3 — split-range sequencing with deadband.** The sequencer demand is mapped to the two valves with a 5 % deadband between them so they never run together:

```
 demand > +5 %   →  heating valve opens 0→100 %, cooling = 0
 −5 % … +5 %     →  both valves closed (deadband)
 demand < −5 %   →  cooling valve opens 0→100 %, heating = 0
```

Both PI integrators are clamped (anti-windup) so the loops recover quickly when leaving a saturated condition.

---

## 6. Fans

The supply fan reference is **ramped** toward its demand at 10 %/s (the JSON `startMode = RAMP`). While running it is held between 20 % and 100 %. The exhaust fan **follows the supply** speed plus the tracking offset (0 %), implementing `tracking = FOLLOW_SUPPLY`. Each fan’s run output is energized whenever its speed reference is above zero.

**Proof of flow:** if a fan is commanded on but its run feedback is missing for longer than 15 s, a latched fan fault is raised and the unit goes to `FAULT`.

---

## 7. Heating coil & frost protection

The heating coil has a hot-water 3-way valve and a circulation pump. Frost protection follows `frostReaction = PUMP_ON | VALVE_100 | FANS_OFF | DAMPERS_CLOSE | ALARM_LATCH`:

- **Trip** when return-water ≤ 7 °C → enter `FROST`: fans off, dampers closed, heating valve 100 %, pump on, alarm **latched**.
- **Release** when return-water ≥ 10 °C (hysteresis), but the latch remains until the operator acknowledges with `DI_Reset`.
- **Pump run-on:** the pump keeps running for 120 s after heating demand ends (and always during frost) to avoid stagnation and freezing.

Return-water limit supervision additionally raises warnings (10/55 °C) and alarms (5/60 °C) independent of the frost state.

---

## 8. Alarms

| Alarm | Type | Cause | Clears when |
|-------|------|-------|-------------|
| Frost | Latched | Return-water ≤ 7 °C | Water ≥ 10 °C **and** reset |
| Fan fault | Latched | Fan fault input or lost proof of flow | Fault gone **and** reset |
| Sensor fault | Latched | Temperature sensor out of −20…80 °C | Reset (re-checked next scan) |
| Return-water alarm | Live | Outside 5…60 °C | Value back in range |
| Filter dirty | Warning | Filter ΔP switch | ΔP normal |

`DO_CommonAlarm` is the OR of all latched alarms plus fire and the return-water alarm.

---

## 9. Deploying the ST program (`AHU_01.st`)

1. In CoDeSys 3.5, create a project for your PLC target.
2. Add a DUT named **`E_AHU_STATE`** with the enum given at the bottom of `AHU_01.st`.
3. Add a POU of type **PROGRAM** named `AHU_01` and paste the program body (everything between `PROGRAM AHU_01` and `END_PROGRAM`).
4. Map the `DI_*`, `AI_*`, `DO_*`, `AO_*` variables to physical channels in the device I/O mapping (or wire them to your fieldbus image). Apply your sensor/actuator scaling there.
5. Assign `AHU_01` to a cyclic task with **interval = 200 ms** (`T#200MS`) to match the parameter timing.
6. Build, download, and commission. Tune the **(default)** parameters in section 4.

The program is self-contained: PI controllers and timers are implemented inline (standard `TON` instances), so no external libraries are required.

---

## 10. Running the Python program (`ahu_01_control.py`)

The Python program is a **soft-PLC**: the same logic in a class `AHUController` whose `scan()` runs once per 200 ms cycle, driven by a real-time scheduler.

**Standalone (with built-in simulator):**

```bash
python3 ahu_01_control.py
```

It prints state, room/supply temperatures, the active supply setpoint, fan speed, valve positions, pump and alarm status roughly every 5 s. Stop with `Ctrl+C`.

**Code structure:**

| Element | Responsibility |
|---------|----------------|
| `class P` | All engineering parameters (section 4) |
| `class State(IntEnum)` | The seven sequence states |
| `class TON` | Software on-delay timer (IEC `TON`) |
| `@dataclass IOImage` | The PLC I/O image (all `DI/AI/DO/AO`) |
| `class AHUController` | Control logic; `scan(io, dt)` = one PLC cycle |
| `class PlantSimulator` | A lumped physical model so it runs with no hardware |
| `run(...)` | 200 ms real-time loop |

**Field deployment:** replace `PlantSimulator` with a driver that talks to real I/O (Modbus, OPC UA, bus card, …). Each cycle: read field inputs into `io.*`, call `ctrl.scan(io, P.DT_S)`, then push `io.*` outputs to the field. The control logic in `AHUController` stays unchanged.

**Headless / accelerated testing:** call `run(cycles=N, realtime=False)` to simulate `N` cycles as fast as possible — useful for regression tests of start-up, frost, fire, and fault scenarios.

---

## 11. Verified behavior

The Python program was exercised through the main scenarios with the built-in simulator:

- **Cold-weather start:** OFF → STARTUP → RUN within the configured waits; cascade control opens the heating valve and the room settles near 22 °C; the exhaust fan exactly tracks the supply fan.
- **Fire:** `FIRE` forces fans off and dampers closed immediately and raises the common alarm; on clearing the input the unit returns toward normal operation.
- **Frost:** return-water ≤ 7 °C trips `FROST` with valve 100 %, fans off, pump on and a latched alarm; recovery requires warm water **and** an operator reset.
- **Fan fault:** a fan commanded on without proof of flow latches a fault and drives the unit to `FAULT`.

Both programs implement the same decisions in the same order, so the ST program is expected to reproduce this behavior on the PLC. Re-validate on the real plant during commissioning, especially the **(default)** tuning parameters.

> **Note on defaults.** The PI gains, fan ramp rate, and proof-of-flow timeout are not specified in the JSON selection. The values used are reasonable starting points only and must be tuned to the actual AHU during commissioning.
