# AHU_04 Control — Technical Documentation

**Project:** 10AHUs · **Unit:** AHU_04 (canvas group **"Exhaust_Vent_Room_Temp"**)
**Source selection:** `10ahus_selection_AHU4.json`

| File | Platform | Role |
|------|----------|------|
| `AHU_04.st` | CoDeSys 3.5, IEC 61131-3 Structured Text | PLC control program (single `PROGRAM`) |
| `ahu_04_control.py` | Python 3.8+ | Soft-PLC control program for a virtual PC (alternative to the licensed CoDeSys runtime) |

Both files implement the same logic on a **200 ms** cycle. This is the simplest unit of the set.

---

## 1. Unit type

AHU_04 is an **exhaust-only ventilation unit** that controls **room temperature** by varying the extract ventilation rate. Its entire plant is:

- one **extract VFD fan** (`FAN_EXHAUST_VFD`),
- an **inlet make-up-air damper** and an **exhaust damper** (both modulating, fail-closed),
- a single **room-temperature sensor** (`SENSOR_T_ROOM`).

There is **no heating coil, cooling coil, heat recovery, humidity sensor, water circuit or supply fan**. Because there is no water coil and no return-water sensor, there is **no frost protection**: the `FROST` state still exists in the state enumeration for consistency with the other units, but it is never entered on this unit (and fails safe to `OFF` if ever forced).

The control idea: when the room is warmer than its setpoint, the unit extracts harder, drawing in cooler make-up air through the inlet damper to cool the space (free cooling by extraction). When the room is at or below setpoint, the fan falls back to a minimum ventilation rate for air quality.

---

## 2. Operating sequence

`OFF → STARTUP → RUN → SHUTDOWN`, with protective `FIRE` and `FAULT`. Priority each scan: **FIRE > FAULT** (no FROST on this unit).

| State | Behavior |
|-------|----------|
| **OFF** | Fan stopped, both dampers closed. Waits for a run request. |
| **STARTUP** | Open dampers, wait `startupDamperWaitS` = 20 s; start fan, wait `startupFanWaitS` = 10 s; on proof of flow → RUN. |
| **RUN** | Dampers open, fan modulated by the room-temperature controller. |
| **SHUTDOWN** | Purge `shutdownPurgeS` = 0 s, then → OFF. |
| **FIRE** | `fireBehavior = FANS_OFF_DAMPERS_CLOSE`: fan off, both dampers closed. |
| **FAULT** | Safe stop on latched fan or sensor fault; clears after the cause is gone and acknowledged. |

---

## 3. I/O list

### Digital inputs

| Signal | Meaning |
|--------|---------|
| `DI_FireAlarm` | Fire signal → `FIRE` |
| `DI_FanExhaust_Run` | Extract fan run feedback / proof of flow (fan ΔP) |
| `DI_FanExhaust_Fault` | Extract fan hardware fault |
| `DI_Reset` | Operator acknowledge/reset |
| `DI_RemoteEnable` | External enable (BMS / schedule) |

### Analog input

| Signal | JSON sensor | Notes |
|--------|-------------|-------|
| `AI_T_Room_C` | `SENSOR_T_ROOM` (`aiT_Room`) | The single control variable |

### Digital outputs

| Signal | Meaning |
|--------|---------|
| `DO_FanExhaust_Run` | Extract fan enable |
| `DO_CommonAlarm` | Grouped alarm lamp / BMS bit |

### Analog outputs (0…100 %)

| Signal | Meaning |
|--------|---------|
| `AO_FanExhaust_Pct` | Extract VFD speed reference |
| `AO_DamperInlet_Pct` | Inlet make-up-air damper |
| `AO_DamperExhaust_Pct` | Exhaust-air damper |

---

## 4. Parameters

Values from the JSON unless marked **(default)**.

| Group | Parameter | Value | Origin |
|-------|-----------|-------|--------|
| Sequence | Room setpoint | 22 °C | `roomTempSpC` |
| | Damper / fan / purge waits | 20 / 10 / 0 s | `startup*`, `shutdownPurgeS` |
| Extract fan | Min / max | 20 / 100 % | `minSpeedPct`, `maxSpeedPct` |
| | Base ventilation | 60 % | `speedSpPct` |
| | Ramp | 10 %/s | **(default)** from `startMode = RAMP` |
| | Proof of flow | 15 s | **(default)** |
| Dampers | Open time | 120 s | `openTimeS` |
| Sensor range | Plausibility | −20…80 °C | **(default)** |
| Ventilation PI | Kp / Ti | 15 %/K · 600 s | **(default)** |

The temperature setpoints `supplyTempSpC`, `min/maxSupplyTempC` and the `CASCADE_ROOM_TO_SUPPLY` mode are present in the JSON sequence block but do not apply here — there is no supply-air sensor or heating/cooling device to drive, so only the room setpoint is used.

---

## 5. Room-temperature ventilation control

Active only in `RUN`. A **direct-acting PI** compares room temperature to the room setpoint (22 °C). Its output is the extract fan speed, expressed around the base ventilation rate:

```
 fan speed = base 60 %  +  Kp · (T_room − 22 °C)  +  integral
             clamped to 20 … 100 %
```

A warmer room produces a higher speed (more extraction → more cool make-up air); a cooler room reduces the speed down to the 20 % minimum. The integrator has anti-windup so the term `base + integral` never drives outside the 20…100 % band, giving clean recovery from the limits. The exhaust fan is enabled whenever its speed reference is above zero, and the speed reference is ramped at 10 %/s.

The exhaust fan's `tracking = FOLLOW_SUPPLY` attribute from the JSON is not applicable because the unit has no supply fan; the fan simply runs at the speed demanded by the room controller.

---

## 6. Alarms

| Alarm | Type | Cause | Clears when |
|-------|------|-------|-------------|
| Fan fault | Latched | Fault input or lost proof of flow (15 s) | Fault gone **and** reset |
| Sensor fault | Latched | Room sensor out of −20…80 °C | Reset (re-checked next scan) |
| Fire | Live | Fire input | Input clears |

`DO_CommonAlarm` is the OR of the two latched faults and the fire input.

---

## 7. Deploying & running

**ST (`AHU_04.st`):** in CoDeSys 3.5 add the `E_AHU_STATE` DUT (listed at the bottom of the file), add a PROGRAM `AHU_04` with the body, map the I/O channels, and run it in a 200 ms cyclic task. No external libraries (inline PI + standard `TON`).

**Python (`ahu_04_control.py`):** run standalone with the built-in simulator:

```bash
python3 ahu_04_control.py
```

It prints state, room temperature, extract speed, damper position and alarm status. Structure mirrors the other units (`IOImage`, `AHUController.scan()`, `PlantSimulator`, `run()`); for field use replace `PlantSimulator` with a real I/O driver and keep `scan()` unchanged. Use `run(cycles=N, realtime=False)` for fast headless testing.

---

## 8. Verified behavior

Exercised with deterministic scans and the built-in simulator:

- **Warm room:** at 26 °C the controller commands maximum extract (100 %); the closed loop then ventilates the room back down toward setpoint.
- **Cool room:** at 18 °C the controller falls back to the 20 % minimum ventilation rate.
- **Fire:** the fan stops and both dampers close immediately, common alarm raised; on clearing the input the unit restarts through STARTUP.
- **Fan fault:** a fan commanded on without proof of flow latches a fault and drives the unit to `FAULT`.

Both programs make the same decisions in the same order, so the ST program is expected to reproduce this on the PLC. Re-validate on the real plant, especially the **(default)** parameters (PI gains, fan ramp, proof-of-flow time).

> **Note on defaults.** The ventilation PI gains, the fan ramp rate and the proof-of-flow time are not specified in the JSON selection. The values used are reasonable starting points only and must be tuned to the actual unit during commissioning. Also confirm the control direction (extraction cools the space) matches the make-up-air arrangement on site.
