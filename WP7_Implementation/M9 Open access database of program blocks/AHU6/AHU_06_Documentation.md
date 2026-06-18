# AHU_06 Control — Technical Documentation

**Project:** 10AHUs · **Unit:** AHU_06 (canvas group **"Fresh_air_unit_El_Heater"**)
**Source selection:** `10ahus_selection_AHU6.json`

| File | Platform | Role |
|------|----------|------|
| `AHU_06.st` | CoDeSys 3.5, IEC 61131-3 Structured Text | PLC control program (single `PROGRAM`) |
| `ahu_06_control.py` | Python 3.8+ | Soft-PLC control program for a virtual PC (alternative to the licensed CoDeSys runtime) |

Both files implement the same logic on a **200 ms** cycle.

---

## 1. Unit type

AHU_06 is a **supply-only fresh-air heating unit with an electric heater**. Its plant is:

- one **supply VFD fan** (`FAN_SUPPLY_VFD`),
- **inlet (outdoor) and exhaust (relief) dampers** (modulating, fail-closed),
- an **electric heating coil** (`Coil-Electric`, 0–10 V) — the only conditioning device,
- sensors: supply T/RH, room temperature, and a **supply-air high-limit thermostat**.

There is **no cooling coil, no heat recovery, no water circuit and no exhaust fan**. Conditioning is heating-only: the electric coil warms the incoming fresh air to hold room/supply temperature, and when the space is at or above setpoint the coil simply closes. Because there is no water coil or return-water sensor, there is **no water frost protection** — the `FROST` state remains in the enum for consistency but is never entered. The relevant electric-coil protection is instead the **supply-air high-limit** (overheat) cutout.

---

## 2. Operating sequence

`OFF → STARTUP → RUN → SHUTDOWN`, with protective `FIRE` and `FAULT`. Priority each scan: **FIRE > FAULT** (no FROST on this unit). The **SHUTDOWN** state holds the fan running with dampers open until the electric cool-down timer elapses, dissipating residual element heat before stopping.

| State | Behavior |
|-------|----------|
| **OFF** | Fan stopped, dampers closed, heater off. |
| **STARTUP** | Open dampers, wait 20 s; start fan, wait 10 s; on proof of flow → RUN. |
| **RUN** | Dampers open, fan at fixed supply rate, electric heater modulated by the cascade. |
| **SHUTDOWN** | Heater off; fan + dampers held during electric cool-down, then → OFF. |
| **FIRE** | `FANS_OFF_DAMPERS_CLOSE`: fan off, dampers closed, heater off. |
| **FAULT** | Safe stop on latched fan or sensor fault. |

---

## 3. I/O list

### Digital inputs

| Signal | Meaning |
|--------|---------|
| `DI_FireAlarm` | Fire signal → `FIRE` |
| `DI_FanSupply_Run` | Supply fan run feedback / proof of flow (fan ΔP) |
| `DI_FanSupply_Fault` | Supply fan hardware fault |
| `DI_FilterSupply_Dirty` | Inlet filter ΔP switch |
| **`DI_SupplyHiLimit`** | **Supply-air high-limit thermostat (electric overheat cutout)** |
| `DI_Reset` | Operator acknowledge/reset |
| `DI_RemoteEnable` | External enable (BMS / schedule) |

### Analog inputs

| Signal | JSON sensor | Notes |
|--------|-------------|-------|
| `AI_T_Supply_C` / `AI_RH_Supply_pct` | `Supply-Sensor-TempHumidity` | Supply-air control + monitoring |
| `AI_T_Room_C` | `SENSOR_T_ROOM` (`aiT_Room`) | Cascade master |

### Digital outputs

| Signal | Meaning |
|--------|---------|
| `DO_FanSupply_Run` | Supply fan enable |
| **`DO_ElHeater_Enable`** | **Electric coil safety contactor (energized only while heating)** |
| `DO_CommonAlarm` | Grouped alarm lamp / BMS bit |

### Analog outputs (0…100 %)

| Signal | Meaning |
|--------|---------|
| `AO_FanSupply_Pct` | Supply VFD speed reference |
| `AO_DamperInlet_Pct` | Outdoor-air damper |
| `AO_DamperExhaust_Pct` | Exhaust / relief damper |
| **`AO_ElHeater_Pct`** | **Electric heater modulation 0…100 % (0–10 V)** |

---

## 4. Parameters

Values from the JSON unless marked **(default)**.

| Group | Parameter | Value | Origin |
|-------|-----------|-------|--------|
| Sequence | Supply / room SP | 18 / 22 °C | `supplyTempSpC`, `roomTempSpC` |
| | Supply limits | 12 / 30 °C | `min/maxSupplyTempC` |
| | Damper / fan / purge waits | 20 / 10 / 0 s | `startup*`, `shutdownPurgeS` |
| Supply fan | Min / max / rate | 20 / 100 / 60 % | `minSpeedPct`, `maxSpeedPct`, `speedSpPct` |
| | Ramp / proof | 10 %/s · 15 s | **(default)** |
| Dampers | Open time | 120 s | `openTimeS` |
| Electric coil | Output range | 0 / 100 % | `valveType = AO_0_10V` |
| | High-limit cutout | 40 °C | **(default)** |
| | Cool-down overrun | 60 s | **(default)** |
| Room / supply PI | Kp / Ti | 1.5/600 · 12/180 | **(default)** |

---

## 5. Heating control

Active only in `RUN`. The supply fan runs at its fixed supply rate (60 %). The electric heater is driven by a **cascade**:

- the **room PI** compares room temperature to the room setpoint (22 °C) and produces a supply-air setpoint, clamped to 12…30 °C;
- the **supply PI** compares supply-air temperature to that setpoint and produces the **electric heating demand**, clamped to 0…100 % (heating only — there is no cooling device, so a warm space simply closes the coil).

The electric output equals this demand, subject to the safety interlocks below.

---

## 6. Electric-coil safety

Three independent protections, applied after the heating logic:

- **Airflow interlock:** `AO_ElHeater_Pct` is forced to 0 unless the unit is in `RUN` **and** supply-fan run feedback (`DI_FanSupply_Run`) is present. No airflow ⇒ no electric heat.
- **Supply high-limit (overheat):** the hardware thermostat `DI_SupplyHiLimit`, **or** a soft trip when supply-air temperature reaches 40 °C, latches the coil off and raises the common alarm; it clears only after the condition is gone and the operator presses reset.
- **Cool-down overrun:** a timer tracks the time since the coil last switched off. On shutdown the fan and dampers stay active for 60 s to dissipate residual element heat before the unit stops.

`DO_ElHeater_Enable` (the safety contactor) is energized only while the coil is actually modulating above zero, providing a hard electrical interlock in addition to the analog signal.

---

## 7. Alarms

| Alarm | Type | Cause | Clears when |
|-------|------|-------|-------------|
| Fan fault | Latched | Fault input or lost proof of flow (15 s) | Fault gone **and** reset |
| Sensor fault | Latched | Supply/room sensor out of −20…80 °C | Reset (re-checked next scan) |
| Electric high-limit | Latched | Thermostat or supply ≥ 40 °C | Condition gone **and** reset |
| Fire | Live | Fire input | Input clears |

`DO_CommonAlarm` is the OR of the three latched faults and the fire input. Filter ΔP raises a non-latching warning.

---

## 8. Deploying & running

**ST (`AHU_06.st`):** in CoDeSys 3.5 add the `E_AHU_STATE` DUT (listed at the bottom of the file), add a PROGRAM `AHU_06` with the body, map the I/O including `DI_SupplyHiLimit`, `DO_ElHeater_Enable` and `AO_ElHeater_Pct`, and run it in a 200 ms cyclic task. No external libraries (inline PI + standard `TON`).

**Python (`ahu_06_control.py`):** run standalone with the built-in simulator:

```bash
python3 ahu_06_control.py
```

It prints state, room/supply temperatures, the active supply setpoint, fan speed, electric output, the contactor state and alarm status. Structure mirrors the other units (`IOImage`, `AHUController.scan()`, `PlantSimulator`, `run()`); for field use replace `PlantSimulator` with a real I/O driver and keep `scan()` unchanged. Use `run(cycles=N, realtime=False)` for fast headless testing.

---

## 9. Verified behavior

Exercised with the built-in simulator:

- **Cold-weather heating:** OFF → RUN; the electric heater modulates and warms the room toward the 22 °C setpoint.
- **Supply high-limit:** the thermostat forces the coil off and latches the alarm until reset.
- **Airflow interlock:** loss of supply-fan proof drives the electric output to 0 immediately.
- **Cool-down overrun:** after a stop request the unit stays in `SHUTDOWN` with the fan running and the coil off, then reaches `OFF` once the 60 s cool-down elapses.
- **Fire:** the fan stops and dampers close immediately, common alarm raised.
- **Fan fault:** a fan commanded on without proof of flow latches a fault and drives the unit to `FAULT`.

Both programs make the same decisions in the same order, so the ST program is expected to reproduce this on the PLC. Re-validate on the real plant, especially the **(default)** parameters.

> **Note on defaults.** The supply high-limit cutout temperature, the electric cool-down time, the PI gains, the fan ramp and the proof-of-flow time are not specified in the JSON selection. The values used are reasonable starting points only and must be tuned to the actual unit during commissioning. The hardware high-limit thermostat remains the primary overheat protection regardless of the soft limit.
