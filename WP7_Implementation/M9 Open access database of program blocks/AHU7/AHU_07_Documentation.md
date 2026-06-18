# AHU_07 Control — Technical Documentation

**Project:** 10AHUs · **Unit:** AHU_07 (canvas group **"Circulation_Clean_Hepa"**)
**Source selection:** `10ahus_selection_AHU7.json`

| File | Platform | Role |
|------|----------|------|
| `AHU_07.st` | CoDeSys 3.5, IEC 61131-3 Structured Text | PLC control program (single `PROGRAM`) |
| `ahu_07_control.py` | Python 3.8+ | Soft-PLC control program for a virtual PC (alternative to the licensed CoDeSys runtime) |

Both files implement the same logic on a **200 ms** cycle.

---

## 1. Unit type

AHU_07 is a **recirculation cleanroom unit with HEPA filtration**. Room air is drawn back, filtered through a pre-filter and a HEPA bank, conditioned, and returned to the room. Its plant is:

- one **supply VFD fan**, run at **constant volume** (to keep the air-change rate and HEPA face velocity constant),
- a single **relief / recirculation damper** (modulating, fail-closed),
- a **water heating coil** (3-way valve) with **frost protection** and a circulation pump,
- a **cooling coil** (2-way valve),
- **pre-filter and HEPA filter banks** with differential-pressure monitoring,
- sensors: supply T/RH, return-air T/RH, room T/RH, return-water temperature.

There is **no outdoor-air sensor, no exhaust fan and no heat recovery** — the unit recirculates room air. Temperature is held by the heating and cooling coils under cascade control; the fan does not modulate for temperature.

---

## 2. Operating sequence

`OFF → STARTUP → RUN → SHUTDOWN`, with protective `FROST`, `FIRE`, `FAULT` at priority **FIRE > FROST > FAULT**.

| State | Behavior |
|-------|----------|
| **OFF** | Fan stopped, damper closed, valves closed. |
| **STARTUP** | Open damper, wait 20 s; start fan, wait 10 s; on proof of flow → RUN. |
| **RUN** | Damper open, fan at constant volume, coils under cascade control. |
| **SHUTDOWN** | Purge 0 s → OFF. |
| **FROST** | Fan off, damper closed, heating valve 100 %, pump on, alarm latched. |
| **FIRE** | `FANS_OFF_DAMPERS_CLOSE`: fan off, damper closed, valves closed. |
| **FAULT** | Safe stop on latched fan or sensor fault. |

---

## 3. I/O list

### Digital inputs

| Signal | Meaning |
|--------|---------|
| `DI_FireAlarm` | Fire signal → `FIRE` |
| `DI_FanSupply_Run` | Supply fan run feedback (proof of flow, fan ΔP) |
| `DI_FanSupply_Fault` | Supply fan hardware fault |
| `DI_FilterInlet_Dirty` | Pre-filter ΔP switch (warning) |
| **`DI_FilterHepa_High`** | **HEPA differential-pressure high (alarm)** |
| `DI_Reset` | Operator acknowledge/reset |
| `DI_RemoteEnable` | External enable (BMS / schedule) |

### Analog inputs

| Signal | JSON sensor | Notes |
|--------|-------------|-------|
| `AI_T_Supply_C` / `AI_RH_Supply_pct` | `th_supply` | Supply-air control + monitoring |
| `AI_T_Return_C` / `AI_RH_Return_pct` | `th_exhaust` | Return-air monitoring |
| `AI_T_Room_C` | `temp_room` | Cascade master |
| `AI_RH_Room_pct` | `rh_room` | Monitoring |
| `AI_T_WaterReturn_C` | `temp_water_return` | Frost protection |

### Digital outputs

`DO_FanSupply_Run`, `DO_HeatPump_Run`, `DO_CommonAlarm`.

### Analog outputs (0…100 %)

| Signal | Meaning |
|--------|---------|
| `AO_FanSupply_Pct` | Supply VFD speed reference (constant in RUN) |
| `AO_Damper_Pct` | Relief / recirculation damper |
| `AO_HeatValve_Pct` | Heating 3-way valve (`AO_0_10V`) |
| `AO_CoolValve_Pct` | Cooling valve (`AO_0_10V`) |

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
| Damper | Open time | 120 s | `openTimeS` |
| Heating/frost | Trip / release | 7 / 10 °C | `tReturnTripC`, `tReturnReleaseC` |
| | Pump run-on | 120 s | `pumpRunOnS` |
| Return water | Alarm / warning | 5/60 · 10/55 °C | sensor limits |
| Room / supply PI | Kp / Ti | 1.5/600 · 12/180 | **(default)** |
| Sequencer | Deadband | 5 % | **(default)** |

---

## 5. Temperature control

Active only in `RUN`. The supply fan runs at its constant volume setpoint (60 %). Temperature is held by a **cascade with split-range heat/cool**, identical in form to AHU_01:

- the **room PI** turns the room error into a supply-air setpoint clamped to 12…30 °C;
- the **supply PI** turns the supply error into a single demand of −100…+100 %;
- a 5 % deadband splits that demand: positive opens the heating valve, negative opens the cooling valve, and the two never run together.

Because the unit recirculates room air, the coils act on air close to room temperature, so in normal cleanroom operation the cooling coil typically trims the internal heat gain while the heating coil handles cold starts.

---

## 6. Cleanroom filtration & alarms

The two filter banks are treated differently to suit cleanroom operation:

- **Pre-filter ΔP** (`DI_FilterInlet_Dirty`) raises a **warning** only — it signals routine pre-filter loading and does not affect operation or the common alarm.
- **HEPA ΔP high** (`DI_FilterHepa_High`) raises a **common alarm** (HEPA loading or integrity issue) but **does not stop the unit**, so room cleanliness and pressurisation are maintained until the filter can be serviced.

| Alarm | Type | Cause | Stops unit? |
|-------|------|-------|-------------|
| Frost | Latched | Return-water ≤ 7 °C | Yes (FROST) |
| Fan fault | Latched | Fault input or lost proof of flow | Yes (FAULT) |
| Sensor fault | Latched | Supply/room/return-water out of range | Yes (FAULT) |
| HEPA dP high | Live | HEPA ΔP switch | **No** (alarm only) |
| Return-water alarm | Live | Outside 5…60 °C | No |
| Pre-filter dirty | Warning | Pre-filter ΔP | No |
| Fire | Live | Fire input | Yes (FIRE) |

`DO_CommonAlarm` is the OR of the latched faults, fire, the return-water alarm and the HEPA alarm.

Frost protection, the heating pump (demand + 120 s run-on + frost) and the fan proof-of-flow behave exactly as in AHU_01.

---

## 7. Deploying & running

**ST (`AHU_07.st`):** in CoDeSys 3.5 add the `E_AHU_STATE` DUT (listed at the bottom of the file), add a PROGRAM `AHU_07` with the body, map the I/O including `DI_FilterHepa_High` and `DI_FilterInlet_Dirty`, and run it in a 200 ms cyclic task. No external libraries (inline PI + standard `TON`).

**Python (`ahu_07_control.py`):** run standalone with the built-in simulator:

```bash
python3 ahu_07_control.py
```

It prints state, room/supply temperatures, the active supply setpoint, fan speed, valve positions, a HEPA flag and alarm status. Structure mirrors the other units (`IOImage`, `AHUController.scan()`, `PlantSimulator`, `run()`); for field use replace `PlantSimulator` with a real I/O driver and keep `scan()` unchanged. Use `run(cycles=N, realtime=False)` for fast headless testing.

---

## 8. Verified behavior

Exercised with the built-in simulator and deterministic scans:

- **Warm cleanroom:** the cooling valve trims the internal heat gain to hold setpoint while the fan stays at constant volume.
- **Cold start:** the heating valve opens and the pump runs.
- **HEPA dP high:** the common alarm is raised but the unit keeps running at constant volume (cleanroom pressurisation preserved).
- **Pre-filter dirty:** a warning is raised without affecting the common alarm.
- **Frost:** return-water ≤ 7 °C trips `FROST` with valve 100 % and pump on, latched until warm and acknowledged.
- **Fire:** the fan stops and the damper closes, common alarm raised.
- **Fan fault:** loss of proof of flow latches a fault and drives the unit to `FAULT`.

Both programs make the same decisions in the same order, so the ST program is expected to reproduce this on the PLC. Re-validate on the real plant, especially the **(default)** parameters.

> **Note on defaults.** The PI gains, the fan ramp and the proof-of-flow time are not specified in the JSON selection. The values used are reasonable starting points only and must be tuned to the actual unit during commissioning. Confirm with the cleanroom design whether the fan should remain constant volume or modulate for pressurisation, and whether a HEPA high-dP event should escalate beyond an alarm.
