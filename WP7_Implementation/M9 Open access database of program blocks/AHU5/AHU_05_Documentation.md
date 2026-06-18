# AHU_05 Control — Technical Documentation

**Project:** 10AHUs · **Unit:** AHU_05 (canvas group **"FreshAir Unit1"**)
**Source selection:** `10ahus_selection_AHU5.json`

| File | Platform | Role |
|------|----------|------|
| `AHU_05.st` | CoDeSys 3.5, IEC 61131-3 Structured Text | PLC control program (single `PROGRAM`) |
| `ahu_05_control.py` | Python 3.8+ | Soft-PLC control program for a virtual PC (alternative to the licensed CoDeSys runtime) |

Both files implement the same logic on a **200 ms** cycle.

---

## 1. Unit type

AHU_05 is a **100 % fresh-air supply/exhaust unit** with **plate energy recovery (ERV)** and a **modulating recovery bypass damper** (canvas group "Klap_Bypass"). Its plant is:

- supply + exhaust **VFD fans** (the exhaust fan tracks the supply fan, `FOLLOW_SUPPLY`),
- **outdoor-air and exhaust isolation dampers** (modulating, fail-closed),
- a **plate ERV** with a modulating **bypass damper** for free heating / free cooling,
- a **water heating coil** (3-way valve) with **frost protection** and a circulation pump,
- a **cooling coil** (2-way valve),
- sensors: outdoor temperature, supply T/RH, exhaust T/RH, room T/RH, return-water temperature.

This is the same plant as AHU_01. The difference in this program is that the **ERV bypass damper is actively modulated** as the first, free-energy stage of heating and cooling, ahead of the coils. There is no electric coil, no supply low-limit thermostat and no dehumidification on this unit, so the control is the AHU_01 core plus active recovery.

**Bypass convention.** `AO_ERVBypass_Pct = 0 %` sends all air through the recovery core (maximum recovery); `100 %` bypasses the core entirely (no recovery). Internally the program computes a recovery fraction 0…100 % and drives the bypass actuator with `100 − recovery`.

---

## 2. Operating sequence

Identical to AHU_01: `OFF → STARTUP → RUN → SHUTDOWN`, with protective `FROST`, `FIRE`, `FAULT` evaluated each scan at priority **FIRE > FROST > FAULT**.

| State | Behavior |
|-------|----------|
| **OFF** | Everything stopped, dampers closed, recovery off (full bypass). |
| **STARTUP** | Open dampers, wait 20 s; start fans, wait 10 s; on supply-fan proof of flow → RUN. |
| **RUN** | Dampers open; recovery + coils control supply temperature. |
| **SHUTDOWN** | Purge 0 s → OFF. |
| **FROST** | Fans off, dampers closed, full bypass, heating valve 100 %, pump on, alarm latched. |
| **FIRE** | `FANS_OFF_DAMPERS_CLOSE`: fans off, dampers closed, full bypass, valves closed. |
| **FAULT** | Safe stop on latched fan or sensor fault. |

---

## 3. I/O list

### Digital inputs

| Signal | Meaning |
|--------|---------|
| `DI_FireAlarm` | Fire signal → `FIRE` |
| `DI_FanSupply_Run` / `DI_FanExhaust_Run` | Fan run feedback (proof of flow) |
| `DI_FanSupply_Fault` / `DI_FanExhaust_Fault` | Fan hardware fault |
| `DI_FilterSupply_Dirty` / `DI_FilterExhaust_Dirty` | Filter ΔP switches |
| `DI_Reset` | Operator acknowledge/reset |
| `DI_RemoteEnable` | External enable (BMS / schedule) |

### Analog inputs

| Signal | JSON sensor | Notes |
|--------|-------------|-------|
| `AI_T_Outdoor_C` | `temp_outdoor` (Ni1000) | Outdoor air + recovery benefit |
| `AI_T_Supply_C` / `AI_RH_Supply_pct` | `th_supply` | Supply-air control + monitoring |
| `AI_T_Exhaust_C` / `AI_RH_Exhaust_pct` | `th_exhaust` | Extract air (recovery benefit) |
| `AI_T_Room_C` | `temp_room` | Cascade master |
| `AI_RH_Room_pct` | `rh_room` | Monitoring |
| `AI_T_WaterReturn_C` | `temp_water_return` | Frost protection |

### Digital outputs

`DO_FanSupply_Run`, `DO_FanExhaust_Run`, `DO_HeatPump_Run`, `DO_CommonAlarm`.

### Analog outputs (0…100 %)

| Signal | Meaning |
|--------|---------|
| `AO_FanSupply_Pct` / `AO_FanExhaust_Pct` | Supply / exhaust VFD speed |
| `AO_DamperInlet_Pct` | Outdoor-air damper |
| `AO_DamperExhaust_Pct` | Exhaust-air damper |
| **`AO_ERVBypass_Pct`** | **ERV bypass damper (0 = full recovery, 100 = full bypass)** |
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
| Fans | Min / max / base | 20 / 100 / 60 % | `minSpeedPct`, `maxSpeedPct`, `speedSpPct` |
| | Exhaust offset (FOLLOW_SUPPLY) | 0 % | `trackingOffsetPct` |
| | Ramp / proof | 10 %/s · 15 s | **(default)** |
| Dampers | Open time | 120 s | `openTimeS` |
| ERV | Min benefit dT | 1.0 K | **(default)** |
| Heating/frost | Trip / release | 7 / 10 °C | `tReturnTripC`, `tReturnReleaseC` |
| | Pump run-on | 120 s | `pumpRunOnS` |
| Return water | Alarm / warning | 5/60 · 10/55 °C | sensor limits |
| Room / supply PI | Kp / Ti | 1.5/600 · 12/180 | **(default)** |
| Sequencer | Deadband | 5 % | **(default)** |

---

## 5. ERV recovery + coil sequencing

Control is active only in `RUN`. The cascade (room PI → supply setpoint 12…30 °C → supply PI) produces one sequencer demand of −100…+100 % (positive = heat, negative = cool). That demand is split so the **free-energy ERV is used before the coils**:

**Heating (demand > +5 %).** When the extract air is warmer than the outdoor air by more than 1 K, the recovery fraction rises 0→100 % over the first half of the heating band (bypass damper closing from 100 % toward 0 %), then the heating valve takes the second half. If recovery is not beneficial, the bypass stays fully open and the heating valve covers the whole band.

**Cooling (demand < −5 %).** Symmetrically: when the extract air is cooler than the outdoor air by more than 1 K (hot day, cooler return air), the ERV recovers "coolth" first, then the cooling valve trims. Otherwise the cooling valve covers the whole band.

```
 heating demand  0% ─────── 50% ─────── 100%
                 │ recovery 0→100 │ heat valve 0→100 │   (bypass 100→0 then valve)
 cooling demand  0% ─────── 50% ─────── 100%
                 │ recovery 0→100 │ cool valve 0→100 │
```

The benefit threshold (1 K) and the stage split are **defaults** to tune. The bypass actuator output is always `100 − recovery`.

---

## 6. Frost protection, fans, pump, alarms

These behave exactly as in AHU_01:

- **Frost:** return-water ≤ 7 °C trips the protective `FROST` state (heating valve 100 %, pump on, fans off, dampers closed, full bypass, latched alarm); release at ≥ 10 °C with operator reset.
- **Fans:** supply fan ramps at 10 %/s and is held 20…100 %; the exhaust fan follows it plus the 0 % tracking offset. Loss of proof of flow for 15 s latches a fan fault → `FAULT`.
- **Pump:** runs on heating demand or frost, with a 120 s run-on after demand ends.
- **Common alarm:** OR of frost, fan-fault, sensor-fault, fire and the return-water alarm. Filter ΔP raises a non-latching warning.

---

## 7. Deploying & running

**ST (`AHU_05.st`):** in CoDeSys 3.5 add the `E_AHU_STATE` DUT (listed at the bottom of the file), add a PROGRAM `AHU_05` with the body, map the I/O including the `AO_ERVBypass_Pct` channel, and run it in a 200 ms cyclic task. No external libraries (inline PI + standard `TON`).

**Python (`ahu_05_control.py`):** run standalone with the built-in simulator:

```bash
python3 ahu_05_control.py
```

It prints state, outdoor/room/supply temperatures, fan speed, recovery (and bypass) position, valve positions and alarm status. Structure mirrors the other units (`IOImage`, `AHUController.scan()`, `PlantSimulator`, `run()`); for field use replace `PlantSimulator` with a real I/O driver and keep `scan()` unchanged. Use `run(cycles=N, realtime=False)` for fast headless testing.

---

## 8. Verified behavior

Exercised with the built-in simulator:

- **Cold-weather heating:** OFF → RUN; the ERV recovery drives to 100 % (bypass fully closed) before the heating valve opens, since the extract air is much warmer than outdoor.
- **Free cooling:** on a hot day with cooler return air, the ERV is used for cooling first and the cooling valve only trims the remainder.
- **Frost:** return-water ≤ 7 °C trips `FROST` with valve 100 %, full bypass and pump on, latched until warm and acknowledged.
- **Fire:** dampers close, fans off, bypass full, common alarm raised.
- **Fan fault:** loss of proof of flow latches a fault and drives the unit to `FAULT`.

Both programs make the same decisions in the same order, so the ST program is expected to reproduce this on the PLC. Re-validate on the real plant, especially the **(default)** parameters.

> **Note on defaults.** The ERV benefit threshold, the heat/cool stage split, the PI gains, the fan ramp and the proof-of-flow time are not specified in the JSON selection. The values used are reasonable starting points only and must be tuned to the actual AHU during commissioning. Confirm the bypass-damper polarity (0 % = recovery) matches the installed actuator.
