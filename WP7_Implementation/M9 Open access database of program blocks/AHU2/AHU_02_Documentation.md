# AHU_02 Control — Technical Documentation

**Project:** 10AHUs · **Unit:** AHU_02 (canvas group **"DeHumi_V1"**)
**Source selection:** `10ahus_selection_AHU2.json`

| File | Platform | Role |
|------|----------|------|
| `AHU_02.st` | CoDeSys 3.5, IEC 61131-3 Structured Text | PLC control program (single `PROGRAM`) |
| `ahu_02_control.py` | Python 3.8+ | Soft-PLC control program for a virtual PC (alternative to the licensed CoDeSys runtime) |

Both files implement the same logic on a **200 ms** cycle. This unit reuses the AHU_01 control core and adds the features specific to AHU_02.

---

## 1. What is different from AHU_01

AHU_02 keeps the identical sequence state machine, VFD fans, cascade temperature control, heating-coil frost protection and alarm handling described for AHU_01. It adds four things that come from this unit's selection:

1. **Rotary heat-recovery wheel** (`HX_ROTARY`, mode `OPTIMIZE_SUPPLY_TEMP`, 0–100 %) — used as the **first stage** of both heating and cooling so that free energy from the extract air is taken before the coils run.
2. **Supply-air low-limit thermostat** (`Supply-AIR-Sensor-LowLimit`) — a hardware frost/low-temperature cut-out on the supply air, plus a soft supply-temperature low-limit trip.
3. **Dehumidification function** (the group is named *DeHumi*) using the room humidity sensor — cool the air to condense moisture, then reheat to the supply setpoint.
4. **A third (recirculation/bypass) damper** in addition to the outdoor-air and exhaust dampers.

Everything in sections that are unchanged (sequence timings, fan ramp/tracking, frost return-water trip, PI cascade) matches the AHU_01 documentation; this document focuses on the additions.

---

## 2. Operating sequence (state machine)

Identical to AHU_01: `OFF → STARTUP → RUN → SHUTDOWN`, with protective `FROST`, `FIRE`, `FAULT` evaluated each scan at priority **FIRE > FROST > FAULT**.

| State | Behavior |
|-------|----------|
| **OFF** | Everything stopped, all dampers closed, wheel off. |
| **STARTUP** | Open outdoor/exhaust dampers, wait 20 s; start fans, wait 10 s; on proof of flow → RUN. |
| **RUN** | Dampers open, recirc closed (full fresh air), recovery + coils control supply temperature. |
| **SHUTDOWN** | Purge 0 s → OFF. |
| **FROST** | Fans off, all dampers closed, wheel off, heating valve 100 %, pump on, alarm latched. |
| **FIRE** | `FANS_OFF_DAMPERS_CLOSE`: fans off, all dampers closed, wheel and valves off. |
| **FAULT** | Safe stop on latched fan or sensor fault. |

---

## 3. I/O list

Working values are in engineering units; field scaling lives in the I/O layer.

### 3.1 Digital inputs

| Signal | Meaning |
|--------|---------|
| `DI_FireAlarm` | Fire signal → `FIRE` |
| `DI_FanSupply_Run` / `DI_FanExhaust_Run` | Fan run feedback (proof of flow) |
| `DI_FanSupply_Fault` / `DI_FanExhaust_Fault` | Fan hardware fault |
| `DI_FilterSupply_Dirty` / `DI_FilterExhaust_Dirty` | Filter ΔP switches |
| **`DI_SupplyLowLimit`** | **Supply-air low-limit thermostat (hardware frost stat)** |
| `DI_Reset` | Operator acknowledge/reset |
| `DI_RemoteEnable` | External enable (BMS/schedule) |

### 3.2 Analog inputs

| Signal | JSON sensor | Notes |
|--------|-------------|-------|
| `AI_T_Inlet_C` | `Inlet-Sensor-TempBulb-Top` | Outdoor/inlet air temperature |
| `AI_T_Supply_C` / `AI_RH_Supply_pct` | `Supply-Sensor-TempHumidity-Bottom` | Supply-air control + monitoring |
| `AI_T_Exhaust_C` / `AI_RH_Exhaust_pct` | `EXHAUST-Sensor-TempHumidity-Top` | Extract air (recovery benefit) |
| `AI_T_Room_C` | `SENSOR_T_ROOM` (`aiT_Room`) | Cascade master |
| `AI_RH_Room_pct` | `SENSOR_RH_ROOM` (`aiRH_Room`) | Dehumidification control |
| `AI_T_WaterReturn_C` | `Sensor-T-WaterTemp-RETURN` | Frost protection |

### 3.3 Digital outputs

`DO_FanSupply_Run`, `DO_FanExhaust_Run`, `DO_HeatPump_Run`, `DO_CommonAlarm` — as in AHU_01.

### 3.4 Analog outputs (0…100 %)

| Signal | Meaning |
|--------|---------|
| `AO_FanSupply_Pct` / `AO_FanExhaust_Pct` | Supply / exhaust VFD speed |
| `AO_DamperInlet_Pct` | Outdoor-air damper |
| `AO_DamperExhaust_Pct` | Exhaust-air damper |
| **`AO_DamperRecirc_Pct`** | **Recirculation/bypass damper** |
| **`AO_HeatRecovery_Pct`** | **Rotary heat-recovery wheel** |
| `AO_HeatValve_Pct` | Heating 3-way valve (`AO_0_10V`) |
| `AO_CoolValve_Pct` | Cooling valve (`AO_0_10V`) |

---

## 4. Parameters

Values from the JSON unless marked **(default)** — those are commissioning choices not present in the selection and must be tuned on site.

| Group | Parameter | Value | Origin |
|-------|-----------|-------|--------|
| Sequence | Supply / room SP | 18 / 22 °C | `supplyTempSpC`, `roomTempSpC` |
| | Supply limits | 12 / 30 °C | `min/maxSupplyTempC` |
| | Damper / fan / purge waits | 20 / 10 / 0 s | `startup*`, `shutdownPurgeS` |
| Fans | Min / max / base | 20 / 100 / 60 % | `minSpeedPct`, `maxSpeedPct`, `speedSpPct` |
| | Exhaust offset (FOLLOW_SUPPLY) | 0 % | `trackingOffsetPct` |
| | Ramp / proof | 10 %/s · 15 s | **(default)** |
| Dampers | Open time | 120 s | `openTimeS` |
| **Heat recovery** | Min / max / fixed | 0 / 100 / 50 % | `minPct`, `maxPct`, `fixedPct` |
| | Min benefit dT | 1.0 K | **(default)** |
| Heating/frost | Trip / release | 7 / 10 °C | `tReturnTripC`, `tReturnReleaseC` |
| | Pump run-on | 120 s | `pumpRunOnS` |
| **Supply low-limit** | Soft trip | 5 °C | **(default)** |
| Return water | Alarm / warning | 5/60 · 10/55 °C | **(default)** range |
| Room / supply PI | Kp / Ti | 1.5/600 · 12/180 | **(default)** |
| Sequencer | Deadband | 5 % | **(default)** |
| **Dehumidification** | RH SP / deadband / min cool | 50 % / 5 % / 60 % | **(default)** — no RH SP in JSON |

---

## 5. Heat recovery + coil sequencing

Control is active only in `RUN`. The cascade (room PI → supply setpoint 12…30 °C → supply PI) produces one sequencer demand in **−100…+100 %** (positive = heat, negative = cool), exactly as in AHU_01. That demand is then split across **three stages** so the free-energy device runs before the coils:

**Heating (demand > +5 %).** If the extract air is warmer than the outdoor air by more than 1 K (`recovery beneficial`), the wheel modulates 0→100 % over the first half of the heating band, and the heating valve takes the second half. If recovery is not beneficial, the wheel stays off and the heating valve covers the whole band.

**Cooling (demand < −5 %).** Symmetrically: if the extract air is cooler than the outdoor air by more than 1 K, the wheel is used for free cooling first, then the cooling valve trims. Otherwise the cooling valve covers the whole band.

```
 heating demand  0% ───────── 50% ───────── 100%
                 │  wheel 0→100 │ heat valve 0→100 │   (when recovery helps)
 cooling demand  0% ───────── 50% ───────── 100%
                 │  wheel 0→100 │ cool valve 0→100 │   (when recovery helps)
```

This implements the wheel's `OPTIMIZE_SUPPLY_TEMP` intent in a deterministic, commissionable way. The benefit threshold (1 K) and the stage split are **defaults** to tune.

---

## 6. Dehumidification (DeHumi)

When the unit is running and room humidity rises above the setpoint plus deadband (default 50 % + 5 %), dehumidification activates:

- The **cooling valve is forced to at least 60 %** to drop the supply air below its dew point so moisture condenses on the cooling coil.
- The **heating valve reheats** the over-cooled air back toward the supply-temperature setpoint (a proportional reheat term), so the room does not get cold while being dried.

Dehumidification is released when room humidity falls back below the setpoint. Because no humidity setpoint exists in the JSON, the RH setpoint, deadband and minimum cooling are **defaults** and must be set during commissioning.

---

## 7. Supply low-limit & frost protection

Two independent conditions trip the protective `FROST` state (heating valve 100 %, pump on, fans off, all dampers closed, wheel off, latched alarm):

- **Return-water frost:** return-water temperature ≤ 7 °C (release at ≥ 10 °C, with operator reset) — same as AHU_01.
- **Supply low-limit:** the hardware low-limit thermostat `DI_SupplyLowLimit`, **or** a soft trip when supply-air temperature falls to ≤ 5 °C while running.

The pump run-on (120 s after heating demand ends, always during frost) protects the coil against stagnation.

---

## 8. Deploying & running

**ST (`AHU_02.st`):** in CoDeSys 3.5 add the `E_AHU_STATE` DUT (listed at the bottom of the file), add a PROGRAM `AHU_02` with the program body, map the I/O including the new `DI_SupplyLowLimit`, `AO_DamperRecirc_Pct` and `AO_HeatRecovery_Pct` channels, and run it in a 200 ms cyclic task. No external libraries are required (inline PI + standard `TON`).

**Python (`ahu_02_control.py`):** run standalone with the built-in simulator:

```bash
python3 ahu_02_control.py
```

It prints state, room/supply temperatures, room humidity, fan speed, wheel and valve positions, a DEHUM flag and alarm status. Structure mirrors AHU_01 (`IOImage`, `AHUController.scan()`, `PlantSimulator`, `run()`); for field use replace `PlantSimulator` with a real I/O driver and keep `scan()` unchanged. Use `run(cycles=N, realtime=False)` for fast headless testing.

---

## 9. Verified behavior

Exercised with the built-in simulator:

- **Cold-weather heating:** OFF → RUN; the recovery wheel drives to 100 % first (extract warmer than outdoor) and the heating valve trims the remainder.
- **Dehumidification:** with room RH forced high, cooling opens to the dehumidify minimum and the heating valve reheats simultaneously to hold the supply setpoint.
- **Supply low-limit:** the hardware stat trips `FROST` with heating valve 100 % and pump on.
- **Return-water frost:** ≤ 7 °C trips `FROST`, latched until warm and acknowledged.
- **Fire:** all dampers close, wheel and fans off, common alarm raised.

Both programs make the same decisions in the same order, so the ST program is expected to reproduce this on the PLC. Re-validate on the real plant, especially the **(default)** parameters (PI gains, recovery threshold, dehumidification setpoints, low-limit value).

> **Note on defaults.** The recovery benefit threshold, the heat/cool stage split, the dehumidification setpoints, the supply low-limit value, the PI gains, the fan ramp and the proof-of-flow time are not specified in the JSON selection. The values used are reasonable starting points only and must be tuned to the actual AHU during commissioning.
