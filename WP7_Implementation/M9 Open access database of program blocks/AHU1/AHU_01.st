(*
================================================================================
 AHU_01 - Air Handling Unit Control Program
 Target  : CoDeSys 3.5 (IEC 61131-3), Structured Text
 Task     : cyclic, 200 ms (T#200MS)
 Source  : 10ahus_selection_AHU1.json  (project "10AHUs", AHU "AHU_01")
 Author  : generated control logic
--------------------------------------------------------------------------------
 SCOPE
   Single PROGRAM containing the complete control of one AHU:
     - Operating-mode / sequence state machine
         OFF -> STARTUP -> RUN -> SHUTDOWN -> OFF   (+ FROST, FIRE, FAULT)
     - Outdoor-air & exhaust isolation dampers (modulating, fail-close)
     - Supply / exhaust VFD fans (ramped, exhaust tracks supply)
     - Cascade temperature control (room -> supply setpoint)
     - Heating / cooling split-range sequencing
     - Heating coil frost protection (return-water + pump run-on)
     - Filter / fan-proof / sensor supervision and alarms
--------------------------------------------------------------------------------
 NOTE ON "single program" style
   To keep everything in one POU the PI controllers and timers are implemented
   inline with static variables (no auxiliary function blocks). The standard
   IEC timer TON instances are declared as program-local instances, which is
   permitted inside a single PROGRAM.
================================================================================
*)

PROGRAM AHU_01
VAR CONSTANT
    (* ---- Engineering parameters taken from the JSON selection ---- *)
    C_CYCLE_MS          : TIME := T#200MS;     (* task cycle                         *)

    (* Sequence (AHU_SEQUENCE) *)
    C_SUPPLY_SP_C        : REAL := 18.0;        (* supplyTempSpC                      *)
    C_ROOM_SP_C          : REAL := 22.0;        (* roomTempSpC                        *)
    C_MIN_SUPPLY_C       : REAL := 12.0;        (* minSupplyTempC                     *)
    C_MAX_SUPPLY_C       : REAL := 30.0;        (* maxSupplyTempC                     *)
    C_STARTUP_DAMPER_S   : TIME := T#20S;       (* startupDamperWaitS                 *)
    C_STARTUP_FAN_S      : TIME := T#10S;       (* startupFanWaitS                    *)
    C_SHUTDOWN_PURGE_S   : TIME := T#0S;        (* shutdownPurgeS                     *)

    (* Fans (FAN_SUPPLY_VFD / FAN_EXHAUST_VFD) *)
    C_FAN_MIN_PCT        : REAL := 20.0;        (* minSpeedPct                        *)
    C_FAN_MAX_PCT        : REAL := 100.0;       (* maxSpeedPct                        *)
    C_FAN_SP_PCT         : REAL := 60.0;        (* speedSpPct (base demand)           *)
    C_FAN_RAMP_PCT_S     : REAL := 10.0;        (* ramp rate %/s (startMode = RAMP)   *)
    C_EXH_OFFSET_PCT     : REAL := 0.0;         (* trackingOffsetPct (FOLLOW_SUPPLY)  *)
    C_FAN_PROOF_S        : TIME := T#15S;       (* time to prove run feedback         *)

    (* Dampers (DAMPER_INLET, MODULATING, fail CLOSE) *)
    C_DAMPER_OPEN_S      : TIME := T#120S;      (* openTimeS                          *)

    (* Heating coil + frost (HEATING_COIL_3WV) *)
    C_FROST_TRIP_C       : REAL := 7.0;         (* tReturnTripC                       *)
    C_FROST_RELEASE_C    : REAL := 10.0;        (* tReturnReleaseC                    *)
    C_PUMP_RUNON_S       : TIME := T#120S;      (* pumpRunOnS                         *)

    (* Return-water sensor supervision (temp_water_return) *)
    C_RW_ALARM_LOW_C     : REAL := 5.0;
    C_RW_ALARM_HIGH_C    : REAL := 60.0;
    C_RW_WARN_LOW_C      : REAL := 10.0;
    C_RW_WARN_HIGH_C     : REAL := 55.0;

    (* Sensor plausibility (Ni1000 range -20..80) *)
    C_TEMP_RANGE_LO_C    : REAL := -20.0;
    C_TEMP_RANGE_HI_C    : REAL := 80.0;

    (* Cascade room PI controller *)
    C_ROOM_KP            : REAL := 1.5;         (* deg supply per deg room error      *)
    C_ROOM_TI_S          : REAL := 600.0;       (* integral time, s                   *)

    (* Supply PI controller (output -100..+100 ; + = heat, - = cool) *)
    C_SUP_KP             : REAL := 12.0;        (* % per deg error                    *)
    C_SUP_TI_S           : REAL := 180.0;       (* integral time, s                   *)
    C_SEQ_DEADBAND_PCT   : REAL := 5.0;         (* heat/cool deadband, %              *)
END_VAR

VAR
    (* =========================== I/O IMAGE =========================== *)
    (* ---- Digital inputs ---- *)
    DI_FireAlarm         : BOOL;        (* TRUE = fire signal active                *)
    DI_FanSupply_Run     : BOOL;        (* supply fan run feedback (hasRunFb)       *)
    DI_FanSupply_Fault   : BOOL;        (* supply fan fault (hasFault)              *)
    DI_FanExhaust_Run    : BOOL;        (* exhaust fan run feedback                 *)
    DI_FanExhaust_Fault  : BOOL;        (* exhaust fan fault                        *)
    DI_FilterSupply_Dirty: BOOL;        (* supply/inlet filter dP switch           *)
    DI_FilterExhaust_Dirty: BOOL;       (* exhaust filter dP switch                 *)
    DI_Reset             : BOOL;        (* operator alarm reset / acknowledge       *)
    DI_RemoteEnable      : BOOL := TRUE;(* external enable (BMS / schedule)         *)

    (* ---- Analog inputs (engineering units) ---- *)
    AI_T_Outdoor_C       : REAL;        (* temp_outdoor      (Ni1000)              *)
    AI_T_Supply_C        : REAL;        (* th_supply temperature                   *)
    AI_RH_Supply_pct     : REAL;        (* th_supply humidity                      *)
    AI_T_Exhaust_C       : REAL;        (* th_exhaust temperature                  *)
    AI_RH_Exhaust_pct    : REAL;        (* th_exhaust humidity                     *)
    AI_T_Room_C          : REAL;        (* temp_room                               *)
    AI_RH_Room_pct       : REAL;        (* rh_room                                 *)
    AI_T_WaterReturn_C   : REAL;        (* temp_water_return (frost coil)          *)

    (* ---- Digital outputs ---- *)
    DO_FanSupply_Run     : BOOL;        (* supply fan enable                       *)
    DO_FanExhaust_Run    : BOOL;        (* exhaust fan enable                      *)
    DO_HeatPump_Run      : BOOL;        (* heating coil circulation pump           *)
    DO_CommonAlarm       : BOOL;        (* grouped alarm lamp / BMS bit            *)

    (* ---- Analog outputs (0..100 % ; field scaling 0..10 V done in I/O layer) ---- *)
    AO_FanSupply_Pct     : REAL;        (* supply VFD speed reference              *)
    AO_FanExhaust_Pct    : REAL;        (* exhaust VFD speed reference             *)
    AO_DamperInlet_Pct   : REAL;        (* outdoor-air damper                      *)
    AO_DamperExhaust_Pct : REAL;        (* exhaust-air damper                      *)
    AO_HeatValve_Pct     : REAL;        (* heating 3-way valve (AO_0_10V)          *)
    AO_CoolValve_Pct     : REAL;        (* cooling valve     (AO_0_10V)            *)

    (* =========================== STATE =========================== *)
    eState               : E_AHU_STATE := AHU_OFF;
    eState_prev          : E_AHU_STATE := AHU_OFF;
    bEnableCmd           : BOOL;        (* combined run request                    *)

    (* Latched faults *)
    bFrostLatched        : BOOL;        (* frost protection latched alarm          *)
    bFanFaultLatched     : BOOL;
    bSensorFaultLatched  : BOOL;

    (* Warnings (non-latching) *)
    bFilterWarn          : BOOL;
    bRWaterWarn          : BOOL;
    bRWaterAlarm         : BOOL;

    (* Setpoints / controller working vars *)
    rSupplySetpoint_C    : REAL := 18.0;(* active supply-air setpoint              *)
    rRoom_I              : REAL;        (* room PI integral term                   *)
    rSup_I               : REAL;        (* supply PI integral term                 *)
    rSeqOut_Pct          : REAL;        (* sequencer demand -100..+100             *)
    rFanDemand_Pct       : REAL;        (* commanded supply fan speed (target)     *)

    (* Frost protection *)
    bFrostActive         : BOOL;

    (* dt in seconds, derived from cycle *)
    rDt_s                : REAL := 0.2;

    (* =========================== TIMERS =========================== *)
    tonDamperWait        : TON;         (* damper opening wait at startup          *)
    tonFanWait           : TON;         (* fan stabilisation wait                  *)
    tonPurge             : TON;         (* shutdown purge                          *)
    tonSupplyProof       : TON;         (* supply fan proof-of-flow                *)
    tonExhaustProof      : TON;         (* exhaust fan proof-of-flow               *)
    tonPumpRunOn         : TON;         (* heating pump overrun                    *)

    (* helper temporaries *)
    rErr                 : REAL;
    rRamp                : REAL;
    bHeatDemand          : BOOL;
END_VAR

(* ================================================================
   0)  HOUSEKEEPING
   ================================================================ *)
rDt_s := DINT_TO_REAL(TIME_TO_DINT(C_CYCLE_MS)) / 1000.0;
eState_prev := eState;

(* Combined enable: remote enable AND no blocking latched fault *)
bEnableCmd := DI_RemoteEnable AND NOT bFanFaultLatched AND NOT bSensorFaultLatched;

(* ================================================================
   1)  SENSOR SUPERVISION  (plausibility + return-water limits)
   ================================================================ *)
(* Out-of-range temperature sensors -> latched sensor fault *)
IF  (AI_T_Outdoor_C   < C_TEMP_RANGE_LO_C) OR (AI_T_Outdoor_C   > C_TEMP_RANGE_HI_C)
 OR (AI_T_Supply_C    < C_TEMP_RANGE_LO_C) OR (AI_T_Supply_C    > C_TEMP_RANGE_HI_C)
 OR (AI_T_Room_C      < C_TEMP_RANGE_LO_C) OR (AI_T_Room_C      > C_TEMP_RANGE_HI_C)
 OR (AI_T_WaterReturn_C < C_TEMP_RANGE_LO_C) OR (AI_T_WaterReturn_C > C_TEMP_RANGE_HI_C)
THEN
    bSensorFaultLatched := TRUE;
END_IF

(* Return-water supervision *)
bRWaterWarn  := (AI_T_WaterReturn_C < C_RW_WARN_LOW_C)  OR (AI_T_WaterReturn_C > C_RW_WARN_HIGH_C);
bRWaterAlarm := (AI_T_WaterReturn_C < C_RW_ALARM_LOW_C) OR (AI_T_WaterReturn_C > C_RW_ALARM_HIGH_C);

(* Dirty-filter warning (informational, does not stop the unit) *)
bFilterWarn := DI_FilterSupply_Dirty OR DI_FilterExhaust_Dirty;

(* ================================================================
   2)  FAN FAULT SUPERVISION
       - hard fault input
       - missing proof of flow while commanded ON
   ================================================================ *)
tonSupplyProof(IN := DO_FanSupply_Run AND NOT DI_FanSupply_Run, PT := C_FAN_PROOF_S);
tonExhaustProof(IN := DO_FanExhaust_Run AND NOT DI_FanExhaust_Run, PT := C_FAN_PROOF_S);

IF DI_FanSupply_Fault OR DI_FanExhaust_Fault
   OR tonSupplyProof.Q OR tonExhaustProof.Q THEN
    bFanFaultLatched := TRUE;
END_IF

(* ================================================================
   3)  FROST PROTECTION  (return-water trip with hysteresis)
       frostReaction = PUMP_ON | VALVE_100 | FANS_OFF | DAMPERS_CLOSE | ALARM_LATCH
   ================================================================ *)
IF AI_T_WaterReturn_C <= C_FROST_TRIP_C THEN
    bFrostActive  := TRUE;
    bFrostLatched := TRUE;                       (* ALARM_LATCH *)
ELSIF AI_T_WaterReturn_C >= C_FROST_RELEASE_C THEN
    bFrostActive := FALSE;                        (* condition cleared; latch stays *)
END_IF

(* ================================================================
   4)  ALARM RESET  (operator acknowledge)
       Only clears latches when the physical cause is gone.
   ================================================================ *)
IF DI_Reset THEN
    IF NOT bFrostActive THEN
        bFrostLatched := FALSE;
    END_IF
    IF NOT (DI_FanSupply_Fault OR DI_FanExhaust_Fault) THEN
        bFanFaultLatched := FALSE;
    END_IF
    bSensorFaultLatched := FALSE;  (* re-evaluated next scan if still bad *)
END_IF

(* ================================================================
   5)  SEQUENCE STATE MACHINE
       States: OFF, STARTUP, RUN, FROST, FIRE, FAULT, SHUTDOWN
       Priority: FIRE > FROST > FAULT > normal flow
   ================================================================ *)

(* --- High-priority overrides evaluated every scan --- *)
IF DI_FireAlarm THEN
    eState := AHU_FIRE;
ELSIF bFrostLatched AND (eState <> AHU_OFF) THEN
    eState := AHU_FROST;
ELSIF (bFanFaultLatched OR bSensorFaultLatched) AND (eState <> AHU_OFF) THEN
    eState := AHU_FAULT;
END_IF

CASE eState OF

    AHU_OFF:
        (* Wait for a run request and a healthy unit *)
        IF bEnableCmd AND NOT DI_FireAlarm AND NOT bFrostLatched THEN
            eState := AHU_STARTUP;
        END_IF

    AHU_STARTUP:
        (* Open dampers, wait, then start fans and wait for proof *)
        tonDamperWait(IN := TRUE, PT := C_STARTUP_DAMPER_S);
        IF tonDamperWait.Q THEN
            tonFanWait(IN := TRUE, PT := C_STARTUP_FAN_S);
        END_IF
        IF tonFanWait.Q AND DI_FanSupply_Run THEN
            eState := AHU_RUN;
        END_IF
        IF NOT bEnableCmd THEN
            eState := AHU_SHUTDOWN;
        END_IF

    AHU_RUN:
        tonDamperWait(IN := FALSE);
        tonFanWait(IN := FALSE);
        IF NOT bEnableCmd THEN
            eState := AHU_SHUTDOWN;
        END_IF

    AHU_SHUTDOWN:
        tonPurge(IN := TRUE, PT := C_SHUTDOWN_PURGE_S);
        IF tonPurge.Q THEN
            tonPurge(IN := FALSE);
            eState := AHU_OFF;
        END_IF

    AHU_FROST:
        (* Recover automatically only after latch acknowledged & cleared *)
        IF NOT bFrostLatched THEN
            eState := AHU_OFF;
        END_IF

    AHU_FIRE:
        IF NOT DI_FireAlarm THEN
            eState := AHU_OFF;          (* manual restart sequence afterwards *)
        END_IF

    AHU_FAULT:
        IF NOT (bFanFaultLatched OR bSensorFaultLatched) THEN
            eState := AHU_OFF;
        END_IF

END_CASE

(* ================================================================
   6)  TEMPERATURE CONTROL
       Cascade: room PI -> supply setpoint (clamped) ;
       supply PI -> split-range heat/cool sequencer.
       Active only in RUN.
   ================================================================ *)
IF eState = AHU_RUN THEN

    (* --- 6a) Outer (room) PI loop : output is a supply-air setpoint --- *)
    rErr   := C_ROOM_SP_C - AI_T_Room_C;
    rRoom_I := rRoom_I + (C_ROOM_KP / C_ROOM_TI_S) * rErr * rDt_s;
    (* Anti-windup clamp of integral expressed in setpoint units around base SP *)
    IF rRoom_I >  (C_MAX_SUPPLY_C - C_SUPPLY_SP_C) THEN rRoom_I := (C_MAX_SUPPLY_C - C_SUPPLY_SP_C); END_IF
    IF rRoom_I <  (C_MIN_SUPPLY_C - C_SUPPLY_SP_C) THEN rRoom_I := (C_MIN_SUPPLY_C - C_SUPPLY_SP_C); END_IF

    rSupplySetpoint_C := C_SUPPLY_SP_C + (C_ROOM_KP * rErr) + rRoom_I;
    (* Hard clamp to commissioned supply limits *)
    IF rSupplySetpoint_C > C_MAX_SUPPLY_C THEN rSupplySetpoint_C := C_MAX_SUPPLY_C; END_IF
    IF rSupplySetpoint_C < C_MIN_SUPPLY_C THEN rSupplySetpoint_C := C_MIN_SUPPLY_C; END_IF

    (* --- 6b) Inner (supply) PI loop : output -100..+100 % --- *)
    rErr  := rSupplySetpoint_C - AI_T_Supply_C;
    rSup_I := rSup_I + (C_SUP_KP / C_SUP_TI_S) * rErr * rDt_s;
    IF rSup_I >  100.0 THEN rSup_I :=  100.0; END_IF
    IF rSup_I < -100.0 THEN rSup_I := -100.0; END_IF

    rSeqOut_Pct := (C_SUP_KP * rErr) + rSup_I;
    IF rSeqOut_Pct >  100.0 THEN rSeqOut_Pct :=  100.0; END_IF
    IF rSeqOut_Pct < -100.0 THEN rSeqOut_Pct := -100.0; END_IF

ELSE
    (* Hold controllers reset when not running *)
    rRoom_I := 0.0;
    rSup_I  := 0.0;
    rSeqOut_Pct := 0.0;
    rSupplySetpoint_C := C_SUPPLY_SP_C;
END_IF

(* --- 6c) Split-range sequencing with deadband --- *)
IF rSeqOut_Pct > C_SEQ_DEADBAND_PCT THEN
    AO_HeatValve_Pct := (rSeqOut_Pct - C_SEQ_DEADBAND_PCT) / (100.0 - C_SEQ_DEADBAND_PCT) * 100.0;
    AO_CoolValve_Pct := 0.0;
    bHeatDemand := TRUE;
ELSIF rSeqOut_Pct < -C_SEQ_DEADBAND_PCT THEN
    AO_HeatValve_Pct := 0.0;
    AO_CoolValve_Pct := (-rSeqOut_Pct - C_SEQ_DEADBAND_PCT) / (100.0 - C_SEQ_DEADBAND_PCT) * 100.0;
    bHeatDemand := FALSE;
ELSE
    AO_HeatValve_Pct := 0.0;
    AO_CoolValve_Pct := 0.0;
    bHeatDemand := FALSE;
END_IF

(* ================================================================
   7)  OUTPUT MAPPING PER STATE
   ================================================================ *)
CASE eState OF

    AHU_OFF, AHU_SHUTDOWN:
        rFanDemand_Pct      := 0.0;
        AO_DamperInlet_Pct  := 0.0;            (* fail/closed *)
        AO_DamperExhaust_Pct:= 0.0;
        AO_HeatValve_Pct    := 0.0;
        AO_CoolValve_Pct    := 0.0;

    AHU_STARTUP:
        AO_DamperInlet_Pct  := 100.0;          (* drive dampers open *)
        AO_DamperExhaust_Pct:= 100.0;
        IF tonFanWait.IN THEN
            rFanDemand_Pct := C_FAN_SP_PCT;    (* request speed once damper wait done *)
        ELSE
            rFanDemand_Pct := 0.0;
        END_IF
        AO_HeatValve_Pct    := 0.0;
        AO_CoolValve_Pct    := 0.0;

    AHU_RUN:
        AO_DamperInlet_Pct  := 100.0;
        AO_DamperExhaust_Pct:= 100.0;
        rFanDemand_Pct      := C_FAN_SP_PCT;   (* heat/cool valves set in section 6 *)

    AHU_FROST:
        (* FANS_OFF | DAMPERS_CLOSE | PUMP_ON | VALVE_100 *)
        rFanDemand_Pct      := 0.0;
        AO_DamperInlet_Pct  := 0.0;
        AO_DamperExhaust_Pct:= 0.0;
        AO_HeatValve_Pct    := 100.0;
        AO_CoolValve_Pct    := 0.0;
        bHeatDemand         := TRUE;           (* keep pump running *)

    AHU_FIRE:
        (* fireBehavior = FANS_OFF_DAMPERS_CLOSE *)
        rFanDemand_Pct      := 0.0;
        AO_DamperInlet_Pct  := 0.0;
        AO_DamperExhaust_Pct:= 0.0;
        AO_HeatValve_Pct    := 0.0;
        AO_CoolValve_Pct    := 0.0;

    AHU_FAULT:
        rFanDemand_Pct      := 0.0;
        AO_DamperInlet_Pct  := 0.0;
        AO_DamperExhaust_Pct:= 0.0;
        AO_HeatValve_Pct    := 0.0;
        AO_CoolValve_Pct    := 0.0;

END_CASE

(* ================================================================
   8)  FAN SPEED RAMP + EXHAUST TRACKING (FOLLOW_SUPPLY)
   ================================================================ *)
(* Ramp the supply speed reference toward the demand (startMode = RAMP) *)
rRamp := C_FAN_RAMP_PCT_S * rDt_s;
IF AO_FanSupply_Pct < rFanDemand_Pct THEN
    AO_FanSupply_Pct := AO_FanSupply_Pct + rRamp;
    IF AO_FanSupply_Pct > rFanDemand_Pct THEN AO_FanSupply_Pct := rFanDemand_Pct; END_IF
ELSIF AO_FanSupply_Pct > rFanDemand_Pct THEN
    AO_FanSupply_Pct := AO_FanSupply_Pct - rRamp;
    IF AO_FanSupply_Pct < rFanDemand_Pct THEN AO_FanSupply_Pct := rFanDemand_Pct; END_IF
END_IF

(* Clamp to min/max while running (0 allowed when stopping) *)
IF rFanDemand_Pct > 0.0 THEN
    IF AO_FanSupply_Pct > 0.0 AND AO_FanSupply_Pct < C_FAN_MIN_PCT THEN
        AO_FanSupply_Pct := C_FAN_MIN_PCT;
    END_IF
    IF AO_FanSupply_Pct > C_FAN_MAX_PCT THEN AO_FanSupply_Pct := C_FAN_MAX_PCT; END_IF
END_IF

(* Exhaust follows supply + tracking offset *)
AO_FanExhaust_Pct := AO_FanSupply_Pct + C_EXH_OFFSET_PCT;
IF AO_FanExhaust_Pct > C_FAN_MAX_PCT THEN AO_FanExhaust_Pct := C_FAN_MAX_PCT; END_IF
IF AO_FanExhaust_Pct < 0.0 THEN AO_FanExhaust_Pct := 0.0; END_IF

(* Fan run enables *)
DO_FanSupply_Run  := (AO_FanSupply_Pct  > 0.0);
DO_FanExhaust_Run := (AO_FanExhaust_Pct > 0.0);

(* ================================================================
   9)  HEATING CIRCULATION PUMP  (demand + run-on + frost)
   ================================================================ *)
(* Pump runs on heating demand or frost; keeps running for pumpRunOnS after demand ends *)
tonPumpRunOn(IN := NOT bHeatDemand, PT := C_PUMP_RUNON_S);
IF bHeatDemand OR bFrostActive OR bFrostLatched THEN
    DO_HeatPump_Run := TRUE;
ELSIF NOT tonPumpRunOn.Q THEN
    DO_HeatPump_Run := TRUE;            (* still within run-on window *)
ELSE
    DO_HeatPump_Run := FALSE;
END_IF

(* ================================================================
   10) COMMON ALARM
   ================================================================ *)
DO_CommonAlarm := bFrostLatched OR bFanFaultLatched OR bSensorFaultLatched
                  OR DI_FireAlarm OR bRWaterAlarm;

END_PROGRAM


(*
================================================================================
 Enumeration used by this program.
 In CoDeSys declare this as a DUT (Data Unit Type) named E_AHU_STATE.
================================================================================
TYPE E_AHU_STATE :
(
    AHU_OFF      := 0,
    AHU_STARTUP  := 1,
    AHU_RUN      := 2,
    AHU_SHUTDOWN := 3,
    AHU_FROST    := 4,
    AHU_FIRE     := 5,
    AHU_FAULT    := 6
);
END_TYPE
================================================================================
*)
