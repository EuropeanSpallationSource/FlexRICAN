(*
================================================================================
 AHU_06 - Air Handling Unit Control Program  (group "Fresh_air_unit_El_Heater")
 Target  : CoDeSys 3.5 (IEC 61131-3), Structured Text
 Task     : cyclic, 200 ms (T#200MS)
 Source  : 10ahus_selection_AHU6.json  (project "10AHUs")
--------------------------------------------------------------------------------
 UNIT TYPE
   Supply-only fresh-air HEATING unit with an ELECTRIC heater:
     - one supply VFD fan (FAN_SUPPLY_VFD)
     - inlet (outdoor) and exhaust (relief) dampers (modulating, fail close)
     - an ELECTRIC heating coil (Coil-Electric, AO_0_10V) - the only conditioning
       device (NO cooling, NO heat recovery, NO water coil)
     - sensors: supply T/RH, room T, supply-air HIGH-limit thermostat
   Heating only: when the space is at/above setpoint the heater simply closes;
   there is no cooling device.
--------------------------------------------------------------------------------
 ELECTRIC-COIL SAFETY
   - airflow interlock : no electric heat without proven supply airflow
   - supply HIGH-limit : thermostat (or soft supply-temp limit) latches the
                         heater off and raises the alarm
   - fan cool-down     : on shutdown the fan keeps running to dissipate residual
                         element heat before stopping
   The water-coil FROST state exists in the enum for consistency but is never
   entered on this (electric, water-less) unit.
================================================================================
*)

PROGRAM AHU_06
VAR CONSTANT
    C_CYCLE_MS          : TIME := T#200MS;

    (* Sequence (AHU_SEQUENCE) *)
    C_SUPPLY_SP_C        : REAL := 18.0;        (* supplyTempSpC      *)
    C_ROOM_SP_C          : REAL := 22.0;        (* roomTempSpC        *)
    C_MIN_SUPPLY_C       : REAL := 12.0;        (* minSupplyTempC     *)
    C_MAX_SUPPLY_C       : REAL := 30.0;        (* maxSupplyTempC     *)
    C_STARTUP_DAMPER_S   : TIME := T#20S;       (* startupDamperWaitS *)
    C_STARTUP_FAN_S      : TIME := T#10S;       (* startupFanWaitS    *)
    C_SHUTDOWN_PURGE_S   : TIME := T#0S;        (* shutdownPurgeS     *)

    (* Supply fan (FAN_SUPPLY_VFD) *)
    C_FAN_MIN_PCT        : REAL := 20.0;        (* minSpeedPct        *)
    C_FAN_MAX_PCT        : REAL := 100.0;       (* maxSpeedPct        *)
    C_FAN_SP_PCT         : REAL := 60.0;        (* speedSpPct (fixed supply rate) *)
    C_FAN_RAMP_PCT_S     : REAL := 10.0;        (* ramp %/s (RAMP)    *)
    C_FAN_PROOF_S        : TIME := T#15S;

    (* Dampers (MODULATING, fail CLOSE) *)
    C_DAMPER_OPEN_S      : TIME := T#120S;      (* openTimeS          *)

    (* Electric heater (Coil-Electric) *)
    C_EL_MIN_PCT         : REAL := 0.0;
    C_EL_MAX_PCT         : REAL := 100.0;
    C_EL_COOLDOWN_S      : TIME := T#60S;        (* (default) fan overrun after electric heat *)
    C_SUPPLY_HILIMIT_C   : REAL := 40.0;         (* (default) soft supply high-limit cutout *)

    (* Sensor plausibility *)
    C_TEMP_RANGE_LO_C    : REAL := -20.0;
    C_TEMP_RANGE_HI_C    : REAL := 80.0;

    (* Cascade room / supply PI (heating only) *)
    C_ROOM_KP            : REAL := 1.5;
    C_ROOM_TI_S          : REAL := 600.0;
    C_SUP_KP             : REAL := 12.0;
    C_SUP_TI_S           : REAL := 180.0;
END_VAR

VAR
    (* =========================== I/O IMAGE =========================== *)
    (* ---- Digital inputs ---- *)
    DI_FireAlarm         : BOOL;
    DI_FanSupply_Run     : BOOL;        (* supply fan run feedback (fan dP) *)
    DI_FanSupply_Fault   : BOOL;
    DI_FilterSupply_Dirty: BOOL;        (* inlet filter dP switch *)
    DI_SupplyHiLimit     : BOOL;        (* supply-air HIGH-limit thermostat (electric overheat) *)
    DI_Reset             : BOOL;
    DI_RemoteEnable      : BOOL := TRUE;

    (* ---- Analog inputs ---- *)
    AI_T_Supply_C        : REAL;        (* th_supply temperature *)
    AI_RH_Supply_pct     : REAL;        (* th_supply humidity    *)
    AI_T_Room_C          : REAL;        (* SENSOR_T_ROOM (aiT_Room) *)

    (* ---- Digital outputs ---- *)
    DO_FanSupply_Run     : BOOL;
    DO_ElHeater_Enable   : BOOL;        (* electric coil safety contactor *)
    DO_CommonAlarm       : BOOL;

    (* ---- Analog outputs (0..100 %) ---- *)
    AO_FanSupply_Pct     : REAL;
    AO_DamperInlet_Pct   : REAL;        (* outdoor-air damper *)
    AO_DamperExhaust_Pct : REAL;        (* exhaust / relief damper *)
    AO_ElHeater_Pct      : REAL;        (* electric heater 0..100 (0-10V) *)

    (* =========================== STATE =========================== *)
    eState               : E_AHU_STATE := AHU_OFF;
    eState_prev          : E_AHU_STATE := AHU_OFF;
    bEnableCmd           : BOOL;

    bFanFaultLatched     : BOOL;
    bSensorFaultLatched  : BOOL;
    bElHiLimitLatched    : BOOL;        (* electric high-limit latched *)
    bHiLimitActive       : BOOL;

    bFilterWarn          : BOOL;

    rSupplySetpoint_C    : REAL := 18.0;
    rRoom_I              : REAL;
    rSup_I               : REAL;
    rHeatDemand_Pct      : REAL;        (* electric heating demand 0..100 *)
    rFanDemand_Pct       : REAL;

    bElActive            : BOOL;
    bElCooldownActive    : BOOL;

    rDt_s                : REAL := 0.2;

    (* helpers *)
    rErr                 : REAL;
    rRamp                : REAL;

    (* =========================== TIMERS =========================== *)
    tonDamperWait        : TON;
    tonFanWait           : TON;
    tonPurge             : TON;
    tonSupplyProof       : TON;
    tonElCooldown        : TON;
END_VAR

(* ================================================================
   0)  HOUSEKEEPING
   ================================================================ *)
rDt_s := DINT_TO_REAL(TIME_TO_DINT(C_CYCLE_MS)) / 1000.0;
eState_prev := eState;
bEnableCmd := DI_RemoteEnable AND NOT bFanFaultLatched AND NOT bSensorFaultLatched;

(* ================================================================
   1)  SENSOR SUPERVISION
   ================================================================ *)
IF  (AI_T_Supply_C < C_TEMP_RANGE_LO_C) OR (AI_T_Supply_C > C_TEMP_RANGE_HI_C)
 OR (AI_T_Room_C   < C_TEMP_RANGE_LO_C) OR (AI_T_Room_C   > C_TEMP_RANGE_HI_C)
THEN
    bSensorFaultLatched := TRUE;
END_IF

bFilterWarn := DI_FilterSupply_Dirty;

(* ================================================================
   2)  FAN FAULT SUPERVISION
   ================================================================ *)
tonSupplyProof(IN := DO_FanSupply_Run AND NOT DI_FanSupply_Run, PT := C_FAN_PROOF_S);
IF DI_FanSupply_Fault OR tonSupplyProof.Q THEN
    bFanFaultLatched := TRUE;
END_IF

(* ================================================================
   3)  ELECTRIC SUPPLY HIGH-LIMIT (overheat)
       hardware thermostat OR soft supply-temperature limit
   ================================================================ *)
bHiLimitActive := DI_SupplyHiLimit OR (AI_T_Supply_C >= C_SUPPLY_HILIMIT_C);
IF bHiLimitActive THEN
    bElHiLimitLatched := TRUE;
END_IF

(* ================================================================
   4)  ALARM RESET
   ================================================================ *)
IF DI_Reset THEN
    IF NOT DI_FanSupply_Fault THEN bFanFaultLatched := FALSE; END_IF
    IF NOT bHiLimitActive      THEN bElHiLimitLatched := FALSE; END_IF
    bSensorFaultLatched := FALSE;
END_IF

(* ================================================================
   5)  SEQUENCE STATE MACHINE  (FIRE > FAULT > flow ; no FROST)
   ================================================================ *)
IF DI_FireAlarm THEN
    eState := AHU_FIRE;
ELSIF (bFanFaultLatched OR bSensorFaultLatched) AND (eState <> AHU_OFF) THEN
    eState := AHU_FAULT;
END_IF

CASE eState OF

    AHU_OFF:
        IF bEnableCmd AND NOT DI_FireAlarm THEN
            eState := AHU_STARTUP;
        END_IF

    AHU_STARTUP:
        tonDamperWait(IN := TRUE, PT := C_STARTUP_DAMPER_S);
        IF tonDamperWait.Q THEN
            tonFanWait(IN := TRUE, PT := C_STARTUP_FAN_S);
        END_IF
        IF tonFanWait.Q AND DI_FanSupply_Run THEN
            eState := AHU_RUN;
        END_IF
        IF NOT bEnableCmd THEN eState := AHU_SHUTDOWN; END_IF

    AHU_RUN:
        tonDamperWait(IN := FALSE);
        tonFanWait(IN := FALSE);
        IF NOT bEnableCmd THEN eState := AHU_SHUTDOWN; END_IF

    AHU_SHUTDOWN:
        (* hold fan until purge done AND electric cool-down elapsed *)
        tonPurge(IN := TRUE, PT := C_SHUTDOWN_PURGE_S);
        bElCooldownActive := NOT tonElCooldown.Q;
        IF tonPurge.Q AND NOT bElCooldownActive THEN
            tonPurge(IN := FALSE);
            eState := AHU_OFF;
        END_IF

    AHU_FROST:
        (* not used on this unit *)
        eState := AHU_OFF;

    AHU_FIRE:
        IF NOT DI_FireAlarm THEN eState := AHU_OFF; END_IF

    AHU_FAULT:
        IF NOT (bFanFaultLatched OR bSensorFaultLatched) THEN eState := AHU_OFF; END_IF

END_CASE

(* ================================================================
   6)  CASCADE HEATING CONTROL  (room -> supply setpoint -> heater)
       Heating only: demand clamped to 0..100 (no cooling device).
   ================================================================ *)
IF eState = AHU_RUN THEN
    (* outer room PI -> supply setpoint *)
    rErr    := C_ROOM_SP_C - AI_T_Room_C;
    rRoom_I := rRoom_I + (C_ROOM_KP / C_ROOM_TI_S) * rErr * rDt_s;
    IF rRoom_I > (C_MAX_SUPPLY_C - C_SUPPLY_SP_C) THEN rRoom_I := (C_MAX_SUPPLY_C - C_SUPPLY_SP_C); END_IF
    IF rRoom_I < (C_MIN_SUPPLY_C - C_SUPPLY_SP_C) THEN rRoom_I := (C_MIN_SUPPLY_C - C_SUPPLY_SP_C); END_IF
    rSupplySetpoint_C := C_SUPPLY_SP_C + (C_ROOM_KP * rErr) + rRoom_I;
    IF rSupplySetpoint_C > C_MAX_SUPPLY_C THEN rSupplySetpoint_C := C_MAX_SUPPLY_C; END_IF
    IF rSupplySetpoint_C < C_MIN_SUPPLY_C THEN rSupplySetpoint_C := C_MIN_SUPPLY_C; END_IF

    (* inner supply PI -> heating demand 0..100 *)
    rErr   := rSupplySetpoint_C - AI_T_Supply_C;
    rSup_I := rSup_I + (C_SUP_KP / C_SUP_TI_S) * rErr * rDt_s;
    IF rSup_I > 100.0 THEN rSup_I := 100.0; END_IF
    IF rSup_I < 0.0   THEN rSup_I := 0.0;   END_IF
    rHeatDemand_Pct := (C_SUP_KP * rErr) + rSup_I;
    IF rHeatDemand_Pct > 100.0 THEN rHeatDemand_Pct := 100.0; END_IF
    IF rHeatDemand_Pct < 0.0   THEN rHeatDemand_Pct := 0.0;   END_IF
ELSE
    rRoom_I := 0.0;  rSup_I := 0.0;  rHeatDemand_Pct := 0.0;
    rSupplySetpoint_C := C_SUPPLY_SP_C;
END_IF

(* ================================================================
   7)  OUTPUT MAPPING PER STATE
   ================================================================ *)
CASE eState OF

    AHU_OFF:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;
        AO_DamperExhaust_Pct := 0.0;
        AO_ElHeater_Pct      := 0.0;

    AHU_SHUTDOWN:
        AO_ElHeater_Pct      := 0.0;
        IF bElCooldownActive THEN
            rFanDemand_Pct       := C_FAN_MIN_PCT;     (* keep airflow for cool-down *)
            AO_DamperInlet_Pct   := 100.0;
            AO_DamperExhaust_Pct := 100.0;
        ELSE
            rFanDemand_Pct       := 0.0;
            AO_DamperInlet_Pct   := 0.0;
            AO_DamperExhaust_Pct := 0.0;
        END_IF

    AHU_STARTUP:
        AO_DamperInlet_Pct   := 100.0;
        AO_DamperExhaust_Pct := 100.0;
        IF tonFanWait.IN THEN rFanDemand_Pct := C_FAN_SP_PCT; ELSE rFanDemand_Pct := 0.0; END_IF
        AO_ElHeater_Pct      := 0.0;

    AHU_RUN:
        AO_DamperInlet_Pct   := 100.0;
        AO_DamperExhaust_Pct := 100.0;
        rFanDemand_Pct       := C_FAN_SP_PCT;          (* fixed supply rate *)
        AO_ElHeater_Pct      := rHeatDemand_Pct;       (* limited by safety below *)

    AHU_FROST:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;
        AO_DamperExhaust_Pct := 0.0;
        AO_ElHeater_Pct      := 0.0;

    AHU_FIRE:
        (* fireBehavior = FANS_OFF_DAMPERS_CLOSE *)
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;
        AO_DamperExhaust_Pct := 0.0;
        AO_ElHeater_Pct      := 0.0;

    AHU_FAULT:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;
        AO_DamperExhaust_Pct := 0.0;
        AO_ElHeater_Pct      := 0.0;

END_CASE

(* ================================================================
   8)  ELECTRIC HEATER SAFETY INTERLOCKS
       - only in RUN, only with proven supply airflow
       - blocked by high-limit latch
   ================================================================ *)
IF bElHiLimitLatched OR (eState <> AHU_RUN) OR NOT DI_FanSupply_Run THEN
    AO_ElHeater_Pct := 0.0;
END_IF
IF AO_ElHeater_Pct > C_EL_MAX_PCT THEN AO_ElHeater_Pct := C_EL_MAX_PCT; END_IF
bElActive          := (AO_ElHeater_Pct > 0.0);
DO_ElHeater_Enable := bElActive;

(* cool-down timer: counts time since the electric coil last switched off *)
tonElCooldown(IN := NOT bElActive, PT := C_EL_COOLDOWN_S);

(* ================================================================
   9)  FAN SPEED RAMP
   ================================================================ *)
rRamp := C_FAN_RAMP_PCT_S * rDt_s;
IF AO_FanSupply_Pct < rFanDemand_Pct THEN
    AO_FanSupply_Pct := AO_FanSupply_Pct + rRamp;
    IF AO_FanSupply_Pct > rFanDemand_Pct THEN AO_FanSupply_Pct := rFanDemand_Pct; END_IF
ELSIF AO_FanSupply_Pct > rFanDemand_Pct THEN
    AO_FanSupply_Pct := AO_FanSupply_Pct - rRamp;
    IF AO_FanSupply_Pct < rFanDemand_Pct THEN AO_FanSupply_Pct := rFanDemand_Pct; END_IF
END_IF

IF rFanDemand_Pct > 0.0 THEN
    IF AO_FanSupply_Pct > 0.0 AND AO_FanSupply_Pct < C_FAN_MIN_PCT THEN
        AO_FanSupply_Pct := C_FAN_MIN_PCT;
    END_IF
    IF AO_FanSupply_Pct > C_FAN_MAX_PCT THEN AO_FanSupply_Pct := C_FAN_MAX_PCT; END_IF
END_IF

DO_FanSupply_Run := (AO_FanSupply_Pct > 0.0);

(* ================================================================
   10) COMMON ALARM
   ================================================================ *)
DO_CommonAlarm := bFanFaultLatched OR bSensorFaultLatched
                  OR bElHiLimitLatched OR DI_FireAlarm;

END_PROGRAM


(*
================================================================================
 DUT used by this program (declare as E_AHU_STATE):
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
