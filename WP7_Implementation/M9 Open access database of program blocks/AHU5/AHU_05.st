(*
================================================================================
 AHU_05 - Air Handling Unit Control Program  (group "FreshAir Unit1")
 Target  : CoDeSys 3.5 (IEC 61131-3), Structured Text
 Task     : cyclic, 200 ms (T#200MS)
 Source  : 10ahus_selection_AHU5.json  (project "10AHUs")
--------------------------------------------------------------------------------
 UNIT TYPE
   100 % fresh-air supply/exhaust unit with PLATE energy recovery (ERV) and a
   modulating recovery BYPASS damper (canvas group "Klap_Bypass"):
     - supply + exhaust VFD fans (exhaust tracks supply, FOLLOW_SUPPLY)
     - outdoor-air and exhaust isolation dampers (modulating, fail close)
     - plate ERV with modulating bypass damper (free heating / free cooling)
     - water heating coil (3-way valve) with frost protection + circulation pump
     - cooling coil (2-way valve)
     - sensors: outdoor T, supply T/RH, exhaust T/RH, room T/RH, return-water T
--------------------------------------------------------------------------------
 RELATION TO AHU_01
   Same plant as AHU_01 but the ERV bypass damper is actively modulated here as
   the first (free-energy) stage of heating and cooling, ahead of the coils.
--------------------------------------------------------------------------------
 BYPASS CONVENTION
   AO_ERVBypass_Pct : 0 % = damper fully through the recovery core (MAX recovery)
                    100 % = air fully bypasses the core            (NO recovery)
   Internally a recovery fraction 0..100 % is computed; the bypass actuator is
   driven with (100 - recovery).
================================================================================
*)

PROGRAM AHU_05
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

    (* Fans *)
    C_FAN_MIN_PCT        : REAL := 20.0;        (* minSpeedPct        *)
    C_FAN_MAX_PCT        : REAL := 100.0;       (* maxSpeedPct        *)
    C_FAN_SP_PCT         : REAL := 60.0;        (* speedSpPct         *)
    C_FAN_RAMP_PCT_S     : REAL := 10.0;        (* ramp %/s (RAMP)    *)
    C_EXH_OFFSET_PCT     : REAL := 0.0;         (* trackingOffsetPct  *)
    C_FAN_PROOF_S        : TIME := T#15S;

    (* Dampers (MODULATING, fail CLOSE) *)
    C_DAMPER_OPEN_S      : TIME := T#120S;      (* openTimeS          *)

    (* ERV plate + bypass damper *)
    C_HR_DT_MIN_K        : REAL := 1.0;         (* (default) min extract<->outdoor dT to use recovery *)

    (* Water heating coil + frost (HEATING_COIL_3WV) *)
    C_FROST_TRIP_C       : REAL := 7.0;         (* tReturnTripC       *)
    C_FROST_RELEASE_C    : REAL := 10.0;        (* tReturnReleaseC    *)
    C_PUMP_RUNON_S       : TIME := T#120S;      (* pumpRunOnS         *)

    (* Return-water supervision (temp_water_return) *)
    C_RW_ALARM_LOW_C     : REAL := 5.0;
    C_RW_ALARM_HIGH_C    : REAL := 60.0;
    C_RW_WARN_LOW_C      : REAL := 10.0;
    C_RW_WARN_HIGH_C     : REAL := 55.0;

    (* Sensor plausibility *)
    C_TEMP_RANGE_LO_C    : REAL := -20.0;
    C_TEMP_RANGE_HI_C    : REAL := 80.0;

    (* Cascade room / supply PI *)
    C_ROOM_KP            : REAL := 1.5;
    C_ROOM_TI_S          : REAL := 600.0;
    C_SUP_KP             : REAL := 12.0;
    C_SUP_TI_S           : REAL := 180.0;
    C_SEQ_DEADBAND_PCT   : REAL := 5.0;
END_VAR

VAR
    (* =========================== I/O IMAGE =========================== *)
    (* ---- Digital inputs ---- *)
    DI_FireAlarm         : BOOL;
    DI_FanSupply_Run     : BOOL;
    DI_FanSupply_Fault   : BOOL;
    DI_FanExhaust_Run    : BOOL;
    DI_FanExhaust_Fault  : BOOL;
    DI_FilterSupply_Dirty: BOOL;
    DI_FilterExhaust_Dirty: BOOL;
    DI_Reset             : BOOL;
    DI_RemoteEnable      : BOOL := TRUE;

    (* ---- Analog inputs (engineering units) ---- *)
    AI_T_Outdoor_C       : REAL;        (* temp_outdoor (Ni1000)      *)
    AI_T_Supply_C        : REAL;        (* th_supply temperature      *)
    AI_RH_Supply_pct     : REAL;        (* th_supply humidity         *)
    AI_T_Exhaust_C       : REAL;        (* th_exhaust temperature     *)
    AI_RH_Exhaust_pct    : REAL;        (* th_exhaust humidity        *)
    AI_T_Room_C          : REAL;        (* temp_room                  *)
    AI_RH_Room_pct       : REAL;        (* rh_room                    *)
    AI_T_WaterReturn_C   : REAL;        (* temp_water_return          *)

    (* ---- Digital outputs ---- *)
    DO_FanSupply_Run     : BOOL;
    DO_FanExhaust_Run    : BOOL;
    DO_HeatPump_Run      : BOOL;
    DO_CommonAlarm       : BOOL;

    (* ---- Analog outputs (0..100 %) ---- *)
    AO_FanSupply_Pct     : REAL;
    AO_FanExhaust_Pct    : REAL;
    AO_DamperInlet_Pct   : REAL;        (* outdoor-air damper         *)
    AO_DamperExhaust_Pct : REAL;        (* exhaust-air damper         *)
    AO_ERVBypass_Pct     : REAL;        (* 0 = full recovery, 100 = full bypass *)
    AO_HeatValve_Pct     : REAL;        (* heating 3-way valve        *)
    AO_CoolValve_Pct     : REAL;        (* cooling valve              *)

    (* =========================== STATE =========================== *)
    eState               : E_AHU_STATE := AHU_OFF;
    eState_prev          : E_AHU_STATE := AHU_OFF;
    bEnableCmd           : BOOL;

    bFrostLatched        : BOOL;
    bFanFaultLatched     : BOOL;
    bSensorFaultLatched  : BOOL;

    bFilterWarn          : BOOL;
    bRWaterWarn          : BOOL;
    bRWaterAlarm         : BOOL;

    rSupplySetpoint_C    : REAL := 18.0;
    rRoom_I              : REAL;
    rSup_I               : REAL;
    rSeqOut_Pct          : REAL;
    rFanDemand_Pct       : REAL;
    rRecovery_Pct        : REAL;        (* internal recovery fraction 0..100 *)

    bFrostActive         : BOOL;
    bHeatDemand          : BOOL;
    bRecoveryHeatOK      : BOOL;
    bRecoveryCoolOK      : BOOL;

    rDt_s                : REAL := 0.2;

    (* helpers *)
    rErr                 : REAL;
    rRamp                : REAL;
    rAuth                : REAL;

    (* =========================== TIMERS =========================== *)
    tonDamperWait        : TON;
    tonFanWait           : TON;
    tonPurge             : TON;
    tonSupplyProof       : TON;
    tonExhaustProof      : TON;
    tonPumpRunOn         : TON;
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
IF  (AI_T_Outdoor_C    < C_TEMP_RANGE_LO_C) OR (AI_T_Outdoor_C    > C_TEMP_RANGE_HI_C)
 OR (AI_T_Supply_C     < C_TEMP_RANGE_LO_C) OR (AI_T_Supply_C     > C_TEMP_RANGE_HI_C)
 OR (AI_T_Room_C       < C_TEMP_RANGE_LO_C) OR (AI_T_Room_C       > C_TEMP_RANGE_HI_C)
 OR (AI_T_WaterReturn_C< C_TEMP_RANGE_LO_C) OR (AI_T_WaterReturn_C> C_TEMP_RANGE_HI_C)
THEN
    bSensorFaultLatched := TRUE;
END_IF

bRWaterWarn  := (AI_T_WaterReturn_C < C_RW_WARN_LOW_C)  OR (AI_T_WaterReturn_C > C_RW_WARN_HIGH_C);
bRWaterAlarm := (AI_T_WaterReturn_C < C_RW_ALARM_LOW_C) OR (AI_T_WaterReturn_C > C_RW_ALARM_HIGH_C);
bFilterWarn  := DI_FilterSupply_Dirty OR DI_FilterExhaust_Dirty;

(* ================================================================
   2)  FAN FAULT SUPERVISION
   ================================================================ *)
tonSupplyProof(IN := DO_FanSupply_Run AND NOT DI_FanSupply_Run, PT := C_FAN_PROOF_S);
tonExhaustProof(IN := DO_FanExhaust_Run AND NOT DI_FanExhaust_Run, PT := C_FAN_PROOF_S);
IF DI_FanSupply_Fault OR DI_FanExhaust_Fault OR tonSupplyProof.Q OR tonExhaustProof.Q THEN
    bFanFaultLatched := TRUE;
END_IF

(* ================================================================
   3)  FROST PROTECTION (water coil)
   ================================================================ *)
IF AI_T_WaterReturn_C <= C_FROST_TRIP_C THEN
    bFrostActive  := TRUE;
    bFrostLatched := TRUE;
ELSIF AI_T_WaterReturn_C >= C_FROST_RELEASE_C THEN
    bFrostActive := FALSE;
END_IF

(* ================================================================
   4)  ALARM RESET
   ================================================================ *)
IF DI_Reset THEN
    IF NOT bFrostActive THEN bFrostLatched := FALSE; END_IF
    IF NOT (DI_FanSupply_Fault OR DI_FanExhaust_Fault) THEN bFanFaultLatched := FALSE; END_IF
    bSensorFaultLatched := FALSE;
END_IF

(* ================================================================
   5)  SEQUENCE STATE MACHINE  (FIRE > FROST > FAULT > flow)
   ================================================================ *)
IF DI_FireAlarm THEN
    eState := AHU_FIRE;
ELSIF bFrostLatched AND (eState <> AHU_OFF) THEN
    eState := AHU_FROST;
ELSIF (bFanFaultLatched OR bSensorFaultLatched) AND (eState <> AHU_OFF) THEN
    eState := AHU_FAULT;
END_IF

CASE eState OF

    AHU_OFF:
        IF bEnableCmd AND NOT DI_FireAlarm AND NOT bFrostLatched THEN
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
        tonPurge(IN := TRUE, PT := C_SHUTDOWN_PURGE_S);
        IF tonPurge.Q THEN
            tonPurge(IN := FALSE);
            eState := AHU_OFF;
        END_IF

    AHU_FROST:
        IF NOT bFrostLatched THEN eState := AHU_OFF; END_IF

    AHU_FIRE:
        IF NOT DI_FireAlarm THEN eState := AHU_OFF; END_IF

    AHU_FAULT:
        IF NOT (bFanFaultLatched OR bSensorFaultLatched) THEN eState := AHU_OFF; END_IF

END_CASE

(* ================================================================
   6)  CASCADE TEMPERATURE CONTROL -> sequencer demand
   ================================================================ *)
IF eState = AHU_RUN THEN
    rErr    := C_ROOM_SP_C - AI_T_Room_C;
    rRoom_I := rRoom_I + (C_ROOM_KP / C_ROOM_TI_S) * rErr * rDt_s;
    IF rRoom_I > (C_MAX_SUPPLY_C - C_SUPPLY_SP_C) THEN rRoom_I := (C_MAX_SUPPLY_C - C_SUPPLY_SP_C); END_IF
    IF rRoom_I < (C_MIN_SUPPLY_C - C_SUPPLY_SP_C) THEN rRoom_I := (C_MIN_SUPPLY_C - C_SUPPLY_SP_C); END_IF
    rSupplySetpoint_C := C_SUPPLY_SP_C + (C_ROOM_KP * rErr) + rRoom_I;
    IF rSupplySetpoint_C > C_MAX_SUPPLY_C THEN rSupplySetpoint_C := C_MAX_SUPPLY_C; END_IF
    IF rSupplySetpoint_C < C_MIN_SUPPLY_C THEN rSupplySetpoint_C := C_MIN_SUPPLY_C; END_IF

    rErr   := rSupplySetpoint_C - AI_T_Supply_C;
    rSup_I := rSup_I + (C_SUP_KP / C_SUP_TI_S) * rErr * rDt_s;
    IF rSup_I >  100.0 THEN rSup_I :=  100.0; END_IF
    IF rSup_I < -100.0 THEN rSup_I := -100.0; END_IF
    rSeqOut_Pct := (C_SUP_KP * rErr) + rSup_I;
    IF rSeqOut_Pct >  100.0 THEN rSeqOut_Pct :=  100.0; END_IF
    IF rSeqOut_Pct < -100.0 THEN rSeqOut_Pct := -100.0; END_IF
ELSE
    rRoom_I := 0.0;  rSup_I := 0.0;  rSeqOut_Pct := 0.0;
    rSupplySetpoint_C := C_SUPPLY_SP_C;
END_IF

(* ================================================================
   7)  ERV RECOVERY (bypass) + COIL SEQUENCING
       Recovery used first, coil trims. Recovery only when the
       extract<->outdoor temperature difference makes it beneficial.
   ================================================================ *)
bRecoveryHeatOK := (AI_T_Exhaust_C - AI_T_Outdoor_C) > C_HR_DT_MIN_K;  (* extract warmer -> free heat *)
bRecoveryCoolOK := (AI_T_Outdoor_C - AI_T_Exhaust_C) > C_HR_DT_MIN_K;  (* extract cooler -> free cool *)

rRecovery_Pct    := 0.0;
AO_HeatValve_Pct := 0.0;
AO_CoolValve_Pct := 0.0;
bHeatDemand      := FALSE;

IF rSeqOut_Pct > C_SEQ_DEADBAND_PCT THEN
    (* ---- HEATING ---- *)
    rAuth := (rSeqOut_Pct - C_SEQ_DEADBAND_PCT) / (100.0 - C_SEQ_DEADBAND_PCT) * 100.0;
    IF bRecoveryHeatOK THEN
        rRecovery_Pct := rAuth * 2.0;                 (* 0..50 demand -> recovery 0..100 *)
        IF rRecovery_Pct > 100.0 THEN rRecovery_Pct := 100.0; END_IF
        AO_HeatValve_Pct := (rAuth - 50.0) * 2.0;     (* 50..100 demand -> valve 0..100 *)
        IF AO_HeatValve_Pct < 0.0 THEN AO_HeatValve_Pct := 0.0; END_IF
    ELSE
        rRecovery_Pct    := 0.0;
        AO_HeatValve_Pct := rAuth;
    END_IF
    bHeatDemand := (AO_HeatValve_Pct > 0.0);

ELSIF rSeqOut_Pct < -C_SEQ_DEADBAND_PCT THEN
    (* ---- COOLING ---- *)
    rAuth := (-rSeqOut_Pct - C_SEQ_DEADBAND_PCT) / (100.0 - C_SEQ_DEADBAND_PCT) * 100.0;
    IF bRecoveryCoolOK THEN
        rRecovery_Pct := rAuth * 2.0;
        IF rRecovery_Pct > 100.0 THEN rRecovery_Pct := 100.0; END_IF
        AO_CoolValve_Pct := (rAuth - 50.0) * 2.0;
        IF AO_CoolValve_Pct < 0.0 THEN AO_CoolValve_Pct := 0.0; END_IF
    ELSE
        rRecovery_Pct    := 0.0;
        AO_CoolValve_Pct := rAuth;
    END_IF
END_IF

(* ================================================================
   8)  OUTPUT MAPPING PER STATE
       Bypass actuator = 100 - recovery fraction.
   ================================================================ *)
CASE eState OF

    AHU_OFF, AHU_SHUTDOWN:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;
        AO_DamperExhaust_Pct := 0.0;
        rRecovery_Pct        := 0.0;
        AO_HeatValve_Pct     := 0.0;
        AO_CoolValve_Pct     := 0.0;

    AHU_STARTUP:
        AO_DamperInlet_Pct   := 100.0;
        AO_DamperExhaust_Pct := 100.0;
        IF tonFanWait.IN THEN rFanDemand_Pct := C_FAN_SP_PCT; ELSE rFanDemand_Pct := 0.0; END_IF
        rRecovery_Pct        := 0.0;
        AO_HeatValve_Pct     := 0.0;
        AO_CoolValve_Pct     := 0.0;

    AHU_RUN:
        AO_DamperInlet_Pct   := 100.0;
        AO_DamperExhaust_Pct := 100.0;
        rFanDemand_Pct       := C_FAN_SP_PCT;   (* recovery/coils set in section 7 *)

    AHU_FROST:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;
        AO_DamperExhaust_Pct := 0.0;
        rRecovery_Pct        := 0.0;
        AO_HeatValve_Pct     := 100.0;
        AO_CoolValve_Pct     := 0.0;
        bHeatDemand          := TRUE;

    AHU_FIRE:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;
        AO_DamperExhaust_Pct := 0.0;
        rRecovery_Pct        := 0.0;
        AO_HeatValve_Pct     := 0.0;
        AO_CoolValve_Pct     := 0.0;

    AHU_FAULT:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;
        AO_DamperExhaust_Pct := 0.0;
        rRecovery_Pct        := 0.0;
        AO_HeatValve_Pct     := 0.0;
        AO_CoolValve_Pct     := 0.0;

END_CASE

(* drive the bypass actuator from the recovery fraction *)
AO_ERVBypass_Pct := 100.0 - rRecovery_Pct;
IF AO_ERVBypass_Pct < 0.0   THEN AO_ERVBypass_Pct := 0.0;   END_IF
IF AO_ERVBypass_Pct > 100.0 THEN AO_ERVBypass_Pct := 100.0; END_IF

(* ================================================================
   9)  FAN SPEED RAMP + EXHAUST TRACKING
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

AO_FanExhaust_Pct := AO_FanSupply_Pct + C_EXH_OFFSET_PCT;
IF AO_FanExhaust_Pct > C_FAN_MAX_PCT THEN AO_FanExhaust_Pct := C_FAN_MAX_PCT; END_IF
IF AO_FanExhaust_Pct < 0.0 THEN AO_FanExhaust_Pct := 0.0; END_IF

DO_FanSupply_Run  := (AO_FanSupply_Pct  > 0.0);
DO_FanExhaust_Run := (AO_FanExhaust_Pct > 0.0);

(* ================================================================
   10) HEATING CIRCULATION PUMP (demand + run-on + frost)
   ================================================================ *)
tonPumpRunOn(IN := NOT bHeatDemand, PT := C_PUMP_RUNON_S);
IF bHeatDemand OR bFrostActive OR bFrostLatched THEN
    DO_HeatPump_Run := TRUE;
ELSIF NOT tonPumpRunOn.Q THEN
    DO_HeatPump_Run := TRUE;
ELSE
    DO_HeatPump_Run := FALSE;
END_IF

(* ================================================================
   11) COMMON ALARM
   ================================================================ *)
DO_CommonAlarm := bFrostLatched OR bFanFaultLatched OR bSensorFaultLatched
                  OR DI_FireAlarm OR bRWaterAlarm;

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
