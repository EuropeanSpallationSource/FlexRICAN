(*
================================================================================
 AHU_04 - Air Handling Unit Control Program  (group "Exhaust_Vent_Room_Temp")
 Target  : CoDeSys 3.5 (IEC 61131-3), Structured Text
 Task     : cyclic, 200 ms (T#200MS)
 Source  : 10ahus_selection_AHU4.json  (project "10AHUs")
--------------------------------------------------------------------------------
 UNIT TYPE
   Exhaust-only ventilation unit that controls ROOM TEMPERATURE by modulating
   the extract ventilation rate (free cooling by extraction). It has:
     - one extract VFD fan (FAN_EXHAUST_VFD)
     - an inlet make-up-air damper and an exhaust damper (modulating, fail close)
     - a single room-temperature sensor (SENSOR_T_ROOM)
   There is NO heating coil, cooling coil, heat recovery, humidity sensor,
   water circuit or supply fan. Consequently there is no frost protection and
   no FROST trigger (the FROST state exists in the enum but is never entered on
   this unit).
--------------------------------------------------------------------------------
 CONTROL PRINCIPLE
   A direct-acting PI on room temperature sets the extract fan speed around a
   base ventilation rate: the warmer the room above setpoint, the higher the
   extract rate (pulling in cooler make-up air); below setpoint the fan falls
   back to its minimum ventilation speed.
================================================================================
*)

PROGRAM AHU_04
VAR CONSTANT
    C_CYCLE_MS          : TIME := T#200MS;

    (* Sequence (AHU_SEQUENCE) *)
    C_ROOM_SP_C          : REAL := 22.0;        (* roomTempSpC        *)
    C_STARTUP_DAMPER_S   : TIME := T#20S;       (* startupDamperWaitS *)
    C_STARTUP_FAN_S      : TIME := T#10S;       (* startupFanWaitS    *)
    C_SHUTDOWN_PURGE_S   : TIME := T#0S;        (* shutdownPurgeS     *)

    (* Extract fan (FAN_EXHAUST_VFD) *)
    C_FAN_MIN_PCT        : REAL := 20.0;        (* minSpeedPct        *)
    C_FAN_MAX_PCT        : REAL := 100.0;       (* maxSpeedPct        *)
    C_VENT_BASE_PCT      : REAL := 60.0;        (* speedSpPct (base ventilation) *)
    C_FAN_RAMP_PCT_S     : REAL := 10.0;        (* ramp %/s (startMode = RAMP) *)
    C_FAN_PROOF_S        : TIME := T#15S;

    (* Dampers (MODULATING, fail CLOSE) *)
    C_DAMPER_OPEN_S      : TIME := T#120S;      (* openTimeS          *)

    (* Sensor plausibility *)
    C_TEMP_RANGE_LO_C    : REAL := -20.0;
    C_TEMP_RANGE_HI_C    : REAL := 80.0;

    (* Room-temperature ventilation PI (direct acting: warm -> more extract) *)
    C_ROOM_KP            : REAL := 15.0;        (* % speed per K above setpoint - (default) *)
    C_ROOM_TI_S          : REAL := 600.0;       (* integral time, s         - (default) *)
END_VAR

VAR
    (* =========================== I/O IMAGE =========================== *)
    (* ---- Digital inputs ---- *)
    DI_FireAlarm         : BOOL;
    DI_FanExhaust_Run    : BOOL;        (* extract fan run feedback (fan dP) *)
    DI_FanExhaust_Fault  : BOOL;        (* extract fan fault                 *)
    DI_Reset             : BOOL;        (* operator alarm reset              *)
    DI_RemoteEnable      : BOOL := TRUE;(* external enable (BMS / schedule)  *)

    (* ---- Analog inputs ---- *)
    AI_T_Room_C          : REAL;        (* SENSOR_T_ROOM (aiT_Room)          *)

    (* ---- Digital outputs ---- *)
    DO_FanExhaust_Run    : BOOL;        (* extract fan enable                *)
    DO_CommonAlarm       : BOOL;

    (* ---- Analog outputs (0..100 %) ---- *)
    AO_FanExhaust_Pct    : REAL;        (* extract VFD speed reference       *)
    AO_DamperInlet_Pct   : REAL;        (* inlet make-up-air damper          *)
    AO_DamperExhaust_Pct : REAL;        (* exhaust-air damper                *)

    (* =========================== STATE =========================== *)
    eState               : E_AHU_STATE := AHU_OFF;
    eState_prev          : E_AHU_STATE := AHU_OFF;
    bEnableCmd           : BOOL;

    bFanFaultLatched     : BOOL;
    bSensorFaultLatched  : BOOL;

    rRoom_I              : REAL;        (* PI integral term                  *)
    rFanDemand_Pct       : REAL;        (* commanded extract speed (target)  *)

    rDt_s                : REAL := 0.2;

    (* helpers *)
    rErr                 : REAL;
    rRamp                : REAL;

    (* =========================== TIMERS =========================== *)
    tonDamperWait        : TON;
    tonFanWait           : TON;
    tonPurge             : TON;
    tonExhaustProof      : TON;
END_VAR

(* ================================================================
   0)  HOUSEKEEPING
   ================================================================ *)
rDt_s := DINT_TO_REAL(TIME_TO_DINT(C_CYCLE_MS)) / 1000.0;
eState_prev := eState;
bEnableCmd := DI_RemoteEnable AND NOT bFanFaultLatched AND NOT bSensorFaultLatched;

(* ================================================================
   1)  SENSOR SUPERVISION  (room temperature plausibility)
   ================================================================ *)
IF (AI_T_Room_C < C_TEMP_RANGE_LO_C) OR (AI_T_Room_C > C_TEMP_RANGE_HI_C) THEN
    bSensorFaultLatched := TRUE;
END_IF

(* ================================================================
   2)  FAN FAULT SUPERVISION
       - hard fault input
       - missing proof of flow while commanded ON
   ================================================================ *)
tonExhaustProof(IN := DO_FanExhaust_Run AND NOT DI_FanExhaust_Run, PT := C_FAN_PROOF_S);
IF DI_FanExhaust_Fault OR tonExhaustProof.Q THEN
    bFanFaultLatched := TRUE;
END_IF

(* ================================================================
   3)  ALARM RESET
   ================================================================ *)
IF DI_Reset THEN
    IF NOT DI_FanExhaust_Fault THEN bFanFaultLatched := FALSE; END_IF
    bSensorFaultLatched := FALSE;
END_IF

(* ================================================================
   4)  SEQUENCE STATE MACHINE  (FIRE > FAULT > flow ; no FROST on this unit)
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
        IF tonFanWait.Q AND DI_FanExhaust_Run THEN
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
        (* Not used on this unit (no water coil); fail safe back to OFF *)
        eState := AHU_OFF;

    AHU_FIRE:
        IF NOT DI_FireAlarm THEN eState := AHU_OFF; END_IF

    AHU_FAULT:
        IF NOT (bFanFaultLatched OR bSensorFaultLatched) THEN eState := AHU_OFF; END_IF

END_CASE

(* ================================================================
   5)  ROOM-TEMPERATURE VENTILATION CONTROL (direct acting PI)
       Active only in RUN. Output = extract fan speed around base rate.
   ================================================================ *)
IF eState = AHU_RUN THEN
    rErr := AI_T_Room_C - C_ROOM_SP_C;            (* >0 when room too warm *)
    rRoom_I := rRoom_I + (C_ROOM_KP / C_ROOM_TI_S) * rErr * rDt_s;
    (* anti-windup: keep base + I inside [min..max] *)
    IF (C_VENT_BASE_PCT + rRoom_I) > C_FAN_MAX_PCT THEN rRoom_I := C_FAN_MAX_PCT - C_VENT_BASE_PCT; END_IF
    IF (C_VENT_BASE_PCT + rRoom_I) < C_FAN_MIN_PCT THEN rRoom_I := C_FAN_MIN_PCT - C_VENT_BASE_PCT; END_IF

    rFanDemand_Pct := C_VENT_BASE_PCT + (C_ROOM_KP * rErr) + rRoom_I;
    IF rFanDemand_Pct > C_FAN_MAX_PCT THEN rFanDemand_Pct := C_FAN_MAX_PCT; END_IF
    IF rFanDemand_Pct < C_FAN_MIN_PCT THEN rFanDemand_Pct := C_FAN_MIN_PCT; END_IF
ELSE
    rRoom_I := 0.0;
END_IF

(* ================================================================
   6)  OUTPUT MAPPING PER STATE
   ================================================================ *)
CASE eState OF

    AHU_OFF, AHU_SHUTDOWN, AHU_FROST:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;
        AO_DamperExhaust_Pct := 0.0;

    AHU_STARTUP:
        AO_DamperInlet_Pct   := 100.0;
        AO_DamperExhaust_Pct := 100.0;
        IF tonFanWait.IN THEN rFanDemand_Pct := C_VENT_BASE_PCT; ELSE rFanDemand_Pct := 0.0; END_IF

    AHU_RUN:
        AO_DamperInlet_Pct   := 100.0;
        AO_DamperExhaust_Pct := 100.0;
        (* rFanDemand_Pct set by the room-temperature PI in section 5 *)

    AHU_FIRE:
        (* fireBehavior = FANS_OFF_DAMPERS_CLOSE *)
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;
        AO_DamperExhaust_Pct := 0.0;

    AHU_FAULT:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;
        AO_DamperExhaust_Pct := 0.0;

END_CASE

(* ================================================================
   7)  FAN SPEED RAMP
   ================================================================ *)
rRamp := C_FAN_RAMP_PCT_S * rDt_s;
IF AO_FanExhaust_Pct < rFanDemand_Pct THEN
    AO_FanExhaust_Pct := AO_FanExhaust_Pct + rRamp;
    IF AO_FanExhaust_Pct > rFanDemand_Pct THEN AO_FanExhaust_Pct := rFanDemand_Pct; END_IF
ELSIF AO_FanExhaust_Pct > rFanDemand_Pct THEN
    AO_FanExhaust_Pct := AO_FanExhaust_Pct - rRamp;
    IF AO_FanExhaust_Pct < rFanDemand_Pct THEN AO_FanExhaust_Pct := rFanDemand_Pct; END_IF
END_IF

IF rFanDemand_Pct > 0.0 THEN
    IF AO_FanExhaust_Pct > 0.0 AND AO_FanExhaust_Pct < C_FAN_MIN_PCT THEN
        AO_FanExhaust_Pct := C_FAN_MIN_PCT;
    END_IF
    IF AO_FanExhaust_Pct > C_FAN_MAX_PCT THEN AO_FanExhaust_Pct := C_FAN_MAX_PCT; END_IF
END_IF

DO_FanExhaust_Run := (AO_FanExhaust_Pct > 0.0);

(* ================================================================
   8)  COMMON ALARM
   ================================================================ *)
DO_CommonAlarm := bFanFaultLatched OR bSensorFaultLatched OR DI_FireAlarm;

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
