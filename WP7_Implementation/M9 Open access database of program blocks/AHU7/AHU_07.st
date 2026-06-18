(*
================================================================================
 AHU_07 - Air Handling Unit Control Program  (group "Circulation_Clean_Hepa")
 Target  : CoDeSys 3.5 (IEC 61131-3), Structured Text
 Task     : cyclic, 200 ms (T#200MS)
 Source  : 10ahus_selection_AHU7.json  (project "10AHUs")
--------------------------------------------------------------------------------
 UNIT TYPE
   Recirculation CLEANROOM unit with HEPA filtration. Room air is drawn back,
   filtered (pre-filter + HEPA), conditioned and returned to the room:
     - one supply VFD fan, run at CONSTANT volume (cleanroom air changes / HEPA
       face velocity)
     - a single relief/recirculation damper (modulating, fail close)
     - water heating coil (3-way valve) with frost protection + circulation pump
     - cooling coil (2-way valve)
     - pre-filter and HEPA filter banks with differential-pressure monitoring
     - sensors: supply T/RH, exhaust(return) T/RH, room T/RH, return-water T
   No outdoor-air sensor, no exhaust fan and no heat recovery (recirculating).
--------------------------------------------------------------------------------
 CLEANROOM NOTES
   - The fan runs at a fixed speed in RUN so air-change rate / HEPA face velocity
     stays constant.
   - HEPA differential-pressure high is raised as an ALARM (filter loading /
     integrity) but does NOT stop the unit, so room cleanliness/pressurisation
     is maintained until service. The pre-filter dP is a warning only.
================================================================================
*)

PROGRAM AHU_07
VAR CONSTANT
    C_CYCLE_MS          : TIME := T#200MS;

    (* Sequence (AHU_SEQUENCE) *)
    C_SUPPLY_SP_C        : REAL := 18.0;
    C_ROOM_SP_C          : REAL := 22.0;
    C_MIN_SUPPLY_C       : REAL := 12.0;
    C_MAX_SUPPLY_C       : REAL := 30.0;
    C_STARTUP_DAMPER_S   : TIME := T#20S;
    C_STARTUP_FAN_S      : TIME := T#10S;
    C_SHUTDOWN_PURGE_S   : TIME := T#0S;

    (* Supply fan (FAN_SUPPLY_VFD) - constant volume *)
    C_FAN_MIN_PCT        : REAL := 20.0;
    C_FAN_MAX_PCT        : REAL := 100.0;
    C_FAN_SP_PCT         : REAL := 60.0;        (* speedSpPct (constant CR setpoint) *)
    C_FAN_RAMP_PCT_S     : REAL := 10.0;
    C_FAN_PROOF_S        : TIME := T#15S;

    (* Damper (MODULATING, fail CLOSE) *)
    C_DAMPER_OPEN_S      : TIME := T#120S;

    (* Water heating coil + frost (HEATING_COIL_3WV) *)
    C_FROST_TRIP_C       : REAL := 7.0;
    C_FROST_RELEASE_C    : REAL := 10.0;
    C_PUMP_RUNON_S       : TIME := T#120S;

    (* Return-water supervision *)
    C_RW_ALARM_LOW_C     : REAL := 5.0;
    C_RW_ALARM_HIGH_C    : REAL := 60.0;
    C_RW_WARN_LOW_C      : REAL := 10.0;
    C_RW_WARN_HIGH_C     : REAL := 55.0;

    (* Sensor plausibility *)
    C_TEMP_RANGE_LO_C    : REAL := -20.0;
    C_TEMP_RANGE_HI_C    : REAL := 80.0;

    (* Cascade room / supply PI + split range *)
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
    DI_FilterInlet_Dirty : BOOL;        (* pre-filter dP switch (warning)        *)
    DI_FilterHepa_High   : BOOL;        (* HEPA dP high (alarm)                  *)
    DI_Reset             : BOOL;
    DI_RemoteEnable      : BOOL := TRUE;

    (* ---- Analog inputs ---- *)
    AI_T_Supply_C        : REAL;        (* th_supply temperature                *)
    AI_RH_Supply_pct     : REAL;        (* th_supply humidity                   *)
    AI_T_Return_C        : REAL;        (* th_exhaust / return-air temperature  *)
    AI_RH_Return_pct     : REAL;        (* th_exhaust / return-air humidity     *)
    AI_T_Room_C          : REAL;        (* temp_room                            *)
    AI_RH_Room_pct       : REAL;        (* rh_room                              *)
    AI_T_WaterReturn_C   : REAL;        (* temp_water_return                    *)

    (* ---- Digital outputs ---- *)
    DO_FanSupply_Run     : BOOL;
    DO_HeatPump_Run      : BOOL;
    DO_CommonAlarm       : BOOL;

    (* ---- Analog outputs (0..100 %) ---- *)
    AO_FanSupply_Pct     : REAL;
    AO_Damper_Pct        : REAL;        (* relief / recirculation damper        *)
    AO_HeatValve_Pct     : REAL;        (* heating 3-way valve                  *)
    AO_CoolValve_Pct     : REAL;        (* cooling valve                        *)

    (* =========================== STATE =========================== *)
    eState               : E_AHU_STATE := AHU_OFF;
    eState_prev          : E_AHU_STATE := AHU_OFF;
    bEnableCmd           : BOOL;

    bFrostLatched        : BOOL;
    bFanFaultLatched     : BOOL;
    bSensorFaultLatched  : BOOL;

    bFilterWarn          : BOOL;        (* pre-filter dirty (warning)           *)
    bHepaAlarm           : BOOL;        (* HEPA dP high (alarm, non-stopping)   *)
    bRWaterWarn          : BOOL;
    bRWaterAlarm         : BOOL;

    rSupplySetpoint_C    : REAL := 18.0;
    rRoom_I              : REAL;
    rSup_I               : REAL;
    rSeqOut_Pct          : REAL;
    rFanDemand_Pct       : REAL;

    bFrostActive         : BOOL;
    bHeatDemand          : BOOL;

    rDt_s                : REAL := 0.2;

    (* helpers *)
    rErr                 : REAL;
    rRamp                : REAL;

    (* =========================== TIMERS =========================== *)
    tonDamperWait        : TON;
    tonFanWait           : TON;
    tonPurge             : TON;
    tonSupplyProof       : TON;
    tonPumpRunOn         : TON;
END_VAR

(* ================================================================
   0)  HOUSEKEEPING
   ================================================================ *)
rDt_s := DINT_TO_REAL(TIME_TO_DINT(C_CYCLE_MS)) / 1000.0;
eState_prev := eState;
bEnableCmd := DI_RemoteEnable AND NOT bFanFaultLatched AND NOT bSensorFaultLatched;

(* ================================================================
   1)  SENSOR SUPERVISION + FILTERS
   ================================================================ *)
IF  (AI_T_Supply_C     < C_TEMP_RANGE_LO_C) OR (AI_T_Supply_C     > C_TEMP_RANGE_HI_C)
 OR (AI_T_Room_C       < C_TEMP_RANGE_LO_C) OR (AI_T_Room_C       > C_TEMP_RANGE_HI_C)
 OR (AI_T_WaterReturn_C< C_TEMP_RANGE_LO_C) OR (AI_T_WaterReturn_C> C_TEMP_RANGE_HI_C)
THEN
    bSensorFaultLatched := TRUE;
END_IF

bRWaterWarn  := (AI_T_WaterReturn_C < C_RW_WARN_LOW_C)  OR (AI_T_WaterReturn_C > C_RW_WARN_HIGH_C);
bRWaterAlarm := (AI_T_WaterReturn_C < C_RW_ALARM_LOW_C) OR (AI_T_WaterReturn_C > C_RW_ALARM_HIGH_C);
bFilterWarn  := DI_FilterInlet_Dirty;           (* pre-filter -> warning only *)
bHepaAlarm   := DI_FilterHepa_High;             (* HEPA dP high -> alarm, non-stopping *)

(* ================================================================
   2)  FAN FAULT SUPERVISION
   ================================================================ *)
tonSupplyProof(IN := DO_FanSupply_Run AND NOT DI_FanSupply_Run, PT := C_FAN_PROOF_S);
IF DI_FanSupply_Fault OR tonSupplyProof.Q THEN
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
    IF NOT DI_FanSupply_Fault THEN bFanFaultLatched := FALSE; END_IF
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
   6)  CASCADE TEMPERATURE CONTROL -> split-range heat/cool
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

(* split-range with deadband *)
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
   7)  OUTPUT MAPPING PER STATE  (fan = constant volume in RUN)
   ================================================================ *)
CASE eState OF

    AHU_OFF, AHU_SHUTDOWN:
        rFanDemand_Pct   := 0.0;
        AO_Damper_Pct    := 0.0;
        AO_HeatValve_Pct := 0.0;
        AO_CoolValve_Pct := 0.0;

    AHU_STARTUP:
        AO_Damper_Pct    := 100.0;
        IF tonFanWait.IN THEN rFanDemand_Pct := C_FAN_SP_PCT; ELSE rFanDemand_Pct := 0.0; END_IF
        AO_HeatValve_Pct := 0.0;
        AO_CoolValve_Pct := 0.0;

    AHU_RUN:
        AO_Damper_Pct    := 100.0;
        rFanDemand_Pct   := C_FAN_SP_PCT;       (* constant volume; coils set in section 6 *)

    AHU_FROST:
        rFanDemand_Pct   := 0.0;
        AO_Damper_Pct    := 0.0;
        AO_HeatValve_Pct := 100.0;
        AO_CoolValve_Pct := 0.0;
        bHeatDemand      := TRUE;

    AHU_FIRE:
        rFanDemand_Pct   := 0.0;
        AO_Damper_Pct    := 0.0;
        AO_HeatValve_Pct := 0.0;
        AO_CoolValve_Pct := 0.0;

    AHU_FAULT:
        rFanDemand_Pct   := 0.0;
        AO_Damper_Pct    := 0.0;
        AO_HeatValve_Pct := 0.0;
        AO_CoolValve_Pct := 0.0;

END_CASE

(* ================================================================
   8)  FAN SPEED RAMP
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
   9)  HEATING CIRCULATION PUMP (demand + run-on + frost)
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
   10) COMMON ALARM  (HEPA dP high is included; does not stop the unit)
   ================================================================ *)
DO_CommonAlarm := bFrostLatched OR bFanFaultLatched OR bSensorFaultLatched
                  OR DI_FireAlarm OR bRWaterAlarm OR bHepaAlarm;

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
