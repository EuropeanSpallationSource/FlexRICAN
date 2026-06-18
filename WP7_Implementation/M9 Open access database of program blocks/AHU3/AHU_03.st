(*
================================================================================
 AHU_03 - Air Handling Unit Control Program  (group "DeHumi_Water_El_afterheating")
 Target  : CoDeSys 3.5 (IEC 61131-3), Structured Text
 Task     : cyclic, 200 ms (T#200MS)
 Source  : 10ahus_selection_AHU3.json  (project "10AHUs")
--------------------------------------------------------------------------------
 DIFFERENCES vs AHU_02
   + ELECTRIC after-heater coil downstream of the cooling coil (Coil-Electric).
     -> Heating sequence has THREE stages: recovery wheel -> water coil -> electric.
     -> Dehumidification reheats with the ELECTRIC coil (its design purpose here).
   + Electric-coil safety:
       - airflow interlock (no electric heat without proven supply airflow)
       - overheat thermostat input (latched, disables electric)
       - fan cool-down overrun on shutdown to dissipate residual element heat
   Air path order: wheel -> water heating coil -> cooling coil -> electric -> fan.
   Everything else matches AHU_02 (sequence, fans, dampers, frost, dehumidify).
================================================================================
*)

PROGRAM AHU_03
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

    (* Fans *)
    C_FAN_MIN_PCT        : REAL := 20.0;
    C_FAN_MAX_PCT        : REAL := 100.0;
    C_FAN_SP_PCT         : REAL := 60.0;
    C_FAN_RAMP_PCT_S     : REAL := 10.0;
    C_EXH_OFFSET_PCT     : REAL := 0.0;
    C_FAN_PROOF_S        : TIME := T#15S;

    (* Dampers (MODULATING, fail CLOSE) *)
    C_DAMPER_OPEN_S      : TIME := T#120S;

    (* Heat-recovery wheel (HX_ROTARY) *)
    C_HR_MIN_PCT         : REAL := 0.0;
    C_HR_MAX_PCT         : REAL := 100.0;
    C_HR_FIXED_PCT       : REAL := 50.0;
    C_HR_DT_MIN_K        : REAL := 1.0;         (* (default) min dT to use wheel *)

    (* Water heating coil + frost (HEATING_COIL_3WV) *)
    C_FROST_TRIP_C       : REAL := 7.0;
    C_FROST_RELEASE_C    : REAL := 10.0;
    C_PUMP_RUNON_S       : TIME := T#120S;

    (* Electric after-heater (Coil-Electric) *)
    C_EL_MIN_PCT         : REAL := 0.0;
    C_EL_MAX_PCT         : REAL := 100.0;
    C_EL_COOLDOWN_S      : TIME := T#60S;        (* (default) fan overrun after electric heat *)

    (* Supply-air low-limit protection *)
    C_SUPPLY_LOWLIMIT_C  : REAL := 5.0;         (* (default) *)

    (* Return-water supervision (default range) *)
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

    (* Dehumidification (default; no RH setpoint in JSON) *)
    C_ROOM_RH_SP_PCT     : REAL := 50.0;
    C_RH_DEADBAND_PCT    : REAL := 5.0;
    C_DEHUM_COOL_MIN_PCT : REAL := 60.0;

    (* Heating-stage split points (% of heating authority) - (default) *)
    C_ST1_WHEEL_END      : REAL := 40.0;        (* with recovery: wheel  0..40   *)
    C_ST2_WATER_END      : REAL := 75.0;        (* with recovery: water 40..75   *)
    (* electric: 75..100 ; without recovery: water 0..60, electric 60..100      *)
    C_NR_WATER_END       : REAL := 60.0;
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
    DI_SupplyLowLimit    : BOOL;        (* supply-air low-limit frost stat *)
    DI_ElHeaterOverheat  : BOOL;        (* electric coil overheat thermostat *)
    DI_Reset             : BOOL;
    DI_RemoteEnable      : BOOL := TRUE;

    (* ---- Analog inputs ---- *)
    AI_T_Inlet_C         : REAL;        (* outdoor / inlet air *)
    AI_T_Supply_C        : REAL;
    AI_RH_Supply_pct     : REAL;
    AI_T_Exhaust_C       : REAL;        (* extract air *)
    AI_RH_Exhaust_pct    : REAL;
    AI_T_Room_C          : REAL;
    AI_RH_Room_pct       : REAL;
    AI_T_WaterReturn_C   : REAL;

    (* ---- Digital outputs ---- *)
    DO_FanSupply_Run     : BOOL;
    DO_FanExhaust_Run    : BOOL;
    DO_HeatPump_Run      : BOOL;
    DO_ElHeater_Enable   : BOOL;        (* electric coil safety contactor *)
    DO_CommonAlarm       : BOOL;

    (* ---- Analog outputs (0..100 %) ---- *)
    AO_FanSupply_Pct     : REAL;
    AO_FanExhaust_Pct    : REAL;
    AO_DamperInlet_Pct   : REAL;
    AO_DamperExhaust_Pct : REAL;
    AO_DamperRecirc_Pct  : REAL;
    AO_HeatRecovery_Pct  : REAL;        (* rotary wheel *)
    AO_HeatValve_Pct     : REAL;        (* water heating valve *)
    AO_ElHeater_Pct      : REAL;        (* electric after-heater 0..100 (0-10V) *)
    AO_CoolValve_Pct     : REAL;

    (* =========================== STATE =========================== *)
    eState               : E_AHU_STATE := AHU_OFF;
    eState_prev          : E_AHU_STATE := AHU_OFF;
    bEnableCmd           : BOOL;

    bFrostLatched        : BOOL;
    bFanFaultLatched     : BOOL;
    bSensorFaultLatched  : BOOL;
    bElOverheatLatched   : BOOL;        (* electric overheat latched *)
    bLowLimitActive      : BOOL;

    bFilterWarn          : BOOL;
    bRWaterWarn          : BOOL;
    bRWaterAlarm         : BOOL;

    rSupplySetpoint_C    : REAL := 18.0;
    rRoom_I              : REAL;
    rSup_I               : REAL;
    rSeqOut_Pct          : REAL;
    rFanDemand_Pct       : REAL;

    bFrostActive         : BOOL;
    bDehumActive         : BOOL;
    bHeatDemand          : BOOL;        (* water-coil demand (drives pump) *)
    bElActive            : BOOL;        (* electric coil energised *)
    bElCooldownActive    : BOOL;
    bRecoveryHeatOK      : BOOL;
    bRecoveryCoolOK      : BOOL;

    rDt_s                : REAL := 0.2;

    (* helpers *)
    rErr                 : REAL;
    rRamp                : REAL;
    rAuth                : REAL;
    rReheat              : REAL;

    (* =========================== TIMERS =========================== *)
    tonDamperWait        : TON;
    tonFanWait           : TON;
    tonPurge             : TON;
    tonSupplyProof       : TON;
    tonExhaustProof      : TON;
    tonPumpRunOn         : TON;
    tonElCooldown        : TON;         (* time since electric coil last on *)
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
IF  (AI_T_Inlet_C       < C_TEMP_RANGE_LO_C) OR (AI_T_Inlet_C       > C_TEMP_RANGE_HI_C)
 OR (AI_T_Supply_C      < C_TEMP_RANGE_LO_C) OR (AI_T_Supply_C      > C_TEMP_RANGE_HI_C)
 OR (AI_T_Room_C        < C_TEMP_RANGE_LO_C) OR (AI_T_Room_C        > C_TEMP_RANGE_HI_C)
 OR (AI_T_WaterReturn_C < C_TEMP_RANGE_LO_C) OR (AI_T_WaterReturn_C > C_TEMP_RANGE_HI_C)
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
   3)  FROST + SUPPLY LOW-LIMIT
   ================================================================ *)
bLowLimitActive := DI_SupplyLowLimit
                   OR ((eState = AHU_RUN) AND (AI_T_Supply_C <= C_SUPPLY_LOWLIMIT_C));
IF (AI_T_WaterReturn_C <= C_FROST_TRIP_C) OR bLowLimitActive THEN
    bFrostActive  := TRUE;
    bFrostLatched := TRUE;
ELSIF (AI_T_WaterReturn_C >= C_FROST_RELEASE_C) AND NOT bLowLimitActive THEN
    bFrostActive := FALSE;
END_IF

(* ================================================================
   3b) ELECTRIC OVERHEAT LATCH
   ================================================================ *)
IF DI_ElHeaterOverheat THEN
    bElOverheatLatched := TRUE;
END_IF

(* ================================================================
   4)  ALARM RESET
   ================================================================ *)
IF DI_Reset THEN
    IF NOT bFrostActive THEN bFrostLatched := FALSE; END_IF
    IF NOT (DI_FanSupply_Fault OR DI_FanExhaust_Fault) THEN bFanFaultLatched := FALSE; END_IF
    IF NOT DI_ElHeaterOverheat THEN bElOverheatLatched := FALSE; END_IF
    bSensorFaultLatched := FALSE;
END_IF

(* ================================================================
   5)  SEQUENCE STATE MACHINE   (FIRE > FROST > FAULT > flow)
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
        (* hold fans until purge done AND electric cool-down elapsed *)
        tonPurge(IN := TRUE, PT := C_SHUTDOWN_PURGE_S);
        bElCooldownActive := NOT tonElCooldown.Q;
        IF tonPurge.Q AND NOT bElCooldownActive THEN
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

    bDehumActive := (AI_RH_Room_pct > (C_ROOM_RH_SP_PCT + C_RH_DEADBAND_PCT));
ELSE
    rRoom_I := 0.0;  rSup_I := 0.0;  rSeqOut_Pct := 0.0;
    rSupplySetpoint_C := C_SUPPLY_SP_C;  bDehumActive := FALSE;
END_IF

(* ================================================================
   7)  3-STAGE HEATING / COOLING SEQUENCING
       Heating capacity order (cheapest first):
         recovery wheel -> water coil -> electric after-heater
   ================================================================ *)
bRecoveryHeatOK := (AI_T_Exhaust_C - AI_T_Inlet_C) > C_HR_DT_MIN_K;
bRecoveryCoolOK := (AI_T_Inlet_C - AI_T_Exhaust_C) > C_HR_DT_MIN_K;

AO_HeatRecovery_Pct := 0.0;
AO_HeatValve_Pct    := 0.0;
AO_ElHeater_Pct     := 0.0;
AO_CoolValve_Pct    := 0.0;
bHeatDemand         := FALSE;

IF rSeqOut_Pct > C_SEQ_DEADBAND_PCT THEN
    (* ---- HEATING ---- *)
    rAuth := (rSeqOut_Pct - C_SEQ_DEADBAND_PCT) / (100.0 - C_SEQ_DEADBAND_PCT) * 100.0;
    IF bRecoveryHeatOK THEN
        (* wheel 0..40, water 40..75, electric 75..100 *)
        AO_HeatRecovery_Pct := rAuth / C_ST1_WHEEL_END * 100.0;
        IF AO_HeatRecovery_Pct > C_HR_MAX_PCT THEN AO_HeatRecovery_Pct := C_HR_MAX_PCT; END_IF
        AO_HeatValve_Pct := (rAuth - C_ST1_WHEEL_END) / (C_ST2_WATER_END - C_ST1_WHEEL_END) * 100.0;
        IF AO_HeatValve_Pct < 0.0   THEN AO_HeatValve_Pct := 0.0;   END_IF
        IF AO_HeatValve_Pct > 100.0 THEN AO_HeatValve_Pct := 100.0; END_IF
        AO_ElHeater_Pct := (rAuth - C_ST2_WATER_END) / (100.0 - C_ST2_WATER_END) * 100.0;
        IF AO_ElHeater_Pct < 0.0 THEN AO_ElHeater_Pct := 0.0; END_IF
    ELSE
        (* water 0..60, electric 60..100 *)
        AO_HeatRecovery_Pct := 0.0;
        AO_HeatValve_Pct := rAuth / C_NR_WATER_END * 100.0;
        IF AO_HeatValve_Pct > 100.0 THEN AO_HeatValve_Pct := 100.0; END_IF
        AO_ElHeater_Pct := (rAuth - C_NR_WATER_END) / (100.0 - C_NR_WATER_END) * 100.0;
        IF AO_ElHeater_Pct < 0.0 THEN AO_ElHeater_Pct := 0.0; END_IF
    END_IF
    bHeatDemand := (AO_HeatValve_Pct > 0.0);

ELSIF rSeqOut_Pct < -C_SEQ_DEADBAND_PCT THEN
    (* ---- COOLING ---- *)
    rAuth := (-rSeqOut_Pct - C_SEQ_DEADBAND_PCT) / (100.0 - C_SEQ_DEADBAND_PCT) * 100.0;
    IF bRecoveryCoolOK THEN
        AO_HeatRecovery_Pct := rAuth * 2.0;
        IF AO_HeatRecovery_Pct > C_HR_MAX_PCT THEN AO_HeatRecovery_Pct := C_HR_MAX_PCT; END_IF
        AO_CoolValve_Pct := (rAuth - 50.0) * 2.0;
        IF AO_CoolValve_Pct < 0.0 THEN AO_CoolValve_Pct := 0.0; END_IF
    ELSE
        AO_HeatRecovery_Pct := 0.0;
        AO_CoolValve_Pct    := rAuth;
    END_IF
END_IF

(* ---- Dehumidification: cool to condense, ELECTRIC after-heater reheats ---- *)
IF (eState = AHU_RUN) AND bDehumActive THEN
    IF AO_CoolValve_Pct < C_DEHUM_COOL_MIN_PCT THEN
        AO_CoolValve_Pct := C_DEHUM_COOL_MIN_PCT;
    END_IF
    rReheat := C_SUP_KP * (rSupplySetpoint_C - AI_T_Supply_C);
    IF rReheat < 0.0   THEN rReheat := 0.0;   END_IF
    IF rReheat > 100.0 THEN rReheat := 100.0; END_IF
    AO_ElHeater_Pct  := rReheat;       (* electric reheat = "El_afterheating" *)
    AO_HeatValve_Pct := 0.0;
    bHeatDemand      := FALSE;
END_IF

(* ================================================================
   8)  OUTPUT MAPPING PER STATE
   ================================================================ *)
CASE eState OF

    AHU_OFF:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0;  AO_DamperExhaust_Pct := 0.0;  AO_DamperRecirc_Pct := 0.0;
        AO_HeatRecovery_Pct  := 0.0;  AO_HeatValve_Pct := 0.0;  AO_ElHeater_Pct := 0.0;  AO_CoolValve_Pct := 0.0;

    AHU_SHUTDOWN:
        (* keep fans + dampers during electric cool-down, electric OFF *)
        AO_ElHeater_Pct      := 0.0;
        AO_HeatRecovery_Pct  := 0.0;  AO_HeatValve_Pct := 0.0;  AO_CoolValve_Pct := 0.0;
        IF bElCooldownActive THEN
            rFanDemand_Pct       := C_FAN_MIN_PCT;
            AO_DamperInlet_Pct   := 100.0; AO_DamperExhaust_Pct := 100.0; AO_DamperRecirc_Pct := 0.0;
        ELSE
            rFanDemand_Pct       := 0.0;
            AO_DamperInlet_Pct   := 0.0; AO_DamperExhaust_Pct := 0.0; AO_DamperRecirc_Pct := 0.0;
        END_IF

    AHU_STARTUP:
        AO_DamperInlet_Pct   := 100.0; AO_DamperExhaust_Pct := 100.0; AO_DamperRecirc_Pct := 0.0;
        IF tonFanWait.IN THEN rFanDemand_Pct := C_FAN_SP_PCT; ELSE rFanDemand_Pct := 0.0; END_IF
        AO_HeatRecovery_Pct  := 0.0;  AO_HeatValve_Pct := 0.0;  AO_ElHeater_Pct := 0.0;  AO_CoolValve_Pct := 0.0;

    AHU_RUN:
        AO_DamperInlet_Pct   := 100.0; AO_DamperExhaust_Pct := 100.0; AO_DamperRecirc_Pct := 0.0;
        rFanDemand_Pct       := C_FAN_SP_PCT;   (* recovery/coils/electric set above *)

    AHU_FROST:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0; AO_DamperExhaust_Pct := 0.0; AO_DamperRecirc_Pct := 0.0;
        AO_HeatRecovery_Pct  := 0.0;  AO_HeatValve_Pct := 100.0;  AO_ElHeater_Pct := 0.0;  AO_CoolValve_Pct := 0.0;
        bHeatDemand          := TRUE;

    AHU_FIRE:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0; AO_DamperExhaust_Pct := 0.0; AO_DamperRecirc_Pct := 0.0;
        AO_HeatRecovery_Pct  := 0.0;  AO_HeatValve_Pct := 0.0;  AO_ElHeater_Pct := 0.0;  AO_CoolValve_Pct := 0.0;

    AHU_FAULT:
        rFanDemand_Pct       := 0.0;
        AO_DamperInlet_Pct   := 0.0; AO_DamperExhaust_Pct := 0.0; AO_DamperRecirc_Pct := 0.0;
        AO_HeatRecovery_Pct  := 0.0;  AO_HeatValve_Pct := 0.0;  AO_ElHeater_Pct := 0.0;  AO_CoolValve_Pct := 0.0;

END_CASE

(* ================================================================
   9)  ELECTRIC HEATER SAFETY INTERLOCKS
       - only in RUN, only with proven supply airflow
       - blocked by overheat latch
   ================================================================ *)
IF bElOverheatLatched OR (eState <> AHU_RUN) OR NOT DI_FanSupply_Run THEN
    AO_ElHeater_Pct := 0.0;
END_IF
IF AO_ElHeater_Pct > C_EL_MAX_PCT THEN AO_ElHeater_Pct := C_EL_MAX_PCT; END_IF
bElActive          := (AO_ElHeater_Pct > 0.0);
DO_ElHeater_Enable := bElActive;

(* cool-down timer: counts time since the electric coil last switched off *)
tonElCooldown(IN := NOT bElActive, PT := C_EL_COOLDOWN_S);

(* ================================================================
   10) FAN SPEED RAMP + EXHAUST TRACKING
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
   11) HEATING CIRCULATION PUMP (water coil demand + run-on + frost)
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
   12) COMMON ALARM
   ================================================================ *)
DO_CommonAlarm := bFrostLatched OR bFanFaultLatched OR bSensorFaultLatched
                  OR bElOverheatLatched OR DI_FireAlarm OR bRWaterAlarm;

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
