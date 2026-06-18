#!/usr/bin/env python3
"""
================================================================================
 AHU_02 - Air Handling Unit Control (Python soft-PLC)   group "DeHumi_V1"
================================================================================
 Functional alternative to the licensed CoDeSys 3.5 ST program (AHU_02.st),
 mirroring the same logic on a 200 ms cycle.

 Source : 10ahus_selection_AHU2.json  (project "10AHUs")

 Differences vs AHU_01:
   + Rotary heat-recovery wheel (HX_ROTARY) staged before the coils.
   + Supply-air low-limit thermostat (hardware + soft trip).
   + Dehumidification function (room RH): cool to condense, reheat to setpoint.
   + Third (recirculation/bypass) damper.

 Architecture: IOImage / AHUController.scan() / PlantSimulator / run().
 For field use, replace PlantSimulator with a real I/O driver.
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum


# ============================================================================
# Engineering parameters (from the JSON selection; (default) = commissioning)
# ============================================================================
class P:
    CYCLE_MS = 200
    DT_S = CYCLE_MS / 1000.0

    # Sequence
    SUPPLY_SP_C = 18.0
    ROOM_SP_C = 22.0
    MIN_SUPPLY_C = 12.0
    MAX_SUPPLY_C = 30.0
    STARTUP_DAMPER_S = 20.0
    STARTUP_FAN_S = 10.0
    SHUTDOWN_PURGE_S = 0.0

    # Fans
    FAN_MIN_PCT = 20.0
    FAN_MAX_PCT = 100.0
    FAN_SP_PCT = 60.0
    FAN_RAMP_PCT_S = 10.0
    EXH_OFFSET_PCT = 0.0
    FAN_PROOF_S = 15.0

    # Dampers
    DAMPER_OPEN_S = 120.0

    # Heat-recovery wheel (HX_ROTARY)
    HR_MIN_PCT = 0.0
    HR_MAX_PCT = 100.0
    HR_FIXED_PCT = 50.0
    HR_DT_MIN_K = 1.0          # (default) min extract<->outdoor dT to use wheel

    # Heating coil + frost
    FROST_TRIP_C = 7.0
    FROST_RELEASE_C = 10.0
    PUMP_RUNON_S = 120.0

    # Supply-air low-limit protection
    SUPPLY_LOWLIMIT_C = 5.0    # (default)

    # Return-water supervision (default range)
    RW_ALARM_LOW_C = 5.0
    RW_ALARM_HIGH_C = 60.0
    RW_WARN_LOW_C = 10.0
    RW_WARN_HIGH_C = 55.0

    # Sensor plausibility
    TEMP_RANGE_LO_C = -20.0
    TEMP_RANGE_HI_C = 80.0

    # Cascade room PI / supply PI
    ROOM_KP = 1.5
    ROOM_TI_S = 600.0
    SUP_KP = 12.0
    SUP_TI_S = 180.0
    SEQ_DEADBAND_PCT = 5.0

    # Dehumidification (default; no RH setpoint in JSON)
    ROOM_RH_SP_PCT = 50.0
    RH_DEADBAND_PCT = 5.0
    DEHUM_COOL_MIN_PCT = 60.0


class State(IntEnum):
    OFF = 0
    STARTUP = 1
    RUN = 2
    SHUTDOWN = 3
    FROST = 4
    FIRE = 5
    FAULT = 6


class TON:
    """IEC on-delay timer, evaluated each scan."""
    def __init__(self) -> None:
        self._elapsed = 0.0
        self.Q = False
        self.IN = False

    def __call__(self, IN: bool, PT_s: float, dt_s: float) -> bool:
        self.IN = IN
        if IN:
            if self._elapsed < PT_s:
                self._elapsed += dt_s
            self.Q = self._elapsed >= PT_s
        else:
            self._elapsed = 0.0
            self.Q = False
        return self.Q


@dataclass
class IOImage:
    # ---- Digital inputs ----
    DI_FireAlarm: bool = False
    DI_FanSupply_Run: bool = False
    DI_FanSupply_Fault: bool = False
    DI_FanExhaust_Run: bool = False
    DI_FanExhaust_Fault: bool = False
    DI_FilterSupply_Dirty: bool = False
    DI_FilterExhaust_Dirty: bool = False
    DI_SupplyLowLimit: bool = False       # hardware low-limit frost stat
    DI_Reset: bool = False
    DI_RemoteEnable: bool = True

    # ---- Analog inputs (engineering units) ----
    AI_T_Inlet_C: float = 5.0             # outdoor / inlet air
    AI_T_Supply_C: float = 18.0
    AI_RH_Supply_pct: float = 40.0
    AI_T_Exhaust_C: float = 22.0          # extract air
    AI_RH_Exhaust_pct: float = 45.0
    AI_T_Room_C: float = 21.0
    AI_RH_Room_pct: float = 45.0
    AI_T_WaterReturn_C: float = 40.0

    # ---- Digital outputs ----
    DO_FanSupply_Run: bool = False
    DO_FanExhaust_Run: bool = False
    DO_HeatPump_Run: bool = False
    DO_CommonAlarm: bool = False

    # ---- Analog outputs (0..100 %) ----
    AO_FanSupply_Pct: float = 0.0
    AO_FanExhaust_Pct: float = 0.0
    AO_DamperInlet_Pct: float = 0.0
    AO_DamperExhaust_Pct: float = 0.0
    AO_DamperRecirc_Pct: float = 0.0
    AO_HeatRecovery_Pct: float = 0.0      # rotary wheel
    AO_HeatValve_Pct: float = 0.0
    AO_CoolValve_Pct: float = 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class AHUController:
    def __init__(self) -> None:
        self.state = State.OFF
        self.state_prev = State.OFF

        self.frost_latched = False
        self.fan_fault_latched = False
        self.sensor_fault_latched = False
        self.frost_active = False
        self.low_limit_active = False

        self.filter_warn = False
        self.rwater_warn = False
        self.rwater_alarm = False

        self.supply_setpoint_c = P.SUPPLY_SP_C
        self.room_i = 0.0
        self.sup_i = 0.0
        self.seq_out_pct = 0.0
        self.fan_demand_pct = 0.0

        self.heat_demand = False
        self.dehum_active = False
        self.recovery_heat_ok = False
        self.recovery_cool_ok = False

        self.ton_damper = TON()
        self.ton_fan = TON()
        self.ton_purge = TON()
        self.ton_supply_proof = TON()
        self.ton_exhaust_proof = TON()
        self.ton_pump_runon = TON()

    # ----------------------------------------------------------------
    def scan(self, io: IOImage, dt_s: float = P.DT_S) -> None:
        self.state_prev = self.state
        enable_cmd = (io.DI_RemoteEnable
                      and not self.fan_fault_latched
                      and not self.sensor_fault_latched)

        # --- 1) Sensor supervision ----------------------------------
        for v in (io.AI_T_Inlet_C, io.AI_T_Supply_C,
                  io.AI_T_Room_C, io.AI_T_WaterReturn_C):
            if v < P.TEMP_RANGE_LO_C or v > P.TEMP_RANGE_HI_C:
                self.sensor_fault_latched = True
        self.rwater_warn = (io.AI_T_WaterReturn_C < P.RW_WARN_LOW_C
                            or io.AI_T_WaterReturn_C > P.RW_WARN_HIGH_C)
        self.rwater_alarm = (io.AI_T_WaterReturn_C < P.RW_ALARM_LOW_C
                             or io.AI_T_WaterReturn_C > P.RW_ALARM_HIGH_C)
        self.filter_warn = io.DI_FilterSupply_Dirty or io.DI_FilterExhaust_Dirty

        # --- 2) Fan fault supervision -------------------------------
        sp = self.ton_supply_proof(
            io.DO_FanSupply_Run and not io.DI_FanSupply_Run, P.FAN_PROOF_S, dt_s)
        ex = self.ton_exhaust_proof(
            io.DO_FanExhaust_Run and not io.DI_FanExhaust_Run, P.FAN_PROOF_S, dt_s)
        if io.DI_FanSupply_Fault or io.DI_FanExhaust_Fault or sp or ex:
            self.fan_fault_latched = True

        # --- 3) Frost + supply low-limit ----------------------------
        self.low_limit_active = (io.DI_SupplyLowLimit
                                 or (self.state == State.RUN
                                     and io.AI_T_Supply_C <= P.SUPPLY_LOWLIMIT_C))
        if io.AI_T_WaterReturn_C <= P.FROST_TRIP_C or self.low_limit_active:
            self.frost_active = True
            self.frost_latched = True
        elif io.AI_T_WaterReturn_C >= P.FROST_RELEASE_C and not self.low_limit_active:
            self.frost_active = False

        # --- 4) Alarm reset -----------------------------------------
        if io.DI_Reset:
            if not self.frost_active:
                self.frost_latched = False
            if not (io.DI_FanSupply_Fault or io.DI_FanExhaust_Fault):
                self.fan_fault_latched = False
            self.sensor_fault_latched = False

        # --- 5) State machine ---------------------------------------
        if io.DI_FireAlarm:
            self.state = State.FIRE
        elif self.frost_latched and self.state != State.OFF:
            self.state = State.FROST
        elif (self.fan_fault_latched or self.sensor_fault_latched) \
                and self.state != State.OFF:
            self.state = State.FAULT

        s = self.state
        if s == State.OFF:
            if enable_cmd and not io.DI_FireAlarm and not self.frost_latched:
                self.state = State.STARTUP
        elif s == State.STARTUP:
            self.ton_damper(True, P.STARTUP_DAMPER_S, dt_s)
            if self.ton_damper.Q:
                self.ton_fan(True, P.STARTUP_FAN_S, dt_s)
            if self.ton_fan.Q and io.DI_FanSupply_Run:
                self.state = State.RUN
            if not enable_cmd:
                self.state = State.SHUTDOWN
        elif s == State.RUN:
            self.ton_damper(False, P.STARTUP_DAMPER_S, dt_s)
            self.ton_fan(False, P.STARTUP_FAN_S, dt_s)
            if not enable_cmd:
                self.state = State.SHUTDOWN
        elif s == State.SHUTDOWN:
            if self.ton_purge(True, P.SHUTDOWN_PURGE_S, dt_s):
                self.ton_purge(False, P.SHUTDOWN_PURGE_S, dt_s)
                self.state = State.OFF
        elif s == State.FROST:
            if not self.frost_latched:
                self.state = State.OFF
        elif s == State.FIRE:
            if not io.DI_FireAlarm:
                self.state = State.OFF
        elif s == State.FAULT:
            if not (self.fan_fault_latched or self.sensor_fault_latched):
                self.state = State.OFF

        # --- 6) Cascade temperature control -------------------------
        if self.state == State.RUN:
            err = P.ROOM_SP_C - io.AI_T_Room_C
            self.room_i = _clamp(self.room_i + (P.ROOM_KP / P.ROOM_TI_S) * err * dt_s,
                                 P.MIN_SUPPLY_C - P.SUPPLY_SP_C,
                                 P.MAX_SUPPLY_C - P.SUPPLY_SP_C)
            self.supply_setpoint_c = _clamp(
                P.SUPPLY_SP_C + P.ROOM_KP * err + self.room_i,
                P.MIN_SUPPLY_C, P.MAX_SUPPLY_C)

            err = self.supply_setpoint_c - io.AI_T_Supply_C
            self.sup_i = _clamp(self.sup_i + (P.SUP_KP / P.SUP_TI_S) * err * dt_s,
                                -100.0, 100.0)
            self.seq_out_pct = _clamp(P.SUP_KP * err + self.sup_i, -100.0, 100.0)
            self.dehum_active = io.AI_RH_Room_pct > (P.ROOM_RH_SP_PCT + P.RH_DEADBAND_PCT)
        else:
            self.room_i = 0.0
            self.sup_i = 0.0
            self.seq_out_pct = 0.0
            self.supply_setpoint_c = P.SUPPLY_SP_C
            self.dehum_active = False

        # --- 7) Heat-recovery + coil sequencing ---------------------
        self.recovery_heat_ok = (io.AI_T_Exhaust_C - io.AI_T_Inlet_C) > P.HR_DT_MIN_K
        self.recovery_cool_ok = (io.AI_T_Inlet_C - io.AI_T_Exhaust_C) > P.HR_DT_MIN_K

        io.AO_HeatRecovery_Pct = 0.0
        io.AO_HeatValve_Pct = 0.0
        io.AO_CoolValve_Pct = 0.0
        self.heat_demand = False
        db = P.SEQ_DEADBAND_PCT

        if self.seq_out_pct > db:                       # heating
            auth = (self.seq_out_pct - db) / (100.0 - db) * 100.0
            if self.recovery_heat_ok:
                io.AO_HeatRecovery_Pct = min(auth * 2.0, P.HR_MAX_PCT)
                io.AO_HeatValve_Pct = max((auth - 50.0) * 2.0, 0.0)
            else:
                io.AO_HeatValve_Pct = auth
            self.heat_demand = io.AO_HeatValve_Pct > 0.0
        elif self.seq_out_pct < -db:                    # cooling
            auth = (-self.seq_out_pct - db) / (100.0 - db) * 100.0
            if self.recovery_cool_ok:
                io.AO_HeatRecovery_Pct = min(auth * 2.0, P.HR_MAX_PCT)
                io.AO_CoolValve_Pct = max((auth - 50.0) * 2.0, 0.0)
            else:
                io.AO_CoolValve_Pct = auth

        # dehumidification override: condense + reheat
        if self.state == State.RUN and self.dehum_active:
            io.AO_CoolValve_Pct = max(io.AO_CoolValve_Pct, P.DEHUM_COOL_MIN_PCT)
            reheat = _clamp(P.SUP_KP * (self.supply_setpoint_c - io.AI_T_Supply_C),
                            0.0, 100.0)
            io.AO_HeatValve_Pct = reheat
            self.heat_demand = io.AO_HeatValve_Pct > 0.0

        # --- 8) Output mapping per state ----------------------------
        s = self.state
        if s in (State.OFF, State.SHUTDOWN):
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = io.AO_DamperRecirc_Pct = 0.0
            io.AO_HeatRecovery_Pct = io.AO_HeatValve_Pct = io.AO_CoolValve_Pct = 0.0
        elif s == State.STARTUP:
            io.AO_DamperInlet_Pct = 100.0
            io.AO_DamperExhaust_Pct = 100.0
            io.AO_DamperRecirc_Pct = 0.0
            self.fan_demand_pct = P.FAN_SP_PCT if self.ton_fan.IN else 0.0
            io.AO_HeatRecovery_Pct = io.AO_HeatValve_Pct = io.AO_CoolValve_Pct = 0.0
        elif s == State.RUN:
            io.AO_DamperInlet_Pct = 100.0
            io.AO_DamperExhaust_Pct = 100.0
            io.AO_DamperRecirc_Pct = 0.0
            self.fan_demand_pct = P.FAN_SP_PCT
        elif s == State.FROST:
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = io.AO_DamperRecirc_Pct = 0.0
            io.AO_HeatRecovery_Pct = 0.0
            io.AO_HeatValve_Pct = 100.0
            io.AO_CoolValve_Pct = 0.0
            self.heat_demand = True
        elif s == State.FIRE:
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = io.AO_DamperRecirc_Pct = 0.0
            io.AO_HeatRecovery_Pct = io.AO_HeatValve_Pct = io.AO_CoolValve_Pct = 0.0
        elif s == State.FAULT:
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = io.AO_DamperRecirc_Pct = 0.0
            io.AO_HeatRecovery_Pct = io.AO_HeatValve_Pct = io.AO_CoolValve_Pct = 0.0

        # --- 9) Fan ramp + exhaust tracking -------------------------
        ramp = P.FAN_RAMP_PCT_S * dt_s
        if io.AO_FanSupply_Pct < self.fan_demand_pct:
            io.AO_FanSupply_Pct = min(io.AO_FanSupply_Pct + ramp, self.fan_demand_pct)
        elif io.AO_FanSupply_Pct > self.fan_demand_pct:
            io.AO_FanSupply_Pct = max(io.AO_FanSupply_Pct - ramp, self.fan_demand_pct)
        if self.fan_demand_pct > 0.0:
            if 0.0 < io.AO_FanSupply_Pct < P.FAN_MIN_PCT:
                io.AO_FanSupply_Pct = P.FAN_MIN_PCT
            io.AO_FanSupply_Pct = min(io.AO_FanSupply_Pct, P.FAN_MAX_PCT)
        io.AO_FanExhaust_Pct = _clamp(io.AO_FanSupply_Pct + P.EXH_OFFSET_PCT,
                                      0.0, P.FAN_MAX_PCT)
        io.DO_FanSupply_Run = io.AO_FanSupply_Pct > 0.0
        io.DO_FanExhaust_Run = io.AO_FanExhaust_Pct > 0.0

        # --- 10) Heating pump ---------------------------------------
        runon_done = self.ton_pump_runon(not self.heat_demand, P.PUMP_RUNON_S, dt_s)
        if self.heat_demand or self.frost_active or self.frost_latched:
            io.DO_HeatPump_Run = True
        elif not runon_done:
            io.DO_HeatPump_Run = True
        else:
            io.DO_HeatPump_Run = False

        # --- 11) Common alarm ---------------------------------------
        io.DO_CommonAlarm = (self.frost_latched or self.fan_fault_latched
                             or self.sensor_fault_latched or io.DI_FireAlarm
                             or self.rwater_alarm)


# ============================================================================
# Plant simulator (replace with real I/O driver for field use)
# ============================================================================
@dataclass
class PlantSimulator:
    t_inlet: float = 2.0            # outdoor / inlet air
    t_exhaust: float = 22.0         # extract air (~ room)
    t_supply: float = 15.0
    t_room: float = 20.0
    rh_room: float = 45.0
    t_water_return: float = 35.0
    fault_inject: bool = False

    def step(self, io: IOImage, dt_s: float) -> None:
        flow = io.AO_FanSupply_Pct / 100.0
        wheel = io.AO_HeatRecovery_Pct / 100.0
        heat = io.AO_HeatValve_Pct / 100.0
        cool = io.AO_CoolValve_Pct / 100.0

        # air entering coils after the recovery wheel (moves toward extract temp)
        t_after_wheel = self.t_inlet + wheel * 0.7 * (self.t_exhaust - self.t_inlet)
        target_supply = t_after_wheel + heat * 25.0 - cool * 18.0
        if flow > 0.01:
            self.t_supply += (target_supply - self.t_supply) * min(1.0, 0.6 * dt_s)
        else:
            self.t_supply += (self.t_room - self.t_supply) * 0.05 * dt_s

        # room temperature & humidity
        if flow > 0.01:
            self.t_room += (self.t_supply - self.t_room) * 0.10 * flow * dt_s
            # cooling coil removes moisture; otherwise humidity creeps up
            if cool > 0.3:
                self.rh_room += (35.0 - self.rh_room) * 0.05 * flow * dt_s
            else:
                self.rh_room += 0.20 * dt_s
        else:
            self.t_room += (self.t_inlet - self.t_room) * 0.005 * dt_s
        self.rh_room = _clamp(self.rh_room, 10.0, 95.0)
        self.t_exhaust = self.t_room          # extract ~ room

        # return water
        if io.DO_HeatPump_Run:
            self.t_water_return += (20.0 + heat * 45.0 - self.t_water_return) * 0.15 * dt_s
        else:
            self.t_water_return += (self.t_inlet - self.t_water_return) * 0.02 * dt_s

        # write back sensors
        io.AI_T_Inlet_C = self.t_inlet
        io.AI_T_Supply_C = self.t_supply
        io.AI_T_Exhaust_C = self.t_exhaust
        io.AI_T_Room_C = self.t_room
        io.AI_RH_Room_pct = self.rh_room
        io.AI_T_WaterReturn_C = self.t_water_return
        io.DI_FanSupply_Run = io.DO_FanSupply_Run and not self.fault_inject
        io.DI_FanExhaust_Run = io.DO_FanExhaust_Run and not self.fault_inject


# ============================================================================
# Real-time scheduler
# ============================================================================
def run(cycles: int | None = None, realtime: bool = True) -> None:
    io = IOImage()
    ctrl = AHUController()
    plant = PlantSimulator()

    n = 0
    next_t = time.perf_counter()
    while cycles is None or n < cycles:
        plant.step(io, P.DT_S)
        ctrl.scan(io, P.DT_S)

        if n % 25 == 0:
            print(f"t={n * P.DT_S:6.1f}s {ctrl.state.name:8s} "
                  f"Troom={io.AI_T_Room_C:5.1f} Tsup={io.AI_T_Supply_C:5.1f} "
                  f"RHr={io.AI_RH_Room_pct:4.0f}% "
                  f"Fan={io.AO_FanSupply_Pct:5.1f}% "
                  f"Wheel={io.AO_HeatRecovery_Pct:5.1f}% "
                  f"Heat={io.AO_HeatValve_Pct:5.1f}% Cool={io.AO_CoolValve_Pct:5.1f}% "
                  f"{'DEHUM ' if ctrl.dehum_active else ''}Alarm={int(io.DO_CommonAlarm)}")

        n += 1
        if realtime:
            next_t += P.DT_S
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    print("AHU_02 (DeHumi_V1) soft-PLC starting (200 ms cycle, Ctrl+C to stop)\n")
    try:
        run(cycles=None, realtime=True)
    except KeyboardInterrupt:
        print("\nStopped.")
