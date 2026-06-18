#!/usr/bin/env python3
"""
================================================================================
 AHU_05 - Air Handling Unit Control (Python soft-PLC)   group "FreshAir Unit1"
================================================================================
 Functional alternative to the licensed CoDeSys 3.5 ST program (AHU_05.st),
 mirroring the same logic on a 200 ms cycle.

 Source : 10ahus_selection_AHU5.json  (project "10AHUs")

 Unit type: 100 % fresh-air supply/exhaust unit with PLATE energy recovery and a
 modulating recovery BYPASS damper ("Klap_Bypass"), water heating coil with frost
 protection, cooling coil. Same plant as AHU_01, but the ERV bypass is actively
 modulated as the first (free-energy) stage of heating and cooling.

 Bypass convention:
   AO_ERVBypass_Pct = 0   -> air fully through recovery core (MAX recovery)
                    = 100 -> air fully bypasses the core      (NO recovery)
   = 100 - recovery_fraction.
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum


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

    # ERV plate + bypass
    HR_DT_MIN_K = 1.0          # (default) min extract<->outdoor dT to use recovery

    # Water heating coil + frost
    FROST_TRIP_C = 7.0
    FROST_RELEASE_C = 10.0
    PUMP_RUNON_S = 120.0

    # Return-water supervision
    RW_ALARM_LOW_C = 5.0
    RW_ALARM_HIGH_C = 60.0
    RW_WARN_LOW_C = 10.0
    RW_WARN_HIGH_C = 55.0

    # Sensor plausibility
    TEMP_RANGE_LO_C = -20.0
    TEMP_RANGE_HI_C = 80.0

    # Cascade PI
    ROOM_KP = 1.5
    ROOM_TI_S = 600.0
    SUP_KP = 12.0
    SUP_TI_S = 180.0
    SEQ_DEADBAND_PCT = 5.0


class State(IntEnum):
    OFF = 0
    STARTUP = 1
    RUN = 2
    SHUTDOWN = 3
    FROST = 4
    FIRE = 5
    FAULT = 6


class TON:
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
    DI_Reset: bool = False
    DI_RemoteEnable: bool = True

    # ---- Analog inputs ----
    AI_T_Outdoor_C: float = 5.0
    AI_T_Supply_C: float = 18.0
    AI_RH_Supply_pct: float = 40.0
    AI_T_Exhaust_C: float = 22.0
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
    AO_ERVBypass_Pct: float = 100.0      # default: no recovery (full bypass)
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

        self.filter_warn = False
        self.rwater_warn = False
        self.rwater_alarm = False

        self.supply_setpoint_c = P.SUPPLY_SP_C
        self.room_i = 0.0
        self.sup_i = 0.0
        self.seq_out_pct = 0.0
        self.fan_demand_pct = 0.0
        self.recovery_pct = 0.0

        self.heat_demand = False
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

        # 1) sensor supervision
        for v in (io.AI_T_Outdoor_C, io.AI_T_Supply_C,
                  io.AI_T_Room_C, io.AI_T_WaterReturn_C):
            if v < P.TEMP_RANGE_LO_C or v > P.TEMP_RANGE_HI_C:
                self.sensor_fault_latched = True
        self.rwater_warn = (io.AI_T_WaterReturn_C < P.RW_WARN_LOW_C
                            or io.AI_T_WaterReturn_C > P.RW_WARN_HIGH_C)
        self.rwater_alarm = (io.AI_T_WaterReturn_C < P.RW_ALARM_LOW_C
                             or io.AI_T_WaterReturn_C > P.RW_ALARM_HIGH_C)
        self.filter_warn = io.DI_FilterSupply_Dirty or io.DI_FilterExhaust_Dirty

        # 2) fan fault
        sp = self.ton_supply_proof(
            io.DO_FanSupply_Run and not io.DI_FanSupply_Run, P.FAN_PROOF_S, dt_s)
        ex = self.ton_exhaust_proof(
            io.DO_FanExhaust_Run and not io.DI_FanExhaust_Run, P.FAN_PROOF_S, dt_s)
        if io.DI_FanSupply_Fault or io.DI_FanExhaust_Fault or sp or ex:
            self.fan_fault_latched = True

        # 3) frost protection (water coil)
        if io.AI_T_WaterReturn_C <= P.FROST_TRIP_C:
            self.frost_active = True
            self.frost_latched = True
        elif io.AI_T_WaterReturn_C >= P.FROST_RELEASE_C:
            self.frost_active = False

        # 4) reset
        if io.DI_Reset:
            if not self.frost_active:
                self.frost_latched = False
            if not (io.DI_FanSupply_Fault or io.DI_FanExhaust_Fault):
                self.fan_fault_latched = False
            self.sensor_fault_latched = False

        # 5) state machine
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

        # 6) cascade temperature control
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
        else:
            self.room_i = self.sup_i = self.seq_out_pct = 0.0
            self.supply_setpoint_c = P.SUPPLY_SP_C

        # 7) ERV recovery + coil sequencing
        self.recovery_heat_ok = (io.AI_T_Exhaust_C - io.AI_T_Outdoor_C) > P.HR_DT_MIN_K
        self.recovery_cool_ok = (io.AI_T_Outdoor_C - io.AI_T_Exhaust_C) > P.HR_DT_MIN_K

        self.recovery_pct = 0.0
        io.AO_HeatValve_Pct = 0.0
        io.AO_CoolValve_Pct = 0.0
        self.heat_demand = False
        db = P.SEQ_DEADBAND_PCT

        if self.seq_out_pct > db:                       # heating
            auth = (self.seq_out_pct - db) / (100.0 - db) * 100.0
            if self.recovery_heat_ok:
                self.recovery_pct = min(auth * 2.0, 100.0)
                io.AO_HeatValve_Pct = max((auth - 50.0) * 2.0, 0.0)
            else:
                io.AO_HeatValve_Pct = auth
            self.heat_demand = io.AO_HeatValve_Pct > 0.0
        elif self.seq_out_pct < -db:                    # cooling
            auth = (-self.seq_out_pct - db) / (100.0 - db) * 100.0
            if self.recovery_cool_ok:
                self.recovery_pct = min(auth * 2.0, 100.0)
                io.AO_CoolValve_Pct = max((auth - 50.0) * 2.0, 0.0)
            else:
                io.AO_CoolValve_Pct = auth

        # 8) output mapping per state
        s = self.state
        if s in (State.OFF, State.SHUTDOWN):
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = 0.0
            self.recovery_pct = 0.0
            io.AO_HeatValve_Pct = io.AO_CoolValve_Pct = 0.0
        elif s == State.STARTUP:
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = 100.0
            self.fan_demand_pct = P.FAN_SP_PCT if self.ton_fan.IN else 0.0
            self.recovery_pct = 0.0
            io.AO_HeatValve_Pct = io.AO_CoolValve_Pct = 0.0
        elif s == State.RUN:
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = 100.0
            self.fan_demand_pct = P.FAN_SP_PCT
        elif s == State.FROST:
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = 0.0
            self.recovery_pct = 0.0
            io.AO_HeatValve_Pct = 100.0
            io.AO_CoolValve_Pct = 0.0
            self.heat_demand = True
        elif s in (State.FIRE, State.FAULT):
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = 0.0
            self.recovery_pct = 0.0
            io.AO_HeatValve_Pct = io.AO_CoolValve_Pct = 0.0

        # bypass actuator = 100 - recovery
        io.AO_ERVBypass_Pct = _clamp(100.0 - self.recovery_pct, 0.0, 100.0)

        # 9) fan ramp + exhaust tracking
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

        # 10) heating pump
        runon_done = self.ton_pump_runon(not self.heat_demand, P.PUMP_RUNON_S, dt_s)
        if self.heat_demand or self.frost_active or self.frost_latched:
            io.DO_HeatPump_Run = True
        elif not runon_done:
            io.DO_HeatPump_Run = True
        else:
            io.DO_HeatPump_Run = False

        # 11) common alarm
        io.DO_CommonAlarm = (self.frost_latched or self.fan_fault_latched
                             or self.sensor_fault_latched or io.DI_FireAlarm
                             or self.rwater_alarm)


# ============================================================================
# Plant simulator (replace with real I/O driver for field use)
# ============================================================================
@dataclass
class PlantSimulator:
    t_outdoor: float = 2.0
    t_exhaust: float = 22.0
    t_supply: float = 15.0
    t_room: float = 20.0
    t_water_return: float = 35.0
    fault_inject: bool = False

    def step(self, io: IOImage, dt_s: float) -> None:
        flow = io.AO_FanSupply_Pct / 100.0
        recovery = (100.0 - io.AO_ERVBypass_Pct) / 100.0   # 0..1 recovery fraction
        heat = io.AO_HeatValve_Pct / 100.0
        cool = io.AO_CoolValve_Pct / 100.0

        t_after_erv = self.t_outdoor + recovery * 0.7 * (self.t_exhaust - self.t_outdoor)
        target_supply = t_after_erv + heat * 25.0 - cool * 18.0
        if flow > 0.01:
            self.t_supply += (target_supply - self.t_supply) * min(1.0, 0.6 * dt_s)
        else:
            self.t_supply += (self.t_room - self.t_supply) * 0.05 * dt_s

        if flow > 0.01:
            self.t_room += (self.t_supply - self.t_room) * 0.10 * flow * dt_s
        else:
            self.t_room += (self.t_outdoor - self.t_room) * 0.005 * dt_s
        self.t_exhaust = self.t_room

        if io.DO_HeatPump_Run:
            self.t_water_return += (20.0 + heat * 45.0 - self.t_water_return) * 0.15 * dt_s
        else:
            self.t_water_return += (self.t_outdoor - self.t_water_return) * 0.02 * dt_s

        io.AI_T_Outdoor_C = self.t_outdoor
        io.AI_T_Supply_C = self.t_supply
        io.AI_T_Exhaust_C = self.t_exhaust
        io.AI_T_Room_C = self.t_room
        io.AI_T_WaterReturn_C = self.t_water_return
        io.DI_FanSupply_Run = io.DO_FanSupply_Run and not self.fault_inject
        io.DI_FanExhaust_Run = io.DO_FanExhaust_Run and not self.fault_inject


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
                  f"Tout={io.AI_T_Outdoor_C:5.1f} Troom={io.AI_T_Room_C:5.1f} "
                  f"Tsup={io.AI_T_Supply_C:5.1f} Fan={io.AO_FanSupply_Pct:5.1f}% "
                  f"Recov={100 - io.AO_ERVBypass_Pct:5.1f}% (byp {io.AO_ERVBypass_Pct:3.0f}%) "
                  f"Heat={io.AO_HeatValve_Pct:5.1f}% Cool={io.AO_CoolValve_Pct:5.1f}% "
                  f"Alarm={int(io.DO_CommonAlarm)}")

        n += 1
        if realtime:
            next_t += P.DT_S
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    print("AHU_05 (FreshAir Unit1) soft-PLC starting (200 ms cycle, Ctrl+C to stop)\n")
    try:
        run(cycles=None, realtime=True)
    except KeyboardInterrupt:
        print("\nStopped.")
