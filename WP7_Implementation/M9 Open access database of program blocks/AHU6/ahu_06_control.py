#!/usr/bin/env python3
"""
================================================================================
 AHU_06 - Air Handling Unit Control (Python soft-PLC)
          group "Fresh_air_unit_El_Heater"
================================================================================
 Functional alternative to the licensed CoDeSys 3.5 ST program (AHU_06.st),
 mirroring the same logic on a 200 ms cycle.

 Source : 10ahus_selection_AHU6.json  (project "10AHUs")

 Unit type: supply-only fresh-air HEATING unit with an ELECTRIC heater.
   One supply VFD fan, inlet + exhaust(relief) dampers, an electric heating coil
   (the only conditioning device), supply T/RH and room sensors, and a supply-air
   HIGH-limit thermostat. Heating only - no cooling, no recovery, no water coil.

 Electric-coil safety: airflow interlock, supply high-limit (latched), and a fan
 cool-down overrun on shutdown.
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

    # Supply fan
    FAN_MIN_PCT = 20.0
    FAN_MAX_PCT = 100.0
    FAN_SP_PCT = 60.0          # fixed supply rate
    FAN_RAMP_PCT_S = 10.0
    FAN_PROOF_S = 15.0

    # Dampers
    DAMPER_OPEN_S = 120.0

    # Electric heater
    EL_MIN_PCT = 0.0
    EL_MAX_PCT = 100.0
    EL_COOLDOWN_S = 60.0       # (default) fan overrun after electric heat
    SUPPLY_HILIMIT_C = 40.0    # (default) soft supply high-limit cutout

    # Sensor plausibility
    TEMP_RANGE_LO_C = -20.0
    TEMP_RANGE_HI_C = 80.0

    # Cascade PI (heating only)
    ROOM_KP = 1.5
    ROOM_TI_S = 600.0
    SUP_KP = 12.0
    SUP_TI_S = 180.0


class State(IntEnum):
    OFF = 0
    STARTUP = 1
    RUN = 2
    SHUTDOWN = 3
    FROST = 4         # unused on this unit
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
    DI_FilterSupply_Dirty: bool = False
    DI_SupplyHiLimit: bool = False     # supply-air high-limit thermostat
    DI_Reset: bool = False
    DI_RemoteEnable: bool = True

    # ---- Analog inputs ----
    AI_T_Supply_C: float = 15.0
    AI_RH_Supply_pct: float = 40.0
    AI_T_Room_C: float = 19.0

    # ---- Digital outputs ----
    DO_FanSupply_Run: bool = False
    DO_ElHeater_Enable: bool = False
    DO_CommonAlarm: bool = False

    # ---- Analog outputs (0..100 %) ----
    AO_FanSupply_Pct: float = 0.0
    AO_DamperInlet_Pct: float = 0.0
    AO_DamperExhaust_Pct: float = 0.0
    AO_ElHeater_Pct: float = 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class AHUController:
    def __init__(self) -> None:
        self.state = State.OFF
        self.state_prev = State.OFF

        self.fan_fault_latched = False
        self.sensor_fault_latched = False
        self.el_hilimit_latched = False
        self.hilimit_active = False
        self.filter_warn = False

        self.supply_setpoint_c = P.SUPPLY_SP_C
        self.room_i = 0.0
        self.sup_i = 0.0
        self.heat_demand_pct = 0.0
        self.fan_demand_pct = 0.0

        self.el_active = False
        self.el_cooldown_active = False

        self.ton_damper = TON()
        self.ton_fan = TON()
        self.ton_purge = TON()
        self.ton_supply_proof = TON()
        self.ton_el_cooldown = TON()

    # ----------------------------------------------------------------
    def scan(self, io: IOImage, dt_s: float = P.DT_S) -> None:
        self.state_prev = self.state
        enable_cmd = (io.DI_RemoteEnable
                      and not self.fan_fault_latched
                      and not self.sensor_fault_latched)

        # 1) sensor supervision
        for v in (io.AI_T_Supply_C, io.AI_T_Room_C):
            if v < P.TEMP_RANGE_LO_C or v > P.TEMP_RANGE_HI_C:
                self.sensor_fault_latched = True
        self.filter_warn = io.DI_FilterSupply_Dirty

        # 2) fan fault
        proof = self.ton_supply_proof(
            io.DO_FanSupply_Run and not io.DI_FanSupply_Run, P.FAN_PROOF_S, dt_s)
        if io.DI_FanSupply_Fault or proof:
            self.fan_fault_latched = True

        # 3) electric supply high-limit (overheat)
        self.hilimit_active = io.DI_SupplyHiLimit or (io.AI_T_Supply_C >= P.SUPPLY_HILIMIT_C)
        if self.hilimit_active:
            self.el_hilimit_latched = True

        # 4) reset
        if io.DI_Reset:
            if not io.DI_FanSupply_Fault:
                self.fan_fault_latched = False
            if not self.hilimit_active:
                self.el_hilimit_latched = False
            self.sensor_fault_latched = False

        # 5) state machine (FIRE > FAULT > flow; no FROST)
        if io.DI_FireAlarm:
            self.state = State.FIRE
        elif (self.fan_fault_latched or self.sensor_fault_latched) \
                and self.state != State.OFF:
            self.state = State.FAULT

        s = self.state
        if s == State.OFF:
            if enable_cmd and not io.DI_FireAlarm:
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
            self.ton_purge(True, P.SHUTDOWN_PURGE_S, dt_s)
            self.el_cooldown_active = not self.ton_el_cooldown.Q
            if self.ton_purge.Q and not self.el_cooldown_active:
                self.ton_purge(False, P.SHUTDOWN_PURGE_S, dt_s)
                self.state = State.OFF
        elif s == State.FROST:
            self.state = State.OFF        # not used
        elif s == State.FIRE:
            if not io.DI_FireAlarm:
                self.state = State.OFF
        elif s == State.FAULT:
            if not (self.fan_fault_latched or self.sensor_fault_latched):
                self.state = State.OFF

        # 6) cascade heating control (heating only)
        if self.state == State.RUN:
            err = P.ROOM_SP_C - io.AI_T_Room_C
            self.room_i = _clamp(self.room_i + (P.ROOM_KP / P.ROOM_TI_S) * err * dt_s,
                                 P.MIN_SUPPLY_C - P.SUPPLY_SP_C,
                                 P.MAX_SUPPLY_C - P.SUPPLY_SP_C)
            self.supply_setpoint_c = _clamp(
                P.SUPPLY_SP_C + P.ROOM_KP * err + self.room_i,
                P.MIN_SUPPLY_C, P.MAX_SUPPLY_C)
            err = self.supply_setpoint_c - io.AI_T_Supply_C
            self.sup_i = _clamp(self.sup_i + (P.SUP_KP / P.SUP_TI_S) * err * dt_s, 0.0, 100.0)
            self.heat_demand_pct = _clamp(P.SUP_KP * err + self.sup_i, 0.0, 100.0)
        else:
            self.room_i = self.sup_i = self.heat_demand_pct = 0.0
            self.supply_setpoint_c = P.SUPPLY_SP_C

        # 7) output mapping per state
        s = self.state
        if s == State.OFF:
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = 0.0
            io.AO_ElHeater_Pct = 0.0
        elif s == State.SHUTDOWN:
            io.AO_ElHeater_Pct = 0.0
            if self.el_cooldown_active:
                self.fan_demand_pct = P.FAN_MIN_PCT
                io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = 100.0
            else:
                self.fan_demand_pct = 0.0
                io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = 0.0
        elif s == State.STARTUP:
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = 100.0
            self.fan_demand_pct = P.FAN_SP_PCT if self.ton_fan.IN else 0.0
            io.AO_ElHeater_Pct = 0.0
        elif s == State.RUN:
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = 100.0
            self.fan_demand_pct = P.FAN_SP_PCT
            io.AO_ElHeater_Pct = self.heat_demand_pct
        elif s in (State.FIRE, State.FAULT, State.FROST):
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = io.AO_DamperExhaust_Pct = 0.0
            io.AO_ElHeater_Pct = 0.0

        # 8) electric heater safety interlocks
        if self.el_hilimit_latched or self.state != State.RUN or not io.DI_FanSupply_Run:
            io.AO_ElHeater_Pct = 0.0
        io.AO_ElHeater_Pct = min(io.AO_ElHeater_Pct, P.EL_MAX_PCT)
        self.el_active = io.AO_ElHeater_Pct > 0.0
        io.DO_ElHeater_Enable = self.el_active
        self.ton_el_cooldown(not self.el_active, P.EL_COOLDOWN_S, dt_s)

        # 9) fan ramp
        ramp = P.FAN_RAMP_PCT_S * dt_s
        if io.AO_FanSupply_Pct < self.fan_demand_pct:
            io.AO_FanSupply_Pct = min(io.AO_FanSupply_Pct + ramp, self.fan_demand_pct)
        elif io.AO_FanSupply_Pct > self.fan_demand_pct:
            io.AO_FanSupply_Pct = max(io.AO_FanSupply_Pct - ramp, self.fan_demand_pct)
        if self.fan_demand_pct > 0.0:
            if 0.0 < io.AO_FanSupply_Pct < P.FAN_MIN_PCT:
                io.AO_FanSupply_Pct = P.FAN_MIN_PCT
            io.AO_FanSupply_Pct = min(io.AO_FanSupply_Pct, P.FAN_MAX_PCT)
        io.DO_FanSupply_Run = io.AO_FanSupply_Pct > 0.0

        # 10) common alarm
        io.DO_CommonAlarm = (self.fan_fault_latched or self.sensor_fault_latched
                             or self.el_hilimit_latched or io.DI_FireAlarm)


# ============================================================================
# Plant simulator (replace with real I/O driver for field use)
# ============================================================================
@dataclass
class PlantSimulator:
    t_outdoor: float = 2.0            # outdoor / inlet air
    t_supply: float = 10.0
    t_room: float = 17.0
    fault_inject: bool = False

    def step(self, io: IOImage, dt_s: float) -> None:
        flow = io.AO_FanSupply_Pct / 100.0
        elec = io.AO_ElHeater_Pct / 100.0
        # supply air = outdoor air warmed by the electric coil
        target_supply = self.t_outdoor + elec * 30.0
        if flow > 0.01:
            self.t_supply += (target_supply - self.t_supply) * min(1.0, 0.6 * dt_s)
            self.t_room += (self.t_supply - self.t_room) * 0.10 * flow * dt_s
        else:
            self.t_supply += (self.t_room - self.t_supply) * 0.05 * dt_s
            self.t_room += (self.t_outdoor - self.t_room) * 0.005 * dt_s

        io.AI_T_Supply_C = self.t_supply
        io.AI_T_Room_C = self.t_room
        io.DI_FanSupply_Run = io.DO_FanSupply_Run and not self.fault_inject


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
                  f"SPsup={ctrl.supply_setpoint_c:4.1f} "
                  f"Fan={io.AO_FanSupply_Pct:5.1f}% Elec={io.AO_ElHeater_Pct:5.1f}% "
                  f"ElEn={int(io.DO_ElHeater_Enable)} Alarm={int(io.DO_CommonAlarm)}")

        n += 1
        if realtime:
            next_t += P.DT_S
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    print("AHU_06 (Fresh_air_unit_El_Heater) soft-PLC starting "
          "(200 ms cycle, Ctrl+C to stop)\n")
    try:
        run(cycles=None, realtime=True)
    except KeyboardInterrupt:
        print("\nStopped.")
