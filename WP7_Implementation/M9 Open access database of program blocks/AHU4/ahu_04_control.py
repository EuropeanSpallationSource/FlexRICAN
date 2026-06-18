#!/usr/bin/env python3
"""
================================================================================
 AHU_04 - Air Handling Unit Control (Python soft-PLC)
          group "Exhaust_Vent_Room_Temp"
================================================================================
 Functional alternative to the licensed CoDeSys 3.5 ST program (AHU_04.st),
 mirroring the same logic on a 200 ms cycle.

 Source : 10ahus_selection_AHU4.json  (project "10AHUs")

 Unit type: exhaust-only ventilation unit that controls ROOM TEMPERATURE by
 modulating the extract rate. One extract VFD fan, an inlet make-up damper and
 an exhaust damper, a single room-temperature sensor. No heating, cooling, heat
 recovery, humidity or water circuit, hence no frost protection.

 Control: a direct-acting PI on room temperature drives the extract fan speed
 around a base ventilation rate (warmer room -> more extraction).
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
    ROOM_SP_C = 22.0
    STARTUP_DAMPER_S = 20.0
    STARTUP_FAN_S = 10.0
    SHUTDOWN_PURGE_S = 0.0

    # Extract fan
    FAN_MIN_PCT = 20.0
    FAN_MAX_PCT = 100.0
    VENT_BASE_PCT = 60.0       # speedSpPct (base ventilation)
    FAN_RAMP_PCT_S = 10.0
    FAN_PROOF_S = 15.0

    # Dampers
    DAMPER_OPEN_S = 120.0

    # Sensor plausibility
    TEMP_RANGE_LO_C = -20.0
    TEMP_RANGE_HI_C = 80.0

    # Room-temperature ventilation PI (direct acting) - (default)
    ROOM_KP = 15.0             # % speed per K above setpoint
    ROOM_TI_S = 600.0


class State(IntEnum):
    OFF = 0
    STARTUP = 1
    RUN = 2
    SHUTDOWN = 3
    FROST = 4         # unused on this unit (no water coil)
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
    DI_FanExhaust_Run: bool = False
    DI_FanExhaust_Fault: bool = False
    DI_Reset: bool = False
    DI_RemoteEnable: bool = True

    # ---- Analog inputs ----
    AI_T_Room_C: float = 21.0

    # ---- Digital outputs ----
    DO_FanExhaust_Run: bool = False
    DO_CommonAlarm: bool = False

    # ---- Analog outputs (0..100 %) ----
    AO_FanExhaust_Pct: float = 0.0
    AO_DamperInlet_Pct: float = 0.0
    AO_DamperExhaust_Pct: float = 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class AHUController:
    def __init__(self) -> None:
        self.state = State.OFF
        self.state_prev = State.OFF
        self.fan_fault_latched = False
        self.sensor_fault_latched = False
        self.room_i = 0.0
        self.fan_demand_pct = 0.0

        self.ton_damper = TON()
        self.ton_fan = TON()
        self.ton_purge = TON()
        self.ton_exhaust_proof = TON()

    # ----------------------------------------------------------------
    def scan(self, io: IOImage, dt_s: float = P.DT_S) -> None:
        self.state_prev = self.state
        enable_cmd = (io.DI_RemoteEnable
                      and not self.fan_fault_latched
                      and not self.sensor_fault_latched)

        # 1) sensor supervision
        if io.AI_T_Room_C < P.TEMP_RANGE_LO_C or io.AI_T_Room_C > P.TEMP_RANGE_HI_C:
            self.sensor_fault_latched = True

        # 2) fan fault (hard fault or missing proof of flow)
        proof = self.ton_exhaust_proof(
            io.DO_FanExhaust_Run and not io.DI_FanExhaust_Run, P.FAN_PROOF_S, dt_s)
        if io.DI_FanExhaust_Fault or proof:
            self.fan_fault_latched = True

        # 3) reset
        if io.DI_Reset:
            if not io.DI_FanExhaust_Fault:
                self.fan_fault_latched = False
            self.sensor_fault_latched = False

        # 4) state machine (FIRE > FAULT > flow; no FROST on this unit)
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
            if self.ton_fan.Q and io.DI_FanExhaust_Run:
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
            self.state = State.OFF        # not used; fail safe
        elif s == State.FIRE:
            if not io.DI_FireAlarm:
                self.state = State.OFF
        elif s == State.FAULT:
            if not (self.fan_fault_latched or self.sensor_fault_latched):
                self.state = State.OFF

        # 5) room-temperature ventilation PI (direct acting)
        if self.state == State.RUN:
            err = io.AI_T_Room_C - P.ROOM_SP_C        # >0 when too warm
            self.room_i += (P.ROOM_KP / P.ROOM_TI_S) * err * dt_s
            self.room_i = _clamp(self.room_i,
                                 P.FAN_MIN_PCT - P.VENT_BASE_PCT,
                                 P.FAN_MAX_PCT - P.VENT_BASE_PCT)
            self.fan_demand_pct = _clamp(
                P.VENT_BASE_PCT + P.ROOM_KP * err + self.room_i,
                P.FAN_MIN_PCT, P.FAN_MAX_PCT)
        else:
            self.room_i = 0.0

        # 6) output mapping per state
        s = self.state
        if s in (State.OFF, State.SHUTDOWN, State.FROST):
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = 0.0
            io.AO_DamperExhaust_Pct = 0.0
        elif s == State.STARTUP:
            io.AO_DamperInlet_Pct = 100.0
            io.AO_DamperExhaust_Pct = 100.0
            self.fan_demand_pct = P.VENT_BASE_PCT if self.ton_fan.IN else 0.0
        elif s == State.RUN:
            io.AO_DamperInlet_Pct = 100.0
            io.AO_DamperExhaust_Pct = 100.0
            # fan_demand_pct from PI in step 5
        elif s in (State.FIRE, State.FAULT):
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = 0.0
            io.AO_DamperExhaust_Pct = 0.0

        # 7) fan ramp
        ramp = P.FAN_RAMP_PCT_S * dt_s
        if io.AO_FanExhaust_Pct < self.fan_demand_pct:
            io.AO_FanExhaust_Pct = min(io.AO_FanExhaust_Pct + ramp, self.fan_demand_pct)
        elif io.AO_FanExhaust_Pct > self.fan_demand_pct:
            io.AO_FanExhaust_Pct = max(io.AO_FanExhaust_Pct - ramp, self.fan_demand_pct)
        if self.fan_demand_pct > 0.0:
            if 0.0 < io.AO_FanExhaust_Pct < P.FAN_MIN_PCT:
                io.AO_FanExhaust_Pct = P.FAN_MIN_PCT
            io.AO_FanExhaust_Pct = min(io.AO_FanExhaust_Pct, P.FAN_MAX_PCT)
        io.DO_FanExhaust_Run = io.AO_FanExhaust_Pct > 0.0

        # 8) common alarm
        io.DO_CommonAlarm = (self.fan_fault_latched or self.sensor_fault_latched
                             or io.DI_FireAlarm)


# ============================================================================
# Plant simulator (replace with real I/O driver for field use)
# ============================================================================
@dataclass
class PlantSimulator:
    t_outdoor: float = 16.0          # make-up air temperature
    t_room: float = 24.0             # start warm so ventilation has something to do
    heat_gain_kps: float = 0.010     # internal heat gain (K/s) when no extraction
    fault_inject: bool = False

    def step(self, io: IOImage, dt_s: float) -> None:
        flow = io.AO_FanExhaust_Pct / 100.0
        # extraction pulls in cooler make-up air -> cools the room proportional to flow
        self.t_room += (self.t_outdoor - self.t_room) * 0.08 * flow * dt_s
        # internal/solar heat gain always pushes the room up
        self.t_room += self.heat_gain_kps * dt_s

        io.AI_T_Room_C = self.t_room
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
                  f"Troom={io.AI_T_Room_C:5.1f} (SP {P.ROOM_SP_C:.0f}) "
                  f"Extract={io.AO_FanExhaust_Pct:5.1f}% "
                  f"Damper={io.AO_DamperExhaust_Pct:3.0f}% "
                  f"Alarm={int(io.DO_CommonAlarm)}")

        n += 1
        if realtime:
            next_t += P.DT_S
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    print("AHU_04 (Exhaust_Vent_Room_Temp) soft-PLC starting "
          "(200 ms cycle, Ctrl+C to stop)\n")
    try:
        run(cycles=None, realtime=True)
    except KeyboardInterrupt:
        print("\nStopped.")
