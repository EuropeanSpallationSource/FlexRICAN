#!/usr/bin/env python3
"""
================================================================================
 AHU_01 - Air Handling Unit Control (Python soft-PLC)
================================================================================
 Purpose
   A standalone control program that runs the AHU on a virtual PC as a
   functional alternative to the licensed CoDeSys 3.5 Structured Text program
   (AHU_01.st). The control logic mirrors the ST program 1:1.

 Source
   10ahus_selection_AHU1.json  (project "10AHUs", AHU "AHU_01")

 Architecture
   - IOImage          : the PLC I/O image (digital/analog in & out)
   - AHUController     : the cyclic control logic, scan() runs once per cycle
   - PlantSimulator    : a simple physical model so the program can run with no
                          hardware (replace with a real IO driver for field use)
   - run()             : the 200 ms real-time scheduler

 Field deployment
   Swap PlantSimulator for a driver that reads/writes real I/O (Modbus, OPC UA,
   bus card, ...) by populating IOImage inputs before scan() and pushing
   IOImage outputs after scan(). The control logic stays unchanged.
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum


# ============================================================================
# Engineering parameters (from the JSON selection)
# ============================================================================
class P:
    CYCLE_MS = 200                  # task cycle
    DT_S = CYCLE_MS / 1000.0        # cycle in seconds

    # Sequence (AHU_SEQUENCE)
    SUPPLY_SP_C = 18.0              # supplyTempSpC
    ROOM_SP_C = 22.0               # roomTempSpC
    MIN_SUPPLY_C = 12.0            # minSupplyTempC
    MAX_SUPPLY_C = 30.0           # maxSupplyTempC
    STARTUP_DAMPER_S = 20.0        # startupDamperWaitS
    STARTUP_FAN_S = 10.0          # startupFanWaitS
    SHUTDOWN_PURGE_S = 0.0        # shutdownPurgeS

    # Fans (FAN_SUPPLY_VFD / FAN_EXHAUST_VFD)
    FAN_MIN_PCT = 20.0            # minSpeedPct
    FAN_MAX_PCT = 100.0          # maxSpeedPct
    FAN_SP_PCT = 60.0           # speedSpPct (base demand)
    FAN_RAMP_PCT_S = 10.0        # ramp rate %/s (startMode = RAMP)
    EXH_OFFSET_PCT = 0.0         # trackingOffsetPct (FOLLOW_SUPPLY)
    FAN_PROOF_S = 15.0          # proof-of-flow timeout

    # Dampers (DAMPER_INLET, MODULATING, fail CLOSE)
    DAMPER_OPEN_S = 120.0        # openTimeS

    # Heating coil + frost (HEATING_COIL_3WV)
    FROST_TRIP_C = 7.0          # tReturnTripC
    FROST_RELEASE_C = 10.0       # tReturnReleaseC
    PUMP_RUNON_S = 120.0        # pumpRunOnS

    # Return-water sensor supervision (temp_water_return)
    RW_ALARM_LOW_C = 5.0
    RW_ALARM_HIGH_C = 60.0
    RW_WARN_LOW_C = 10.0
    RW_WARN_HIGH_C = 55.0

    # Sensor plausibility (Ni1000 range -20..80)
    TEMP_RANGE_LO_C = -20.0
    TEMP_RANGE_HI_C = 80.0

    # Cascade room PI controller
    ROOM_KP = 1.5
    ROOM_TI_S = 600.0

    # Supply PI controller (output -100..+100 ; + = heat, - = cool)
    SUP_KP = 12.0
    SUP_TI_S = 180.0
    SEQ_DEADBAND_PCT = 5.0


# ============================================================================
# State machine
# ============================================================================
class State(IntEnum):
    OFF = 0
    STARTUP = 1
    RUN = 2
    SHUTDOWN = 3
    FROST = 4
    FIRE = 5
    FAULT = 6


# ============================================================================
# Software TON timer (IEC 61131-3 on-delay), evaluated each scan
# ============================================================================
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


# ============================================================================
# I/O image
# ============================================================================
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

    # ---- Analog inputs (engineering units) ----
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
    AO_HeatValve_Pct: float = 0.0
    AO_CoolValve_Pct: float = 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ============================================================================
# Controller  (one scan() == one PLC cycle, mirrors AHU_01.st)
# ============================================================================
class AHUController:
    def __init__(self) -> None:
        self.state = State.OFF
        self.state_prev = State.OFF

        # latched faults
        self.frost_latched = False
        self.fan_fault_latched = False
        self.sensor_fault_latched = False
        self.frost_active = False

        # warnings
        self.filter_warn = False
        self.rwater_warn = False
        self.rwater_alarm = False

        # controllers
        self.supply_setpoint_c = P.SUPPLY_SP_C
        self.room_i = 0.0
        self.sup_i = 0.0
        self.seq_out_pct = 0.0
        self.fan_demand_pct = 0.0
        self.heat_demand = False

        # timers
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

        # --- 1) Sensor supervision -----------------------------------
        for v in (io.AI_T_Outdoor_C, io.AI_T_Supply_C,
                  io.AI_T_Room_C, io.AI_T_WaterReturn_C):
            if v < P.TEMP_RANGE_LO_C or v > P.TEMP_RANGE_HI_C:
                self.sensor_fault_latched = True

        self.rwater_warn = (io.AI_T_WaterReturn_C < P.RW_WARN_LOW_C
                            or io.AI_T_WaterReturn_C > P.RW_WARN_HIGH_C)
        self.rwater_alarm = (io.AI_T_WaterReturn_C < P.RW_ALARM_LOW_C
                             or io.AI_T_WaterReturn_C > P.RW_ALARM_HIGH_C)
        self.filter_warn = io.DI_FilterSupply_Dirty or io.DI_FilterExhaust_Dirty

        # --- 2) Fan fault supervision --------------------------------
        sp_proof = self.ton_supply_proof(
            io.DO_FanSupply_Run and not io.DI_FanSupply_Run, P.FAN_PROOF_S, dt_s)
        ex_proof = self.ton_exhaust_proof(
            io.DO_FanExhaust_Run and not io.DI_FanExhaust_Run, P.FAN_PROOF_S, dt_s)
        if io.DI_FanSupply_Fault or io.DI_FanExhaust_Fault or sp_proof or ex_proof:
            self.fan_fault_latched = True

        # --- 3) Frost protection (hysteresis) ------------------------
        if io.AI_T_WaterReturn_C <= P.FROST_TRIP_C:
            self.frost_active = True
            self.frost_latched = True            # ALARM_LATCH
        elif io.AI_T_WaterReturn_C >= P.FROST_RELEASE_C:
            self.frost_active = False

        # --- 4) Alarm reset / acknowledge ----------------------------
        if io.DI_Reset:
            if not self.frost_active:
                self.frost_latched = False
            if not (io.DI_FanSupply_Fault or io.DI_FanExhaust_Fault):
                self.fan_fault_latched = False
            self.sensor_fault_latched = False

        # --- 5) State machine ----------------------------------------
        # high-priority overrides: FIRE > FROST > FAULT
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

        # --- 6) Temperature control (cascade) ------------------------
        if self.state == State.RUN:
            # 6a outer room PI -> supply setpoint
            err = P.ROOM_SP_C - io.AI_T_Room_C
            self.room_i += (P.ROOM_KP / P.ROOM_TI_S) * err * dt_s
            self.room_i = _clamp(self.room_i,
                                 P.MIN_SUPPLY_C - P.SUPPLY_SP_C,
                                 P.MAX_SUPPLY_C - P.SUPPLY_SP_C)
            self.supply_setpoint_c = _clamp(
                P.SUPPLY_SP_C + P.ROOM_KP * err + self.room_i,
                P.MIN_SUPPLY_C, P.MAX_SUPPLY_C)

            # 6b inner supply PI -> sequencer demand -100..+100
            err = self.supply_setpoint_c - io.AI_T_Supply_C
            self.sup_i = _clamp(self.sup_i + (P.SUP_KP / P.SUP_TI_S) * err * dt_s,
                                -100.0, 100.0)
            self.seq_out_pct = _clamp(P.SUP_KP * err + self.sup_i, -100.0, 100.0)
        else:
            self.room_i = 0.0
            self.sup_i = 0.0
            self.seq_out_pct = 0.0
            self.supply_setpoint_c = P.SUPPLY_SP_C

        # 6c split-range sequencing with deadband
        db = P.SEQ_DEADBAND_PCT
        if self.seq_out_pct > db:
            io.AO_HeatValve_Pct = (self.seq_out_pct - db) / (100.0 - db) * 100.0
            io.AO_CoolValve_Pct = 0.0
            self.heat_demand = True
        elif self.seq_out_pct < -db:
            io.AO_HeatValve_Pct = 0.0
            io.AO_CoolValve_Pct = (-self.seq_out_pct - db) / (100.0 - db) * 100.0
            self.heat_demand = False
        else:
            io.AO_HeatValve_Pct = 0.0
            io.AO_CoolValve_Pct = 0.0
            self.heat_demand = False

        # --- 7) Output mapping per state -----------------------------
        s = self.state
        if s in (State.OFF, State.SHUTDOWN):
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = 0.0
            io.AO_DamperExhaust_Pct = 0.0
            io.AO_HeatValve_Pct = 0.0
            io.AO_CoolValve_Pct = 0.0

        elif s == State.STARTUP:
            io.AO_DamperInlet_Pct = 100.0
            io.AO_DamperExhaust_Pct = 100.0
            self.fan_demand_pct = P.FAN_SP_PCT if self.ton_fan.IN else 0.0
            io.AO_HeatValve_Pct = 0.0
            io.AO_CoolValve_Pct = 0.0

        elif s == State.RUN:
            io.AO_DamperInlet_Pct = 100.0
            io.AO_DamperExhaust_Pct = 100.0
            self.fan_demand_pct = P.FAN_SP_PCT

        elif s == State.FROST:
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = 0.0
            io.AO_DamperExhaust_Pct = 0.0
            io.AO_HeatValve_Pct = 100.0
            io.AO_CoolValve_Pct = 0.0
            self.heat_demand = True

        elif s == State.FIRE:                 # FANS_OFF_DAMPERS_CLOSE
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = 0.0
            io.AO_DamperExhaust_Pct = 0.0
            io.AO_HeatValve_Pct = 0.0
            io.AO_CoolValve_Pct = 0.0

        elif s == State.FAULT:
            self.fan_demand_pct = 0.0
            io.AO_DamperInlet_Pct = 0.0
            io.AO_DamperExhaust_Pct = 0.0
            io.AO_HeatValve_Pct = 0.0
            io.AO_CoolValve_Pct = 0.0

        # --- 8) Fan ramp + exhaust tracking --------------------------
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

        # --- 9) Heating pump (demand + run-on + frost) ---------------
        runon_done = self.ton_pump_runon(not self.heat_demand, P.PUMP_RUNON_S, dt_s)
        if self.heat_demand or self.frost_active or self.frost_latched:
            io.DO_HeatPump_Run = True
        elif not runon_done:
            io.DO_HeatPump_Run = True
        else:
            io.DO_HeatPump_Run = False

        # --- 10) Common alarm ----------------------------------------
        io.DO_CommonAlarm = (self.frost_latched or self.fan_fault_latched
                             or self.sensor_fault_latched or io.DI_FireAlarm
                             or self.rwater_alarm)


# ============================================================================
# Plant simulator (replace with real I/O driver for field use)
# ============================================================================
@dataclass
class PlantSimulator:
    """Very small lumped model: supply temp reacts to coil valves and fan flow,
    room temp reacts to supply air, return-water temp reacts to heating valve."""
    t_outdoor: float = 2.0
    t_supply: float = 15.0
    t_room: float = 20.0
    t_water_return: float = 35.0
    fault_inject: bool = False

    def step(self, io: IOImage, dt_s: float) -> None:
        flow = io.AO_FanSupply_Pct / 100.0            # 0..1 air flow

        # heating / cooling effect on supply air
        heat = io.AO_HeatValve_Pct / 100.0
        cool = io.AO_CoolValve_Pct / 100.0
        # supply pulled toward a mix of outdoor air and coil action
        target_supply = (self.t_outdoor
                         + heat * 25.0          # full heating adds up to +25 K
                         - cool * 18.0)         # full cooling subtracts up to 18 K
        if flow > 0.01:
            self.t_supply += (target_supply - self.t_supply) * min(1.0, 0.6 * dt_s)
        else:
            # no flow -> supply sensor drifts toward room/outdoor average
            self.t_supply += ((self.t_room - self.t_supply) * 0.05 * dt_s)

        # room reacts to supply air when fans run, else drifts to outdoor
        if flow > 0.01:
            self.t_room += (self.t_supply - self.t_room) * 0.10 * flow * dt_s
        else:
            self.t_room += (self.t_outdoor - self.t_room) * 0.005 * dt_s

        # return-water temperature: heating valve + pump warm it, idle cools it
        if io.DO_HeatPump_Run:
            warm_target = 20.0 + heat * 45.0
            self.t_water_return += (warm_target - self.t_water_return) * 0.15 * dt_s
        else:
            self.t_water_return += (self.t_outdoor - self.t_water_return) * 0.02 * dt_s

        # write sensors back into the I/O image
        io.AI_T_Outdoor_C = self.t_outdoor
        io.AI_T_Supply_C = self.t_supply
        io.AI_T_Room_C = self.t_room
        io.AI_T_WaterReturn_C = self.t_water_return

        # fan run feedback follows the command (with the fault injection option)
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
        plant.step(io, P.DT_S)      # read field inputs (here: simulate)
        ctrl.scan(io, P.DT_S)       # run control logic
        # (in the field: push io.* outputs to the I/O driver here)

        if n % 25 == 0:             # print ~ every 5 s
            print(f"t={n * P.DT_S:6.1f}s  {ctrl.state.name:8s} "
                  f"Troom={io.AI_T_Room_C:5.1f} Tsup={io.AI_T_Supply_C:5.1f} "
                  f"SPsup={ctrl.supply_setpoint_c:4.1f} "
                  f"Fan={io.AO_FanSupply_Pct:5.1f}% "
                  f"Heat={io.AO_HeatValve_Pct:5.1f}% Cool={io.AO_CoolValve_Pct:5.1f}% "
                  f"Pump={int(io.DO_HeatPump_Run)} Alarm={int(io.DO_CommonAlarm)}")

        n += 1
        if realtime:
            next_t += P.DT_S
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    print("AHU_01 soft-PLC starting (200 ms cycle, Ctrl+C to stop)\n")
    try:
        run(cycles=None, realtime=True)
    except KeyboardInterrupt:
        print("\nStopped.")
