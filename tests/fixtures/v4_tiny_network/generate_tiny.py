"""Generate a minimal SWMM INP for pytest proofs (Gate 2).

2 junctions, 1 conduit, 1 binary pump, 1 outfall, 1 subcatchment, 1 rain gage.
Includes a [CONTROLS] rule: IF NODE J1 DEPTH > 0.5 THEN PUMP P1 SETTING = 1.
"""
from __future__ import annotations

from pathlib import Path

TINY_INP = """\
[TITLE]
V4 Tiny Network for pytest proofs

[OPTIONS]
FLOW_UNITS           CFS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS          DEPTH
MIN_SLOPE            0
ALLOW_PONDING        NO
SKIP_STEADY_STATE    NO

START_DATE           01/01/2020
START_TIME           00:00:00
REPORT_START_DATE    01/01/2020
REPORT_START_TIME    00:00:00
END_DATE             01/01/2020
END_TIME             00:30:00
SWEEP_START          01/01
SWEEP_END            12/31
DRY_DAYS             0
REPORT_STEP          00:05:00
WET_STEP             00:05:00
DRY_STEP             01:00:00
ROUTING_STEP         0:00:30
VARIABLE_STEP         0.75

[EVAPORATION]
;;Data Source    Parameters
;;-------------- ----------------
CONSTANT         0.0
DRY_ONLY         NO

[RAINGAGES]
;;Name           Format    Interval SCF      Source
;;-------------- --------- ------ ------ ----------
RG1              INTENSITY 0:05     1.0      TIMESERIES  rain_ts

[JUNCTIONS]
;;Name           Elev.     Max.Depth  Init.Depth  SurDepth  Aponded
;;-------------- ---------- ---------- ---------- ---------- ----------
J1               0          3          0          0          0
J2               0          3          0          0          0

[OUTFALLS]
;;Name           Elev.    Type       Stage Data       Gated    Route To
;;-------------- ---------- ---------- ---------------- -------- --------
OUT              -1       FREE                        NO

[CONDUITS]
;;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow
;;-------------- ---------------- ---------------- ---------- ---------- ---------- ---------- ---------- ----------
C1               J1               J2               100        0.01       0          0          0          0

[PUMPS]
;;Name           From Node        To Node          Pump Curve       Status   Sartup   Shutoff
;;-------------- ---------------- ---------------- ---------------- ------   ------   --------
P1               J2               OUT              P1_CURVE         ON     0        0

[SUBCATCHMENTS]
;;Name           Rain Gage        Outlet           Area     %Imperv  Width    %Slope   CurbLen  SnowPack
;;-------------- ---------------- ---------------- -------- -------- -------- -------- -------- --------
S1               RG1              J1               10       50       500      0.5      0

[SUBAREAS]
;;Subcatchment   N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo    PctRouted
;;-------------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
S1               0.1        0.1        0.05       0.05       25         OUTLET

[XSECTIONS]
;;Link           Shape        Geom1            Geom2      Geom3      Geom4      Barrels    Culvert
;;-------------- ------------ ---------------- ---------- ---------- ---------- ---------- ----------
C1               CIRCULAR     0.3              0          0          0          1

[INFILTRATION]
;;Subcatchment   Param1     Param2     Param3     Param4     Param5
;;-------------- ---------- ---------- ---------- ---------- ----------
S1               3.0        0.5        4          7          0

[CURVES]
;;Name           Type       Parameters
;;-------------- ---------- ----------
P1_CURVE         Pump4      0 0.0  1.0 0.5

[TIMESERIES]
;;Name           Date       Time       Value
;;-------------- ---------- ---------- ----------
rain_ts                     0:00       0.0
rain_ts                     0:05       2.0
rain_ts                     0:10       4.0
rain_ts                     0:15       6.0
rain_ts                     0:20       3.0
rain_ts                     0:25       1.0

[CONTROLS]
RULE 1
IF NODE J1 DEPTH > 0.5
THEN PUMP P1 SETTING = 1.0
PRIORITY 1

[REPORT]
SUBCATCHMENTS  ALL
NODES          ALL
LINKS          ALL
"""


def write_tiny_inp(path: str | Path) -> Path:
    """Write the tiny INP to *path* and return the Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(TINY_INP, encoding="utf-8")
    return p


if __name__ == "__main__":
    out = Path(__file__).parent / "tiny.inp"
    write_tiny_inp(out)
    print(f"Wrote {out}")
