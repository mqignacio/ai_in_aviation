# Engine Maintenance Program — Turbofan Engine Series TF-2000

**Document Reference:** EMP-TF2000-Rev.4  
**Applicability:** All TF-2000 series commercial turbofan engines  
**Classification:** Internal — Maintenance Planning Use Only

---

## 1. Purpose and Scope

This Engine Maintenance Program (EMP) establishes the inspection intervals, component replacement thresholds, and corrective actions for the TF-2000 series turbofan engine. The program is designed to ensure safe and reliable engine operation while minimizing unnecessary maintenance events.

All maintenance decisions shall be based on the most current engine health data, including sensor readings, cycle counts, and Remaining Useful Life (RUL) predictions from the onboard prognostics system.

---

## 2. Engine Overview

The TF-2000 is a high-bypass-ratio commercial turbofan engine with the following key parameters:

| Parameter | Value |
|---|---|
| Thrust (sea level, static) | 275 kN |
| Bypass ratio | 9.6:1 |
| Overall pressure ratio | 40:1 |
| Maximum turbine inlet temp. | 1,520°C |
| Length | 6.1 m |
| Dry weight | 5,800 kg |
| Sensor channels | 21 (temperatures, pressures, speeds, ratios) |
| Operational settings | 3 (thrust level, altitude regime, ambient condition) |

The engine is monitored by 21 sensor channels measuring temperatures, pressures, rotor speeds, and derived ratios across the fan, compressor, combustor, and turbine sections.

---

## 3. Inspection Intervals

Inspections are categorized by depth and scope. The interval is expressed in operating cycles.

| Inspection Level | Interval (cycles) | Scope | Downtime |
|---|---|---|---|
| **A-Check** | Every 150 cycles | Visual inspection, oil analysis, basic sensor calibration | 4–6 hours |
| **B-Check** | Every 600 cycles | Detailed sensor diagnostics, vibration analysis, borescope of fan and LPC | 1–2 days |
| **C-Check** | Every 3,600 cycles | Full engine teardown, HPC/LPC blade inspection, bearing replacement, combustor inspection | 2–4 weeks |
| **D-Check (Hot Section Overhaul)** | Every 18,000 cycles or 6 years | Complete engine disassembly, all components inspected or replaced, full certification test | 6–12 weeks |

**Note:** Inspection intervals may be shortened based on RUL predictions (see Section 5).

---

## 4. Component Replacement Thresholds

The following table lists critical components and their replacement criteria. Components shall be replaced when ANY of the listed conditions is met.

| Component | Life Limit (cycles) | Condition-Based Trigger | Action |
|---|---|---|---|
| Fan blades | 36,000 | Tip clearance > 3.2 mm or crack detected | Replace set |
| LPC blades | 36,000 | Vibration amplitude > 0.15 in/s RMS | Replace set |
| HPC blades | 18,000 | Efficiency drop > 5% or crack detected | Replace set |
| HPC vanes | 18,000 | EGT margin < 15°C | Replace set |
| Turbine blades (HP) | 12,000 | Creep strain > 0.5% or coating loss > 40% | Replace set |
| Turbine blades (LP) | 18,000 | Tip rub detected or efficiency drop > 3% | Replace set |
| Main bearings | 36,000 | Vibration > 0.20 in/s or oil metal particles | Replace set |
| Combustor liner | 24,000 | EGT spread > 25°C or crack detected | Replace |
| Fuel nozzles | 18,000 | Fuel flow deviation > 3% from baseline | Replace set |
| Oil pump | 36,000 | Oil pressure drop > 10% or metal debris | Replace |

---

## 5. RUL-Based Maintenance Decision Matrix

The onboard prognostics system provides a Remaining Useful Life (RUL) estimate in operating cycles. The following decision matrix maps RUL ranges to required maintenance actions.

| RUL Range | Health Status | Required Action | Timeframe |
|---|---|---|---|
| **RUL > 200 cycles** | **Healthy** | Continue normal operations. Next scheduled A-Check applies. | — |
| **RUL 100–200 cycles** | **Monitor** | Increase sensor monitoring frequency to every flight. Review trend data at next A-Check. | Next A-Check |
| **RUL 50–100 cycles** | **Warning** | Schedule a B-Check within the next 30 cycles. Prepare component replacement parts based on trending. | Within 30 cycles |
| **RUL 30–50 cycles** | **Caution** | Schedule a B-Check immediately. Conduct borescope inspection of HPC and turbine sections. Identify likely failing components. | Within 15 cycles |
| **RUL 15–30 cycles** | **Advisory** | Plan for C-Check or targeted hot-section inspection. Remove engine from service if RUL drops below 20 cycles during operation. | Within 10 cycles |
| **RUL < 15 cycles** | **Critical** | **Remove engine from service immediately.** Schedule full C-Check or engine swap. Do not dispatch with this engine. | Immediate |

### 5.1 Escalation Rules

- If RUL decreases by more than 20 cycles between consecutive predictions, escalate one level (e.g., from Warning to Caution).
- If two or more sensors show simultaneous degradation trends, escalate one level regardless of absolute RUL.
- If EGT margin drops below 25°C, escalate to at least Caution level regardless of RUL.

---

## 6. Corrective Actions by Fault Mode

### 6.1 HPC (High-Pressure Compressor) Degradation

**Symptoms:** Gradual increase in EGT, decrease in compressor efficiency, increased fuel flow for same thrust, rising HPC outlet temperature (sensor 4), falling HPC efficiency (derived from sensors 7 and 12), and gradual erosion of stall margin.

**HPC Degradation Corrective Actions (in order of escalation):**

1. **Verify sensor calibration** (sensors 3, 4, 12, 16). Rule out instrumentation drift before assuming a mechanical fault.
2. **Trend review:** Compare the last 10 cycles of EGT margin and HPC efficiency against baseline. A decline of more than 2% per 50 cycles indicates active degradation.
3. **Borescope inspection** of HPC blades and vanes at the next available maintenance opportunity (do not wait for the next scheduled B-Check if RUL is below 100 cycles).
4. **If blade fouling detected:** perform a compressor chemical wash (can be done within an A-Check, no teardown required). Re-check EGT margin after the wash.
5. **If blade erosion or coating loss detected (< 40%):** continue monitoring; re-inspect at next A-Check.
6. **If blade damage, cracking, or coating loss ≥ 40% detected:** schedule a C-Check for HPC blade and vane replacement (see Section 4 thresholds).
7. **If HPC efficiency drop exceeds 5% or EGT margin falls below 15°C:** this meets the Section 4 replacement trigger — remove and replace the HPC blade/vane set at the next C-Check; do not defer.
8. **If HPC degradation is detected in combination with an EGT spread greater than 25°C:** treat as combined degradation per Section 6.3 and escalate to immediate borescope of the full engine core.
9. **Update the baseline** sensor trend data after any HPC maintenance action so future degradation is measured against the corrected engine state.

**Recommended Timeframe:** If RUL is in the Warning band (50–100 cycles) or better, HPC corrective actions can be scheduled within the next B-Check window. If RUL is in the Caution band (30–50 cycles) or worse, the borescope inspection and any required blade replacement must be completed before the next flight cycle threshold is reached.

### 6.2 Fan Degradation

**Symptoms:** Increased vibration, decreased bypass ratio, abnormal fan discharge pressure.

**Corrective Actions:**
1. Check fan blade tip clearance.
2. Borescope inspection of fan and LPC blades.
3. If foreign object damage (FOD) detected: assess severity, replace affected blades.
4. If bearing wear detected: replace main bearing set.
5. Vibration test after repair.

### 6.3 Combined Degradation (Multiple Fault Modes)

**Symptoms:** Multiple sensor drifts, rapid RUL decline, EGT margin erosion.

**Corrective Actions:**
1. **Immediate borescope of full engine core.**
2. Cross-reference sensor trends with component life limits (Section 4).
3. Prepare for C-Check with full component replacement list.
4. If RUL < 20 cycles: engine swap is mandatory.
5. Post-maintenance certification test required before return to service.

---

## 7. Maintenance Documentation Requirements

All maintenance actions shall be documented in the engine log with the following minimum information:

- Date and cycle count at time of maintenance
- RUL prediction at time of decision
- Sensors flagged as degraded (with values)
- Components inspected and/or replaced
- Maintenance level performed (A/B/C/D-Check)
- Post-maintenance test results
- Technician sign-off and certification number

---

## 8. References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." *Proc. 1st Int. Conf. on Prognostics and Health Management (PHM08)*.
- EASA AI Concept Paper Issue 2 — Level 1/2 AI in aviation maintenance.
- FAA Advisory Circular 43.13 — Acceptable Methods, Techniques, and Practices for Aircraft Inspection and Repair.
