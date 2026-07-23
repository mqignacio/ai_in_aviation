# C-MAPSS Dataset — Source and License

## Origin

The NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset is produced by the NASA Prognostics Center of Excellence (PCoE) at Ames Research Center. It provides simulated run-to-failure data for commercial turbofan engines, widely used as a benchmark in prognostics and health management (PHM) research.

## Subset Used

This project uses **FD001** (one of four available subsets):

- **Operating conditions:** 1 (sea level)
- **Fault modes:** 1 (HPC degradation)
- **Training trajectories:** 100 engines
- **Test trajectories:** 100 engines
- **Features:** 26 columns (unit number, cycle, 3 operational settings, 21 sensor measurements)

## Files

| File | Description |
|---|---|
| `train_FD001.txt` | Training data (run-to-failure trajectories for 100 engines) |
| `test_FD001.txt` | Test data (partial trajectories for 100 engines) |
| `RUL_FD001.txt` | Ground truth Remaining Useful Life values for each test engine |
| `readme.txt` | Original dataset documentation |

## License Terms

**Status: Public domain for research and educational use. No explicit license file shipped with the dataset.**

Verification performed 2026-07-06:

1. **NASA Open Data Portal** (`data.nasa.gov/dataset/cmapss-jet-engine-simulated-data`) — Lists "License not specified" but hosts the dataset under NASA's public open data portal, which operates under the U.S. Government Public Domain policy. NASA data is generally considered public domain unless otherwise stated.

2. **IEEE DataPort** (`ieee-dataport.org/documents/c-mapss-dataset`) — Mirrors the dataset behind a login wall. No explicit license stated on the page. IEEE DataPort datasets typically require attribution.

3. **Original release** — The dataset was first released as part of the PHM 2008/2009 Data Challenge (Prognostics and Health Management conference). The original authors made it freely available for research and benchmarking purposes.

**Assessment:** The dataset is widely used in academic research with attribution to the original PHM08 paper. No commercial license restriction has been found. For this course (educational, non-commercial), usage is appropriate with proper citation.

**Recommendation:** Users planning commercial applications should verify current terms directly with NASA PCoE or IEEE DataPort.

## Citation

Saxena, A., Goebel, K., Simon, D. and Eklund, N., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," in *Proc. 1st Int. Conf. on Prognostics and Health Management (PHM08)*, Denver, CO, Oct. 2008.

## Download Sources

- NASA Open Data Portal: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- IEEE DataPort: https://ieee-dataport.org/documents/c-mapss-dataset
