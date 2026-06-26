# M9 Open Access Database of AHU Program Blocks

This folder contains an open-access database of Air Handling Unit (AHU) program blocks created for the FlexRICAN project. The database includes ten example AHU control programs, AHU1 to AHU10, prepared with the TechSchem Editor.

The TechSchem Editor was used to create the program blocks and is one of the Key Exploitable Results (KER) of the project.

## How to choose the correct AHU

| Folder | Main specialization | Use this unit when you need... |
| --- | --- | --- |
| `AHU1` | Supply/exhaust AHU with energy recovery, water heating and cooling | A complete supply/exhaust unit with ERV, VFD fans, heating coil, cooling coil, room-temperature control and protection functions. |
| `AHU2` | Dehumidification unit | An AHU focused on dehumidification control. |
| `AHU3` | Dehumidification with water and electric after-heating | Dehumidification combined with additional water/electric heating stages. |
| `AHU4` | Exhaust ventilation with room-temperature control | An exhaust/ventilation unit where room temperature is part of the control logic. |
| `AHU5` | Fresh-air unit | A fresh-air AHU example for basic ventilation and air supply. |
| `AHU6` | Fresh-air unit with electric heater | A fresh-air AHU where heating is provided by an electric heater. |
| `AHU7` | Clean-room air circulation with HEPA filtration | A circulation unit for clean-room or high-filtration applications. |
| `AHU8` | Rooftop AHU with heating | A rooftop unit focused on heating operation. |
| `AHU9` | Rooftop AHU with heating and cooling | A rooftop unit requiring both heating and cooling operation. |
| `AHU10` | Fresh-air unit with water heating and cooling | A fresh-air AHU using water-based heating and cooling coils. |

## Contents of each AHU folder

Each AHU folder contains the complete material for one program block:

| File type | Purpose |
| --- | --- |
| Structured Text program, `AHU_XX.st` | PLC-oriented implementation of the AHU control program. |
| Python program, `ahu_xx_control.py` | Python implementation of the same AHU logic, useful for simulation, testing or soft-PLC use. |
| Program description, `AHU_XX_Documentation.md` | Main user manual for the selected AHU. It explains the control logic, I/O, parameters, alarms and how to implement or run the program. |
| TechSchem JSON configuration, `10ahus_selection_AHUXX.json` | Configuration file for opening or editing the AHU program block in the TechSchem Editor. |
| TechSchem project files | Complete TechSchem Editor project material for the AHU1-AHU10 program set. |
| Print screen, `.png` | Visual reference of the selected AHU/program block. |

`XX` represents the AHU number, for example `01` for AHU1 or `10` for AHU10.

## Recommended workflow

1. Select the AHU folder using the table above.
2. Open the `.png` print screen for a quick visual check.
3. Read the `AHU_XX_Documentation.md` file for implementation instructions.
4. Use the `.json` file if you want to open or modify the program in the TechSchem Editor.
5. Use the `.st` file for PLC implementation.
6. Use the `.py` file for simulation, testing or soft-PLC work.

## Important note

The files in this database are reference program blocks. Before use on real HVAC equipment, the selected program must be reviewed, connected to the correct I/O, tested, commissioned and tuned for the actual installation.

For project background and milestone context, see `M9_Milestone_Report_FINAL.docx`.
