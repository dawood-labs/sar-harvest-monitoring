# SOP: Harvest monitoring tools

## 1. In short
1. Tool 1 (`preprocess_classified.py`) reads a classified raster (1 = cane) and L1. It makes cane polygons (81).
2. You check and edit these polygons in QGIS.
3. Tool 2 (`postprocess_remaining.py`) makes the final layer: 81 Remaining cane, 82 Harvested.
4. Tool 2 also checks that 81 + 82 equals the L1 area.

Order: raster + L1 -> TOOL 1 -> check in QGIS -> TOOL 2 -> 81 and 82 polygons

## 2. Set up once
1. Install Python 3.10 or newer from python.org. Tick "Add Python to PATH".
2. Open "Command Prompt". Run this one command:
```
pip install numpy rasterio shapely pyogrio pyproj
```
3. Optional: `pip install psutil`.
4. Test it (you must see a version number, no error):
```
python -c "import numpy, rasterio, shapely, pyogrio, pyproj; print(shapely.__version__)"
```
5. Put BOTH .py files in the SAME folder, for example `C:\tools`.
   Tool 2 needs Tool 1 next to it.
## 3. Before every run
- [ ] Close heavy programs (QGIS projects, many browser tabs). The tool uses almost all free memory.
- [ ] Do not start other heavy work during the run.
- [ ] Charger on. Sleep off.
- [ ] Free disk space on the output drive.
- [ ] The output file is not open in QGIS or ArcMap.

## 4. Tool 1 - make remaining cane
```
cd C:\tools
python preprocess_classified.py
```

Enter keeps the value in [brackets]. Paste paths in any style.
| Question as printed | What you type | Enter gives |
|---|---|---|
| `1) Classified raster (.tif):` | `C:\work\report1\classified.tif` | nothing (first run) |
| `2) L1 layer (.shp / .gpkg):` | `D:\data\L1.shp` | nothing (first run) |
| `3) Output folder [C:\work\report1]:` | a folder, or a full file path | the raster folder |
| `   Output name [classified_remaining_cane]:` | a name, no extension | raster name + `_remaining_cane` |
| `   Save as:  1) Shapefile (.shp)   2) GeoPackage (.gpkg)  [1]:` | `2` for a very big result | 1 (shapefile) |
| `4) Apply the majority filter? [Y/n]:` | `n` = no filter | yes |
| `   Number of neighbours (4 or 8) [4]:` | `8` | 4 |
| `   Choose number, zone (e.g. 47N) or EPSG code [1]:` | a number, `47N` or `32647` | the suggested zone |
| `6) Remove slivers? [Y/n]:` | `n` = keep all | yes |
| `   Minimum area in acres [0.2]:` | e.g. `0.5` (above 5 the tool asks `Use it? [y/N]`) | 0.2 acres |
| `Start? [Y/n]:` | `n` = cancel (`Cancelled.`) | start |

Notes:
- A full file path at the folder question works.
- A bad name asks again: `A file name cannot contain : ? (Windows rule). Spaces, - _ ( ) [ ] are fine.`
- Majority filter = smooths the raster. Slivers = tiny polygons (noise).
- Projection: the tool lists every UTM zone the raster touches. Number 1 = centre zone. Acres use the ONE zone you pick.
  On a later run you may see `Enter keeps your earlier choice: UTM 47N (EPSG:32647)`. Enter keeps that zone.
- A non-UTM choice (e.g. 3857) asks `Use WGS 84 / Pseudo-Mercator anyway? [y/N]:`. Press Enter (no).
- A far zone asks `UTM zone 36 is 11 zones away from the data centre (zone 47); areas get less accurate far from the zone centre.` Press Enter (no).
- If the output exists: `report2.shp was made by an earlier run of this tool. Replace it? [Y/n]:`.
Read the summary (`Your answers:`) before you press Enter at `Start?`.

While it runs you see `Machine: ...` and `Plan: ...` lines, for example:
```
Plan: 15 processes x 1 thread = 15 tiles at a time, 4 tiles of 256 x 256 cells (planned 21 bytes per cell)
```

- Process = a copy of the program. Thread = a worker inside a copy. Tile = a square piece of the raster.
- Processes x threads = tiles done at the same time.
- `memory mode: desktop` = normal PC. `container` = server with a hard memory limit.
- `using up to` = memory the tool may use. `kept free` = memory left for Windows.

What you get at the end (shortened):
```
================================ RESULT ================================
Remaining cane polygons written :         3
Total area                      : 1,597.586 acres
How the result was reached          Polygons      Acres
  Cane pieces inside L1                  156  1,602.158
  Removed: slivers below 0.5 acres       153      4.572
  Written to the output                    3  1,597.586
```

- `Written to the output` = what is in your file. Write these numbers in your report.
- Big results may add: `Note: adding up the area_ac column of the file gives 110,675.646 acres (each polygon is rounded to 3 decimals).` This is rounding.
- Fields: `predicted` (81) and `area_ac` (acres). Coordinate system: WGS 84.
## 5. Check in QGIS
1. Open the Tool 1 file in QGIS with the raster or an image.
2. Edit by hand if needed. Save the layer as a file. Close it in QGIS.

## 6. Tool 2 - remaining cane and harvested
```
python postprocess_remaining.py
```

| Question as printed | What you type | Enter gives |
|---|---|---|
| `1) Checked remaining-cane layer (FINAL, .shp / .gpkg):` | your checked file | nothing (first run) |
| `2) L1 layer (.shp / .gpkg):` | the same L1 as in Tool 1 | nothing (first run) |
| `3) Output folder [...]:` | a folder, or a full file path | the checked file folder |
| `   Output name [checked_harvest_final]:` | a name, no extension | checked name + `_harvest_final` |
| `   Save as: ... [1]:` | `2` = GeoPackage | 1 (shapefile) |
| `4) Turn small harvested pieces that touch remaining cane into remaining cane? [Y/n]:` | `n` = keep all as 82 | yes |
| `   Largest piece size in acres (pieces <= this size change) [0.2]:` | e.g. `0.5` | 0.2 acres |
| `5) Projection used to calculate acres ...` | as in Tool 1 | the suggested zone |
| `6) Classes to write: both, 81 (remaining) or 82 (harvested) [both]:` | `81` or `82` | both |
| `Start? [Y/n]:` | `n` = cancel | start |

Use the same L1 and the same zone as in Tool 1. The end of the run (shortened):

```
================================ RESULT ================================
Class                        Polygons      Acres
  81  Remaining cane                3  1,580.002
  82  Harvested                     5    465.634
  Total (81 + 82)                   8  2,045.636
Area check (remaining + harvested must equal L1)
  L1 area                              2,045.636
  Difference                               0.000  (0.0000 %)  OK
Small harvested pieces (rule: 0.2 acres or smaller AND touching remaining cane)
  Changed to remaining cane         2      0.205
```

- Area check must say `OK`. If it says `CHECK`, a `WARNING: 81 + 82 differs from the area of L1 ...` line follows. Send a screenshot.
- Small harvested pieces: small 82 pieces that touch 81 become 81. The last row shows how many.

## 7. Run again
Start the same tool. You see `A previous run was found:` and the old answers.
Then: `Continue the last run with these settings? [Y/n]:`
| You want to ... | Do this | What happens |
|---|---|---|
| Continue after a stop, power cut or Ctrl+C | Enter at `Continue ...` | Finished work is reused: `resuming: 722 of 2209 tiles already done` |
| Same settings, run was finished | Enter at `Continue ...` | `The output ... was already up to date for these settings; nothing was written again.` |
| Change only sliver size, projection or classes | `n`, then Enter on all, change one value | Only the last step runs, in seconds: `Setting changed: sliver size (0.2 -> 0.5). Only the last step runs again.` |
| Change only the output name | `n`, then type the new name | `Reusing saved work from <folder>` |
| Change the filter (Tool 1) | `n`, then change the filter answer | `Setting changed: number of neighbours (4 -> 8). Tiles must be recomputed.` then `Discard them and recompute? [Y/n]:` Enter = yes |
| Raster, L1 or checked file changed | Run again | `The file ... was modified since the last run (...). Tiles must be recomputed.` (Tool 2: `Inputs changed since the saved overlay: ...`) |
| Tool file is newer | Run again | `This work folder was made by an older version of the tool; its saved tiles cannot be reused.` |
| Start completely fresh | Delete the `<output name>_work` folder, then run | Everything is computed again |
After `n`, every question shows your old answer in [brackets].

## 8. Messages
| Message (short) | Meaning | Do this |
|---|---|---|
| `File not found: nofile.tif. Paste the full path of the raster.` | Wrong path | Paste the right path |
| `A file name cannot contain ...` | Bad character in the name | Type the name again |
| `The L1 layer has no coordinate system (.prj missing).` | L1 has no .prj | Fix it in QGIS. Answer `y` only if sure. Enter stops the run |
| `Please answer y or n.` / `Not understood. Type a list number, a zone like 47N, or an EPSG code like 32647.` / `Type both, 81 or 82.` / `Type a number, e.g. 0.2.` | Wrong answer | Answer again |
| `ERROR: preprocess_classified.py was not found. Put preprocess_classified.py in the same folder as this file.` | Tool 2 is alone in its folder | Put both .py files together |
| `ERROR: python package 'rasterio' is missing. Install it with: pip install rasterio ...` | Package missing | Run the pip command |
| `ERROR: the output X is open in another program (...)` | File open in QGIS/ArcMap | Close it, or use another name |
| `not enough disk space next to the output: ...` | Disk is full | Press Enter at `Choose another output path?` and pick another drive |
| `tile ... does not fit in memory even when split; ...` | Not enough memory | Close programs. Run again |

## 9. Good to know
- Saved work: the folder `<output name>_work` sits next to each output. It can get big (size is in the RESULT).
  Delete it when no run is active. `--clean` does it for you.
- Tool 1 saves a copy of the prepared L1 in your home folder: `.preprocess_classified\l1_prepared` (max 2 GB, safe to delete).
- Acres are calculated in ONE UTM zone, the one you choose. Use the same zone for every report.
- To copy the old workflow: the old files were made with no filter and no sliver removal in Tool 1.
  Answer `n` to the filter and `n` to slivers. Then use the 0.2 acre rule in Tool 2 (the default).
  Areas the analyst added by hand come only from your QGIS check.
- The result does not depend on the computer: every machine gives the same polygons and acres.
- Limits: the 8-neighbour filter and the outer row of the raster are not checked against ArcMap. With under about 2 GB free memory, the tool may use a little more than planned.
- A shapefile cannot be bigger than 2 GB. Use GeoPackage for very big results.
- If something goes wrong, send a screenshot of the whole window.
## 10. Without questions
```
python preprocess_classified.py --raster C:\work\report1\classified.tif --l1 D:\data\L1.shp --out C:\work\report1\remaining_cane.shp --min-acres 0.2 --utm 47N --neighbors 4
python postprocess_remaining.py --final C:\work\report1\checked.shp --l1 D:\data\L1.shp --out C:\work\report1\final.shp --sliver-acres 0.2 --utm 47N
```
