# SOP: Remaining-cane polygons from a classified raster

Script: `preprocess_classified.py` (one file, no other script needed).

## 1. What this tool does

It replaces the ArcMap Model Builder pre-processing.
It reads a classified raster, where value 1 means standing sugarcane.
It can smooth the raster first (majority filter), then turns the cane cells into polygons.
It keeps only the parts inside the L1 layer, calculates acres, and removes tiny polygons (slivers).
The result is a polygon file in WGS 84 with the fields `predicted` (always 81) and `area_ac`.

You need two input files:

1. The classified raster, for example `C:\work\report1\classified.tif`.
2. The L1 polygon layer, for example `D:\data\L1.shp` (a .gpkg also works).
   A shapefile must come with its `.prj` file, so the coordinate system is known.

## 2. One-time setup on Windows

1. Install Python 3.10 or newer from python.org. Tick "Add Python to PATH" in the installer.
   (The script does not state a Python version, so 3.10 or newer is our advice.)
2. Open "Command Prompt" and run this one command:

```
pip install numpy rasterio shapely pyogrio pyproj
```

   Optional but useful: `pip install psutil` (lets the tool read memory more exactly).
   Shapely must be version 2.0 or newer.
   If you use conda, this works too: `conda install -c conda-forge numpy rasterio shapely pyogrio pyproj psutil`.
3. Check that it worked. Run this line:

```
python -c "import numpy, rasterio, shapely, pyogrio, pyproj; print(shapely.__version__)"
```

   You should see a number such as `2.0.6` and no error.
4. Save `preprocess_classified.py` in a folder, for example `C:\tools\`.

If a package is missing, the tool stops at once and says which one: `ERROR: python package 'rasterio' is missing. Install it with: pip install rasterio ...`.

## 3. Before you start

The tool uses almost all free memory and leaves only a small reserve (about 2 GB, or 1.5 GB on a laptop with 8 GB or less). So:

1. Close heavy programs: ArcMap, QGIS and browsers with many tabs.
2. Do not start other heavy work while the tool runs.
3. Keep the laptop on the charger. Switch off sleep (Settings > System > Power > Screen and sleep > "Never").
4. Check that the output drive has free space. The tool also checks this and stops if space is short.
5. The output file must not be open in ArcMap or QGIS. Close it there first.
6. Put the output in a folder where you can write files.

## 4. How to run it

1. Open "Command Prompt" in the script folder: `cd C:\tools`.
2. Run:

```
python preprocess_classified.py
```

3. Answer the questions. Press Enter to accept the value in `[brackets]`.

Paths: paste them in any style. Quotes, backslashes and forward slashes all work.
In Windows Explorer you can use "Copy as path" and paste it, quotes included.

The first screen of a first-time run looks like this:

```
Remaining cane from a classified raster - answer the questions, Enter keeps the [default].
```

Questions, in order:

1. **Classified raster.** Prompt: `1) Classified raster (.tif):`
   Type: `C:\work\report1\classified.tif`. No default the first time.
2. **L1 layer.** Prompt: `2) L1 layer (.shp / .gpkg):`
   Type: `D:\data\L1.shp`.
3. **Output file.** Prompt: `3) Output file [C:\work\report1\classified_remaining_cane.shp]:`
   The default goes next to the raster, named `<raster name>_remaining_cane.shp`.
   Press Enter, or type another path ending in `.shp` or `.gpkg`, for example `E:\out\cane_report1.shp`.
   If the file exists and was not made by this tool, you see `classified_remaining_cane.shp already exists. Overwrite? [y/N]:`. Type `y` to replace it.
   If it was made by an earlier run of this tool, you see `classified_remaining_cane.shp was made by an earlier run of this tool. Replace it? [Y/n]:`. Enter = yes.
4. **Majority filter.** Prompt: `4) Apply the majority filter? [Y/n]:`
   Enter = yes. Type `n` to skip it.
   If yes, the next prompt is `   Number of neighbours (4 or 8) [4]:`. Enter = 4. Type `8` for 8 neighbours.
5. **Projection for acres.** The tool looks at where your data is and suggests a UTM zone:

```
5) Projection used to calculate acres (detected from the data extent):
   1) UTM 47N  EPSG:32647  (suggested: covers the centre of the data)
   Choose number, zone (e.g. 47N) or EPSG code [1]:
```

   Press Enter to take the suggestion. Or type a zone such as `42N`, or an EPSG code such as `32642`.
   If you choose a projection that is not UTM, the tool warns you:

```
   Areas are exact only in an equal-area or local UTM projection; in other projected systems (e.g. Web Mercator EPSG:3857) acres can be badly wrong.
   Use WGS 84 / Pseudo-Mercator anyway? [y/N]:
```

   Press Enter (= no) and choose again. Type `y` only if you really know you need it.
   A UTM zone far from your data gives a similar question: `UTM zone 36 is 11 zones away from the data centre (zone 47); areas get less accurate far from the zone centre.` followed by `Use it anyway? [y/N]:`.
6. **Slivers.** Prompt: `6) Remove slivers? [Y/n]:`
   Enter = yes. Then `   Minimum area in acres [0.2]:`. Enter = 0.2. Type for example `0.5` to remove polygons below 0.5 acre.
   Type `n` to keep every polygon.
7. **Summary and start.** The tool repeats your answers:

```
Your answers:
  raster      : C:\work\report1\classified.tif
  L1 layer    : D:\data\L1.shp
  output      : C:\work\report1\classified_remaining_cane.shp
  majority    : yes, 4 neighbours
  projection  : WGS 84 / UTM zone 47N (EPSG:32647)
  slivers     : remove below 0.2 acres
Start? [Y/n]:
```

   Check every line. Press Enter to start, or type `n` to cancel (`Cancelled.`).

## 5. What you see while it runs

First one line about your machine and the plan:

```
machine: 15 cores, 25.8 GB available, buffer 2.0 GB, budget 23.8 GB -> 15 workers, tiles of 128 x 128 cells, 16 tiles (planned 34 bytes per cell)
```

It means: the cores and free memory found, the memory it may use, the number of parallel workers, and the size and number of tiles.
A tile is one square piece of the raster, processed on its own.

Then progress lines with the elapsed time, in order:

```
[    0.1s] plan (400x400 cells, 88,431 cane cells)
[    0.0s] L1 prepared (reprojected, made valid): 16 of 16 tiles touch L1
[    2.2s] tiles (filter, polygonise, clip)
[    0.0s] merge seams: 17 seam pieces, 1 merged groups, 2 pieces saved
[    0.0s] final stage: areas, slivers, write C:\work\report1\classified_remaining_cane.shp
```

At the end:

```
SUMMARY polygons=1 total_acres=2,589.162 slivers_removed=1 slivers_acres=1.055 noise_pieces_removed=0 before_removal: polygons=2 acres=2,590.217
TOTAL wall time 0.1s
```

The SUMMARY line tells you: polygons written, their total acres, how many slivers were removed (and their acres), and the count before removal.
Write these numbers in your report.
Then the window says `Press Enter to close`.

The output files (next to the path you chose):

1. `classified_remaining_cane.shp` with `.shx`, `.dbf`, `.prj`, `.cpg`. Keep all of them together.
2. Fields: `predicted` = 81 and `area_ac` (acres, 3 decimals).
3. Coordinate system: WGS 84 (EPSG:4326).
4. Folder `classified_remaining_cane_work` (see section 8).

## 6. If it stops or the laptop shuts down

1. Start the tool again: `python preprocess_classified.py`.
2. It shows the last run. Only the `status` line differs for a stopped run (the example below is an illustration of that line):

```
A previous run was found:
  raster      : C:\work\report1\classified.tif
  L1 layer    : D:\data\L1.shp
  output      : C:\work\report1\classified_remaining_cane.shp
  majority    : yes, 4 neighbours, majority
  projection  : EPSG:32647
  slivers     : remove below 0.2 acres
  status      : interrupted, tiles, started 2026-11-25 08:39:52 (7 of 16 tiles saved)
Continue the last run with these settings? [Y/n]:
```

3. Press Enter. Finished tiles are reused, so only the missing work is done.
4. The status shows `finished` for a completed run and `interrupted` for a stopped one.
   If you press Enter on a finished run with no changes, it says `output classified_remaining_cane.shp is already up to date for these settings` and does nothing.

The tool saves a small memory file in your user folder (`.preprocess_classified\last_run.json`). That is how it remembers the last run.

## 7. Changing only one setting afterwards

Use this to test another sliver size or another projection without redoing the slow part.

1. Run `python preprocess_classified.py`.
2. At `Continue the last run with these settings? [Y/n]:` type `n`.
3. Now the answers of the last run are the defaults. Press Enter on every question and change only one value.
   Two things look different from a first run:
   - Question 3: the tool asks `... was made by an earlier run of this tool. Replace it? [Y/n]:`. Enter = yes.
   - Question 5: the default is the earlier EPSG code (for example `[EPSG:32647]`) when it fits the data. Enter keeps it.
     If the earlier code does not fit the new data, the default is the suggested zone `[1]` again.
4. Change the value, for example `Minimum area in acres [0.2]:` type `5`.
5. Check the summary (`slivers     : remove below 5.0 acres`) and press Enter at `Start?`.

The tool then says:

```
reusing the saved tile and merge results (only the final stage runs)
[    0.0s] final stage: areas, slivers, write C:\work\report1\classified_remaining_cane.shp
```

This takes seconds, because the tiles are not computed again.

Changing the majority filter (on/off, 4 or 8), the raster, the L1 layer or the cane value changes the tiles.
Then the tool tells you what changed, for example:

```
Setting changed for the tile stage: number of neighbours (4 -> 8). Tiles must be recomputed.
```

In question mode it first asks `Old tiles in <work folder> cannot be reused. Discard them and recompute? [Y/n]:`. Enter = yes.

## 8. The work folder

Next to the output there is a folder `<output name>_work`, for example `classified_remaining_cane_work`.

1. It holds the saved tiles, `merged.bin` (the merged result, which makes reruns fast), `manifest.json` and a marker file `preprocess_classified.workfolder`.
2. It can be large for big areas.
3. It is safe to delete when you are done and no run is active. Delete the whole folder in Explorer.
4. The tool only ever deletes its own files in it. If a folder of that name exists but was not made by the tool, it refuses to use it.
5. To delete it automatically after a successful run, use the flag `--clean` (see section 10).
6. After deleting it, a rerun starts from the beginning.

## 9. Common messages

| Message as printed | Meaning | What to do |
|---|---|---|
| `ERROR: the output classified_remaining_cane.shp is open in another program (...). Close it in ArcMap/QGIS and run again, or choose another output name.` | Another program holds the output file. | Close the file or the project in ArcMap/QGIS, then run again. Or type another output name. |
| `not enough disk space next to the output: about X GB needed, Y GB free in <folder>` | The drive is too full. | The tool asks `Choose another output path? [Y/n]:`. Press Enter and type a path on a drive with more space. Or free space first. |
| `ERROR: cannot write in the output folder <folder>` (question mode: `Cannot write in <folder>. Choose another folder.`) | No permission or a read-only folder. | Choose a folder you own, such as `C:\work\...`. |
| `ERROR: another run (process N, started ...) is using <work folder>. Wait for it to finish or choose another output name.` | The same output is being processed in another window. | Wait for the other run, or close it. Or choose another output name. |
| `... is not a UTM zone. Areas are exact only in an equal-area or local UTM projection ... Add --allow-non-utm to use it anyway.` | You chose a projection that is not UTM. | Use a UTM zone (press Enter at question 5). |
| `ERROR: the L1 layer has no coordinate system (.prj missing). Add one, ...` (question mode asks: `Assume it uses the same coordinate system as the raster? [y/N]:`) | The L1 file has no `.prj`. | Best: define the coordinate system in ArcMap/QGIS and run again. Type `y` only if you are sure L1 uses the raster's system. |
| `one output polygon is too large for the memory budget (N vertices, about X GB needed); raise --max-memory-gb` | One very large polygon needs more memory than the budget. | Close other programs and run again. If it repeats, ask the developers. |
| `tile ... does not fit in memory even when split; free memory, raise --max-memory-gb, or close other programs` | Not enough free memory. | Close programs and run again. Finished tiles are kept. |
| `the folder <work folder> already exists and was not made by this script. Rename or move it, or choose another output name.` | A folder with the work-folder name already exists. | Rename it or choose another output name. |
| `File not found: <path>. Paste the full path of the raster.` | Wrong path. | Paste the correct path. |
| `ERROR: input ended before all questions were answered.` | The window was closed or input was cut. | Run again. |

## 10. Running without questions (advanced)

Same work in one command (all on one line in Command Prompt):

```
python preprocess_classified.py --raster C:\work\report1\classified.tif --l1 D:\data\L1.shp --out C:\work\report1\remaining_cane.shp --min-acres 0.2 --utm 47N --neighbors 4 --clean
```

Useful flags:

1. `--no-filter` skips the majority filter. `--neighbors 8` uses 8 neighbours.
2. `--keep-slivers` keeps all polygons. `--min-acres 0.5` changes the sliver size.
3. `--utm 47N` or `--utm EPSG:32647` sets the projection. Without it, the tool picks the zone from the data.
4. `--clean` deletes the work folder after a successful run.
5. `--max-memory-gb 8` limits the memory the tool may use.
6. `--help` lists every flag.

Errors in this mode are printed as `ERROR: ...` and the command ends with exit code 2.

## 11. Known limits

1. The 8-neighbour filter follows the Esri documentation. It has not yet been compared with an ArcMap output.
2. Cells on the outermost row and column of the raster may differ from ArcMap.
3. The tool has not yet been tested on Windows by the developers.
   If anything goes wrong, report it with a screenshot of the whole window.
4. A shapefile cannot be larger than 2 GB. For a very large result, give an output name ending in `.gpkg`.
