"""VisIt CLI script: overlay density lineouts at y=0.5 from three solutions,
one saved image per timestep.

Curves:
  Reference -- single-block run in ~/plot            (thick blue)
  ML        -- block B of ~/a_codes/euler-miniapp_AITraining/plot  (thin red)
  No ML     -- block B of ~/run/coupling/plot        (thin purple)

Run on the machine with VisIt installed:
    visit -cli -nowin -s visit_density_lineout.py
Frames land in ./lineout_frames/density_y05_NNNN.png.

Assumption (adjust if wrong): VisIt's CGNS reader can open the solution
files directly. Both solvers write grid and solution as separate files
(the reference run writes its grid just once, at step 0); if VisIt
complains about missing coordinates, that has to be checked interactively
once and the patterns reworked.
"""

import glob
import os

# ---------------- configuration ----------------
HOME = os.path.expanduser("~")
SOURCES = [
    {
        "label": "Reference",
        "dir": os.path.join(HOME, "plot"),
        # Single-block run: per-timestep solutions are ShockTube_2d_*.cgns;
        # the grid is written once (ShockTube_grid_2d_000000.cgns) and the
        # ShockTubePEdist_* pair is the MPI rank-ownership map -- neither
        # matches this glob, so both are ignored.
        "pattern": "ShockTube_2d_*.cgns",
        "color": (0, 0, 255, 255),  # blue
        "line_width": 5,            # thick
    },
    {
        "label": "ML",
        "dir": os.path.join(HOME, "a_codes/euler-miniapp_AITraining/plot"),
        "pattern": "blockB_2d_*.cgns",
        "color": (255, 0, 0, 255),  # red
        "line_width": 1,            # thin
    },
    {
        "label": "No ML",
        "dir": os.path.join(HOME, "run/coupling/plot"),
        "pattern": "blockB_2d_*.cgns",
        "color": (160, 32, 240, 255),  # purple
        "line_width": 1,               # thin
    },
]
VARIABLE = "Density"
Y_VALUE = 0.5
OUT_DIR = os.path.join(os.getcwd(), "lineout_frames")
IMAGE_WIDTH, IMAGE_HEIGHT = 1200, 900
# ------------------------------------------------

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

# Open each source as a VisIt virtual database (the "... database" suffix
# groups the per-timestep files into one time series) and add a Pseudocolor
# plot of the variable; the plots exist only so Lineout has something to
# sample -- the saved image comes from the curve window.
n_states = []
for src in SOURCES:
    db = os.path.join(src["dir"], src["pattern"]) + " database"
    n_files = len(glob.glob(os.path.join(src["dir"], src["pattern"])))
    if n_files == 0:
        raise SystemExit("No files match %s/%s" % (src["dir"], src["pattern"]))
    n_states.append(n_files)
    if OpenDatabase(db) != 1:
        raise SystemExit("VisIt could not open %s" % db)
    AddPlot("Pseudocolor", VARIABLE)
DrawPlots()

# One time slider driving all three databases in lockstep, state index by
# state index. Iterate only as far as the shortest run so a frame never
# mixes timesteps.
db_names = [os.path.join(s["dir"], s["pattern"]) + " database" for s in SOURCES]
CreateDatabaseCorrelation("all_sources", db_names, 0)  # 0 = IndexForIndex
SetActiveTimeSlider("all_sources")
n_frames = min(n_states)

# Lineout endpoints spanning the reference solution's full x extent at y=Y_VALUE.
SetActiveWindow(1)
SetActivePlots(0)
Query("SpatialExtents")
ext = GetQueryOutputValue()  # (xmin, xmax, ymin, ymax, ...)
p0 = (ext[0], Y_VALUE)
p1 = (ext[1], Y_VALUE)

save_atts = SaveWindowAttributes()
save_atts.format = save_atts.PNG
save_atts.outputToCurrentDirectory = 0
save_atts.outputDirectory = OUT_DIR
save_atts.family = 0
save_atts.width = IMAGE_WIDTH
save_atts.height = IMAGE_HEIGHT

annotations_made = False
fixed_view = None

for state in range(n_frames):
    SetActiveWindow(1)
    SetTimeSliderState(state)

    # One lineout per source, in SOURCES order -- each appends a Curve plot
    # to window 2, so curve index j in window 2 corresponds to SOURCES[j].
    for j in range(len(SOURCES)):
        SetActiveWindow(1)
        SetActivePlots(j)
        Lineout(p0, p1, (VARIABLE,))

    SetActiveWindow(2)
    for j, src in enumerate(SOURCES):
        SetActivePlots(j)
        c = CurveAttributes()
        c.curveColorSource = c.Custom
        c.curveColor = src["color"]
        c.lineWidth = src["line_width"]
        c.showLegend = 0
        c.showLabels = 0
        SetPlotOptions(c)
    DrawPlots()

    if not annotations_made:
        # Color-matched labels standing in for the (hidden) per-curve legends.
        y_pos = 0.90
        for src in SOURCES:
            t = CreateAnnotationObject("Text2D")
            t.text = src["label"]
            t.position = (0.72, y_pos)
            t.height = 0.03
            t.useForegroundForTextColor = 0
            t.textColor = src["color"]
            y_pos -= 0.05
        annotations_made = True

    # Lock the axes to the first frame's view so the y-scale doesn't jump
    # from frame to frame (t=0's step IC already spans the density range),
    # padded vertically so the min/max plateaus don't sit on the plot border.
    if fixed_view is None:
        fixed_view = GetViewCurve()
        r0, r1 = fixed_view.rangeCoords
        pad = 0.10 * (r1 - r0)
        fixed_view.rangeCoords = (r0 - pad, r1 + pad)
    SetViewCurve(fixed_view)

    save_atts.fileName = "density_y05_%04d" % state
    SetSaveWindowAttributes(save_atts)
    SaveWindow()

    DeleteAllPlots()  # clear window 2's curves before the next frame

print("Wrote %d frames to %s" % (n_frames, OUT_DIR))
