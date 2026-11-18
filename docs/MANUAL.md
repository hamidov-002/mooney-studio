# Mooney Studio: manual

Mooney Studio is a workbench for two-tone "Mooney" face images: build a set of face photos, turn each into a
Mooney image, run vision systems on both versions, and compare what they return.

## Where your data is kept

The app keeps nothing of yours inside its own folder. Everything goes to the **working folder**
(`~/MooneyStudio` unless you choose another one):

```
<working folder>/
  project/     photos, Mooneys, results, settings (layout under "Folders" below)
  runs/        saved runs
  notes/       optional: your own Markdown notes; if there are any, the app shows them in a Notes tab
```

The Overview page lists these folders and lets you choose another working folder; the change takes effect
after a restart. For one start only: `./run.sh --work FOLDER`. Nothing is moved when you switch: each working
folder has its own project.

## The workflow

| Step | Tab | What happens | What is written |
|---|---|---|---|
| 1 | Dataset | Collect portraits, look through them, remove bad ones | `Original/`, `Raw/`, `dataset.csv` |
| 2 | Batch | One Mooney per photo, same values for all | `Mooney/`, `params.csv` |
| 3 | Mooney | Adjust single photos by hand, mark favourites | `Mooney/`, `params.csv`, `favorites.json` |
| 4 | Vision systems | Each system runs on every original and every Mooney | `Result1/` … `Result4/`, `analysis/` |
| 5 | Results | Summary table, figure, per-photo viewer, export | `figures/` |
| – | Analysis | Results per photo, by group, agreement between systems, ranking; comparison of saved runs | `figures/analysis_*` |
| 6 | 3D faces | Renders with known light and depth, their Mooneys, systems against the truth; comparison of the stimulus sets | `3D/`, `3D_basel/` |

The strip of counters under the title shows where the project stands on every tab. The Overview tab says
what to do next.

## 1 Dataset

Photos come from Wikimedia Commons. For each name the collector looks at up to 80 files and keeps a photo
only if all of these hold:

- the licence is open (CC0, CC BY, CC BY-SA or public domain);
- the file title contains the person's first name and surname, so a relative with the same surname is not
  taken by mistake;
- it is a photograph, not a painting, drawing or statue;
- exactly one face is found;
- the crop from hair to collar lies inside the photo (no padding) and is at least 480 px wide (no upscaling);
- the face is sharp, neither too dark nor too bright, and has contrast;
- the photos of one person come from different occasions (a different day, or a different photographer or
  photo series);
- the photo is not already used for someone else.

Among the photos that pass, the ones with the most different head poses are chosen.

Each kept photo is saved twice: uncropped in `Raw/` and as a 512 × 512 px crop in `Original/`. Licence,
author, source page, date, head turn and sharpness go to `dataset.csv`.

**Reviewing.** Click a photo to select it and press *Remove selected photo*. The photo moves to `_removed/`,
its Mooney and result files are deleted, and its source page is added to `rejected.csv` so it is never
collected again. Press Collect to fetch a replacement; people who already have all their photos are skipped,
and a person with one photo left gets only the missing one.

**Reading the log.** One line per person: `+` photos added, `=` already complete, `-` skipped (with the
reasons), `!` Commons did not answer. A summary line closes the run.

## 2 Batch

The generator has four phases:

1. **Background separation.** U²-Net finds the person; the background is replaced by one gray value.
2. **Grayscale.**
3. **Gaussian blur** with standard deviation σ in pixels.
4. **Threshold.** The threshold is set so that a chosen share of the face is white. Then the background
   of the result is painted black or white.

Main settings and their defaults:

| Setting | Default | Meaning |
|---|---|---|
| Blur σ | 6 px | larger = fewer, smoother shapes; harder to recognise |
| White | 23 % of the face | share of the person's pixels inside the face box that end up white |
| Background of the Mooney | black | colour outside the person |
| Phase-1 gray | 146 | gray value given to the background before blurring; it only changes the outline of the person (0 gives no outline with a black background) |

Optional settings, all off or at a neutral default:

| Setting | Default | Meaning |
|---|---|---|
| White share | same for every photo | *By face brightness* gives bright faces a larger white share |
| White for bright faces | 35 % | used only with *By face brightness* |
| A face is bright from brightness | 124 | mean gray value of the face (0–255) from which a face counts as bright |
| Preprocessing | none | *clahe* equalises local contrast before the blur |
| White share is counted on | face | *whole person* includes hair and clothing |
| Separate the background | on | off = threshold the whole photo |
| Keep Mooneys adjusted by hand | on | Batch skips photos saved by hand in the Mooney tab |
| Sweep | off | try several values per photo and keep the one with the highest quality score |

*Save these values as defaults* stores the settings in the project's `settings.json`.

`params.csv` has one row per Mooney: the values used, the white share actually reached, the threshold, the
face brightness, and `mode` (`batch` or `manual`).

**Quality score** (used only by Sweep): fidelity × simplicity. Fidelity is how well the black/white borders
of the Mooney follow the strong edges of the photo; simplicity rewards few, large regions.

## 3 Mooney

One photo at a time, with all four phases visible. Use *Previous* and *Next* to step through the folder.
Changing a slider redraws the Mooney at once. The line under the buttons shows the white share reached, the
threshold and the face brightness.

- *Save this Mooney* writes the image to `Mooney/` and its values to `params.csv` with `mode = manual`.
  Batch leaves such photos alone.
- *Favourite* marks the photo as a candidate for the subset of ten.
- *Back to defaults* returns the sliders to the saved defaults.

When ten favourites are marked, press *Save favourites to Subset/* in the Batch tab. The pairs are copied to
`Subset/Original` and `Subset/Mooney`.

## 4 Vision systems

Every system takes each original and its Mooney and returns values for both. Nothing about who the person
is is asked at this stage.

| System | Model | Values per image | Files |
|---|---|---|---|
| 1 Face model | YuNet detector, SFace embedding (OpenCV) | face detected, detector score; for the Mooney also the similarity of its embedding to the original's | `Result1/*.json`, `analysis/result1_face.csv` |
| 2 Vision-language model | a local model through Ollama | face present, head pose, light direction, one-sentence description | `Result2/*.txt`, `analysis/result2_vlm.csv` |
| 3 Learned depth | Depth Anything V2, ViT-S (a trained network) | a depth map; for the Mooney its correlation with the original's depth map | `Result3/*.png`, `analysis/result3_depth.csv` |
| 4 Depth from shading | shape from shading, Tsai & Shah 1994 (a formula, nothing learned) | the same two values as system 3 | `Result4/*.png`, `analysis/result4_shading.csv` |

Notes:

- **Alignment in system 1.** The detector does not find faces in Mooneys, so the Mooney is aligned with the
  landmarks found in its original before the embedding is computed.
- **Re-blurred Mooney.** System 1 also reports every value for the Mooney blurred once more (Gaussian,
  σ = 8 px at 512 px). This adds no information to the image; it only removes the hard edges.
- **System 2 protocol.** The same prompt for every image, temperature 0, seed 0. Model and prompt are saved
  in `analysis/result2_vlm_protocol.json`. Earlier answers are reused only if both are unchanged.

- **Raw values and settings are kept.** System 1 stores the detector's box, landmarks and confidence and the
  128-number embedding for every image. Every system writes its settings to
  `analysis/resultN_*_protocol.json`. `docs/VISION_SYSTEMS.md` (also in the Vision systems tab) explains for each
  value whether it comes straight from a model, is one of our settings, or is calculated by us.

Under *Later steps*: identity matching (does the system recognise the person?) and system 5, the 3D face
mesh. Both are switched off by default. The controlled 3D experiment has its own tab, *6 3D faces*.

## 5 Results

- **Summary table**: one row per measure, with columns Original, Mooney and Mooney re-blurred. Counts are
  images; other values are means.
- **Figure**: the counted measures as bars.
- **Viewer**: for one photo, the original and the Mooney next to each system's output, with the values below.
- **Export figures** writes the table, the figure and one panel per favourite to `figures/`.
- **Save this run as…** freezes the current state under a name. It copies `Mooney/`, `params.csv`,
  `Result1/` to `Result4/`, `analysis/`, `settings.json`, `favorites.json` and `dataset.csv` to
  `runs/<date>_<name>/` and writes `run.json` there (the values the Mooneys were made with, counts, date, the
  list of photos). The photos themselves are not copied. Saving twice under one name keeps both.

**Working with runs.** Save the run, change the settings in 2 Batch, generate again, run the systems again,
save under another name. Then compare the runs in the Analysis tab.

## Analysis

The Analysis tab relates what the systems returned to properties of the photos. It runs no model; it only
reads `dataset.csv`, `params.csv` and the tables in `analysis/`. Each photo is one data point.

| Part | What it shows |
|---|---|
| The dataset | photos, people, licences; how face brightness and head turn are distributed |
| Each system: original versus Mooney | per-photo distributions of every value, and counts for the yes/no results |
| By group | the same results for groups of photos: by face brightness (three equal groups), head turn, how the Mooney was made, or the white share reached |
| Do the systems agree? | correlation across photos between every pair of results, and a scatter of the two depth systems |
| Hardest and easiest photos | Mooneys ranked by the average of their ranks on all available results |
| In words | sentences generated from the numbers, each with the number of photos behind it |

Every number is shown with the number of photos it rests on, and groups with fewer than 10 photos are
marked. *Export analysis to figures/* writes the figures as `analysis_*.png` and the tables as
`analysis_*.csv`, including one row per photo with all values (`analysis_per_photo.csv`).

"Hard" and "easy" here mean hard and easy for the models. Nothing in this tab measures what people see.

**Compare runs.** Choose two or more saved runs; *current* stands for the project as it is now. The first
table is the Results summary with one column per run. The second shows what each run's Mooneys were made
with, read from the run's `params.csv`: blur, white target, white share reached, background, preprocessing,
how many were adjusted by hand. Check there that the runs cover the same photos before reading differences.

## 6 3D faces

A controlled version of the experiment. A photo tells us nothing certain about the light or the shape of the
face. A picture we draw ourselves does: we choose the light, and we know the depth of every pixel.

**Two sources of 3D faces.** Choose one at the top of the tab; the three step buttons, the tables, the chart,
the sheet and the export then work on that source. The two sets are kept side by side.

| Source | Folder | What the faces are |
|---|---|---|
| Faces reconstructed from your photos | `3D/` | a mesh estimated from one photo of a real person; face region only, one skin colour |
| Invented faces from the Basel Face Model | `3D_basel/` | random samples of a statistical face model; whole head with skin colour, no hair, no real person |

The first source is described next, the Basel source after it.

**Steps.**

1. *Render.* For each chosen photo a 3D face mesh is estimated (MediaPipe, 478 landmarks, triangulated,
   smoothed) and drawn as a matte surface under each chosen light: front, left 45°, right 45°, top, below.
   The head pose is the one in the photo.
2. *Make Mooneys.* The renders go through the same generator as the photos, with the values saved as defaults
   in 2 Batch.
3. *Run systems 1, 3 and 4* on the renders and their Mooneys, then compare with the truth.

*Run all three steps* does them in order. Step 1 replaces earlier renders together with their Mooneys and
results, and step 2 removes earlier results of step 3, so the folder always holds one consistent set. To keep
the outcome of a setting, export it to `3D/figures/` first.

**Which photos.** The suggestion is the favourites in their order, one photo per person, filled up from the
dataset with the most frontal photos (ten people by default). Strongly turned heads are skipped, because the
simple mesh gets holes; the log says so.

**What is written.** `3D/` has the layout of a project, so every tool works on it unchanged:

```
project/3D/
  Original/<photo>__<light>.png      the renders (they play the role of the originals)
  Mooney/                            their Mooneys
  Result1/, Result3/, Result4/       system outputs, as in the main project
  analysis/                          result tables, and truth_scores.csv
  truth/<photo>__depth.npy           true depth, larger = nearer, 0 outside the face
  truth/<photo>__depth.png, __mask.png, __mesh.png
  truth/renders.csv                  per render: source photo, person, light name and vector, head pose
  figures/                           exported table, chart and one sheet per face
```

**The table, one row per light direction.** `n` is the number of renders in the row.

| Column | Meaning |
|---|---|
| Face found: render / Mooney / re-blurred | images in which the detector finds a face |
| Similarity Mooney to render | face-model similarity of the Mooney to its render (−1 to 1) |
| Similarity render to photo | face-model similarity of the render to the photo it was made from |
| Learned depth vs truth: render / Mooney | correlation, inside the face, between system 3's depth map and the true depth |
| Shading depth vs truth: render / Mooney | the same for system 4 |

The chart shows the four depth columns by light, with one dot per render. The sheet below it shows one face:
photo and true depth, then render and Mooney under every light.

**What the truth is, and is not.**

- Light direction and depth are exact for the render, because the picture is drawn from that mesh under that
  light.
- They are not the true shape of the real person. The mesh is an estimate from one photo.
- The renders show the face region only, without hair, in one skin colour. The low similarity between a
  render and its photo reflects that.
- The depth comparison uses the saved depth pictures (8-bit, clipped at the 1st and 99th percentile).

### Invented faces from the Basel Face Model

The Basel Face Model 2019 describes face shape and skin colour statistically: a face is the model's mean plus a
weighted sum of its principal components. Drawing the weights at random gives a face that looks plausible and
belongs to nobody.

**Getting the model.** It is not part of Mooney Studio and may not be passed on. Register at
<https://faces.dmi.unibas.ch/bfm/bfm2019.html> (free for non-commercial research), download
`model2019_fullHead.h5` and put it in `models/`. The package `h5py` must be installed (`pip install h5py`; it is
in `requirements.txt`). While either is missing the tab says so and the rest of the app works as before.

**Choices.**

| Setting | Default | Meaning |
|---|---|---|
| Number of invented faces | 10 | seeds 1, 2, 3, …; a seed always gives the same face, named `basel-001`, `basel-002`, … |
| Model file | Whole head | *Whole head*: with ears, neck and skull. *Face only*: the face region (`model2019_face12.h5`) |
| Head pose | frontal | also *turned 30° left* and *turned 30° right*, as seen by the viewer; every chosen pose is drawn for every face |
| Cast shadows | on | parts of the face shade other parts (the nose shades the cheek); off = shading from the surface angle only |
| Light directions | the five | the same directions as for the reconstructed faces; the light stays fixed when the head turns |

Fixed in `config.py`: the coefficient scale (`BASEL_SCALE`, 0.9: the random weights are multiplied by it, so 0
would give the mean face and 1 the model's full variation) and the paths of the two model files.

**How the pictures are drawn** (`bfm.py`). Parallel projection at 512 px, the same scale in mm for every face.
Matte surface: brightness = skin colour × (0.1 + 0.9 × cosine of the angle between surface and light). A
z-buffer decides which surface point is seen in each pixel. For cast shadows the head is drawn once more as
seen from the light; a point is lit only if nothing lies between it and the light. The picture is computed at
twice the size and averaged down. One render takes well under a second.

**What is written.** The same layout as `3D/`:

```
project/3D_basel/
  Original/basel-001__<light>.png       the renders; a turned head is basel-001-left30__<light>.png
  Mooney/, Result1/, Result3/, Result4/, analysis/, figures/     as in 3D/
  truth/basel-001__depth.npy            true depth in mm, larger = nearer, 0 outside the head
  truth/basel-001__depth.png, __mask.png
  truth/basel-001__texture.png          the face without shading (skin colour only)
  truth/renders.csv                     per render: seed, model file, coefficient scale, light name and vector,
                                        head pose and turn in degrees, cast shadows on or off
```

The table by light has the same columns as above, without *Similarity render to photo*: there is no photo.
The sheet shows the face without shading and the true depth, then render and Mooney under every light.

**What the truth is here.** Shape, light and head pose are all set by us, so the depth is exact, and it is the
depth of the whole thing shown, not an estimate of somebody. The faces are invented, have no hair and no
glasses, and their skin has no shine. They vary less than real people do.

### Compare stimulus sets

At the bottom of the tab. One table with one row per measure and one column per set: **Photos** (the main
project), **Reconstructed 3D** (`3D/`), **Basel 3D** (`3D_basel/`). Every cell carries its own n.

| Row | Meaning |
|---|---|
| Image pairs | originals or renders that have a Mooney |
| Face found in the original / render, the Mooney, the re-blurred Mooney | images in which the detector finds a face, with the share |
| Similarity of the Mooney to its original / render | face-model similarity, mean |
| Learned depth / depth from shading: Mooney against original / render | correlation between a system's two depth maps, on the whole picture and on its central half |
| Learned depth / depth from shading against the true depth: render, Mooney | correlation with the true depth inside the face; 3D sets only, the photos have *no truth* |

Read the *Mooney against original / render* rows with care: in the 3D sets both pictures have the same black
background, so the whole-picture value mostly reflects the outline of the head. The central half is the fairer
comparison with the photos.

The chart puts the sets side by side: how often a face is found, and for each depth system the correlation
with the true depth (bars) and the Mooney-against-original value (diamonds). The second table lists the two 3D
sets by light direction, one row per light and set. *Export comparison to figures/* writes both tables and
the chart to `project/figures/` (`compare_sets.csv`, `compare_sets_by_light.csv`, `compare_sets.png`).

The sets differ in more than the source of the face: the photos have hair, background and unknown light; the
reconstructed faces have no skin colour; the Basel faces have skin colour, a whole head and cast shadows. A
difference between columns can come from any of these.

## Folders

```
project/
  Original/            512 px crops
  Mooney/              one PNG per photo
  Subset/Original, Subset/Mooney    the best ten pairs
  Result1..4/Original, /Mooney      one file per image and vision system
  Raw/                 uncropped photos
  analysis/            one table per system
  figures/             exported table, figure, panels
  dataset.csv          licence and source of every photo
  params.csv           values used for every Mooney
  settings.json        saved defaults
  favorites.json       favourites
  identities.txt       the names last entered in 1 Dataset
  rejected.csv         photos removed by hand
  _removed/            the removed photos themselves
  3D/                  3D faces reconstructed from photos: same layout again, plus truth/
  3D_basel/            invented 3D faces from the Basel Face Model: same layout as 3D/
runs/                  saved runs: <date>_<name>/ with Mooneys, results and run.json
```

Model files are the only data kept next to the code, in `models/`. They are downloaded the first time a step
needs them.

## Your own notes

Markdown files in `<working folder>/notes/` are shown in a **Notes** tab, one note at a time. A line that
holds only an image, `![caption](pictures/sheet.png)`, is shown as a picture; the path is read relative to the
notes folder. The notes are read when the app starts. The tab is absent while the folder has no Markdown file.

## Running it

```
./setup.sh        once: creates the environment and installs the packages
./run.sh          starts the app
```

By hand: `conda activate mooney` (or `source .venv/bin/activate`), `pip install -r requirements.txt`,
`python app.py`.

For system 2, Ollama must be running with a vision model:

```
ollama serve
ollama pull llava:7b
```

**Settings.** All defaults are in `config.py`. To change some, copy `config.example.json` to `config.json`
and keep only the lines you need; `python config.py` lists every setting. The values you save in 2 Batch are
stored per project in `settings.json` and take precedence over the generation defaults in `config.py`.

`README.md` covers installation, the working folder and making the app reachable from another computer.

## Limits to keep in mind

- Every measure in the app is computed by a model, not by people. Whether a Mooney is recognisable to humans
  is not measured anywhere in the tool.
- Photos of public figures on Commons are mostly event and press photos with flat lighting, which is not
  the lighting classic Mooney faces were made from. The 3D faces tab adds directional light, but on
  simplified faces: reconstructed ones without hair or skin texture, or invented Basel heads without hair.
- Collected photos keep their own licences (recorded in `dataset.csv`). Check them before you publish any
  picture.
