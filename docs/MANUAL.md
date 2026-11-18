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
  notes/       optional: your own Markdown notes; if there are any, the app shows them in Docs → Notes
```

The Home page lists these folders and lets you choose another working folder; the change takes effect
after a restart. For one start only: `./run.sh --work FOLDER`. Nothing is moved when you switch: each working
folder has its own project.

## Finding your way

The menu on the left has six sections. Each section has its pages as tabs across the top.

| Section | Pages | What they are for |
|---|---|---|
| **Home** | – | where the project stands, what to do next, where the data is kept |
| **Dataset** | Photos · Subsets · 3D datasets | what the experiments run on: photos, chosen subsets, rendered 3D faces |
| **Mooney** | Batch · By hand | making the Mooneys: all at once, or one photo at a time |
| **Experiments** | Vision systems · One-button run · Queue · Saved runs | running the systems, whole runs, runs in a row, and what was kept |
| **Analysis** | Results · By photo properties · Compare runs · 3D comparison | reading the outcome |
| **Docs** | Manual · Vision systems · References · Notes | this manual, where every value comes from, sources and credits, your notes |

The strip of counters under the title shows where the project stands, on every page.

The usual order of work:

| Step | Where | What happens | What is written |
|---|---|---|---|
| 1 | Dataset → Photos | Collect portraits, look through them, remove bad ones | `Original/`, `Raw/`, `dataset.csv` |
| 2 | Mooney → Batch | One Mooney per photo, same values for all | `Mooney/`, `params.csv` |
| 3 | Mooney → By hand | Adjust single photos, mark favourites | `Mooney/`, `params.csv`, `favorites.json` |
| 4 | Dataset → Subsets | Choose pairs from your favourites into a named subset | `Subset/`, `Subsets/<name>/`, `subsets.json` |
| 5 | Experiments → Vision systems | The ticked systems run on every original and every Mooney | `Result1/` … `Result4/`, `analysis/` |
| 6 | Analysis | Summary, figures, analysis by photo properties | `figures/` |
| 7 | Dataset → 3D datasets, then 5 and 6 again | The same on rendered 3D faces with known light and depth | `3D/`, `3D_basel_fit/`, `3D_basel/` |
| 8 | Experiments | Save the state as a run; queue several runs; compare them | `runs/` |

**Experiments → One-button run** does steps 2, 5, 6 and 7 in one go.

### Long jobs: one at a time, and how to stop one

Collecting photos, Batch, the vision systems, a 3D step, a one-button run and the queue are long jobs. Only one
of them works at a time; a second one is refused with a line in its log. Each has a **Stop** button next to
its start button. After Stop the job ends at the image it is working on, and the log says
`STOPPED BY YOU`. Files written up to that point stay as they are, so the step is only partly done (for
example Mooneys or results up to that photo); running the step again completes it. A stopped Batch still
records in `params.csv` what it made. A stopped one-button run is not saved as a run.

## Dataset → Photos

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

## Dataset → Subsets

A subset is a chosen set of pairs (original and Mooney) with a name and a size. You can keep several.

1. Open a saved subset at the top, or type a new name. Set its size (10 unless you change it).
2. The left side shows your favourites; switch *Show* to *All Mooneys* to choose from everything. Click a
   Mooney to add it (it gets a ✓), click it again to take it out. The right side shows what is in the subset,
   as Mooneys and as originals. *Fill up from favourites* adds favourites until the size is reached.
3. *Save subset* writes the pairs to `Subsets/<name>/Original` and `Subsets/<name>/Mooney`.

One subset is the project's subset: it is also written to `Subset/Original` and `Subset/Mooney`, and the
counter at the top shows it. Any saved subset can be chosen wherever a step runs on part of the photos
(system 2, the photo-based 3D datasets). Tick *Make it the project's subset*
when saving to choose which one that is. Nothing is saved until you press *Save subset*; the line above the
pictures says whether the choice on screen differs from the saved one.

## Dataset → 3D datasets

A controlled version of the experiment. A photo tells us nothing certain about the light or the shape of the
face. A picture we draw ourselves does: we choose the light, and we know the depth of every pixel.

This page only makes a 3D dataset: the renders and their Mooneys. The systems run on it in
**Experiments → Vision systems**; the comparison with the truth is in **Analysis → 3D comparison**.

**Three sources of 3D faces.** Choose one at the top of the page. Each source has one live folder; the sets
are kept side by side.

| Source | Folder | What the faces are | What it is for |
|---|---|---|---|
| Faces reconstructed from your photos | `3D/` | a mesh estimated from one photo of a real person; face region only, one skin colour | your real people, with the pose of the photo |
| Basel Face Model fitted to your photos | `3D_basel_fit/` | the Basel model fitted to the photo's landmarks: that person's overall proportions on a whole Basel head, skin tone matched, cast shadows | your real people, drawn like the invented faces |
| Invented faces from the Basel Face Model | `3D_basel/` | random samples of a statistical face model; whole head with skin colour, no hair, no real person | perfect truth: nothing is estimated |

**Steps**, for the source chosen:

1. *Render.* The faces are drawn as a matte surface under each chosen light: front, left 45°, right 45°, top,
   below. For *faces reconstructed from your photos* a 3D face mesh is estimated for each chosen photo
   (MediaPipe, 478 landmarks, triangulated, smoothed) and the head pose is the one in the photo.
2. *Make Mooneys.* The renders go through the same generator as the photos, with the values saved as defaults
   in Mooney → Batch.

*Render and make Mooneys* does both. *Render* replaces the earlier renders of that source together with their
Mooneys and results, and *Make Mooneys* removes earlier system results of that 3D set, so a live folder always
holds one consistent set. Save the dataset first if you want to keep it.

**Which photos.** The suggestion is the favourites in their order, one photo per person, filled up from the
dataset with the most frontal photos (ten people by default). Strongly turned heads are skipped, because the
simple mesh gets holes; the log says so.

For the two photo-based sources, *Take the photos from* offers the suggestion, all photos, your favourites or
any saved subset; single photos can still be added or removed.

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

**Saved 3D datasets.** *Save 3D dataset* copies the live folder of the chosen source, with everything in it
(renders, true depth, Mooneys, any system results), to `3D_sets/<name>/` and lists it in the table. *Load* puts
a saved dataset back as the live folder of its source, replacing what is there (the page says what it
replaced); the saved copy stays. *Delete* removes a saved copy. Saving and loading copy files as they are; no
number changes.

### Invented faces from the Basel Face Model

The Basel Face Model 2019 describes face shape and skin colour statistically: a face is the model's mean plus a
weighted sum of its principal components. Drawing the weights at random gives a face that looks plausible and
belongs to nobody.

**Getting the model.** It is not part of Mooney Studio and may not be passed on. Register at
<https://faces.dmi.unibas.ch/bfm/bfm2019.html> (free for non-commercial research), download
`model2019_fullHead.h5` and put it in `models/`. The package `h5py` must be installed (`pip install h5py`; it is
in `requirements.txt`). While either is missing the page says so and the rest of the app works as before.

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

### The Basel Face Model fitted to your photos

The same model, but instead of random weights the shape weights and the head pose are fitted to the photo
(`bfm_fit.py`). It needs the model file and `h5py`, like the invented faces.

**How the fit works.** MediaPipe finds the face's landmarks in the photo. 19 of them have a counterpart among
the landmarks stored in the model file: eye corners and lower lids, nose tip, nose wings and the point below
the nose, philtrum, the outer lip line and mouth corners, chin. The fit looks for the head pose, the scale and
the first 80 shape weights whose landmarks land where MediaPipe found them, while keeping the weights as likely
as the model allows (an expected landmark error of 1.5 mm). It alternates the two parts eight times and takes
under a second.

| Setting | Default | Where |
|---|---|---|
| Photos | the project's subset (else the favourites) | *Take the photos from* on the page; any set can be chosen |
| Head pose of the fitted head | frontal | *as in the photo* draws the head in the pose the fit found |
| Model file, cast shadows, lights | as for the invented faces | the page |
| Shape components, expected landmark error, skip threshold, tone matching | 80, 1.5 mm, 15 % of the eye distance, on | `config.py`: `BASEL_FIT_*` |

**Check the fits.** For every photo `3D_basel_fit/truth/<photo>__fit.png` shows the photo with the outline of
the fitted head (white), the landmarks MediaPipe found (green) and the fitted ones (red). The log gives the
mean landmark error in pixels and as a share of the eye distance; photos above the threshold are skipped.

**What is written.** The layout of `3D/`, with `truth/<photo>__fit.png`, `truth/<photo>__coefficients.npy`
(the fitted weights), `truth/<photo>__texture.png` and, in `truth/renders.csv`, the landmark error and the
tone gain of every fit.

**What the truth is here.** As for the reconstructed faces: exact for the render, an estimate for the person.
Landmarks fix the overall proportions (face width, nose length, position of eyes and mouth) but not fine
detail, so the fitted heads stay close to the model's average face and look more alike than the people do.
The skin has one colour matched to the photo; there is no hair. *Similarity render to photo* is calculated,
with the render aligned on its own landmarks.

## Mooney → Batch

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
| Keep Mooneys adjusted by hand | on | Batch skips photos saved in *By hand* |
| Sweep | off | try several values per photo and keep the one with the highest quality score |

*Save these values as defaults* stores the settings in the project's `settings.json`.

`params.csv` has one row per Mooney: the values used, the white share actually reached, the threshold, the
face brightness, and `mode` (`batch` or `manual`).

**Quality score** (used only by Sweep): fidelity × simplicity. Fidelity is how well the black/white borders
of the Mooney follow the strong edges of the photo; simplicity rewards few, large regions.

## Mooney → By hand

One photo at a time, with all four phases visible. Use *Previous* and *Next* to step through the folder.
Changing a slider redraws the Mooney at once. The lines under the buttons show the white share reached, the
threshold and the face brightness, and whether what you see is what is saved for this photo.

A change affects the photo on screen only, and nothing is written until you press *Save this Mooney*. When you
move to another photo, the sliders jump to the values that photo's own saved Mooney was made with (or to the
defaults if it has none), so an adjustment is never carried over to the next photo.

- *Save this Mooney* writes the image to `Mooney/` and its values to `params.csv` with `mode = manual`.
  Batch leaves such photos alone.
- *Favourite* marks the photo as a candidate for a subset.
- *Back to defaults* returns the sliders to the saved defaults.

## Experiments → Vision systems

Choose what to run on, tick the systems, press *Run the ticked systems*. They run one after another with one
log.

- **Run on.** *Photos* is the main project. A 3D dataset is offered once it has Mooneys; on it systems 1, 3
  and 4 can run, and afterwards the depth maps are compared with the true depth.
- **Systems.** Every system takes each original (or render) and its Mooney and returns values for both.
  Nothing about who the person is is asked at this stage.

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
  in `analysis/result2_vlm_protocol.json`. An earlier answer is reused only if both are unchanged and the image is not newer than the answer.

- **Raw values and settings are kept.** System 1 stores the detector's box, landmarks and confidence and the
  128-number embedding for every image. Every system writes its settings to
  `analysis/resultN_*_protocol.json`. **Docs → Vision systems** (the file `docs/VISION_SYSTEMS.md`) explains for each
  value whether it comes straight from a model, is one of our settings, or is calculated by us.

Below the log the page shows the results of the chosen dataset: the summary table and figure for the photos,
the table by light direction against the truth for a 3D dataset.

**Save this state as a run** freezes the current state under a name (see *Saved runs* below).

**Settings of the systems** (a fold-out):

- *Check which systems are ready* shows whether each model file is there and whether Ollama answers.
- Ollama, the program that serves the model of system 2: the status line says whether it is installed,
  whether its server answers and which models it has. *Start Ollama* starts the server in the background if the
  program is installed; *Download this model* fetches a model by name (for example `llava:7b`) with the progress
  in the log. Nothing else can be started from the app.
- System 2: the Ollama model, the prompt, and the set of photos it runs on (all photos, favourites or a saved
  subset). It is heavy, so a subset is the usual choice; systems 1, 3 and 4 always run on every photo.
- *Identity matching* (a later step, off by default): system 1 ranks each image against the other photos,
  system 2 is asked for the name.
- System 5, the 3D face mesh (MediaPipe), is an optional later step; tick it in the list of systems.

## Experiments → One-button run

The whole workflow with one button: Mooneys, vision systems, 3D faces, figures, and a saved copy.

1. Give the run a name and, if you like, a note.
2. Tick the steps and set their values:
   - **Make the Mooneys**: blur, white share, background and phase-1 gray. The values become the project's
     defaults. Mooneys you adjusted by hand are kept. Untick it to use the Mooneys as they are.
   - **Vision systems**: any of the four. System 2 is skipped if Ollama is not running.
   - **3D faces**: any of the three sources, with the default lights; each is rendered, turned into Mooneys
     and run through systems 1, 3 and 4.
   - **Which photos:** system 2 and the photo-based 3D sources each take a set: all photos, your favourites
     or any saved subset. Systems 1, 3 and 4 always run on every photo, so their tables stay complete.
   - **Export figures** and **Save the run**, with a choice of what the saved copy holds.
3. Press **Run**, or **Add to queue**. The log names each step as it starts; a step that cannot run says why
   and the rest goes on.

Below the log the page shows the summary table and figure of the current state and the photos and the 3D sets
side by side. *Only save the current state as a run* saves without running anything. Every step calls the same
code as its own page, so the two ways give the same results.

## Experiments → Queue

Runs that go one after another without you, for example overnight.

1. Set a run up in *One-button run* and press *Add to queue*. The queue keeps the values of the form as they
   are at that moment. A queued run needs a name and *Save the run* ticked.
2. Open *Queue* and press *Start queue*. The table shows each run's state (waiting, running, done, failed,
   stopped), when it started, how long it took and its last message; the log of the run that is working is
   shown below and refreshes itself.

- A run that fails is marked *failed* with its error, and the next one starts.
- *Stop after the current run* lets the run that is working finish. *Stop now* ends it at the image it is
  working on: it is marked *stopped*, it is not saved, and the runs after it stay waiting.
- *Move up*, *Move down* and *Remove* work on waiting items. *Clear finished* takes finished items off the
  list; their saved runs are untouched.
- The queue is kept in `queue.json` in the project, so it is still there after a restart. The app must stay
  open and the computer awake while the queue runs.

Every run is frozen in its own saved copy. The project itself is one folder, so after the queue it holds the
Mooneys and results of the **last** run; put the run you want to go on working with last.

## Experiments → Saved runs

A run is a frozen copy of the project's state under a name, in `runs/<date>_<name>/`. It can be saved from
*Vision systems*, from *One-button run*, and by the queue.

- The Mooneys, `params.csv`, `settings.json`, `favorites.json`, `subsets.json` and `dataset.csv` are always
  saved. You choose the rest: vision system results and tables, exported figures, the subsets, the three 3D
  folders, and the original photos (off by default).
- `run.json` records the values the Mooneys were made with, the counts, the date, what was included, the note
  and the list of photos. Saving twice under one name keeps both.
- A run without the result tables cannot be used in *Compare runs*.

The page lists every saved run, newest first. Choosing one shows what it was made with and its summary table
and figure, read from the run's folder; nothing in the run is changed.

**Working with runs.** Save the run, change the settings in Mooney → Batch, generate again, run the systems
again, save under another name; or queue the variants. Then compare them in Analysis → Compare runs.

## Analysis → Results

- **Summary table**: one row per measure, with columns Original, Mooney and Mooney re-blurred. Counts are
  images; other values are means.
- **Figure**: the counted measures as bars.
- **Viewer**: for one photo, the original and the Mooney next to each system's output, with the values below.
- **Export figures** writes the table, the figure and one panel per favourite to `figures/`.

## Analysis → By photo properties

This page relates what the systems returned to properties of the photos. It runs no model; it only
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

"Hard" and "easy" here mean hard and easy for the models. Nothing on this page measures what people see.

## Analysis → Compare runs

Choose two or more saved runs; *current* stands for the project as it is now. The first
table is the results summary with one column per run. The second shows what each run's Mooneys were made
with, read from the run's `params.csv`: blur, white target, white share reached, background, preprocessing,
how many were adjusted by hand. Check there that the runs cover the same photos before reading differences.

## Analysis → 3D comparison

The systems against the truth, for the 3D dataset of the source chosen at the top of the page.

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

*Export* writes the table, the chart and one sheet per face to the `figures/` folder of that 3D set.

**What the truth is, and is not.**

- Light direction and depth are exact for the render, because the picture is drawn from that mesh under that
  light.
- They are not the true shape of the real person. The mesh is an estimate from one photo.
- The renders show the face region only, without hair, in one skin colour. The low similarity between a
  render and its photo reflects that.
- The depth comparison uses the saved depth pictures (8-bit, clipped at the 1st and 99th percentile).

### Compare stimulus sets

At the bottom of the page. One table with one row per measure and one column per set: **Photos** (the main
project), **Reconstructed 3D** (`3D/`), **Basel fitted 3D** (`3D_basel_fit/`), **Basel 3D** (`3D_basel/`).
Every cell carries its own n.

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
with the true depth (bars) and the Mooney-against-original value (diamonds). The second table lists the 3D
sets by light direction, one row per light and set. *Export comparison to figures/* writes both tables and
the chart to `project/figures/` (`compare_sets.csv`, `compare_sets_by_light.csv`, `compare_sets.png`).

The sets differ in more than the source of the face: the photos have hair, background and unknown light; the
reconstructed faces have no skin colour; the Basel faces (fitted and invented) have skin colour, a whole head and cast shadows. A
difference between columns can come from any of these.

## Docs

- **Manual**: this text (`docs/MANUAL.md`).
- **Vision systems**: for every value, whether it comes straight from a model, is one of our settings, or is
  calculated by us (`docs/VISION_SYSTEMS.md`).
- **References**: the author, the literature the method is built on, the photo credits of your project and
  the use of AI (`docs/REFERENCES.md`). The photo table is built from `dataset.csv`: author, licence and source
  page of every photo. Photos you add by hand can be credited in a file `credits.csv` in the project folder
  with the columns `file, author, licence, source`; its rows are listed too.
- **Notes**: Markdown files in `<working folder>/notes/`, one note at a time. A line that holds only an image,
  `![caption](pictures/sheet.png)`, is shown as a picture; the path is read relative to the notes folder. The
  notes are read when the app starts. The page is absent while the folder has no Markdown file.

Changing the code with a coding agent, or by hand: read `AGENTS.md` in the code folder first.

## Folders

```
project/
  Original/            512 px crops
  Mooney/              one PNG per photo
  Subset/Original, Subset/Mooney    the project's subset
  Subsets/<name>/                   further named subsets, same two folders each
  Result1..4/Original, /Mooney      one file per image and vision system
  Raw/                 uncropped photos
  analysis/            one table per system
  figures/             exported table, figure, panels
  dataset.csv          licence and source of every photo
  params.csv           values used for every Mooney
  settings.json        saved defaults
  favorites.json       favourites
  identities.txt       the names last entered in Dataset → Photos
  subsets.json         the named subsets
  queue.json           the run queue
  rejected.csv         photos removed by hand
  _removed/            the removed photos themselves
  3D/                  3D faces reconstructed from photos: same layout again, plus truth/
  3D_basel_fit/        the Basel Face Model fitted to your photos: same layout as 3D/
  3D_basel/            invented 3D faces from the Basel Face Model: same layout as 3D/
  3D_sets/<name>/      saved 3D datasets: set.json and a copy of one of the three folders above in data/
runs/                  saved runs: <date>_<name>/ with Mooneys, results and run.json
```

Model files are the only data kept next to the code, in `models/`. They are downloaded the first time a step
needs them.

## Running it

```
./setup.sh        once: creates the environment and installs the packages
./run.sh          starts the app
```

By hand: `conda activate mooney` (or `source .venv/bin/activate`), `pip install -r requirements.txt`,
`python app.py`.

For system 2, Ollama must be running with a vision model. In a terminal:

```
ollama serve
ollama pull llava:7b
```

**Settings.** All defaults are in `config.py`. To change some, copy `config.example.json` to `config.json`
and keep only the lines you need; `python config.py` lists every setting. The values you save in Mooney → Batch are
stored per project in `settings.json` and take precedence over the generation defaults in `config.py`.

`README.md` covers installation, the working folder and making the app reachable from another computer.

## Limits to keep in mind

- Every measure in the app is computed by a model, not by people. Whether a Mooney is recognisable to humans
  is not measured anywhere in the tool.
- Photos of public figures on Commons are mostly event and press photos with flat lighting, which is not
  the lighting classic Mooney faces were made from. The 3D datasets add directional light, but on
  simplified faces: reconstructed ones without hair or skin texture, or invented Basel heads without hair.
- Collected photos keep their own licences (recorded in `dataset.csv`). Check them before you publish any
  picture.
