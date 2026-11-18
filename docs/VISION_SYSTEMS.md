# The vision systems: where every value comes from

For each system this page separates three things:

- **Model output**: what the model itself returns. We store it unchanged.
- **Our settings**: choices we made around the model. They are also written to
  `analysis/resultN_*_protocol.json` every time a system runs.
- **Calculated by us**: numbers we work out from the model output.

Every system gets the same two inputs per photo: the original (`Original/`) and its Mooney (`Mooney/`).

---

## 1 Face model

Two models from OpenCV work together: a detector (YuNet) and a recogniser (SFace).

### Model output

| Model | Returns | Stored in `Result1/.../<photo>.json` as |
|---|---|---|
| YuNet | a box around each face | `model_output.detector.box_x_y_w_h` |
| YuNet | five landmark points: both eyes, nose tip, both mouth corners | `model_output.detector.landmarks_x_y` |
| YuNet | a confidence between 0 and 1 | `model_output.detector.confidence` |
| SFace | an embedding: 128 numbers describing the face | `model_output.embedding_128` |

If the detector finds no face, `detector` is empty. The embedding has no meaning by itself; only the
comparison of two embeddings does.

### Our settings

| Setting | Value | Why |
|---|---|---|
| Detector input | image resized so its longer side is 640 px | the size the detector works best at |
| A face counts as detected from confidence | 0.6 | lenient on purpose (OpenCV's default is 0.9), so a weak detection in a Mooney is not thrown away |
| Several faces | the largest box is used | the collected photos show one person |
| Alignment of the original | its own five landmarks, to 112 × 112 px | what SFace expects |
| Alignment of the Mooney | the landmarks found in its **original** | the detector finds none in a Mooney; without this the embedding would describe a badly placed crop |
| Embedding | divided by its length | so that similarity depends on direction only |
| Re-blurred condition | Gaussian blur of the Mooney, σ = 8 px at 512 px | removes the hard edges without adding information |

### Calculated by us

| Value | How | Reading |
|---|---|---|
| `face_detected` | yes if the detector returned a face with confidence ≥ 0.6 | can a standard detector find a face at all? |
| `detector_score` | the model's confidence, copied | – |
| `similarity_to_own_original` | sum of the products of the two normalised embeddings (cosine similarity), −1 to 1 | near 0: the model sees nothing in common. Two photos of one person usually score above about 0.36 with this model |
| `reblur_face_detected`, `reblur_similarity_to_own_original` | the same two values for the re-blurred Mooney | if they recover, the hard edges were the obstacle, not missing information |

In the Results tab: the number of images with a face detected, and the mean similarity.

---

## 2 Vision-language model

A local model run through Ollama (for example llava:7b).

### Model output

One piece of text, saved exactly as it came back in `Result2/.../<photo>.txt`. We ask for four fields:
`face_present`, `head_pose`, `light_direction`, `description`.

These are the model's statements. There is no score or confidence behind them.

### Our settings

| Setting | Value | Why |
|---|---|---|
| Prompt | the same text for every image, saved in `analysis/result2_vlm_protocol.json` | so differences come from the image, not the question |
| Temperature | 0, seed 0 | the same image gives the same answer every time |
| Answer format | JSON, requested through Ollama | so the four fields can be read reliably |
| Image sent | PNG, longer side at most 768 px | – |
| Reuse | an earlier answer is reused only if model and prompt are unchanged | – |

### Calculated by us

| Value | How |
|---|---|
| The four columns in `analysis/result2_vlm.csv` | copied from the model's answer |
| `parsed` | whether the answer was valid JSON; if not, it counts as no answer |
| Share saying a face is present | number of answers with `face_present = yes`, divided by the number of images |

---

## 3 Learned depth

Depth Anything V2 (ViT-S): a neural network trained on millions of images to estimate depth from one image.

### Model output

One number per pixel: relative inverse depth. Larger means nearer. It has no unit, so it gives shape, not
distance in metres.

### Our settings

| Setting | Value |
|---|---|
| Model input | image resized to 518 × 518 px, scaled to 0–1, ImageNet mean and standard deviation removed |
| Model output | resized back to the image size |

### Calculated by us

| Value | How | Reading |
|---|---|---|
| The picture `Result3/.../<photo>.png` | raw values clipped to their 1st and 99th percentile, scaled to 0–255; bright = near | for looking at |
| `png_0_is_raw`, `png_255_is_raw`, `raw_min`, `raw_max` | the two percentiles and the range, per image | lets the picture be turned back into the model's values |
| `depth_correlation_original_vs_mooney` | Pearson correlation between all raw values of the Mooney's map and of its original's map | 1: the same 3D layout from both; 0: unrelated |
| `depth_correlation_centre_crop` | the same on the central half of the image | less influenced by the background |

The model never sees the two images together. The comparison is ours.

---

## 4 Depth from shading

Shape from shading after Tsai & Shah (1994). It is a formula, not a trained model, and it knows nothing
about faces.

### How it works

It assumes a matte surface of one colour, lit by one distant light. Under that assumption the brightness of
a point depends only on how the surface is tilted there relative to the light. The method writes the tilt as
the difference between neighbouring depth values, and then adjusts the depth step by step until the
brightness it predicts matches the brightness in the image.

### Model output

One number per pixel: relative depth, larger = nearer, no unit.

### Our settings

| Setting | Value | Why |
|---|---|---|
| Light direction | slant 45° from the viewing direction, tilt 90° (from above) | the usual "light comes from above" assumption |
| Working size | 256 px | – |
| Smoothing before | Gaussian, σ = 2 px | the method is sensitive to pixel noise |
| Brightness | divided by its maximum | one albedo is assumed |
| Iterations | 120 | – |

### Calculated by us

The same four values as for system 3, in `Result4/` and `analysis/result4_shading.csv`.

### What to expect, and why it is worth having

On real photos this method mostly reads brightness as nearness. Expect its depth to agree only weakly with
the learned depth inside the face, and its Mooney and original outputs to agree with each other simply
because both follow brightness.

That contrast is the point: system 3 brings knowledge about what heads look like; system 4 brings none.
Comparing the two shows how much of "seeing shape" in a Mooney comes from prior knowledge and how much from
the image.

---

## Later step: identity matching (ranking)

Switched off by default. When it is on:

- **System 1** compares each image's embedding with the embeddings of all originals of the *other* photos and
  sorts them by similarity. This is the only ranking in the project.
  - `rank_of_true_identity`: the position of the same person's other photo in that sorted list (1 = best).
  - `top1_correct`: whether that position is 1.
  - `afc8_accuracy`: how often the right person wins against seven randomly drawn others (200 draws).
  - Only people with two photos can be tested this way.
- **System 2** is also asked for the person's name, and we check whether the answer contains it.

## Later step: 3D face mesh (system 5)

MediaPipe Face Landmarker returns 478 3D landmarks. We report whether a mesh is obtained and how close it is
to the mesh from the original. Results go to `Result5/`.

## 3D faces: the systems against known truth

In the *6 3D faces* tab systems 1, 3 and 4 run unchanged on renders and on the Mooneys of those renders. All
values above keep their meaning, with "original" read as "render". Two values are added, both calculated by
us and stored in `3D/analysis/truth_scores.csv`:

| Value | How | Reading |
|---|---|---|
| `depth3_vs_truth_render`, `depth3_vs_truth_mooney`, and the same for `depth4` | Pearson correlation, over the pixels inside the face mask, between the system's saved depth picture and the true depth of the render | 1: the system's relief is the relief the picture was drawn from; 0: unrelated |
| `similarity_render_to_photo` | cosine similarity of the SFace embeddings of the render and of the photo it was made from, both aligned with the landmarks found in the photo | how much of the person the face model still finds in a face without hair, texture or skin colour |

The true depth is the depth of the mesh the render was drawn from (`3D/truth/<photo>__depth.npy`, larger =
nearer). It is exact for the render and an estimate for the real person. The depth pictures used here are
8-bit and clipped at the 1st and 99th percentile, as described for system 3.

System 4 assumes light from above. That matches the *top* render and not the others, so differences between
its rows are expected.

### The Basel set (`3D_basel/`)

The second source of 3D faces: random samples from the Basel Face Model 2019, drawn by `bfm.py`. Systems 1, 3
and 4 run on it unchanged, and the depth values against the truth are calculated exactly as above and stored in
`3D_basel/analysis/truth_scores.csv`. What differs:

| | Reconstructed faces (`3D/`) | Basel faces (`3D_basel/`) |
|---|---|---|
| Shape | estimated from one photo | set by us: the model's mean plus random components (seed and scale in `truth/renders.csv`) |
| True depth | exact for the render, an estimate for the person | exact; there is no person behind it |
| Light | set by us | set by us |
| Head pose | as in the photo | set by us: frontal, or turned 30° left or right |
| Cast shadows | none | drawn (can be switched off) |
| Surface | one skin colour | the model's skin colour per point, matte |
| Covers | the face region | the whole head with ears and neck, or the face region (face-only model file) |
| Hair | none | none |
| `similarity_render_to_photo` | calculated | empty: the faces are invented, there is no photo |

Our settings for these renders: parallel projection, 512 px, ambient light 0.1, coefficient scale 0.9, true
depth in mm relative to its mean over the head. The identity of a Basel face is `basel-<seed>`; turned views of
the same seed share it.

### Comparing the three stimulus sets

*Compare stimulus sets* in the 3D tab reads, for the photos and for each 3D set, the tables the systems wrote:

| Row | Taken from | Calculated by us |
|---|---|---|
| Face found (original or render, Mooney, re-blurred) | `face_detected`, `reblur_face_detected` in `result1_face.csv` | count and share |
| Similarity of the Mooney to its original or render | `similarity_to_own_original` | mean |
| Mooney against original or render, systems 3 and 4 | `depth_correlation_original_vs_mooney`, `depth_correlation_centre_crop` | mean |
| Against the true depth, systems 3 and 4 | `truth_scores.csv` of the 3D set | mean; the photos have no truth |

Only image pairs that have a Mooney are counted, and every cell states its n. Nothing is recomputed from the
images, so the comparison always shows what the last run of each set produced.
