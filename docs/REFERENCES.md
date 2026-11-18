# References

## Author

**Akbar Hamidov**: architecture of the tool, research questions, method, experiments and their analysis.

## The write-up

Akbar Hamidov. *Mooney faces and vision systems: a controlled comparison with photos and 3D faces.*
Working title; in preparation. It describes the experiments made with this tool and their results.

## Literature the method is built on

| Where it is used | Reference |
|---|---|
| Mooney images | C. M. Mooney (1957). *Age in the development of closure ability in children.* Canadian Journal of Psychology 11(4), 219–226. [doi:10.1037/h0083717](https://doi.org/10.1037/h0083717) |
| Blur and threshold settings | L. C. Reining, T. S. A. Wallis (2024). *A psychophysical evaluation of techniques for Mooney image generation.* PeerJ 12, e18059. [doi:10.7717/peerj.18059](https://doi.org/10.7717/peerj.18059) |
| Black-to-white ratio, default blur | L. C. Reining, T. Matthies, L. Haussner, R. Turon, T. S. A. Wallis (2026). *MooneyMaker: A Python package to create ambiguous two-tone images.* [arXiv:2601.14077](https://arxiv.org/abs/2601.14077) |
| Range of blur values | T.-W. Ke, S. X. Yu, D. Whitney (2017). *Mooney face classification and prediction by learning across tone.* IEEE ICIP. [Paper page](https://web.eecs.umich.edu/~stellayu/publication/2017mooneyICIP.html) |
| Background separation | X. Qin et al. (2020). *U²-Net: Going deeper with nested U-structure for salient object detection.* Pattern Recognition 106. [GitHub](https://github.com/xuebinqin/U-2-Net) |
| System 1, face detection | W. Wu, H. Peng, S. Yu (2023). *YuNet: A tiny millisecond-level face detector.* Machine Intelligence Research. [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| System 1, face embedding | Y. Zhong et al. (2021). *SFace: Sigmoid-constrained hypersphere loss for robust face recognition.* IEEE TIP 30. [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |
| System 2, vision-language model | H. Liu et al. (2023). *Visual instruction tuning* (LLaVA). NeurIPS. Served locally with [Ollama](https://ollama.com) |
| System 3, learned depth | L. Yang et al. (2024). *Depth Anything V2.* NeurIPS. [GitHub](https://github.com/DepthAnything/Depth-Anything-V2) |
| System 4, depth from shading | P.-S. Tsai, M. Shah (1994). *Shape from shading using linear approximation.* Image and Vision Computing 12(8). [doi:10.1016/0262-8856(94)90002-7](https://doi.org/10.1016/0262-8856(94)90002-7) |
| Shape from shading, overview | R. Zhang, P.-S. Tsai, J. E. Cryer, M. Shah (1999). *Shape from shading: a survey.* IEEE TPAMI 21(8), 690–706 |
| 3D faces from photos, face landmarks | C. Lugaresi et al. (2019). *MediaPipe: A framework for building perception pipelines.* [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |
| Invented and fitted 3D faces | T. Gerig et al. (2018). *Morphable face models: an open framework.* IEEE FG. [Basel Face Model 2019](https://faces.dmi.unibas.ch/bfm/bfm2019.html) |
| Fitting a face model to a photo | V. Blanz, T. Vetter (1999). *A morphable model for the synthesis of 3D faces.* SIGGRAPH. B. Egger et al. (2020). *3D morphable face models: past, present and future.* ACM TOG 39(5) |

## Photos

Photos for the experiments were collected automatically from [Wikimedia Commons](https://commons.wikimedia.org),
open licences only. The list below is generated from the project's `dataset.csv`: author, licence and source
page of every photo. Photos you add by hand can be credited in `credits.csv` in the project folder (columns
`file, author, licence, source`); those rows are listed as well.

## Use of AI

Design, research questions, method choices and all decisions: Akbar Hamidov. Code and documentation were
written with AI assistance (Claude by Anthropic, used as a coding agent) under the author's direction and
reviewed by him. How to continue the work with an agent is described in `AGENTS.md` in the code folder.

| Phase | Model (reasoning effort) |
|---|---|
| Understanding the problem | Claude Fable 5.1 (medium, high) |
| First implementation | Claude Opus 5.5 (medium) |
| Main development, discussions, clean-up | Claude Fable 5.2 (high) |
| Final review and development after it | Claude Opus 5.5 (high) |
| Final development and documentation | Claude Fable 5.2 (medium, high) |

## Software

[Gradio](https://www.gradio.app) (interface), [OpenCV](https://opencv.org), [ONNX Runtime](https://onnxruntime.ai),
[rembg](https://github.com/danielgatis/rembg), [NumPy](https://numpy.org), [SciPy](https://scipy.org),
[Pillow](https://python-pillow.org), [Matplotlib](https://matplotlib.org), [h5py](https://www.h5py.org).
