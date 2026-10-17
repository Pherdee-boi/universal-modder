# Face swap + lipsync + tracking (ComfyUI)

Drag `faceswap_lipsync_tracking.json` onto ComfyUI. (`..._api.json` is the same graph in API format, for scripts.)
Only swap faces you have the right to use (your own, or with the person's consent), and label AI-generated video.

## Pipeline

| Stage | Nodes | What it does |
|---|---|---|
| Inputs | VHS Load Video, Load Image, Load Audio | Video forced to 25 fps (LatentSync's rate) |
| **A. Swap** | ReActor Fast Face Swap [OPTIONS] + Face Booster | Detects the face in every frame (tracking), swaps, sharpens |
| **B. Motion** | LivePortrait crop → process → composite | Re-applies the original expressions and smoothed head motion to the swapped face |
| **C. Lipsync** | LatentSync 1.6 | Re-animates the mouth to a *new* audio track |

Output 1 keeps the original audio (stages A+B). Output 2 is lipsynced to your new audio (A+B+C).
Skip a stage by selecting its node(s) and pressing Ctrl+B: bypass "Paste back" to skip B; bypass
"Match video to audio length", LatentSync and OUTPUT 2 to skip C.

## What else you need

1. **Custom nodes** (install via ComfyUI-Manager):
   - `ComfyUI-LatentSyncWrapper` (ShmuelRonen). Delete any older copy first; it needs a clean install. It downloads its checkpoints on first run.
   - `ComfyUI-LivePortraitKJ` (kijai), which you already have. Models auto-download to `models/liveportrait`.
2. **ffmpeg** on PATH (VideoHelperSuite and LatentSync both need it).
3. **ReActor models** (you likely have them):
   - `models/insightface/inswapper_128.onnx`
   - `models/facerestore_models/codeformer-v0.1.0.pth` (and GFPGANv1.4 if you prefer it)
   - InsightFace `buffalo_l` in `models/insightface/models/` (LivePortrait's InsightFace cropper uses it too).
4. **GPU:** about 20 GB VRAM for LatentSync 1.6. Without that, skip C and use a lipsync-only pass elsewhere.
5. In the three input nodes, pick your own files (placeholders are `input.mp4`, `face.png`, `speech.wav`).

## Tips

- Test with `frame_load_cap = 50`, then remove the cap. Long clips: render in chunks (`skip_first_frames`).
- Several people in frame: set `input_faces_index` in the "Faces" node (and the LivePortrait `face_index`) to the right face.
- Face-swap quality: use a sharp, frontal, well-lit source photo. Feed ReActor a blended face model from 3-5 photos
  (Build Blended Face Model) for steadier identity across angles.
- Frames with no detectable face (heavy turn, occlusion) pass through unswapped, and the LivePortrait crop can fail on them.
  Trim those sections or lower the detection threshold.
- Lipsync needs a mostly visible, near-frontal mouth. LatentSync crops and pastes the mouth region back on its own.
- If the swapped face looks "pasted", lower Face Booster `visibility` (0.7) or turn it off; if expressions look
  exaggerated, lower LivePortrait `delta_multiplier`.

## Regenerate

`python3 build_workflow.py` rebuilds both JSON files. Widget order is taken from the node sources, so if a node
pack changes its widgets, edit `NODES` in that file.

Not run end to end here (no GPU or ComfyUI install in the build sandbox); widget names and enums were checked against the node source.
