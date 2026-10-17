#!/usr/bin/env python3
"""Generate the ComfyUI face-swap workflow (UI format + API format) from one graph description.

    python3 build_workflow.py            # writes faceswap_lipsync_tracking.json and ..._api.json

Widget order and names come from the node sources (ReActor, LivePortraitKJ, VideoHelperSuite,
LatentSyncWrapper, ComfyUI core). If a node pack changes its widgets, edit NODES below and re-run.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# type -> (linked inputs [(name, type)], outputs [(name, type)])
IO = {
    "VHS_LoadVideo": ([], [("IMAGE", "IMAGE"), ("frame_count", "INT"), ("audio", "AUDIO"), ("video_info", "VHS_VIDEOINFO")]),
    "LoadImage": ([], [("IMAGE", "IMAGE"), ("MASK", "MASK")]),
    "LoadAudio": ([], [("AUDIO", "AUDIO")]),
    "ReActorOptions": ([], [("OPTIONS", "OPTIONS")]),
    "ReActorFaceBoost": ([], [("FACE_BOOST", "FACE_BOOST")]),
    "ReActorFaceSwapOpt": (
        [("input_image", "IMAGE"), ("source_image", "IMAGE"), ("options", "OPTIONS"), ("face_boost", "FACE_BOOST")],
        [("SWAPPED_IMAGE", "IMAGE"), ("FACE_MODEL", "FACE_MODEL"), ("ORIGINAL_IMAGE", "IMAGE")],
    ),
    "DownloadAndLoadLivePortraitModels": ([], [("live_portrait_pipe", "LIVEPORTRAITPIPE")]),
    "LivePortraitLoadCropper": ([], [("cropper", "LPCROPPER")]),
    "LivePortraitCropper": (
        [("pipeline", "LIVEPORTRAITPIPE"), ("cropper", "LPCROPPER"), ("source_image", "IMAGE")],
        [("cropped_image", "IMAGE"), ("crop_info", "CROPINFO")],
    ),
    "LivePortraitProcess": (
        [("pipeline", "LIVEPORTRAITPIPE"), ("crop_info", "CROPINFO"), ("source_image", "IMAGE"), ("driving_images", "IMAGE")],
        [("cropped_image", "IMAGE"), ("output", "LP_OUT")],
    ),
    "LivePortraitComposite": (
        [("source_image", "IMAGE"), ("cropped_image", "IMAGE"), ("liveportrait_out", "LP_OUT")],
        [("full_images", "IMAGE"), ("mask", "MASK")],
    ),
    "VideoLengthAdjuster": ([("images", "IMAGE"), ("audio", "AUDIO")], [("images", "IMAGE"), ("audio", "AUDIO")]),
    "LatentSyncNode": ([("images", "IMAGE"), ("audio", "AUDIO")], [("images", "IMAGE"), ("audio", "AUDIO")]),
    "VHS_VideoCombine": ([("images", "IMAGE"), ("audio", "AUDIO")], [("Filenames", "VHS_FILENAMES")]),
    "Note": ([], []),
}

OPTIONAL_INPUTS = {"source_image", "options", "face_boost", "audio"}  # shape 7 in the UI for these


def combine(prefix, rate=25):
    return [("frame_rate", rate), ("loop_count", 0), ("filename_prefix", prefix), ("format", "video/h264-mp4"),
            ("pix_fmt", "yuv420p"), ("crf", 17), ("save_metadata", True), ("pingpong", False), ("save_output", True)]


# key: (type, title, (col, row), widgets [(name, value)], links {input: (src_key, out_slot)}, mode)
NODES = {
    # ---- inputs -------------------------------------------------------------------------------
    "video": ("VHS_LoadVideo", "1. TARGET VIDEO (upload here)", (0, 0),
              [("video", "input.mp4"), ("force_rate", 25), ("custom_width", 0), ("custom_height", 0),
               ("frame_load_cap", 0), ("skip_first_frames", 0), ("select_every_nth", 1), ("format", "AnimateDiff")], {}, 0),
    "face": ("LoadImage", "2. SOURCE FACE (clear, frontal photo)", (0, 1),
             [("image", "face.png"), ("upload", "image")], {}, 0),
    "audio": ("LoadAudio", "3. NEW AUDIO for lipsync (stage C only)", (0, 2),
              [("audio", "speech.wav")], {}, 0),

    # ---- stage A: face swap ---------------------------------------------------------------------
    "opts": ("ReActorOptions", "Faces: pick who gets swapped", (1, 0),
             [("input_faces_order", "large-small"), ("input_faces_index", "0"), ("detect_gender_input", "no"),
              ("source_faces_order", "large-small"), ("source_faces_index", "0"), ("detect_gender_source", "no"),
              ("console_log_level", 1), ("restore_swapped_only", True)], {}, 0),
    "boost": ("ReActorFaceBoost", "Face Booster (sharper swap)", (1, 1),
              [("enabled", True), ("boost_model", "codeformer-v0.1.0.pth"), ("interpolation", "Bicubic"),
               ("visibility", 1.0), ("codeformer_weight", 0.6), ("restore_with_main_after", False)], {}, 0),
    "swap": ("ReActorFaceSwapOpt", "A. Face swap (per-frame detect + track)", (2, 0),
             [("enabled", True), ("swap_model", "inswapper_128.onnx"), ("facedetection", "retinaface_resnet50"),
              ("face_restore_model", "none"), ("face_restore_visibility", 1.0), ("codeformer_weight", 0.5)],
             {"input_image": ("video", 0), "source_image": ("face", 0), "options": ("opts", 0), "face_boost": ("boost", 0)}, 0),

    # ---- stage B: LivePortrait expression + motion transfer --------------------------------------
    "lp_models": ("DownloadAndLoadLivePortraitModels", "LivePortrait models", (1, 3),
                  [("precision", "auto"), ("mode", "human")], {}, 0),
    "lp_crop_loader": ("LivePortraitLoadCropper", "LP face detector (InsightFace)", (1, 4),
                       [("onnx_device", "CUDA"), ("keep_model_loaded", True), ("detection_threshold", 0.5)], {}, 0),
    "lp_crop_src": ("LivePortraitCropper", "Crop swapped face", (2, 3),
                    [("dsize", 512), ("scale", 2.3), ("vx_ratio", 0.0), ("vy_ratio", -0.125), ("face_index", 0),
                     ("face_index_order", "large-small"), ("rotate", True)],
                    {"pipeline": ("lp_models", 0), "cropper": ("lp_crop_loader", 0), "source_image": ("swap", 0)}, 0),
    "lp_crop_drv": ("LivePortraitCropper", "Crop original face (driver)", (2, 4),
                    [("dsize", 512), ("scale", 2.3), ("vx_ratio", 0.0), ("vy_ratio", -0.125), ("face_index", 0),
                     ("face_index_order", "large-small"), ("rotate", True)],
                    {"pipeline": ("lp_models", 0), "cropper": ("lp_crop_loader", 0), "source_image": ("video", 0)}, 0),
    "lp_process": ("LivePortraitProcess", "B. Expression + head-motion transfer", (3, 3),
                   [("lip_zero", False), ("lip_zero_threshold", 0.03), ("stitching", True), ("delta_multiplier", 1.0),
                    ("mismatch_method", "constant"), ("relative_motion_mode", "source_video_smoothed"),
                    ("driving_smooth_observation_variance", 3e-06), ("expression_friendly", False),
                    ("expression_friendly_multiplier", 1.0)],
                   {"pipeline": ("lp_models", 0), "crop_info": ("lp_crop_src", 1),
                    "source_image": ("lp_crop_src", 0), "driving_images": ("lp_crop_drv", 0)}, 0),
    "lp_comp": ("LivePortraitComposite", "Paste back (bypass = skip stage B)", (4, 3), [],
                {"source_image": ("swap", 0), "cropped_image": ("lp_process", 0), "liveportrait_out": ("lp_process", 1)}, 0),
    "out_swap": ("VHS_VideoCombine", "OUTPUT 1: swapped video + original audio", (5, 3), combine("faceswap"),
                 {"images": ("lp_comp", 0), "audio": ("video", 2)}, 0),

    # ---- stage C: lipsync -----------------------------------------------------------------------
    "len_adj": ("VideoLengthAdjuster", "Match video to audio length", (5, 5),
                [("mode", "pingpong"), ("fps", 25.0), ("silent_padding_sec", 0.5)],
                {"images": ("lp_comp", 0), "audio": ("audio", 0)}, 0),
    "lipsync": ("LatentSyncNode", "C. Lipsync to new audio (LatentSync)", (6, 5),
                [("seed", 1247), ("control_after_generate", "randomize"), ("lips_expression", 1.5), ("inference_steps", 20)],
                {"images": ("len_adj", 0), "audio": ("len_adj", 1)}, 0),
    "out_lip": ("VHS_VideoCombine", "OUTPUT 2: swapped + lipsynced", (7, 5), combine("faceswap_lipsync"),
                {"images": ("lipsync", 0), "audio": ("lipsync", 1)}, 0),
}

GROUPS = [
    ("INPUTS", [(0, 0), (0, 2)], "#3f789e"),
    ("A. FACE SWAP (ReActor)", [(1, 0), (2, 1)], "#8A8"),
    ("B. EXPRESSION + MOTION (LivePortrait) - bypass 'Paste back' to skip", [(1, 3), (5, 4)], "#b58b2a"),
    ("C. LIPSYNC (LatentSync) - bypass LatentSync + OUTPUT 2 to skip", [(5, 5), (7, 5)], "#a1309b"),
]

NOTES = [
    ((0, 3), "HOW TO USE\n1. Upload video, face photo, audio in the three INPUTS nodes.\n"
             "2. Test with frame_load_cap = 50 first.\n3. Queue. OUTPUT 1 = swap (+LivePortrait), OUTPUT 2 = + lipsync.\n"
             "Skip a stage: select its node(s), Ctrl+B (bypass). No new audio? Bypass LatentSync + OUTPUT 2 + 'Match video to audio length'."),
]

COL_W, ROW_H = 440, 330
SIZE = {"VHS_LoadVideo": [330, 400], "VHS_VideoCombine": [330, 420], "LoadImage": [330, 330], "Note": [330, 200]}


def pos(c, r):
    return [60 + c * COL_W, 60 + r * ROW_H]


def build_ui():
    ids = {k: i + 1 for i, k in enumerate(NODES)}
    nid = len(NODES) + 1
    links, link_id = [], 0
    out_links = {k: {} for k in NODES}
    in_links = {k: {} for k in NODES}
    for k, (_, _, _, _, lk, _) in NODES.items():
        for iname, (src, slot) in lk.items():
            link_id += 1
            tytype = IO[NODES[src][0]][1][slot][1]
            tslot = [n for n, _t in IO[NODES[k][0]][0]].index(iname)
            links.append([link_id, ids[src], slot, ids[k], tslot, tytype])
            out_links[src].setdefault(slot, []).append(link_id)
            in_links[k][iname] = link_id
    nodes = []
    for order, (k, (typ, title, (c, r), widgets, lk, mode)) in enumerate(NODES.items()):
        ins = []
        for n, t in IO[typ][0]:
            e = {"name": n, "type": t, "link": in_links[k].get(n)}
            if n in OPTIONAL_INPUTS:
                e["shape"] = 7
            ins.append(e)
        outs = [{"name": n, "type": t, "links": out_links[k].get(i, []) or None, "slot_index": i}
                for i, (n, t) in enumerate(IO[typ][1])]
        if typ == "VHS_LoadVideo":
            w = {n: v for n, v in widgets}
            w["choose video to upload"] = "image"
            w["videopreview"] = {"hidden": False, "paused": False,
                                 "params": {"filename": w["video"], "type": "input", "format": "video/mp4",
                                            "force_rate": w["force_rate"], "frame_load_cap": 0,
                                            "skip_first_frames": 0, "select_every_nth": 1}}
        elif typ == "VHS_VideoCombine":
            w = {n: v for n, v in widgets}
            w["videopreview"] = {"hidden": False, "paused": False,
                                 "params": {"filename": w["filename_prefix"] + ".mp4", "subfolder": "",
                                            "type": "output", "format": w["format"], "frame_rate": w["frame_rate"]}}
        elif typ == "LoadAudio":
            w = [widgets[0][1], "", ""]
        else:
            w = [v for _n, v in widgets]
        nodes.append({"id": ids[k], "type": typ, "pos": pos(c, r), "size": SIZE.get(typ, [360, 260]), "flags": {},
                      "order": order, "mode": mode, "inputs": ins, "outputs": outs, "title": title,
                      "properties": {"Node name for S&R": typ}, "widgets_values": w})
    for (c, r), text in NOTES:
        nodes.append({"id": nid, "type": "Note", "pos": pos(c, r), "size": [420, 220], "flags": {}, "order": 99,
                      "mode": 0, "inputs": [], "outputs": [], "title": "Read me", "properties": {}, "widgets_values": [text]})
        nid += 1
    groups = []
    for i, (title, ((c0, r0), (c1, r1)), color) in enumerate(GROUPS, 1):
        x, y = pos(c0, r0)
        x2, y2 = pos(c1, r1)
        groups.append({"id": i, "title": title, "bounding": [x - 20, y - 70, x2 - x + COL_W - 40, y2 - y + ROW_H], "color": color,
                       "font_size": 24, "flags": {}})
    return {"id": "faceswap-lipsync-tracking", "revision": 0, "last_node_id": nid, "last_link_id": link_id, "nodes": nodes,
            "links": links, "groups": groups, "config": {}, "extra": {"ds": {"scale": 0.6, "offset": [0, 0]}}, "version": 0.4}


def build_api():
    ids = {k: str(i + 1) for i, k in enumerate(NODES)}
    api = {}
    for k, (typ, title, _p, widgets, lk, mode) in NODES.items():
        if typ == "Note":
            continue
        inputs = {}
        for n, v in widgets:
            if n in ("control_after_generate", "upload"):
                continue
            inputs[n] = v
        for iname, (src, slot) in lk.items():
            inputs[iname] = [ids[src], slot]
        api[ids[k]] = {"class_type": typ, "inputs": inputs, "_meta": {"title": title}}
    return api


if __name__ == "__main__":
    with open(os.path.join(HERE, "faceswap_lipsync_tracking.json"), "w") as f:
        json.dump(build_ui(), f, indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, "faceswap_lipsync_tracking_api.json"), "w") as f:
        json.dump(build_api(), f, indent=1, ensure_ascii=False)
    print("wrote workflow files in", HERE)
