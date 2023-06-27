#!/usr/bin/env python3

"""ffprobe.py: A wrapper around ffprobe."""


import json
import numpy as np
import re
import sys

from common import run


def run_ffprobe_command(infile, **kwargs):
    stream_id = kwargs.get("stream_id", None)
    ffprobe_bin = kwargs.get("ffprobe", "ffprobe")
    debug = kwargs.get("debug", 0)
    add_qp = kwargs.get("add_qp", False)
    command = f"{ffprobe_bin}"
    if stream_id is not None:
        command += f"-select_streams {stream_id}"
    command += " -print_format json"
    command += " -count_frames -show_frames"
    if add_qp:
        command += " -debug qp"
    command += f" {infile}"
    returncode, out, err = run(command, debug=debug)
    assert returncode == 0, f'error running "{command}"'
    return out, err


def get_frames_information(infile, **kwargs):
    debug = kwargs.get("debug", 0)
    out, err = run_ffprobe_command(infile, **kwargs)
    # parse the output
    keys, vals = parse_ffprobe_frames_output(out, debug)
    # sort the frames
    vals = sort_frames(keys, vals, "frame_num")
    return keys, vals


def get_frames_qp_information(infile, **kwargs):
    debug = kwargs.get("debug", 0)
    out, err = run_ffprobe_command(infile, add_qp=True, **kwargs)
    # parse the output
    ffprobe_keys, ffprobe_vals = parse_ffprobe_frames_output(out, debug)
    qp_keys, qp_vals = parse_qp_information(err, debug)
    # join the output
    # ensure the same number of frames in both sources
    assert len(ffprobe_vals) == len(qp_vals), f"error: ffprobe produced {len(ffprobe_vals)} frames while QP produced {len(qp_vals)} frames"
    # assume same order for both lists
    # keep only the first field ("frame_num") from the ffprobe list
    qp_keys = ffprobe_keys[:1] + qp_keys[3:]
    qp_vals = [v1[:1] + v2[3:] for (v1, v2) in zip(ffprobe_vals, qp_vals)]
    # sort the frames
    qp_vals = sort_frames(qp_keys, qp_vals, "frame_num")
    return qp_keys, qp_vals


def parse_ffprobe_frames_output(out, debug):
    frame_dict = parse_ffprobe_output(out, debug)
    # punt if more than 1
    assert len(frame_dict.keys()) == 1, f"error: video contains {len(frame_dict.keys())} video streams"
    keys, vals = list(frame_dict.values())[0]
    return keys, vals


PREFERRED_KEY_ORDER = [
    # frame order
    "frame_num",
    "coded_picture_number",
    "display_picture_number",
    # time and position
    "pts",
    "pts_time",
    "pkt_pos",
    "pkt_dts",
    "pkt_dts_time",
    "duration",
    "duration_time",
    "pkt_duration",
    "pkt_duration_time",
    "best_effort_timestamp",
    "best_effort_timestamp_time",
    "framerate",
    # frame type
    "pix_fmt",
    "pict_type",
    "key_frame",
    "chroma_location",
    "interlaced_frame",
    "top_field_first",
    # frame size
    "pkt_size",
    "bpp",
    "bitrate",
    "width",
    "height",
    "crop_left",
    "crop_right",
    "crop_bottom",
    "crop_top",
    "sample_aspect_ratio",
    # other
    "repeat_pict",
    "side_data_list",
]


def parse_ffprobe_output(out, debug):
    # load the output
    parsed_out = json.loads(out)
    assert "frames" in parsed_out, f"error: invalid ffprobe output (keys are {parsed_out.keys()})"
    frames = parsed_out["frames"]

    # select video frames and add derived values
    video_frames = {}
    frame_num = {}
    for frame in frames:
        if frame["media_type"] != "video":
            continue
        del frame["media_type"]
        stream_index = frame["stream_index"]
        del frame["stream_index"]
        if stream_index not in video_frames:
            video_frames[stream_index] = []
            frame_num[stream_index] = 0
        # check whether there is a coded_picture_number key
        coded_picture_number = frame.get("coded_picture_number", None)
        # add derived values
        # add the frame_num
        if coded_picture_number is None or coded_picture_number == 0:
            frame["frame_num"] = frame_num[stream_index]
        else:
            frame["frame_num"] = coded_picture_number
        frame_num[stream_index] += 1
        # add bits per pixel (bpp)
        frame["bpp"] = (int(frame["pkt_size"]) * 8) / (
            int(frame["width"]) * int(frame["height"])
        )
        # add bitrate (bps)
        frame["bitrate"] = (int(frame["pkt_size"]) * 8) / float(
            frame["pkt_duration_time"]
        )
        # add framerate (fps)
        frame["framerate"] = 1.0 / float(frame["pkt_duration_time"])
        # store the frame
        video_frames[stream_index].append(frame)

    # ensure all the frames have the same keys
    new_video_frames = {}
    for stream_id in video_frames:
        ffprobe_keys = (list(frame.keys()) for frame in video_frames[stream_id])
        # flatten the list
        ffprobe_keys = list(set([key for item in ffprobe_keys for key in item]))
        # sort the list according to PREFERRED_KEY_ORDER
        intersection = set(ffprobe_keys).intersection(set(PREFERRED_KEY_ORDER))
        difference = set(ffprobe_keys).difference(set(PREFERRED_KEY_ORDER))
        # start with the intersection
        ffprobe_keys = [key for key in PREFERRED_KEY_ORDER if key in intersection]
        # then add the difference
        ffprobe_keys += list(difference)
        # convert the dictionary into a list
        ffprobe_vals = []
        for frame in video_frames[stream_id]:
            vals = []
            for key in ffprobe_keys:
                val = frame.get(key, "")
                if not isinstance(val, (int, float, str)):
                    val = str(val)
                vals.append(val)
            ffprobe_vals.append(vals)
        new_video_frames[stream_id] = (ffprobe_keys, ffprobe_vals)

    return new_video_frames


def sort_frames(keys, vals, field_name):
    # sort the frames by <field_name> (using deco sort)
    field_name_index = keys.index(field_name)
    # 1. decorate the list
    deco = [(val[field_name_index], val) for val in vals]
    # 2. sort the decorated list
    deco.sort()
    # 3. undecorate the list
    return [val for _, val in deco]


def get_qp_statistics(qp_list):
    qp_arr = np.array(qp_list)
    qp_min = qp_arr.min()
    qp_max = qp_arr.max()
    qp_mean = qp_arr.mean()
    qp_var = qp_arr.var()
    return qp_min, qp_max, qp_mean, qp_var


def parse_qp_information(out, debug):
    qp_vals = []
    frame_num = -1
    pix_fmt = None
    pict_type = None
    qp_list = []

    reinit_pattern = (
        r"\[[^\]]+\] Reinit context to (?P<resolution>\d+x\d+), "
        r"pix_fmt: (?P<pix_fmt>.+)"
    )
    newframe_pattern = r"\[[^\]]+\] New frame, type: (?P<pict_type>.+)"
    qp_pattern = r"\[[^\]]+\] (?P<qp_str>\d+)$"

    qp_keys = ["frame_num", "pix_fmt", "pict_type", "qp_min", "qp_max", "qp_mean", "qp_var"]

    for line in out.splitlines():
        try:
            line = line.decode("ascii").strip()
        except UnicodeDecodeError:
            # ignore the line
            continue
        if "Reinit context to" in line:
            # [h264 @ 0x30d1a80] Reinit context to 1280x720, pix_fmt: yuv420p
            match = re.search(reinit_pattern, line)
            if not match:
                print(f'warning: invalid reinit line ("{line}")')
                sys.exit(-1)
            # reinit: flush all previous data
            _resolution = match.group("resolution")
            pix_fmt = match.group("pix_fmt")
            qp_vals = []
            frame_num = -1
            qp_list = []

        elif "New frame, type:" in line:
            # [h264 @ 0x30d1a80] New frame, type: I
            match = re.search(newframe_pattern, line)
            if not match:
                print(f'warning: invalid newframe line ("{line}")')
                sys.exit(-1)
            # store the old frame info
            if frame_num != -1:
                # get derived QP statistics
                qp_min, qp_max, qp_mean, qp_var = get_qp_statistics(qp_list)
                qp_vals.append([frame_num, pix_fmt, pict_type, qp_min, qp_max, qp_mean, qp_var])
                qp_list = []
            # new frame
            pict_type = match.group("pict_type")
            frame_num += 1

        else:
            # [h264 @ 0x30d1a80] 3535353535353535353535...
            match = re.search(qp_pattern, line)
            if not match:
                continue
            qp_str = match.group("qp_str")
            qp_list += [int(qp_str[i: i + 2]) for i in range(0, len(qp_str), 2)]

    # dump the last state
    if qp_list:
        qp_min, qp_max, qp_mean, qp_var = get_qp_statistics(qp_list)
        qp_vals.append([frame_num, pix_fmt, pict_type, qp_min, qp_max, qp_mean, qp_var])
        qp_list = []

    return qp_keys, qp_vals
