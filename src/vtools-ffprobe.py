#!/usr/bin/env python3

"""ffprobe.py: A wrapper around ffprobe."""


import json
import numpy as np
import pandas as pd
import re
import sys

from common import run


def run_ffprobe_command(infile, analysis="frames", **kwargs):
    stream_id = kwargs.get("stream_id", None)
    ffprobe_bin = kwargs.get("ffprobe", "ffprobe")
    debug = kwargs.get("debug", 0)
    add_qp = kwargs.get("add_qp", False)
    add_mb_type = kwargs.get("add_mb_type", False)
    command = f"{ffprobe_bin}"
    if stream_id is not None:
        command += f"-select_streams {stream_id}"
    command += " -print_format json"
    if analysis == "streams":
        command += " -show_streams"
    else:
        command += " -count_frames -show_frames"
    if add_qp:
        command += " -debug qp"
    elif add_mb_type:
        command += " -debug mb_type"
    command += f" '{infile}'"
    returncode, out, err = run(command, debug=debug)
    assert returncode == 0, f'error running "{command}"'
    return out, err


def get_info(infile, **kwargs):
    out, err = run_ffprobe_command(infile, analysis="streams", **kwargs)
    # parse the output
    streams = parse_ffprobe_streams_output(out, **kwargs)
    return streams


def get_frames_information(infile, **kwargs):
    debug = kwargs.get("debug", 0)
    out, err = run_ffprobe_command(infile, **kwargs)
    # parse the output
    df = parse_ffprobe_frames_output(out, debug)
    # sort the frames
    df = df.sort_values(by=["frame_num"])
    return df


def get_frames_qp_information(infile, **kwargs):
    debug = kwargs.get("debug", 0)
    out, err = run_ffprobe_command(infile, add_qp=True, **kwargs)
    # parse the output
    ffprobe_df = parse_ffprobe_frames_output(out, debug)
    qp_df = parse_qp_information(err, debug)
    # join the output
    # ensure the same number of frames in both sources
    assert (
        ffprobe_df.shape[0] == qp_df.shape[0]
    ), f"error: ffprobe produced {ffprobe_df.shape[0]} frames while QP produced {qp_df.shape[0]} frames"
    df = ffprobe_df.join(
        qp_df.set_index("frame_num"), on="frame_num", rsuffix="_remove"
    )
    duplicated_columns = list(k for k in df.keys() if k.endswith("_remove"))
    df.drop(columns=duplicated_columns, inplace=True)
    return df


def get_frames_mb_information(infile, **kwargs):
    debug = kwargs.get("debug", 0)
    out, err = run_ffprobe_command(infile, add_mb_type=True, **kwargs)
    # parse the output
    ffprobe_df = parse_ffprobe_frames_output(out, debug)
    mb_df = parse_mb_information(err, debug)
    # join the output
    # ensure the same number of frames in both sources
    assert (
        ffprobe_df.shape[0] == mb_df.shape[0]
    ), f"error: ffprobe produced {ffprobe_df.shape[0]} frames while MB-type produced {mb_df.shape[0]} frames"
    df = ffprobe_df.join(
        mb_df.set_index("frame_num"), on="frame_num", rsuffix="_remove"
    )
    duplicated_columns = list(k for k in df.keys() if k.endswith("_remove"))
    df.drop(columns=duplicated_columns, inplace=True)
    return df


def parse_ffprobe_streams_output(out, debug):
    # load the output
    parsed_out = json.loads(out)
    assert (
        "streams" in parsed_out
    ), f"error: invalid ffprobe output (keys are {parsed_out.keys()})"
    streams = parsed_out["streams"]
    return streams


def parse_ffprobe_frames_output(out, debug):
    frame_dict = parse_ffprobe_output(out, debug)
    # punt if more than 1
    assert (
        len(frame_dict.keys()) == 1
    ), f"error: video contains {len(frame_dict.keys())} video streams"
    stream_id = list(frame_dict.keys())[0]
    return frame_dict[stream_id]


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
    assert (
        "frames" in parsed_out
    ), f"error: invalid ffprobe output (keys are {parsed_out.keys()})"
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

        # convert the dictionary into a dataframe
        df = pd.DataFrame(columns=ffprobe_keys)
        for frame in video_frames[stream_id]:
            vals = []
            for key in ffprobe_keys:
                val = frame.get(key, "")
                if not isinstance(val, (int, float, str)):
                    val = str(val)
                vals.append(val)
            df.loc[len(df.index)] = vals
        new_video_frames[stream_id] = df
    return new_video_frames


def get_qp_statistics(qp_list):
    qp_arr = np.array(qp_list)
    qp_min = qp_arr.min()
    qp_max = qp_arr.max()
    qp_mean = qp_arr.mean()
    qp_var = qp_arr.var()
    return qp_min, qp_max, qp_mean, qp_var


def parse_qp_information(out, debug):
    frame_num = -1
    pix_fmt = None
    pict_type = None
    qp_list = []

    reinit_pattern = (
        r"\[[^\]]+\] Reinit context to (?P<resolution>\d+x\d+), "
        r"pix_fmt: (?P<pix_fmt>.+)"
    )
    newframe_pattern = r"\[[^\]]+\] New frame, type: (?P<pict_type>.+)"
    qp_pattern = r"\[[^\]]+\] (?P<qp_str>[\d ]+)$"

    qp_keys = [
        "frame_num",
        "pix_fmt",
        "pict_type",
        "qp_min",
        "qp_max",
        "qp_mean",
        "qp_var",
    ]
    df = pd.DataFrame(columns=qp_keys)

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
            df.drop(df.index, inplace=True)
            frame_num = -1

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
                df.loc[len(df.index)] = [
                    frame_num,
                    pix_fmt,
                    pict_type,
                    qp_min,
                    qp_max,
                    qp_mean,
                    qp_var,
                ]
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
            qp_list += [int(qp_str[i : i + 2]) for i in range(0, len(qp_str), 2)]

    # dump the last state
    if qp_list:
        qp_min, qp_max, qp_mean, qp_var = get_qp_statistics(qp_list)
        df.loc[len(df.index)] = [
            frame_num,
            pix_fmt,
            pict_type,
            qp_min,
            qp_max,
            qp_mean,
            qp_var,
        ]
        qp_list = []

    return df


MB_TYPE_LIST = [
    "P",  # IS_PCM(mb_type)  // MB_TYPE_INTRA_PCM
    "A",  # IS_INTRA(mb_type) && IS_ACPRED(mb_type)  // MB_TYPE_ACPRED
    "i",  # IS_INTRA4x4(mb_type)  // MB_TYPE_INTRA4x4
    "I",  # IS_INTRA16x16(mb_type)  // MB_TYPE_INTRA16x16
    "d",  # IS_DIRECT(mb_type) && IS_SKIP(mb_type)
    "D",  # IS_DIRECT(mb_type)  // MB_TYPE_DIRECT2
    "g",  # IS_GMC(mb_type) && IS_SKIP(mb_type)
    "G",  # IS_GMC(mb_type)  // MB_TYPE_GMC
    "S",  # IS_SKIP(mb_type)  // MB_TYPE_SKIP
    ">",  # !USES_LIST(mb_type, 1)
    "<",  # !USES_LIST(mb_type, 0)
    "X",  # av_assert2(USES_LIST(mb_type, 0) && USES_LIST(mb_type, 1))
]

# simplified types
MB_STYPE_DICT = {
    "intra": [
        "A",
        "i",
        "I",
    ],
    "intra_pcm": [
        "P",
    ],
    "inter": [
        "<",
        ">",
    ],
    "skip_direct": [
        "S",
        "d",
        "D",
    ],
    "other": [
        "X",
    ],
    "gmc": [
        "g",
        "G",
    ],
}


def parse_mb_information(out, debug):
    frame_num = -1
    resolution = None
    pix_fmt = None
    pict_type = None
    mb_dict = {}

    reinit_pattern = (
        r"\[[^\]]+\] Reinit context to (?P<resolution>\d+x\d+), "
        r"pix_fmt: (?P<pix_fmt>.+)"
    )
    newframe_pattern = r"\[[^\]]+\] New frame, type: (?P<pict_type>.+)"
    mb_pattern = r"\[[^\]]+\] (?P<mb_str>[PAiIdDgGS><X+\-|= ]+)$"

    mb_only_keys = [f"mb_type_{mb_type}" for mb_type in MB_TYPE_LIST]
    mb_only_keys += [f"mb_stype_{mb_stype}" for mb_stype in MB_STYPE_DICT]
    mb_keys = ["frame_num", "pix_fmt", "pict_type"]
    mb_keys += mb_only_keys
    df = pd.DataFrame(columns=mb_keys)

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
            resolution = match.group("resolution")
            pix_fmt = match.group("pix_fmt")
            df.drop(df.index, inplace=True)
            frame_num = -1
            mb_dict = {}

        elif "New frame, type:" in line:
            # [h264 @ 0x30d1a80] New frame, type: I
            match = re.search(newframe_pattern, line)
            if not match:
                print(f'warning: invalid newframe line ("{line}")')
                sys.exit(-1)
            # store the old frame info
            if frame_num != -1:
                mb_list = mb_dict_convert(mb_only_keys, mb_dict)
                df.loc[len(df.index)] = [frame_num, pix_fmt, pict_type, *mb_list]
                mb_dict = {}
            # new frame
            pict_type = match.group("pict_type")
            frame_num += 1
        else:
            # "[h264 @ ...] S  S  S  S  S  >- S  S  S  S  S  S  >  S  S  S  "
            match = re.search(mb_pattern, line)
            if not match:
                # print(f'error: invalid line: {line}')
                continue
            mb_str = match.group("mb_str")
            # make sure mb_str length is a multiple of 3
            while (len(mb_str) % 3) != 0:
                mb_str += " "
            mb_row_list = [mb_str[i : i + 1] for i in range(0, len(mb_str), 3)]
            row_mb_dict = {
                mb_type: mb_row_list.count(mb_type) for mb_type in mb_row_list
            }
            for k, v in row_mb_dict.items():
                mb_dict[k] = (mb_dict[k] if k in mb_dict else 0) + v

    # dump the last state
    if mb_dict:
        mb_list = mb_dict_convert(mb_only_keys, mb_dict)
        df.loc[len(df.index)] = [frame_num, pix_fmt, pict_type, *mb_list]
        mb_dict = {}

    return df


def mb_dict_convert(mb_only_keys, mb_dict):
    mb_info = {}
    # 1. read the mb_dict
    for mb_type in MB_TYPE_LIST:
        mb_info[f"mb_type_{mb_type}"] = mb_dict.get(mb_type, 0) / sum(mb_dict.values())
    # 2. add derived values
    for mb_stype in MB_STYPE_DICT.keys():
        mb_info[f"mb_stype_{mb_stype}"] = 0
    for mb_stype, mb_type_list in MB_STYPE_DICT.items():
        for mb_type in mb_type_list:
            mb_info[f"mb_stype_{mb_stype}"] += mb_info[f"mb_type_{mb_type}"]
    # 3. convert to mb_list
    mb_list = []
    for key in mb_only_keys:
        mb_list.append(mb_info[key])
    return mb_list
