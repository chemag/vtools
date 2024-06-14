#!/usr/bin/env python3

"""ffprobe.py: A wrapper around ffprobe."""


import importlib
import io
import json
import numpy as np
import pandas as pd
import re
import sys
import tempfile

vtools_common = importlib.import_module("vtools-common")

def get_audio_info(infile, **kwargs):
    ffprobe_bin = kwargs.get("ffprobe", "ffprobe")
    debug = kwargs.get("debug", 0)
    
    # Construct the ffprobe command to extract audio stream information
    command = f"{ffprobe_bin} -i '{infile}' -show_streams -select_streams a -print_format json"
    
    # Execute the command
    # print(command)  # Optional: print the command to debug
    returncode, out, err = vtools_common.run(command, debug=debug)
    
    # Check if the command was successful
    assert returncode == 0, f'Error running "{command}"'
    
    # Parse the output to extract audio properties
    import json
    data = json.loads(out)
    if len(data['streams']) == 0:
        raise ValueError("No audio stream found in the file.")
    
    audio_stream = data['streams'][0]
    sample_rate = int(audio_stream.get('sample_rate', '0'))
    bitrate = int(audio_stream.get('bit_rate', '0'))
    duration = float(audio_stream.get('duration', '0.0'))
    # Hz, bps, seconds
    return sample_rate, bitrate, duration

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
    returncode, out, err = vtools_common.run(command, debug=debug)
    assert returncode == 0, f'error running "{command}"'
    return out, err


def get_info(infile, config_dict, debug):
    out, err = run_ffprobe_command(infile, analysis="streams", debug=debug)
    # parse the output
    streams = parse_ffprobe_streams_output(out, config_dict, debug)
    return streams


def get_frames_information(infile, config_dict, debug):
    out, err = run_ffprobe_command(infile, debug=debug)
    # parse the output
    df = parse_ffprobe_frames_output(out, debug)
    # sort the frames
    df = df.sort_values(by=["frame_num"])
    return df


QPEXTRACT_FIELDS = ("qp_avg", "qp_stddev", "qp_num", "qp_min", "qp_max")
CTU_SIZE_VALUES = (8, 16, 32, 64)


def parse_qpextract_bin_output(output, mode):
    df = pd.read_csv(io.StringIO(output.decode("ascii")))
    if mode in ("qpy", "qpcb", "qpcr"):
        # drop numeric columns
        df = df.drop(columns=list(str(i) for i in range(53)))
        rename_suffix = ("num", "min", "max", "avg", "stddev")
        rename_dict = {f"qp_{suffix}": f"{mode}:{suffix}" for suffix in rename_suffix}
        df = df.rename(columns=rename_dict)
        return df
    elif mode == "ctu":
        # get statistics
        df_out = pd.DataFrame(columns=("frame", "ctu:mean", "ctu:stddev"))
        for frame in df["frame"].unique():
            df_out.loc[df_out.size] = [
                frame,
                df[df["frame"] == frame]["size"].mean(),
                df[df["frame"] == frame]["size"].std(),
            ]
        df_out.frame = df_out.frame.apply(int)
        return df_out


def get_frames_qp_information(infile, config_dict, debug):
    # check the file is h264
    file_info = get_info(infile, config_dict, debug)
    video_codec_name = None
    for stream_info in file_info:
        if stream_info["codec_type"] == "audio":
            continue
        elif stream_info["codec_type"] == "video":
            video_codec_name = stream_info["codec_name"]
            break
    else:
        # no video stream
        return None
    if video_codec_name == "h264":
        # 1. use ffmpeg QP mode (h264 only)
        # run the QP analysis command
        out, err = run_ffprobe_command(infile, add_qp=True, debug=debug)
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
    elif video_codec_name == "hevc":
        # 2. use qpextract analysis (h265 only)
        qpextract_bin = config_dict.get("qpextract_bin", None)
        if qpextract_bin is None:
            return None
        # qpextract needs an Annex B file
        tmp265 = tempfile.NamedTemporaryFile(prefix="vtools.hevc.", suffix=".265").name
        command = f"ffmpeg -i {infile} -vcodec copy -an {tmp265}"
        returncode, out, err = vtools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        # extract the QP-Y info for the first tile
        command = f"{qpextract_bin} --qpymode -w -i {tmp265}"
        returncode, out, err = vtools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        df = parse_qpextract_bin_output(out, "qpy")
        # extract the QP-Cb info for the first tile
        command = f"{qpextract_bin} --qpcbmode -w -i {tmp265}"
        returncode, out, err = vtools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        qpcb_df = parse_qpextract_bin_output(out, "qpcb")
        df = df.join(qpcb_df.set_index("frame"), on="frame", rsuffix="_remove")
        # extract the QP-Cr info for the first tile
        command = f"{qpextract_bin} --qpcrmode -w -i {tmp265}"
        returncode, out, err = vtools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        qpcr_df = parse_qpextract_bin_output(out, "qpcr")
        df = df.join(qpcr_df.set_index("frame"), on="frame", rsuffix="_remove")
        # extract the CTU info for the first tile
        command = f"{qpextract_bin} --ctumode -w -i {tmp265}"
        returncode, out, err = vtools_common.run(command, debug=debug)
        assert returncode == 0, f"error in {command}\n{err}"
        ctu_df = parse_qpextract_bin_output(out, "ctu")
        df = df.join(ctu_df.set_index("frame"), on="frame", rsuffix="_remove")
        df = df.rename(columns={"frame": "frame_num"})
    else:
        return None

    return df


def get_frames_mb_information(infile, config_dict, debug):
    out, err = run_ffprobe_command(infile, add_mb_type=True, debug=debug)
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


def parse_ffprobe_streams_output(out, config_dict, debug):
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
    "pkt_duration_time_ms",
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
    # audio information
    "audio_sample_rate",
    "audio_bitrate",
    "audio_duration_time",
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
    prev_frame_pts_time = None
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
        # pkt_duration_time is deprecated, use duration_time
        if "pkt_duration_time" in frame:
            pkt_duration_time = float(frame["pkt_duration_time"])
        elif "duration_time" in frame:
            pkt_duration_time = float(frame["duration_time"])
        else:
            # TODO(chema): this is measuring the time after this frame
            if prev_frame_pts_time is None:
                pkt_duration_time = np.NaN
            else:
                pkt_duration_time = float(frame["pts_time"]) - prev_frame_pts_time
            prev_frame_pts_time = float(frame["pts_time"])
        frame["pkt_duration_time_ms"] = pkt_duration_time * 1000.0
        # add bitrate (bps)
        frame["bitrate"] = (
            ((int(frame["pkt_size"]) * 8) / pkt_duration_time)
            if (pkt_duration_time != np.NaN and pkt_duration_time != 0.0)
            else np.NaN
        )
        # add framerate (fps)
        frame["framerate"] = (
            (1.0 / pkt_duration_time)
            if (pkt_duration_time != np.NaN and pkt_duration_time != 0.0)
            else np.NaN
        )
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
            # old format
            # [h264 @ 0x30d1a80] 3535353535353535353535...
            # new format
            # [h264 @ 0x9cabc0]      0               128             256  ... 1664            1792
            # [h264 @ 0x9cabc0]    0 343434343434343434343434343435373737 ... 7404040404040393940
            # [h264 @ 0x9cabc0]   16 363636363936363636363636363636363633 ... 3333333333333363636
            # [h264 @ 0x9cabc0]   32 404040404040404039373737373636363636 ... 5353535353535353535
            # [h264 @ 0x9cabc0]   48 363636363636363636363636363636363636 ... 6363636363434343434
            # ...
            # [h264 @ 0x9cabc0]  784 393939393939393939393939393939393939 ... 9393939393939393939
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
