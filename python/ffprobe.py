#!/usr/bin/env python3

"""ffprobe.py: A wrapper around ffprobe."""


import json

from common import run


def get_frames_information(infile, **kwargs):
    stream_id = kwargs.get("stream_id", None)
    ffprobe_bin = kwargs.get("ffprobe", "ffprobe")
    debug = kwargs.get("debug", 0)
    command = f"{ffprobe_bin}"
    if stream_id is not None:
        command += f"-select_streams {stream_id}"
    command += " -print_format json"
    command += f" -count_frames -show_frames {infile}"
    returncode, out, err = run(command, debug=debug)
    assert returncode == 0, f'error running "{command}"'
    # parse the output
    return parse_ffprobe_frames_output(out, debug)


def parse_ffprobe_frames_output(out, debug):
    frame_dict = parse_ffprobe_output(out, debug)
    # punt if more than 1
    assert len(frame_dict.keys()) == 1, f"error: video contains {len(frame_dict.keys())} video streams"
    frame_list = list(frame_dict.values())[0]
    # add derived values
    return frame_list


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

    # sort the frames by frame_num
    new_video_frames = {}
    for stream_id in video_frames:
        # sort the corresponding video frames (using deco sort)
        # 1. decorate the list
        deco = [(frame["frame_num"], frame) for frame in video_frames[stream_id]]
        # 2. sort the decorated list
        deco.sort()
        # 3. undecorate the list
        new_video_frames[stream_id] = [frame for _, frame in deco]

    # ensure all the frames have the same keys
    video_frames = new_video_frames
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
