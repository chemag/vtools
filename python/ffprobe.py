#!/usr/bin/env python3

"""ffprobe.py: A wrapper around ffprobe."""


from common import run


def get_frames_information(infile, **kwargs):
    stream_id = kwargs.get("stream_id", None)
    ffprobe_bin = kwargs.get("ffprobe", "ffprobe")
    debug = kwargs.get("debug", 0)
    command = f"{ffprobe_bin}"
    if stream_id is not None:
        command += f"-select_streams {stream_id}"
    command += f" -count_frames -show_frames {infile}"
    returncode, out, err = run(command, debug=debug)
    assert returncode == 0, f'error running "{command}"'
    # parse the output
    return parse_ffprobe_frames_output(out, debug)


def parse_ffprobe_frames_output(out, debug):
    frame_list = parse_ffprobe_output(out, "FRAME", debug)
    # add frame numbers
    frame_num = 0
    new_frame_list = []
    for frame_info in frame_list:
        new_frame_info = {
            "frame_num": frame_num,
        }
        new_frame_info.update(frame_info)
        # add derived values
        # add bits per pixel (bpp)
        new_frame_info["bpp"] = (int(new_frame_info["pkt_size"]) * 8) / (
            int(new_frame_info["width"]) * int(new_frame_info["height"])
        )
        # add bitrate (bps)
        new_frame_info["bitrate"] = (int(frame_info["pkt_size"]) * 8) / float(
            frame_info["pkt_duration_time"]
        )
        # add framerate (fps)
        new_frame_info["framerate"] = 1.0 / float(frame_info["pkt_duration_time"])
        new_frame_list.append(new_frame_info)
        frame_num += 1
    return new_frame_list


def parse_ffprobe_output(out, label, debug):
    item_list = []
    item_info = {}
    start_item = f"[{label}]"
    end_item = f"[/{label}]"
    for line in out.splitlines():
        try:
            line = line.decode("ascii").strip()
        except UnicodeDecodeError:
            # ignore the line
            continue
        if line == start_item:
            item_info = {}
        elif line == end_item:
            item_list.append(item_info)
        elif "=" in line:
            key, value = line.split("=", 1)
            item_info[key] = value
        else:
            if debug > 0:
                print(f'warning: unknown line ("{line}")')
    return item_list
