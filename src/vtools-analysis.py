#!/usr/bin/env python3

"""vtools-analysis.py module description.

Analyzes a series of video files.
"""
# https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
# https://docs.opencv.org/3.1.0/d7/d9e/tutorial_video_write.html


import argparse
import importlib
import math
import numpy as np
import os
import pandas as pd
import sys

vtools_common = importlib.import_module("vtools-common")
vtools_ffprobe = importlib.import_module("vtools-ffprobe")
vtools_opencv = importlib.import_module("vtools-opencv")

DEFAULT_NOISE_LEVEL = 50
PSNR_K = math.log10(2**8 - 1)

FILTER_CHOICES = {
    "help": "show help options",
    "frames": "per-frame analysis",
    "summary": "per-video analysis",
}

default_values = {
    "debug": 0,
    "dry_run": False,
    "add_opencv_analysis": True,
    "add_mse": False,
    "mse_delta": 10.0,
    "add_ffprobe_frames": True,
    "add_qp": False,
    "add_mb_type": False,
    "qpextract_bin": None,
    "frame_dups": True,
    "frame_dups_psnr": 35.0,
    "filter": "frames",
    "infile_list": [],
    "outfile": None,
}


SUMMARY_FIELDS_SINGLE = (
    "pix_fmt",
    "chroma_location",
    "interlaced_frame",
    "top_field_first",
    "width",
    "height",
    "crop_left",
    "crop_right",
    "crop_bottom",
    "crop_top",
    "color_range",
    "color_transfer",
    "color_primaries",
    "color_space",
)


SUMMARY_FIELDS_AVERAGE = (
    "delta_timestamp_ms",
    "duration_time",
    "framerate",
    "pkt_size",
    "bpp",
    "bitrate",
    "qp_mean",
    "mb_type_P",
    "mb_type_A",
    "mb_type_i",
    "mb_type_I",
    "mb_type_d",
    "mb_type_D",
    "mb_type_g",
    "mb_type_G",
    "mb_type_S",
    "mb_type_>",
    "mb_type_<",
    "mb_type_X",
    "mb_stype_intra",
    "mb_stype_intra_pcm",
    "mb_stype_inter",
    "mb_stype_skip_direct",
    "mb_stype_other",
    "mb_stype_gmc",
)


def summarize(infile, df, frame_dups, frame_dups_psnr, debug):
    keys, vals = ["infile", "num_frames", "file_size_bytes"], [
        infile,
        len(df),
        os.stat(infile).st_size,
    ]
    # add single-value fields
    for key in SUMMARY_FIELDS_SINGLE:
        if key not in df:
            # field not in analysis
            continue
        assert (
            len(df[key].unique()) == 1
        ), f"error: more than 1 value for {key}: {list(df[key].unique())}"
        val = df[key].unique()[0]
        keys.append(key)
        vals.append(val)
    # add derived values
    if "pkt_duration_time" in df:
        key = "file_duration_time"
        val = df["pkt_duration_time"].astype(float).sum()
        keys.append(key)
        vals.append(val)
    if "pict_type" in df:
        num_iframes = len(df[df["pict_type"] == "I"])
        num_pframes = len(df[df["pict_type"] == "P"])
        num_bframes = len(df[df["pict_type"] == "B"])
        key = "p_i_ratio"
        val = num_pframes / num_iframes
        keys.append(key)
        vals.append(val)
        key = "b_i_ratio"
        val = num_bframes / num_iframes
        keys.append(key)
        vals.append(val)
    # add averaged fields
    for key in SUMMARY_FIELDS_AVERAGE:
        if key not in df:
            # field not in analysis
            continue
        val = df[key].astype(float).mean()
        keys.append(key)
        vals.append(val)
    # add max/min fields
    key = "qp_min"
    if key in df:
        val = df[key].min()
        keys.append(key)
        vals.append(val)
    key = "qp_max"
    if key in df:
        val = df[key].max()
        keys.append(key)
        vals.append(val)
    # get frame dup/drop info
    if frame_dups:
        frame_dups_ratio, frame_dups_average_length = get_frame_dups_info(
            df, frame_dups_psnr, debug
        )
        keys.append("frame_dups_ratio")
        vals.append(frame_dups_ratio)
        keys.append("frame_dups_average_length")
        vals.append(frame_dups_average_length)
    frame_drop_ratio, frame_drop_average_length, frame_drop_text_list = (
        get_frame_drop_info(df, debug)
    )
    keys.append("frame_drop_ratio")
    vals.append(frame_drop_ratio)
    keys.append("frame_drop_average_length")
    vals.append(frame_drop_average_length)
    keys.append("frame_drop_text_list")
    vals.append(frame_drop_text_list)
    # return summary dataframe
    df = pd.DataFrame(columns=keys)
    df.loc[len(df.index)] = vals
    return df


# count frame dups (average, variance)
# frame_dups_ratio: ratio of duplicated frames over the total
# frame_dups_average_length: average length of a dup (in frame units)
def get_frame_dups_info(df, frame_dups_psnr, debug):
    frame_total = len(df)
    frame_dups_list = list(df[df["psnr_y"] > frame_dups_psnr]["frame_num"])
    frame_dups = len(frame_dups_list)
    frame_dups_ratio = frame_dups / frame_total
    # get frame dup clumpiness factor
    last_frame_num = None
    dup_length = 0
    dup_length_list = []
    # TODO(chema): what if frame_dups_list = [..., 81, 83, 85, 87, ...]
    for frame_num in frame_dups_list:
        if last_frame_num == None:
            pass
        elif last_frame_num == frame_num - 1:
            dup_length += 1
        else:
            if dup_length > 0:
                dup_length_list.append(dup_length + 1)
                if debug > 0:
                    print(f"{frame_num=} {dup_length=}")
            dup_length = 0
        last_frame_num = frame_num
    if frame_dups == 0:
        frame_dups_average_length = 0.0
    else:
        frame_dups_average_length = sum(dup_length_list) / len(dup_length_list)
    return frame_dups_ratio, frame_dups_average_length


# count frame drops (average, variance)
# frame_drop_ratio: ratio of dropped frames over the total
# frame_drop_average_length: average length of a drop (in frame units)
def get_frame_drop_info(df, debug):
    frame_total = len(df)
    col_name = None
    if "delta_timestamp_ms" in df.columns:
        col_name = "delta_timestamp_ms"
    elif "pkt_duration_time_ms" in df.columns:
        col_name = "pkt_duration_time_ms"
    assert col_name is not None, "error: need a column with frame timestamps"
    delta_timestamp_ms_mean = df[col_name].mean()
    delta_timestamp_ms_threshold = delta_timestamp_ms_mean * 0.75 * 2
    drop_length_list = list(df[df[col_name] > delta_timestamp_ms_threshold][col_name])
    # drop_length_list: [66.68900000000022, 100.25600000000168, ...]
    frame_drop_ratio = sum(drop_length_list) / (frame_total * delta_timestamp_ms_mean)
    frame_drop_average_length = 0.0
    normalized_frame_drop_average_length = 0.0
    if drop_length_list:
        frame_drop_average_length = sum(drop_length_list) / len(drop_length_list)
        normalized_frame_drop_average_length = (
            frame_drop_average_length / delta_timestamp_ms_mean
        )
    frame_drop_text_list = " ".join(
        str(drop_length) for drop_length in drop_length_list
    )
    return frame_drop_ratio, normalized_frame_drop_average_length, frame_drop_text_list


def run_frame_analysis(options):
    # read input values
    config_dict = {
        k: v for (k, v) in vars(options).items() if k in vtools_common.CONFIG_KEY_LIST
    }
    # multiple infiles only supported in summary mode
    assert (
        len(options.infile_list) == 1 or options.filter == "summary"
    ), "error: multiple infiles only supported in summary mode"

    # process input files
    df_list = []
    for infile in options.infile_list:
        df = process_file(
            infile,
            options.add_opencv_analysis,
            options.add_mse,
            options.mse_delta,
            options.add_ffprobe_frames,
            options.add_qp,
            options.add_mb_type,
            config_dict,
            options.debug,
        )
        # implement summary mode
        if options.filter == "summary":
            df = summarize(
                infile, df, options.frame_dups, options.frame_dups_psnr, options.debug
            )
        df_list.append(df)

    if options.filter == "summary":
        # coalesce summary entries
        df = None
        for tmp_df in df_list:
            df = tmp_df if df is None else pd.concat([df, tmp_df])
    else:
        df = df_list[0]
    # write up to output file
    df.to_csv(options.outfile, index=False)


# process input
def process_file(
    infile,
    add_opencv_analysis,
    add_mse,
    mse_delta,
    add_ffprobe_frames,
    add_qp,
    add_mb_type,
    config_dict,
    debug,
):
    df = None
    if debug > 1:
        print(f"{add_opencv_analysis=}")
        print(f"{add_ffprobe_frames=}")
        print(f"{add_qp=}")
        print(f"{add_mb_type=}")

    # run opencv analysis
    if add_opencv_analysis:
        opencv_df = vtools_opencv.run_opencv_analysis(infile, add_mse, mse_delta, debug)
        # join 2x dataframes
        df = (
            opencv_df
            if df is None
            else df.join(
                opencv_df.set_index("frame_num"), on="frame_num", rsuffix="_remove"
            )
        )
        duplicated_columns = list(k for k in df.keys() if k.endswith("_remove"))
        df.drop(columns=duplicated_columns, inplace=True)

    # add other sources of information
    if add_ffprobe_frames:
        ffprobe_df = vtools_ffprobe.get_frames_information(infile, config_dict, debug)
        # join 2x dataframes
        df = (
            ffprobe_df
            if df is None
            else df.join(
                ffprobe_df.set_index("frame_num"), on="frame_num", rsuffix="_remove"
            )
        )
        duplicated_columns = list(k for k in df.keys() if k.endswith("_remove"))
        df.drop(columns=duplicated_columns, inplace=True)

    if add_qp:
        qp_df = vtools_ffprobe.get_frames_qp_information(infile, config_dict, debug)
        if qp_df is not None:
            # join 2x dataframes
            df = (
                qp_df
                if df is None
                else df.join(
                    qp_df.set_index("frame_num"), on="frame_num", rsuffix="_remove"
                )
            )
            duplicated_columns = list(k for k in df.keys() if k.endswith("_remove"))
            df.drop(columns=duplicated_columns, inplace=True)

    if add_mb_type:
        mb_df = vtools_ffprobe.get_frames_mb_information(infile, config_dict, debug)
        # join 2x dataframes
        df = (
            mb_df
            if df is None
            else df.join(
                mb_df.set_index("frame_num"), on="frame_num", rsuffix="_remove"
            )
        )
        duplicated_columns = list(k for k in df.keys() if k.endswith("_remove"))
        df.drop(columns=duplicated_columns, inplace=True)

    # fix the column types
    df = df.astype({"frame_num": int})
    return df


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    # usage = 'usage: %prog [options] arg1 arg2'
    # parser = argparse.OptionParser(usage=usage)
    # parser.print_help() to get argparse.usage (large help)
    # parser.print_usage() to get argparse.usage (just usage line)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        dest="version",
        default=False,
        help="Print version",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="count",
        dest="debug",
        default=default_values["debug"],
        help="Increase verbosity (use multiple times for more)",
    )
    parser.add_argument(
        "--quiet",
        action="store_const",
        dest="debug",
        const=-1,
        help="Zero verbosity",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        default=default_values["dry_run"],
        help="Dry run",
    )
    parser.add_argument(
        "--add-opencv-analysis",
        dest="add_opencv_analysis",
        action="store_true",
        default=default_values["add_opencv_analysis"],
        help="Add opencv frame values to frame analysis%s"
        % (" [default]" if default_values["add_opencv_analysis"] else ""),
    )
    parser.add_argument(
        "--no-add-opencv-analysis",
        dest="add_opencv_analysis",
        action="store_false",
        help="Do not add opencv frame values to frame analysis%s"
        % (" [default]" if not default_values["add_opencv_analysis"] else ""),
    )
    parser.add_argument(
        "--add-mse",
        dest="add_mse",
        action="store_true",
        default=default_values["add_mse"],
        help="Add inter-frame MSE values to frame analysis%s"
        % (" [default]" if default_values["add_mse"] else ""),
    )
    parser.add_argument(
        "--no-add-mse",
        dest="add_mse",
        action="store_false",
        help="Add inter-frame MSE values to frame analysis%s"
        % (" [default]" if not default_values["add_mse"] else ""),
    )
    parser.add_argument(
        "--mse-delta",
        action="store",
        type=float,
        dest="mse_delta",
        default=default_values["mse_delta"],
        help="MSE delta to identify duplicate frames",
    )
    parser.add_argument(
        "--add-ffprobe-frames",
        dest="add_ffprobe_frames",
        action="store_true",
        default=default_values["add_ffprobe_frames"],
        help="Add ffprobe frame values to frame analysis%s"
        % (" [default]" if default_values["add_ffprobe_frames"] else ""),
    )
    parser.add_argument(
        "--no-add-ffprobe-frames",
        dest="add_ffprobe_frames",
        action="store_false",
        help="Add ffprobe frame values to frame analysis%s"
        % (" [default]" if not default_values["add_ffprobe_frames"] else ""),
    )
    parser.add_argument(
        "--add-qp",
        action="store_const",
        default=default_values["add_qp"],
        dest="add_qp",
        const=True,
        help="Add QP columns (min, max, mean, var) (h264 only)%s"
        % (" [default]" if default_values["add_qp"] else ""),
    )
    parser.add_argument(
        "--no-add-qp",
        action="store_const",
        dest="add_qp",
        const=False,
        help="Do not add QP columns (min, max, mean, var) (h264 only)%s"
        % (" [default]" if not default_values["add_qp"] else ""),
    )
    parser.add_argument(
        "--add-mb-type",
        action="store_const",
        default=default_values["add_mb_type"],
        dest="add_mb_type",
        const=True,
        help="Add MB type columns (h264 only)%s"
        % (" [default]" if default_values["add_mb_type"] else ""),
    )
    parser.add_argument(
        "--no-add-mb-type",
        action="store_const",
        dest="add_mb_type",
        const=False,
        help="Do not add MB type columns (h264 only)%s"
        % (" [default]" if not default_values["add_mb_type"] else ""),
    )
    parser.add_argument(
        "--qpextract-bin",
        action="store",
        type=str,
        dest="qpextract_bin",
        default=default_values["qpextract_bin"],
        help="Path to the qpextract bin",
    )
    parser.add_argument(
        "--frame-dups",
        dest="frame_dups",
        action="store_true",
        default=default_values["frame_dups"],
        help="Add frame dups to frame analysis%s"
        % (" [default]" if default_values["frame_dups"] else ""),
    )
    parser.add_argument(
        "--no-frame-dups",
        dest="frame_dups",
        action="store_false",
        help="Do not add frame dups to frame analysis%s"
        % (" [default]" if not default_values["frame_dups"] else ""),
    )
    parser.add_argument(
        "--frame-dups-psnr",
        action="store",
        type=float,
        dest="frame_dups_psnr",
        default=default_values["frame_dups_psnr"],
        help="PSNR Y threshold for duplicate frame",
    )
    parser.add_argument(
        "--filter",
        action="store",
        type=str,
        dest="filter",
        default=default_values["filter"],
        choices=FILTER_CHOICES.keys(),
        metavar="{%s}" % (" | ".join(f"{k}" for k in FILTER_CHOICES.keys())),
        help="%s"
        % (
            " | ".join(
                f"{k}: {v}{' [default]' if k == default_values['filter'] else ''}"
                for k, v in FILTER_CHOICES.items()
            )
        ),
    )
    parser.add_argument(
        dest="infile_list",
        type=str,
        nargs="+",
        default=default_values["infile_list"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        type=str,
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    # force analysis coherence
    if options.add_mse:
        options.add_opencv_analysis = True
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    # get outfile
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)

    if options.filter in ("frames", "summary"):
        run_frame_analysis(options)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
