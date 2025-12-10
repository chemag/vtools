#!/usr/bin/env python3
"""vtools-analysis.py module description.
Analyzes a series of video files.
"""
import argparse
import importlib
import math
import sys

import pandas as pd

vtools_common = importlib.import_module("vtools-common")
vtools_ffprobe = importlib.import_module("vtools-ffprobe")
vtools_opencv = importlib.import_module("vtools-opencv")
vtools_version = importlib.import_module("vtools-version")
vtools_analysis_liblcvm = importlib.import_module("vtools-analysis-liblcvm")

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
    "dump_audio_info": False,
    "policy_file": None,
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


def summarize(infile, df, config_dict, debug):
    # 1. Get summary info from liblcvm
    try:
        lcvm_info = vtools_analysis_liblcvm.get_liblcvm_summary_info(infile)
    except Exception as e:
        if debug > 0:
            print(f"liblcvm error: {e}")
        lcvm_info = {}

    # 2. Start with liblcvm fields
    keys = list(lcvm_info.keys())
    vals = list(lcvm_info.values())

    # 3. Add number of frames from pandas (if available)
    keys.append("num_frames")
    vals.append(len(df))

    # 4. Add single-value fields from pandas if not in liblcvm
    for key in SUMMARY_FIELDS_SINGLE:
        if key not in df or key in lcvm_info:
            continue
        unique_vals = df[key].unique()
        if len(unique_vals) == 1:
            keys.append(key)
            vals.append(unique_vals[0])
        else:
            if debug > 0:
                print(f"Warning: more than 1 value for {key}: {list(unique_vals)}")

    # 5. Add derived values from pandas if not in liblcvm
    if "pkt_duration_time_ms" in df and "duration_video_sec" not in lcvm_info:
        keys.append("video_duration_time")
        vals.append(df["pkt_duration_time_ms"].astype(float).sum() / 1000)

    if "pict_type" in df:
        num_iframes = len(df[df["pict_type"] == "I"])
        num_pframes = len(df[df["pict_type"] == "P"])
        num_bframes = len(df[df["pict_type"] == "B"])
        if num_iframes > 0:
            keys.append("p_i_ratio")
            vals.append(num_pframes / num_iframes)
            keys.append("b_i_ratio")
            vals.append(num_bframes / num_iframes)

    # 6. Add averaged fields from pandas if not in liblcvm
    for key in SUMMARY_FIELDS_AVERAGE:
        if key not in df or key in lcvm_info:
            continue
        vals.append(df[key].astype(float).mean())
        keys.append(key)

    # 7. Add max/min fields
    for key in ["qp_min", "qp_max"]:
        if key in df:
            val = df[key].min() if key == "qp_min" else df[key].max()
            keys.append(key)
            vals.append(val)

    # 8. Frame dups info (still pandas-based)
    if config_dict.get("frame_dups", False):
        frame_dups_ratio, frame_dups_average_length, frame_dups_text_list = (
            get_frame_dups_info(df, config_dict["frame_dups_psnr"], debug)
        )
        keys.extend(
            ["frame_dups_ratio", "frame_dups_average_length", "frame_dups_text_list"]
        )
        vals.extend([frame_dups_ratio, frame_dups_average_length, frame_dups_text_list])

    # 9. Frame drop info: now from liblcvm, not pandas (already in lcvm_info)

    # 10. Optionally, add ffprobe audio info if requested
    if config_dict.get("dump_audio_info", False):
        sample_rate, bitrate, duration = vtools_ffprobe.get_audio_info(infile)
        keys.extend(
            [
                "audio_sample_rate_ffprobe",
                "audio_bitrate_ffprobe",
                "audio_duration_time_ffprobe",
            ]
        )
        vals.extend([sample_rate, bitrate, duration])

    # 11. Return summary dataframe
    summary_df = pd.DataFrame([vals], columns=keys)
    return summary_df


# count frame dups (average, variance)
def get_frame_dups_info(df, frame_dups_psnr, debug):
    frame_total = len(df)
    frame_dups_list = list(df[df["psnr_y"] > frame_dups_psnr]["frame_num"])
    frame_dups = len(frame_dups_list)
    frame_dups_ratio = frame_dups / frame_total if frame_total > 0 else 0.0
    last_frame_num = None
    dup_length = 0
    dup_length_list = []
    for frame_num in frame_dups_list:
        if last_frame_num is None:
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
    if frame_dups == 0 or not dup_length_list:
        frame_dups_average_length = 0.0
    else:
        frame_dups_average_length = sum(dup_length_list) / len(dup_length_list)
    frame_dups_text_list = " ".join(str(frame_dup) for frame_dup in frame_dups_list)
    return frame_dups_ratio, frame_dups_average_length, frame_dups_text_list


def run_frame_analysis(infile_list, outfile, the_filter, config_dict, debug):
    assert (
        len(infile_list) == 1 or the_filter == "summary"
    ), "error: multiple infiles only supported in summary mode"
    if the_filter == "summary":
        # Use liblcvm for all files
        summary_list = []
        for infile in infile_list:
            try:
                summary = get_liblcvm_full_summary(infile)
                summary_list.append(summary)
            except Exception as e:
                if debug > 0:
                    print(f"liblcvm error for {infile}: {e}")
        df = pd.DataFrame(summary_list)
        df.to_csv(outfile, index=False)
    else:
        # Per-frame analysis: keep old logic
        df_list = []
        for infile in infile_list:
            df = process_file(
                infile,
                config_dict,
                debug,
            )
            df_list.append(df)
        df = df_list[0]
        df.to_csv(outfile, index=False)


def process_file(
    infile,
    config_dict,
    debug,
):
    df = None
    if debug > 1:
        for key in vtools_common.CONFIG_KEY_LIST:
            print(f'config_dict["{key}"]: {config_dict[key]}')
    # run opencv analysis
    if config_dict["add_opencv_analysis"]:
        opencv_df = vtools_opencv.run_opencv_analysis(
            infile, config_dict["add_mse"], config_dict["mse_delta"], debug
        )
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
    if config_dict["add_ffprobe_frames"]:
        ffprobe_df = vtools_ffprobe.get_frames_information(infile, config_dict, debug)
        df = (
            ffprobe_df
            if df is None
            else df.join(
                ffprobe_df.set_index("frame_num"), on="frame_num", rsuffix="_remove"
            )
        )
        duplicated_columns = list(k for k in df.keys() if k.endswith("_remove"))
        df.drop(columns=duplicated_columns, inplace=True)
    if config_dict["add_qp"]:
        qp_df = vtools_ffprobe.get_frames_qp_information(infile, config_dict, debug)
        if qp_df is not None:
            df = (
                qp_df
                if df is None
                else df.join(
                    qp_df.set_index("frame_num"), on="frame_num", rsuffix="_remove"
                )
            )
            duplicated_columns = list(k for k in df.keys() if k.endswith("_remove"))
            df.drop(columns=duplicated_columns, inplace=True)
    if config_dict["add_mb_type"]:
        mb_df = vtools_ffprobe.get_frames_mb_information(infile, config_dict, debug)
        df = (
            mb_df
            if df is None
            else df.join(
                mb_df.set_index("frame_num"), on="frame_num", rsuffix="_remove"
            )
        )
        duplicated_columns = list(k for k in df.keys() if k.endswith("_remove"))
        df.drop(columns=duplicated_columns, inplace=True)
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
        action="version",
        version=vtools_version.__version__,
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
    parser.add_argument(
        "--dump-audio-info",
        dest="dump_audio_info",
        action="store_true",
        default=default_values["dump_audio_info"],
        help="Dump audio information%s"
        % (" [default]" if default_values["dump_audio_info"] else ""),
    )
    parser.add_argument(
        "--no-dump-audio-info",
        dest="dump_audio_info",
        action="store_false",
        help="Do not dump audio information%s"
        % (" [default]" if not default_values["dump_audio_info"] else ""),
    )
    parser.add_argument(
        "-p",
        "--policy",
        dest="policy_file",
        type=str,
        default=default_values["policy_file"],
        metavar="policy-file",
        help="policy file for video quality checks (requires liblcvm built with ADD_POLICY=ON)",
    )

    # do the parsing
    options = parser.parse_args(argv[1:])
    # force analysis coherence
    if options.add_mse:
        options.add_opencv_analysis = True
    return options


def get_config_dict(options):
    # read input values
    config_dict = {
        k: v for (k, v) in vars(options).items() if k in vtools_common.CONFIG_KEY_LIST
    }
    # TODO(chema): check config coherence
    return config_dict


def main(argv):
    # Parse options
    options = get_options(argv)
    # Set output file
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # Print options if debugging
    if options.debug > 0:
        print(options)
    config_dict = get_config_dict(options)
    if options.filter == "summary":
        # liblcvm-only summary mode
        summary_rows = []
        for infile in options.infile_list:
            try:
                row = vtools_analysis_liblcvm.get_liblcvm_summary_info(
                    infile, options.policy_file
                )
                summary_rows.append(row)
            except Exception as e:
                print(f"liblcvm error for {infile}: {e}")
        if summary_rows:
            import csv

            with open(options.outfile, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow(row)
    elif options.filter == "frames":
        # Per-frame analysis: keep old logic
        run_frame_analysis(
            options.infile_list,
            options.outfile,
            options.filter,
            config_dict,
            options.debug,
        )
    else:
        print(f"Unknown filter: {options.filter}")


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
