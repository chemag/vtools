#!/usr/bin/env python3

"""vtools-analysis.py module description.

Runs an video filter.
"""
# https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
# https://docs.opencv.org/3.1.0/d7/d9e/tutorial_video_write.html


import argparse
import cv2
import math
import numpy as np
import sys

import ffprobe

DEFAULT_NOISE_LEVEL = 50
PSNR_K = math.log10(2**8 - 1)

FILTER_CHOICES = {
    "help": "show help options",
    "frames": "per-frame analysis",
}

default_values = {
    "debug": 0,
    "dry_run": False,
    "add_opencv_analysis": True,
    "add_mse": False,
    "add_ffprobe_frames": True,
    "add_qp": False,
    "add_mb_type": False,
    "filter": "frames",
    "infile": None,
    "outfile": None,
}


def calculate_diff_mse(img, prev_img):
    if prev_img is None:
        return (None, None, None)
    # YCbCr diff**2 / (width*height)
    yuvimg = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    yuvprev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2YCrCb)
    diff = yuvimg.astype(np.int16) - yuvprev_img.astype(np.int16)
    height, width, channels = img.shape
    diff_mse = (diff**2).mean(axis=(1, 0)) / (width * height)
    return list(diff_mse)


def run_opencv_analysis(infile, add_mse, debug):
    # open the input
    # use "0" to capture from the camera
    video_capture = cv2.VideoCapture(infile)
    if not video_capture.isOpened():
        print(f"error: {infile = } is not open")
        sys.exit(-1)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    size = (width, height)
    fourcc_input = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if debug > 0:
        print(f"input: {infile = } {size = } {fourcc_input = } {fps = }")

    # process the input
    opencv_vals = []
    prev_timestamp_ms = video_capture.get(cv2.CAP_PROP_POS_MSEC)
    prev_img = None
    frame_num = 0
    while True:
        # get image
        ret, img = video_capture.read()
        if not ret:
            break
        # process image
        val = [frame_num, ]
        # get timestamps
        timestamp_ms = video_capture.get(cv2.CAP_PROP_POS_MSEC)
        delta_timestamp_ms = timestamp_ms - prev_timestamp_ms
        val += [timestamp_ms, delta_timestamp_ms]
        if add_mse:
            # get mse
            diff_msey, diff_mseu, diff_msev = calculate_diff_mse(img, prev_img)
            log10_msey = (
                None
                if diff_msey is None
                else (math.log10(diff_msey) if diff_msey != 0.0 else "-inf")
            )

            psnr_y = (
                None
                if ((log10_msey is None) or (log10_msey == "-inf"))
                else 20 * PSNR_K - 10 * log10_msey
            )
            val += [log10_msey, psnr_y, diff_msey, diff_mseu, diff_msev]
        opencv_vals.append(val)
        # update previous info
        prev_timestamp_ms = timestamp_ms
        prev_img = img
        frame_num += 1

    # get the tuple keys
    opencv_keys = ["frame_num", "timestamp_ms", "delta_timestamp_ms"]
    if add_mse:
        opencv_keys += ["log10_msey", "psnr_y", "diff_msey", "diff_mseu", "diff_msev"]

    # release the video objects
    video_capture.release()

    return opencv_keys, opencv_vals


def run_frame_analysis(options):
    keys, vals = [], []

    # run the opencv analysis
    if options.add_opencv_analysis:
        opencv_keys, opencv_vals = run_opencv_analysis(options.infile, options.add_mse, options.debug)
        keys, vals = opencv_keys, opencv_vals

    # add other sources of information
    if options.add_ffprobe_frames:
        ffprobe_keys, ffprobe_vals = ffprobe.get_frames_information(options.infile, debug=options.debug)
        if not keys and not vals:
            keys, vals = ffprobe_keys, ffprobe_vals
        else:
            # join the 2x sources of information
            # ensure the same number of frames in both sources
            assert len(ffprobe_vals) == len(vals), f"error: ffprobe produced {len(ffprobe_vals)} frames while previously produced {len(vals)} frames"
            # join by frame_num in both lists
            # assume frame_num-sorted lists
            keys = keys + ffprobe_keys[1:]
            vals = [v1 + v2[1:] for (v1, v2) in zip(vals, ffprobe_vals)]

    if options.add_qp:
        qp_keys, qp_vals = ffprobe.get_frames_qp_information(options.infile, debug=options.debug)
        if not keys and not vals:
            keys, vals = qp_keys, qp_vals
        else:
            # join the 2x sources of information
            # ensure the same number of frames in both sources
            assert len(qp_vals) == len(vals), f"error: ffprobe-qp produced {len(qp_vals)} frames while previously produced {len(vals)} frames"
            # join by frame_num in both lists
            # assume frame_num-sorted lists
            keys = keys + qp_keys[1:]
            vals = [v1 + v2[1:] for (v1, v2) in zip(vals, qp_vals)]

    if options.add_mb_type:
        mb_keys, mb_vals = ffprobe.get_frames_mb_information(options.infile, debug=options.debug)
        if not keys and not vals:
            keys, vals = mb_keys, mb_vals
        else:
            # join the 2x sources of information
            # ensure the same number of frames in both sources
            assert len(mb_vals) == len(vals), f"error: ffprobe-mb produced {len(mb_vals)} frames while previously produced {len(vals)} frames"
            # join by frame_num in both lists
            # assume frame_num-sorted lists
            keys = keys + mb_keys[1:]
            vals = [v1 + v2[1:] for (v1, v2) in zip(vals, mb_vals)]

    # calculate the output
    with open(options.outfile, "w") as fd:
        # write the header
        fd.write(",".join(keys) + "\n")
        for val in vals:
            fd.write(",".join(str(v) for v in val) + "\n")


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
        "--noadd-opencv-analysis",
        dest="add_opencv_analysis",
        action="store_false",
        help="Add opencv frame values to frame analysis%s"
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
        "--noadd-mse",
        dest="add_mse",
        action="store_false",
        help="Add inter-frame MSE values to frame analysis%s"
        % (" [default]" if not default_values["add_mse"] else ""),
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
        "--noadd-ffprobe-frames",
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
        "--noadd-qp",
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
        "--noadd-mb-type",
        action="store_const",
        dest="add_mb_type",
        const=False,
        help="Do not add MB type columns (h264 only)%s"
        % (" [default]" if not default_values["add_mb_type"] else ""),
    )
    parser.add_argument(
        "--filter",
        action="store",
        type=str,
        dest="filter",
        default=default_values["filter"],
        choices=FILTER_CHOICES.keys(),
        metavar="{%s}" % (" | ".join("{}".format(k) for k in FILTER_CHOICES.keys())),
        help="%s"
        % (" | ".join("{}: {}".format(k, v) for k, v in FILTER_CHOICES.items())),
    )
    parser.add_argument(
        "-i",
        "--infile",
        dest="infile",
        type=str,
        default=default_values["infile"],
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
    # get infile/outfile
    if options.infile == "-"or options.infile is None:
        options.infile = "/dev/fd/0"
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)

    run_frame_analysis(options)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
