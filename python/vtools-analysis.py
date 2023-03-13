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
import os
import sys


DEFAULT_NOISE_LEVEL = 50

FILTER_CHOICES = {
    "help": "show help options",
    "analyze": "analyze file",
}

default_values = {
    "debug": 0,
    "dry_run": False,
    "filter": "analyze",
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


def run_video_filter(options):
    # open the input
    # use "0" to capture from the camera
    video_capture = cv2.VideoCapture(options.infile)
    if not video_capture.isOpened():
        print(f"error: {options.infile = } is not open")
        sys.exit(-1)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    size = (width, height)
    fourcc_input = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"input: {options.infile = } {size = } {fourcc_input = } {fps = }")

    # process the input
    analysis_results = []
    prev_timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)
    prev_img = None
    while True:
        # get image
        ret, img = video_capture.read()
        if not ret:
            break
        # process image
        if options.filter == "analyze":
            # get timestamps
            timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)
            delta_timestamp = timestamp - prev_timestamp
            # get mse
            diff_msey, diff_mseu, diff_msev = calculate_diff_mse(img, prev_img)
            analysis_results.append(
                (timestamp, delta_timestamp, diff_msey, diff_mseu, diff_msev)
            )

        # update previous info
        prev_timestamp = timestamp
        prev_img = img

    # release the video objects
    video_capture.release()
    if options.filter == "analyze":
        with open(options.outfile, "w") as fd:
            fd.write(
                f"frame_num,timestamp,delta_timestamp,log10_msey,diff_msey,diff_mseu,diff_msev\n"
            )
            for frame_num, (
                ts,
                delta_timestamp,
                diff_msey,
                diff_mseu,
                diff_msev,
            ) in enumerate(analysis_results):
                log10_msey = (
                    None
                    if diff_msey is None
                    else (math.log10(diff_msey) if diff_msey != 0.0 else "-inf")
                )
                fd.write(
                    f"{frame_num},{ts},{delta_timestamp},{log10_msey},{diff_msey},{diff_mseu},{diff_msev}\n"
                )


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
        "infile",
        type=str,
        nargs="?",
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "outfile",
        type=str,
        nargs="?",
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    # get infile/outfile
    if options.infile == "-":
        options.infile = "/dev/fd/0"
    if options.outfile == "-":
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)

    run_video_filter(options)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
