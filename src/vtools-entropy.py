#!/usr/bin/env python3

"""vtools-entropy.py module description."""


import argparse
import importlib
import math
import numpy as np
import pandas as pd
import re
import sys

vtools_ffmpeg = importlib.import_module("vtools-ffmpeg")


__version__ = "0.1"

default_values = {
    "debug": 0,
    "frameto": None,
    "framefrom": None,
    "dump_file": None,
    "dump_list": [],
    "infile": None,
    "outfile": None,
}


# helper functions
def interframe_diff_energy(
    infile, framefrom=None, frameto=None, dump_list=None, debug=0
):
    columns = (
        "frame_num",
        "frame_num_prev",
        "pts_time_sec",
        "pts_time_delta_sec",
        "width",
        "height",
        "cwidth",
        "cheight",
        "pix_fmt",
        "y_mse",
        "y_log10_mse",
        "u_mse",
        "u_log10_mse",
        "v_mse",
        "v_log10_mse",
    )
    df = pd.DataFrame(columns=columns)
    yarr_prev = uarr_prev = varr_prev = None
    meta_prev = None
    with vtools_ffmpeg.FFmpegYUVFrameReader(infile) as reader:
        while True:
            # get a frame
            out = reader.get_next_frame()
            if out is None:
                break
            yarr, uarr, varr, meta = out
            produce_row = True
            # check the frame
            frame_num = meta["frame_num"]
            if yarr_prev is None:
                produce_row = False
            if framefrom is not None and frame_num < framefrom:
                produce_row = False
            if frameto is not None and frame_num > frameto:
                break
            # process the frame
            if produce_row:
                frame_num_prev = meta_prev["frame_num"]
                pts_time_sec = meta["pts_time_sec"]
                pts_time_sec_prev = meta_prev["pts_time_sec"]
                pts_time_delta_sec = pts_time_sec - pts_time_sec_prev
                width = meta["width"]
                height = meta["height"]
                pix_fmt = meta["pix_fmt"]
                # TODO(chema): add label based on dump_list
                y_mse, y_log10_mse = calculate_mse(yarr, yarr_prev, width, height)
                cwidth, cheight = uarr.shape
                u_mse, u_log10_mse = calculate_mse(uarr, uarr_prev, cwidth, cheight)
                v_mse, v_log10_mse = calculate_mse(varr, varr_prev, cwidth, cheight)
                df.loc[df.size] = [
                    frame_num,
                    frame_num_prev,
                    pts_time_sec,
                    pts_time_delta_sec,
                    width,
                    height,
                    cwidth,
                    cheight,
                    pix_fmt,
                    y_mse,
                    y_log10_mse,
                    u_mse,
                    u_log10_mse,
                    v_mse,
                    v_log10_mse,
                ]
            # update prev information
            meta_prev = meta
            yarr_prev = yarr
            uarr_prev = uarr
            varr_prev = varr
    return df


def calculate_mse(arr, arr_prev, width, height, label=None):
    # use np.subtract() instead of "-" to allow setting the dtype
    diff = np.subtract(arr, arr_prev, dtype=np.int32)
    mse = (diff**2).mean()
    mse /= width * height
    log10_mse = math.log10(mse) if mse != 0.0 else np.nan
    if label is not None:
        raw_outfile = f"{label}.gray"  # gray10???
        diff.astype(np.uint8).tofile(raw_outfile)
        # convert the file to png
        outfile = f"{label}.png"
        stream = ffmpeg.input(
            raw_outfile, format="rawvideo", pix_fmt="y8", s=f"{width}x{height}"
        )
        stream = ffmpeg.output(stream, outfile)
        stream = ffmpeg.overwrite_output(stream)
        out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
    return mse, log10_mse


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
        "--frameto",
        action="store",
        type=int,
        dest="frameto",
        default=default_values["frameto"],
        metavar="TO",
        help="Early stop",
    )
    parser.add_argument(
        "--framefrom",
        action="store",
        type=int,
        dest="framefrom",
        default=default_values["framefrom"],
        metavar="FROM",
        help="Late start",
    )
    parser.add_argument(
        "--dump-file",
        type=str,
        dest="dump_file",
        default=default_values["dump_file"],
        metavar="DUMPFILE",
        help="dump file",
    )
    parser.add_argument(
        "--dump-list",
        type=int,
        nargs="+",
        dest="dump_list",
        default=default_values["dump_list"],
        metavar="DUMPLIST",
        help="dump list",
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
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print("version: %s" % __version__)
        sys.exit(0)
    # get infile
    if options.infile == "-" or options.infile is None:
        options.infile = "/dev/fd/0"
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # do something
    df = interframe_diff_energy(
        options.infile,
        options.framefrom,
        options.frameto,
        options.dump_list,
        options.debug,
    )
    df.to_csv(options.outfile, index=False)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
