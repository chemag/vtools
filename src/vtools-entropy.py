#!/usr/bin/env python3

"""vtools-entropy.py module description."""


import argparse
import ffmpeg
import math
import numpy as np
import sys


__version__ = "0.1"

default_values = {
    "debug": 0,
    "frameto": -1,
    "framefrom": -1,
    "dump_file": None,
    "dump_list": [],
    "infile": None,
    "outfile": None,
}


def process_diff(frame2, frame1, frame_counter1, width, height, dfid, options):
    if options.framefrom != -1 and options.framefrom > frame_counter1:
        return
    if options.frameto != -1 and options.frameto < frame_counter1:
        sys.exit(0)
    # use np.subtract() instead of "-" to allow setting the dtype
    diff = np.subtract(frame2, frame1, dtype=np.int32)
    if frame_counter1 in options.dump_list:
        raw_outfile = (
            options.infile
            + f".diff_{frame_counter1}_"
            + f"{frame_counter1 + 1}.{width}x{height}.y8"
        )
        diff.astype(np.uint8).tofile(raw_outfile)
        # convert the file to png
        outfile = (
            options.infile + f".diff_{frame_counter1}_" + f"{frame_counter1 + 1}.png"
        )
        stream = ffmpeg.input(
            raw_outfile, format="rawvideo", pix_fmt="y8", s=f"{width}x{height}"
        )
        stream = ffmpeg.output(stream, outfile)
        stream = ffmpeg.overwrite_output(stream)
        out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
    if dfid is not None:
        diff.astype(np.uint8).tofile(dfid)
    mse = (diff**2).mean()
    mse /= width * height
    return frame_counter1, frame_counter1 + 1, mse, math.log10(mse) if mse != 0.0 else "-inf"


def diff_consecutive_frames(options):
    # 1. parse the file
    # https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md
    probe = ffmpeg.probe(options.infile)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    timestamp = 0
    chunk_size_sec = 10
    frame_counter = 0
    last_frame = None

    # diff video fid
    raw_outfile = f"{options.dump_file}.{width}x{height}.y8.yuv"
    dfid = open(raw_outfile, "w") if options.dump_file else None

    results = []
    while True:
        # 2. convert the video into images (luma-only)
        stream = ffmpeg.input(
            options.infile, ss=timestamp, to=timestamp + chunk_size_sec
        )
        stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="gray")
        out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 1])
        if len(frames) == 0:
            if timestamp == 0:
                # not a single frame
                print(f"error: did not find any frame in {options.infile}")
                sys.exit(-1)
            break
        if last_frame is not None:
            frame_counter, frame_counter_next, mse, log_mse = process_diff(
                frames[0], last_frame, frame_counter, width, height, dfid, options
            )
            results.append([frame_counter, frame_counter_next, mse, log_mse])
        for i in range(len(frames) - 1):
            frame_counter, frame_counter_next, mse, log_mse = process_diff(
                frames[i + 1],
                frames[i],
                frame_counter + i,
                width,
                height,
                dfid,
                options,
            )
            results.append([frame_counter, frame_counter_next, mse, log_mse])
        # keep last frame
        last_frame = frames[-1]
        timestamp += chunk_size_sec
        frame_counter += len(frames)

    # print header
    with open(options.outfile, "w") as fout:
        fout.write("frame1,frame2,mse,log10_mse\n")
        for frame_counter, frame_counter_next, mse, log_mse in results:
            fout.write(f"{frame_counter},{frame_counter_next},{mse},{log_mse}\n")

    if options.dump_file:
        # convert the file to mp4
        stream = ffmpeg.input(
            raw_outfile, format="rawvideo", pix_fmt="y8", s=f"{width}x{height}"
        )
        stream = ffmpeg.output(stream, options.dump_file)
        stream = ffmpeg.overwrite_output(stream)
        out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)


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
    diff_consecutive_frames(options)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
