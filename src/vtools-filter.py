#!/usr/bin/env python3

"""filter.py module description.

Runs generic image transformation on input images.
"""
# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html


import argparse
import cv2
import enum
import itertools
import math
import numpy as np
import os.path
import collections
import pandas as pd
import pathlib
import sys
import importlib

vtools_common = importlib.import_module("vtools-common")


Filter = collections.namedtuple("Filter", ["description"])

FILTER_DICT = {
    "help": Filter(
        description="show help options",
    ),
    "stack-waveform": Filter(
        description="stack video and waveform audio",
    ),
}


default_values = {
    "debug": 0,
    "dry_run": False,
    "proc_color": None,
    "filter": "help",
    "stack_waveform_audio_points": 600,
    "stack_waveform_output_width": 1280,
    "stack_waveform_output_video_height": 720,
    "stack_waveform_output_audio_height": 240,
    "stack_waveform_silence_samplerate": 48000,
    "infile": None,
    "infile2": None,
    "outfile": None,
    "cleanup": True,
    "logfile": None,
}


def probe_audio_start_time_seconds(in_path: str, debug: int = 0) -> float:
    # try stream.start_time first
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=start_time",
        "-of",
        "default=nw=1:nk=1",
        in_path,
    ]
    returncode, out, err = vtools_common.run(cmd, debug=debug)
    if out and out != "N/A":
        return max(0.0, float(out))

    # fallback: compute from start_pts * time_base if start_time is missing
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=start_pts,time_base",
        "-of",
        "default=nw=1",
        in_path,
    ]
    returncode, out, err = vtools_common.run(cmd, debug=debug)
    kv = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    start_pts = kv.get("start_pts")
    time_base = kv.get("time_base")  # like "1/48000"
    if start_pts and time_base and start_pts != "N/A" and time_base != "N/A":
        num, den = time_base.split("/")
        tb = float(num) / float(den)
        return max(0.0, float(start_pts) * tb)
    return 0.0


def stack_waveform(
    infile,
    outfile,
    audio_points,
    output_width,
    output_video_height,
    output_audio_height,
    silence_samplerate,
    debug,
):
    # encoding parameters (h264 CRF)
    crf = 18
    preset = "veryfast"

    # 1. check whether audio and video do not start at the same time (e.g. edit lists)
    in_path = str(pathlib.Path(infile))
    out_path = str(pathlib.Path(outfile))
    delay = probe_audio_start_time_seconds(in_path, debug=debug)
    osr = audio_points * 30
    if debug > 0:
        print(f"detected audio start delay: {delay:.6f} s")
        print(f"waveform resample rate: {osr} Hz (N={audio_points} @ 30fps)")

    # 2. create the ffmpeg filter
    # Apply drawtext directly to padded main video to preserve original timestamps.
    # Overlay showwaves separately, letting main video drive the timing.
    output_total_height = output_video_height + output_audio_height
    filter_complex = (
        f"[0:v]scale={output_width}:{output_video_height}:force_original_aspect_ratio=decrease,"
        f"pad={output_width}:{output_video_height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"pad={output_width}:{output_total_height}:0:0:black,"
        f"drawtext=text='frame\\: %{{n}} ts\\: %{{pts\\:hms}} NO AUDIO TRACK (starts at {delay:.3f}s)':"
        f"fontcolor=white:fontsize=24:fontfile=/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf:"
        f"x=(w-text_w)/2:y={output_video_height}+({output_audio_height}-text_h)/2:enable='lt(t,{delay})'[base];"
        f"anullsrc=r={silence_samplerate}:cl=mono:d={delay}[sil];"
        f"[0:a]aformat=channel_layouts=mono,asetpts=PTS-STARTPTS[a0];"
        f"[sil][a0]concat=n=2:v=0:a=1[aud];"
        f"[aud]showwaves=s={output_width}x{output_audio_height}:r=30:mode=line:scale=lin,format=rgba[wave];"
        f"[base][wave]overlay=0:{output_video_height}:eof_action=repeat:repeatlast=1,format=yuv420p[outv]"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        in_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-map",
        "0:a",
        "-shortest",
        "-c:v",
        "libx264",
        "-bf",
        "0",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        out_path,
    ]
    returncode, out, err = vtools_common.run(cmd, debug=debug)


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
        "--cleanup",
        action="store_const",
        dest="cleanup",
        const=1,
        default=default_values["cleanup"],
        help="Cleanup Raw Files%s"
        % (" [default]" if default_values["cleanup"] == 1 else ""),
    )
    parser.add_argument(
        "--full-cleanup",
        action="store_const",
        dest="cleanup",
        const=2,
        default=default_values["cleanup"],
        help="Cleanup All Files%s"
        % (" [default]" if default_values["cleanup"] == 2 else ""),
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_const",
        dest="cleanup",
        const=0,
        help="Do Not Cleanup Files%s"
        % (" [default]" if not default_values["cleanup"] == 0 else ""),
    )

    parser.add_argument(
        "--stack-waveform-audio-points",
        action="store",
        type=int,
        dest="stack_waveform_audio_points",
        default=default_values["stack_waveform_audio_points"],
        help="Number of audio points per video frame",
    )
    parser.add_argument(
        "--stack-waveform-output-width",
        action="store",
        type=int,
        dest="stack_waveform_output_width",
        default=default_values["stack_waveform_output_width"],
        help="Output video width",
    )
    parser.add_argument(
        "--stack-waveform-video-height",
        action="store",
        type=int,
        dest="stack_waveform_output_video_height",
        default=default_values["stack_waveform_output_video_height"],
        help="Output video height for the video track",
    )
    parser.add_argument(
        "--stack-waveform-audio-height",
        action="store",
        type=int,
        dest="stack_waveform_output_audio_height",
        default=default_values["stack_waveform_output_audio_height"],
        help="Output video height for the audio track",
    )
    parser.add_argument(
        "--stack-waveform-silence-samplerate",
        action="store",
        type=int,
        dest="stack_waveform_silence_samplerate",
        default=default_values["stack_waveform_silence_samplerate"],
        help="Sample rate for generated silence",
    )
    parser.add_argument(
        "--filter",
        action="store",
        type=str,
        dest="filter",
        default=default_values["filter"],
        choices=FILTER_DICT.keys(),
        metavar="{%s}" % (" | ".join("{}".format(k) for k in FILTER_DICT.keys())),
        help="%s" % (" | ".join("{}: {}".format(k, v) for k, v in FILTER_DICT.items())),
    )
    parser.add_argument(
        "-i",
        "--infile",
        action="store",
        type=str,
        dest="infile",
        default=default_values["infile"],
        metavar="input-file",
        help="input file",
    )
    parser.add_argument(
        "-j",
        "--infile2",
        action="store",
        type=str,
        dest="infile2",
        default=default_values["infile2"],
        metavar="input-file-2",
        help="input file 2",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        action="store",
        type=str,
        dest="outfile",
        default=default_values["outfile"],
        metavar="output-file",
        help="output file",
    )
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    # implement help
    if options.filter == "help":
        parser.print_help()
        sys.exit(0)
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print("version: %s" % __version__)
        sys.exit(0)
    # get infile/outfile
    if options.infile == "-" or options.infile is None:
        options.infile = "/dev/fd/0"
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(f"debug: {options}")

    if options.filter == "stack-waveform":
        stack_waveform(
            options.infile,
            options.outfile,
            options.stack_waveform_audio_points,
            options.stack_waveform_output_width,
            options.stack_waveform_output_video_height,
            options.stack_waveform_output_audio_height,
            options.stack_waveform_silence_samplerate,
            options.debug,
        )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
