#!/usr/bin/env python3

"""vtools-ffmpeg.py module description.

Runs a generic filter using python ffmpeg to read the file.
"""


import argparse
import ffmpeg
import math
import numpy as np
import pandas as pd
import re
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


# helper functions


def probe_video_stream(infile):
    """Return the first video stream dict from ffprobe."""
    info = ffmpeg.probe(infile)
    vstreams = [s for s in info["streams"] if s["codec_type"] == "video"]
    if not vstreams:
        raise RuntimeError("No video stream found")
    return vstreams[0]


def parse_pix_fmt(pix_fmt):
    """
    Parse a YUV planar pixel format string like:
      yuv420p, yuv422p, yuv444p, yuv420p10le, yuv422p12le, yuv444p16le, etc.
    Returns:
      subsample (chroma_w_div, chroma_h_div),
      bit_depth (8,10,12,16),
      dtype (np.uint8 or np.uint16)
    """
    if not pix_fmt.startswith("yuv"):
        raise NotImplementedError(f"Unsupported (non-YUV) pix_fmt: {pix_fmt}")
    # subsampling
    if pix_fmt.startswith("yuv420"):
        subsample = (2, 2)
    elif pix_fmt.startswith("yuv422"):
        subsample = (2, 1)
    elif pix_fmt.startswith("yuv444"):
        subsample = (1, 1)
    else:
        raise NotImplementedError(f"Unsupported YUV sampling: {pix_fmt}")
    # bit depth
    # 8-bit formats are usually ...p (no number), higher bit depths have digits + endianness
    digits = "".join(ch for ch in pix_fmt if ch.isdigit())
    if digits == "":
        bit_depth = 8
    else:
        bit_depth = int(digits)
    if bit_depth <= 8:
        dtype = np.uint8
    else:
        # ffmpeg packs >8-bit planar as little-endian 16-bit words for rawvideo
        dtype = np.uint16

    return subsample, bit_depth, dtype


def frame_byte_size(width, height, pix_fmt):
    """Compute the total bytes per frame for planar YUV in the given pix_fmt."""
    (cw_div, ch_div), bit_depth, dtype = parse_pix_fmt(pix_fmt)
    # luma plane
    y_w, y_h = width, height
    # chroma planes
    c_w = (width + (cw_div - 1)) // cw_div
    c_h = (height + (ch_div - 1)) // ch_div
    bytes_per_sample = 1 if dtype == np.uint8 else 2
    y_bytes = y_w * y_h * bytes_per_sample
    c_bytes = c_w * c_h * bytes_per_sample
    # planar Y + U + V
    return y_bytes + c_bytes + c_bytes


def split_planes(buf, width, height, pix_fmt):
    """Return (Y, U, V) as NumPy arrays of shape (H, W) for their native plane sizes."""
    (cw_div, ch_div), bit_depth, dtype = parse_pix_fmt(pix_fmt)
    y_w, y_h = width, height
    c_w = (width + (cw_div - 1)) // cw_div
    c_h = (height + (ch_div - 1)) // ch_div
    bytes_per_sample = 1 if dtype == np.uint8 else 2
    y_sz = y_w * y_h
    c_sz = c_w * c_h
    if dtype == np.uint8:
        arr = np.frombuffer(buf, dtype=np.uint8, count=y_sz + 2 * c_sz)
    else:
        arr = np.frombuffer(buf, dtype=np.uint16, count=y_sz + 2 * c_sz)
    yarr = arr[0:y_sz].reshape((y_h, y_w))
    uarr = arr[y_sz : y_sz + c_sz].reshape((c_h, c_w))
    varr = arr[y_sz + c_sz : y_sz + 2 * c_sz].reshape((c_h, c_w))
    return yarr, uarr, varr


def plane_energy(plane):
    """Sum of squares of samples as float64."""
    # convert to float to avoid overflow in multiplication
    p = plane.astype(np.float64)
    return float(np.sum(p * p))


class FFmpegYUVFrameReader:
    """
    Read raw planar YUV frames (native pix_fmt) using ffmpeg, with true per-frame PTS
    obtained from the showinfo filter. Detached from the parent TTY so interactive
    debugging with `code.interact` works reliably.

    API:
      reader = FFmpegYUVFrameReader(infile)
      reader.process()
      yarr,uarr,varr,meta = reader.get_next_frame()  # or None at EOF
      reader.close()
    """

    def __init__(self, infile: str):
        self.infile = infile
        self.proc = None
        self.stdin = None
        self.stderr = None
        self.stdout = None
        self.stream = None
        self.width = None
        self.height = None
        self.pix_fmt = None
        self._frame_size = None
        self._frame_num = 0

    def process(self):
        self.stream = probe_video_stream(self.infile)
        self.width = int(self.stream["width"])
        self.height = int(self.stream["height"])
        self.pix_fmt = self.stream["pix_fmt"]
        self._frame_size = frame_byte_size(self.width, self.height, self.pix_fmt)

        # stream = ffmpeg.input(
        #    infile, ss=timestamp, to=timestamp + chunk_size_sec
        # )
        # stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="gray")
        # out, err = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        # frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 1])
        self.proc = (
            ffmpeg.input(self.infile, hwaccel="auto").output(
                "pipe:",
                format="rawvideo",
                pix_fmt=self.pix_fmt,
                vsync="passthrough",
                vf="showinfo",
                loglevel="info",
            )
            # Pipe ALL stdio so the child does not inherit your TTY.
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )
        self.stdin = self.proc.stdin
        self.stdout = self.proc.stdout
        self.stderr = self.proc.stderr

        # Close ffmpeg's stdin immediately; we won't feed it anything.
        # This prevents accidental reads from the parent's stdin, which would
        # interfere with interactive debugging.
        try:
            if self.stdin:
                self.stdin.close()
        except Exception:
            pass
        self.stdin = None

    _showinfo_pts_re = re.compile(r"showinfo.*pts_time:([0-9.+-eE]+)")

    def _read_next_pts_time_sec(self):
        while True:
            line = self.stderr.readline()
            if not line:
                raise EOFError
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            m = self._showinfo_pts_re.search(line)
            if m:
                return float(m.group(1))

    def get_next_frame(self):
        """
        Returns (yarr, uarr, varr, metadata) or None at end-of-stream.
        metadata includes: framenum, pts_time_sec, width, height, pix_fmt.
        """
        if self.proc is None:
            raise RuntimeError("Call process() before get_next_frame().")

        # 1. read the metadata
        try:
            pts_time_sec = self._read_next_pts_time_sec()
        except EOFError:
            pts_time_sec = None

        # 2. read the actual frame
        buf = self.stdout.read(self._frame_size)
        if not buf or len(buf) < self._frame_size:
            return None

        # 3. process the frame
        yarr, uarr, varr = split_planes(buf, self.width, self.height, self.pix_fmt)
        meta = {
            "frame_num": self._frame_num,
            "pts_time_sec": pts_time_sec,
            "width": self.width,
            "height": self.height,
            "pix_fmt": self.pix_fmt,
        }
        self._frame_num += 1
        return yarr, uarr, varr, meta

    def close(self):
        if self.stdout:
            try:
                self.stdout.close()
            except Exception:
                pass
        if self.stderr:
            try:
                self.stderr.read()  # drain remaining logs
                self.stderr.close()
            except Exception:
                pass
        if self.proc:
            try:
                self.proc.wait()
            except Exception:
                pass
        self.proc = self.stdin = self.stdout = self.stderr = None

    def __enter__(self):
        self.process()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def example_filter(infile, outfile, debug):
    cols = [
        "frame_num",
        "pts_time_sec",
        "width",
        "height",
        "pix_fmt",
    ]
    rows = []

    with FFmpegYUVFrameReader(infile) as reader:
        while True:
            # get a frame
            out = reader.get_next_frame()
            if out is None:
                break
            yarr, uarr, varr, meta = out
            # process the frame
            rows.append(
                {
                    **meta,
                }
            )
    # convert to dataframe
    df = pd.DataFrame.from_records(rows, columns=cols)
    # write to csv file
    df.to_csv(outfile, index=False)


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
    example_filter(options.infile, options.outfile, options.debug)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
