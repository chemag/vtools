#!/usr/bin/env python3

"""vtools-pyav.py module description.

Runs a generic filter using python av to read the file.
"""


import argparse
import av
import math
import numpy as np
import pandas as pd
import sys
import typing


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


def _parse_pix_fmt(fmt_name: str) -> typing.Tuple[int, int, int, np.dtype]:
    """
    Parse PyAV/FFmpeg pixel format names like:
      "yuv420p", "yuv422p10le", "yuv444p12be", "yuvj420p"
    Returns: (chroma_w_div, chroma_h_div, bit_depth, dtype)
    """
    name = fmt_name
    if name.startswith("yuvj"):
        # treat JPEG-range YUV like the corresponding yuv* (8-bit)
        name = "yuv" + name[4:]

    if not name.startswith("yuv"):
        raise NotImplementedError(f"Unsupported (non-YUV) pix_fmt: {fmt_name}")

    if name.startswith("yuv420"):
        cw_div, ch_div = 2, 2
    elif name.startswith("yuv422"):
        cw_div, ch_div = 2, 1
    elif name.startswith("yuv444"):
        cw_div, ch_div = 1, 1
    else:
        raise NotImplementedError(f"Unsupported YUV subsampling: {fmt_name}")

    digits = "".join(ch for ch in name if ch.isdigit())
    bit_depth = int(digits) if digits else 8

    if bit_depth <= 8:
        dtype = np.uint8
    else:
        # High bit depth YUV is stored in 16-bit containers with endianness suffix
        if name.endswith("be"):
            dtype = np.dtype(">u2")
        else:
            # default to little-endian if unspecified (most common in practice)
            dtype = np.dtype("<u2")

    return cw_div, ch_div, bit_depth, dtype


def _plane_view(
    plane: av.video.plane.VideoPlane,
    width: int,
    height: int,
    dtype: np.dtype,
    copy: bool,
) -> np.ndarray:
    """
    Make a 2D numpy view of a PyAV plane honoring stride (line_size).
    Slices columns to 'width' to drop right-side padding. Optionally copy.
    """
    itemsize = np.dtype(dtype).itemsize
    stride_elems = plane.line_size // itemsize
    buf = np.frombuffer(plane, dtype=dtype, count=stride_elems * height)
    arr = buf.reshape((height, stride_elems))[:, :width]
    return arr.copy() if copy else arr


class PyAVYUVFrameReader:
    """
    Frame-by-frame YUV reader using PyAV (FFmpeg bindings).

    Usage:
      with PyAVYUVFrameReader(path) as r:
          while True:
              got = r.get_next_frame()
              if got is None: break
              yarr, uarr, varr, meta = got

    Meta dict keys:
      "frame_num", "pts_time_sec", "width", "height", "pix_fmt"
    """

    def __init__(
        self,
        path: str,
        stream_index: typing.Optional[int] = None,
        copy_planes: bool = False,
    ):
        self.path = path
        self.stream_index = stream_index
        self.copy_planes = copy_planes

        self.container: typing.Optional[av.container.input.InputContainer] = None
        self.stream: typing.Optional[av.video.stream.VideoStream] = None
        self._frame_iter = None
        self._frame_num = 0

    def __enter__(self):
        self.container = av.open(self.path)
        if self.stream_index is not None:
            self.stream = self.container.streams.video[self.stream_index]
        else:
            self.stream = next(s for s in self.container.streams if s.type == "video")
        # Decode in native format; PyAV yields frames with frame.format.name
        self._frame_iter = self.container.decode(self.stream)
        self._frame_num = 0
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
        self.container = None
        self.stream = None
        self._frame_iter = None

    def get_next_frame(self):
        """
        Returns (yarr, uarr, varr, meta) or None at EOF.
        y/u/v are 2D numpy arrays shaped to their plane sizes (respecting subsampling).
        """
        if self._frame_iter is None:
            raise RuntimeError(
                "Call within a 'with PyAVYUVFrameReader(...) as reader' block."
            )

        try:
            frame: av.VideoFrame = next(self._frame_iter)
        except StopIteration:
            return None

        pix_fmt = frame.format.name  # e.g., "yuv420p10le"
        width, height = frame.width, frame.height
        cw_div, ch_div, bit_depth, dtype = _parse_pix_fmt(pix_fmt)

        y_w, y_h = width, height
        c_w = math.ceil(width / cw_div)
        c_h = math.ceil(height / ch_div)

        # True presentation timestamp (seconds) if available
        pts_time = frame.time
        if pts_time is None and frame.pts is not None and frame.time_base is not None:
            pts_time = float(frame.pts * frame.time_base)

        # Planes: 0=Y, 1=U, 2=V (planar)
        if len(frame.planes) < 3:
            raise RuntimeError(f"Expected planar YUV with 3 planes; got '{pix_fmt}'")

        yarr = _plane_view(frame.planes[0], y_w, y_h, dtype, self.copy_planes)
        uarr = _plane_view(frame.planes[1], c_w, c_h, dtype, self.copy_planes)
        varr = _plane_view(frame.planes[2], c_w, c_h, dtype, self.copy_planes)

        meta = {
            "frame_num": self._frame_num,
            "pts_time_sec": float(pts_time) if pts_time is not None else None,
            "width": width,
            "height": height,
            "pix_fmt": pix_fmt,
        }
        self._frame_num += 1
        return yarr, uarr, varr, meta


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
