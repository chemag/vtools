#!/usr/bin/env python3

"""vtools-lcvm-analysis.py

Video analysis module utilizing ISOBMFF information.
Accelerated by the liblcvm C++ shared library (liblcvm_pybind.so).
"""

import argparse
import importlib
import sys

import liblcvm_pybind

vtools_version = importlib.import_module("vtools-version")


default_values = {
    "debug": 0,
    "outfile_timestamps_sort_pts": True,
    "outfile_timestamps": "outfile_timestamps.csv",
    "infile_list": [],
    "outfile": None,
}


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
        type=int,  # Change to accept a value
        dest="debug",
        default=default_values["debug"],
        help="Set debug level (e.g., -d 1)",
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

    options.outfile_timestamps = default_values["outfile_timestamps"]
    options.outfile_timestamps_sort_pts = default_values["outfile_timestamps_sort_pts"]

    liblcvm_pybind.parse_files(
        options.infile_list,
        options.outfile,
        options.outfile_timestamps,
        options.outfile_timestamps_sort_pts,
        options.debug,
    )


if __name__ == "__main__":
    main(sys.argv)
