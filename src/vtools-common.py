#!/usr/bin/env python3

"""vtools-common.py: Common vtools code.

Module that contains common code.
"""

import subprocess
import sys

CONFIG_KEY_LIST = (
    "add_opencv_analysis",
    "add_mse",
    "mse_delta",
    "add_ffprobe_frames",
    "add_qp",
    "add_mb_type",
    "frame_dups",
    "frame_dups_psnr",
    "qpextract_bin",
)


class InvalidCommand(Exception):
    pass


def run(command, **kwargs):
    debug = kwargs.get("debug", 0)
    dry_run = kwargs.get("dry_run", False)
    env = kwargs.get("env", None)
    stdin = subprocess.PIPE if kwargs.get("stdin", False) else None
    bufsize = kwargs.get("bufsize", 0)
    universal_newlines = kwargs.get("universal_newlines", False)
    default_close_fds = True if sys.platform == "linux2" else False
    close_fds = kwargs.get("close_fds", default_close_fds)
    shell = type(command) in (type(""), type(""))
    if debug > 0:
        print(f"running $ {command}")
    if dry_run:
        return 0, b"stdout", b"stderr"
    p = subprocess.Popen(
        command,
        stdin=stdin,  # noqa: P204
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=bufsize,
        universal_newlines=universal_newlines,
        env=env,
        close_fds=close_fds,
        shell=shell,
    )
    # wait for the command to terminate
    if stdin is not None:
        out, err = p.communicate(stdin)
    else:
        out, err = p.communicate()
    returncode = p.returncode
    # clean up
    del p
    # return results
    return returncode, out, err
