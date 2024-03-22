#!/usr/bin/env python3

"""vmaf.py module description."""


import argparse
import json
import subprocess
import sys
import tempfile


__version__ = "0.1"

ENUM_CHOICES = ["one", "two", "three"]

FUNC_CHOICES = {
    "help": "show help options",
    "copy_file": "copy input to output",
}

VMAF_MODEL = "/usr/share/model/vmaf_v0.6.1neg.json"
VMAF_MODEL = "/usr/share/model/vmaf_4k_v0.6.1.json"
VMAF_MODEL = "/usr/share/model/vmaf_v0.6.1.json"

default_values = {
    "debug": 0,
    "dry_run": False,
    "json_output": False,
    "grammar": "default_grammar.txt",
    "enum_arg": "one",
    "width": 1,
    "height": 2,
    "distorted": None,
    "reference": None,
    "outfile": None,
}


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
    p = subprocess.Popen(  # noqa: E501
        command,
        stdin=stdin,
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


def run_vmaf(distorted, reference, outfile, debug):
    # 1. run the vmaf command
    vmaf_file = tempfile.NamedTemporaryFile().name + ".json"
    command = f'ffmpeg -i {distorted} -i {reference} -lavfi libvmaf="model=path={VMAF_MODEL}:feature=name=psnr:log_fmt=json:log_path={vmaf_file}:n_threads=32" -f null -'
    ret, out, err = run(command, debug=debug)
    assert ret == 0, f"error: {err = }"
    # 2. parse the per-frame output
    with open(vmaf_file, "r") as fin:
        vmaf_json = fin.read()
    vmaf_dict = json.loads(vmaf_json)
    # 3. write the per-frame values
    wrote_header = False
    with open(outfile, "w") as fout:
        for frame in vmaf_dict["frames"]:
            frame_num = frame["frameNum"]
            if not wrote_header:
                # get column list now
                col_names = tuple(frame["metrics"].keys())
                # write header
                fout.write("frame_num," + ",".join(col_names) + "\n")
                wrote_header = True
            # write row for the frame
            vals = tuple(str(frame["metrics"][col]) for col in col_names)
            fout.write(f"{frame_num}," + ",".join(vals) + "\n")


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
        "--distorted",
        action="store",
        dest="distorted",
        type=str,
        default=default_values["distorted"],
        metavar="distorted-input-file",
        help="distorted input file",
    )
    parser.add_argument(
        "--reference",
        action="store",
        dest="reference",
        type=str,
        default=default_values["reference"],
        metavar="reference-input-file",
        help="reference input file",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        action="store",
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

    # get outfile
    if options.distorted is None:
        print("error: need a distorted file")
        sys.exit(-1)
    if options.reference is None:
        print("error: need a reference file")
        sys.exit(-1)
    if options.outfile == "-" or options.outfile is None:
        options.outfile = "/dev/fd/1"
    # print results
    if options.debug > 0:
        print(options)
    # do something
    run_vmaf(options.distorted, options.reference, options.outfile, options.debug)


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
