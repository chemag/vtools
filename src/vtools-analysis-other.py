#!/usr/bin/env python3

"""vtools-analysis-other.py: Duration analysis using MP4Box and ffprobe."""


import argparse
import importlib
import json
import sys

vtools_analysis_mp4box = importlib.import_module("vtools-analysis-mp4box")
vtools_analysis_ffprobe = importlib.import_module("vtools-analysis-ffprobe")

FILTER_CHOICES = {
    "help": "show help options",
    "durations": "extract durations from various boxes",
    "stts": "extract stts (time-to-sample) sums per track",
    "summary": "combined duration and stts analysis",
}

default_values = {
    "debug": 0,
    "dry_run": False,
    "filter": "summary",
    "output_format": "json",
    "infile": None,
    "outfile": None,
}


def write_output(data, outfile, output_format):
    """Write data to output file in the specified format.

    Args:
        data: Data to serialize
        outfile: Output file path (use "-" for stdout)
        output_format: "json" or "duration"
    """
    if output_format == "json":
        output = json.dumps(data, indent=2) + "\n"
    elif output_format == "duration":
        output = format_duration_text(data)
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    if outfile == "-":
        sys.stdout.write(output)
    else:
        with open(outfile, "w") as f:
            f.write(output)


def format_duration_text(data):
    """Format summary data as flat key: value text.

    Args:
        data: Summary dictionary from run_analysis()

    Returns:
        Formatted text string
    """
    lines = []

    # MP4Box data
    mp4box = data.get("mp4box", {})
    if mp4box:
        lines.append("# mp4box")

        # Movie-level info
        movie = mp4box.get("movie", {})
        if movie:
            lines.append(f"mp4box_movie_timescale: {movie.get('timescale', 0)}")
            lines.append(f"mp4box_movie_duration_units: {movie.get('duration', 0)}")
            lines.append(f"mp4box_movie_duration_sec: {movie.get('duration_sec', 0)}")
            lines.append("")

        # Per-track info
        for track in mp4box.get("tracks", []):
            track_type = track.get("track_type", "unknown")
            prefix = f"mp4box_trak_{track_type}"

            # mdhd
            mdhd = track.get("mdhd", {})
            if mdhd:
                lines.append(f"{prefix}_mdhd_timescale: {mdhd.get('timescale', 0)}")
                lines.append(f"{prefix}_mdhd_duration_units: {mdhd.get('duration', 0)}")
                lines.append(
                    f"{prefix}_mdhd_duration_sec: {mdhd.get('duration_sec', 0)}"
                )

            # tkhd
            tkhd = track.get("tkhd", {})
            if tkhd:
                lines.append(f"{prefix}_tkhd_duration_units: {tkhd.get('duration', 0)}")
                lines.append(
                    f"{prefix}_tkhd_duration_sec: {tkhd.get('duration_sec', 0)}"
                )

            # edts
            edts = track.get("edts", {})
            if edts:
                lines.append(f"{prefix}_edts_duration_units: {edts.get('duration', 0)}")
                lines.append(
                    f"{prefix}_edts_duration_sec: {edts.get('duration_sec', 0)}"
                )
                for j, entry in enumerate(edts.get("elst", [])):
                    lines.append(
                        f"{prefix}_edts_elst_{j}_duration_units: {entry.get('duration', 0)}"
                    )
                    lines.append(
                        f"{prefix}_edts_elst_{j}_duration_sec: {entry.get('duration_sec', 0)}"
                    )
                    lines.append(
                        f"{prefix}_edts_elst_{j}_media_time: {entry.get('media_time', 0)}"
                    )
                    lines.append(
                        f"{prefix}_edts_elst_{j}_media_rate: {entry.get('media_rate', '1')}"
                    )

            # stts
            stts = track.get("stts", {})
            if stts:
                lines.append(
                    f"{prefix}_stts_sample_count_total: {stts.get('sample_count_total', 0)}"
                )
                lines.append(
                    f"{prefix}_stts_sample_delta_total: {stts.get('sample_delta_total', 0)}"
                )
                lines.append(
                    f"{prefix}_stts_sample_delta_average: {stts.get('sample_delta_average', 0)}"
                )
                lines.append(
                    f"{prefix}_stts_sample_delta_stddev: {stts.get('sample_delta_stddev', 0)}"
                )
                lines.append(
                    f"{prefix}_stts_duration_sec: {stts.get('duration_sec', 0)}"
                )

            # mdhd_stts_mismatch
            mismatch = track.get("mdhd_stts_mismatch", {})
            if mismatch:
                lines.append(
                    f"{prefix}_mdhd_stts_mismatch_diff_units: {mismatch.get('diff_units', 0)}"
                )
                lines.append(
                    f"{prefix}_mdhd_stts_mismatch_diff_sec: {mismatch.get('diff_sec', 0)}"
                )

            lines.append("")

    # FFprobe data
    ffprobe = data.get("ffprobe", {})
    if ffprobe:
        lines.append("# ffprobe")

        for stream in ffprobe.get("streams", []):
            stream_type = stream.get("stream_type", "unknown")
            prefix = f"ffprobe_{stream_type}"

            lines.append(f"{prefix}_codec_name: {stream.get('codec_name', '')}")
            if stream.get("profile"):
                lines.append(f"{prefix}_profile: {stream.get('profile', '')}")
            lines.append(f"{prefix}_nb_read_frames: {stream.get('nb_read_frames', 0)}")
            if stream.get("sample_rate"):
                lines.append(f"{prefix}_sample_rate: {stream.get('sample_rate', 0)}")
            if stream.get("samples_per_frame"):
                lines.append(
                    f"{prefix}_samples_per_frame: {stream.get('samples_per_frame', 0)}"
                )
            if stream.get("duration_samples"):
                lines.append(
                    f"{prefix}_duration_samples: {stream.get('duration_samples', 0)}"
                )
            lines.append(f"{prefix}_duration_sec: {stream.get('duration_sec', 0)}")
            lines.append("")

    return "\n".join(lines)


def run_analysis(infile, outfile, filter_type, output_format, debug):
    """Run the specified analysis on the input file.

    Args:
        infile: Input MP4 file path
        outfile: Output file path
        filter_type: Type of analysis to run
        output_format: Output format ("json", "duration", or "xml")
        debug: Debug level
    """
    # Handle XML mode separately - just output raw MP4Box XML
    if output_format == "xml":
        xml_path = vtools_analysis_mp4box.run_mp4box_command(infile, debug=debug)
        with open(xml_path, "r", encoding="utf-8", errors="replace") as f:
            xml_content = f.read()
        if outfile == "-":
            sys.stdout.write(xml_content)
        else:
            with open(outfile, "w") as f:
                f.write(xml_content)
        return

    # Get MP4Box analysis
    mp4box_data = vtools_analysis_mp4box.get_mp4box_info(infile, debug=debug)

    # Get ffprobe analysis
    ffprobe_data = vtools_analysis_ffprobe.get_ffprobe_info(infile, debug=debug)

    # Combine and output
    combined = {
        "mp4box": mp4box_data,
        "ffprobe": ffprobe_data,
    }
    write_output(combined, outfile, output_format)


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
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
        metavar="{%s}" % (" | ".join(f"{k}" for k in FILTER_CHOICES.keys())),
        help="%s"
        % (
            " | ".join(
                f"{k}: {v}{' [default]' if k == default_values['filter'] else ''}"
                for k, v in FILTER_CHOICES.items()
            )
        ),
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
    # Output format options (mutually exclusive)
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--json",
        action="store_const",
        dest="output_format",
        const="json",
        help="output as JSON (default)",
    )
    format_group.add_argument(
        "--duration",
        action="store_const",
        dest="output_format",
        const="duration",
        help="output duration info as flat key: value text",
    )
    format_group.add_argument(
        "--xml",
        action="store_const",
        dest="output_format",
        const="xml",
        help="output raw MP4Box XML",
    )
    parser.set_defaults(output_format=default_values["output_format"])
    # do the parsing
    options = parser.parse_args(argv[1:])
    if options.version:
        return options
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    # get outfile
    if options.outfile is None:
        options.outfile = "-"
    # print results
    if options.debug > 0:
        print(options)

    if options.filter == "help":
        for k, v in FILTER_CHOICES.items():
            print(f"  {k}: {v}")
        return

    run_analysis(
        options.infile,
        options.outfile,
        options.filter,
        options.output_format,
        options.debug,
    )


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
