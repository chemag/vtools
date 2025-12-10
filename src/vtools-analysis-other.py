#!/usr/bin/env python3

"""vtools-analysis-other.py: Duration analysis using MP4Box and ffprobe."""


import argparse
import importlib
import json
import sys
import tempfile
import xml.etree.ElementTree as ET

vtools_common = importlib.import_module("vtools-common")

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

# XML namespace for ISOBMFF schema
NS = {"mp4": "urn:mpeg:isobmff:schema:file:2016"}


def run_mp4box_command(infile, outfile=None, **kwargs):
    """Run MP4Box to extract ISOBMFF structure as XML.

    Args:
        infile: Input MP4 file path
        outfile: Output XML file path (optional, will use temp file if None)
        **kwargs: Additional options (debug, mp4box binary path)

    Returns:
        Path to the XML output file
    """
    mp4box_bin = kwargs.get("mp4box", "MP4Box")
    debug = kwargs.get("debug", 0)

    if outfile is None:
        outfile = tempfile.NamedTemporaryFile(
            prefix="vtools.mp4box.", suffix=".xml", delete=False
        ).name

    command = f"{mp4box_bin} -diso '{infile}' -out '{outfile}'"
    returncode, out, err = vtools_common.run(command, debug=debug)
    assert returncode == 0, f'error running "{command}": {err}'
    return outfile


def parse_xml_file(xml_path):
    """Parse the MP4Box XML output file.

    Args:
        xml_path: Path to the XML file

    Returns:
        Tuple of (root element, list of TrackBox elements)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracks = root.findall(".//mp4:TrackBox", NS)
    return root, tracks


def get_durations(root, tracks, debug=0):
    """Extract durations from various ISOBMFF boxes.

    Args:
        root: XML root element
        tracks: List of TrackBox elements
        debug: Debug level

    Returns:
        Dictionary containing duration information
    """
    result = {
        "movie": {},
        "tracks": [],
    }

    # Get MovieHeaderBox (mvhd) duration
    mvhd = root.find(".//mp4:MovieHeaderBox", NS)
    if mvhd is not None:
        movie_timescale = int(mvhd.get("TimeScale", 0))
        movie_duration = int(mvhd.get("Duration", 0))
        movie_duration_sec = (
            movie_duration / movie_timescale if movie_timescale > 0 else 0
        )
        result["movie"] = {
            "timescale": movie_timescale,
            "duration": movie_duration,
            "duration_sec": movie_duration_sec,
        }

    # Get per-track durations
    for i, track in enumerate(tracks):
        track_info = {
            "track_id": i + 1,
            "track_type": "unknown",
            "tkhd": {},
            "mdhd": {},
            "edts": {},
        }

        # Get handler type to determine video/audio
        hdlr = track.find(".//mp4:HandlerBox", NS)
        if hdlr is not None:
            track_info["track_type"] = hdlr.get("hdlrType", "unknown")

        # Get TrackHeaderBox (tkhd) duration
        tkhd = track.find(".//mp4:TrackHeaderBox", NS)
        if tkhd is not None:
            tkhd_duration = int(tkhd.get("Duration", 0))
            # tkhd uses movie timescale
            movie_timescale = result["movie"].get("timescale", 0)
            tkhd_duration_sec = (
                tkhd_duration / movie_timescale if movie_timescale > 0 else 0
            )
            track_info["tkhd"] = {
                "duration": tkhd_duration,
                "duration_sec": tkhd_duration_sec,
            }

        # Get MediaHeaderBox (mdhd) duration and timescale
        mdhd = track.find(".//mp4:MediaHeaderBox", NS)
        if mdhd is not None:
            mdhd_timescale = int(mdhd.get("TimeScale", 0))
            mdhd_duration = int(mdhd.get("Duration", 0))
            mdhd_duration_sec = (
                mdhd_duration / mdhd_timescale if mdhd_timescale > 0 else 0
            )
            track_info["mdhd"] = {
                "timescale": mdhd_timescale,
                "duration": mdhd_duration,
                "duration_sec": mdhd_duration_sec,
            }

        # Get EditListBox (elst) entries
        elst = track.find(".//mp4:EditListBox", NS)
        if elst is not None:
            elst_entries = []
            total_duration = 0
            movie_timescale = result["movie"].get("timescale", 0)
            for entry in elst.findall("mp4:EditListEntry", NS):
                duration = int(entry.get("Duration", 0))
                media_time = int(entry.get("MediaTime", 0))
                media_rate = entry.get("MediaRate", "1")
                # Duration is in movie timescale
                duration_sec = duration / movie_timescale if movie_timescale > 0 else 0
                total_duration += duration
                elst_entries.append(
                    {
                        "duration": duration,
                        "duration_sec": duration_sec,
                        "media_time": media_time,
                        "media_rate": media_rate,
                    }
                )
            total_duration_sec = (
                total_duration / movie_timescale if movie_timescale > 0 else 0
            )
            track_info["edts"] = {
                "duration": total_duration,
                "duration_sec": total_duration_sec,
                "elst": elst_entries,
            }

        result["tracks"].append(track_info)

    return result


def get_stts_info(root, tracks, debug=0):
    """Extract stts (time-to-sample) information per track.

    Args:
        root: XML root element
        tracks: List of TrackBox elements
        debug: Debug level

    Returns:
        List of dictionaries containing stts information per track
    """
    result = []

    for i, track in enumerate(tracks):
        track_info = {
            "track_id": i + 1,
            "track_type": "unknown",
            "timescale": 0,
            "sample_count_total": 0,
            "sample_delta_total": 0,
            "sample_delta_average": 0,
            "sample_delta_stddev": 0,
            "duration_sec": 0,
        }

        # Get handler type to determine video/audio
        hdlr = track.find(".//mp4:HandlerBox", NS)
        if hdlr is not None:
            track_info["track_type"] = hdlr.get("hdlrType", "unknown")

        # Get timescale from MediaHeaderBox
        mdhd = track.find(".//mp4:MediaHeaderBox", NS)
        if mdhd is not None:
            track_info["timescale"] = int(mdhd.get("TimeScale", 0))

        # Get stts entries from TimeToSampleBox
        stts = track.find(".//mp4:TimeToSampleBox", NS)
        if stts is not None:
            total_duration = 0
            total_units = 0
            # Collect all sample deltas for stddev calculation
            deltas = []
            for entry in stts.findall("mp4:TimeToSampleEntry", NS):
                delta = int(entry.get("SampleDelta", 0))
                count = int(entry.get("SampleCount", 0))
                total_duration += delta * count
                total_units += count
                deltas.extend([delta] * count)

            track_info["sample_count_total"] = total_units
            track_info["sample_delta_total"] = total_duration
            if total_units > 0:
                track_info["sample_delta_average"] = total_duration / total_units
                # Calculate stddev
                mean = track_info["sample_delta_average"]
                variance = sum((d - mean) ** 2 for d in deltas) / total_units
                track_info["sample_delta_stddev"] = variance**0.5
            if track_info["timescale"] > 0:
                track_info["duration_sec"] = total_duration / track_info["timescale"]

        result.append(track_info)

    return result


# Audio codec samples per frame mapping
AUDIO_SAMPLES_PER_FRAME = {
    # AAC variants
    ("aac", "LC"): 1024,
    ("aac", "HE-AAC"): 2048,
    ("aac", "HE-AACv2"): 2048,
    ("aac", "LD"): 480,
    ("aac", "ELD"): 480,
    # MP3
    ("mp3", None): 1152,
    # Opus
    ("opus", None): 960,  # default, can vary
    # FLAC (variable, use 4096 as typical)
    ("flac", None): 4096,
    # PCM (1 sample per frame)
    ("pcm_s16le", None): 1,
    ("pcm_s24le", None): 1,
    ("pcm_s32le", None): 1,
    ("pcm_f32le", None): 1,
}


def get_samples_per_frame(codec_name, profile):
    """Get the number of samples per frame for an audio codec.

    Args:
        codec_name: Audio codec name (e.g., "aac", "mp3")
        profile: Audio profile (e.g., "LC", "HE-AAC")

    Returns:
        Number of samples per frame, or None if unknown
    """
    # Try exact match first
    key = (codec_name.lower() if codec_name else None, profile)
    if key in AUDIO_SAMPLES_PER_FRAME:
        return AUDIO_SAMPLES_PER_FRAME[key]

    # Try without profile
    key = (codec_name.lower() if codec_name else None, None)
    if key in AUDIO_SAMPLES_PER_FRAME:
        return AUDIO_SAMPLES_PER_FRAME[key]

    return None


def run_ffprobe_stream_info(infile, stream_type, stream_index=0, **kwargs):
    """Run ffprobe to get stream information.

    Args:
        infile: Input file path
        stream_type: Stream type ('v' for video, 'a' for audio)
        stream_index: Stream index (default 0)
        **kwargs: Additional options (debug, ffprobe binary path)

    Returns:
        Dictionary with stream info, or None if stream doesn't exist
    """
    ffprobe_bin = kwargs.get("ffprobe", "ffprobe")
    debug = kwargs.get("debug", 0)

    stream_specifier = f"{stream_type}:{stream_index}"
    command = (
        f"{ffprobe_bin} -v error -count_frames "
        f"-select_streams {stream_specifier} "
        f"-show_entries stream=codec_name,profile,sample_rate,nb_read_frames "
        f"-of json '{infile}'"
    )

    returncode, out, err = vtools_common.run(command, debug=debug)
    if returncode != 0:
        if debug > 0:
            print(f"ffprobe error for {stream_specifier}: {err}")
        return None

    try:
        data = json.loads(out)
        streams = data.get("streams", [])
        if not streams:
            return None
        return streams[0]
    except json.JSONDecodeError:
        return None


def get_ffprobe_info(infile, **kwargs):
    """Get duration information from ffprobe for all streams.

    Args:
        infile: Input file path
        **kwargs: Additional options (debug)

    Returns:
        Dictionary with ffprobe duration information
    """
    debug = kwargs.get("debug", 0)
    result = {"streams": []}

    # Get video stream info
    video_info = run_ffprobe_stream_info(infile, "v", 0, debug=debug)
    if video_info:
        nb_frames = int(video_info.get("nb_read_frames", 0))
        # For video, duration in seconds needs frame rate which we don't have here
        # Just report frame count
        result["streams"].append(
            {
                "stream_type": "video",
                "codec_name": video_info.get("codec_name", ""),
                "profile": video_info.get("profile", ""),
                "nb_read_frames": nb_frames,
                "duration_sec": 0,  # Would need framerate to calculate
            }
        )

    # Get audio stream info
    audio_info = run_ffprobe_stream_info(infile, "a", 0, debug=debug)
    if audio_info:
        codec_name = audio_info.get("codec_name", "")
        profile = audio_info.get("profile", "")
        nb_frames = int(audio_info.get("nb_read_frames", 0))
        sample_rate = int(audio_info.get("sample_rate", 0))

        samples_per_frame = get_samples_per_frame(codec_name, profile)
        duration_samples = None
        duration_sec = 0

        if samples_per_frame and nb_frames > 0:
            duration_samples = nb_frames * samples_per_frame
            if sample_rate > 0:
                duration_sec = duration_samples / sample_rate

        stream_data = {
            "stream_type": "audio",
            "codec_name": codec_name,
            "profile": profile,
            "nb_read_frames": nb_frames,
            "sample_rate": sample_rate,
            "duration_sec": duration_sec,
        }
        if samples_per_frame:
            stream_data["samples_per_frame"] = samples_per_frame
        if duration_samples:
            stream_data["duration_samples"] = duration_samples

        result["streams"].append(stream_data)

    return result


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
        data: Summary dictionary from build_combined_summary()

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


def build_summary(durations, stts_info):
    """Build combined duration and stts summary.

    Args:
        durations: Duration dictionary from get_durations()
        stts_info: stts list from get_stts_info()

    Returns:
        Dictionary containing the combined summary
    """
    result = {
        "movie": durations["movie"],
        "tracks": [],
    }

    for i, track in enumerate(durations["tracks"]):
        stts = stts_info[i] if i < len(stts_info) else {}

        track_summary = {
            "track_id": track["track_id"],
            "track_type": track["track_type"],
            "mdhd": track["mdhd"],
            "tkhd": track["tkhd"],
            "edts": track["edts"],
            "stts": (
                {
                    "sample_count_total": stts.get("sample_count_total", 0),
                    "sample_delta_total": stts.get("sample_delta_total", 0),
                    "sample_delta_average": stts.get("sample_delta_average", 0),
                    "sample_delta_stddev": stts.get("sample_delta_stddev", 0),
                    "duration_sec": stts.get("duration_sec", 0),
                }
                if stts
                else {}
            ),
        }

        # Check for duration mismatches
        if track["mdhd"] and stts:
            mdhd_dur = track["mdhd"]["duration"]
            stts_sum = stts["sample_delta_total"]
            if mdhd_dur != stts_sum:
                diff = stts_sum - mdhd_dur
                diff_sec = (
                    diff / track["mdhd"]["timescale"]
                    if track["mdhd"]["timescale"] > 0
                    else 0
                )
                track_summary["mdhd_stts_mismatch"] = {
                    "diff_units": diff,
                    "diff_sec": diff_sec,
                }

        result["tracks"].append(track_summary)

    return result


def run_analysis(infile, outfile, filter_type, output_format, debug):
    """Run the specified analysis on the input file.

    Args:
        infile: Input MP4 file path
        outfile: Output file path
        filter_type: Type of analysis to run
        output_format: Output format ("json" or "duration")
        debug: Debug level
    """
    # Step 1: Run MP4Box to generate XML
    xml_path = run_mp4box_command(infile, debug=debug)

    if debug > 0:
        print(f"XML output: {xml_path}")

    # Step 2: Parse the XML
    root, tracks = parse_xml_file(xml_path)

    # Step 3: Get MP4Box analysis
    if filter_type == "durations":
        mp4box_data = get_durations(root, tracks, debug)
    elif filter_type == "stts":
        mp4box_data = get_stts_info(root, tracks, debug)
    elif filter_type == "summary":
        durations = get_durations(root, tracks, debug)
        stts_info = get_stts_info(root, tracks, debug)
        mp4box_data = build_summary(durations, stts_info)

    # Step 4: Get ffprobe analysis
    ffprobe_data = get_ffprobe_info(infile, debug=debug)

    # Step 5: Combine and output
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
