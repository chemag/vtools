#!/usr/bin/env python3

"""vtools-analysis-mp4box.py: MP4Box-based ISOBMFF duration analysis."""


import importlib
import tempfile
import xml.etree.ElementTree as ET

vtools_common = importlib.import_module("vtools-common")

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


def sanitize_xml(content):
    """Remove invalid XML characters from content.

    Args:
        content: String containing XML content

    Returns:
        Sanitized string with invalid XML characters removed
    """

    # Valid XML 1.0 characters:
    # #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
    def is_valid_xml_char(c):
        codepoint = ord(c)
        return (
            codepoint == 0x9
            or codepoint == 0xA
            or codepoint == 0xD
            or (0x20 <= codepoint <= 0xD7FF)
            or (0xE000 <= codepoint <= 0xFFFD)
            or (0x10000 <= codepoint <= 0x10FFFF)
        )

    return "".join(c for c in content if is_valid_xml_char(c))


def parse_xml_file(xml_path):
    """Parse the MP4Box XML output file.

    Args:
        xml_path: Path to the XML file

    Returns:
        Tuple of (root element, list of TrackBox elements)
    """
    # Read and sanitize the XML content to handle invalid characters
    with open(xml_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    content = sanitize_xml(content)

    root = ET.fromstring(content)
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


def get_mp4box_info(infile, **kwargs):
    """Get complete MP4Box analysis for a file.

    Args:
        infile: Input file path
        **kwargs: Additional options (debug)

    Returns:
        Dictionary with MP4Box analysis (summary format)
    """
    debug = kwargs.get("debug", 0)

    # Run MP4Box to generate XML
    xml_path = run_mp4box_command(infile, debug=debug)

    if debug > 0:
        print(f"XML output: {xml_path}")

    # Parse the XML
    root, tracks = parse_xml_file(xml_path)

    # Build summary
    durations = get_durations(root, tracks, debug)
    stts_info = get_stts_info(root, tracks, debug)
    return build_summary(durations, stts_info)
