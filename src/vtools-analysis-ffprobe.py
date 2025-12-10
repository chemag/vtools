#!/usr/bin/env python3

"""vtools-analysis-ffprobe.py: ffprobe-based duration analysis."""


import importlib
import json

vtools_common = importlib.import_module("vtools-common")


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
