#!/usr/bin/env python3

"""vtools-analysis-liblcvm.py: liblcvm-based video analysis."""


import os
import sys

import numpy as np

# Import liblcvm (pybind11 bindings)
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "build", "lib", "liblcvm")
)

import liblcvm


def get_liblcvm_summary_info(infile, policy_file=None):
    """Get summary information from liblcvm for a video file.

    Args:
        infile: Input video file path
        policy_file: Optional path to policy file for quality checks

    Returns:
        Dictionary containing all liblcvm analysis fields
    """
    config = liblcvm.LiblcvmConfig()
    policy_content = None
    if policy_file:
        # Read policy file content
        with open(policy_file, "r") as f:
            policy_content = f.read()
    info = liblcvm.parse(infile, config)
    frame = info.get_frame()
    timing = info.get_timing()
    audio = info.get_audio()

    # All FrameInformation fields
    frame_fields = {
        "filesize": frame.get_filesize(),
        "bitrate_bps": frame.get_bitrate_bps(),
        "width": frame.get_width(),
        "height": frame.get_height(),
        "video_codec_type": frame.get_video_codec_type(),
        "horizresolution": frame.get_horizresolution(),
        "vertresolution": frame.get_vertresolution(),
        "depth": frame.get_depth(),
        "chroma_format": frame.get_chroma_format(),
        "bit_depth_luma": frame.get_bit_depth_luma(),
        "bit_depth_chroma": frame.get_bit_depth_chroma(),
        "video_full_range_flag": frame.get_video_full_range_flag(),
        "colour_primaries": frame.get_colour_primaries(),
        "transfer_characteristics": frame.get_transfer_characteristics(),
        "matrix_coeffs": frame.get_matrix_coeffs(),
    }

    # All TimingInformation fields
    timing_fields = {
        "video_freeze": timing.get_video_freeze(),
        "audio_video_ratio": timing.get_audio_video_ratio(),
        "duration_video_sec": timing.get_duration_video_sec(),
        "duration_audio_sec": timing.get_duration_audio_sec(),
        "timescale_video_hz": timing.get_timescale_video_hz(),
        "timescale_audio_hz": timing.get_timescale_audio_hz(),
        "pts_duration_sec_average": timing.get_pts_duration_sec_average(),
        "pts_duration_sec_median": timing.get_pts_duration_sec_median(),
        "pts_duration_sec_stddev": timing.get_pts_duration_sec_stddev(),
        "pts_duration_sec_mad": timing.get_pts_duration_sec_mad(),
        "num_video_frames": timing.get_num_video_frames(),
        "frame_rate_fps_median": timing.get_frame_rate_fps_median(),
        "frame_rate_fps_average": timing.get_frame_rate_fps_average(),
        "frame_rate_fps_reverse_average": timing.get_frame_rate_fps_reverse_average(),
        "frame_rate_fps_stddev": timing.get_frame_rate_fps_stddev(),
        "frame_drop_count": timing.get_frame_drop_count(),
        "frame_drop_ratio": timing.get_frame_drop_ratio(),
        "normalized_frame_drop_average_length": timing.get_normalized_frame_drop_average_length(),
        "num_video_keyframes": timing.get_num_video_keyframes(),
        "key_frame_ratio": timing.get_key_frame_ratio(),
        # Lists (optional, can be large)
        "frame_num_orig_list": timing.get_frame_num_orig_list(),
        "stts_unit_list": timing.get_stts_unit_list(),
        "ctts_unit_list": timing.get_ctts_unit_list(),
        "dts_sec_list": timing.get_dts_sec_list(),
        "pts_sec_list": timing.get_pts_sec_list(),
        "pts_duration_sec_list": timing.get_pts_duration_sec_list(),
        "pts_duration_delta_sec_list": timing.get_pts_duration_delta_sec_list(),
    }

    # All AudioInformation fields
    audio_fields = {
        "audio_type": audio.get_audio_type(),
        "channel_count": audio.get_channel_count(),
        "sample_rate": audio.get_sample_rate(),
        "sample_size": audio.get_sample_size(),
    }

    # Top-level fields
    top_fields = {
        "filename": info.get_filename(),
    }

    # Merge all fields
    all_fields = {}
    all_fields.update(top_fields)
    all_fields.update(frame_fields)
    all_fields.update(timing_fields)
    all_fields.update(audio_fields)

    # Run policy checks if policy was provided
    policy_fields = {}
    if policy_content:
        try:
            # Prepare keys and values for policy runner
            # Filter out list/array fields - policy runner only accepts scalar values
            keys = []
            vals = []
            for key, value in all_fields.items():
                # Skip list fields
                if isinstance(value, (list, np.ndarray)):
                    continue
                keys.append(key)
                vals.append(value)

            # Call policy_runner
            result, warn_list, error_list, version = liblcvm.policy_runner(
                policy_content, keys, vals
            )

            # Add policy results to fields
            policy_fields = {
                "policy_version": version,
                "warn_list": "; ".join(warn_list) if warn_list else "",
                "error_list": "; ".join(error_list) if error_list else "",
            }
        except Exception as e:
            policy_fields = {
                "policy_version": "",
                "warn_list": "",
                "error_list": f"Policy execution failed: {str(e)}",
            }

    all_fields.update(policy_fields)
    return all_fields
