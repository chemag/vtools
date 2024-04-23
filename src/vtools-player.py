#!/usr/bin/python
"""Videoplayer.

Play video, step and extract frames.
Can also show waveform if audio is present.
"""

import argparse
import asyncio
import csv
import cv2
import importlib
import math
import numpy as np
import os
import re
import soundfile
import sys
import tempfile
import threading
import timeit


vtools_common = importlib.import_module("vtools-common")

default_values = {
    "debug": 0,
    "dry_run": False,
    "seek": 0,
    "format": None,
    "resolution": None,
    "label": None,
    "fps": -1,
}


clickPoints = []
activeVideo = None
BORDER = 30

PAUSE = 1
PREVIOUS_FRAME = 2
NEXT_FRAME = 3
PREVIOUS_SECOND = 4
NEXT_SECOND = 5
QUIT = 6
WRITE = 7
MARKER_1 = 8
MARKER_2 = 9
MARKER_3 = 10
ZOOM_IN = 11
ZOOM_OUT = 12
AUDIO_AMP_INCR = 13
AUDIO_AMP_DECR = 14
AUDIO_WAVE_LEFT = 15
AUDIO_WAVE_RIGHT = 16
WAVE_FIND_MAX = 17

shortcuts = {
    ord(" "): PAUSE,
    ord("a"): PREVIOUS_FRAME,
    ord(","): PREVIOUS_FRAME,
    ord("s"): NEXT_FRAME,
    ord("."): NEXT_FRAME,
    ord("z"): PREVIOUS_SECOND,
    ord("x"): NEXT_SECOND,
    ord("1"): MARKER_1,
    ord("2"): MARKER_2,
    ord("3"): MARKER_3,
    ord("i"): ZOOM_IN,
    ord("o"): ZOOM_OUT,
    ord("l"): AUDIO_WAVE_RIGHT,
    ord("k"): AUDIO_WAVE_LEFT,
    ord("q"): QUIT,
    ord("w"): WRITE,
    ord("m"): AUDIO_AMP_INCR,
    ord("n"): AUDIO_AMP_DECR,
    ord("v"): WAVE_FIND_MAX,
}

descr = {
    PAUSE: "pause",
    PREVIOUS_FRAME: "previous frame",
    NEXT_FRAME: "next frame",
    PREVIOUS_SECOND: "previous second",
    NEXT_SECOND: "next second",
    QUIT: "quit",
    WRITE: "write",
    MARKER_1: "marker 1",
    MARKER_2: "marker 2",
    MARKER_3: "marker 3",
    ZOOM_IN: "zoom in",
    ZOOM_OUT: "zoom out",
    AUDIO_AMP_INCR: "audio amp increase",
    AUDIO_AMP_DECR: "audio amp decrease",
    AUDIO_WAVE_RIGHT: "audio wave right",
    AUDIO_WAVE_LEFT: "audio wave left",
    WAVE_FIND_MAX: "straight sampleing or plot absolute max sampled range",
}

c1 = (255, 255, 255)
c2 = (200, 100, 100)
c3 = (0, 0, 255)


class AudioWaveform:
    def __init__(self, filename, width, height, debug):
        self.filename = filename
        self.width = width
        self.height = height
        self.show_sec = 4
        # extract the audio stream into a single, mono audio file
        if debug > 0:
            print(f"Create audio file {audiofile}")
        audiofile = tempfile.NamedTemporaryFile(prefix="vtools-player.").name + ".wav"
        command = f"ffmpeg -i {filename} -ac 1 {audiofile}"
        returncode, out, err = vtools_common.run(command, debug=debug)
        if returncode != 0:
            raise vtools_common.InvalidCommand(f'error running "{command}"')
        # check file
        self.soundfile = soundfile.SoundFile(f"{audiofile}")
        self.samplerate = self.soundfile.samplerate
        self.waveData = self.soundfile.read()
        # TODO(johan): remove file?
        self.start = 0
        self.end = len(self.waveData)
        self.last_draw_time = 0
        self.markers = [None] * 3
        self.amplification = 1
        self.find_max = True
        self.plot_wav_data()

    def get_wave_data(self):
        return self.waveData

    def plot_wav_data(self):
        # TODO: aliasing?
        img = np.zeros(
            (int(self.height + 2 * BORDER), int(self.width + 2 * BORDER), 3), np.uint8
        )
        nbr = self.samplerate * self.show_sec
        if self.start + len(self.waveData) > self.start + nbr:
            self.end = int(self.start + self.samplerate * self.show_sec)
        else:
            self.end = int(len(self.waveData) - self.start)
        self.samples = self.end - self.start
        data = (
            self.height
            * (self.amplification * self.waveData[self.start : self.end] + 1)
            / 2
        )
        self.xstep = self.samples / self.width
        mid = int(self.height / 2) + BORDER
        # cv2.line(img, (BORDER, mid), (self.width - 2 * BORDER, mid), c1, 2);

        yval_ = 0
        index = 0
        x = 0
        yval_ = mid
        prev = 0
        while index < len(data) - 1:
            val = data[int(index)]

            if self.find_max and index > 0:
                mi = np.argmax(data[prev : int(index)])
                val = data[mi + prev]
            yval = int(val)
            cv2.line(
                img,
                (BORDER + x, BORDER + yval_),
                (BORDER + x + 1, BORDER + yval),
                c2,
                1,
            )
            yval_ = yval
            prev = int(index)
            index += self.xstep
            x += 1

        time = round(self.start / self.samplerate, 2)
        cv2.putText(
            img,
            f"{time:.2f} s",
            (BORDER, BORDER + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            c1,
            1,
            cv2.LINE_AA,
        )
        cv2.line(img, (BORDER, BORDER), (BORDER, self.height), c2, 1)

        time = round(self.end / self.samplerate, 2)
        cv2.putText(
            img,
            f"{time:.2f} s",
            (self.width, BORDER + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            c1,
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            img,
            (self.width + BORDER, BORDER),
            (self.width + BORDER, self.height),
            c2,
            1,
        )
        self.img = img

    def get_wave_image(self, timestamp, text):
        # Check time

        sample = timestamp * self.samplerate
        self.last_draw_time = timestamp

        if sample > self.end:
            if sample > len(self.waveData):
                return self.img
            self.start += int(
                self.show_sec * self.samplerate
            )  # int(sample/self.samplerate) * self.samplerate
            self.end = self.start + self.show_sec * self.samplerate
            self.plot_wav_data()

        elif sample < self.start:
            self.start -= int(
                self.show_sec * self.samplerate
            )  # int(sample/self.samplerate) * self.samplerate
            self.end = self.start + self.show_sec * self.samplerate
            self.plot_wav_data()

        sample = sample - self.start
        step = 1 / self.xstep
        x = int(sample * step)
        if self.img is not None and len(self.img) > 0:
            img_ = self.img.copy()
            cv2.line(img_, (BORDER + x, BORDER), (BORDER + x, self.height), c3, 1)
            timestamp = round(timestamp, 3)
            cv2.putText(
                img_,
                f"{timestamp:0>4.3f} sec, {text}",
                (int(self.width / 2), BORDER + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                c1,
                1,
                cv2.LINE_AA,
            )
            # markers
            for i in range(len(self.markers)):
                if self.markers[i] is not None:
                    sample = self.markers[i] * self.samplerate
                    if sample > self.start and sample < self.end:
                        sample = sample - self.start
                        x = int(sample * step)
                        cv2.line(
                            img_,
                            (BORDER + x, BORDER * 2),
                            (BORDER + x, self.height - 2 * BORDER),
                            c1,
                            1,
                        )
                        ts = round(self.markers[i], 3)
                        cv2.putText(
                            img_,
                            f"{ts:0>4} sec",
                            (BORDER + x + 10, (3 + i) * BORDER + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            c1,
                            1,
                            cv2.LINE_AA,
                        )

            return img_
        return None

    def set_marker(self, number):
        if number - 1 < len(self.markers):
            if self.markers[number - 1] == self.last_draw_time:
                self.markers[number - 1] = None
            else:
                self.markers[number - 1] = self.last_draw_time
        else:
            print(f"need more markers {len(self.markers)}, {number = }")

    def zoom_in(self):
        self.show_sec -= 0.1
        if self.show_sec <= 0:
            self.show_sec = 0.1
        self.plot_wav_data()

    def zoom_out(self):
        self.show_sec += 0.1
        self.plot_wav_data()

    def amp_decr(self):
        self.amplification -= 0.1
        if self.amplification <= 1:
            self.amplification = 1
        self.plot_wav_data()

    def amp_incr(self):
        self.amplification += 0.1
        self.plot_wav_data()

    def wave_left(self):
        self.start -= int(self.samplerate / 10)
        if self.start < 0:
            self.start = 0
        self.plot_wav_data()

    def wave_right(self):
        self.start += int(self.samplerate / 10)
        if self.start + self.show_sec * self.samplerate > len(self.waveData):
            self.start = len(self.waveData) - self.show_sec * self.samplerate
        if self.start > self.last_draw_time * self.samplerate:
            self.start = int(self.last_draw_time * self.samplerate)
        self.plot_wav_data()

    def set_find_max(self, value):
        self.find_max = value
        self.plot_wav_data()


class VideoCaptureYUV:
    def __init__(self, filename, size, colorformat):
        self.height, self.width = size
        self.frame_len = math.ceil(self.width * self.height * 3 / 2)
        self.f = open(filename, "rb")
        self.shape = (int(self.height * 1.5), self.width)
        self.colorformat = colorformat

    def read_raw(self):
        try:
            raw = self.f.read(self.frame_len)
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, self.colorformat)
        return ret, bgr


class Video:
    def __init__(self, cap, rot_90, filename):
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.cap = cap
        self.rot_90 = rot_90
        self.filename = filename
        cv2.namedWindow(self.filename)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.audio = None
        self.current = 0

    async def seek_frame(self, framenum):
        if framenum > self.framecount:
            framenum = self.framecount - 1
        self.current = self.cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
        await self.next()
        return framenum

    async def next(self):
        if self.current < self.framecount:
            status, img_ = self.cap.read()
            if status:
                self.current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                if self.rot_90:
                    self.img = cv2.rotate(img_, cv2.ROTATE_90_CLOCKWISE)
                else:
                    self.img = img_
            else:
                print("Failed to capture frame")
        return self.img

    def current_frame(self):
        return self.img

    def get_filename(self):
        return self.filename

    def set_audio_wavform(self, audioWaveForm):
        self.audio = audioWaveForm

    async def show_current_frame(self, showInfo=False):
        cv2.imshow(self.filename, self.img)
        if self.audio and self.current < self.framecount:
            time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            wave_ = self.audio.get_wave_image(time, f"frame: {int(self.current):0>3}")
            if wave_ is not None and len(wave_) > 0:
                cv2.imshow(self.filename + "_audio", wave_)

    def add_overlay_region(self, topleft, bottomright):
        self.topleft = topleft
        self.bottomright = bottomright

    def release():
        cap.release()


def key_press_check():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        print("With listener")
        listener.join()


async def analyze_files(files, raw, options):
    global videoList
    # all files needs to have same settings if raw
    if options.resolution is not None and len(options.resolution) > 0:
        raw = True
        size_ = options.resolution.split("x")
        size = [int(size_[1]), int(size_[0])]

    vnum = len(files)
    print(f"Number of videoList: {vnum}")
    videoList = [None] * vnum
    audioList = [None] * vnum
    img = [None] * vnum
    oldImg = [None] * vnum
    widths = []
    waveimg = []
    count = 0
    if raw:
        for filename in files:
            print(
                "Cap raw {:s} {:s} {:s}".format(
                    options.input, str(size), str(pixFormat)
                )
            )
            cap = VideoCaptureYUV(filename, size, pixFormat)
            videoList[count] = Video(cap, options.rot_90, filename)
            cv2.namedWindow(filename)
            count += 1
    else:
        for filename in files:
            print(f"Count: {count}, open: {filename}")
            cap = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)
            videoList[count] = Video(cap, options.rot_90, filename)
            # get the number of audio streams
            command = f"ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 {filename}"
            returncode, out, err = vtools_common.run(command, debug=options.debug)
            if returncode != 0:
                raise vtools_common.InvalidCommand(f'error running "{command}"')
            num_audio_streams = 0 if not out else int(out)
            # add audio streams to list (if existing)
            if num_audio_streams > 0:
                audio = AudioWaveform(filename, 600, 300, options.debug)
                videoList[count].set_audio_wavform(audio)
            cv2.namedWindow(filename)
            count += 1

    total = 0

    clickPoints.append((0, 0))
    clickPoints.append((0, 0))

    done = False
    capture = True

    meanB_box = 0
    meanG_box = 0
    meanR_box = 0
    status = True
    img_ = None
    pause = False
    label = ""
    if options.label is not None:
        label = options.label.replace(" ", "_")

    amp = 2
    wavenum = -1
    old_shape = (0, 0, 0)
    extract_list = []
    if options.extract:
        extract_list = [int(x) for x in options.extract.split(",")]
    tasks = [None] * len(videoList)
    while not done:
        videolock = asyncio.Lock()
        start = timeit.default_timer()

        if capture:
            counter = 0
            for video in videoList:
                async with videolock:
                    tasks[counter] = asyncio.create_task(video.next())
        capDone = timeit.default_timer()
        frameduration = 1 / video.fps
        if options.fps > 0:
            frameduration = 1 / options.fps

        for task in tasks:
            if task is not None:
                await task

        delay = frameduration - (capDone - start)

        if delay <= 0 or options.fps == 0:
            delay = 1
        if pause:
            delay = -1
        start = timeit.default_timer()
        k = cv2.waitKey(int(delay * 1000))  # 30fps?

        if k == ord("m"):
            amp += 1
        elif k == ord("n"):
            amp -= 1
        step = 0

        command = shortcuts.get(k, None)
        if command is None and k == 27:
            command = QUIT

        if command == QUIT:
            done = True
        elif command == PAUSE:
            pause = not pause
        elif command == NEXT_FRAME:
            step = 1
        elif command == PREVIOUS_FRAME:
            for video in videoList:
                await video.seek_frame(video.current - 2)
                capture = False
        elif command == NEXT_SECOND:
            for video in videoList:
                await video.seek_frame(video.current + video.fps)  # TODO: framerate
                capture = False
        elif command == PREVIOUS_SECOND:
            for video in videoList:
                await video.seek_frame(video.current - video.fps - 1)
                capture = False
        elif command == MARKER_1:
            for video in videoList:
                if video.audio:
                    video.audio.set_marker(1)
        elif command == MARKER_2:
            for video in videoList:
                if video.audio:
                    video.audio.set_marker(2)
        elif command == MARKER_3:
            for video in videoList:
                if video.audio:
                    video.audio.set_marker(3)
        elif command == ZOOM_IN:
            for video in videoList:
                if video.audio:
                    video.audio.zoom_in()
        elif command == ZOOM_OUT:
            for video in videoList:
                if video.audio:
                    video.audio.zoom_out()
        elif command == AUDIO_AMP_INCR:
            for video in videoList:
                if video.audio:
                    video.audio.amp_incr()
        elif command == AUDIO_AMP_DECR:
            for video in videoList:
                if video.audio:
                    video.audio.amp_decr()
        elif command == AUDIO_WAVE_LEFT:
            for video in videoList:
                if video.audio:
                    video.audio.wave_left()
        elif command == AUDIO_WAVE_RIGHT:
            for video in videoList:
                if video.audio:
                    video.audio.wave_right()
        elif command == WAVE_FIND_MAX:
            for video in videoList:
                if video.audio:
                    video.audio.set_find_max(not audio.find_max)

        if pause and step == 0:
            capture = False
        else:
            capture = True
        step = 0

        if k == WRITE or int(video.current) in extract_list:
            print("Call save image")
            for video in videoList:
                img = video.current_frame()
                filename = video.get_filename()
                cv2.imwrite(
                    f"{filename}_fr_{video.current}_{img.shape[1]}x{img.shape[0]}.png",
                    img,
                )
        for video in videoList:
            await video.show_current_frame(True)
    try:
        for video in videoList:
            video.release()
    except Exception as e:
        pass


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
    commands = [
        f"\n'{chr(x)}' - {descr.get(shortcuts.get(x))}\n" for x in shortcuts.keys()
    ]
    command_string = "\n".join(commands)
    description = f"Play video files. Keyboard commands: {command_string}"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
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
        "-f",
        "--format",
        required=False,
        default=default_values["format"],
        help="pixel format for raw files",
    )
    parser.add_argument(
        "-s",
        "--resolution",
        required=False,
        default=default_values["resolution"],
        help="resolution for raw files",
    )
    parser.add_argument(
        "-l", "--label", default=default_values["label"], required=False
    )
    parser.add_argument(
        "--seek", required=False, type=int, default=default_values["seek"]
    )

    parser.add_argument("-p", "--paused", required=False, action="store_true")
    parser.add_argument("--rot_90", required=False, action="store_true")
    parser.add_argument(
        "-x",
        "--extract",
        required=False,
        help="Extract a list of frames (comma separated)",
    )
    parser.add_argument(
        "-r",
        "--fps",
        required=False,
        type=int,
        default=default_values["fps"],
        help="Input Framerate",
    )

    parser.add_argument("files", nargs="*", help="file(s)")

    options = parser.parse_args()
    return options


def main(argv):
    # parse options
    options = get_options(argv)
    if options.version:
        print("version: %s" % __version__)
        sys.exit(0)

    raw = False

    if options.format is not None and len(options.format) > 0:
        raw = True
        if options.format == "yuv420p":
            pixFormat = cv2.COLOR_YUV2BGR_I420
        elif options.format == "nv12":
            pixFormat = cv2.COLOR_YUV2BGR_NV12
        elif options.format == "nv21":
            pixFormat = cv2.COLOR_YUV2BGR_NV21
        else:
            print("Unknown color format")
            exit(-1)

    # ensure FFMPEG backend is supported
    assert cv2.videoio_registry.hasBackend(
        cv2.CAP_FFMPEG
    ), "error: player requires cv2 to support FFMPEG videoio"

    asyncio.run(analyze_files(options.files, raw, options))


if __name__ == "__main__":
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
