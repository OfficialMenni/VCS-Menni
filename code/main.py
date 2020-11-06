import argparse
import skvideo.io
from service_locator import serviceLocator
from util import *


def arg_parse():
    """
    Parse arguments to the main module
    """
    parser = argparse.ArgumentParser(description='Project Module')
    parser.add_argument("--video", dest='video', help="Video in media folder to process.", default="VIRB0392.MP4",
                        type=str)
    parser.add_argument("--topn", dest='topn',
                        help="Number of results displayed in descending likelihood order for paint retrieval. (INT)",
                        default=5)
    parser.add_argument("--start_frame", dest="first_frame", help="First frame to process. (INT)", default=0)
    parser.add_argument("--skip", dest="skip", help="Process 1 frame every N (INT)", default=1)
    parser.add_argument('--outH', dest="outH", help="Height of the output image detection. Just for better"
                                                    "output visualization.", type=int, default=0)
    parser.add_argument('--outW', dest="outW", help="Width of the output image detection. Just for better"
                                                    "output visualization.", type=int, default=0)
    parser.add_argument("--mk_video", dest="mk_video", help="Create a video highlighting people and paintings. "
                                                            "Enter the name of the output video")
    return parser.parse_args()


# Arguments parsing and module initialization
args = arg_parse()
video = args.video
mk_video = args.mk_video
topn = int(args.topn)
first_frame = int(args.first_frame)
skip = int(args.skip)
outW = int(args.outW)
outH = int(args.outH)
outsize = (outW, outH)

if outW == 0 or outH == 0:
    outsize = None
assert first_frame >= 0
assert skip >= 1

locator = serviceLocator()

locator.check_db()
print("Reading video. This may take a while...")
cap = None
try:
    cap = skvideo.io.vread(MEDIA_DIR + video)
except FileNotFoundError:
    print("No video found. Try again with a video contained in the media folder")
    exit(1)

if mk_video:
    locator.create_video(cap, mk_video)
else:
    locator.analyse_video(first_frame=first_frame, cap=cap, skip=skip, topn=topn, outsize=outsize)

