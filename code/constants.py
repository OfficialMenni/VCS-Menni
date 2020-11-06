import os
from ctypes import c_long, sizeof

ROOT_DIR = path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEDIA_DIR = ROOT_DIR + "\\media\\"
DATABASE_DIR = ROOT_DIR + "\\paintings_db\\"
CFG_DIR = ROOT_DIR + "\\cfg\\"
bit_size = sizeof(c_long) * 8
limit = 2 ** (bit_size - 1)
