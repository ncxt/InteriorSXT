import os
import sys

# Get the repo base directory dynamically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "config"))

import conf

JANELIA_FOLDER = conf.JANELIA_FOLDER
WD_FOLDER = conf.WD_FOLDER

BACT_FOLDER = conf.BACT_FOLDER
BCELL_FOLDER = conf.BCELL_FOLDER

from .cellphantom_bwl import InteriorPhantom
from .cellphantom_psf import InteriorPhantomPSF
