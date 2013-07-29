"""
Settings for evolve-music
"""

from path import path
import os
import sys
from private import *

#Various paths
PROJECT_PATH = path(__file__).dirname().dirname()

#Where to cache values during the run
CACHE = "percept.fields.caches.MemoryCache"
#Do we use json to serialize the values in in the cache?
SERIALIZE_CACHE_VALUES = False

#How to run the workflows
RUNNER = "percept.workflows.runners.SingleThreadedRunner"

#What to use as a datastore
DATASTORE = "percept.workflows.datastores.FileStore"

#Namespace to give the modules in the registry
NAMESPACE = "evolve-music"

#What severity of error to log to file and console.  One of "DEBUG", "WARN", "INFO", "ERROR"
LOG_LEVEL = "DEBUG"

MUSIC_PATH = "/media/vik/FreeAgent GoFlex Drive/Music/evolve"

MIDI_MUSIC_PATH = "/media/vik/FreeAgent GoFlex Drive/Music/evolve/midi"

MUSIC_TIME_LIMIT = 30 #seconds
ITER_COUNT = 1000

#Used to save and retrieve workflows and other data
DATA_PATH = os.path.abspath(os.path.join(PROJECT_PATH, "stored_data"))
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

MIDI_PATH = "/media/vik/FreeAgent GoFlex Drive/Music/evolve/processed_midi"

FEATURE_PATH = os.path.abspath(os.path.join(DATA_PATH, "features.csv"))
MIDI_FEATURE_PATH = os.path.abspath(os.path.join(DATA_PATH, "midi_features.csv"))
VIZ_PATH = os.path.abspath(os.path.join(DATA_PATH, "visualize.csv"))

SOUNDFONT_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"

MUSIC_STORE_PATH = os.path.abspath(os.path.join(DATA_PATH, "generated"))
MIDI_STORE_PATH = os.path.abspath(os.path.join(DATA_PATH, "generated_midi"))

#Commands are discovered here, and tasks/inputs/formats are imported using only these modules
INSTALLED_APPS = [
    'evolve-music.inputs',
    'evolve-music.formatters',
    'evolve-music.tasks',
    'evolve-music.workflows'
]