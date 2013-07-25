from pandas import DataFrame
import numpy as np
from percept.utils.models import FieldModel

from percept.fields.base import Dict
from percept.conf.base import settings
from percept.utils.models import RegistryCategories, get_namespace
from percept.utils.input import DataFormats
from percept.tests.framework import JSONFormatTester
from percept.datahandlers.formatters import BaseFormat, JSONFormat
from inputs.inputs import MusicFormats
import os
import re
import logging
log = logging.getLogger(__name__)

class MusicFormatter(JSONFormat):
    namespace = get_namespace(__module__)

    def from_mjson(self,input_data):
        """
        Reads subtitle format input data and converts to json.
        """
        return input_data

    def to_dataframe(self):
        return self.data



