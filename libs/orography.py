"""
ASTER-GDEM Orography Data

TODO: Implement ingestion of ASTER-GDEM elevation data.
      Static field — one frame per extent, tiled to T during ingest.
"""

import numpy as np


class OrographyData:
    def __init__(self, extent, dim=84, **kwargs):
        # TODO: Implement ASTER-GDEM ingestion
        self.data = np.zeros((1, dim, dim), dtype=np.float32)
