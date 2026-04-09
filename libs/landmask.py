"""
Land/Sea Mask Data

TODO: Implement land/sea mask ingestion.
      Static field — one frame per extent, tiled to T during ingest.
"""

import numpy as np


class LandMaskData:
    def __init__(self, extent, dim=84, **kwargs):
        # TODO: Implement land/sea mask ingestion
        self.data = np.zeros((1, dim, dim), dtype=np.float32)
