import os
import sys
from pathlib import Path

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

from .config.config import config, update_config
CFGS = Path(__file__).parent / 'cfgs'

__all__ = [
    'CFGS'
]