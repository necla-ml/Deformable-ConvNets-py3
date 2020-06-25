import os
import sys
from pathlib import Path

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

#_root = Path(__file__).parents[3]
#DCN = _root / 'Deformable-ConvNets-py3'

#DCN_EXP  = DCN / 'experiments'
#DCN_PARAMS  = DCN / 'model'
#DCN_DEMO = DCN / 'demo'

#DCN_LIB  = DCN / 'lib'
#DCN_RFCN = DCN / 'rfcn'
#DCN_EXP_RFCN_CFGS = DCN_EXP / 'rfcn' / 'cfgs'
#sys.add_path(str(DCN_LIB))
#sys.add_path(str(DCN_RFCN))

from .config.config import config, update_config
from .models import RFCN
CFGS = Path(__file__).parent / 'cfgs'

__all__ = [
    'RFCN',
    'CFGS'
]