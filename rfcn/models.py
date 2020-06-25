# Copyright (c) 2017-present, NEC Laboratories America, Inc. ("NECLA"). 
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import numpy as np
import cv2

from dcn.nms import gpu_nms_wrapper, cpu_softnms_wrapper, soft_nms
from dcn.utils.image import resize, transform
from .config.config import config, update_config

"""
def soft_nms_wrapper(thresh, max_dets=-1):
    def _soft_nms(dets):
        return soft_nms(dets, thresh, max_dets)
    return _soft_nms
"""

import os
import time
from pathlib import Path

def attempt_download(weights, epoch=8, force=False):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip()
    msg = f'Failed to download {weights}'

    r = 1
    if len(weights) > 0 and not os.path.isfile(weights):
        d = { 
            'rfcn_dcn_coco': '0B6T5quL13CdHZ3ZrRVNjcnFmZk0',
            'rfcn_coco': None,
        }

        #file = Path(weights).name
        #cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
        #cache.mkdir(exist_ok=False)
        to = f"/tmp/{weights}-{epoch:04d}.params"
        if not Path(to).is_file() or force:
            if weights in d:
                r = gdrive_download(id=d[weights], to=to)
            if not (r == 0 and os.path.exists(to) and os.path.getsize(to) > 1E6):  # weights exist and > 1MB
                os.remove(to) if os.path.exists(to) else None  # remove partial downloads
                raise Exception(msg)
        else:
            print(f"Already downloaded at {to}")

def gdrive_download(id='0B6T5quL13CdHZ3ZrRVNjcnFmZk0', to='/tmp/rfcn_dcn_coco-0008.params'):
    # https://gist.github.com/tanaikech/f0f2d122e05bf5f971611258c22c110f
    # Downloads a file from Google Drive, accepting presented query
    # from utils.google_utils import *; gdrive_download()
    t = time.time()

    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, to), end='')
    os.remove(to) if os.path.exists(to) else None  # remove existing
    os.remove('cookie') if os.path.exists('cookie') else None

    # Attempt file download
    os.system("curl -c ./cookie -s -L \"https://drive.google.com/uc?export=download&id=%s\" > /dev/null" % id)
    if os.path.exists('cookie'):  # large file
        s = "curl -Lb ./cookie \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=%s\" -o %s" % (id, to)
    else:  # small file
        s = "curl -s -L -o %s 'https://drive.google.com/uc?export=download&id=%s'" % (name, id)
    r = os.system(s)  # execute, capture return values
    os.remove('cookie') if os.path.exists('cookie') else None

    # Error check
    if r != 0:
        os.remove(to) if os.path.exists(to) else None  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if to.endswith('.zip'):
        print('unzipping... ', end='')
        os.system('unzip -q %s' % to)  # unzip
        os.remove(to)  # remove zip to free space

    print('Done (%.1fs)' % (time.time() - t))
    return r

class RFCN(object):
    def __init__(self, dcn=True, softNMS=True, pretrained=True, gpu=0):
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
        os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
        from .core.tester import Predictor
        from .symbols import resnet_v1_101_rfcn
        from .symbols import resnet_v1_101_rfcn_dcn
        from .symbols import resnet_v1_101_rfcn_dcn_rpn
        from dcn.utils.load_model import load_param
        from . import CFGS
        import mxnet as mx

        cfg    = CFGS / f"rfcn_coco_demo{softNMS and '_softNMS' or ''}.yaml"
        update_config(cfg)
        
        config.symbol = 'resnet_v1_101_rfcn_dcn_rpn' if dcn else 'resnet_v1_101_rfcn'
        params = None
        if pretrained:
            params = dcn and 'rfcn_dcn_coco' or 'rfcn_coco'
            attempt_download(params, dcn and 8 or 0)
        instance = eval(config.symbol + '.' + config.symbol)()
        sym = instance.get_symbol(config, is_train=False)

        # Build the model predictor
        data_names  = ['data', 'im_info']
        label_names = []
        data        = self.preprocess()
        data        = [[mx.nd.array(data[i][name]) for name in data_names] for i in range(len(data))]                   # [[data, im_info], ...]
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]   # [[('data', (1, 3, 600, 1000))]]
        provide_data  = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in range(len(data))]                 # [[('data', (1, 3, h, w)), ('im_info', (h, w, scale))], ...]]
        provide_label = [None for i in range(len(data))]                                                                # [None, ...]
        arg_params, aux_params = None, None
        if pretrained:
            chkpt = softNMS and 8 or 0
            arg_params, aux_params = load_param(f"/tmp/{params}", chkpt, process=True)
        
        gpus = [gpu] if type(gpu) == int else gpu
        self.predictor = Predictor(sym, data_names, label_names,
                              context=[mx.gpu(gpu) for gpu in gpus], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
        self.data_names = data_names
        self.config = config
        self.nms = gpu_nms_wrapper(config.TEST.NMS, 0)  # 0.2 for SNMS and 0.3 otherwise
        if softNMS:
            self.snms = cpu_softnms_wrapper()

    def preprocess(self, frames=None, interpolation=cv2.INTER_LINEAR):
        if frames is None:
            frames = []
            for im_name in ['COCO_test2015_000000000891.jpg', 'COCO_test2015_000000001669.jpg']:
                #path = DCN_DEMO / im_name
                #assert path.exists(), ('%s does not exist'.format('../demo/' + im_name))
                #im = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                im = np.zeros((480, 640, 3))
                frames.append(im)
                break

        # resize at some scale
        # transform from BGR HxWxC to RGB CxHxW with normalization
        data = []
        for im in frames:
            target_size = config.SCALES[0][0]
            max_size = config.SCALES[0][1]
            im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE, interpolation=interpolation)
            im_tensor = transform(im, config.network.PIXEL_MEANS)
            im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
            data.append({'data': im_tensor, 'im_info': im_info})

        return data

    def rpn(self, frames):
        r"""Return RoIs from RPN only.        
        """

        import mxnet as mx
        data = [[mx.nd.array(frames[i][name]) for name in self.data_names] for i in range(len(frames))]                 # [[data, im_info], ...]
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]   # [[('data', (1, 3, 600, 1000))]]
        provide_data = [[(k, v.shape) for k, v in zip(self.data_names, data[i])] for i in range(len(data))]             # [[('data', (1, 3, h, w)), ('im_info', (h, w, scale))], ...]]
        provide_label = [None for i in range(len(data))]                                                                # [None, ...]
        data_batch = mx.io.DataBatch(data=data, label=[], pad=0, index=None, 
                                     provide_data=provide_data, 
                                     provide_label=provide_label)
        output_all      = self.predictor.predict(data_batch)
        data_dict_all   = [dict(zip(self.data_names, idata)) for idata in data_batch.data]
        scales          = [data_batch.data[i][1].asnumpy()[0, 2] for i in range(len(data_batch.data))]
        rois_output_all = []
        rois_scores_all = []
        for output, data_dict, scale in zip(output_all, data_dict_all, scales):
            if config.TEST.HAS_RPN: # from updated yaml
                # ROIs from RPN
                rois = output['rois_output'].asnumpy()[:, 1:]
            else:
                # ROIs from RFCN? 
                rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
            rois_output_all.append(rois / scale)
            rois_scores_all.append(output['rois_score'].asnumpy())
        return rois_output_all, rois_scores_all

    def detect(self, frames):
        from bbox.bbox_transform import bbox_pred, clip_boxes
        import mxnet as mx
        data = [[mx.nd.array(frames[i][name]) for name in self.data_names] for i in range(len(frames))]                 # [[data, im_info], ...]
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]   # [[('data', (1, 3, 600, 1000))]]
        provide_data = [[(k, v.shape) for k, v in zip(self.data_names, data[i])] for i in range(len(data))]             # [[('data', (1, 3, h, w)), ('im_info', (h, w, scale))], ...]]
        provide_label = [None for i in range(len(data))]                                                                # [None, ...]
        data_batch = mx.io.DataBatch(data=data, label=[], pad=0, index=None, 
                                     provide_data=provide_data, 
                                     provide_label=provide_label)

        output_all      = self.predictor.predict(data_batch)
        data_dict_all   = [dict(zip(self.data_names, idata)) for idata in data_batch.data]
        scales          = [data_batch.data[i][1].asnumpy()[0, 2] for i in range(len(data_batch.data))]
        scores_all      = []
        pred_boxes_all  = []
        rois_output_all = []
        rois_scores_all = []
        for output, data_dict, scale in zip(output_all, data_dict_all, scales):
            if config.TEST.HAS_RPN: # from updated yaml
                # ROIs from RPN
                rois = output['rois_output'].asnumpy()[:, 1:]
            else:
                # ROIs from RFCN? 
                rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
            im_shape = data_dict['data'].shape

            # save output
            scores = output['cls_prob_reshape_output'].asnumpy()[0]
            bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

            # post processing
            pred_boxes = bbox_pred(rois, bbox_deltas)
            pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])
            pred_boxes = pred_boxes / scale            
            scores_all.append(scores)
            pred_boxes_all.append(pred_boxes)

            # batch of one frame
            rois_output_all.append(rois / scale)
            rois_scores_all.append(output['rois_score'].asnumpy())

        return pred_boxes_all, scores_all, rois_output_all, rois_scores_all

    def postprocess(self, pred_boxes_all, scores_all):
        r"""CLS NMS for the final predictions.
        """
        # TODO potentially slow => parallelization
        # RPN_POST_NMS_TOP_N: 300
        dets = []
        for scores, boxes in zip(scores_all, pred_boxes_all):
            boxes = boxes.astype('f')
            scores = scores.astype('f')
            dets_nms = []
            for c in range(1, scores.shape[1]): # Starting with person FG objects
                cls_scores  = scores[:, c, np.newaxis]
                cls_boxes   = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, c*4: (c+1)*4]
                cls_dets    = np.hstack((cls_boxes, cls_scores))
                keep        = self.nms(cls_dets)
                cls_dets    = cls_dets[keep, :]
                cls_dets    = cls_dets[cls_dets[:, -1] > 0.7, :]
                dets_nms.append(cls_dets)
        
            # per class bboxes (#bboxes, 4+1)
            dets.append(dets_nms)
        return dets
