# -*- coding: utf-8 -*

from copy import deepcopy

import numpy as np
import math

import torch

from videoanalyst.pipeline.pipeline_base import TRACK_PIPELINES, PipelineBase
from videoanalyst.pipeline.utils import (cxywh2xywh,
                                         imarray_to_tensor, tensor_to_numpy,
                                         xywh2cxywh, xyxy2cxywh, cxywh2xyxy)
from videoanalyst.pipeline.utils.crop import get_crop_single
import videoanalyst.utils.visualize_score_map as vsm


# -----------------------------self defined func-----------------------------------------------
import sys
import numpy as np
import cv2
sys.path.append("/root/LY/pytracking-master/pytracking")
from DPE_Module.depth.options import LiteMonoOptions
from torchvision import transforms, datasets
from DPE_Module.depth import networks
from scipy.stats import norm
import matplotlib as mpl # not necessary
import matplotlib.cm as cm # not necessary
def depth_check(label, data, threshold=1.5):
    median = np.median(label)
    mad = np.median(np.abs(label - median))
    modified_z_scores = np.abs(0.6745 * (data - median) / mad)
    outliers = modified_z_scores < threshold
    return outliers


def PSR(response):
    response_map=response.copy()
    max_loc=np.unravel_index(np.argmax(response_map, axis=None),response_map.shape)
    y,x=max_loc
    F_max = np.max(response_map)
    response_map[y-5:y+6,x-5:x+6]=0.
    mean=np.mean(response_map[response_map>0])
    std=np.std(response_map[response_map>0])
    psr=(F_max-mean)/std
    return psr

def Ck1(psrlist):
    mu = np.mean(psrlist)
    sigma = np.std(psrlist)
    gaussian_probabilities = norm.pdf(psrlist, mu, sigma)
    tmp = gaussian_probabilities[-1]/gaussian_probabilities.sum()
    return tmp
    
# -----------------------------self defined func-----------------------------------------------


# ============================== Tracker definition ============================== #
@TRACK_PIPELINES.register
class STMTrackTracker(PipelineBase):
    r"""
    default_hyper_params setting rules:
    0/0.0: to be set in config file manually.
    -1: to be calculated in code automatically.
    >0: default value.
    """

    default_hyper_params = dict(
        total_stride=8,
        score_size=0,
        score_offset=-1,
        test_lr=0.0,
        penalty_k=0.0,
        window_influence=0.0,
        windowing="cosine",
        m_size=0,
        q_size=0,
        min_w=10,
        min_h=10,
        phase_memorize="memorize",
        phase_track="track",
        corr_fea_output=False,
        num_segments=4,
        confidence_threshold=0.6,
        gpu_memory_threshold=-1,
        search_area_factor=0.0,
        visualization=False,
    )

    def __init__(self, *args, **kwargs):
        super(STMTrackTracker, self).__init__(*args, **kwargs)
        self.update_params()

        # set underlying model to device
        self.device = torch.device("cpu")
        self.debug = False
        self.set_model(self._model)

    def set_model(self, model):
        """model to be set to pipeline. change device & turn it into eval mode
        
        Parameters
        ----------
        model : ModuleBase
            model to be set to pipeline
        """
        self._model = model
        # self._model = model.to(self.device)
        self._model.eval()

    def set_device(self, device):
        self.device = device
        self._model = self._model.to(device)
        if self.device != torch.device('cuda:0'):
            self._hyper_params['gpu_memory_threshold'] = 3000

    def update_params(self):
        hps = self._hyper_params
        assert hps['q_size'] == hps['m_size']
        hps['score_offset'] = (
            hps['q_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        if hps['gpu_memory_threshold'] == -1:
            hps['gpu_memory_threshold'] = 1 << 30  # infinity
        self._hyper_params = hps

        self._hp_score_size = self._hyper_params['score_size']
        self._hp_m_size = self._hyper_params['m_size']
        self._hp_q_size = self._hyper_params['q_size']
        self._hp_num_segments = self._hyper_params['num_segments']
        self._hp_gpu_memory_threshold = self._hyper_params['gpu_memory_threshold']
        self._hp_confidence_threshold = self._hyper_params['confidence_threshold']
        self._hp_visualization = self._hyper_params['visualization']

    def create_fg_bg_label_map(self, bbox, size):
        r"""

        Args:
            bbox: target box. (cx, cy, w, h) format.
            size: int
        Returns:

        """
        bbox = cxywh2xyxy(bbox).astype(np.int)
        fg_bg_label_map = torch.zeros(size=(1, 1, size, size), dtype=torch.float32, device=self.device)
        fg_bg_label_map[:, :, bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1] = 1
        return fg_bg_label_map

    def memorize(self, im: np.array, target_pos, target_sz, avg_chans):
        m_size = self._hp_m_size

        scale_m = math.sqrt(np.prod(target_sz) / np.prod(self._state['base_target_sz']))
        im_m_crop, real_scale = get_crop_single(im, target_pos, scale_m, m_size, avg_chans)

        phase = self._hyper_params['phase_memorize']
        with torch.no_grad():
            data = imarray_to_tensor(im_m_crop).to(self.device)
            bbox_m = np.concatenate([np.array([(m_size - 1) / 2, (m_size - 1) / 2]),
                                    target_sz * real_scale], axis=0)
            fg_bg_label_map = self.create_fg_bg_label_map(bbox_m, m_size)
            fm = self._model(data, fg_bg_label_map, phase=phase)
        return fm

    def select_representatives(self, cur_frame_idx):
        num_segments = self._hp_num_segments
        assert cur_frame_idx > num_segments

        dur = cur_frame_idx // num_segments
        indexes = np.concatenate([
            np.array([1]),
            np.array(list(range(num_segments))) * dur + dur // 2 + 1
        ])
        if self._state['pscores'][-1] > self._hp_confidence_threshold:
            indexes = np.append(indexes, 0)
        indexes = np.unique(indexes)

        representatives = []
        for idx in indexes:
            fm = self._state['all_memory_frame_feats'][idx - 1]
            if not fm.is_cuda:
                fm = fm.to(self.device)
            representatives.append(fm)

        assert len(representatives[0].shape) == 5
        representatives = torch.cat(representatives, dim=2)
        return representatives

    def init(self, im, state):
        r"""Initialize tracker
            Internal target state representation: self._state['state'] = (target_pos, target_sz)
        
        Arguments
        ---------
        im : np.array
            initial frame image
        state
            target state on initial frame (bbox in case of SOT), format: xywh
        """
        #print(type(im),im.shape)
        torch.cuda.empty_cache()
        # -----------------------------get DPE Module-----------------------------------------------
        self.output_state = 0
        self.frame_num =1
        encoder_path = r"/root/LY/pytracking-master/pytracking/DPE_Module/depth/networks/encoder.pth"
        decoder_path = r"/root/LY/pytracking-master/pytracking/DPE_Module/depth/networks/depth.pth"
        # need to be changed according to your path
        self.encoder_dict = torch.load(encoder_path)
        decoder_dict = torch.load(decoder_path)
        self.encoder = networks.LiteMono(model="lite-mono",
                                    height=self.encoder_dict['height'],
                                    width=self.encoder_dict['width'])
        self.depth_decoder = networks.DepthDecoder(self.encoder.num_ch_enc, scales=range(3))
        model_dict = self.encoder.state_dict()
        depth_model_dict = self.depth_decoder.state_dict()
        self.encoder.load_state_dict({k: v for k, v in self.encoder_dict.items() if k in model_dict})
        self.depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

        self.encoder.cuda() # need to be changed if you inference on multiple gpu
        self.encoder.eval()
        self.depth_decoder.cuda() # need to be changed if you inference on multiple gpu
        self.depth_decoder.eval()
        self.k1 = 0.10
        self.psr = []

        self.refined_depth = 1
        self.disp_resized_np = 0
        rect = state
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]
        self.init_state = [target_pos[0],target_pos[1],target_sz[0],target_sz[1]]
        # -----------------------------get DPE Module-----------------------------------------------
        
        
        rect = state  # bbox in xywh format is given for initialization in case of tracking
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]

        self._state['im_h'] = im.shape[0]
        self._state['im_w'] = im.shape[1]

        score_size = self._hp_score_size
        if self._hyper_params['windowing'] == 'cosine':
            window = np.outer(np.hanning(score_size), np.hanning(score_size))
            window = window.reshape(-1)
        elif self._hyper_params['windowing'] == 'uniform':
            window = np.ones((score_size, score_size))
        else:
            window = np.ones((score_size, score_size))

        self._state['avg_chans'] = (np.mean(im[..., 0]), np.mean(im[..., 1]), np.mean(im[..., 2]))
        self._state['window'] = window
        self._state['state'] = (target_pos, target_sz)
        self._state['last_img'] = im
        self._state['track_rects'] = [{'target_pos': target_pos, 'target_sz': target_sz}]
        self._state['all_memory_frame_feats'] = []
        self._state['pscores'] = [1.0, ]
        self._state['cur_frame_idx'] = 1
        self._state["rng"] = np.random.RandomState(123456)
        search_area = np.prod(target_sz * self._hyper_params['search_area_factor'])
        self._state['target_scale'] = math.sqrt(search_area) / self._hp_q_size
        self._state['base_target_sz'] = target_sz / self._state['target_scale']
        if self._hp_visualization:
            vsm.rename_dir()

    def get_avg_chans(self):
        return self._state['avg_chans']

    def track(self,
              im_q,
              target_pos,
              target_sz,
              features,
              update_state=False,
              **kwargs):
        if 'avg_chans' in kwargs:
            avg_chans = kwargs['avg_chans']
        else:
            avg_chans = self._state['avg_chans']

        q_size = self._hp_q_size
        #print(type(im_q),im_q.shape)
        # ---------------------------------------------------------------------
        image = im_q
        self.frame_num += 1
        # PREDICTION    
        
        if (self.frame_num==2) or ((self.frame_num-1)%60 ==0):
            input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_img = cv2.resize(input_img,(self.encoder_dict['width'],self.encoder_dict['height']))
            input_image = transforms.ToTensor()(input_img).unsqueeze(0)
            input_image = input_image.to("cuda")
            features_ = self.encoder(input_image)
            outputs = self.depth_decoder(features_)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (image.shape[0], image.shape[1]), mode="bilinear", align_corners=False)
            disp_resized_np_ = disp_resized.squeeze().cpu().detach().numpy()
            self.disp_resized_np = disp_resized_np_

        if not isinstance(self.output_state,int):
            depth_label = self.disp_resized_np[int(self.output_state[1]):int(self.output_state[1]+self.output_state[3]),int(self.output_state[0]):int(self.output_state[0]+self.output_state[2])]
            self.psr.append(PSR(self.disp_resized_np))
            if (self.frame_num==2) or ((self.frame_num-1)%5==0):
                tmp = Ck1(self.psr)
                if tmp<0.5:
                    self.k1 = tmp
        else:
            depth_label = self.disp_resized_np[int(self.init_state[1]):int(self.init_state[1]+self.init_state[3]),int(self.init_state[0]):int(self.init_state[0]+self.init_state[2])]
            self.psr.append(PSR(self.disp_resized_np))
            if (self.frame_num==2) or ((self.frame_num-1)%5 ==0):
                tmp = Ck1(self.psr)
                if tmp<0.5:
                    self.k1 = tmp
                    
        self.refined_depth = depth_check(depth_label,self.disp_resized_np)
        

        
        image = (self.k1*image + (1-self.k1)*image*self.refined_depth[:,:,None]).astype(np.uint8)
        im_q = image
        
        # ---------------------------------------------------------------------
        
        
        phase_track = self._hyper_params['phase_track']
        im_q_crop, scale_q = get_crop_single(im_q, target_pos, self._state['target_scale'], q_size, avg_chans)
        self._state["scale_q"] = deepcopy(scale_q)
        with torch.no_grad():
            #print(type(imarray_to_tensor(im_q_crop).to(self.device)),type(features),type(phase_track))
            score, box, cls, ctr, extra = self._model(
                imarray_to_tensor(im_q_crop).to(self.device),
                features,
                phase=phase_track)
        if self._hyper_params["corr_fea_output"]:
            self._state["corr_fea"] = extra["corr_fea"]

        if self._hp_visualization:
            score1 = tensor_to_numpy(score[0])[:, 0]
            vsm.visualize(score1, self._hp_score_size, im_q_crop, self._state['cur_frame_idx'], 'raw_score')

        box = tensor_to_numpy(box[0])
        score = tensor_to_numpy(score[0])[:, 0]
        cls = tensor_to_numpy(cls[0])
        ctr = tensor_to_numpy(ctr[0])
        box_wh = xyxy2cxywh(box)

        # score post-processing
        best_pscore_id, pscore, penalty = self._postprocess_score(
            score, box_wh, target_sz, scale_q, im_q_crop)
        # box post-processing
        new_target_pos, new_target_sz = self._postprocess_box(
            best_pscore_id, score, box_wh, target_pos, target_sz, scale_q,
            q_size, penalty)

        if self.debug:
            box = self._cvt_box_crop2frame(box_wh, target_pos, q_size, scale_q)

        # restrict new_target_pos & new_target_sz
        new_target_pos, new_target_sz = self._restrict_box(
            new_target_pos, new_target_sz)
        # ---------------------attn---------------------------
        self.output_state = [new_target_pos[0],new_target_pos[1],new_target_sz[0],new_target_sz[1]]
        # ---------------------attn---------------------------
        # record basic mid-level info
        self._state['q_crop'] = im_q_crop
        bbox_pred_in_crop = np.rint(box[best_pscore_id]).astype(np.int)
        self._state['bbox_pred_in_crop'] = bbox_pred_in_crop
        # record optional mid-level info
        if update_state:
            self._state['score'] = score
            self._state['pscore'] = pscore[best_pscore_id]
            self._state['all_box'] = box
            self._state['cls'] = cls
            self._state['ctr'] = ctr

        return new_target_pos, new_target_sz

    def set_state(self, state):
        self._state["state"] = state

    def get_track_score(self):
        return float(self._state["pscore"])

    def update(self, im, state=None):
        """ Perform tracking on current frame
            Accept provided target state prior on current frame
            e.g. search the target in another video sequence simutanously

        Arguments
        ---------
        im : np.array
            current frame image
        state
            provided target state prior (bbox in case of SOT), format: xywh
        """
        # use prediction on the last frame as target state prior
        if state is None:
            target_pos_prior, target_sz_prior = self._state['state']
        # use provided bbox as target state prior
        else:
            rect = state  # bbox in xywh format is given for initialization in case of tracking
            box = xywh2cxywh(rect).reshape(4)
            target_pos_prior, target_sz_prior = box[:2], box[2:]

        fidx = self._state['cur_frame_idx']
        prev_frame_feat = self.memorize(self._state['last_img'],
                                 self._state['track_rects'][fidx - 1]['target_pos'],
                                 self._state['track_rects'][fidx - 1]['target_sz'],
                                 self._state['avg_chans'])

        if fidx > self._hp_gpu_memory_threshold:
            prev_frame_feat = prev_frame_feat.detach().cpu()
        self._state['all_memory_frame_feats'].append(prev_frame_feat)

        if fidx <= self._hp_num_segments:
            features = torch.cat(self._state['all_memory_frame_feats'], dim=2)
        else:
            features = self.select_representatives(fidx)

        # forward inference to estimate new state
        target_pos, target_sz = self.track(im,
                                           target_pos_prior,
                                           target_sz_prior,
                                           features,
                                           update_state=True)

        # save underlying state
        # self.state['target_pos'], self.state['target_sz'] = target_pos, target_sz
        self._state['state'] = target_pos, target_sz

        # return rect format
        track_rect = cxywh2xywh(np.concatenate([target_pos, target_sz],
                                               axis=-1))
        self._state['last_img'] = im
        self._state['track_rects'].append({'target_pos': target_pos, 'target_sz': target_sz})
        self._state['target_scale'] = math.sqrt(np.prod(target_sz) / np.prod(self._state['base_target_sz']))
        self._state['pscores'].append(self._state['pscore'])
        self._state['cur_frame_idx'] += 1
        if self._hyper_params["corr_fea_output"]:
            return target_pos, target_sz, self._state["corr_fea"]
        return track_rect

    # ======== tracking processes ======== #

    def _postprocess_score(self, score, box_wh, target_sz, scale_x, im_x_crop):
        r"""
        Perform SiameseRPN-based tracker's post-processing of score
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_sz: previous state (w & h)
        :param scale_x:
        :return:
            best_pscore_id: index of chosen candidate along axis HW
            pscore: (HW, ), penalized score
            penalty: (HW, ), penalty due to scale/ratio change
        """
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        penalty_k = self._hyper_params['penalty_k']
        target_sz_in_crop = target_sz * scale_x
        s_c = change(
            sz(box_wh[:, 2], box_wh[:, 3]) /
            (sz_wh(target_sz_in_crop)))  # scale penalty
        r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) /
                     (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score
        if self._hp_visualization:
            vsm.visualize(pscore, self._hp_score_size, im_x_crop, self._state['cur_frame_idx'], 'pscore_0')

        # ipdb.set_trace()
        # cos window (motion model)
        window_influence = self._hyper_params['window_influence']
        pscore = pscore * (
            1 - window_influence) + self._state['window'] * window_influence
        best_pscore_id = np.argmax(pscore)
        if self._hp_visualization:
            vsm.visualize(pscore, self._hp_score_size, im_x_crop, self._state['cur_frame_idx'], 'pscore_1')

        return best_pscore_id, pscore, penalty

    def _postprocess_box(self, best_pscore_id, score, box_wh, target_pos,
                         target_sz, scale_x, x_size, penalty):
        r"""
        Perform SiameseRPN-based tracker's post-processing of box
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_pos: (2, ) previous position (x & y)
        :param target_sz: (2, ) previous state (w & h)
        :param scale_x: scale of cropped patch of current frame
        :param x_size: size of cropped patch
        :param penalty: scale/ratio change penalty calculated during score post-processing
        :return:
            new_target_pos: (2, ), new target position
            new_target_sz: (2, ), new target size
        """
        pred_in_crop = box_wh[best_pscore_id, :] / np.float32(scale_x)
        # about np.float32(scale_x)
        # attention!, this casting is done implicitly
        # which can influence final EAO heavily given a model & a set of hyper-parameters

        # box post-postprocessing
        test_lr = self._hyper_params['test_lr']
        lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
        res_x = pred_in_crop[0] + target_pos[0] - (x_size // 2) / scale_x
        res_y = pred_in_crop[1] + target_pos[1] - (x_size // 2) / scale_x
        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        new_target_pos = np.array([res_x, res_y])
        new_target_sz = np.array([res_w, res_h])

        return new_target_pos, new_target_sz

    def _restrict_box(self, target_pos, target_sz):
        r"""
        Restrict target position & size
        :param target_pos: (2, ), target position
        :param target_sz: (2, ), target size
        :return:
            target_pos, target_sz
        """
        target_pos[0] = max(0, min(self._state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(self._state['im_h'], target_pos[1]))
        target_sz[0] = max(self._hyper_params['min_w'],
                           min(self._state['im_w'], target_sz[0]))
        target_sz[1] = max(self._hyper_params['min_h'],
                           min(self._state['im_h'], target_sz[1]))

        return target_pos, target_sz

    def _cvt_box_crop2frame(self, box_in_crop, target_pos, scale_x, x_size):
        r"""
        Convert box from cropped patch to original frame
        :param box_in_crop: (4, ), cxywh, box in cropped patch
        :param target_pos: target position
        :param scale_x: scale of cropped patch
        :param x_size: size of cropped patch
        :return:
            box_in_frame: (4, ), cxywh, box in original frame
        """
        x = (box_in_crop[..., 0]) / scale_x + target_pos[0] - (x_size //
                                                               2) / scale_x
        y = (box_in_crop[..., 1]) / scale_x + target_pos[1] - (x_size //
                                                               2) / scale_x
        w = box_in_crop[..., 2] / scale_x
        h = box_in_crop[..., 3] / scale_x
        box_in_frame = np.stack([x, y, w, h], axis=-1)

        return box_in_frame
