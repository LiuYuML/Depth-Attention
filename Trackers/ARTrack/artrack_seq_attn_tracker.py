import math

from lib.models.artrack_seq import build_artrack_seq
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


# -----------------------------self defined func-----------------------------------------------
import sys
import numpy as np
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


class ARTrackSeq(BaseTracker):
    def __init__(self, params, dataset_name):
        super(ARTrackSeq, self).__init__(params)
        network = build_artrack_seq(params.cfg, training=False)
        #print("hi,leo")
        #print("hi.leo")
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.bins = self.cfg.MODEL.BINS
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.store_result = None
        self.save_all = 7
        self.x_feat = None
        self.update = None
        self.update_threshold = 5.0
        self.update_intervals = 1
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
        # -----------------------------get DPE Module-----------------------------------------------
        

    def initialize(self, image, info: dict):
        # forward the template once
        self.x_feat = None
        # ------------------------------------------------attn-----------------------------
        self.init_state = info['init_bbox']
        # ------------------------------------------------attn-----------------------------
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)  # output_sz=self.params.template_size
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        
        # save states
        self.state = info['init_bbox']
        self.store_result = [info['init_bbox'].copy()]
        for i in range(self.save_all - 1):
            self.store_result.append(info['init_bbox'].copy())
        self.frame_id = 0
        self.update = None
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        
        # ---------------------------------------------------------------------
        
        self.frame_num += 1
        # PREDICTION    
        
        if (self.frame_num==2) or ((self.frame_num-1)%60 ==0):
            input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_img = cv2.resize(input_img,(self.encoder_dict['width'],self.encoder_dict['height']))
            input_image = transforms.ToTensor()(input_img).unsqueeze(0)
            input_image = input_image.to("cuda")
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (image.shape[0], image.shape[1]), mode="bilinear", align_corners=False)
            disp_resized_np_ = disp_resized.squeeze().cpu().detach().numpy()
            self.disp_resized_np = disp_resized_np_
        if not isinstance(self.output_state,int):
            depth_label = self.disp_resized_np[int(self.output_state[1]):int(self.output_state[1]+self.output_state[3]),int(self.output_state[0]):int(self.output_state[0]+self.output_state[2])]
            self.psr.append(PSR(self.disp_resized_np))
            if (self.frame_num==2) or ((self.frame_num-1)%5 ==0):
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
        #---------------------visualize the depth estimatio colored map----------
        #vmax = np.percentile(self.disp_resized_np, 95)
        #normalizer = mpl.colors.Normalize(vmin=self.disp_resized_np.min(), vmax=vmax)
        #mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        #colormapped_im = (mapper.to_rgba(self.disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        #---------------------visualize the depth estimatio colored map----------
        image = (self.k1*image + (1-self.k1)*image*self.refined_depth[:,:,None]).astype(np.uint8)
        #self.image = image
        
        # ---------------------------------------------------------------------
        
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        for i in range(len(self.store_result)):
            box_temp = self.store_result[i].copy()
            box_out_i = transform_image_to_crop(torch.Tensor(self.store_result[i]), torch.Tensor(self.state),
                                                resize_factor,
                                                torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                                normalize=True)
            box_out_i[2] = box_out_i[2] + box_out_i[0]
            box_out_i[3] = box_out_i[3] + box_out_i[1]
            box_out_i = box_out_i.clamp(min=-0.5, max=1.5)
            box_out_i = (box_out_i + 0.5) * (self.bins - 1)
            if i == 0:
                seqs_out = box_out_i
            else:
                seqs_out = torch.cat((seqs_out, box_out_i), dim=-1)
        seqs_out = seqs_out.unsqueeze(0)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors,
                seq_input=seqs_out, stage="sequence", search_feature=self.x_feat, update=None)

        self.x_feat = out_dict['x_feat']

        pred_boxes = out_dict['seqs'][:, 0:4] / (self.bins - 1) - 0.5
        pred_boxes = pred_boxes.view(-1, 4).mean(dim=0)
        pred_new = pred_boxes
        pred_new[2] = pred_boxes[2] - pred_boxes[0]
        pred_new[3] = pred_boxes[3] - pred_boxes[1]
        pred_new[0] = pred_boxes[0] + pred_new[2] / 2
        pred_new[1] = pred_boxes[1] + pred_new[3] / 2
        pred_boxes = (pred_new * self.params.search_size / resize_factor).tolist()
        # ---------------------attn---------------------------
        #print(pred_boxes)
        self.output_state = pred_boxes
        # ---------------------attn---------------------------
        # Baseline: Take the mean of all pred boxes as the final result
        # pred_box = (pred_boxes.mean(
        #    dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_boxes, resize_factor), H, W, margin=10)
        if len(self.store_result) < self.save_all:
            self.store_result.append(self.state.copy())
        else:
            for i in range(self.save_all):
                if i != self.save_all - 1:
                    self.store_result[i] = self.store_result[i + 1]
                else:
                    self.store_result[i] = self.state.copy()

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap',
                                     1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        # cx_real = cx + cx_prev
        # cy_real = cy + cy_prev
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return ARTrackSeq
