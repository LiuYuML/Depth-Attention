import math

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import copy
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

import matplotlib as mpl # not necessary
import matplotlib.cm as cm # not necessary
def depth_check(label, data, threshold=1.5):
    median = np.median(label)
    mad = np.median(np.abs(label - median))
    modified_z_scores = np.abs(0.6745 * (data - median) / mad)
    outliers = modified_z_scores < threshold
    return outliers

from scipy.stats import norm
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


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        #print("leo",self.params.checkpoint)
        print("leoyu")
        self.params.checkpoint = "/root/autodl-tmp/OSTrack_ep0300.pth.tar"
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = False
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
        # ------------------------------------------------attn-----------------------------
        self.init_state = info['init_bbox']
        # ------------------------------------------------attn-----------------------------
        
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
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

        # ---------------------------------------------------------------------
        
        
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_boxes_neighbor = self.network.box_head.cal_bbox_neighbor(pred_score_map.clone(), out_dict['size_map'], out_dict['offset_map'])
        pred_boxes_neighbor = pred_boxes_neighbor.view(-1, 4)
        
        #print(pred_boxes)
        #print(pred_boxes_neighbor)
        
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        if self.network.head_type == 'CENTER_NEIGHBOR':
            self.old_state = copy.deepcopy(self.state)
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # ---------------------attn---------------------------
        #print(pred_boxes)
        self.output_state = self.state
        # ---------------------attn---------------------------
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                #image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break
                        
        if self.network.head_type == 'CENTER_NEIGHBOR':
            all_boxes_neighbor = self.map_box_back_batch(pred_boxes_neighbor.clone() * self.params.search_size / resize_factor, resize_factor)
            all_boxes_neighbor_save = all_boxes_neighbor.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_neighbor_save}


        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}
        
    def track_neighbor(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_boxes_neighbor = self.network.box_head.cal_bbox_neighbor(pred_score_map.clone(), out_dict['size_map'], out_dict['offset_map'])
        pred_boxes_neighbor = pred_boxes_neighbor.view(-1, 4)
        
        #print(pred_boxes)
        #print(pred_boxes_neighbor)
        
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        if self.network.head_type == 'CENTER_NEIGHBOR':
            self.old_state = copy.deepcopy(self.state)
        #self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        xywh = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        score = 1
                        
        all_boxes_neighbor = self.map_box_back_batch(pred_boxes_neighbor.clone() * self.params.search_size / resize_factor, resize_factor)
        all_boxes_neighbor_save = all_boxes_neighbor.view(-1,4).tolist()  # (4N, )

        nscore = []
        for ind_score in range(len(all_boxes_neighbor_save)):
            nscore.append(0.8)

        return xywh,score,all_boxes_neighbor_save,nscore
    def update_center(self,bbox):
        self.state = bbox


    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        if self.network.head_type == 'CENTER_NEIGHBOR':
            cx_prev, cy_prev = self.old_state[0] + 0.5 * self.old_state[2], self.old_state[1] + 0.5 * self.old_state[3]
        else:
            cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
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
    return OSTrack
