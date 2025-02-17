import os

import cv2
import torch

from lib.models.aiatrack import build_aiatrack
from lib.test.tracker.utils import Preprocessor
from lib.test.tracker.basetracker import BaseTracker
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import clip_box
from lib.utils.merge import merge_feature_sequence


# -----------------------------self defined func-----------------------------------------------
import sys
import numpy as np
sys.path.append("/root/LY/pytracking-master/pytracking")
from DPE_Module.depth.options import LiteMonoOptions
from torchvision import transforms, datasets
from DPE_Module.depth import networks
import numpy as np
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


class AIATRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(AIATRACK, self).__init__(params)
        network = build_aiatrack(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.net = network.cuda()
        self.net.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # For debug
        self.debug = False
        self.frame_id = 0
        # Set the hyper-parameters
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.HYPER, DATASET_NAME):
            self.cache_siz = self.cfg.TEST.HYPER[DATASET_NAME][0]
            self.refer_cap = 1 + self.cfg.TEST.HYPER[DATASET_NAME][1]
            self.threshold = self.cfg.TEST.HYPER[DATASET_NAME][2]
        else:
            self.cache_siz = self.cfg.TEST.HYPER.DEFAULT[0]
            self.refer_cap = 1 + self.cfg.TEST.HYPER.DEFAULT[1]
            self.threshold = self.cfg.TEST.HYPER.DEFAULT[2]
        if self.debug:
            self.save_dir = 'debug'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # For save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

    def initialize(self, image, info: dict, seq_name: str = None):
        print("this is attn")
        # Forward the long-term reference once
        # -----------------------------get DPE Module-----------------------------------------------
        #options = LiteMonoOptions()
        #opt = options.parse()
        self.output_state = 0
        self.frame_num =1
        encoder_path = r"Please configure the Path"
        decoder_path = r"Please configure the Path"
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
        self.refined_depth = 1
        self.disp_resized_np = 0
        self.init_state = info['init_bbox']
        self.psr = []
        # -----------------------------get DPE Module-----------------------------------------------
        
        
        refer_crop, resize_factor, refer_att_mask = sample_target(image, info['init_bbox'], self.params.search_factor,
                                                                  output_sz=self.params.search_size)
        refer_box = transform_image_to_crop(torch.Tensor(info['init_bbox']), torch.Tensor(info['init_bbox']),
                                            resize_factor,
                                            torch.Tensor([self.params.search_size, self.params.search_size]),
                                            normalize=True)
        self.feat_size = self.params.search_size // 16
        refer_img = self.preprocessor.process(refer_crop, refer_att_mask)
        with torch.no_grad():
            refer_dict = self.net.forward_backbone(refer_img)
            refer_dict_list = [refer_dict]
            refer_dict = merge_feature_sequence(refer_dict_list)
            refer_mem = self.net.transformer.run_encoder(refer_dict['feat'], refer_dict['mask'], refer_dict['pos'],
                                                         refer_dict['inr'])
        target_region = torch.zeros((self.feat_size, self.feat_size))
        x, y, w, h = (refer_box * self.feat_size).round().int()
        target_region[max(y, 0):min(y + h, self.feat_size), max(x, 0):min(x + w, self.feat_size)] = 1
        target_region = target_region.view(self.feat_size * self.feat_size, -1)
        background_region = 1 - target_region
        refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0).cuda()
        embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight],
                               dim=0).unsqueeze(0)
        self.refer_mem_cache = [refer_mem]
        self.refer_emb_cache = [torch.bmm(refer_region, embed_bank).transpose(0, 1)]
        self.refer_pos_cache = [refer_dict['inr']]
        self.refer_msk_cache = [refer_dict['mask']]
        self.refer_mem_list = list()
        for _ in range(self.refer_cap):
            self.refer_mem_list.append(self.refer_mem_cache[0])
        self.refer_emb_list = list()
        for _ in range(self.refer_cap):
            self.refer_emb_list.append(self.refer_emb_cache[0])
        self.refer_pos_list = list()
        for _ in range(self.refer_cap):
            self.refer_pos_list.append(self.refer_pos_cache[0])
        self.refer_msk_list = list()
        for _ in range(self.refer_cap):
            self.refer_msk_list.append(self.refer_msk_cache[0])
        # Save states
        self.state = info['init_bbox']
        if self.save_all_boxes:
            # Save all predicted boxes
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {'all_boxes': all_boxes_save}

    def track(self, image, info: dict = None, seq_name: str = None):
        H, W, _ = image.shape
        self.frame_id += 1
        
        # Get the t-th search region
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
            #self.disp_resized_np = self.k2*disp_resized_np_ + (1-self.k2)*self.disp_resized_np
            #print(self.output_state)
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
            
        #print(type(self.refined_depth),self.refined_depth.shape)
        #---------------------visualize the depth estimatio colored map----------
        #vmax = np.percentile(self.disp_resized_np, 95)
        #normalizer = mpl.colors.Normalize(vmin=self.disp_resized_np.min(), vmax=vmax)
        #mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        #colormapped_im = (mapper.to_rgba(self.disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        #---------------------visualize the depth estimatio colored map----------
        
        
        image = (self.k1*image*self.refined_depth[:,:,None] + (1-self.k1)*image).astype(np.uint8)
        #self.image = image
        
        # ---------------------------------------------------------------------
        
        search_crop, resize_factor, search_att_mask = sample_target(image, self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
        search_img = self.preprocessor.process(search_crop, search_att_mask)
        with torch.no_grad():
            search_dict = self.net.forward_backbone(search_img)
            # Merge the feature sequence
            search_dict_list = [search_dict]
            search_dict = merge_feature_sequence(search_dict_list)
            # Run the transformer
            out_embed, search_mem, pos_emb, key_mask = self.net.forward_transformer(search_dic=search_dict,
                                                                                    refer_mem_list=self.refer_mem_list,
                                                                                    refer_emb_list=self.refer_emb_list,
                                                                                    refer_pos_list=self.refer_pos_list,
                                                                                    refer_msk_list=self.refer_msk_list)
            # Forward the corner head
            out_dict, outputs_coord = self.net.forward_box_head(out_embed)
            # out_dict: (B, N, C), outputs_coord: (1, B, N, C)

            pred_iou = self.net.forward_iou_head(out_embed, outputs_coord.unsqueeze(0).unsqueeze(0))

        # Get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # ---------------------attn---------------------------
        self.output_state = pred_boxes[0]
        # ---------------------attn---------------------------
        # Baseline: Take the mean of all predicted boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # Get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        predicted_iou = pred_iou['pred_iou'][0][0][0].item()

        # Update state
        if predicted_iou > self.threshold:
            if len(self.refer_mem_cache) == self.cache_siz:
                _ = self.refer_mem_cache.pop(1)
                _ = self.refer_emb_cache.pop(1)
                _ = self.refer_pos_cache.pop(1)
                _ = self.refer_msk_cache.pop(1)
            target_region = torch.zeros((self.feat_size, self.feat_size))
            x, y, w, h = (outputs_coord[0] * self.feat_size).round().int()
            target_region[max(y, 0):min(y + h, self.feat_size), max(x, 0):min(x + w, self.feat_size)] = 1
            target_region = target_region.view(self.feat_size * self.feat_size, -1)
            background_region = 1 - target_region
            refer_region = torch.cat([target_region, background_region], dim=1).unsqueeze(0).cuda()
            embed_bank = torch.cat([self.net.foreground_embed.weight, self.net.background_embed.weight],
                                   dim=0).unsqueeze(0)
            new_emb = torch.bmm(refer_region, embed_bank).transpose(0, 1)
            self.refer_mem_cache.append(search_mem)
            self.refer_emb_cache.append(new_emb)
            self.refer_pos_cache.append(pos_emb)
            self.refer_msk_cache.append(key_mask)

            self.refer_mem_list = [self.refer_mem_cache[0]]
            self.refer_emb_list = [self.refer_emb_cache[0]]
            self.refer_pos_list = [self.refer_pos_cache[0]]
            self.refer_msk_list = [self.refer_msk_cache[0]]
            max_idx = len(self.refer_mem_cache) - 1
            ensemble = self.refer_cap - 1
            for part in range(ensemble):
                self.refer_mem_list.append(self.refer_mem_cache[max_idx * (part + 1) // ensemble])
                self.refer_emb_list.append(self.refer_emb_cache[max_idx * (part + 1) // ensemble])
                self.refer_pos_list.append(self.refer_pos_cache[max_idx * (part + 1) // ensemble])
                self.refer_msk_list.append(self.refer_msk_cache[max_idx * (part + 1) // ensemble])

        # For debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 255, 0), thickness=3)
            save_seq_dir = os.path.join(self.save_dir, seq_name)
            if not os.path.exists(save_seq_dir):
                os.makedirs(save_seq_dir)
            save_path = os.path.join(save_seq_dir, '%04d.jpg' % self.frame_id)
            cv2.imwrite(save_path, image_BGR)
        if self.save_all_boxes:
            # Save all predictions
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N,)
            return {'target_bbox': self.state,
                    'all_boxes': all_boxes_save}
        else:
            return {'target_bbox': self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return AIATRACK
