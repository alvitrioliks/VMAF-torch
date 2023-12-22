"""VMAF metric

Based on:
    https://github.com/Netflix/vmaf

References:
    https://netflixtechblog.com/toward-a-practical-perceptual-video-quality-metric-653f208b9652

"""

import numpy as np
import pandas as pd
import torch

from .motion import Motion
from .adm import ADM
from .vif import VIF
from .svm_predict import SVMPredict


class VMAF(torch.nn.Module):
    """VMAF module
    Args:
        temporal_pooling (bool): if False output tensor of scores for each frame,
            if True output mean score computed over frame dimension, i.e. score for the whole video
        enable_motion (bool): if False set motion for all frames to 0 e.g. when computing on batch of images instead of video 
        clip_score (bool): if True clip final VMAF score to [0,100]
        NEG (bool): if True compute VMAF NEG version, if False compute regular VMAF
        model_json_path (str): path to custom SVM model json, if None default model is used

    """

    def __init__(self,
                 temporal_pooling=False,
                 enable_motion=True,
                 clip_score=False,
                 NEG=False,
                 model_json_path=None,
                 ):
        super().__init__()
        self.temporal_pooling = temporal_pooling
        self.enable_motion = enable_motion

        if self.enable_motion:
            self.motion = Motion()
        else:
            self.motion = None
        self.vif = VIF(NEG=NEG)
        self.adm = ADM(NEG=NEG)
        self.svm = SVMPredict(clip_score=clip_score, model_json_path=model_json_path)

    def forward(self, ref, dist):
        return self.compute_vmaf_score(ref, dist)

    def compute_vmaf_score(self, ref, dist):
        """Computation of VMAF score
        Args:
            ref: An input tensor with [N, 1, H, W] shape, reference image, Y channel only, in range [0,255]
                first dimension is frame dimension for video or batch dimension for batch of images
            dist: An input tensor with [N, 1, H, W] shape, distorted image, Y channel only, in range [0,255],
                order of arguments matters!
        Returns:
            Value of VMAF metric.
        """

        motion2_score = self.compute_motion2(ref)
        adm_score = self.compute_adm_score(ref, dist)
        vif_features = self.compute_vif_features(ref, dist)

        vmaf_score = self.predict(adm_score, motion2_score, vif_features)

        if self.temporal_pooling:
            vmaf_score = torch.mean(vmaf_score, dim=0)  # mean over frames

        return vmaf_score

    def compute_vmaf_features_and_score(self, ref, dist):
        """Compute VMAF score and needed features (faster then calling individual functions due to computation sharing)"""
        motion2_score = self.compute_motion2(ref)
        adm_score = self.compute_adm_score(ref, dist)
        vif_features = self.compute_vif_features(ref, dist)
        vmaf_score = self.predict(adm_score, motion2_score, vif_features)
        return motion2_score, adm_score, vif_features, vmaf_score

    def compute_motion(self, ref):
        """Compute motion feature"""        
        if self.enable_motion:
            return self.motion.motion(ref)
        else:
            return torch.zeros((ref.shape[0], 1), device=ref.device, dtype=ref.dtype)

    def compute_motion2(self, ref):
        """Compute motion2 feature"""
        if self.enable_motion:
            return self.motion.motion2(ref)
        else:
            return torch.zeros((ref.shape[0], 1), device=ref.device, dtype=ref.dtype)

    def compute_vif_features(self, ref, dist):
        """Compute one VIF feature for each scale"""
        return self.vif.vif_features(ref, dist)

    def compute_vif_score(self, ref, dist):
        """Compute VIF score using all 4 image scales"""
        return self.vif.vif_score(ref, dist)

    def compute_vif_features_and_score(self, ref, dist):
        """Compute VIF features and score (faster then calling individual functions due to computation sharing)"""
        return self.vif.vif_features_and_score(ref, dist)

    def compute_adm_features(self, ref, dist):
        """Compute one ADM feature for each scale"""
        return self.adm.adm_features(ref, dist)

    def compute_adm_score(self, ref, dist):
        """Compute ADM score using all 4 image scales"""
        return self.adm.adm_score(ref, dist)

    def compute_adm_features_and_score(self, ref, dist):
        """Compute ADM features and score (faster then calling individual functions due to computation sharing)"""
        return self.adm.adm_features_and_score(ref, dist)

    def predict(self, adm, motion, vif):
        features = torch.cat([adm, motion, vif], dim=-1)
        score = self.svm(features)
        return score

    @torch.no_grad()
    def table(self, ref, dist):
        '''Compute table with features and score for each frame similar to csv produced by vmaf command line tool with --csv argument
        Args:
            ref: An input tensor with [N, 1, H, W] shape, reference image, Y channel only, in range [0,255]
                first dimension is frame dimension for video or batch dimension for batch of images
            dist: An input tensor with [N, 1, H, W] shape, distorted image, Y channel only, in range [0,255],
                order of arguments matters!
        Returns:
            pandas DataFrame
        '''

        motion_score = self.compute_motion(ref)
        motion2_score = self.compute_motion2(ref)
        adm_features, adm_score = self.compute_adm_features_and_score(ref, dist)
        vif_features = self.compute_vif_features(ref, dist)
        vmaf_score = self.predict(adm_score, motion2_score, vif_features)

        features = torch.cat([motion2_score, motion_score, adm_score, adm_features, vif_features, vmaf_score], dim=-1).detach().cpu().numpy()

        # features computed here do not use integers, column names match names produced by vmaf executable for convenience
        df = pd.DataFrame(columns=['Frame',
                                   'integer_motion2',
                                   'integer_motion',
                                   'integer_adm2',
                                   'integer_adm_scale0',
                                   'integer_adm_scale1',
                                   'integer_adm_scale2',
                                   'integer_adm_scale3',
                                   'integer_vif_scale0',
                                   'integer_vif_scale1',
                                   'integer_vif_scale2',
                                   'integer_vif_scale3',
                                   'vmaf'])
        num_frames = ref.shape[0]
        df['Frame'] = np.arange(num_frames)
        df.iloc[:num_frames, 1:] = features
        return df
