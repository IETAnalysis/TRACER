# -*- coding: utf-8 -*-
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew, kurtosis, entropy
import warnings

warnings.filterwarnings('ignore')

# =========================================================================
# ZERO-HYPERPARAMETER ABLATION CONFIGURATION
# =========================================================================
# Tunable coefficient-type hyperparameters are set to zero. Structural/runtime
# settings such as tensor dimensions, batch size, epoch count, number of trees,
# and class count are kept valid so that the program can still execute.
METRIC_BIAS = 0.0
METRIC_BETA = 0.0
FLOW_SCALE_FACTOR = 0.0
PROTO_TEMPERATURE = 0.0
OT_REG = 0.0
OT_MAX_ITER = 0
OT_LR = 0.0
SSL_RADIUS_MARGIN = 0.0
SSL_PUSH_WEIGHT = 0.0
SSL_NLL_WEIGHT = 0.0


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_simad_metrics(y_true, y_pred, bias=METRIC_BIAS, beta=METRIC_BETA):
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    u_pre = (pre - bias) / (1.0 - bias + 1e-8)
    n_pre = (pre - beta) / (1.0 - beta + 1e-8)
    
    def calc_f1(p, r):
        if p < 0: return - (2 * abs(p) * r) / (abs(p) + r + 1e-8)
        elif p + r == 0: return 0.0
        else: return (2 * p * r) / (p + r)
            
    return calc_f1(u_pre, rec), calc_f1(n_pre, rec)

def compute_pa_k(y_true, y_scores, k=None):
    if len(y_scores) == 0: return 0.0
    if k is None:
        total_anomalies = np.sum(y_true)
        k = (total_anomalies / len(y_true)) * 100 if len(y_true) > 0 else 0
        k = max(k, 0.01)
    threshold = np.percentile(y_scores, 100 - k)
    y_pred_adjusted = (y_scores >= threshold).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred_adjusted, average='binary', pos_label=1, zero_division=0)
    return f1

def extract_macro_stats(X_raw):
    B, L, C = X_raw.shape
    iats = X_raw[:, :, 0]
    lens = X_raw[:, :, 1]
    dirs = X_raw[:, :, 2]
    
    valid_mask = np.any(X_raw != 0, axis=2).astype(float)
    valid_count = np.sum(valid_mask, axis=1, keepdims=True) + 1e-6 
    
    duration = np.sum(iats * valid_mask, axis=1, keepdims=True) + 1e-6
    pkts_per_sec = valid_count / duration 
    bytes_per_sec = np.sum(lens * valid_mask, axis=1, keepdims=True) / duration
    
    iat_mean = duration / valid_count
    iat_std = np.sqrt(np.sum(((iats - iat_mean) * valid_mask)**2, axis=1, keepdims=True) / valid_count)
    iat_cv = iat_std / (iat_mean + 1e-8)
    iat_max = np.max(iats * valid_mask, axis=1, keepdims=True)
    iats_masked = np.where(valid_mask == 1, iats, np.inf)
    iat_min = np.min(iats_masked, axis=1, keepdims=True)
    iat_min[iat_min == np.inf] = 0.0
    
    total_bytes = np.sum(lens * valid_mask, axis=1, keepdims=True)
    len_mean = total_bytes / valid_count
    len_std = np.sqrt(np.sum(((lens - len_mean) * valid_mask)**2, axis=1, keepdims=True) / valid_count)
    len_cv = len_std / (len_mean + 1e-8)
    len_max = np.max(lens * valid_mask, axis=1, keepdims=True)
    lens_masked = np.where(valid_mask == 1, lens, np.inf)
    len_min = np.min(lens_masked, axis=1, keepdims=True)
    len_min[len_min == np.inf] = 0.0
    
    cov_len_iat = np.sum((lens - len_mean) * (iats - iat_mean) * valid_mask, axis=1, keepdims=True) / valid_count
    corr_len_iat = cov_len_iat / (len_std * iat_std + 1e-8)
    
    up_mask = (dirs == 0.0).astype(float) * valid_mask
    down_mask = (dirs == 1.0).astype(float) * valid_mask
    
    up_pkts = np.sum(up_mask, axis=1, keepdims=True) + 1e-6
    down_pkts = np.sum(down_mask, axis=1, keepdims=True) + 1e-6
    up_bytes = np.sum(lens * up_mask, axis=1, keepdims=True) + 1e-6
    down_bytes = np.sum(lens * down_mask, axis=1, keepdims=True) + 1e-6
    
    pkt_asym = up_pkts / down_pkts              
    byte_asym = up_bytes / down_bytes           
    up_mean_len = up_bytes / up_pkts            
    down_mean_len = down_bytes / down_pkts      
    up_iat_mean = np.sum(iats * up_mask, axis=1, keepdims=True) / up_pkts
    down_iat_mean = np.sum(iats * down_mask, axis=1, keepdims=True) / down_pkts
    
    up_len_std = np.sqrt(np.sum(((lens - up_mean_len) * up_mask)**2, axis=1, keepdims=True) / up_pkts)
    down_len_std = np.sqrt(np.sum(((lens - down_mean_len) * down_mask)**2, axis=1, keepdims=True) / down_pkts)
    up_iat_std = np.sqrt(np.sum(((iats - up_iat_mean) * up_mask)**2, axis=1, keepdims=True) / up_pkts)
    down_iat_std = np.sqrt(np.sum(((iats - down_iat_mean) * down_mask)**2, axis=1, keepdims=True) / down_pkts)
    
    first_pkt_len = lens[:, 0].reshape(B, 1)
    first_iat = iats[:, 1].reshape(B, 1) if L > 1 else np.zeros((B, 1))
    
    s_64, s_128, s_256, s_512 = 64.0/1500.0, 128.0/1500.0, 256.0/1500.0, 512.0/1500.0
    storm_ratio = np.sum((iats < 0.005).astype(float) * valid_mask, axis=1, keepdims=True) / valid_count
    hist_1 = np.sum((lens <= s_64).astype(float) * valid_mask, axis=1, keepdims=True) / valid_count
    hist_2 = np.sum(((lens > s_64) & (lens <= s_128)).astype(float) * valid_mask, axis=1, keepdims=True) / valid_count
    hist_3 = np.sum(((lens > s_128) & (lens <= s_256)).astype(float) * valid_mask, axis=1, keepdims=True) / valid_count
    hist_4 = np.sum(((lens > s_256) & (lens <= s_512)).astype(float) * valid_mask, axis=1, keepdims=True) / valid_count
    hist_5 = np.sum((lens > s_512).astype(float) * valid_mask, axis=1, keepdims=True) / valid_count
    
    max_burst_pkts = np.zeros((B, 1)); max_burst_bytes = np.zeros((B, 1))
    mean_burst_pkts = np.zeros((B, 1)); burst_count_feat = np.zeros((B, 1))
    dir_switches = np.zeros((B, 1))
    
    for i in range(B):
        v_mask = valid_mask[i]
        if np.sum(v_mask) == 0: continue
        flow_dirs, flow_lens = dirs[i][v_mask == 1], lens[i][v_mask == 1]
        if len(flow_dirs) == 0: continue
        
        switches = 0
        current_burst_dir = flow_dirs[0]
        current_burst_pkt, current_burst_byte = 0, 0
        burst_pkts_list, burst_bytes_list = [], []
        
        for j in range(len(flow_dirs)):
            if flow_dirs[j] == current_burst_dir:
                current_burst_pkt += 1; current_burst_byte += flow_lens[j]
            else:
                switches += 1
                burst_pkts_list.append(current_burst_pkt); burst_bytes_list.append(current_burst_byte)
                current_burst_dir = flow_dirs[j]
                current_burst_pkt, current_burst_byte = 1, flow_lens[j]
                
        burst_pkts_list.append(current_burst_pkt); burst_bytes_list.append(current_burst_byte)
        max_burst_pkts[i, 0] = max(burst_pkts_list) if burst_pkts_list else 0
        max_burst_bytes[i, 0] = max(burst_bytes_list) if burst_bytes_list else 0
        mean_burst_pkts[i, 0] = np.mean(burst_pkts_list) if burst_pkts_list else 0
        burst_count_feat[i, 0] = len(burst_pkts_list)
        dir_switches[i, 0] = switches

    dir_switch_ratio = dir_switches / valid_count

    curr_dirs, next_dirs = dirs[:, :-1], dirs[:, 1:]
    curr_lens, next_lens = lens[:, :-1], lens[:, 1:]
    curr_iats, next_iats = iats[:, :-1], iats[:, 1:]
    v_mask_trans = (valid_mask[:, :-1] * valid_mask[:, 1:]) == 1

    up_up = np.sum((curr_dirs == 0.0) & (next_dirs == 0.0) & v_mask_trans, axis=1, keepdims=True)
    up_down = np.sum((curr_dirs == 0.0) & (next_dirs == 1.0) & v_mask_trans, axis=1, keepdims=True)
    down_up = np.sum((curr_dirs == 1.0) & (next_dirs == 0.0) & v_mask_trans, axis=1, keepdims=True)
    down_down = np.sum((curr_dirs == 1.0) & (next_dirs == 1.0) & v_mask_trans, axis=1, keepdims=True)

    trans_total = np.sum(v_mask_trans, axis=1, keepdims=True) + 1e-6
    p_up_up = up_up / trans_total
    p_up_down = up_down / trans_total
    p_down_up = down_up / trans_total
    p_down_down = down_down / trans_total

    len_diff = np.abs(next_lens - curr_lens)
    lsv = np.sum(len_diff * v_mask_trans, axis=1, keepdims=True) / trans_total
    
    iat_diff = np.abs(next_iats - curr_iats)
    isv = np.sum(iat_diff * v_mask_trans, axis=1, keepdims=True) / trans_total

    is_fixed_len = (len_std < 1e-4).astype(float)

    final_macro_features = np.concatenate([
        duration, valid_count, pkts_per_sec, bytes_per_sec,
        iat_mean, iat_std, iat_max, iat_min, iat_cv,
        len_mean, len_std, len_max, len_min, len_cv, corr_len_iat,
        up_pkts, down_pkts, pkt_asym,
        up_bytes, down_bytes, byte_asym,
        up_mean_len, down_mean_len, up_iat_mean, down_iat_mean,
        storm_ratio, hist_1, hist_2, hist_3, hist_4, hist_5,
        max_burst_pkts, max_burst_bytes, mean_burst_pkts, burst_count_feat,
        dir_switches, dir_switch_ratio,
        up_len_std, down_len_std, up_iat_std, down_iat_std,
        first_pkt_len, first_iat,
        p_up_up, p_up_down, p_down_up, p_down_down, lsv, isv, is_fixed_len 
    ], axis=1)
    
    return np.nan_to_num(final_macro_features, nan=0.0, posinf=0.0, neginf=0.0)

# =========================================================================
# EXACT LIKELIHOOD NORMALIZING FLOWS
# =========================================================================
class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.mask = torch.arange(dim) % 2 
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim * 2)
        )
        
    def forward(self, x):
        device = x.device
        mask = self.mask.to(device).float()
        
        x_masked = x * mask
        st = self.net(x_masked)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * FLOW_SCALE_FACTOR 
        
        y = x_masked + (1 - mask) * (x * torch.exp(s) + t)
        log_det_jacobian = torch.sum((1 - mask) * s, dim=1)
        
        return y, log_det_jacobian

class NormalizingFlow(nn.Module):
    def __init__(self, dim=128, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([AffineCouplingLayer(dim) for _ in range(n_layers)])
        
    def forward(self, x):
        log_det_total = 0
        for layer in self.layers:
            x, ldj = layer(x)
            log_det_total += ldj
            x = torch.flip(x, dims=[1])
        return x, log_det_total

class XBM_DAF_Encoder(nn.Module):
    def __init__(self, seq_in=3, stats_in=24, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(seq_in, 16, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(seq_in, 16, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(seq_in, 16, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv1d(seq_in, 16, kernel_size=3, padding=8, dilation=8)
        self.bn_tcn = nn.BatchNorm1d(64)
        
        self.lstm = nn.LSTM(seq_in + 64, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.res_norm = nn.LayerNorm(hidden_dim)
        
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.query_proj = nn.Sequential(nn.Linear(stats_in, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, seq_x, stats_x):
        x = seq_x.transpose(1, 2)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x))
        
        tcn_concat = torch.cat([x1, x2, x3, x4], dim=1)
        tcn_out = self.bn_tcn(tcn_concat).transpose(1, 2)
        
        lstm_in = torch.cat([seq_x, tcn_out], dim=-1)
        H, _ = self.lstm(lstm_in)
        
        H_self, _ = self.self_attn(H, H, H)
        H_res = self.res_norm(H + H_self) 
        
        Q = self.query_proj(stats_x).unsqueeze(1)
        Z, _ = self.cross_attn(Q, H_res, H_res)
        
        H_max = torch.max(H_res, dim=1, keepdim=True)[0]
        
        return (Z + H_max).squeeze(1)

class ContrastiveProtoNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.temperature = PROTO_TEMPERATURE
        self.fc = nn.Linear(128, 128)
        
    def forward(self, seq_x, stats_x):
        emb = self.encoder(seq_x, stats_x)
        out = self.fc(emb)
        return F.normalize(out, p=2, dim=1)
        
    def proto_loss(self, embedding, n_cls, n_query):
        n_per_class = embedding.size(0) // n_cls
        n_shot = n_per_class - n_query
        
        emb_reshaped = embedding.view(n_per_class, n_cls, -1).transpose(0, 1)
        support = emb_reshaped[:, :n_shot].mean(1)
        query = emb_reshaped[:, n_shot:].contiguous().view(n_cls * n_query, -1)
        
        support_norm = F.normalize(support, p=2, dim=1)
        query_norm = F.normalize(query, p=2, dim=1)
        
        temperature = self.temperature if self.temperature > 0 else 1.0
        logits = torch.mm(query_norm, support_norm.t()) / temperature
        
        target_base = torch.arange(0, n_cls).view(n_cls, 1)
        target = target_base.expand(n_cls, n_query).long().contiguous().view(-1).to(embedding.device)
        
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (((1 - pt) ** 2.0) * ce_loss).mean()
        
        accuracy = torch.eq(logits.argmax(1), target).float().mean()
        return focal_loss, accuracy

class CategoriesSampler(Sampler):
    def __init__(self, labels, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.m_ind = [np.argwhere(labels == i).reshape(-1) for i in sorted(np.unique(labels))]
        
    def __len__(self): return self.n_batch
    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            for c in range(self.n_cls):
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                if len(pos) < self.n_per: pos = torch.randint(0, len(l), (self.n_per,))
                batch.append(torch.from_numpy(l[pos]))
            yield torch.stack(batch).t().reshape(-1)

class EvidentialFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, e_net, e_rf):
        alpha_net = e_net + 1.0
        alpha_rf = e_rf + 1.0
        S_net = torch.sum(alpha_net, dim=1, keepdim=True)
        S_rf = torch.sum(alpha_rf, dim=1, keepdim=True)
        u_net = self.num_classes / S_net
        u_rf = self.num_classes / S_rf
        
        mass_net = e_net / S_net
        mass_rf = e_rf / S_rf
        
        mass_fused = mass_net * mass_rf + mass_net * u_rf + mass_rf * u_net
        u_fused = u_net * u_rf
        
        e_fused = mass_fused * (self.num_classes / (u_fused + 1e-8))
        alpha_fused = e_fused + 1.0
        p_fused = alpha_fused / torch.sum(alpha_fused, dim=1, keepdim=True)
        return p_fused, u_fused, u_rf

class RiemannianOTProtoNet(nn.Module):
    def __init__(self, reg=OT_REG, max_iter=OT_MAX_ITER, ot_lr=OT_LR):
        super().__init__()
        self.reg = reg             
        self.max_iter = max_iter   
        self.ot_lr = ot_lr         

    def sinkhorn(self, C, a, b):
        K = torch.exp(-C / self.reg)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(self.max_iter):
            u = a / (torch.mv(K, v) + 1e-8)
            v = b / (torch.mv(K.t(), u) + 1e-8)
        return torch.diag(u) @ K @ torch.diag(v)

    def update_prototypes(self, prototypes, query_features):
        # Zero-valued OT hyperparameters disable prototype evolution safely.
        if self.reg <= 0 or self.max_iter <= 0 or self.ot_lr <= 0:
            return prototypes.clone()

        cos_sim = torch.mm(query_features, prototypes.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        C = torch.acos(cos_sim) 
        N_q, K = query_features.shape[0], prototypes.shape[0]
        a = torch.ones(N_q, device=query_features.device) / N_q
        b = torch.ones(K, device=query_features.device) / K
        Gamma = self.sinkhorn(C, a, b)
        
        updated_protos = prototypes.clone()
        for c in range(K):
            pull_direction = torch.mv(query_features.t(), Gamma[:, c])
            p_c = prototypes[c]
            v = pull_direction - torch.dot(pull_direction, p_c) * p_c
            v_norm = torch.norm(v).clamp(min=1e-8)
            p_new = p_c * torch.cos(v_norm * self.ot_lr) + (v / v_norm) * torch.sin(v_norm * self.ot_lr)
            updated_protos[c] = F.normalize(p_new, p=2, dim=0)
        return updated_protos

class XBMDAFTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        self.save_pth = os.path.join(args.save_pth, args.data_name)
        if not os.path.exists(self.save_pth): os.makedirs(self.save_pth, exist_ok=True)
            
        self.shared_save_pth = os.path.join(args.save_pth, 'shared_pretrain')
        if not os.path.exists(self.shared_save_pth): os.makedirs(self.shared_save_pth, exist_ok=True)
            
        self.label_enc = LabelEncoder()
        
        self.normal_idx = 0
        self.class_names = []
        self.is_etei = "etei" in self.args.data_name.lower()

    def compute_advanced_stats(self, x_np):
        B, L, C = x_np.shape
        features = []
        x_noisy = x_np + np.random.normal(0, 1e-6, x_np.shape)
        fft_mag = np.abs(np.fft.rfft(x_noisy, axis=1))
        for c in range(C):
            cd = x_noisy[:, :, c]
            md = fft_mag[:, :, c]
            mean_val = np.mean(cd, 1)
            std_val = np.std(cd, 1)
            skew_val = skew(cd, 1)
            kurt_val = kurtosis(cd, 1)
            fft_max = np.max(md, 1)
            fft_energy = np.sum(md**2, 1) / L
            fft_entropy = entropy(md / (np.sum(md, 1, keepdims=True) + 1e-9), axis=1)
            fft_mean = np.mean(md, 1)
            stats_c = np.stack([mean_val, std_val, skew_val, kurt_val, fft_max, fft_energy, fft_entropy, fft_mean], 1)
            features.append(np.nan_to_num(stats_c))
        return np.concatenate(features, axis=1)

    def pretrain_dual_purified_ssl(self):
        model_path = os.path.join(self.shared_save_pth, 'backbone_hypersphere_MTSSL.pth')
        center_path = os.path.join(self.shared_save_pth, 'hypersphere_center.npy')
        flow_path = os.path.join(self.shared_save_pth, 'backbone_flow.pth')
        
        if os.path.exists(model_path) and os.path.exists(center_path) and os.path.exists(flow_path): 
            print("\n   -> [Cache] Found Tri-Head SSL Backbone (Hypersphere 7-Class + Exact Likelihood NF).")
            self.fixed_center = torch.from_numpy(np.load(center_path)).to(self.device)
            return 
            
        source_dirs = [d.strip() for d in self.args.source_pth.split(',')]
        benign_pool = []
        for s_dir in source_dirs:
            if not os.path.exists(s_dir): continue
            x_f = [f for f in os.listdir(s_dir) if f.startswith('x_train')]
            y_f = [f for f in os.listdir(s_dir) if f.startswith('y_train')]
            c_f = [f for f in os.listdir(s_dir) if f.startswith('classes')]
            if not (x_f and y_f and c_f): continue
            X = np.load(os.path.join(s_dir, x_f[0]))
            Y = np.load(os.path.join(s_dir, y_f[0]))
            classes = np.load(os.path.join(s_dir, c_f[0]), allow_pickle=True)
            b_idx = -1
            for i, name in enumerate(classes):
                if 'benign' in str(name).lower() or 'normal' in str(name).lower(): 
                    b_idx = i; break
            X_pure = X[Y == b_idx]
            if len(X_pure) > 0: 
                np.random.shuffle(X_pure)
                benign_pool.append(X_pure[:8000])
                
        if not benign_pool: return
        base_x = np.concatenate(benign_pool)
        B, L, C = base_x.shape
        X_list, Y_list = [], []
        
        # 0: Normal + Jitter
        x0 = base_x.copy()
        x0[:,:,0] += np.random.normal(0, 0.01, size=x0[:,:,0].shape)
        x0[:,:,0] = np.clip(x0[:,:,0], 0, None)
        X_list.append(x0); Y_list.append(np.zeros(B)) 
        
        # 1: IAT Stretch
        x1 = base_x.copy()
        x1[:,:,0] *= 5.0
        X_list.append(x1); Y_list.append(np.ones(B) * 1) 
        
        # 2: Length shrink
        x2 = base_x.copy()
        x2[:,:,1] *= 0.5
        X_list.append(x2); Y_list.append(np.ones(B) * 2) 
        
        # 3: Direction inversion
        x3 = base_x.copy()
        x3[:,:,2] = 1.0 - x3[:,:,2]
        X_list.append(x3); Y_list.append(np.ones(B) * 3) 
        
        # 4: Pure Noise
        x4 = np.random.uniform(0, 1, size=(B, L, C))
        X_list.append(x4); Y_list.append(np.ones(B) * 4) 
        
        # 5: Micro-Burst Injection
        x5 = base_x.copy()
        for i in range(B):
            start = np.random.randint(0, max(1, L - 5))
            length = np.random.randint(3, 6)
            x5[i, start:start+length, 0] = np.random.uniform(0, 0.001, size=length) 
            x5[i, start:start+length, 1] = 60.0 / 1500.0 
        X_list.append(x5); Y_list.append(np.ones(B) * 5)
        
        # 6: Local Sequence Shuffling
        x6 = base_x.copy()
        for i in range(B):
            start = np.random.randint(0, max(1, L - 10))
            block = x6[i, start:start+10, :].copy()
            np.random.shuffle(block)
            x6[i, start:start+10, :] = block
        X_list.append(x6); Y_list.append(np.ones(B) * 6)
            
        X_pre, Y_pre = np.concatenate(X_list), np.concatenate(Y_list)
        stats_pre = self.compute_advanced_stats(X_pre)
        stats_tensor = torch.from_numpy(stats_pre).float()
        t_stats = (stats_tensor - stats_tensor.mean(0)) / (stats_tensor.std(0) + 1e-8)
        dataset = TensorDataset(torch.from_numpy(X_pre).float(), t_stats, torch.from_numpy(Y_pre).long())
        train_dl = DataLoader(dataset, batch_size=256, shuffle=True)
        
        encoder = XBM_DAF_Encoder(seq_in=3, stats_in=24).to(self.device)
        protonet_ssl = ContrastiveProtoNet(encoder).to(self.device)
        flow_net = NormalizingFlow(dim=128, n_layers=4).to(self.device)
        
        clf_ssl = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 7)).to(self.device)
        
        opt = torch.optim.AdamW(list(protonet_ssl.parameters()) + list(flow_net.parameters()) + list(clf_ssl.parameters()), lr=0.001)
        
        fixed_center = torch.ones(128).to(self.device)
        fixed_center = F.normalize(fixed_center, p=2, dim=0)
        radius_margin = SSL_RADIUS_MARGIN 
        
        print("\n   -> Starting Tri-Head SSL: Hypersphere 7-Class + Exact Likelihood NF (30 Epochs)...")
        for epoch in range(30):
            epoch_ce, epoch_pull, epoch_push, epoch_nll = 0.0, 0.0, 0.0, 0.0
            n_batches = len(train_dl)
            for bx, bs, by in train_dl:
                opt.zero_grad()
                
                z_raw = protonet_ssl.encoder(bx.to(self.device), bs.to(self.device))
                z_fc = protonet_ssl.fc(z_raw)
                z_norm = F.normalize(z_fc, p=2, dim=1) 
                
                logits = clf_ssl(z_norm)
                loss_ce = F.cross_entropy(logits, by.to(self.device))
                
                mask_normal = (by == 0)
                mask_anomaly = (by > 0)
                
                if mask_normal.sum() > 0:
                    z_benign = z_norm[mask_normal]
                    loss_pull = torch.mean(torch.sum((z_benign - fixed_center)**2, dim=1))
                else: 
                    loss_pull = torch.tensor(0.0).to(self.device)
                    
                if mask_anomaly.sum() > 0:
                    z_anom = z_norm[mask_anomaly]
                    dist_anom = torch.sqrt(torch.sum((z_anom - fixed_center)**2, dim=1) + 1e-8)
                    loss_push = torch.mean(F.relu(radius_margin - dist_anom)**2)
                else: 
                    loss_push = torch.tensor(0.0).to(self.device)
                    
                if mask_normal.sum() > 0:
                    z_raw_benign = z_raw[mask_normal]
                    u_norm, ldj = flow_net(z_raw_benign)
                    loss_nll = torch.mean(0.5 * torch.sum(u_norm**2, dim=1) - ldj)
                else:
                    loss_nll = torch.tensor(0.0).to(self.device)
                    
                total_loss = loss_ce + loss_pull + SSL_PUSH_WEIGHT * loss_push + SSL_NLL_WEIGHT * loss_nll
                total_loss.backward()
                opt.step()
                
        torch.save(protonet_ssl.encoder.state_dict(), model_path)
        torch.save(flow_net.state_dict(), flow_path)
        np.save(center_path, fixed_center.cpu().numpy())
        self.fixed_center = fixed_center
        print("   -> Tri-Head SSL Pretraining Completed.")

    def run_system(self):
        print("\n=======================================================")
        print(">>> [SYSTEM] ER-ADAPT 3D MULTI-MODE ENGINE <<<")
        print("    * Innovation 1: Riemannian OT-ProtoNet Evolution")
        print("    * Innovation 2: Subjective Evidential Fusion (EDL)")
        print("    * Innovation 3: Tri-Head SSL (7-Class Hypersphere + NF Regularization)")
        print("    * Innovation 4: Frozen Backbone Anti-Overfitting")
        print("    * Innovation 5: PURE Baseline Inference (No Hacky Gating)")
        print("=======================================================")
        print("    * Zero-hyperparameter ablation: ENABLED")
        print(f"      temperature={PROTO_TEMPERATURE}, OT=({OT_REG}, {OT_MAX_ITER}, {OT_LR}), "
              f"margin={SSL_RADIUS_MARGIN}, push_w={SSL_PUSH_WEIGHT}, nll_w={SSL_NLL_WEIGHT}")
        print("    * ARP traffic restriction: REMOVED")
        self.pretrain_dual_purified_ssl()
        
        print(f"\n>>> [Target Domain] CROSS-DOMAIN FEW-SHOT MODE ({self.args.fraction*100}%) <<<")
        X = np.load(os.path.join(self.args.data_pth, 'x_train.npy'))
        Y_raw_old = np.load(os.path.join(self.args.data_pth, 'y_train.npy'))
        cls_path = os.path.join(self.args.data_pth, 'classes.npy')
        if not os.path.exists(cls_path): cls_path = os.path.join(self.args.data_pth, 'classes_edge.npy')
        classes_old = np.load(cls_path, allow_pickle=True)
        y_str = np.array([classes_old[i] for i in Y_raw_old])
        
        y_enc = self.label_enc.fit_transform(y_str)
        self.class_names = [str(c) for c in self.label_enc.classes_]
        for i, n in enumerate(self.class_names):
            if 'benign' in n.lower() or 'normal' in n.lower(): 
                self.normal_idx = i; break
                
        stats = self.compute_advanced_stats(X)
        stats_tensor = torch.from_numpy(stats).float()
        full_s_t = (stats_tensor - stats_tensor.mean(0)) / (stats_tensor.std(0) + 1e-8)
        full_x_t = torch.from_numpy(X).float()
        
        print("   -> [Feature Engineering] Extracting 48D Physical Radar Stats for ML Expert...")
        full_x_tabular = extract_macro_stats(X)

        sup_idx, qry_idx = [], []
        for c in np.unique(y_enc):
            idx = np.random.permutation(np.where(y_enc == c)[0])
            n_sup = max(2, int(len(idx) * self.args.fraction))
            sup_idx.extend(idx[:n_sup])
            
            # Use all remaining samples for every class, including ARP traffic.
            qry_idx.extend(idx[n_sup:])
            
        encoder = XBM_DAF_Encoder(seq_in=3, stats_in=24).to(self.device)
        pretrained_path = os.path.join(self.shared_save_pth, 'backbone_hypersphere_MTSSL.pth')
        if os.path.exists(pretrained_path): 
            encoder.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        
        model = ContrastiveProtoNet(encoder).to(self.device)
        
        for p in model.encoder.parameters(): p.requires_grad = False
        for p in model.encoder.query_proj.parameters(): p.requires_grad = True
        for p in model.encoder.cross_attn.parameters(): p.requires_grad = True
        
        train_dataset = TensorDataset(full_x_t[sup_idx], full_s_t[sup_idx], torch.from_numpy(y_enc[sup_idx]).long())
        train_sampler = CategoriesSampler(y_enc[sup_idx], 100, len(self.class_names), 5)
        train_dl = DataLoader(train_dataset, batch_sampler=train_sampler)
        
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
        
        model.train()
        print("   -> Fine-Tuning Cross-Attention on Target Few-Shot Data (50 Epochs)...")
        for _ in range(50):
            for bx, bs, by in train_dl:
                opt.zero_grad()
                emb = model(bx.to(self.device), bs.to(self.device))
                loss, _ = model.proto_loss(emb, len(self.class_names), 2)
                loss.backward()
                opt.step()
        
        model.eval()
        with torch.no_grad():
            scaler_rf = RobustScaler()
            X_sup_rf_scaled = scaler_rf.fit_transform(full_x_tabular[sup_idx])
            
            clf_rf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=2021, n_jobs=-1)
            clf_rf.fit(X_sup_rf_scaled, y_enc[sup_idx])
            
            X_sup_f = model(full_x_t[sup_idx].to(self.device), full_s_t[sup_idx].to(self.device)).cpu().numpy()
            protos = {}
            for c in range(len(self.class_names)):
                class_features = X_sup_f[y_enc[sup_idx] == c]
                if len(class_features) > 0:
                    protos[c] = F.normalize(torch.from_numpy(class_features.mean(0)).unsqueeze(0), p=2, dim=1).numpy()
                else:
                    protos[c] = np.zeros((1, 128))

        print(f"\n>>> [Phase 4] Evaluation on Query Set <<<")
        feats_q = []
        q_dataset = TensorDataset(full_x_t[qry_idx], full_s_t[qry_idx])
        q_dl = DataLoader(q_dataset, batch_size=256)
        
        with torch.no_grad():
            for bx, bs in q_dl:
                z_norm_q = model(bx.to(self.device), bs.to(self.device))
                feats_q.append(z_norm_q.cpu().numpy())
                
        feats_q = np.concatenate(feats_q)
        
        X_qry_rf_scaled = scaler_rf.transform(full_x_tabular[qry_idx])
        prob_rf_q = clf_rf.predict_proba(X_qry_rf_scaled)
        
        feats_q_tensor = torch.from_numpy(feats_q).float().to(self.device)
        protos_tensor = torch.stack([torch.from_numpy(protos[c]).squeeze(0) for c in range(len(self.class_names))]).float().to(self.device)
        
        print("   -> [Architecture] Riemannian OT-ProtoNet Evolution...")
        ot_protonet = RiemannianOTProtoNet(reg=OT_REG, max_iter=OT_MAX_ITER, ot_lr=OT_LR)
        updated_protos = ot_protonet.update_prototypes(protos_tensor, feats_q_tensor)
        cos_sim_q = torch.mm(feats_q_tensor, updated_protos.t()) 
        
        print("   -> [Architecture] Subjective Evidential Fusion (EDL)...")
        evidence_net = F.relu(cos_sim_q * 10.0)
        
        rf_prob_tensor = torch.from_numpy(prob_rf_q).float().to(self.device)
        evidence_rf = rf_prob_tensor * 10.0 
        
        ev_fusion = EvidentialFusion(num_classes=len(self.class_names)).to(self.device)
        prob_fused_tensor, _, u_rf = ev_fusion(evidence_net, evidence_rf)
        
        prob_fused = prob_fused_tensor.cpu().numpy()
        print(f"      [Info] Average ML Expert Vacuity (Ignorance): {u_rf.mean().item():.4f}")
        
        final_preds = np.argmax(prob_fused, axis=1)

        y_true = y_enc[qry_idx]
        y_bin_true = (y_true != self.normal_idx).astype(int)
        y_bin_pred = (final_preds != self.normal_idx).astype(int)
        u_f1, n_f1 = compute_simad_metrics(y_bin_true, y_bin_pred, bias=METRIC_BIAS, beta=METRIC_BETA)
        pak = compute_pa_k(y_bin_true, 1.0 - prob_fused[:, self.normal_idx])
        
        print(f"\n[Metrics Summary]")
        print(f"   Binary Accuracy          : {accuracy_score(y_bin_true, y_bin_pred)*100:.2f}%")
        print(f"   Vanilla F1 (Binary)      : {f1_score(y_bin_true, y_bin_pred, pos_label=1):.4f}")
        print(f"   UAff-F1 (Unbiased)       : {u_f1:.4f}")
        print(f"   NAff-F1 (Normalized)     : {n_f1:.4f}")
        print(f"   PA%K (Point Adjustment)  : {pak:.4f}\n")

        print(">>> Classification Report <<<")
        print(classification_report(y_true, final_preds, target_names=self.class_names, digits=4))
        
        print("\n>>> Confusion Matrix <<<")
        cm = confusion_matrix(y_true, final_preds)
        cm_df = pd.DataFrame(cm, index=[f"T_{n}" for n in self.class_names], columns=[f"P_{n}" for n in self.class_names])
        pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000)
        print(cm_df)

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='edge_iot_cross')
parser.add_argument('--data_pth', type=str, required=True)
parser.add_argument('--source_pth', type=str, required=True)
parser.add_argument('--fraction', type=float, default=0.03) 
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--save_pth', type=str, default='./checkpoints/')
parser.add_argument('--win_size', type=int, default=128)
args = parser.parse_args()

if __name__ == '__main__':
    fix_seed(2021)
    XBMDAFTrainer(args).run_system()