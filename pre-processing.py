# -*- coding: utf-8 -*-
import sys
import os
import glob
import random
import numpy as np
from scapy.all import PcapReader, IP, IPv6, TCP, UDP
from tqdm import tqdm
import concurrent.futures
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration (Aligned with 3D XBM-DAF)
# ==========================================
DATA_DIR = r"/home/njust/data/yqj/SimAD-main later/dataset/BoT_IoT/"
# 输出路径修改为 3D 专属文件夹，防止覆盖
OUTPUT_DIR = r"/home/njust/data/yqj/SimAD-main later/dataset/BoT_IoT_3D_128"
MAX_PKT = 128  # Must match main.py win_size
IN_CHANS = 3   # 维度修改为 3

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

NORMAL_KEYWORDS = [
    "distance", "flame", "heart", "ir_receiver", "modbus",
    "phvalue", "soil", "sound", "temperature",
    "ultrasonic", "water", "normal", "benign"
]

def get_unified_label(folder_name):
    """Merges all sensor types into 'Benign' and cleans attack names."""
    name_lower = folder_name.lower()
    for kw in NORMAL_KEYWORDS:
        if kw in name_lower:
            return "Benign"
    
    clean_name = folder_name.rsplit('.', 1)[0]
    import re
    clean_name = re.sub(r'_\d+$', '', clean_name)
    return clean_name.capitalize()

# ==========================================
# 2. 3D Feature Extraction (IAT, Length, Direction)
# ==========================================
def process_single_pcap(pcap_file):
    stats_features = []
    src_ip = None
    last_time = None

    try:
        with PcapReader(pcap_file) as pcap_reader:
            for i, pkt in enumerate(pcap_reader):
                if i >= MAX_PKT: 
                    break
                
                if i == 0:
                    if IP in pkt: src_ip = pkt[IP].src
                    elif IPv6 in pkt: src_ip = pkt[IPv6].src
                    last_time = float(pkt.time)
                
                curr_time = float(pkt.time)
                iat = max(0.0, curr_time - last_time) if last_time is not None else 0.0
                last_time = curr_time
                
                # Length (为了与之前 3D 脚本保持一致，进行归一化，否则直接使用 len(pkt))
                pkt_len = float(len(pkt)) / 1500.0
                
                direction = 0.0 # Forward
                
                if IP in pkt:
                    if src_ip and pkt[IP].src != src_ip: direction = 1.0 # Backward
                elif IPv6 in pkt:
                    if src_ip and pkt[IPv6].src != src_ip: direction = 1.0
                        
                # Order: [IAT, Length, Direction]
                stats_features.append([iat, pkt_len, direction])
                
    except Exception: 
        return None
    
    if len(stats_features) < 2: 
        return None
        
    # 使用 3 维 0 向量进行 Padding
    while len(stats_features) < MAX_PKT:
        stats_features.append([0.0, 0.0, 0.0])
        
    return np.array(stats_features, dtype=np.float32)

# ==========================================
# 3. Multiprocessing Pipeline
# ==========================================
def main():
    folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    unified_classes = list(set([get_unified_label(f) for f in folders]))
    unified_classes.sort()
    print(f">>> Target Classes: {unified_classes}")
    
    all_seq, all_y = [], []
    
    for folder in folders:
        unified_label = get_unified_label(folder)
        label_idx = unified_classes.index(unified_label)
        
        pcap_files = glob.glob(os.path.join(DATA_DIR, folder, "*.pcap*"))
        if not pcap_files: continue
            
        # ?? 采样限制：正常 10000，其余 1000
        target_num = 10000 if unified_label == 'Benign' else 1000
        if len(pcap_files) > target_num:
            pcap_files = random.sample(pcap_files, target_num)
            
        print(f"Processing [{folder}] -> Class [{unified_label}] ({len(pcap_files)} files)")
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_single_pcap, pcap_files), total=len(pcap_files)))
            
        for res in results:
            if res is not None:
                all_seq.append(res)
                all_y.append(label_idx)

    if not all_seq:
        print("Error: No data extracted!")
        return

    all_seq = np.array(all_seq, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.int32)
    
    indices = np.arange(len(all_y))
    np.random.shuffle(indices)

    np.save(os.path.join(OUTPUT_DIR, "x_train.npy"), all_seq[indices])
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), all_y[indices])
    np.save(os.path.join(OUTPUT_DIR, "classes.npy"), np.array(unified_classes))

    print(f"Success! Final shape: {all_seq.shape}. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()