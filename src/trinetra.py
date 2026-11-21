import os, math, random, time, copy
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Imports: Transformers / PyG / Sklearn
# -----------------------
try:
    from transformers import GPT2Model, GPT2Config
    print("Hugging Face Transformers (GPT-2) loaded successfully.")
except ImportError:
    raise ImportError("Please install 'transformers' via: pip install transformers")

try:
    from torch_geometric.nn import SAGEConv, GATv2Conv
except Exception as e:
    raise RuntimeError("torch_geometric not found. Install correct torch-geometric.") from e

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# -----------------------
# 1. Config / Hyperparams
# -----------------------
CSV_PATH = os.environ.get("CSV_PATH", "data/foursquare_nyc.csv")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SPEED OPTIMIZATION SETTINGS ---
if torch.cuda.device_count() > 1:
    BATCH = 128
    print(f"🚀 Multi-GPU Activated: {torch.cuda.device_count()} GPUs. Using Batch Size {BATCH}")
else:
    BATCH = 128
    print("⚠️ Single GPU detected. Using Batch Size 128")

# --- PAPER REPLICATION SETTINGS ---
# "filtered out inactive users ... fewer than 10 check-ins"
MIN_CHECKINS_PER_USER = 10 
# "venues with fewer than 10 visits"
POI_MIN = 10            

MAX_SEQ_LEN = 100 
EPOCHS = 100                    
LR = 1e-3 
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
NUM_WORKERS = 4 
EARLY_STOP = 5

# Advanced Params
LAMBDA_CL = 0.1      
LAMBDA_LOC = 1.0    
CL_TEMP = 0.07      

# Clustering & Graph
N_CLUSTERS = 50             
SPATIAL_K = 10              
CAT_EMB_DIM = 32            
CLUSTER_EMB_DIM = 16        

EMB_SIZES =[128]

# UPDATED: Specifically requested K values for evaluation
KS = (1, 5, 10, 20)        

# -----------------------
# Utilities (UPDATED WITH METRICS FROM IMAGE)
# -----------------------
def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371000.0 # Meters
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*(np.sin(dlambda/2.0)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def accuracy_topk_batch(logits, targets, ks=(1,5,10,20)):
    """
    Computes Acc@k as per Equation (25) in the image.
    Acc@k = 1/|X| * sum(indicator(label in top-k))
    """
    max_k = min(max(ks), logits.size(1))
    _, topk_idx = torch.topk(logits, max_k, dim=1)
    targets_expand = targets.view(-1,1).expand_as(topk_idx)
    correct = topk_idx.eq(targets_expand)
    res = []
    for k in ks:
        if k <= max_k:
            correct_k = correct[:, :k].any(dim=1).float().mean().item()
            res.append(correct_k)
        else:
            res.append(0.0)
    return res

def get_mrr_at_k_batch(logits, targets, ks=(1,5,10,20)):
    """
    Computes MRR@k as per Equation (26) in the image.
    If the rank of the target > k, the reciprocal rank is considered 0.
    """
    max_k = min(max(ks), logits.size(1))
    # Get top k indices (descending sort happens inside topk)
    _, topk_idx = torch.topk(logits, max_k, dim=1)
    
    targets_expand = targets.view(-1, 1).expand_as(topk_idx)
    # hits is a boolean tensor [Batch, max_k]
    hits = (topk_idx == targets_expand)
    
    res = []
    for k in ks:
        if k <= max_k:
            # Slice to current k
            hits_k = hits[:, :k] 
            
            # Mask: 1 if target is in top-k, 0 otherwise
            mask = hits_k.any(dim=1).float()
            
            # Find the rank (0-based index) of the hit
            # argmax finds the first True. If no True, it returns 0 (handled by mask)
            # We add 1.0 to get 1-based rank.
            ranks = hits_k.float().argmax(dim=1) + 1.0
            
            # Calculate Reciprocal Rank: 1/rank if hit, else 0
            recip_ranks = (1.0 / ranks) * mask
            
            # Mean over the batch
            res.append(recip_ranks.mean().item())
        else:
            res.append(0.0)
    return res

def get_ndcg_batch(logits, targets, ks=(1,5,10,20)):
    max_k = min(max(ks), logits.size(1))
    _, topk_idx = torch.topk(logits, max_k, dim=1) 
    device = logits.device
    ranks = torch.arange(max_k, device=device).float()
    discounts = 1.0 / torch.log2(ranks + 2.0) 
    targets_expand = targets.view(-1, 1).expand_as(topk_idx)
    hits = (topk_idx == targets_expand).float() 
    dcg_matrix = hits * discounts 
    res = []
    for k in ks:
        if k <= max_k:
            ndcg_val = dcg_matrix[:, :k].sum(dim=1).mean().item()
            res.append(ndcg_val)
        else:
            res.append(0.0)
    return res

def print_metrics_row(split_name, loss, accs, mrrs, m_acc, n_samples=0, elapsed=0):
    """
    Updated to print Acc@K, MRR@K, and mAcc explicitly as requested.
    """
    acc_str = ", ".join([f"{x:.4f}" for x in accs])
    mrr_str = ", ".join([f"{x:.4f}" for x in mrrs])
    
    print(f"[{split_name}] {elapsed:.1f}s | Loss: {loss:.4f} | mAcc: {m_acc:.4f}")
    print(f"   Acc@{KS} : {acc_str}")
    print(f"   MRR@{KS} : {mrr_str}")

# -----------------------
# 2. Load & Preprocessing
# -----------------------
print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# Normalize columns
if 'utcTimestamp' in df.columns and 'timestamp' not in df.columns:
    df["timestamp"] = df["utcTimestamp"]
col_map = {}
if 'userId' not in df.columns and 'user_id' in df.columns: col_map['user_id'] = 'userId'
if 'venueId' not in df.columns and 'venue_id' in df.columns: col_map['venue_id'] = 'venueId'
if 'venueCategoryId' not in df.columns and 'category' in df.columns: col_map['category'] = 'venueCategoryId'
if 'timestamp' not in df.columns and 'time' in df.columns: col_map['time'] = 'timestamp'
if col_map:
    df = df.rename(columns=col_map)

df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
df = df.dropna(subset=['timestamp']).reset_index(drop=True)

# --- 2.1 PAPER REPLICATION: FILTERING ---
print("Applying filtering rules from Paper (Min 10 checkins, Min 10 visits)...")

# 1. Filter Venues < 10 visits
venue_counts_raw = df['venueId'].value_counts()
valid_venues_list = venue_counts_raw[venue_counts_raw >= POI_MIN].index
df = df[df['venueId'].isin(valid_venues_list)].reset_index(drop=True)

# 2. Filter Users < 10 check-ins
user_counts = df['userId'].value_counts()
valid_users = user_counts[user_counts >= MIN_CHECKINS_PER_USER].index
df = df[df['userId'].isin(valid_users)].reset_index(drop=True)

# Encodings
user_enc = LabelEncoder(); venue_enc = LabelEncoder(); cat_enc = LabelEncoder()
df['user_enc'] = user_enc.fit_transform(df['userId'])
df['venue_enc'] = venue_enc.fit_transform(df['venueId']) + 1 # 0 is padding
df['venueCategoryId'] = df['venueCategory'].astype(str)
df['cat_enc'] = cat_enc.fit_transform(df['venueCategoryId']) + 1

n_users = df['user_enc'].nunique()
n_venues = df['venue_enc'].max() + 1 
n_cats = df['cat_enc'].max() + 1

print(f"Filtered Dataset: {len(df)} check-ins.")
print(f"Vocab Size: {n_venues} venues, {n_users} users.")

# Coords Calculation
venue_coords = df.groupby('venue_enc')[['latitude','longitude']].mean()
lat_mean, lat_std = venue_coords['latitude'].mean(), venue_coords['latitude'].std()
lon_mean, lon_std = venue_coords['longitude'].mean(), venue_coords['longitude'].std()

venue_coords['lat_norm'] = (venue_coords['latitude'] - lat_mean) / (lat_std + 1e-9)
venue_coords['lon_norm'] = (venue_coords['longitude'] - lon_mean) / (lon_std + 1e-9)

# ------------------------------------------------------------------------
# 3. GENERATE SAMPLES (UPDATED: PAPER SPLITTING STRATEGY)
# ------------------------------------------------------------------------
print("\nGenerating sequences using (n-2) Train, (n-1) Val, (n) Test splits...")

train_samples = []
val_samples = []
test_samples = []

# Helper to process a specific slice of history for a specific target
def create_sample_from_indices(venues, times, lats, lons, uid, target_idx):
    """
    Creates a sample where venues[target_idx] is the label.
    Context is venues[target_idx-MAX_SEQ_LEN : target_idx]
    """
    tgt_id = venues[target_idx]
    tgt_lat, tgt_lon = lats[target_idx], lons[target_idx]
    
    # Define Context Window
    start_idx = max(0, target_idx - MAX_SEQ_LEN)
    
    seq_v = venues[start_idx : target_idx]
    seq_t = times[start_idx : target_idx]
    seq_lat = lats[start_idx : target_idx]
    seq_lon = lons[start_idx : target_idx]
    
    # Calculate intervals (Time & Dist)
    d_times, d_dists = [0.0], [0.0]
    
    for k in range(1, len(seq_v)):
        dt = (seq_t[k] - seq_t[k-1]).total_seconds()
        dd = haversine_np(seq_lat[k-1], seq_lon[k-1], seq_lat[k], seq_lon[k])
        d_times.append(dt)
        d_dists.append(dd)
        
    st_arr = [(t_obj.hour, t_obj.dayofweek) for t_obj in seq_t]
        
    # Padding
    pad_len = MAX_SEQ_LEN - len(seq_v)
    
    seq_v_final = [0]*pad_len + seq_v
    st_arr_final = [(0,0)]*pad_len + st_arr
    
    d_times_log = [math.log(x + 1.0) for x in d_times]
    d_dists_log = [math.log(x + 1.0) for x in d_dists]
    
    d_times_final = [0.0]*pad_len + d_times_log
    d_dists_final = [0.0]*pad_len + d_dists_log
    
    # Target Geo Normalization
    tgt_y = (tgt_lat - lat_mean) / (lat_std + 1e-9)
    tgt_x = (tgt_lon - lon_mean) / (lon_std + 1e-9)
    
    return (
        seq_v_final, st_arr_final, d_times_final, d_dists_final, 
        int(uid), int(tgt_id), (tgt_y, tgt_x) 
    )

# Iterate Users
for uid, g in tqdm(df.groupby('user_enc'), desc="Processing Users"):
    # Sort chronologically to form trajectory 1...n
    g = g.sort_values('timestamp')
    
    venues = g['venue_enc'].tolist()
    times = g['timestamp'].tolist()
    lats = g['latitude'].tolist()
    lons = g['longitude'].tolist()
    
    n = len(venues)
    
    # We need at least 3 items to have Train(n-2), Val(n-1), Test(n)
    # The filter ensures >= 10, so we are safe, but good to check.
    if n < 3:
        continue
        
    # Indices (0-based)
    # n (length) -> Indices are 0 to n-1
    # Test Item: Index n-1 (The last item)
    # Val Item:  Index n-2 (The second to last item)
    # Train Items: Indices 1 to n-3 (The rest)
    
    idx_test = n - 1
    idx_val  = n - 2
    idx_train_end = n - 3 # Inclusive of the last training target
    
    # 1. Generate Test Sample
    test_samples.append(create_sample_from_indices(venues, times, lats, lons, uid, idx_test))
    
    # 2. Generate Validation Sample
    val_samples.append(create_sample_from_indices(venues, times, lats, lons, uid, idx_val))
    
    # 3. Generate Training Samples
    # We start from index 1 (since index 0 has no history before it) up to idx_train_end
    for i in range(1, idx_train_end + 1):
        train_samples.append(create_sample_from_indices(venues, times, lats, lons, uid, i))

print(f"Split Complete.")
print(f"Train Samples: {len(train_samples)}")
print(f"Val Samples:   {len(val_samples)}")
print(f"Test Samples:  {len(test_samples)}")

# ------------------------------------------------------------------------
# 4. GRAPH & CLUSTERS
# ------------------------------------------------------------------------
print("Building Graphs...")

# A. Coords Array
coords_full = np.zeros((n_venues, 2), dtype=np.float32)
for vid, row in venue_coords.iterrows():
    coords_full[int(vid), 0] = row['latitude']
    coords_full[int(vid), 1] = row['longitude']
valid_coords = coords_full[1:] 

# B. KMeans
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
valid_clusters = kmeans.fit_predict(valid_coords) 
venue_clusters_full = np.zeros(n_venues, dtype=np.int64)
venue_clusters_full[1:] = valid_clusters + 1 
venue_cluster_tensor = torch.tensor(venue_clusters_full, dtype=torch.long)

# C. Categories
venue_cat_ids = df.groupby('venue_enc')['cat_enc'].first()
venue_cat_tensor_full = torch.zeros(n_venues, dtype=torch.long)
for vid, cat in venue_cat_ids.items():
    venue_cat_tensor_full[int(vid)] = int(cat)
venue_cat_tensor = venue_cat_tensor_full

# D. Transition Graph
# Graph is built ONLY from edges present in the Training set to avoid leakage
train_edges_dict = defaultdict(int)
for seq, _, _, _, _, _, _ in tqdm(train_samples):
    # seq contains 0-padding, filter those out
    valid_seq = [x for x in seq if x != 0]
    for i in range(len(valid_seq) - 1):
        u, v = valid_seq[i], valid_seq[i+1]
        if u == v: continue
        train_edges_dict[(u,v)] += 1
        train_edges_dict[(v,u)] += 1 

edge_list_trans = []
for (u,v), count in train_edges_dict.items():
    edge_list_trans.append([u, v])
if not edge_list_trans:
    for v in range(1, n_venues): edge_list_trans.append([v,v])
edge_index_trans = torch.tensor(edge_list_trans, dtype=torch.long).t().contiguous()

# E. Geo Graph (KNN)
nbrs = NearestNeighbors(n_neighbors=SPATIAL_K+1, algorithm='ball_tree').fit(valid_coords)
_, indices = nbrs.kneighbors(valid_coords)
edge_list_geo = []
for i in range(len(valid_coords)):
    src_id = i + 1
    for neighbor_idx in indices[i, 1:]:
        tgt_id = neighbor_idx + 1
        edge_list_geo.append([src_id, tgt_id])
        edge_list_geo.append([tgt_id, src_id]) 
edge_index_geo = torch.tensor(edge_list_geo, dtype=torch.long).t().contiguous()

# Node Features
venue_cat_onehot = np.zeros((n_venues, n_cats), dtype=np.float32)
latlon_norm = np.zeros((n_venues, 2), dtype=np.float32)
cat_mode = df.groupby('venue_enc')['cat_enc'].agg(lambda x: x.mode().iloc[0]).to_dict()
for vid in range(1, n_venues):
    if vid in cat_mode:
        venue_cat_onehot[vid, int(cat_mode[vid])] = 1.0
    if vid in venue_coords.index:
        latlon_norm[vid, 0] = venue_coords.loc[vid, 'lat_norm']
        latlon_norm[vid, 1] = venue_coords.loc[vid, 'lon_norm']

graph_input_feats_np = np.concatenate([venue_cat_onehot, latlon_norm], axis=1).astype(np.float32)
graph_input_feats = torch.tensor(graph_input_feats_np, dtype=torch.float32)

# -----------------------
# Dataset
# -----------------------
class SeqDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq, time_pair, d_time, d_dist, uid, tgt, (ty, tx) = self.samples[idx]
        return (torch.tensor(seq, dtype=torch.long), 
                torch.tensor(time_pair, dtype=torch.long),
                torch.tensor(d_time, dtype=torch.float), 
                torch.tensor(d_dist, dtype=torch.float), 
                torch.tensor(uid, dtype=torch.long),
                torch.tensor(tgt, dtype=torch.long),
                torch.tensor([ty, tx], dtype=torch.float)) 

# OPTIMIZED DATALOADERS
train_loader = DataLoader(
    SeqDataset(train_samples), 
    batch_size=BATCH, 
    shuffle=True, # Shuffle training data samples
    num_workers=NUM_WORKERS, 
    pin_memory=True, 
    prefetch_factor=2 
)
# Do not shuffle Val/Test to keep consistent evaluation order (optional but good practice)
val_loader = DataLoader(SeqDataset(val_samples), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(SeqDataset(test_samples), batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# -----------------------
# MODEL COMPONENTS
# -----------------------
class IntervalEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(2, emb_dim)
        self.act = nn.Tanh()
    def forward(self, d_time, d_dist):
        x = torch.stack([d_time, d_dist], dim=-1)
        return self.act(self.linear(x))

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim, hour_dim=16, dow_dim=8):
        super().__init__()
        self.hour_emb = nn.Embedding(24, hour_dim)
        self.dow_emb = nn.Embedding(7, dow_dim)
        self.proj = nn.Linear(hour_dim + dow_dim, emb_dim)
    def forward(self, time_pair):
        h = self.hour_emb(time_pair[:,:,0])
        d = self.dow_emb(time_pair[:,:,1])
        return self.proj(torch.cat([h,d], dim=-1))

class DualGraphEncoder_GAT(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.25):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.conv_trans = GATv2Conv(hidden, hidden, heads=2, concat=False, dropout=dropout)
        self.conv_geo = GATv2Conv(hidden, hidden, heads=2, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)
        self.fusion_gate = nn.Linear(hidden * 2, 1)

    def forward(self, feats, edge_index_trans, edge_index_geo):
        x = F.elu(self.input_proj(feats))
        h_trans = self.conv_trans(x, edge_index_trans)
        h_geo = self.conv_geo(x, edge_index_geo)
        combined = torch.cat([h_trans, h_geo], dim=-1)
        alpha = torch.sigmoid(self.fusion_gate(combined))
        return self.norm(self.dropout(alpha * h_trans + (1 - alpha) * h_geo))

class VenuePredictorGPT2_Advanced(nn.Module):
    def __init__(self, num_venues, num_cats, num_clusters, emb_dim=256, dropout=0.3):
        super().__init__()
        
        # 1. Embeddings
        self.venue_emb = nn.Embedding(num_venues, emb_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(num_cats, CAT_EMB_DIM, padding_idx=0)
        self.cluster_emb = nn.Embedding(num_clusters, CLUSTER_EMB_DIM, padding_idx=0) 
        
        self.time_emb = TimeEmbedding(emb_dim)
        self.interval_emb = IntervalEmbedding(emb_dim)
        
        # Input projection to mix features back to emb_dim
        input_dim_total = emb_dim + CAT_EMB_DIM + CLUSTER_EMB_DIM
        self.input_mix = nn.Linear(input_dim_total, emb_dim)
        
        # Cross Attention (Graph Fusion)
        # Note: num_heads must divide emb_dim evenly. 256 / 8 = 32 (Valid)
        self.cross_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=8, batch_first=True, dropout=dropout)
        self.norm_cross = nn.LayerNorm(emb_dim)
        
        # 2. GPT-2 Configuration (SCALED UP)
        config = GPT2Config(
            vocab_size=1, 
            n_embd=emb_dim,         # 256
            n_layer=2,              # 2 Layers
            n_head=4,               # 8 Heads
            n_inner=emb_dim * 4,    # 1024
            resid_pdrop=dropout, 
            embd_pdrop=dropout, 
            attn_pdrop=dropout, 
            max_position_embeddings=MAX_SEQ_LEN + 10
        )
        self.gpt2 = GPT2Model(config)
        
        # 3. Prediction Heads
        self.classifier = nn.Linear(emb_dim, num_venues)
        self.coord_regressor = nn.Linear(emb_dim, 2) 

    def forward(self, seq_ids, time_pair, dt, dd, graph_view, cat_lookup, cluster_lookup):
        B, L = seq_ids.shape
        flat_seq = seq_ids.view(-1)
        
        # Lookup Embeddings
        v_e = self.venue_emb(seq_ids)           
        c_e = self.cat_emb(cat_lookup[flat_seq].view(B, L))             
        cl_e = self.cluster_emb(cluster_lookup[flat_seq].view(B, L))    
        
        # Mix features
        base = self.input_mix(torch.cat([v_e, c_e, cl_e], dim=-1))
        
        # Add Spatio-Temporal context
        t_e = self.time_emb(time_pair)
        i_e = self.interval_emb(dt, dd)
        seq_rep = base + t_e + i_e
        
        # Cross Attention with Graph
        gv = graph_view[seq_ids]  
        # attn_out, _ = self.cross_attn(query=seq_rep, key=gv, value=gv)
        # gv = graph_view[seq_ids]

        attn_out, _ = self.cross_attn(
            query=seq_rep,
            key=gv,
            value=gv,
            key_padding_mask=(seq_ids == 0),
            need_weights=False
        )
        fused_input = self.norm_cross(seq_rep + attn_out)
        
        # Transformer Pass
        mask = (seq_ids != 0).long()
        outputs = self.gpt2(inputs_embeds=fused_input, attention_mask=mask)
        last_hidden = outputs.last_hidden_state[:, -1, :] 
        
        return self.classifier(last_hidden), self.coord_regressor(last_hidden), last_hidden

# -----------------------
# WRAPPER FOR MULTI-GPU (Used in Phase 2)
# -----------------------
class JointModel(nn.Module):
    def __init__(self, encoder, predictor, g_feats, ei_trans, ei_geo, v_cat, v_clus):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.register_buffer('g_feats', g_feats)
        self.register_buffer('ei_trans', ei_trans)
        self.register_buffer('ei_geo', ei_geo)
        self.register_buffer('v_cat', v_cat)
        self.register_buffer('v_clus', v_clus)

    def forward(self, seq, time_p, dt, dd):
        gf = self.encoder(self.g_feats, self.ei_trans, self.ei_geo)
        return self.predictor(seq, time_p, dt, dd, gf, self.v_cat, self.v_clus)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
    def forward(self, seq_rep, target_rep):
        logits = torch.matmul(seq_rep, target_rep.transpose(0, 1)) / self.temperature
        return self.ce(logits, torch.arange(logits.size(0), device=logits.device))

# -----------------------
# Training Loop
# -----------------------

# Helper for Phase 1
def get_negative_samples(edge_index, num_nodes, num_neg):
    neg_u = torch.randint(0, num_nodes, (num_neg,), device=DEVICE)
    neg_v = torch.randint(0, num_nodes, (num_neg,), device=DEVICE)
    return torch.stack([neg_u, neg_v], dim=0)

# -----------------------
# 5. FIVE-SEED EXPERIMENT RUNNER
# -----------------------
SEEDS = [42]
seed_results = []

for RUN_SEED in SEEDS:
    print("\\n" + "=" * 90)
    print(f"STARTING RUN FOR SEED {RUN_SEED}")
    print("=" * 90)
    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RUN_SEED)
    # Keep graph construction/preprocessing unchanged; reset stochastic training/model init per run.
    for emb_dim in EMB_SIZES:
        print(f"\n>>> EXPERIMENT EMB_DIM: {emb_dim} (Phase 1: GNN Pre-train -> Phase 2: Multi-GPU)")
        
        # 1. Init Individual Models on GPU 0
        #   
        graph_encoder = DualGraphEncoder_GAT(in_dim=graph_input_feats.shape[1], hidden=emb_dim).to(DEVICE)
        predictor = VenuePredictorGPT2_Advanced(n_venues, n_cats, N_CLUSTERS+1, emb_dim).to(DEVICE)
        
        # ---------------------------------------------------------
        # >>> PHASE 1: GNN PRE-TRAINING (Single GPU) <<<
        # ---------------------------------------------------------
        
        print(f"\n--- [PHASE 1] Pre-training GNN Embeddings on {DEVICE} ---")
        
        optimizer_gnn = torch.optim.Adam(graph_encoder.parameters(), lr=1e-3)
        bce_loss = nn.BCEWithLogitsLoss()
        PHASE1_EPOCHS = 400 # Reduced for demonstration speed
        
        graph_encoder.train()
        for p1_epoch in range(1, PHASE1_EPOCHS + 1):
            optimizer_gnn.zero_grad()
            
            # Move graph data to same device as encoder
            g_in = graph_input_feats.to(DEVICE)
            et = edge_index_trans.to(DEVICE)
            eg = edge_index_geo.to(DEVICE)
            
            node_embeds = graph_encoder(g_in, et, eg)
            
            # Link Prediction Task
            src_pos, dst_pos = et[0], et[1]
            pos_scores = (node_embeds[src_pos] * node_embeds[dst_pos]).sum(dim=-1)
            
            neg_edges = get_negative_samples(et, n_venues, num_neg=src_pos.size(0))
            src_neg, dst_neg = neg_edges[0], neg_edges[1]
            neg_scores = (node_embeds[src_neg] * node_embeds[dst_neg]).sum(dim=-1)
            
            preds = torch.cat([pos_scores, neg_scores])
            targets = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            
            loss_p1 = bce_loss(preds, targets)
            loss_p1.backward()
            optimizer_gnn.step()
            
            if p1_epoch % 2 == 0:
                print(f"   Phase 1 Epoch {p1_epoch}/{PHASE1_EPOCHS} | Link Pred Loss: {loss_p1.item():.4f}")

        print(">> Phase 1 Complete. Transferring Weights...")

        # ---------------------------------------------------------
        # >>> TRANSFER WEIGHTS
        # ---------------------------------------------------------
        with torch.no_grad():
            graph_encoder.eval()
            gf = graph_encoder(graph_input_feats.to(DEVICE), edge_index_trans.to(DEVICE), edge_index_geo.to(DEVICE))
            num_init = min(predictor.venue_emb.num_embeddings, gf.size(0))
            predictor.venue_emb.weight[:num_init].copy_(gf[:num_init])
            print(">> Weights Transferred.")

        # ---------------------------------------------------------
        # >>> PREPARE PHASE 2: WRAP & PARALLELIZE
        # ---------------------------------------------------------
        
        model = JointModel(
            graph_encoder, 
            predictor, 
            graph_input_feats, 
            edge_index_trans, 
            edge_index_geo, 
            venue_cat_tensor, 
            venue_cluster_tensor
        )

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(">> JointModel wrapped in DataParallel for Phase 2.")
        
        model = model.to(DEVICE)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        
        crit_cls = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)
        crit_cl = ContrastiveLoss(CL_TEMP)
        crit_reg = nn.MSELoss()

        best_mrr = 0.0; patience = 0

        # ---------------------------------------------------------
        # >>> PHASE 2: STANDARD TRAINING (Multi-GPU) <<<
        # ---------------------------------------------------------
        for epoch in range(1, EPOCHS+1):
            
            # --- TRAIN ---
            model.train()
            total_loss = 0.0; ns = 0
            acc_sums = np.zeros(len(KS)); mrr_sums = np.zeros(len(KS))
            
            start_time = time.time()
            pbar = tqdm(train_loader, desc=f"Ep {epoch} (Ph2)", mininterval=10.0) 
            
            for batch in pbar:
                seq, time_p, dt, dd, uid, tgt, coord_tgt = [x.to(DEVICE) for x in batch]
                optimizer.zero_grad()
                
                logits, pred_coords, seq_rep = model(seq, time_p, dt, dd)
                
                # Accessing module for contrastive target
                if isinstance(model, nn.DataParallel):
                    target_emb = model.module.predictor.venue_emb(tgt)
                else:
                    target_emb = model.predictor.venue_emb(tgt)

                loss = crit_cls(logits, tgt) + (LAMBDA_CL * crit_cl(seq_rep, target_emb)) + (LAMBDA_LOC * crit_reg(pred_coords, coord_tgt))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                bs = tgt.size(0); ns += bs; total_loss += loss.item() * bs
                
                with torch.no_grad():
                    acc_sums += np.array(accuracy_topk_batch(logits, tgt, KS)) * bs
                    mrr_sums += np.array(get_mrr_at_k_batch(logits, tgt, KS)) * bs

            elapsed = time.time() - start_time
            
            # Calculate Train Metrics
            train_accs = acc_sums / ns
            train_mrrs = mrr_sums / ns
            train_macc = np.mean(train_accs) # mAcc (mean of Acc@1, 5, 10, 20)
            
            print_metrics_row("TRAIN", total_loss/ns, train_accs, train_mrrs, train_macc, ns, elapsed)
            
            # --- VALIDATION ---
            model.eval()
            v_loss = 0.0; v_ns = 0
            v_acc_sums = np.zeros(len(KS)); v_mrr_sums = np.zeros(len(KS))
            
            with torch.no_grad():
                for batch in val_loader:
                    seq, time_p, dt, dd, uid, tgt, coord_tgt = [x.to(DEVICE) for x in batch]
                    logits, pred_coords, _ = model(seq, time_p, dt, dd)
                    
                    bs = tgt.size(0); v_ns += bs
                    v_loss += crit_cls(logits, tgt).item() * bs
                    v_acc_sums += np.array(accuracy_topk_batch(logits, tgt, KS)) * bs
                    v_mrr_sums += np.array(get_mrr_at_k_batch(logits, tgt, KS)) * bs
            
            val_accs = v_acc_sums / v_ns
            val_mrrs = v_mrr_sums / v_ns
            val_macc = np.mean(val_accs)
            
            print_metrics_row("VALID", v_loss/v_ns, val_accs, val_mrrs, val_macc, v_ns)

            # --- TEST ---
            t_loss = 0.0; t_ns = 0
            t_acc_sums = np.zeros(len(KS)); t_mrr_sums = np.zeros(len(KS))
            
            with torch.no_grad():
                for batch in test_loader:
                    seq, time_p, dt, dd, uid, tgt, coord_tgt = [x.to(DEVICE) for x in batch]
                    logits, pred_coords, _ = model(seq, time_p, dt, dd)
                    
                    loss = crit_cls(logits, tgt)
                    bs = tgt.size(0); t_ns += bs; t_loss += loss.item() * bs
                    
                    t_acc_sums += np.array(accuracy_topk_batch(logits, tgt, KS)) * bs
                    t_mrr_sums += np.array(get_mrr_at_k_batch(logits, tgt, KS)) * bs
            
            test_accs = t_acc_sums / max(1, t_ns)
            test_mrrs = t_mrr_sums / max(1, t_ns)
            test_macc = np.mean(test_accs)

            print_metrics_row("TEST ", t_loss/max(1, t_ns), test_accs, test_mrrs, test_macc, t_ns)

            # Save Best based on mAcc (or you can choose MRR@20)
            current_metric = val_macc
            if current_metric > best_mrr:
                best_mrr = current_metric
                patience = 0
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state, 'best_gpt2_model.pth')
                print(">> Saved Best (based on mAcc)")
            else:
                patience += 1
                if patience >= EARLY_STOP:
                    print("Early Stopping"); break

            scheduler.step(current_metric)

    print("Done.")

    # Record this seed's final TEST metrics (last evaluated epoch).
    seed_results.append({
        "seed": RUN_SEED,
        **{f"Acc@{k}": float(v) for k, v in zip(KS, test_accs)},
        **{f"MRR@{k}": float(v) for k, v in zip(KS, test_mrrs)},
        "Test_Loss": float(t_loss / max(1, t_ns)),
        "Test_mAcc": float(test_macc),
    })
    latest = seed_results[-1]
    print(f"\\nRUN SUMMARY | Seed {RUN_SEED} | Test Loss: {latest['Test_Loss']:.4f} | Test mAcc: {latest['Test_mAcc']:.4f}")
    print("Acc@K: " + ", ".join(f"Acc@{k}={latest[f'Acc@{k}']:.4f}" for k in KS))
    print("MRR@K: " + ", ".join(f"MRR@{k}={latest[f'MRR@{k}']:.4f}" for k in KS))

print("\\n" + "=" * 90)
print("FIVE-SEED SUMMARY (mean ± population standard deviation)")
print("=" * 90)
results_df = pd.DataFrame(seed_results).sort_values("seed").reset_index(drop=True)
print("\\nPer-seed results:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

metric_cols = [c for c in results_df.columns if c != "seed"]
summary_rows = []
for col in metric_cols:
    mean_val = results_df[col].mean()
    std_val = results_df[col].std(ddof=0)
    summary_rows.append({"Metric": col, "Mean": mean_val, "Std": std_val, "Mean ± Std": f"{mean_val * 100:.1f}±{std_val * 100:.1f}" if col.startswith(("Acc@", "MRR@", "Test_mAcc")) else f"{mean_val:.4f}±{std_val:.4f}"})
summary_df = pd.DataFrame(summary_rows)
print("\\nFinal table (ranking metrics shown as percentage points, e.g. 24.6±0.5):")
print(summary_df.to_string(index=False))
summary_df.to_csv("five_seed_summary.csv", index=False)
results_df.to_csv("five_seed_per_run_results.csv", index=False)
print("\\nSaved: five_seed_summary.csv and five_seed_per_run_results.csv")
