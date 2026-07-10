import os, random, re, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn.functional as F
import json
import optuna

# ==============================
# 0. Device & Seed
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

dof_cols = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
print(f"Target Features: 6-DOF (All)")

# ==============================
# 1. Inverse Transform 6-DOF Setup
# ==============================
SCALER_PATH = 'scaler_standard_6dof.json' 

def inverse_scale_data_6dof(scaled_array, scaler_path=SCALER_PATH):
    """
    Receives an array of shape [Batch, Time, 6] or [Time, 6] 
    and performs inverse transformation for all 6 features respectively.
    """
    if not os.path.exists(scaler_path):
        print(f"Scaler file not found at {scaler_path}. Returning scaled data.")
        return scaled_array
    
    with open(scaler_path, 'r', encoding='utf-8') as f:
        scaler_info = json.load(f)
    
    inversed_array = np.zeros_like(scaled_array)
    
    for i, feature_name in enumerate(dof_cols):
        mean_val = scaler_info['global_mean'][feature_name]
        std_val = scaler_info['global_std'][feature_name]
        
        if scaled_array.ndim == 3: # [Batch, Time, 6]
            inversed_array[:, :, i] = (scaled_array[:, :, i] * std_val) + mean_val
        else: # [Time, 6]
            inversed_array[:, i] = (scaled_array[:, i] * std_val) + mean_val
            
    return inversed_array

# ==============================
# 2. Data Load and Metadata Extraction
# ==============================
DATA_DIR = "newdata_scaled"
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

if len(file_paths) == 0:
    raise RuntimeError(f"No csv files found in the {DATA_DIR} folder.")

all_list, file_lengths, file_metadata = [], [], []

for p in file_paths:
    filename = os.path.basename(p)
    match = re.search(r'(\d+)_Tm_([\d\.]+)_HS_([\d\.]+)_V_([\d\.]+)', filename)
    if match:
        direction = match.group(1)      
        period = float(match.group(2))  
        height = float(match.group(3))  
        velocity = float(match.group(4)) 
    else:
        direction, period, height, velocity = "Unknown", -1.0, -1.0, -1.0
        
    df = pd.read_csv(p)
    df.columns = ["t", "surge", "sway", "heave", "roll", "pitch", "yaw"]
    all_list.append(df[dof_cols].values.astype(np.float32))
    file_lengths.append(len(df))
        
    file_metadata.append({
        "wave_dir": direction, "wave_height": height, 
        "wave_period": period, "velocity": velocity
    })

data_6dof = np.concatenate(all_list, axis=0)   

# ==============================
# 3. Window Calculation and DataLoader (6-DOF Target)
# ==============================
sampling_rate = 2        
IN_LEN  = 256              
OUT_LEN = 256              
STRIDE = 4              

valid_starts, valid_file_idx, current_offset = [], [], 0
for f_idx, L in enumerate(file_lengths):
    max_start = L - IN_LEN - OUT_LEN
    if max_start >= 0:
        starts = np.arange(0, max_start + 1, STRIDE) + current_offset
        valid_starts.extend(starts)
        valid_file_idx.extend([f_idx] * len(starts))
    current_offset += L

valid_starts = np.array(valid_starts)
valid_file_idx = np.array(valid_file_idx)
N_window = len(valid_starts)

permuted_positions = np.random.permutation(N_window)
n_train = int(N_window * 0.7)
n_val   = int(N_window * 0.15)
n_test  = N_window - n_train - n_val

pos_train = permuted_positions[:n_train]
pos_val   = permuted_positions[n_train:n_train + n_val]
pos_test  = permuted_positions[n_train + n_val:]

idx_train, idx_val, idx_test = valid_starts[pos_train], valid_starts[pos_val], valid_starts[pos_test]
test_file_indices = valid_file_idx[pos_test]

class TimeWindowDataset(Dataset):
    def __init__(self, data, indices, in_len=IN_LEN, out_len=OUT_LEN):
        self.data = torch.from_numpy(data.astype(np.float32))  
        self.indices = indices.astype(np.int64)
        self.in_len, self.out_len = in_len, out_len

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        s = int(self.indices[idx])
        e_in = s + self.in_len
        e_out = e_in + self.out_len
        x = self.data[s:e_in, :]
        #  Set target to all 6 features (6-DOF) instead of 1
        y = self.data[e_in:e_out, :]
        return x, y

def build_loaders(batch_size):
    train_ds = TimeWindowDataset(data_6dof, idx_train)
    val_ds   = TimeWindowDataset(data_6dof, idx_val)
    test_ds  = TimeWindowDataset(data_6dof, idx_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=8) 
    return train_loader, val_loader, test_loader

# ==============================
# 4. ModernTCN (Expanded Output Dimension)
# ==============================
class PermuteToChannelsFirst(nn.Module):
    def forward(self, x): return x.transpose(1, 2)

class PermuteToTimeFirst(nn.Module):
    def forward(self, x): return x.transpose(1, 2)

class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    def forward(self, x): return self.norm(x.transpose(1, 2)).transpose(1, 2)

class ModernTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=31, expansion=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(channels * expansion)
        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=True)
        self.norm = ChannelLayerNorm(channels)
        self.pwffn = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, channels, kernel_size=1), nn.Dropout(dropout),
        )
    def forward(self, x): return x + self.pwffn(self.norm(self.dwconv(x)))

class ModernTCNBackbone(nn.Module):
    def __init__(self, in_dof=6, channels=128, num_blocks=6, kernel_size=31, expansion=2.0, dropout=0.1, stem_kernel_size=7):
        super().__init__()
        self.input_proj = nn.Sequential(
            PermuteToChannelsFirst(),
            nn.Conv1d(in_dof, channels, kernel_size=stem_kernel_size, padding=stem_kernel_size//2),
        )
        self.blocks = nn.ModuleList([ModernTCNBlock(channels, kernel_size, expansion, dropout) for _ in range(num_blocks)])
        self.out_permute = PermuteToTimeFirst()
    def forward(self, x):
        x = self.input_proj(x)
        for b in self.blocks: x = b(x)
        return self.out_permute(x)

class SelectiveModernTCNRegressor(nn.Module):
    def __init__(self, in_dof=6, out_dof=6, in_len=256, out_len=256, channels=128, num_blocks=6, kernel_size=31, expansion=2.0, dropout=0.1, stem_kernel_size=7, head_hidden_dim=256):
        super().__init__()
        self.out_len, self.out_dof = out_len, out_dof
        self.backbone = ModernTCNBackbone(in_dof, channels, num_blocks, kernel_size, expansion, dropout, stem_kernel_size)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.shared_head = nn.Sequential(nn.Flatten(), nn.Linear(channels, head_hidden_dim), nn.GELU(), nn.Dropout(dropout))
        
        self.head_mu_h = nn.Linear(head_hidden_dim, out_len * out_dof)
        self.head_sigma_h = nn.Linear(head_hidden_dim, out_len * out_dof)
        
        self.head_mu_a = nn.Linear(head_hidden_dim, out_len * out_dof)
        self.head_sigma_a = nn.Linear(head_hidden_dim, out_len * out_dof)
        
        self.head_g = nn.Sequential(nn.Linear(head_hidden_dim, 1), nn.Sigmoid())

    def forward(self, src):
        x = self.global_pool(self.backbone(src).transpose(1, 2))
        features = self.shared_head(x)
        
        mu_h = self.head_mu_h(features).view(-1, self.out_len, self.out_dof)
        mu_a = self.head_mu_a(features).view(-1, self.out_len, self.out_dof)
        
        sigma_h = F.softplus(self.head_sigma_h(features)).view(-1, self.out_len, self.out_dof) + 1e-6
        sigma_a = F.softplus(self.head_sigma_a(features)).view(-1, self.out_len, self.out_dof) + 1e-6
        
        g = self.head_g(features) 
        return mu_h, sigma_h, mu_a, sigma_a, g

# ==============================
# 5. Loss & Utils (Multi-dimensional NLL Processing)
# ==============================
class SelectiveNLLLoss(nn.Module):
    def __init__(self, target_coverage=0.8, alpha=0.5, lambda_c=32.0):
        super().__init__()
        self.target_coverage = target_coverage
        self.alpha = alpha
        self.lambda_c = lambda_c

    def forward(self, mu_h, sigma_h, mu_a, sigma_a, g, y):
        var_h = sigma_h ** 2
        nll_h = 0.5 * torch.log(2 * np.pi * var_h) + ((y - mu_h) ** 2) / (2 * var_h)
        #  [Batch, Time, 6] -> Average over time and all features to derive window-level Risk
        nll_h_per_sample = nll_h.mean(dim=(1, 2)).unsqueeze(-1) # Shape: [Batch, 1]
        
        var_a = sigma_a ** 2
        nll_a = 0.5 * torch.log(2 * np.pi * var_a) + ((y - mu_a) ** 2) / (2 * var_a)
        loss_a_mean = nll_a.mean()
        
        emp_coverage = g.mean()
        selective_risk = (nll_h_per_sample * g).mean() / (emp_coverage + 1e-8)
        coverage_penalty = F.relu(self.target_coverage - emp_coverage) ** 2

        selective_loss = selective_risk + self.lambda_c * coverage_penalty
        total_loss = self.alpha * selective_loss + (1 - self.alpha) * loss_a_mean
        return total_loss, emp_coverage

class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='models/best_model.pt'):
        self.patience, self.delta, self.path = patience, delta, path
        self.counter, self.best_score, self.early_stop = 0, None, False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss, epoch_coverage = 0.0, 0.0
    for src, trg in tqdm(loader, desc="Train", leave=False):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        mu_h, sigma_h, mu_a, sigma_a, g = model(src)
        loss, coverage = criterion(mu_h, sigma_h, mu_a, sigma_a, g, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_coverage += coverage.item()
    return epoch_loss / len(loader), epoch_coverage / len(loader)

@torch.no_grad()
def evaluate_epoch(model, loader, criterion):
    model.eval()
    epoch_loss, epoch_coverage, epoch_mse_h = 0.0, 0.0, 0.0
    for src, trg in tqdm(loader, desc="Eval", leave=False):
        src, trg = src.to(device), trg.to(device)
        mu_h, sigma_h, mu_a, sigma_a, g = model(src)
        loss, coverage = criterion(mu_h, sigma_h, mu_a, sigma_a, g, trg)
        
        epoch_loss += loss.item()
        epoch_coverage += coverage.item()
        epoch_mse_h += F.mse_loss(mu_h, trg).item() 
    return epoch_loss / len(loader), epoch_coverage / len(loader), epoch_mse_h / len(loader)

# ==============================
# 6. Visualization & Analysis Utilities (6-DOF Subplots)
# ==============================
def plot_input_output_seq_6dof(X_input, pred_mu, pred_sigma, truth_seq, fs, title, save_path, shared_ylims=None):
    # Apply inverse transform
    X_input_inv = inverse_scale_data_6dof(X_input)
    pred_mu_inv = inverse_scale_data_6dof(pred_mu)
    truth_seq_inv = inverse_scale_data_6dof(truth_seq)
    
    with open('scaler_standard_6dof.json', 'r', encoding='utf-8') as f:
        scaler_info = json.load(f)

    t_input = np.arange(-X_input_inv.shape[0], 0) / fs
    t_pred  = np.arange(pred_mu_inv.shape[0]) / fs

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16)

    for i, col in enumerate(dof_cols):
        ax = axes[i // 2, i % 2]
        std_val = scaler_info['global_std'][col]
        
        # Calculate inverse-transformed variance boundaries for each feature
        pred_upper_inv = pred_mu_inv[:, i] + pred_sigma[:, i] * std_val
        pred_lower_inv = pred_mu_inv[:, i] - pred_sigma[:, i] * std_val   

        ax.plot(t_input, X_input_inv[:, i], label="Input (History)", linestyle="--", alpha=0.6)
        ax.plot(t_pred, truth_seq_inv[:, i], label="Truth", linewidth=1.5, color='green')
        ax.plot(t_pred, pred_mu_inv[:, i], label="Pred Mean", linewidth=1.5, color='red')
        ax.fill_between(t_pred, pred_lower_inv, pred_upper_inv, color='red', alpha=0.2, label="±1σ")
        
        ax.axvline(0, color="k", linestyle=":", alpha=0.5)
        ax.set_title(col.capitalize(), fontweight='bold')
        ax.set_ylabel("Original Unit")
        ax.grid(True)
        
        #  Apply externally specified common y-axis range for each DOF
        if shared_ylims is not None and len(shared_ylims) == 6:
            ax.set_ylim(shared_ylims[i])
        else:
            # Calculate automatic margin to prevent exceptions
            c_min = min(np.min(X_input_inv[:, i]), np.min(truth_seq_inv[:, i]), np.min(pred_lower_inv))
            c_max = max(np.max(X_input_inv[:, i]), np.max(truth_seq_inv[:, i]), np.max(pred_upper_inv))
            margin = (c_max - c_min) * 0.1
            if margin == 0: margin = 0.5
            ax.set_ylim(c_min - margin, c_max + margin)
        
        if i == 0:
            ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

@torch.no_grad()
def analyze_selective_results(model, test_loader):
    model.eval()
    all_mu, all_sigma, all_g, all_y = [], [], [], []
    for src, trg in test_loader:
        src, trg = src.to(device), trg.to(device)
        mu_h, sigma_h, _, _, g = model(src)
        all_mu.append(mu_h.cpu().numpy())
        all_sigma.append(sigma_h.cpu().numpy())
        all_g.append(g.cpu().numpy())
        all_y.append(trg.cpu().numpy())
            
    all_mu = np.concatenate(all_mu, axis=0)
    all_sigma = np.concatenate(all_sigma, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_g = np.concatenate(all_g, axis=0).squeeze() 
    
    # Since the dimension is [Batch, Time, 6], derive average MSE over both Time (1) and Feature (2)
    sample_mse = np.mean((all_mu - all_y) ** 2, axis=(1, 2))
    return all_mu, all_sigma, all_g, all_y, sample_mse

# ==============================
# 7. Optuna Objective (Trial Training & Saving Loop)
# ==============================
def objective(trial):
    # Define hyperparameter search space
    lr = 0.000112
    batch_size = 128

    channels = 64
    num_blocks = 3
    kernel_size = 511
    expansion = 3.15
    dropout = 0.04
    head_hidden_dim = 512

    MAX_EPOCHS = 300 # Adjustable based on resource availability
    PATIENCE = 10
    TARGET_COVERAGE = 0.85

    train_loader, val_loader, test_loader = build_loaders(batch_size)

    model = SelectiveModernTCNRegressor(
            in_dof=6, out_dof=6, in_len=IN_LEN, out_len=OUT_LEN,
            channels=channels, 
            num_blocks=num_blocks, 
            kernel_size=kernel_size,        
            expansion=expansion,            
            dropout=dropout,                
            stem_kernel_size=7,            
            head_hidden_dim=head_hidden_dim 
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = SelectiveNLLLoss(target_coverage=TARGET_COVERAGE, alpha=0.5, lambda_c=32.0)
    
    model_save_path = f'models/best_model_trial_{trial.number}_6dof.pt'
    es = EarlyStopping(patience=PATIENCE, path=model_save_path)

    print(f"\n" + "="*60)
    print(f" [Trial {trial.number}] Training started")
    print(f" Current hyperparameter combination:")
    for key, value in trial.params.items():
        # Cleanly format float values that might have long decimals
        if isinstance(value, float):
            print(f"   - {key}: {value:.6f}")
        else:
            print(f"   - {key}: {value}")
    print("="*60)
    
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_cov = train_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_cov, va_mse = evaluate_epoch(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Trial {trial.number}] Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"LR: {current_lr:.6f} | "
            f"Tr NLL Loss: {tr_loss:.4f} (Cov:{tr_cov:.2f}) | "
            f"Va NLL Loss: {va_loss:.4f} (Cov:{va_cov:.2f}, Va_MSE:{va_mse:.4f})")
        # Report the current epoch performance of the Trial to Optuna (for pruning purposes)
        trial.report(va_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        es(va_loss, model)
        if es.early_stop:
            print(f"Trial {trial.number} EarlyStopping Triggered at Epoch {epoch}")
            break

    # ==========================================
    # 8. Trial Evaluation, Result Graphs, and CSV Saving
    # ==========================================
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    mu_preds, sigma_preds, g_scores, y_truths, sample_mses = analyze_selective_results(model, test_loader)
    
    REJECT_TARGET_PERCENTAGE = 15.0
    REJECT_THRESHOLD = np.percentile(g_scores, REJECT_TARGET_PERCENTAGE)
    
    rejected_indices = np.where(g_scores <= REJECT_THRESHOLD)[0]
    accepted_indices = np.where(g_scores > REJECT_THRESHOLD)[0]

    test_ds = TimeWindowDataset(data_6dof, idx_test)

    rej_data, acc_data = None, None
    target_idx_reject, target_idx_accept = None, None

    # Acquire Worst (Rejected) data
    if len(rejected_indices) >= 1:
        target_idx_reject = rejected_indices[np.argsort(g_scores[rejected_indices])][0] 
        X_seq_r, y_seq_r = test_ds[target_idx_reject]
        X_start_r = X_seq_r.unsqueeze(0).to(device)
        with torch.no_grad(): pred_mu_r, pred_sigma_r, _, _, g_r = model(X_start_r)
        rej_data = (X_seq_r.cpu().numpy(), pred_mu_r.squeeze(0).cpu().numpy(), pred_sigma_r.squeeze(0).cpu().numpy(), y_seq_r.cpu().numpy(), g_r.item())

    # Acquire Best (Accepted) data
    if len(accepted_indices) >= 1:
        target_idx_accept = accepted_indices[np.argsort(-g_scores[accepted_indices])][0] 
        X_seq_a, y_seq_a = test_ds[target_idx_accept]
        X_start_a = X_seq_a.unsqueeze(0).to(device)
        with torch.no_grad(): pred_mu_a, pred_sigma_a, _, _, g_a = model(X_start_a)
        acc_data = (X_seq_a.cpu().numpy(), pred_mu_a.squeeze(0).cpu().numpy(), pred_sigma_a.squeeze(0).cpu().numpy(), y_seq_a.cpu().numpy(), g_a.item())

    #  [Individual unification per DOF] Calculate a common ylim for each DOF across both Best and Worst
    shared_ylims = [None] * 6
    if rej_data or acc_data:
        with open('scaler_standard_6dof.json', 'r', encoding='utf-8') as f:
            scaler_info = json.load(f)
            
        for i in range(6):
            col = dof_cols[i]
            std_val = scaler_info['global_std'][col]
            
            vals_min, vals_max = [], []
            
            # Add Worst sample to range analysis
            if rej_data is not None:
                X_inv_r = inverse_scale_data_6dof(rej_data[0])
                mu_inv_r = inverse_scale_data_6dof(rej_data[1])
                y_inv_r = inverse_scale_data_6dof(rej_data[3])
                sigma_r = rej_data[2]
                lower_r = mu_inv_r[:, i] - sigma_r[:, i] * std_val
                upper_r = mu_inv_r[:, i] + sigma_r[:, i] * std_val
                
                vals_min.extend([np.min(X_inv_r[:, i]), np.min(y_inv_r[:, i]), np.min(lower_r)])
                vals_max.extend([np.max(X_inv_r[:, i]), np.max(y_inv_r[:, i]), np.max(upper_r)])
                
            # Add Best sample to range analysis
            if acc_data is not None:
                X_inv_a = inverse_scale_data_6dof(acc_data[0])
                mu_inv_a = inverse_scale_data_6dof(acc_data[1])
                y_inv_a = inverse_scale_data_6dof(acc_data[3])
                sigma_a = acc_data[2]
                lower_a = mu_inv_a[:, i] - sigma_a[:, i] * std_val
                upper_a = mu_inv_a[:, i] + sigma_a[:, i] * std_val
                
                vals_min.extend([np.min(X_inv_a[:, i]), np.min(y_inv_a[:, i]), np.min(lower_a)])
                vals_max.extend([np.max(X_inv_a[:, i]), np.max(y_inv_a[:, i]), np.max(upper_a)])
                
            g_min = min(vals_min)
            g_max = max(vals_max)
            margin = (g_max - g_min) * 0.1
            if margin == 0: margin = 0.5
            shared_ylims[i] = (g_min - margin, g_max + margin)

    #  Generate plot by injecting the computed 1:1 matching list for each DOF
    if rej_data:
        save_name_rej = f"results/Trial_{trial.number}_REJECTED_Worst.png"
        title_reject = f"Trial {trial.number} REJECTED_Worst | Conf {rej_data[4]:.4f} | MSE {sample_mses[target_idx_reject]:.4f}"
        plot_input_output_seq_6dof(rej_data[0], rej_data[1], rej_data[2], rej_data[3], sampling_rate, title_reject, save_name_rej, shared_ylims)

    if acc_data:
        save_name_acc = f"results/Trial_{trial.number}_ACCEPTED_Best.png"
        title_accept = f"Trial {trial.number} ACCEPTED_Best | Conf {acc_data[4]:.4f} | MSE {sample_mses[target_idx_accept]:.4f}"
        plot_input_output_seq_6dof(acc_data[0], acc_data[1], acc_data[2], acc_data[3], sampling_rate, title_accept, save_name_acc, shared_ylims)
    
    # Extract full test data results to CSV
    all_test_info_list = []
    for test_idx in range(len(g_scores)):
        file_idx = test_file_indices[test_idx] 
        meta = file_metadata[file_idx]          
        
        info = {
            "Trial": trial.number,
            "Test_Index": test_idx,
            "File_Index": file_idx,
            "Confidence_Score": round(float(g_scores[test_idx]), 4),
            "MSE_Error_6DOF": round(float(sample_mses[test_idx]), 6),
            "Predicted_Sigma_Mean": round(float(np.mean(sigma_preds[test_idx])), 6),
            "Wave_Dir": meta["wave_dir"],
            "Wave_Period": meta["wave_period"],
            "Wave_Height": meta["wave_height"],
            "Velocity": meta["velocity"]
        }
        all_test_info_list.append(info)
        
    df_all_results = pd.DataFrame(all_test_info_list)
    csv_path = f"results/Test_Results_Trial_{trial.number}_6DOF.csv"
    df_all_results.to_csv(csv_path, index=False)
    
    # Optimization criterion to return to Optuna (Return the minimum Validation NLL Loss)
    return es.best_score * -1.0 # (Since score was -val_loss, convert back to positive)

# ==============================
# 9. Main Execution Block (Start Optuna Study)
# ==============================
if __name__ == '__main__':
    print(f"\n=== [6-DOF NLL Architecture & Optuna] Start Optimization Search ===")
    
    # Create Optuna study (Objective: minimize val_loss)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    
    # n_trials is the number of searches. (e.g., 20 searches)
    study.optimize(objective, n_trials=1)
    
    print("\n Optimization completed!")
    print("  Best Trial Number:", study.best_trial.number)
    print("  Best Validation Loss:", study.best_value)
    print("  Best Hyperparameters:", study.best_trial.params)
    
    # Save the final Best parameters to a file
    with open("results/Best_Hyperparameters.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    #  Save the entire study (Trial) history as a summary CSV
    df_study = study.trials_dataframe()
    df_study.to_csv("results/TCN_Optuna_History.csv", index=False)
    print(" Full Optuna test record saved: results/TCN_Optuna_History.csv")
