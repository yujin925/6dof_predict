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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import optuna

# ==============================
# 0. Device & Seed
# ==============================
device_ids = [6, 7]
primary_device = "cuda:6"
device = torch.device(primary_device if torch.cuda.is_available() else "cpu")
print("Using GPUs:", device_ids)

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
print(f"🎯 Target Features: 6-DOF (All)")

# ==============================
# 1. 역변환 (Inverse Transform) 6-DOF 셋업
# ==============================
SCALER_PATH = 'scaler_standard_6dof.json' 

def inverse_scale_data_6dof(scaled_array, scaler_path=SCALER_PATH):
    if not os.path.exists(scaler_path):
        print(f"⚠️ Scaler file not found at {scaler_path}. Returning scaled data.")
        return scaled_array
    
    with open(scaler_path, 'r', encoding='utf-8') as f:
        scaler_info = json.load(f)
    
    inversed_array = np.zeros_like(scaled_array)
    
    for i, feature_name in enumerate(dof_cols):
        mean_val = scaler_info['global_mean'][feature_name]
        std_val = scaler_info['global_std'][feature_name]
        
        if scaled_array.ndim == 3: 
            inversed_array[:, :, i] = (scaled_array[:, :, i] * std_val) + mean_val
        else: 
            inversed_array[:, i] = (scaled_array[:, i] * std_val) + mean_val
            
    return inversed_array

# ==============================
# 2. 데이터 로드 및 메타데이터 추출
# ==============================
DATA_DIR = "newdata_scaled"
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"발견된 파일 수: {len(file_paths)}")

if len(file_paths) == 0:
    raise RuntimeError(f"{DATA_DIR} 폴더에 csv 파일이 없습니다.")

all_list, file_lengths, file_metadata = [], [], []

for p in file_paths:
    df = pd.read_csv(p)
    df.columns = ["t", "surge", "sway", "heave", "roll", "pitch", "yaw"]
    all_list.append(df[dof_cols].values.astype(np.float32))
    file_lengths.append(len(df))
    
    filename = os.path.basename(p)
    match = re.search(r'(\d+)_Tm_([\d\.]+)_HS_([\d\.]+)_V_([\d\.]+)', filename)
    
    if match:
        direction = match.group(1)      
        period = float(match.group(2))  
        height = float(match.group(3))  
        velocity = float(match.group(4)) 
    else:
        direction, period, height, velocity = "Unknown", -1.0, -1.0, -1.0
        
    file_metadata.append({
        "wave_dir": direction, "wave_height": height, 
        "wave_period": period, "velocity": velocity
    })

data_6dof = np.concatenate(all_list, axis=0)   

# ==============================
# 3. 윈도우 계산 및 DataLoader 
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
        y = self.data[e_in:e_out, :]  
        return x, y

def build_loaders(batch_size):
    train_ds = TimeWindowDataset(data_6dof, idx_train)
    val_ds   = TimeWindowDataset(data_6dof, idx_val)
    test_ds  = TimeWindowDataset(data_6dof, idx_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) 
    return train_loader, val_loader, test_loader

# ==============================
# 4. Selective NLL BiLSTM 정의 (🌟 Sigma 출력 추가)
# ==============================
class SelectiveSingleShotRNN(nn.Module):
    def __init__(self, input_dim=6, output_dim=6, in_len=256, out_len=256, 
                 hidden_dim=256, n_layers=4, bidirectional=True):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.output_dim = output_dim
        
        # LSTM으로 고정
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
        
        D = 2 if bidirectional else 1
        shared_out_dim = hidden_dim * D
        
        # 🌟 NLL 예측을 위해 mu와 sigma 헤드로 분리
        self.head_mu_h = nn.Linear(shared_out_dim, out_len * output_dim)
        self.head_sigma_h = nn.Linear(shared_out_dim, out_len * output_dim)
        
        self.head_mu_a = nn.Linear(shared_out_dim, out_len * output_dim)
        self.head_sigma_a = nn.Linear(shared_out_dim, out_len * output_dim)
        
        self.head_g = nn.Sequential(
            nn.Linear(shared_out_dim, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        out, (h_n, c_n) = self.rnn(x)
        # h_n shape: [n_layers * num_directions, B, hidden_dim]
        # bidirectional이면 마지막 레이어의 forward, backward가 각각 h_n[-2], h_n[-1]
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        last_hidden = torch.cat([h_forward, h_backward], dim=1)  # [B, hidden_dim*2] 
        
        mu_h = self.head_mu_h(last_hidden).view(-1, self.out_len, self.output_dim)
        mu_a = self.head_mu_a(last_hidden).view(-1, self.out_len, self.output_dim)
        
        sigma_h = F.softplus(self.head_sigma_h(last_hidden)).view(-1, self.out_len, self.output_dim) + 1e-6
        sigma_a = F.softplus(self.head_sigma_a(last_hidden)).view(-1, self.out_len, self.output_dim) + 1e-6
        
        g = self.head_g(last_hidden) 
        
        return mu_h, sigma_h, mu_a, sigma_a, g

# ==============================
# 5. NLL Loss & Train/Eval Utilities
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
        nll_h_per_sample = nll_h.mean(dim=(1, 2)).unsqueeze(-1)
        
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
# 6. 역변환 시각화 유틸리티
# ==============================
def plot_input_output_seq_6dof(X_input, pred_mu, pred_sigma, truth_seq, fs, title, save_path, shared_ylims=None):
    # 역변환 적용
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
        
        # 각 피쳐별 역변환된 분산 경계 계산
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
        
        # 🌟 자유도별로 외부에서 지정된 공통 y축 범위 적용
        if shared_ylims is not None and len(shared_ylims) == 6:
            ax.set_ylim(shared_ylims[i])
        else:
            # 예외 방지용 자동 마진 계산
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
    for src, trg in tqdm(test_loader, desc="Testing", leave=False):
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
    
    sample_mse = np.mean((all_mu - all_y) ** 2, axis=(1, 2))
    return all_mu, all_sigma, all_g, all_y, sample_mse

# ==============================
# 7. Optuna Objective (Trial 학습 & 저장 루프)
# ==============================
def objective(trial):
    # 하이퍼파라미터 탐색 공간 (BiLSTM 맞춤형)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 2, 5)
    
    MAX_EPOCHS = 150
    PATIENCE = 10
    TARGET_COVERAGE = 0.85

    train_loader, val_loader, test_loader = build_loaders(batch_size)

    # 🌟 BiLSTM NLL 모델 초기화
    model = SelectiveSingleShotRNN(
        input_dim=6, output_dim=6, in_len=IN_LEN, out_len=OUT_LEN,
        hidden_dim=hidden_dim, n_layers=n_layers, bidirectional=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    criterion = SelectiveNLLLoss(target_coverage=TARGET_COVERAGE, alpha=0.5, lambda_c=32.0)
    
    model_save_path = f'models/best_model_trial_{trial.number}_bilstm.pt'
    es = EarlyStopping(patience=PATIENCE, path=model_save_path)

    print(f"\n" + "="*60)
    print(f"🚀 [Trial {trial.number}] 학습 시작")
    print(f"🔍 현재 하이퍼파라미터 조합:")
    for key, value in trial.params.items():
        # 소수점이 길어질 수 있는 실수형 값은 깔끔하게 포맷팅
        if isinstance(value, float):
            print(f"   - {key}: {value:.6f}")
        else:
            print(f"   - {key}: {value}")
    print("="*60)
    
    epoch_pbar = tqdm(range(1, MAX_EPOCHS + 1), desc=f"Trial {trial.number} Progress", leave=False)
    for epoch in epoch_pbar:
        tr_loss, tr_cov = train_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_cov, va_mse = evaluate_epoch(model, val_loader, criterion)
        
        epoch_pbar.set_postfix({"Tr_NLL": f"{tr_loss:.4f}", "Va_NLL": f"{va_loss:.4f}", "Cov": f"{va_cov:.2f}"})
        
        scheduler.step(va_loss)
        trial.report(va_loss, epoch)
        if trial.should_prune():
            epoch_pbar.close()
            raise optuna.exceptions.TrialPruned()
            
        es(va_loss, model)
        if es.early_stop:
            epoch_pbar.close()
            print(f"Trial {trial.number} EarlyStopping Triggered at Epoch {epoch} (Best Loss: {-es.best_score:.4f})")
            break

    # ==========================================
    # 8. 평가, 시각화 및 CSV 저장 (매 Trial 마다)
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

    # Worst (Rejected) 데이터 확보
    if len(rejected_indices) >= 1:
        target_idx_reject = rejected_indices[np.argsort(g_scores[rejected_indices])][0] 
        X_seq_r, y_seq_r = test_ds[target_idx_reject]
        X_start_r = X_seq_r.unsqueeze(0).to(device)
        with torch.no_grad(): pred_mu_r, pred_sigma_r, _, _, g_r = model(X_start_r)
        rej_data = (X_seq_r.cpu().numpy(), pred_mu_r.squeeze(0).cpu().numpy(), pred_sigma_r.squeeze(0).cpu().numpy(), y_seq_r.cpu().numpy(), g_r.item())

    # Best (Accepted) 데이터 확보
    if len(accepted_indices) >= 1:
        target_idx_accept = accepted_indices[np.argsort(-g_scores[accepted_indices])][0] 
        X_seq_a, y_seq_a = test_ds[target_idx_accept]
        X_start_a = X_seq_a.unsqueeze(0).to(device)
        with torch.no_grad(): pred_mu_a, pred_sigma_a, _, _, g_a = model(X_start_a)
        acc_data = (X_seq_a.cpu().numpy(), pred_mu_a.squeeze(0).cpu().numpy(), pred_sigma_a.squeeze(0).cpu().numpy(), y_seq_a.cpu().numpy(), g_a.item())

    # 🌟 [자유도별 개별 통일] Best와 Worst 전체를 통틀어 자유도마다 공통의 ylim 계산
    shared_ylims = [None] * 6
    if rej_data or acc_data:
        with open('scaler_standard_6dof.json', 'r', encoding='utf-8') as f:
            scaler_info = json.load(f)
            
        for i in range(6):
            col = dof_cols[i]
            std_val = scaler_info['global_std'][col]
            
            vals_min, vals_max = [], []
            
            # Worst 샘플 범위 분석에 추가
            if rej_data is not None:
                X_inv_r = inverse_scale_data_6dof(rej_data[0])
                mu_inv_r = inverse_scale_data_6dof(rej_data[1])
                y_inv_r = inverse_scale_data_6dof(rej_data[3])
                sigma_r = rej_data[2]
                lower_r = mu_inv_r[:, i] - sigma_r[:, i] * std_val
                upper_r = mu_inv_r[:, i] + sigma_r[:, i] * std_val
                
                vals_min.extend([np.min(X_inv_r[:, i]), np.min(y_inv_r[:, i]), np.min(lower_r)])
                vals_max.extend([np.max(X_inv_r[:, i]), np.max(y_inv_r[:, i]), np.max(upper_r)])
                
            # Best 샘플 범위 분석에 추가
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

    # 🌟 계산 완료된 자유도별 1:1 매칭 딕셔너리/리스트를 주입하여 플롯 생성
    if rej_data:
        save_name_rej = f"results/Trial_{trial.number}_REJECTED_Worst.png"
        title_reject = f"Trial {trial.number} REJECTED_Worst | Conf {rej_data[4]:.4f} | MSE {sample_mses[target_idx_reject]:.4f}"
        plot_input_output_seq_6dof(rej_data[0], rej_data[1], rej_data[2], rej_data[3], sampling_rate, title_reject, save_name_rej, shared_ylims)

    if acc_data:
        save_name_acc = f"results/Trial_{trial.number}_ACCEPTED_Best.png"
        title_accept = f"Trial {trial.number} ACCEPTED_Best | Conf {acc_data[4]:.4f} | MSE {sample_mses[target_idx_accept]:.4f}"
        plot_input_output_seq_6dof(acc_data[0], acc_data[1], acc_data[2], acc_data[3], sampling_rate, title_accept, save_name_acc, shared_ylims)
    
    # 전체 결과 CSV 저장
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
    csv_path = f"results/BiLSTM_Test_Results_Trial_{trial.number}_6DOF.csv"
    df_all_results.to_csv(csv_path, index=False)
    
    return es.best_score * -1.0 

# ==============================
# 9. Main 실행 블록
# ==============================
if __name__ == '__main__':
    print(f"\n=== [BiLSTM 6-DOF NLL & Optuna] 최적화 탐색 시작 ===")
    
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=5)
    
    print("\n✅ 전체 탐색이 완료되었습니다!")
    print("  Best Trial Number:", study.best_trial.number)
    print("  Best Validation Loss:", study.best_value)
    print("  Best Hyperparameters:", study.best_trial.params)
    
    with open("results/BiLSTM_Best_Hyperparameters.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    # 🌟 전체 스터디(Trial) 히스토리를 요약본 CSV로 저장
    df_study = study.trials_dataframe()
    df_study.to_csv("results/BiLSTM_Optuna_History.csv", index=False)
    print("📊 전체 Optuna 테스트 기록이 저장되었습니다: results/BiLSTM_Optuna_History.csv")    
