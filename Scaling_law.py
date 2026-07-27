import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ============================================================
# 0. Device & Seed (device_ids 및 BASE_SEED 복원 수정)
# ============================================================
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    device_ids = [0, 1] if gpu_count >= 2 else [0]
    device = torch.device("cuda:0")
else:
    gpu_count = 0
    device_ids = []
    device = torch.device("cpu")

print(f"Using device: {device}")
print(f"Visible GPU count: {gpu_count}")
print(f"DataParallel device_ids: {device_ids}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


BASE_SEED = 42
set_seed(BASE_SEED)

# ============================================================
# 1. 실험 설정 (1015개 조건, 100개 단위 증가)
# ============================================================
DATA_DIR = Path("csv_dataset")

SAMPLING_INTERVAL = 0.5
SAMPLING_RATE = 1.0 / SAMPLING_INTERVAL

IN_LEN = 256
OUT_LEN = 256
STRIDE = 16

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 1015개 중 Train ratio 70%는 약 710개 -> 100개 단위로 증가
REQUESTED_TRAIN_CONDITION_SIZES = [
    100,
    200,
    300,
    400,
    500,
    600,
    700,
]

NUM_REPEATS = 5
SAVE_TEST_PREDICTIONS = True

EXP_NAME = (
    f"CausalModernTCN_NLL_UQ_learning_curve_"
    f"in{IN_LEN}_out{OUT_LEN}_stride{STRIDE}"
)

EXP_DIR = Path("experiments") / EXP_NAME
RUNS_DIR = EXP_DIR / "runs"
RESULT_DIR = EXP_DIR / "results"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("Experiment directory:", EXP_DIR)
print("Sampling interval:", SAMPLING_INTERVAL, "sec")
print("Input duration:", IN_LEN * SAMPLING_INTERVAL, "sec")
print("Output duration:", OUT_LEN * SAMPLING_INTERVAL, "sec")

# ============================================================
# 2. 데이터 로드 (안정성 강화)
# ============================================================
file_paths = sorted([
    p for p in DATA_DIR.glob("*.csv")
    if not p.name.startswith("Wave_")
])

if len(file_paths) == 0:
    raise RuntimeError("newdata 폴더에서 CSV 파일을 찾지 못했습니다.")

print("전체 조건 파일 수:", len(file_paths))

sample_df = pd.read_csv(file_paths[0])
time_col = sample_df.columns[0]
dof_cols = list(sample_df.columns[1:7])

print("Time column:", time_col)
print("DOF columns:", dof_cols)

all_arrays = []
file_lengths = []
valid_file_paths = []

for path in tqdm(file_paths, desc="Loading CSVs"):
    try:
        df = pd.read_csv(path, usecols=dof_cols)
    except ValueError as e:
        raise RuntimeError(f"{path.name}에서 6자유도 컬럼을 읽을 수 없습니다.") from e

    arr = df.to_numpy(dtype=np.float32)

    if not np.isfinite(arr).all():
        raise RuntimeError(f"{path.name}에 NaN/Inf 값이 있습니다.")

    if len(arr) < IN_LEN + OUT_LEN:
        continue

    all_arrays.append(arr)
    file_lengths.append(len(arr))
    valid_file_paths.append(path)

file_paths = valid_file_paths
num_conditions = len(file_paths)
print("사용 가능한 조건 파일 수:", num_conditions)

# ============================================================
# 3. 조건 단위 Train / Validation / Test 분할
# ============================================================
condition_indices = np.arange(num_conditions)
split_rng = np.random.default_rng(BASE_SEED)
split_rng.shuffle(condition_indices)

num_test_conditions = max(1, int(round(num_conditions * TEST_RATIO)))
num_val_conditions = max(1, int(round(num_conditions * VAL_RATIO)))
num_train_conditions = num_conditions - num_val_conditions - num_test_conditions

test_condition_indices = np.sort(condition_indices[:num_test_conditions])
val_condition_indices = np.sort(
    condition_indices[num_test_conditions : num_test_conditions + num_val_conditions]
)
train_pool_indices = np.sort(
    condition_indices[num_test_conditions + num_val_conditions :]
)

print("Train condition pool:", len(train_pool_indices))
print("Validation conditions:", len(val_condition_indices))
print("Test conditions:", len(test_condition_indices))

train_condition_sizes = sorted(set([
    n for n in REQUESTED_TRAIN_CONDITION_SIZES
    if 1 <= n <= len(train_pool_indices)
]))

if len(train_pool_indices) not in train_condition_sizes:
    train_condition_sizes.append(len(train_pool_indices))

print("실험할 Train condition sizes:", train_condition_sizes)

# ============================================================
# 4. 조건 분할 정보 및 Window 생성
# ============================================================
def make_window_meta(condition_indices):
    meta = []
    for file_idx in condition_indices:
        n = len(all_arrays[file_idx])
        max_start = n - IN_LEN - OUT_LEN
        for start_idx in range(0, max_start + 1, STRIDE):
            meta.append((file_idx, start_idx))
    return np.asarray(meta, dtype=np.int64) if meta else np.empty((0, 2), dtype=np.int64)

fixed_val_meta = make_window_meta(val_condition_indices)
fixed_test_meta = make_window_meta(test_condition_indices)

# ============================================================
# 5. Scaling 함수
# ============================================================
def calculate_train_scaler(selected_train_indices):
    train_arrays = [all_arrays[idx] for idx in selected_train_indices]
    train_concat = np.concatenate(train_arrays, axis=0)
    data_mean = train_concat.mean(axis=0).astype(np.float32)
    data_std = train_concat.std(axis=0).astype(np.float32)
    data_std[data_std == 0] = 1.0
    return data_mean, data_std

def standard_scale(x, mean, std):
    return (x - mean) / std

def standard_inverse(x_scaled, mean, std):
    return x_scaled * std + mean

# ============================================================
# 6. Dataset
# ============================================================
class TimeWindowDataset(Dataset):
    def __init__(self, raw_arrays, meta, data_mean, data_std, in_len=IN_LEN, out_len=OUT_LEN):
        self.raw_arrays = raw_arrays
        self.meta = np.asarray(meta, dtype=np.int64)
        self.data_mean = data_mean.astype(np.float32)
        self.data_std = data_std.astype(np.float32)
        self.in_len = in_len
        self.out_len = out_len

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        file_idx, start_idx = self.meta[idx]
        e_in = start_idx + self.in_len
        e_out = e_in + self.out_len
        raw_arr = self.raw_arrays[file_idx]

        x_scaled = standard_scale(raw_arr[start_idx:e_in], self.data_mean, self.data_std)
        y_scaled = standard_scale(raw_arr[e_in:e_out], self.data_mean, self.data_std)

        return (
            torch.from_numpy(x_scaled),
            torch.from_numpy(y_scaled),
            torch.tensor(file_idx, dtype=torch.long),
            torch.tensor(start_idx, dtype=torch.long),
        )

# ============================================================
# 7. Causal ModernTCN (Dual-Head NLL Output 수정)
# ============================================================
class PermuteToChannelsFirst(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class PermuteToTimeFirst(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=0, groups=groups, bias=bias
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)

class ModernTCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=31, expansion=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(channels * expansion)
        self.dwconv = CausalConv1d(channels, channels, kernel_size=kernel_size, groups=channels)
        self.norm = ChannelLayerNorm(channels)
        self.pwffn = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.pwffn(self.norm(self.dwconv(x)))

class ModernTCNBackbone(nn.Module):
    def __init__(self, in_dof=6, channels=128, num_blocks=6, kernel_size=31, expansion=2.0, dropout=0.1, stem_kernel_size=7):
        super().__init__()
        self.input_proj = nn.Sequential(
            PermuteToChannelsFirst(),
            CausalConv1d(in_dof, channels, kernel_size=stem_kernel_size),
        )
        self.blocks = nn.ModuleList([
            ModernTCNBlock(channels, kernel_size, expansion, dropout)
            for _ in range(num_blocks)
        ])
        self.out_permute = PermuteToTimeFirst()

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.out_permute(x)

class ModernTCNRegressor(nn.Module):
    def __init__(
        self, in_dof=6, out_dof=6, in_len=256, out_len=256,
        channels=128, num_blocks=6, kernel_size=31, expansion=2.0,
        dropout=0.1, stem_kernel_size=7, head_dropout=0.1,
    ):
        super().__init__()
        self.out_dof = out_dof
        self.backbone = ModernTCNBackbone(
            in_dof, channels, num_blocks, kernel_size, expansion, dropout, stem_kernel_size
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(in_len, out_len),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )
        # NLL을 위해 평균(mu) 6개 + 로그 분산(log_var) 6개 = 총 12개 채널 출력
        self.output_projection = nn.Linear(channels, out_dof * 2)

    def forward(self, src):
        x = self.backbone(src)          # (B, IN_LEN, C)
        x = x.transpose(1, 2)           # (B, C, IN_LEN)
        x = self.temporal_projection(x) # (B, C, OUT_LEN)
        x = x.transpose(1, 2)           # (B, OUT_LEN, C)
        x = self.output_projection(x)   # (B, OUT_LEN, DOF * 2)

        # 평균과 로그 분산으로 분리
        mu, log_var = torch.split(x, self.out_dof, dim=-1)
        # 수치적 안정성을 위한 로그 분산 클램핑 [-8.0, 8.0]
        log_var = torch.clamp(log_var, min=-8.0, max=8.0)
        return mu, log_var

# ============================================================
# 8. 모델 설정, 손실 함수, UQ Helper
# ============================================================
MODEL_CONFIG = {
    "model_type": "CausalModernTCN_NLL",
    "in_dof": 6, "out_dof": 6, "in_len": IN_LEN, "out_len": OUT_LEN,
    "channels": 128, "num_blocks": 6, "kernel_size": 31,
    "expansion": 2.0, "dropout": 0.1, "stem_kernel_size": 7, "head_dropout": 0.1,
}

TRAIN_CONFIG = {
    "batch_size": 64, "learning_rate": 1e-3, "max_epochs": 100,
    "patience": 10, "weight_decay": 0.0, "mc_samples": 20,
}

def create_model():
    return ModernTCNRegressor(**{k: v for k, v in MODEL_CONFIG.items() if k != "model_type"})

class HeteroscedasticGaussianNLLLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_mu, pred_log_var, target):
        var = torch.exp(pred_log_var) + self.eps
        loss = 0.5 * (pred_log_var + ((target - pred_mu) ** 2) / var)
        return loss.mean()

def enable_mc_dropout(model):
    """LayerNorm 등은 eval 모드로 유지하되, Dropout만 train 모드로 활성화"""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

# ============================================================
# 9. 학습 및 MC Dropout 평가 함수
# ============================================================
class EarlyStopping:
    def __init__(self, patience, checkpoint_path):
        self.patience = patience
        self.checkpoint_path = Path(checkpoint_path)
        self.counter = 0
        self.best_val_loss = float("inf")
        self.best_epoch = None
        self.early_stop = False

    def update(self, epoch, val_loss, save_checkpoint_func):
        if val_loss < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.counter = 0
            save_checkpoint_func(self.checkpoint_path, epoch, val_loss)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_samples = 0.0, 0
    for src, trg, _, _ in tqdm(loader, desc="Train", leave=False):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad(set_to_none=True)
        mu, log_var = model(src)
        loss = criterion(mu, log_var, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * src.size(0)
        total_samples += src.size(0)
    return total_loss / max(1, total_samples)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_samples = 0.0, 0
    for src, trg, _, _ in tqdm(loader, desc="Evaluate", leave=False):
        src, trg = src.to(device), trg.to(device)
        mu, log_var = model(src)
        loss = criterion(mu, log_var, trg)
        total_loss += loss.item() * src.size(0)
        total_samples += src.size(0)
    return total_loss / max(1, total_samples)

@torch.no_grad()
def evaluate_mc_dropout_uq(model, loader, data_mean, data_std, mc_samples=20):
    """MC Dropout K회 반복을 통한 AU, EU, TU 불확실성 분해 및 지표 산출"""
    enable_mc_dropout(model)

    mu_list_k, var_list_k = [], []
    y_true_list, x_list = [], []
    file_idx_list, start_idx_list = [], []

    # 1. K번 MC Dropout 샘플링
    for k in range(mc_samples):
        mu_k_batch, var_k_batch = [], []
        for batch_idx, (src, trg, file_idx, start_idx) in enumerate(loader):
            src_device = src.to(device)
            mu, log_var = model(src_device)
            var = torch.exp(log_var)

            mu_k_batch.append(mu.cpu().numpy().astype(np.float32))
            var_k_batch.append(var.cpu().numpy().astype(np.float32))

            if k == 0:
                y_true_list.append(trg.numpy().astype(np.float32))
                x_list.append(src.numpy().astype(np.float32))
                file_idx_list.append(file_idx.numpy().astype(np.int64))
                start_idx_list.append(start_idx.numpy().astype(np.int64))

        mu_list_k.append(np.concatenate(mu_k_batch, axis=0))
        var_list_k.append(np.concatenate(var_k_batch, axis=0))

    # (K, N_samples, OUT_LEN, DOF)
    mu_samples = np.stack(mu_list_k, axis=0)
    var_samples = np.stack(var_list_k, axis=0)
    y_true_scaled = np.concatenate(y_true_list, axis=0)

    # 2. Scaled Space에서 불확실성 계산
    mu_mean_scaled = np.mean(mu_samples, axis=0)
    au_scaled = np.mean(var_samples, axis=0) # 평균 분산
    eu_scaled = np.var(mu_samples, axis=0)   # 평균들의 분산
    tu_scaled = au_scaled + eu_scaled

    # 3. Raw Space로 역스케일링
    mean_3d = data_mean.reshape(1, 1, -1)
    std_3d = data_std.reshape(1, 1, -1)
    std_var_3d = (data_std ** 2).reshape(1, 1, -1) # 분산 변환은 표준편차의 제곱!

    y_true_raw = standard_inverse(y_true_scaled, mean_3d, std_3d)
    mu_mean_raw = standard_inverse(mu_mean_scaled, mean_3d, std_3d)
    au_raw = au_scaled * std_var_3d
    eu_raw = eu_scaled * std_var_3d
    tu_raw = tu_scaled * std_var_3d

    # 4. 정량적 오차 지표 (MAE, RMSE, R2) 계산
    eps = 1e-12
    error = mu_mean_raw - y_true_raw
    mae = np.mean(np.abs(error), axis=(0, 1))
    rmse = np.sqrt(np.mean(error ** 2, axis=(0, 1)))
    ss_res = np.sum(error ** 2, axis=(0, 1))
    ss_tot = np.sum((y_true_raw - np.mean(y_true_raw, axis=(0, 1), keepdims=True)) ** 2, axis=(0, 1))
    r2 = 1.0 - ss_res / (ss_tot + eps)

    # 5. 불확실성 평균 지표 산출 (시간 및 윈도우 평균)
    au_dof = np.mean(au_raw, axis=(0, 1))
    eu_dof = np.mean(eu_raw, axis=(0, 1))
    tu_dof = np.mean(tu_raw, axis=(0, 1))
    eu_au_ratio_dof = eu_dof / (au_dof + eps)

    metrics_df = pd.DataFrame({
        "DOF": dof_cols,
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "AU": au_dof, "EU": eu_dof, "TU": tu_dof,
        "EU_AU_ratio": eu_au_ratio_dof,
    })

    overall_df = pd.DataFrame({
        "DOF": ["overall_mean"],
        "MAE": [float(np.mean(mae))], "RMSE": [float(np.mean(rmse))], "R2": [float(np.mean(r2))],
        "AU": [float(np.mean(au_dof))], "EU": [float(np.mean(eu_dof))], "TU": [float(np.mean(tu_dof))],
        "EU_AU_ratio": [float(np.mean(eu_dof) / (np.mean(au_dof) + eps))],
    })

    return {
        "metrics_df": pd.concat([metrics_df, overall_df], ignore_index=True),
        "predictions": {
            "y_true_raw": y_true_raw, "y_pred_raw": mu_mean_raw,
            "au_raw": au_raw, "eu_raw": eu_raw, "tu_raw": tu_raw,
            "file_indices": np.concatenate(file_idx_list, axis=0),
            "start_indices": np.concatenate(start_idx_list, axis=0),
        }
    }

# ============================================================
# 10. 전체 실험 설정 저장
# ============================================================
with open(RESULT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "experiment_name": EXP_NAME, "data_dir": str(DATA_DIR),
        "train_condition_sizes": train_condition_sizes, "num_repeats": NUM_REPEATS,
        "model_config": MODEL_CONFIG, "train_config": TRAIN_CONFIG,
    }, f, indent=4, ensure_ascii=False)

np.savez_compressed(
    RESULT_DIR / "fixed_condition_split.npz",
    train_pool_indices=train_pool_indices,
    val_condition_indices=val_condition_indices,
    test_condition_indices=test_condition_indices,
)

# ============================================================
# 11. 데이터 규모별 반복 학습 및 UQ 평가
# ============================================================
all_run_summary_records = []
all_run_metric_records = []

for train_size in train_condition_sizes:
    print("\n" + "=" * 80)
    print(f"Train condition size: {train_size}")
    print("=" * 80)
    size_dir = RUNS_DIR / f"train_conditions_{train_size:04d}"
    size_dir.mkdir(parents=True, exist_ok=True)

    for repeat_idx in range(1, NUM_REPEATS + 1):
        run_seed = BASE_SEED + train_size * 100 + repeat_idx
        set_seed(run_seed)

        subset_rng = np.random.default_rng(run_seed)
        selected_train_indices = np.sort(
            subset_rng.choice(train_pool_indices, size=train_size, replace=False)
        )

        run_dir = size_dir / f"repeat_{repeat_idx:02d}"
        model_dir, result_dir = run_dir / "models", run_dir / "results"
        model_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "best_model.pt"

        print(f"\nTrain size={train_size}, Repeat={repeat_idx}/{NUM_REPEATS}, Seed={run_seed}")

        data_mean, data_std = calculate_train_scaler(selected_train_indices)
        train_meta = make_window_meta(selected_train_indices)
        val_meta, test_meta = fixed_val_meta.copy(), fixed_test_meta.copy()

        train_loader = DataLoader(TimeWindowDataset(all_arrays, train_meta, data_mean, data_std), batch_size=TRAIN_CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(TimeWindowDataset(all_arrays, val_meta, data_mean, data_std), batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)
        test_loader = DataLoader(TimeWindowDataset(all_arrays, test_meta, data_mean, data_std), batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)

        model = create_model()
        if len(device_ids) >= 2:
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"], weight_decay=TRAIN_CONFIG["weight_decay"])
        criterion = HeteroscedasticGaussianNLLLoss()
        early_stopping = EarlyStopping(patience=TRAIN_CONFIG["patience"], checkpoint_path=model_path)

        def save_checkpoint(path, epoch, val_loss):
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                "model_state_dict": state_dict, "best_epoch": int(epoch),
                "best_val_loss": float(val_loss), "data_mean": data_mean, "data_std": data_std,
            }, path)

        for epoch in range(1, TRAIN_CONFIG["max_epochs"] + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss = evaluate(model, val_loader, criterion)

            print(f"Epoch {epoch:03d} | Train NLL: {train_loss:.6f} | Val NLL: {val_loss:.6f}")
            early_stopping.update(epoch, val_loss, save_checkpoint)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # 안전 검사: 학습 발산으로 체크포인트가 생성되지 않은 경우 처리
        if not model_path.exists():
            raise RuntimeError(f"학습 실패: 최적 모델 체크포인트가 생성되지 않았습니다 -> {model_path}")

        # 최적 모델 로드 및 MC Dropout UQ 평가 (weights_only=False 명시)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        best_model = create_model()
        best_model.load_state_dict(checkpoint["model_state_dict"])
        if len(device_ids) >= 2:
            best_model = nn.DataParallel(best_model, device_ids=device_ids)
        best_model = best_model.to(device)

        uq_results = evaluate_mc_dropout_uq(
            best_model, test_loader, data_mean, data_std, mc_samples=TRAIN_CONFIG["mc_samples"]
        )
        metrics_df = uq_results["metrics_df"]
        metrics_df.insert(0, "train_condition_size", train_size)
        metrics_df.insert(1, "repeat", repeat_idx)
        metrics_df.to_csv(result_dir / "test_metrics_uq.csv", index=False)

        if SAVE_TEST_PREDICTIONS:
            preds = uq_results["predictions"]
            np.savez_compressed(
                result_dir / "test_predictions_uq.npz",
                y_true_raw=preds["y_true_raw"], y_pred_raw=preds["y_pred_raw"],
                au_raw=preds["au_raw"], eu_raw=preds["eu_raw"], tu_raw=preds["tu_raw"],
                file_indices=preds["file_indices"], start_indices=preds["start_indices"]
            )

        overall_m = metrics_df[metrics_df["DOF"] == "overall_mean"].iloc[0]
        run_summary = {
            "train_condition_size": int(train_size), "repeat": int(repeat_idx),
            "best_epoch": int(checkpoint["best_epoch"]), "best_val_nll": float(checkpoint["best_val_loss"]),
            "overall_rmse": float(overall_m["RMSE"]), "overall_r2": float(overall_m["R2"]),
            "overall_au": float(overall_m["AU"]), "overall_eu": float(overall_m["EU"]),
            "overall_tu": float(overall_m["TU"]), "eu_au_ratio": float(overall_m["EU_AU_ratio"]),
        }
        all_run_summary_records.append(run_summary)
        for _, row in metrics_df.iterrows():
            all_run_metric_records.append(row.to_dict())

        print(f"Completed: train_size={train_size}, repeat={repeat_idx} | RMSE={run_summary['overall_rmse']:.4f} | AU={run_summary['overall_au']:.4f} | EU={run_summary['overall_eu']:.4f} | EU/AU={run_summary['eu_au_ratio']:.4f}")

        del model, best_model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================================
# 12. 집계 및 기하평균(Geometric Mean) 산출
# ============================================================
all_runs_df = pd.DataFrame(all_run_summary_records)
all_runs_df.to_csv(RESULT_DIR / "all_runs_summary.csv", index=False)
pd.DataFrame(all_run_metric_records).to_csv(RESULT_DIR / "all_runs_metrics.csv", index=False)

def geometric_mean(x):
    x_pos = np.maximum(x, 1e-12)
    return float(np.exp(np.mean(np.log(x_pos))))

learning_curve_df = (
    all_runs_df.groupby("train_condition_size")
    .agg(
        num_repeats=("repeat", "count"),
        rmse_mean=("overall_rmse", "mean"),
        rmse_gmean=("overall_rmse", geometric_mean),
        au_mean=("overall_au", "mean"),
        au_gmean=("overall_au", geometric_mean),
        eu_mean=("overall_eu", "mean"),
        eu_gmean=("overall_eu", geometric_mean),
        tu_mean=("overall_tu", "mean"),
        tu_gmean=("overall_tu", geometric_mean),
        eu_au_ratio_mean=("eu_au_ratio", "mean"),
        eu_au_ratio_gmean=("eu_au_ratio", geometric_mean),
    )
    .reset_index()
    .sort_values("train_condition_size")
)

learning_curve_df.to_csv(RESULT_DIR / "learning_curve.csv", index=False)

print("\n" + "=" * 80)
print("All NLL UQ experiments completed successfully.")
print("=" * 80)
print(learning_curve_df[["train_condition_size", "rmse_gmean", "au_gmean", "eu_gmean", "eu_au_ratio_gmean"]])
