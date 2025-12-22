import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn.functional as F
import glob
import optuna
from optuna.pruners import MedianPruner

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

# ==============================
# 1. 데이터 로드 (스케일된 6자유도, 여러 파일)
# ==============================
DATA_DIR = "data_6dof_scaled"
file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print("발견된 파일 수:", len(file_paths))

if len(file_paths) == 0:
    raise RuntimeError("data_6dof_scaled 폴더에 csv가 없습니다.")

dof_cols = ["surge", "sway", "heave", "roll", "pitch", "yaw"]

all_list = []
for p in file_paths:
    df = pd.read_csv(p)
    all_list.append(df[dof_cols].values.astype(np.float32))

data_6dof = np.concatenate(all_list, axis=0)   # (T_total, 6)
print("전체 6자유도 데이터 shape:", data_6dof.shape)

# ==============================
# 2. 윈도우 개수 계산 + 인덱스 랜덤 Split
# ==============================
sampling_rate = 10        # Hz
IN_LEN  = 256             # 입력 시퀀스 길이
OUT_LEN = 256            # 출력 시퀀스 길이
STRIDE = 16
N_total   = len(data_6dof)
N_window  = N_total - IN_LEN - OUT_LEN + 1  # 만들 수 있는 전체 윈도우 수
print("전체 윈도우 개수:", N_window)

all_indices = np.random.permutation(N_window)

train_ratio = 0.7
val_ratio   = 0.15  # 나머지 0.15는 test

n_train = int(N_window * train_ratio)
n_val   = int(N_window * val_ratio)
n_test  = N_window - n_train - n_val

idx_train = all_indices[:n_train]
idx_val   = all_indices[n_train:n_train + n_val]
idx_test  = all_indices[n_train + n_val:]

print("\n랜덤 윈도우 분할 완료.")
print(f"Train windows: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

class TimeWindowDataset(Dataset):
    """
    data_6dof 전체에서
    시작 인덱스 리스트(indices)를 기준으로
    (IN_LEN, 6) 입력과 (OUT_LEN, 6) 타깃을 잘라내는 Dataset
    """
    def __init__(self, data: np.ndarray, indices: np.ndarray,
                 in_len: int = IN_LEN, out_len: int = OUT_LEN):
        self.data    = torch.from_numpy(data.astype(np.float32))  # (T_total, 6)
        self.indices = indices.astype(np.int64)
        self.in_len  = in_len
        self.out_len = out_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = int(self.indices[idx])
        e_in  = s + self.in_len
        e_out = e_in + self.out_len
        x = self.data[s:e_in, :]      # (IN_LEN, 6)
        y = self.data[e_in:e_out, :]  # (OUT_LEN, 6)
        return x, y


# ==============================
# 4. Flexible Time-CNN Encoder & Regressor
# ==============================
class FlexibleTimeCNN(nn.Module):
    """
    Conv1d + BN + ReLU 블록을 n_layers개 쌓는 Time-CNN 인코더
    각 레이어마다 다른 kernel_size를 사용할 수 있음.
    """
    def __init__(
        self,
        in_dof: int = 6,
        channels: int = 64,
        kernel_sizes=None,   # 리스트 (len = n_layers)
        use_bn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        if kernel_sizes is None or len(kernel_sizes) == 0:
            raise ValueError("kernel_sizes 리스트를 1개 이상 지정해야 합니다.")
        self.n_layers = len(kernel_sizes)

        # 홀수 커널만 허용 (same padding)
        for ks in kernel_sizes:
            assert ks % 2 == 1, "kernel_size는 same padding을 위해 홀수로 설정하세요."

        self.layers = nn.ModuleList()

        for i in range(self.n_layers):
            ks = kernel_sizes[i]
            padding = ks // 2
            in_ch = in_dof if i == 0 else channels

            conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=channels,
                kernel_size=ks,
                stride=1,
                padding=padding,
            )
            bn   = nn.BatchNorm1d(channels) if use_bn else nn.Identity()
            block = nn.Sequential(conv, bn, nn.ReLU())
            self.layers.append(block)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, in_dof)
        return: (B, T, channels)
        """
        x = x.permute(0, 2, 1)   # (B, in_dof, T)
        for block in self.layers:
            x = block(x)         # (B, channels, T)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)   # (B, T, channels)
        return x


class TimeCNNRegressor(nn.Module):
    """
    Time-CNN 기반 6자유도 seq2seq 회귀 모델
    입력: (B, IN_LEN, 6)
    출력: (B, OUT_LEN, 6)
    """
    def __init__(
        self,
        in_dof: int = 6,
        out_dof: int = 6,
        in_len: int = IN_LEN,
        out_len: int = OUT_LEN,
        cnn_channels: int = 64,
        kernel_sizes=None,        # 리스트
        conv_use_bn: bool = True,
        conv_dropout: float = 0.0,
    ):
        super().__init__()
        self.out_dof = out_dof
        self.out_len = out_len

        if kernel_sizes is None or len(kernel_sizes) == 0:
            raise ValueError("kernel_sizes 리스트를 1개 이상 지정해야 합니다.")

        self.cnn = FlexibleTimeCNN(
            in_dof=in_dof,
            channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            use_bn=conv_use_bn,
            dropout=conv_dropout,
        )

        # 시간축 Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, C, T) -> (B, C, 1)

        # FC로 OUT_LEN * out_dof로 매핑
        self.fc = nn.Sequential(
            nn.Flatten(),            # (B, C, 1) -> (B, C)
            nn.Linear(cnn_channels, 256),
            nn.ReLU(),
            nn.Linear(256, out_len * out_dof),
            nn.Sigmoid(),           # 0~1 스케일에 맞게
        )

    def forward(self, src, trg=None, teacher_forcing_ratio: float = 0.0):
        x = self.cnn(src)          # (B, T, C)
        x = x.permute(0, 2, 1)     # (B, C, T)
        x = self.global_pool(x)    # (B, C, 1)
        x = self.fc(x)             # (B, OUT_LEN * out_dof)
        x = x.view(-1, self.out_len, self.out_dof)
        return x

# ==============================
# 5. EarlyStopping & Train/Eval
# ==============================
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(model, loader, optimizer, criterion, tfr=0.0):
    model.train()
    epoch_loss = 0.0
    for src, trg in tqdm(loader, desc="Train(step)", leave=False):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=tfr)  # TimeCNN은 trg/tfr 무시
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0.0
    for src, trg in tqdm(loader, desc="Valid/Test(step)", leave=False):
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = criterion(output, trg)
        epoch_loss += loss.item()
    return epoch_loss / max(1, len(loader))

# ==============================
# 6. 1024스텝 입력 + 1024스텝 예측 (그래프)
# ==============================
def plot_input_output_seq(X_input, pred_seq, truth_seq, fs, title,
                          dof_names=None):
    X_input  = np.asarray(X_input)
    pred_seq = np.asarray(pred_seq)
    truth_seq= np.asarray(truth_seq)

    T_in  = X_input.shape[0]
    T_out = pred_seq.shape[0]

    t_input = np.arange(-T_in, 0) / fs
    t_pred  = np.arange(T_out) / fs

    if dof_names is None:
        dof_names = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    for d in range(6):
        ax = axes[d]
        ax.plot(t_input, X_input[:, d],
                label="Input", linestyle="--", alpha=0.7)
        ax.plot(t_pred,  truth_seq[:, d],
                label="Truth", linewidth=1.0)
        ax.plot(t_pred,  pred_seq[:, d],
                label="Pred",  linewidth=1.0)
        ax.axvline(0, color="k", linestyle=":", alpha=0.5)
        ax.set_ylabel(dof_names[d])
        ax.grid(True)
        if d == 0:
            ax.set_title(title)

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/ALLDOF_{title.replace(' ','_')}_seq.png")
    plt.close()

@torch.no_grad()
def predict_seq(model, X_start):
    model.eval()
    src = X_start.to(device)
    out = model(src, trg=None, teacher_forcing_ratio=0.0)   # (1, OUT_LEN, 6)
    return out.squeeze(0).cpu().numpy()

# ==============================
# 7. DataLoader & build_model
# ==============================
MAX_EPOCHS = 100
PATIENCE   = 10
LR         = 1e-3
TFR        = 0.0   # Time-CNN에서는 의미 없음

def build_loaders(batch_size):
    train_ds = TimeWindowDataset(data_6dof, idx_train, IN_LEN, OUT_LEN)
    val_ds   = TimeWindowDataset(data_6dof, idx_val,   IN_LEN, OUT_LEN)
    test_ds  = TimeWindowDataset(data_6dof, idx_test,  IN_LEN, OUT_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader   = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    test_loader  = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return train_loader, val_loader, test_loader


def build_model(model_type, n_layers, hidden_size, cnn_params=None):
    """
    model_type: 'TIME-CNN' 만 사용 (n_layers, hidden_size는 기록용)
    cnn_params: {
        "cnn_channels": int,
        "kernel_sizes": List[int],
        "conv_use_bn": bool,
        "conv_dropout": float,
    }
    """
    model_type = model_type.upper()
    if model_type != "TIME-CNN":
        raise ValueError(f"Only 'TIME-CNN' is supported in this script, got {model_type}")

    cnn_params = cnn_params or {}
    kernel_sizes = cnn_params.get("kernel_sizes", [9])

    model = TimeCNNRegressor(
        in_dof=6,
        out_dof=6,
        in_len=IN_LEN,
        out_len=OUT_LEN,
        cnn_channels=cnn_params.get("cnn_channels", 64),
        kernel_sizes=kernel_sizes,
        conv_use_bn=cnn_params.get("conv_use_bn", True),
        conv_dropout=cnn_params.get("conv_dropout", 0.0),
    ).to(device)
    return model

# ==============================
# 8. Optuna Objective (Time-CNN)
# ==============================
def objective(trial):
    # ---- 공통 하이퍼파라미터 (기록용 포함) ----
    model_type = trial.suggest_categorical("model_type", ["TIME-CNN"])
    n_layers   = trial.suggest_int("n_layers", 1, 1)          # 기록용
    hidden_size= trial.suggest_categorical("hidden_size", [64]) # 기록용
    batch_size = trial.suggest_categorical("batch_size", [64])

    # ---- Time-CNN 관련 하이퍼파라미터 ----
    max_layers = 4           # 나중에 5로 늘릴 수 있음
    n_cnn_layers = trial.suggest_int("n_cnn_layers", 4, max_layers)

    kernel_sizes_all = []
    for i in range(max_layers):
        # 레이어 인덱스별 kernel 후보 다르게 설정
        if i == 0:          # 1층: 가장 넓은 패턴
            candidates = [51]
        elif i == 1:        # 2층: 중간~넓은 패턴
            candidates = [41]
        elif i == 2:        # 3층: 중간 패턴
            candidates = [31]
        elif i == 3:        # 4층: 비교적 local
            candidates = [9]
        else:               # 5층: 가장 local
            candidates = [3, 5]

        ks = trial.suggest_categorical(
            f"kernel_size_l{i+1}",
            candidates
        )
        kernel_sizes_all.append(ks)

    kernel_sizes = kernel_sizes_all[:n_cnn_layers]

    cnn_channels = trial.suggest_categorical("cnn_channels", [64])
    cnn_dropout  = trial.suggest_float("cnn_dropout", 0.1, 0.1)  # 현재는 0 고정

    cnn_params = dict(
        cnn_channels = cnn_channels,
        kernel_sizes = kernel_sizes,
        conv_use_bn  = True,
        conv_dropout = cnn_dropout,
    )

    # ---- 시드 & 데이터로더 ----
    set_seed(42 + trial.number)
    train_loader, val_loader, _ = build_loaders(batch_size)

    # ---- 모델/옵티마이저 ----
    model = build_model(model_type, n_layers, hidden_size, cnn_params=cnn_params)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    ckpt_path = f"models/optuna_trial_{trial.number}.pt"
    es = EarlyStopping(patience=PATIENCE, path=ckpt_path)

    # ---- 학습 루프 ----
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss = train(model, train_loader, optimizer, criterion, tfr=TFR)
        va_loss = evaluate(model, val_loader, criterion)

        trial.report(va_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        es(va_loss, model)
        if es.early_stop:
            break

    return es.val_loss_min

# ==============================
# 9. Optuna Study 실행
# ==============================
N_TRIALS = 1
study = optuna.create_study(
    direction="minimize",
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)
study.optimize(objective, n_trials=N_TRIALS)

os.makedirs("results", exist_ok=True)
df_trials = study.trials_dataframe()
df_trials.to_csv("results/optuna_trials_6dof_timecnn.csv", index=False)

print("=== Optuna Best Trial ===")
print("Number:", study.best_trial.number)
print("Value (best val MSE):", study.best_trial.value)
print("Params:", study.best_trial.params)

# ==============================
# 10. Optuna Best Trial 불러와서 Test 평가 & 예측
# ==============================
csv_path = "results/optuna_trials_6dof_timecnn.csv"
df = pd.read_csv(csv_path)

df_complete = df[df["state"] == "COMPLETE"]
if len(df_complete) == 0:
    raise RuntimeError("CSV 안에 state == 'COMPLETE' 인 trial 이 없습니다. Optuna를 먼저 실행해 주세요.")

best_row = df_complete.loc[df_complete["value"].idxmin()]

best_trial_number = int(best_row["number"])
print("\n[Best Trial from CSV]")
print("  trial number :", best_trial_number)
print("  best val MSE:", best_row["value"])

# 기본 스칼라 파라미터들
best_params = {
    "model_type":      best_row["params_model_type"],
    "n_layers":        int(best_row["params_n_layers"]),
    "hidden_size":     int(best_row["params_hidden_size"]),
    "batch_size":      int(best_row["params_batch_size"]),
    "n_cnn_layers":    int(best_row["params_n_cnn_layers"]),
    "cnn_channels":    int(best_row["params_cnn_channels"]),
    "cnn_dropout":     float(best_row["params_cnn_dropout"]),
}

# 레이어별 kernel size 복원
kernel_sizes_all = []
for i in range(1, 5):  # max_layers = 1 기준
    col_name = f"params_kernel_size_l{i}"
    if col_name in df_complete.columns:
        ks = int(best_row[col_name])
        kernel_sizes_all.append(ks)

kernel_sizes_used = kernel_sizes_all[:best_params["n_cnn_layers"]]
best_params["kernel_sizes_used"] = kernel_sizes_used

print("\n[Best Params from CSV]")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print("  kernel_sizes_used:", kernel_sizes_used)

best_model_type = best_params["model_type"]
best_n_layers   = best_params["n_layers"]
best_hidden     = best_params["hidden_size"]
best_batch      = best_params["batch_size"]

best_cnn_params = dict(
    cnn_channels = best_params["cnn_channels"],
    kernel_sizes = kernel_sizes_used,
    n_cnn_layers = best_params["n_cnn_layers"],
    conv_use_bn  = True,
    conv_dropout = best_params["cnn_dropout"],
)

# DataLoader (batch_size 맞춰서)
_, _, test_loader = build_loaders(best_batch)

# 모델 생성 & 체크포인트 로드
best_model = build_model(
    best_model_type,
    best_n_layers,
    best_hidden,
    cnn_params=best_cnn_params,
)

best_ckpt_path = f"models/optuna_trial_{best_trial_number}.pt"
print(f"\n[Load Best Checkpoint] {best_ckpt_path}")

best_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
best_model.to(device)

criterion = nn.MSELoss()
test_mse = evaluate(best_model, test_loader, criterion)
print(f"\n[Best Trial Model] Test MSE: {test_mse:.6f}")

# 1024스텝 입력 + 1024스텝 예측 (그래프 저장)
if len(idx_test) >= 1:
    # 테스트용 Dataset 하나 다시 생성
    test_ds = TimeWindowDataset(data_6dof, idx_test, IN_LEN, OUT_LEN)

    idx = 0  # 테스트 윈도우 중 첫 번째
    X_seq, y_seq = test_ds[idx]      # X_seq: (IN_LEN, 6), y_seq: (OUT_LEN, 6)

    # 배치 차원 추가해서 모델에 넣기
    X_start = X_seq.unsqueeze(0)     # (1, IN_LEN, 6)
    pred_seq = predict_seq(best_model, X_start)  # (OUT_LEN, 6)

    X_input_seq = X_seq.cpu().numpy()
    truth_seq   = y_seq.cpu().numpy()

    ks_str = "-".join(map(str, kernel_sizes_used))  # [3] → "3", [15,9,5] → "15-9-5"

    model_name = (
        f"{best_model_type}_"
        f"B{best_batch}_C{best_cnn_params['cnn_channels']}_"
        f"L{best_cnn_params['n_cnn_layers']}_K{ks_str}"
    )

    plot_input_output_seq(
        X_input_seq,
        pred_seq,
        truth_seq,
        sampling_rate,
        f"{model_name}_6DOF_TIMECNN_OPTUNA"
    )
    print("입력 + 예측 시퀀스 (모든 자유도) 그래프 저장 완료.")
else:
    print("테스트 윈도우가 없습니다. 그래프를 그릴 수 없습니다.")
