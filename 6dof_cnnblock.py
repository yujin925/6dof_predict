import os, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from joblib import dump
from tqdm.auto import tqdm, trange
import torch.nn.functional as F
import glob
import optuna
from optuna.pruners import MedianPruner

device_ids = [6, 7]
primary_device = "cuda:6"

device = torch.device(primary_device if torch.cuda.is_available() else "cpu")
print("Using GPUs:", device_ids)

# 결과 저장을 위한 디렉토리 생성
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
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
IN_LEN  = 256            # 입력 시퀀스 길이
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
# 3. STRIDE 적용한 윈도우 시작 인덱스 생성 + Dataset/DataLoader 준비
# ==============================
# STRIDE 반영: 시작점 후보를 0, STRIDE, 2*STRIDE, ...
starts = np.arange(0, N_total - IN_LEN - OUT_LEN + 1, STRIDE, dtype=np.int64)
print("STRIDE 적용 윈도우 개수:", len(starts))

all_indices = np.random.permutation(len(starts))
n_train = int(len(starts) * train_ratio)
n_val   = int(len(starts) * val_ratio)
n_test  = len(starts) - n_train - n_val

idx_train = all_indices[:n_train]
idx_val   = all_indices[n_train:n_train + n_val]
idx_test  = all_indices[n_train + n_val:]

train_starts = starts[idx_train]
val_starts   = starts[idx_val]
test_starts  = starts[idx_test]

train_ds = TimeWindowDataset(data_6dof, train_starts, IN_LEN, OUT_LEN)
val_ds   = TimeWindowDataset(data_6dof, val_starts,   IN_LEN, OUT_LEN)
test_ds  = TimeWindowDataset(data_6dof, test_starts,  IN_LEN, OUT_LEN)

def build_loaders(batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

# plotting에서 쓰는 test_scaled도 정의(현재 data_6dof가 이미 scaled라고 가정)
test_scaled = data_6dof

# ==============================
# 4. Encoder / Decoder / Seq2Seq (멀티출력 지원)
# ==============================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, bidirectional, rnn_type="LSTM"):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        self.bidirectional = bidirectional
        RNN = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.rnn = RNN(input_dim, hidden_dim, n_layers,
                       batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        # x: (B, Tin, input_dim)
        if self.rnn_type == "LSTM":
            outputs, (hidden, cell) = self.rnn(x)    # hidden/cell: (D*L, B, H)
            return hidden, cell
        else:  # GRU
            outputs, hidden = self.rnn(x)            # hidden: (D*L, B, H)
            cell = None
            return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, rnn_type="LSTM"):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        self.output_dim = output_dim
        RNN = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.rnn = RNN(output_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()          # MinMax(0~1) 스케일에 맞게; 원하면 제거 가능
        )

    def forward(self, input, hidden, cell):
        # input: (B, output_dim) → (B,1,output_dim)
        input = input.unsqueeze(1)
        if self.rnn_type == "LSTM":
            output, (hidden, cell) = self.rnn(input, (hidden, cell))  # output: (B,1,H)
        else:
            output, hidden = self.rnn(input, hidden)
            cell = None
        prediction = self.fc_out(output.squeeze(1))  # (B, output_dim)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, rnn_type="LSTM", out_dim=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.rnn_type = rnn_type.upper()
        self.out_dim = out_dim

    def _bridge(self, hidden, cell):
        """
        hidden/cell: (D*L, B, Henc) -> (L, B, Hdec)
        Bi-RNN이면 방향차원 concat. hidden_size mismatch면 선형 프로젝션.
        GRU는 cell=None.
        """
        enc = self.encoder.rnn
        dec = self.decoder.rnn
        L = enc.num_layers
        Henc = enc.hidden_size
        D = 2 if getattr(self.encoder, "bidirectional", False) else 1
        B = hidden.size(1)

        if D == 2:  # bi-dir concat
            hidden_ = hidden.view(L, D, B, Henc)
            hidden = torch.cat((hidden_[:,0], hidden_[:,1]), dim=2)  # (L,B,2H)
            if self.rnn_type == "LSTM" and cell is not None:
                cell_ = cell.view(L, D, B, Henc)
                cell = torch.cat((cell_[:,0], cell_[:,1]), dim=2)    # (L,B,2H)

        Hdec = dec.hidden_size
        if hidden.size(2) != Hdec:
            proj_h = nn.Linear(hidden.size(2), Hdec).to(self.device)
            hidden = proj_h(hidden)
            if self.rnn_type == "LSTM" and cell is not None:
                proj_c = nn.Linear(cell.size(2),   Hdec).to(self.device)
                cell   = proj_c(cell)

        if self.rnn_type == "GRU":
            cell = None
        return hidden.contiguous(), (cell.contiguous() if cell is not None else None)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.0):
        """
        src: (B, Tin, in_dim=6)
        trg: (B, Tout, out_dim=6)
        반환: (B, Tout, out_dim=6)
        """
        B, _, _ = src.size()
        out_dim = self.decoder.output_dim

        trg_len = trg.size(1) if trg is not None else OUT_LEN
        outputs = torch.zeros(B, trg_len, out_dim, device=self.device)

        hidden, cell = self.encoder(src)
        hidden, cell = self._bridge(hidden, cell)

        # 첫 decoder 입력: 마지막 관측값 (B, out_dim)
        input_t = src[:, -1, :]  # (B, 6)

        for t in range(trg_len):
            out_t, hidden, cell = self.decoder(input_t, hidden, cell)  # (B, 6)
            outputs[:, t, :] = out_t
            use_tf = (trg is not None) and (random.random() < teacher_forcing_ratio)
            input_t = trg[:, t, :] if use_tf else out_t

        return outputs

# ==============================
# 4-0. Time-CNN 블럭 + CNN-Bi-GRU Encoder
# ==============================
class TimeCNNEncoder(nn.Module):
    """
    Time-CNN encoder 블럭 (10 Hz, 10초→10초 기준)

    입력  : (B, T_in=100, D=6)   # 6자유도 시계열
    출력  : (B, T_cnn=50, C_out=128)  # Bi-GRU로 들어가는 feature 시퀀스
    """
    def __init__(
        self,
        in_dof: int = 6,      # 입력 채널 수 (6자유도)
        c1: int = 64,         # 1번 conv 출력 채널 수
        c2: int = 128,        # 2/3번 conv 출력 채널 수
        k1: int = 15,         # conv1 커널 길이
        k2: int = 15,         # conv2 커널 길이
        k3: int = 9,          # conv3 커널 길이
        use_bn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        padding1 = k1 // 2    # same padding
        padding2 = k2 // 2
        padding3 = k3 // 2

        # --- Stage 1: Conv(6→64, k=15) + ReLU + MaxPool(2) ---
        self.conv1 = nn.Conv1d(
            in_channels=in_dof,
            out_channels=c1,
            kernel_size=k1,
            stride=1,
            padding=padding1,
        )
        self.bn1 = nn.BatchNorm1d(c1) if use_bn else nn.Identity()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- Stage 2: Conv(64→128, k=15) + ReLU ---
        self.conv2 = nn.Conv1d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k2,
            stride=1,
            padding=padding2,
        )
        self.bn2 = nn.BatchNorm1d(c2) if use_bn else nn.Identity()

        # --- Stage 3: Conv(128→128, k=9) + ReLU ---
        self.conv3 = nn.Conv1d(
            in_channels=c2,
            out_channels=c2,
            kernel_size=k3,
            stride=1,
            padding=padding3,
        )
        self.bn3 = nn.BatchNorm1d(c2) if use_bn else nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_in, in_dof)  # 예: (batch, 100, 6)
        return: (B, T_cnn, C_out)  # 예: (batch, 50, 128)
        """
        # (B, T, D) -> (B, D, T) : Conv1d 입력 형식
        x = x.permute(0, 2, 1)             # (B, 6, T)

        # Stage 1
        x = self.conv1(x)                  # (B, 64, 100)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)                  # (B, 64, 50)

        # Stage 2
        x = self.conv2(x)                  # (B, 128, 50)
        x = self.bn2(x)
        x = F.relu(x)

        # Stage 3
        x = self.conv3(x)                  # (B, 128, 50)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # (B, C, T) -> (B, T, C) : Bi-GRU 입력 형식
        x = x.permute(0, 2, 1)             # (B, 50, 128)
        return x


class CNNEncoderBiGRU(nn.Module):
    """
    Time-CNN + Bi-GRU Encoder
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers,
        cnn_c1: int = 64,
        cnn_c2: int = 128,
        k1: int = 15,
        k2: int = 15,
        k3: int = 9,
        conv_use_bn: bool = True,
        conv_dropout: float = 0.1,
    ):
        super().__init__()
        self.rnn_type = "GRU"
        self.bidirectional = True

        # Time-CNN 블럭
        self.cnn = TimeCNNEncoder(
            in_dof=input_dim,
            c1=cnn_c1,
            c2=cnn_c2,
            k1=k1,
            k2=k2,
            k3=k3,
            use_bn=conv_use_bn,
            dropout=conv_dropout,
        )

        # CNN 출력 feature 크기 = cnn_c2
        self.rnn = nn.GRU(
            input_size=cnn_c2,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        x_feat = self.cnn(x)       # (B, T_cnn, cnn_c2)
        outputs, hidden = self.rnn(x_feat)
        cell = None
        return hidden, cell


# ==============================
# 6. EarlyStopping & 학습/평가
# ==============================
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):
        self.patience, self.delta, self.path = patience, delta, path
        self.counter, self.best_score, self.early_stop = 0, None, False
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

def train(model, loader, optimizer, criterion, tfr=0.5):
    model.train()
    epoch_loss = 0.0
    for src, trg in tqdm(loader, desc="Train(step)", leave=False):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=tfr)
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

def plot_history(train_hist, val_hist, model_name):
    plt.figure(figsize=(10, 4))
    plt.plot(train_hist, label='Train Loss')
    plt.plot(val_hist, label='Validation Loss')
    plt.title(f'{model_name} - Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/PITCH_{model_name}_history.png')
    plt.close()

# ==============================
# 7. 예측(그래프)
# ==============================
def truth_10s_scaled_6dof(test_scaled, start_idx, in_len, fs):
    """
    test_scaled: (T, 6)
    start_idx  : 윈도우 시작 인덱스 (make_windows에서의 s)
    반환: (10*fs, 6)
    """
    need = int(25.6 * fs)
    s = start_idx + in_len
    e = s + need
    return test_scaled[s:e, :]   # (need, 6)

def plot_input_output_10s_all(X_input, pred_10s, truth_10s, fs, title,
                              dof_names=None):
    """
    X_input  : (IN_LEN, 6)   - 입력 10초 (모든 자유도, scaled)
    pred_10s : (10*fs, 6)    - 예측 10초 (모든 자유도, scaled)
    truth_10s: (10*fs, 6)    - 실제 10초 (모든 자유도, scaled)
    """
    X_input  = np.asarray(X_input)
    pred_10s = np.asarray(pred_10s)
    truth_10s= np.asarray(truth_10s)

    T_in  = X_input.shape[0]
    T_out = pred_10s.shape[0]

    t_input = np.arange(-T_in, 0) / fs    # -10s ~ 0s
    t_pred  = np.arange(T_out) / fs       # 0s ~ 10s

    if dof_names is None:
        dof_names = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for d in range(6):
        ax = axes[d]
        ax.plot(t_input, X_input[:, d],
                label="Input (last 25.6s)", linestyle="--", alpha=0.7)
        ax.plot(t_pred,  truth_10s[:, d],
                label="Truth (next 25.6s)", linewidth=1.0)
        ax.plot(t_pred,  pred_10s[:, d],
                label="Pred (next 25.6s)",  linewidth=1.0)
        ax.axvline(0, color="k", linestyle=":", alpha=0.5)
        ax.set_ylabel(dof_names[d])
        ax.grid(True)
        if d == 0:
            ax.set_title(title)

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/ALLDOF_{title.replace(' ','_')}_input+output.png")
    plt.close()


@torch.no_grad()
def predict_10s_direct(model, X_start):
    """
    X_start: (1, IN_LEN, 6) scaled
    return : (10*fs, 6) scaled
    """
    model.eval()
    src = X_start.to(device)
    out = model(src, trg=None, teacher_forcing_ratio=0.0)   # (1, OUT_LEN, 6)
    return out.squeeze(0).cpu().numpy()   # (OUT_LEN, 6)

# ==============================
# 8. 하이퍼파라미터 & 모델 빌더
# ==============================
MAX_EPOCHS = 100
PATIENCE   = 10
LR         = 1e-3
TFR        = 0.5

def build_model(model_type, n_layers, hidden_size, cnn_params=None):
    """
    model_type: 'LSTM', 'Bi-LSTM', 'GRU', 'Bi-GRU', 'CNN-Bi-GRU'
    cnn_params: CNNEncoderBiGRU 에 넘길 dict (Optuna에서 샘플링)
    """
    model_type = model_type.upper()

    input_dim  = 6
    output_dim = 6

    if model_type == "CNN-BI-GRU":
        cnn_params = cnn_params or {}
        enc = CNNEncoderBiGRU(
            input_dim=input_dim,
            hidden_dim=hidden_size,
            n_layers=n_layers,
            **cnn_params,   # <- 여기!
        ).to(device)
        dec_hidden = hidden_size * 2  # Bi-GRU라 2배
        dec = Decoder(output_dim, dec_hidden, n_layers, rnn_type="GRU").to(device)
        return Seq2Seq(enc, dec, device, rnn_type="GRU", out_dim=output_dim).to(device)



    # ==== 이하 기존 RNN 분기는 그대로 유지 ====
    if model_type not in ["LSTM", "BI-LSTM", "GRU", "BI-GRU"]:
        raise ValueError(f"Unknown model_type: {model_type}")

    is_bi = ("BI" in model_type)
    rnn_core = "GRU" if "GRU" in model_type else "LSTM"

    enc = Encoder(input_dim, hidden_size, n_layers, bidirectional=is_bi, rnn_type=rnn_core).to(device)
    dec_hidden = hidden_size * (2 if is_bi else 1)
    dec = Decoder(output_dim, dec_hidden, n_layers, rnn_type=rnn_core).to(device)
    return Seq2Seq(enc, dec, device, rnn_type=rnn_core, out_dim=output_dim).to(device)

def objective(trial):
    # ---- 공통 RNN 하이퍼파라미터 샘플링 ----
    model_type = trial.suggest_categorical("model_type", ["CNN-Bi-GRU"])
    n_layers   = trial.suggest_int("n_layers", low = 2, high = 2)
    hidden_size= trial.suggest_categorical("hidden_size", [128])
    batch_size = trial.suggest_categorical("batch_size", [32])
    

    # ---- CNN block 하이퍼파라미터 샘플링 ----
    cnn_c1 = trial.suggest_categorical("cnn_c1", [64])
    cnn_c2 = trial.suggest_categorical("cnn_c2", [64])
    k1     = trial.suggest_categorical("cnn_k1", [15])   # 첫 conv kernel
    k2     = trial.suggest_categorical("cnn_k2", [11])
    k3     = trial.suggest_categorical("cnn_k3", [7])
    cnn_dropout = trial.suggest_float("cnn_dropout", 0.1, 0.1)

    cnn_params = dict(
        cnn_c1=cnn_c1,
        cnn_c2=cnn_c2,
        k1=k1,
        k2=k2,
        k3=k3,
        conv_use_bn=True,
        conv_dropout=cnn_dropout,
    )

    # ---- 시드 & 데이터로더 ----
    set_seed(42 + trial.number)
    train_loader, val_loader, test_loader = build_loaders(batch_size)

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


# # ==============================
# # 10. Optuna Study 실행
# # ==============================
N_TRIALS = 1  # 하고 싶은 만큼 

study = optuna.create_study(
    direction="minimize",
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
)
study.optimize(objective, n_trials=N_TRIALS)

# 결과 저장
os.makedirs("results", exist_ok=True)
df_trials = study.trials_dataframe()
df_trials.to_csv("results/optuna_trials_6dof.csv", index=False)

print("=== Optuna Best Trial ===")
print("Number:", study.best_trial.number)
print("Value (best val MSE):", study.best_trial.value)
print("Params:", study.best_trial.params)

# ==============================
# 11. Optuna Best Trial 체크포인트 로드해서 그대로 평가 & 예측
# ==============================
# 1) Optuna 결과 CSV 읽기
csv_path = "results/optuna_trials_6dof.csv"
df = pd.read_csv(csv_path)

# 2) state == COMPLETE 인 trial 중에서 value(=val MSE)가 최소인 trial 찾기
df_complete = df[df["state"] == "COMPLETE"]
if len(df_complete) == 0:
    raise RuntimeError("CSV 안에 state == COMPLETE 인 trial 이 없습니다. Optuna를 먼저 실행해 주세요.")

best_row = df_complete.loc[df_complete["value"].idxmin()]

best_trial_number = int(best_row["number"])
print("\n[Best Trial from CSV]")
print("  trial number :", best_trial_number)
print("  best val MSE:", best_row["value"])

# 3) 하이퍼파라미터 복원 (trials_dataframe() 포맷 기준)
best_params = {
    "model_type":      best_row["params_model_type"],
    "n_layers":        int(best_row["params_n_layers"]),
    "hidden_size":     int(best_row["params_hidden_size"]),
    "batch_size":      int(best_row["params_batch_size"]),
    "cnn_c1":          int(best_row["params_cnn_c1"]),
    "cnn_c2":          int(best_row["params_cnn_c2"]),
    "cnn_k1":          int(best_row["params_cnn_k1"]),
    "cnn_k2":          int(best_row["params_cnn_k2"]),
    "cnn_k3":          int(best_row["params_cnn_k3"]),
    "cnn_dropout":     float(best_row["params_cnn_dropout"]),
}
print("\n[Best Params from CSV]")
for k, v in best_params.items():
    print(f"  {k}: {v}")

best_model_type = best_params["model_type"]
best_n_layers   = best_params["n_layers"]
best_hidden     = best_params["hidden_size"]
best_batch      = best_params["batch_size"]

best_cnn_params = dict(
    cnn_c1       = best_params["cnn_c1"],
    cnn_c2       = best_params["cnn_c2"],
    k1           = best_params["cnn_k1"],
    k2           = best_params["cnn_k2"],
    k3           = best_params["cnn_k3"],
    conv_use_bn  = True,
    conv_dropout = best_params["cnn_dropout"],
)

# 4) DataLoader (batch_size 맞춰주기)
_, _, test_loader = build_loaders(best_batch)

# 5) 모델 구조 생성
best_model = build_model(
    best_model_type,
    best_n_layers,
    best_hidden,
    cnn_params=best_cnn_params,
)

# 6) Optuna objective에서 저장했던 체크포인트 로드
best_ckpt_path = f"models/optuna_trial_{best_trial_number}.pt"
print(f"\n[Load Best Checkpoint] {best_ckpt_path}")

best_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
best_model.to(device)

criterion = nn.MSELoss()

# 7) Test MSE 계산
test_mse = evaluate(best_model, test_loader, criterion)
print(f"\n[Best Trial Model] Test MSE: {test_mse:.6f}")

# 8) 입력/예측/정답 그래프 저장
if len(test_ds) >= 1:
    idx = 0
    X_start, _ = test_ds[idx]                 # (IN_LEN, 6), (OUT_LEN, 6)
    X_start = X_start.unsqueeze(0)            # (1, IN_LEN, 6)

    pred_10s_all = predict_10s_direct(best_model, X_start)  # (OUT_LEN, 6)

    X_input_all = X_start.squeeze(0).cpu().numpy()

    # 여기 중요: start index는 "idx"가 아니라 test_starts[idx]
    start_idx = int(test_starts[idx])
    truth_10s_all = truth_10s_scaled_6dof(test_scaled, start_idx, IN_LEN, sampling_rate)

    model_name = f"{best_model_type}_L{best_n_layers}_H{best_hidden}_B{best_batch}_6DOF_OPTUNA"
    plot_input_output_10s_all(
        X_input_all,
        pred_10s_all,
        truth_10s_all,
        sampling_rate,
        f"{model_name}_ALL_DOF"
    )
