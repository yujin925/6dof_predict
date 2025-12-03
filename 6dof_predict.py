import os, random, json
import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from joblib import dump, load
from tqdm.auto import tqdm, trange

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

N_DOF = 6  # 6자유도 입력/출력 차원
# --- 1. 데이터 로드 및 전처리 ---
# -----------------------------
# 1) 데이터 로드 (기존 형식 유지 + TITLE/VARIABLES 백업 파서)
# -----------------------------
def load_mot_data_6dof(file_path,
                       columns=('Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw')):
    """
    LIN_001_IRR_MOT.mot 형식 전용 로더
    - TITLE / VARIABLES 라인 뒤에 바로 데이터(숫자 7개: Time + 6 DOF)
    - 반환: (T, 6) numpy array
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 1) 혹시라도 endheader 형식인 파일도 대응 (지금 파일에는 없음)
    try:
        header_end_line = next(i for i, line in enumerate(lines)
                               if 'endheader' in line.lower())
        columns_all = lines[header_end_line + 1].strip().split()
        data_lines = lines[header_end_line + 2:]
        data_list = [list(map(float, ln.split()))
                     for ln in data_lines if ln.strip()]
        df = pd.DataFrame(data_list, columns=columns_all)

        for c in columns:
            if c not in df.columns:
                raise ValueError(f"'{c}' 컬럼 없음. 사용 가능한 컬럼: {df.columns.tolist()}")

        return df[list(columns)].values.astype(float)  # (T, 6)

    except StopIteration:
        # 2) TITLE / VARIABLES 포맷(지금 LIN_001_IRR_MOT.mot)이면 여기로 옴
        #   - VARIABLES 라인에서 컬럼 이름 추출
        var_line = next((ln for ln in lines
                         if ln.strip().upper().startswith("VARIABLES")), None)
        if var_line is None:
            raise ValueError("VARIABLES 라인을 찾을 수 없습니다.")

        # "Time", "Surge", ... "Yaw" 추출
        names = re.findall(r'"([^"]+)"', var_line)
        if not names:
            raise ValueError(f"VARIABLES 파싱 실패: {var_line}")

        # 숫자 라인만 모아서 float로 변환
        clean = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            # TITLE / VARIABLES 라인은 건너뜀
            up = s.upper()
            if up.startswith("TITLE") or up.startswith("VARIABLES"):
                continue
            try:
                floats = list(map(float, s.split()))
                clean.append(floats)
            except ValueError:
                pass

        if not clean:
            raise ValueError("데이터 숫자 라인을 찾을 수 없습니다.")

        arr = np.array(clean, dtype=float)   # (T, 7)  = Time + 6 DOF
        if arr.shape[1] != len(names):
            raise ValueError(f"컬럼 수 불일치: VARIABLES={len(names)}, 데이터={arr.shape[1]}")

        # 이름으로 인덱스 매핑
        name_to_idx = {name: i for i, name in enumerate(names)}  # Time=0, Surge=1, ...

        idx = [name_to_idx[c] for c in columns]   # ('Surge',...,'Yaw')
        return arr[:, idx].astype(float)          # (T, 6)

# main       
try:
    file_path = 'LIN_001_IRR_MOT.mot'
    data_6dof = load_mot_data_6dof(file_path)
    print("데이터 로드 완료:", data_6dof.shape)  # (T, 6)
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    print("예시용 데이터로 분석을 계속합니다.")
    T = 6000
    # 예시: 6자유도 가짜 데이터
    base = np.sin(np.linspace(0, 200, T))[:, None]
    noise = 0.4 * np.random.randn(T, 6)
    data_6dof = np.concatenate([base + noise[:, [i]] for i in range(6)], axis=1)  # (T, 6)

# Split
train_size = int(len(data_6dof) * 0.7)
val_size   = int(len(data_6dof) * 0.15)
train_data = data_6dof[:train_size]
val_data   = data_6dof[train_size:train_size + val_size]
test_data  = data_6dof[train_size + val_size:]

# MinMaxScaler는 자동으로 feature-wise로 동작하므로 그대로 사용 가능
scaler = MinMaxScaler().fit(train_data)
train_scaled = scaler.transform(train_data)
val_scaled   = scaler.transform(val_data)
test_scaled  = scaler.transform(test_data)
dump(scaler, 'models/scaler_6dof.joblib')


print("\n데이터 분할 및 표준화 완료.")
print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

# # -----------------------------
# # 3) 슬라이딩 윈도우 (seq2seq 10초→10초)
# # -----------------------------
# # ★ 샘플링레이트를 실제 데이터에 맞게 설정하세요 (예: 10Hz)
sampling_rate = 10
IN_LEN  = int(25.6 * sampling_rate)
OUT_LEN = int(25.6 * sampling_rate)
STRIDE  = 1

def make_windows(arr, in_len=IN_LEN, out_len=OUT_LEN, stride=STRIDE):
    """
    arr: (T, 1) numpy array (scaled)
    return: torch.Tensor X:(N,in_len,1), Y:(N,out_len,1)
    """
    X, Y = [], []
    N = len(arr)
    for s in range(0, N - in_len - out_len + 1, stride):
        X.append(arr[s:s+in_len])
        Y.append(arr[s+in_len:s+in_len+out_len])
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y)

X_train, y_train = make_windows(train_scaled)
X_val,   y_val   = make_windows(val_scaled)
X_test,  y_test  = make_windows(test_scaled)

print(f"윈도우 생성 완료 - X_train:{tuple(X_train.shape)}, y_train:{tuple(y_train.shape)}")

# -----------------------------
# Generic Encoder / Decoder for LSTM or GRU (uni/bi)
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, bidirectional, rnn_type="LSTM"):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        self.bidirectional = bidirectional
        RNN = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.rnn = RNN(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        # x: (B, Tin, 1)
        if self.rnn_type == "LSTM":
            outputs, (hidden, cell) = self.rnn(x)    # hidden/cell: (D*L, B, H)
            return hidden, cell
        else:  # GRU
            outputs, hidden = self.rnn(x)            # hidden: (D*L, B, H)
            cell = None
            return hidden, cell


class Decoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_layers, rnn_type="LSTM"):
        """
        feature_dim: 출력 차원 = 입력 차원 (여기서는 6자유도)
        """
        super().__init__()
        self.rnn_type = rnn_type.upper()
        RNN = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        # RNN input_size = feature_dim (6)
        self.rnn = RNN(feature_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim)
            # MinMax(0~1) 스케일링이면 Sigmoid 추가 가능, 아니면 빼도 됨
            # nn.Sigmoid()
        )

    def forward(self, input, hidden, cell):
        """
        input: (B, feature_dim) = (B, 6)
        hidden/cell: (L, B, H)
        """
        input = input.unsqueeze(1)  # (B, 1, 6)
        if self.rnn_type == "LSTM":
            output, (hidden, cell) = self.rnn(input, (hidden, cell))  # output: (B,1,H)
        else:
            output, hidden = self.rnn(input, hidden)
            cell = None
        prediction = self.fc_out(output.squeeze(1))  # (B, 6)
        return prediction, hidden, cell



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, rnn_type="LSTM"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.rnn_type = rnn_type.upper()

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
        D = 2 if self.encoder.bidirectional else 1
        B = hidden.size(1)

        if D == 2:  # bi-dir concat
            hidden = hidden.view(L, D, B, Henc)
            hidden = torch.cat((hidden[:,0], hidden[:,1]), dim=2)  # (L,B,2H)
            if self.rnn_type == "LSTM":
                cell = cell.view(L, D, B, Henc)
                cell = torch.cat((cell[:,0], cell[:,1]), dim=2)    # (L,B,2H)

        Hdec = dec.hidden_size
        if hidden.size(2) != Hdec:
            proj_h = nn.Linear(hidden.size(2), Hdec).to(self.device)
            hidden = proj_h(hidden)
            if self.rnn_type == "LSTM":
                proj_c = nn.Linear(cell.size(2),   Hdec).to(self.device)
                cell   = proj_c(cell)

        if self.rnn_type == "GRU":
            cell = None
        return hidden.contiguous(), (cell.contiguous() if cell is not None else None)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.0):
        """
        src: (B, Tin, 6)
        trg: (B, Tout, 6)
        """
        B, _, F = src.size()  # F = feature_dim = 6
        trg_len = trg.size(1) if trg is not None else OUT_LEN
        outputs = torch.zeros(B, trg_len, F, device=self.device)

        hidden, cell = self.encoder(src)
        hidden, cell = self._bridge(hidden, cell)

        # 첫 디코더 입력: 마지막 타임스텝 전체 6자유도
        input_t = src[:, -1, :]  # (B, 6)

        for t in range(trg_len):
            out_t, hidden, cell = self.decoder(input_t, hidden, cell)  # (B, 6)
            outputs[:, t, :] = out_t
            use_tf = (trg is not None) and (random.random() < teacher_forcing_ratio)
            input_t = trg[:, t, :] if use_tf else out_t

        return outputs

# --- 3. 조기 종료 클래스 ---
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pt'):  # ★ patience=10 고정
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
            if self.counter >= self.patience: self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- 4. 학습/평가 루프 및 시각화 함수 ---
def train(model, loader, optimizer, criterion, tfr=0.5):
    model.train()
    epoch_loss = 0.0
    # 배치 진행률 바
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
    for src, trg in tqdm(loader, desc="Valid(step)", leave=False):
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
    plt.savefig(f'results/{model_name}_history.png')
    plt.close()

def plot_100s_segment(truth, pred, fs, model_name):
    """truth/pred: (N,1), N=100*fs"""
    N = truth.shape[0]
    t = np.arange(N) / fs
    plt.figure(figsize=(10,4))
    plt.plot(t, truth, label='Truth')
    plt.plot(t, pred,  label='Pred')
    plt.title(f'{model_name} - 100s segment (scaled)')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (scaled)')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_100s.png')
    plt.close()

@torch.no_grad()
def predict_ar(model, X_start, fs, horizon_steps=None, horizon_s=None):
    """
    자가회귀 예측을 원하는 길이만큼 반복.
    - horizon_steps: 예측 스텝 수(샘플 수)
    - horizon_s: 예측 시간(초). 둘 다 None이면 기존 호환성 위해 100초.
    """
    model.eval()
    if horizon_steps is not None:
        need = int(horizon_steps)
    elif horizon_s is not None:
        need = int(horizon_s * fs)
    else:
        need = 100 * fs  # backward compatibility

    cur = X_start.clone().to(device)  # (1, IN_LEN, 1)
    preds = []
    for _ in range(need):
        dummy_trg = torch.zeros(1, 1, 1, device=device)
        out = model(cur, dummy_trg, teacher_forcing_ratio=0.0)  # (1,1,1)
        nxt = out[:, -1:, :]
        preds.append(nxt.squeeze().item())
        cur = torch.cat([cur[:, 1:, :], nxt], dim=1)
    return np.array(preds, dtype=np.float32)


def ar_truth_from_test_scaled(test_scaled, start_idx, in_len, fs, horizon_steps=None, horizon_s=None):
    """
    테스트 정답 구간을 예측 길이에 맞춰 잘라서 반환.
    - in_len: 입력 길이(스텝)
    - horizon_steps/horizon_s: 위와 동일(둘 다 None이면 100초)
    """
    if horizon_steps is not None:
        need = int(horizon_steps)
    elif horizon_s is not None:
        need = int(horizon_s * fs)
    else:
        need = 100 * fs  # backward compatibility

    s = start_idx + in_len
    e = s + need
    return test_scaled[s:e].squeeze()


def plot_ar_with_input(X_input, pred_seq, truth_seq, fs, title):
    """
    입력 구간(-len(X_input)/fs ~ 0s) + 예측/정답(0s ~ len(pred_seq)/fs)을 한 그래프에 표시
    """
    X_input  = np.asarray(X_input).squeeze()
    pred_seq = np.asarray(pred_seq).squeeze()
    truth_seq= np.asarray(truth_seq).squeeze()

    t_input = np.arange(-len(X_input), 0) / fs
    t_pred  = np.arange(len(pred_seq)) / fs

    plt.figure(figsize=(12,5))
    plt.plot(t_input, X_input,    label=f"Input ({len(X_input)/fs:.2f}s)", linestyle="--", alpha=0.7)
    plt.plot(t_pred,  truth_seq,  label=f"Truth ({len(pred_seq)/fs:.2f}s)", linewidth=1.2)
    plt.plot(t_pred,  pred_seq,   label=f"Pred  ({len(pred_seq)/fs:.2f}s)", linewidth=1.2)
    plt.axvline(0, color="k", linestyle=":", alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (scaled)")
    plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{title.replace(' ','_')}.png", dpi=150)
    plt.close()

def build_loaders(batch_size):
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

def build_model(model_type, n_layers, hidden_size):
    """
    model_type: 'LSTM', 'Bi-LSTM', 'GRU', 'Bi-GRU'
    """
    model_type = model_type.upper()
    if model_type not in ["LSTM", "BI-LSTM", "GRU", "BI-GRU"]:
        raise ValueError(f"Unknown model_type: {model_type}")

    is_bi = ("BI" in model_type)
    rnn_core = "GRU" if "GRU" in model_type else "LSTM"

    input_dim   = N_DOF  # 6
    feature_dim = N_DOF  # 6

    # ----- 모델 생성 -----
    enc = Encoder(input_dim, hidden_size, n_layers,
                  bidirectional=is_bi, rnn_type=rnn_core)
    dec_hidden = hidden_size * (2 if is_bi else 1)
    dec = Decoder(feature_dim, dec_hidden, n_layers,
                  rnn_type=rnn_core)

    model = Seq2Seq(enc, dec, device, rnn_type=rnn_core)

    # ----- DataParallel 적용 (GPU 6,7 사용) -----
    device_ids = [6, 7]       # 병렬로 사용할 GPU 번호
    primary = "cuda:6"        # 메인 GPU

    if torch.cuda.is_available():
        # 모델을 먼저 GPU 메인으로 옮기기
        model = model.to(primary)

        # GPU가 2개 이상이면 DataParallel 적용
        if len(device_ids) > 1:
            print(f"Using DataParallel on GPUs {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)

    return model

# ===========================
# 🔧 Optuna: n_layers만 1~2, lr/tfr는 고정
# ===========================
import optuna
from optuna.pruners import MedianPruner

USE_GRAD_TRACKER_FOR_BEST = True
MAX_EPOCHS = 100
PATIENCE   = 10

# lr, tfr 고정값 (원래 스크립트의 LR=1e-3, train에서 tfr=0.5를 그대로 사용)
LR = 1e-3
DEFAULT_TFR = 0.5

def objective(trial: optuna.Trial):
    #  탐색 공간: model_type, n_layers(1~2), hidden_size, batch_size만
    model_type  = trial.suggest_categorical("model_type", ["Bi-GRU"])
    n_layers    = trial.suggest_int("n_layers", low=1, high=3)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    batch_size  = trial.suggest_categorical("batch_size", [16, 32, 48])

    #  고정 하이퍼파라미터
    lr  = LR
    tfr = DEFAULT_TFR

    set_seed(100 + trial.number)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False, drop_last=False)

    model = build_model(model_type, n_layers, hidden_size)
    opt   = optim.Adam(model.parameters(), lr=lr)
    crit  = nn.MSELoss()

    ckpt_path = f"models/{model_type}_L{n_layers}_H{hidden_size}_B{batch_size}.pt"
    trial.set_user_attr("ckpt_path", ckpt_path)
    es = EarlyStopping(patience=PATIENCE, path=ckpt_path)

    tr_hist, va_hist = [], []
    best_va = float("inf")

    for ep in range(1, MAX_EPOCHS + 1):
        tr_loss = train(model, train_loader, opt, crit, tfr=tfr)
        va_loss = evaluate(model, val_loader, crit)
        tr_hist.append(tr_loss); va_hist.append(va_loss)

        trial.report(va_loss, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        es(va_loss, model)
        if va_loss < best_va: best_va = va_loss
        if es.early_stop: break

    model_name = f"{model_type}_L{n_layers}_H{hidden_size}_B{batch_size}"
    try:
        json.dump({'train': tr_hist, 'val': va_hist}, open(f"results/{model_name}_history.json","w"))
        plot_history(tr_hist, va_hist, model_name)
    except Exception:
        pass

    return best_va

# study = optuna.create_study(direction="minimize", pruner=MedianPruner(n_warmup_steps=10))
# study.optimize(objective, n_trials=10, show_progress_bar=True)

# # === Optuna 결과 테이블 CSV 저장 ===
# os.makedirs("results", exist_ok=True)
# df = study.trials_dataframe(
#     attrs=("number","value","params","state","user_attrs","datetime_start","datetime_complete","duration")
# )
# df.to_csv("results/optuna_trials.csv", index=False)
# print("Trials CSV saved → results/optuna_trials.csv")

# # === 베스트 트라이얼 메타 JSON 저장 ===
# bp = study.best_trial.params
# best_ckpt = study.best_trial.user_attrs.get(
#     "ckpt_path",
#     f"models/OPTUNA_{bp['model_type']}_L{bp['n_layers']}_H{bp['hidden_size']}_B{bp['batch_size']}.pt"
# )
# best_meta = {"best_value": float(study.best_value), "best_params": bp, "best_ckpt": best_ckpt}
# with open("results/optuna_best.json", "w", encoding="utf-8") as f:
#     json.dump(best_meta, f, ensure_ascii=False, indent=2)
# print(f"Best checkpoint: {best_ckpt}")
# print("Best meta JSON saved → results/optuna_best.json")


# print("\n===== Optuna 결과 =====")
# print(f"Best value (val MSE): {study.best_value:.6f}")
# print("Best params:")
# for k, v in study.best_trial.params.items():
#     print(f"  - {k}: {v}")

# ===========================
# ✅ 베스트 모델 로드 & 테스트 & 25.6초(=IN_LEN) → 25.6초 예측, 6자유도 전체 플롯
# ===========================

# 1) IN/OUT 길이는 위에서 이미 정의한 값을 그대로 사용 (IN_LEN, OUT_LEN)

# === 저장된 Optuna 결과 불러오기 ===
with open("results/optuna_best.json", "r", encoding="utf-8") as f:
    best_meta = json.load(f)

bp = best_meta["best_params"]
best_ckpt = best_meta["best_ckpt"]

# 2) 베스트 파라미터로 모델 재구성 + 체크포인트 로드
best_model_type  = bp["model_type"]          # 예: 'Bi-GRU'
best_n_layers    = int(bp["n_layers"])
best_hidden_size = int(bp["hidden_size"])
best_batch_size  = int(bp["batch_size"])

# 베스트 로더 (평가 용)
test_loader = DataLoader(TensorDataset(X_test, y_test),
                         batch_size=best_batch_size, shuffle=False, drop_last=False)

# 네가 이미 정의해둔 build_model() 재사용
best_model = build_model(best_model_type, best_n_layers, best_hidden_size)
best_model.load_state_dict(torch.load(best_ckpt, map_location=device))
best_model.to(device)
best_model.eval()

crit = nn.MSELoss()

# 3) 테스트 MSE
with torch.no_grad():
    test_mse = evaluate(best_model, test_loader, crit)
print(f"[BEST] Test MSE: {test_mse:.6f}")

# 4) 25.6초 입력 -> 25.6초 예측 (한 번의 forward로 OUT_LEN만큼 예측, 6자유도 동시에)
@torch.no_grad()
def predict_1shot_next(model, X_start):
    """
    모델이 teacher forcing 없이 다음 OUT_LEN 시퀀스를 한 번에 출력.
    (우리 Seq2Seq의 forward는 trg=None이면 OUT_LEN 길이로 생성함)
    X_start: (1, IN_LEN, 6)
    return: (OUT_LEN, 6) numpy array (scaled)
    """
    # DataParallel로 감싸져 있으면 안쪽 모듈만 꺼내서 사용
    if isinstance(model, nn.DataParallel):
        base_model = model.module
    else:
        base_model = model

    base_model.eval()
    # trg=None, teacher_forcing_ratio=0.0 은 기본값이라 굳이 안 넘겨도 됨
    out = base_model(X_start.to(device))              # (1, OUT_LEN, 6)
    return out.squeeze(0).cpu().numpy()              # (OUT_LEN, 6)


# 테스트 윈도 하나 골라서 시각화 (원하면 idx 바꿔가면서 확인 가능)
idx = 0
X_start = X_test[idx:idx+1]     # (1, IN_LEN, 6)
Y_truth = y_test[idx:idx+1]     # (1, OUT_LEN, 6)

pred_all  = predict_1shot_next(best_model, X_start)        # (OUT_LEN, 6)
truth_all = Y_truth.squeeze(0).cpu().numpy()               # (OUT_LEN, 6)
X_all     = X_start.squeeze(0).cpu().numpy()               # (IN_LEN, 6)

# 시간 축 (입력: -25.6s ~ 0s, 출력: 0s ~ +25.6s)
t_input = np.arange(-X_all.shape[0], 0) / sampling_rate
t_pred  = np.arange(pred_all.shape[0]) / sampling_rate

# DOF 이름 (데이터 로딩 순서와 맞춰서)
DOF_NAMES = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']

plt.figure(figsize=(14, 10))

for d in range(N_DOF):
    plt.subplot(N_DOF, 1, d+1)

    # 입력 구간 (과거 25.6초)
    plt.plot(t_input, X_all[:, d], linestyle="--", alpha=0.7,
             label=f"Input ({DOF_NAMES[d]})")

    # 정답 / 예측 (미래 25.6초)
    plt.plot(t_pred, truth_all[:, d], linewidth=1.2,
             label=f"Truth ({DOF_NAMES[d]})")
    plt.plot(t_pred, pred_all[:, d], linewidth=1.2,
             label=f"Pred ({DOF_NAMES[d]})")

    if d == 0:
        plt.title(
            f"{best_model_type}_L{best_n_layers}_H{best_hidden_size}_B{best_batch_size} — "
            f"25.6s Input → 25.6s Prediction (6-DOF, scaled)"
        )

    plt.axvline(0, color="k", linestyle=":", alpha=0.4)
    plt.ylabel(DOF_NAMES[d])
    plt.grid(True)
    plt.legend(loc="upper right", fontsize=8)

plt.xlabel("Time (s)")
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/OPTUNA_BEST_25.6s_input_output_6DOF.png", dpi=150)
plt.close()

print("Saved: results/OPTUNA_BEST_25.6s_input_output_6DOF.png")

