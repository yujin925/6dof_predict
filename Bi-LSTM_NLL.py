"""
BiLSTM Single-Shot 기반 6-DOF 시계열 예측 학습 스크립트 (NLL 버전)
- ID(2Hz 시뮬레이션) 데이터 로딩 → 조건 단위 train/val/test 분할
- z-score 정규화 (train 통계만 사용)
- Gaussian NLL 기반 학습 + 조기 종료(Early Stopping)
  (BiLSTM 인코더 → single-shot FC로 평균과 log 분산을 함께 예측)
- 학습 곡선 및 테스트 예측 시각화 (물리 단위 축)
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ============================================================
# 0. Device & Seed
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

BASE_SEED = 42
set_seed(BASE_SEED)

# ============================================================
# 1. Configuration & Hyperparameters
# ============================================================
ID_DATA_DIR = Path("csv_dataset")       # 2Hz 시뮬레이션 데이터 폴더

IN_LEN = 128
OUT_LEN = 128
STRIDE = 16
ID_HZ = 2.0
DT_SEC = 1.0 / ID_HZ                    # 시각화용 샘플 간격(초)

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

dof_cols = ["QTM-X_Filtered", "QTM-Y_Filtered", "QTM-Z_Filtered",
            "QTM-Roll_Filtered", "QTM-Pitch_Filtered", "QTM-Yaw_Filtered"]
dof_names = ["Surge (X)", "Sway (Y)", "Heave (Z)", "Roll", "Pitch", "Yaw"]
# QTM 원본 단위에 맞게 수정 (위치: mm 또는 m / 각도: deg 또는 rad)
dof_units = ["m", "m", "m", "deg", "deg", "deg"]

# log 분산 클램프 범위 (수치 안정성)
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0

# ============================================================
# 2. Data Loading & Preprocessing
# ============================================================
def load_id_data(data_dir):
    """ID 데이터(2Hz) 로드. (배열 리스트, 파일명 리스트) 반환"""
    file_paths = sorted([p for p in data_dir.glob("*.csv") if not p.name.startswith("Wave_")])
    if not file_paths:
        raise RuntimeError(f"{data_dir}에서 CSV 파일을 찾지 못했습니다.")

    all_arrays, kept_names = [], []
    for path in tqdm(file_paths, desc="Loading ID CSVs"):
        try:
            df = pd.read_csv(path, usecols=dof_cols)
        except ValueError:
            df = pd.read_csv(path)
            df = df.iloc[:, -6:]
        arr = df.to_numpy(dtype=np.float32)
        if len(arr) >= IN_LEN + OUT_LEN and np.isfinite(arr).all():
            all_arrays.append(arr)
            kept_names.append(path.name)
    print(f" - 유효 조건(파일) 수: {len(all_arrays)}")
    return all_arrays, kept_names

def make_window_meta(condition_indices, all_arrays):
    """조건 분할에 따른 슬라이딩 윈도우 인덱스 맵 생성"""
    meta = []
    for file_idx in condition_indices:
        n = len(all_arrays[file_idx])
        max_start = n - IN_LEN - OUT_LEN
        for start_idx in range(0, max_start + 1, STRIDE):
            meta.append((file_idx, start_idx))
    return np.asarray(meta, dtype=np.int64) if meta else np.empty((0, 2), dtype=np.int64)

class TimeWindowDataset(Dataset):
    """ID 데이터를 위한 PyTorch Dataset (Meta 기반 메모리 최적화)"""
    def __init__(self, raw_arrays, meta, data_mean, data_std):
        self.raw_arrays = raw_arrays
        self.meta = meta
        self.data_mean = data_mean
        self.data_std = data_std

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        file_idx, start_idx = self.meta[idx]
        e_in = start_idx + IN_LEN
        e_out = e_in + OUT_LEN
        raw_arr = self.raw_arrays[file_idx]
        x_scaled = (raw_arr[start_idx:e_in] - self.data_mean) / self.data_std
        y_scaled = (raw_arr[e_in:e_out] - self.data_mean) / self.data_std
        return torch.from_numpy(x_scaled).float(), torch.from_numpy(y_scaled).float()

# ============================================================
# 3. Model: BiLSTM Single-Shot 예측기 (NLL 헤드)
# ============================================================
class SingleShotRNN(nn.Module):
    """
    과거 IN_LEN 스텝(6-DOF)을 RNN 인코더로 요약한 뒤,
    FC 한 번(single-shot)으로 미래 OUT_LEN 스텝 전체의
    평균(mu)과 log 분산(log_var)을 예측한다.
    bidirectional=True일 때 forward/backward 각각의 '시퀀스 전체를 본'
    최종 은닉 상태를 h_n에서 올바르게 추출하여 결합한다.
    (out[:, -1, :] 사용 시 backward 절반은 마지막 1스텝만 본 상태라 부적절)
    """
    def __init__(self, input_dim=6, output_dim=6, in_len=128, out_len=128,
                 hidden_dim=256, n_layers=3, bidirectional=True,
                 rnn_type="LSTM", dropout=0.0):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.upper()
        RNN = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.rnn = RNN(input_dim, hidden_dim, n_layers,
                       batch_first=True, bidirectional=bidirectional,
                       dropout=dropout if n_layers > 1 else 0.0)
        D = 2 if bidirectional else 1
        # 마지막 은닉 상태 → 미래 전체의 mu + log_var를 한 번에 예측 (출력 2배)
        self.fc = nn.Linear(hidden_dim * D, out_len * output_dim * 2)

    def forward(self, x):
        # x: (B, IN_LEN, 6)
        if self.rnn_type == "LSTM":
            _, (h_n, _) = self.rnn(x)     # h_n: (n_layers * D, B, H)
        else:
            _, h_n = self.rnn(x)
        if self.bidirectional:
            # 마지막 레이어의 forward(-2) / backward(-1) 최종 상태 결합
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2H)
        else:
            last_hidden = h_n[-1]                                # (B, H)
        pred = self.fc(last_hidden)                              # (B, OUT_LEN*6*2)
        pred = pred.view(-1, self.out_len, self.output_dim * 2)  # (B, OUT_LEN, 12)
        mu, log_var = pred.chunk(2, dim=-1)                      # 각각 (B, OUT_LEN, 6)
        log_var = torch.clamp(log_var, LOGVAR_MIN, LOGVAR_MAX)
        return mu, log_var

# ============================================================
# 4. 모델 및 학습 설정
# ============================================================
MODEL_CONFIG = {
    "model_type": "BiLSTM_SingleShot_NLL",
    "hidden_dim": 256,
    "n_layers": 4,
    "bidirectional": True,
    "rnn_type": "LSTM",       # BiLSTM → 반드시 "LSTM" (기본값 GRU 주의)
    "dropout": 0.1,           # n_layers > 1 일 때 레이어 사이에 적용
}

TRAIN_CONFIG = {
    "batch_size": 64,
    "num_workers": 4,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "patience": 10,
}

EXP_NAME = (f"{MODEL_CONFIG['model_type']}_in{IN_LEN}_out{OUT_LEN}"
            f"_stride{STRIDE}_h{MODEL_CONFIG['hidden_dim']}"
            f"_L{MODEL_CONFIG['n_layers']}")
RESULT_DIR = Path("experiments0829") / EXP_NAME
MODEL_DIR = RESULT_DIR / "models"
PLOT_DIR = RESULT_DIR / "plots"
for d in (MODEL_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_DIR / "best_model.pt"
SCALER_SAVE_PATH = RESULT_DIR / "scaler.json"
HISTORY_SAVE_PATH = RESULT_DIR / "history.json"


def create_model():
    return SingleShotRNN(
        input_dim=6, output_dim=6,
        in_len=IN_LEN, out_len=OUT_LEN,
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        n_layers=MODEL_CONFIG["n_layers"],
        bidirectional=MODEL_CONFIG["bidirectional"],
        rnn_type=MODEL_CONFIG["rnn_type"],
        dropout=MODEL_CONFIG["dropout"],
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
# 5. 학습/평가 루프 (Gaussian NLL 기준)
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    for src, trg in tqdm(loader, desc="Train", leave=False):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        mu, log_var = model(src)
        var = torch.exp(log_var)
        loss = criterion(mu, trg, var)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion):
    model.eval()
    epoch_loss, epoch_mae = 0.0, 0.0
    for src, trg in tqdm(loader, desc="Eval", leave=False):
        src, trg = src.to(device), trg.to(device)
        mu, log_var = model(src)
        var = torch.exp(log_var)
        epoch_loss += criterion(mu, trg, var).item()
        epoch_mae += F.l1_loss(mu, trg).item()
    return epoch_loss / len(loader), epoch_mae / len(loader)

# ============================================================
# 6. 시각화 유틸리티
# ============================================================
def plot_training_curves(history, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train NLL")
    ax.plot(history["val_loss"], label="Val NLL")
    best_ep = int(np.argmin(history["val_loss"]))
    ax.axvline(best_ep, color="red", ls="--", lw=0.8,
               label=f"Best epoch ({best_ep + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gaussian NLL (z-score)")
    ax.set_title("Training / Validation Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f" 학습 곡선 저장: {save_path}")


def describe_window(dataset, idx, file_names):
    """윈도우 인덱스 → 원본 파일명 / 시작 위치 / 시간 구간 정보"""
    file_idx, start_idx = (int(v) for v in dataset.meta[idx])
    hist_start = start_idx
    pred_start = start_idx + IN_LEN
    pred_end = pred_start + OUT_LEN
    return {
        "window_index": int(idx),
        "source_file": file_names[file_idx],
        "file_index": file_idx,
        "input_sample_range": [hist_start, pred_start],
        "target_sample_range": [pred_start, pred_end],
        "input_time_sec": [round(hist_start * DT_SEC, 2), round(pred_start * DT_SEC, 2)],
        "target_time_sec": [round(pred_start * DT_SEC, 2), round(pred_end * DT_SEC, 2)],
    }


def plot_test_predictions(model, dataset, data_mean, data_std, n_samples, save_dir,
                          file_names):
    """테스트 윈도우 예측 시각화. x축=시간(초), y축=물리 단위."""
    model.eval()
    # 테스트 셋에서 균등 간격으로 샘플 선택
    n_total = len(dataset)
    if n_total == 0:
        print(" 테스트 윈도우가 없어 시각화를 건너뜁니다.")
        return
    sample_ids = np.linspace(0, n_total - 1, min(n_samples, n_total), dtype=int)

    t_in = np.arange(-IN_LEN, 0) * DT_SEC      # 과거 구간: 음수 시간
    t_out = np.arange(OUT_LEN) * DT_SEC        # 예측 구간: 0부터 시작

    records = []
    for s_idx in sample_ids:
        info = describe_window(dataset, s_idx, file_names)
        src, trg = dataset[s_idx]
        with torch.no_grad():
            mu, log_var = model(src.unsqueeze(0).to(device))
            pred = mu.squeeze(0).cpu().numpy()
            sigma = torch.exp(0.5 * log_var).squeeze(0).cpu().numpy()
        src = src.numpy()
        trg = trg.numpy()

        # z-score 역변환 → 물리 단위
        src_phys = src * data_std + data_mean
        trg_phys = trg * data_std + data_mean
        pred_phys = pred * data_std + data_mean
        sigma_phys = sigma * data_std

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        for i in range(6):
            ax = axes[i // 2, i % 2]
            ax.plot(t_in, src_phys[:, i], ls="--", alpha=0.6, color="gray",
                    label="Input (History)")
            ax.plot(t_out, trg_phys[:, i], lw=1.5, color="green", label="Truth")
            ax.plot(t_out, pred_phys[:, i], lw=1.5, color="red", label="Pred (mean)")
            ax.fill_between(t_out,
                            pred_phys[:, i] - 2 * sigma_phys[:, i],
                            pred_phys[:, i] + 2 * sigma_phys[:, i],
                            color="red", alpha=0.15, label="±2σ")
            ax.axvline(0, color="k", ls=":", alpha=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"{dof_names[i]} ({dof_units[i]})")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper left", fontsize=9)

        rmse = float(np.sqrt(np.mean((pred_phys - trg_phys) ** 2)))
        info["rmse_physical"] = round(rmse, 6)

        t0, t1 = info["target_time_sec"]
        fig.suptitle(
            f"{info['source_file']} | window #{s_idx} | "
            f"target t = {t0:.1f}–{t1:.1f}s (sample {info['target_sample_range'][0]}"
            f"–{info['target_sample_range'][1]}) | RMSE = {rmse:.4f}",
            fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        # 파일명에도 출처를 남겨 그림만 봐도 추적 가능
        stem = Path(info["source_file"]).stem
        safe_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
        save_path = save_dir / f"test_pred_w{s_idx}_{safe_stem}_s{info['target_sample_range'][0]}.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        info["plot_file"] = save_path.name
        records.append(info)
        print(f" 예측 시각화 저장: {save_path.name}")
        print(f"    출처: {info['source_file']} | "
              f"입력 {info['input_sample_range'][0]}–{info['input_sample_range'][1]} → "
              f"예측 {info['target_sample_range'][0]}–{info['target_sample_range'][1]} 샘플")

    # 시각화한 윈도우의 출처 정보를 파일로 저장
    meta_path = save_dir / "plotted_window_sources.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f" 윈도우 출처 정보 저장: {meta_path}")

# ============================================================
# 7. Main
# ============================================================
if __name__ == "__main__":
    # ---------- 데이터 준비 ----------
    print("데이터 로딩 중...")
    all_arrays, id_file_names = load_id_data(ID_DATA_DIR)
    num_conditions = len(all_arrays)

    # 조건(파일) 단위 분할 — seed 고정으로 재현 가능
    condition_indices = np.arange(num_conditions)
    np.random.shuffle(condition_indices)
    n_test = max(1, int(round(num_conditions * TEST_RATIO)))
    n_val = max(1, int(round(num_conditions * VAL_RATIO)))
    test_idx = condition_indices[:n_test]
    val_idx = condition_indices[n_test:n_test + n_val]
    train_idx = condition_indices[n_test + n_val:]
    print(f" - 분할: train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} 조건")

    # 정규화 통계는 train 조건만 사용 (데이터 누수 방지)
    train_concat = np.concatenate([all_arrays[i] for i in train_idx], axis=0)
    data_mean = train_concat.mean(axis=0).astype(np.float32)
    data_std = train_concat.std(axis=0).astype(np.float32)
    data_std[data_std == 0] = 1.0

    # 스케일러 저장 (추론/시각화 스크립트에서 재사용)
    with open(SCALER_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "dof_cols": dof_cols,
            "mean": data_mean.tolist(),
            "std": data_std.tolist(),
            "train_files": [id_file_names[i] for i in train_idx],
            "val_files": [id_file_names[i] for i in val_idx],
            "test_files": [id_file_names[i] for i in test_idx],
        }, f, indent=2, ensure_ascii=False)
    print(f" 스케일러/분할 정보 저장: {SCALER_SAVE_PATH}")

    # 윈도우 메타 및 데이터로더
    train_meta = make_window_meta(train_idx, all_arrays)
    val_meta = make_window_meta(val_idx, all_arrays)
    test_meta = make_window_meta(test_idx, all_arrays)
    print(f" - 윈도우: train {len(train_meta)} / val {len(val_meta)} / test {len(test_meta)}")

    # 전체 테스트 윈도우의 출처 매핑 저장 (윈도우 인덱스 → 파일/구간)
    test_map = pd.DataFrame({
        "window_index": np.arange(len(test_meta)),
        "source_file": [id_file_names[i] for i in test_meta[:, 0]],
        "input_start": test_meta[:, 1],
        "target_start": test_meta[:, 1] + IN_LEN,
        "target_end": test_meta[:, 1] + IN_LEN + OUT_LEN,
    })
    test_map["target_start_sec"] = test_map["target_start"] * DT_SEC
    test_map.to_csv(RESULT_DIR / "test_window_index.csv", index=False)
    print(f" 테스트 윈도우 매핑 저장: {RESULT_DIR / 'test_window_index.csv'}")

    train_ds = TimeWindowDataset(all_arrays, train_meta, data_mean, data_std)
    val_ds = TimeWindowDataset(all_arrays, val_meta, data_mean, data_std)
    test_ds = TimeWindowDataset(all_arrays, test_meta, data_mean, data_std)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"],
                              shuffle=True, num_workers=TRAIN_CONFIG["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CONFIG["batch_size"],
                            shuffle=False, num_workers=TRAIN_CONFIG["num_workers"],
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=TRAIN_CONFIG["batch_size"],
                             shuffle=False, num_workers=TRAIN_CONFIG["num_workers"],
                             pin_memory=True)

    # ---------- 모델/옵티마이저 ----------
    model = create_model().to(device)
    print(f"\n모델: {MODEL_CONFIG['model_type']} | 파라미터 수: {count_parameters(model):,}")

    criterion = nn.GaussianNLLLoss(full=False, eps=1e-6)
    optimizer = optim.AdamW(model.parameters(),
                            lr=TRAIN_CONFIG["learning_rate"],
                            weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    # ---------- 학습 루프 (조기 종료) ----------
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0

    print("\n 학습 시작")
    for epoch in range(1, TRAIN_CONFIG["max_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_mae = evaluate_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(cur_lr)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        marker = " (best 저장)" if improved else ""
        print(f"[Epoch {epoch:3d}/{TRAIN_CONFIG['max_epochs']}] "
              f"train NLL {train_loss:.5f} | val NLL {val_loss:.5f} | "
              f"val MAE {val_mae:.5f} | lr {cur_lr:.2e}{marker}")

        if patience_counter >= TRAIN_CONFIG["patience"]:
            print(f"조기 종료: {TRAIN_CONFIG['patience']} epoch 동안 val 개선 없음")
            break

    with open(HISTORY_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, PLOT_DIR / "training_curves.png")

    # ---------- 테스트 평가 ----------
    print("\n 테스트 평가 (best 가중치 로드)")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss, test_mae = evaluate_epoch(model, test_loader, criterion)
    print(f" - Test NLL (z-score): {test_loss:.5f}")
    print(f" - Test MAE (z-score): {test_mae:.5f}")

    # DOF별 물리 단위 RMSE도 계산 (평균 예측 기준)
    model.eval()
    sq_err_sum = np.zeros(6, dtype=np.float64)
    n_elems = 0
    with torch.no_grad():
        for src, trg in test_loader:
            mu, _ = model(src.to(device))
            pred = mu.cpu().numpy()
            trg = trg.numpy()
            # 물리 단위 오차 = z-score 오차 * std
            err_phys = (pred - trg) * data_std
            sq_err_sum += (err_phys ** 2).sum(axis=(0, 1))
            n_elems += pred.shape[0] * pred.shape[1]
    rmse_per_dof = np.sqrt(sq_err_sum / n_elems)
    print(" - DOF별 Test RMSE (물리 단위):")
    for name, unit, r in zip(dof_names, dof_units, rmse_per_dof):
        print(f"     {name:12s}: {r:.4f} {unit}")

    # ---------- 예측 시각화 ----------
    print("\n 테스트 예측 시각화")
    plot_test_predictions(model, test_ds, data_mean, data_std,
                          n_samples=4, save_dir=PLOT_DIR,
                          file_names=id_file_names)

    print(f"\n 완료! 결과 폴더: {RESULT_DIR}")
    print(f" - 모델 가중치 : {MODEL_SAVE_PATH}")
    print(f" - 스케일러    : {SCALER_SAVE_PATH}")
    print(f" - 학습 이력   : {HISTORY_SAVE_PATH}")
    print(f" - 그래프      : {PLOT_DIR}")
