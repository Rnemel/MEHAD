import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import mne
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Load signals
def load_ecg_run(ecg_path):
    raw = mne.io.read_raw_edf(str(ecg_path), preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    df = raw.to_data_frame()
    df = df.set_index("time")
    return df, sfreq


def load_events_for_run(root, subject, run_stem):
    if run_stem.endswith("_ecg"):
        base_stem = run_stem[:-4]
    elif run_stem.endswith("_mov"):
        base_stem = run_stem[:-4]
    elif run_stem.endswith("_eeg"):
        base_stem = run_stem[:-4]
    else:
        base_stem = run_stem
    eeg_events_path = (
        Path(root)
        / subject
        / "ses-01"
        / "eeg"
        / f"{base_stem}_events.tsv"
    )
    if not eeg_events_path.exists():
        print("لم يتم العثور على ملف events.tsv المتوقع في:", eeg_events_path)
        return None
    events_df = pd.read_csv(eeg_events_path, sep="\t")
    return events_df


# Build sample labels
def build_sample_labels(times, events_df, pre_ictal_minutes=30.0):
    labels = np.zeros(len(times), dtype=int)
    ignore_mask = np.zeros(len(times), dtype=bool)
    for _, row in events_df.iterrows():
        onset = float(row["onset"])
        duration = float(row["duration"])
        event_type = str(row["eventType"])
        start = onset
        end = onset + duration
        if event_type == "impd":
            m = (times >= start) & (times < end)
            ignore_mask[m] = True
        if event_type.startswith("sz_"):
            ictal_mask = (times >= start) & (times < end)
            labels[ictal_mask] = 2
            pre_start = max(0.0, start - pre_ictal_minutes * 60.0)
            pre_mask = (times >= pre_start) & (times < start) & (~ictal_mask)
            pre_mask = pre_mask & (~ignore_mask)
            labels[pre_mask] = 1
    labels[ignore_mask] = -1
    return labels


# Handle missing/noisy data
def handle_missing_and_noisy(signals_df, max_nan_fraction=0.3):
    nan_fraction = signals_df.isna().mean(axis=1)
    valid_mask = nan_fraction < max_nan_fraction
    cleaned = signals_df[valid_mask]
    cleaned = cleaned.interpolate(limit_direction="both")
    return cleaned


# Normalize signals
def normalize_signals(signals_df, labels, inter_ictal_label=0):
    if labels is None:
        stats_df = signals_df
    else:
        mask = labels == inter_ictal_label
        stats_df = signals_df[mask] if mask.any() else signals_df
    mean = stats_df.mean()
    std = stats_df.std().replace(0, 1.0)
    normalized = (signals_df - mean) / std
    return normalized, mean, std


# Segment into windows
def segment_into_windows(signals_df, labels, sfreq, window_sec=60.0, step_sec=30.0):
    window_size = int(window_sec * sfreq)
    step_size = int(step_sec * sfreq)
    values = signals_df.values
    y = np.asarray(labels)
    X_windows = []
    y_windows = []
    start = 0
    while start + window_size <= len(values):
        end = start + window_size
        window_labels = y[start:end]
        if (window_labels == -1).any():
            start += step_size
            continue
        # Assign window labels
        counts = np.bincount(window_labels)
        window_label = int(counts.argmax())
        X_windows.append(values[start:end])
        y_windows.append(window_label)
        start += step_size
    if not X_windows:
        return np.empty((0, window_size, values.shape[1])), np.empty((0,), dtype=int)
    X_arr = np.stack(X_windows, axis=0)
    y_arr = np.asarray(y_windows, dtype=int)
    return X_arr, y_arr


# Compute class weights
def compute_class_weights(y_windows):
    if len(y_windows) == 0:
        return {}
    classes = np.unique(y_windows)
    weights = compute_class_weight("balanced", classes=classes, y=y_windows)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def plot_pre_seizure_segment(times, signals_df, labels, seizure_label=2, minutes_before=30.0, minutes_after=5.0, acc_prefix="ACC"):
    if len(times) == 0 or len(labels) == 0:
        return
    labels = np.asarray(labels)
    seizure_indices = np.where(labels == seizure_label)[0]
    if seizure_indices.size == 0:
        return
    onset_idx = seizure_indices[0]
    onset_time = times[onset_idx]
    start_time = onset_time - minutes_before * 60.0
    end_time = onset_time + minutes_after * 60.0
    mask = (times >= start_time) & (times <= end_time)
    seg = signals_df.iloc[mask]
    rel_time = (seg.index.values - onset_time) / 60.0
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ecg_cols = [c for c in seg.columns if "ECG" in c]
    if ecg_cols:
        axes[0].plot(rel_time, seg[ecg_cols[0]], label=ecg_cols[0])
    axes[0].axvline(0.0, color="red", linestyle="--")
    axes[0].set_ylabel("ECG")
    axes[0].legend(loc="upper right")
    acc_cols = [c for c in seg.columns if acc_prefix in c]
    for c in acc_cols:
        axes[1].plot(rel_time, seg[c], label=c)
    axes[1].axvline(0.0, color="red", linestyle="--")
    axes[1].set_ylabel("Movement")
    axes[1].set_xlabel("Time (min relative to seizure onset)")
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def process_seizeit2_ecg(
    root="/Volumes/SEIZE_DATA/SeizeIT2",
    out_dir=None,
    window_sec=60.0,
    step_sec=30.0,
    pre_ictal_minutes=30.0,
    max_subjects=None,
):
    root_path = Path(root)
    if out_dir is None:
        out_path = root_path.parent / "SeizeIT2_preprocessed_windows"
    else:
        out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ecg_files = sorted(root_path.glob("sub-*/ses-01/ecg/*_ecg.edf"))
    all_y = []
    subjects_seen = set()
    for ecg_path in ecg_files:
        subject = ecg_path.parts[-4]
        subjects_seen.add(subject)
        if max_subjects is not None and len(subjects_seen) > max_subjects:
            break
        run_stem = ecg_path.stem
        events_df = load_events_for_run(root, subject, run_stem)
        if events_df is None:
            print("لا يوجد events.tsv لهذا التشغيل:", run_stem)
            continue
        signals_df, sfreq = load_ecg_run(ecg_path)
        signals_df = handle_missing_and_noisy(signals_df)
        times = signals_df.index.values.astype(float)
        sample_labels = build_sample_labels(times, events_df, pre_ictal_minutes=pre_ictal_minutes)
        if (sample_labels == 2).sum() == 0:
            print("تشغيل بدون نوبات (لا يوجد ictal) سيتم تجاوزه:", run_stem)
            continue
        normalized_df, _, _ = normalize_signals(signals_df, sample_labels)
        X, y = segment_into_windows(
            normalized_df,
            sample_labels,
            sfreq,
            window_sec=window_sec,
            step_sec=step_sec,
        )
        if len(y) == 0:
            print("لم يتم استخراج أي نوافذ صالحة من:", run_stem)
            continue
        print(f"run: {run_stem}, windows: {len(y)}, class counts:", {int(c): int((y == c).sum()) for c in np.unique(y)})
        # Save .npz files
        try:
            np.savez_compressed(
                out_path / f"{run_stem}_windows.npz",
                X=X,
                y=y,
            )
        except OSError as e:
            print("تعذر حفظ النوافذ بسبب مشكلة في الكتابة (غالبًا المساحة ممتلئة):", e)
            break
        all_y.append(y)
    if all_y:
        y_all = np.concatenate(all_y, axis=0)
        weights = compute_class_weights(y_all)
    else:
        weights = {}
    return weights

# دالة لحساب إحصائيات عامة عن النوافذ من ملفات .npz الجاهزة
def summarize_preprocessed_windows(pre_dir=None):
    if pre_dir is None:
        pre_path = Path("/Volumes/SEIZE_DATA/SeizeIT2_preprocessed_windows")
    else:
        pre_path = Path(pre_dir)
    all_y = []
    npz_files = sorted(pre_path.glob("*_windows.npz"))
    if not npz_files:
        print("لا توجد ملفات نوافذ جاهزة في المجلد:", pre_path)
        return
    for npz_path in npz_files:
        data = np.load(npz_path)
        y = data["y"]
        all_y.append(y)
    y_all = np.concatenate(all_y)
    total = int(len(y_all))
    print("إجمالي عدد النوافذ في مجموعة SeizeIT2:", total)
    for c in np.unique(y_all):
        count = int((y_all == c).sum())
        pct = 100.0 * count / total
        if c == 0:
            name = "inter-ictal"
        elif c == 1:
            name = "pre-ictal"
        elif c == 2:
            name = "ictal"
        else:
            name = "other"
        print(f"label {c} ({name}): count={count}, percent={pct:.2f}%")



def load_mov_run(mov_path):
    raw_mov = mne.io.read_raw_edf(str(mov_path), preload=True, verbose=False)
    df_mov = raw_mov.to_data_frame()
    df_mov = df_mov.set_index("time")
    return df_mov

#--------------------Rnem----------------------------------------

def plot_example_pre_seizure(
    root="/Volumes/SEIZE_DATA/SeizeIT2",
    subject="sub-001",
    run_stem="sub-001_ses-01_task-szMonitoring_run-03_ecg",
    minutes_before=30.0,
    minutes_after=5.0,
    save_path="/Users/rnemalmalki/Desktop/prof plan/pre_seizure_example.png",
):
  
    ecg_path = Path(root) / subject / "ses-01" / "ecg" / f"{run_stem}.edf"
    
    if run_stem.endswith("_ecg") or run_stem.endswith("_mov") or run_stem.endswith("_eeg"):
        base_stem = run_stem[:-4]
    else:
        base_stem = run_stem
    mov_path = Path(root) / subject / "ses-01" / "mov" / f"{base_stem}_mov.edf"
    if not ecg_path.exists():
        print("ملف ECG غير موجود:", ecg_path)
        return
    events_df = load_events_for_run(root, subject, run_stem)

    if events_df is None:
        print("لا يمكن إيجاد events.tsv لهذا التشغيل:", run_stem)
        return
    
    signals_df, sfreq = load_ecg_run(ecg_path)
    signals_df = handle_missing_and_noisy(signals_df)

    if mov_path.exists():
        try:
            mov_df = load_mov_run(mov_path)
            mov_df = mov_df.reindex(signals_df.index).interpolate(limit_direction="both")
            for col in mov_df.columns:
                signals_df[f"ACC_{col}"] = mov_df[col]
        except Exception as e:
            print("تعذر قراءة ملف الحركة MOV، سيتم الرسم بدون ACC:", e)
    times = signals_df.index.values.astype(float)
    sample_labels = build_sample_labels(times, events_df, pre_ictal_minutes=minutes_before)
    plot_pre_seizure_segment(
        times,
        signals_df,
        sample_labels,
        seizure_label=2,
        minutes_before=minutes_before,
        minutes_after=minutes_after,
    )
    try:
        plt.savefig(save_path, dpi=150)
        print("تم حفظ الشكل في:", save_path)
    except Exception as e:
        print("تعذر حفظ الشكل:", e)


class CNNLSTMSeizureNet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm_hidden_size = 64
        self.lstm_layers = 1
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * self.lstm_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Channels, Time) -> (B, 1, 15360)
        # Conv1d expects (B, C, L)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        logits = self.fc(h_last)
        return logits


class SeizeIT2LazyDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        
        # We need to know the size of each file to map global index -> (file_index, local_index)
        # This might take a few seconds to scan, but it's safe for RAM.
        self.file_info = []
        self.total_samples = 0
        
        print(f"Scanning {len(file_paths)} files for indexing...")
        for i, f in enumerate(file_paths):
            try:
                # Quick scan: read only shape, don't load data
                with np.load(f) as data:
                    n = data["y"].shape[0]
                    self.file_info.append((f, n, self.total_samples))
                    self.total_samples += n
            except Exception as e:
                print(f"Error scanning {f}: {e}")
                
            if i % 50 == 0:
                 print(f"Scanned {i}/{len(file_paths)}...", end='\r')
                 
        print(f"\nTotal samples found: {self.total_samples}")
        
        # Pre-calculate start indices for binary search
        self.start_indices = [info[2] for info in self.file_info]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Binary search to find which file contains 'idx'
        import bisect
        
        # Find the right file index
        file_idx = bisect.bisect_right(self.start_indices, idx) - 1
        
        file_path, count, start_idx = self.file_info[file_idx]
        local_idx = idx - start_idx
        
        # Load ONLY the specific sample from the specific file
        # Using mmap_mode='r' allows accessing a slice without reading the whole file to RAM
        try:
            # Check if file exists
            if not file_path.exists():
                # Should not happen, but safe fallback
                return torch.zeros(15360, 1), torch.tensor(0, dtype=torch.long)

            # Use mmap_mode='r' to read specific slice from disk
            data = np.load(file_path, mmap_mode='r')
            x_np = data["X"][local_idx] # Shape (15360, 1) or similar
            y_np = data["y"][local_idx] # Scalar
            
            # Copy to RAM (small chunk)
            x = torch.from_numpy(np.array(x_np)).float()
            y = torch.tensor(int(y_np), dtype=torch.long)
            
            # Ensure shape is (Channels, Time) for Conv1d
            # Current shape might be (Time, 1) -> (1, Time)
            if x.shape[0] == 15360: # (Time, Channels) or (Time,)
                 if x.ndim == 1:
                     x = x.unsqueeze(0) # (1, Time)
                 elif x.shape[1] == 1:
                     x = x.permute(1, 0) # (1, Time)
            
            return x, y
            
        except Exception as e:
            print(f"Error reading index {idx} from {file_path}: {e}")
            return torch.zeros(1, 15360), torch.tensor(0, dtype=torch.long)

def train_initial_model(
    pre_dir="/Volumes/SEIZE_DATA/SeizeIT2_preprocessed_windows",
    mmap_dir=None, # Not used in Lazy mode
    batch_size=64,
    num_epochs=5,
    lr=1e-3,
    use_mmap=False, # We use LazyDataset instead
):
    print(f"Starting training setup (Lazy Loading Mode)...")
    
    # Auto-detect device: CUDA -> MPS (Mac) -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal) GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU (Warning: Training might be slow)")
    
    pre_path = Path(pre_dir)
    
    # 1. Gather all files
    all_files = sorted(list(pre_path.glob("*_windows.npz")))
    if not all_files:
        raise ValueError(f"No .npz files found in {pre_dir}")
        
    # 2. Split subjects (Train/Val/Test)
    subjects_map = {} # file_path -> subject
    for f in all_files:
        subjects_map[f] = f.name.split('_')[0]
        
    unique_subjects = sorted(list(set(subjects_map.values())))
    rng = np.random.RandomState(42)
    rng.shuffle(unique_subjects)
    
    n_subjs = len(unique_subjects)
    n_train = int(0.70 * n_subjs)
    n_val = int(0.15 * n_subjs)
    
    train_subjs = set(unique_subjects[:n_train])
    val_subjs = set(unique_subjects[n_train:n_train+n_val])
    test_subjs = set(unique_subjects[n_train+n_val:])
    
    print(f"Subjects Split -> Train: {len(train_subjs)}, Val: {len(val_subjs)}, Test: {len(test_subjs)}")
    
    # 3. Create file lists for each split
    train_files = [f for f in all_files if subjects_map[f] in train_subjs]
    val_files = [f for f in all_files if subjects_map[f] in val_subjs]
    test_files = [f for f in all_files if subjects_map[f] in test_subjs]
    
    # 4. Create Datasets
    print("Initializing Train Dataset...")
    train_dataset = SeizeIT2LazyDataset(train_files)
    print("Initializing Val Dataset...")
    val_dataset = SeizeIT2LazyDataset(val_files)
    print("Initializing Test Dataset...")
    test_dataset = SeizeIT2LazyDataset(test_files)
    
    # 5. DataLoaders
    # Optimize for Mac: Use num_workers > 0 for parallel loading, but keep it moderate
    # Enable pin_memory for faster host-to-device transfer
    workers = 4 
    print(f"Using {workers} workers for data loading...")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, persistent_workers=True)
    
    # Model Setup
    # Hardcode 1 channel as per our data knowledge (ECG)
    model = CNNLSTMSeizureNet(input_channels=1, num_classes=3).to(device)
    
    # Simple class weights (approximate) or standard CE
    # Calculating exact weights requires full scan, skipping for speed now
    criterion = nn.CrossEntropyLoss() 
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * y_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
            if i % 5 == 0:
                print(f"  Batch {i}: Loss {loss.item():.4f} (Accumulated Acc: {correct/total:.2%})", end='\r')
        
        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
                
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"\nSummary: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # Final Test Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            preds = logits.argmax(dim=1)
            
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)
            
    test_acc = test_correct / test_total if test_total > 0 else 0.0
    print(f"Final Test Accuracy: {test_acc:.4f}")
        
    return model


if __name__ == "__main__":
    summarize_preprocessed_windows()
    plot_example_pre_seizure()
