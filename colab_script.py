import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import os
import re
import random
import time
from tqdm.auto import tqdm
from torch.utils.data import Sampler
import warnings
import shutil
import sys
import subprocess

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

QUICK_TRIAL = os.environ.get("QUICK_TRIAL", "0") == "1"

# --- Google Colab Setup & Environment Check ---
def setup_colab_env():
    print("="*50)
    print("Setting up Google Colab Environment...")
    print("="*50)
    
    if os.path.exists("/kaggle") or ("KAGGLE_KERNEL_RUN_TYPE" in os.environ):
        dataset_path = "/kaggle/working/data"
        input_root = Path("/kaggle/input")

        print("="*50)
        print("Detected Kaggle environment")
        print("="*50)

        zip_candidates = list(input_root.rglob("SeizeIT2_preprocessed_windows.zip")) if input_root.exists() else []
        zip_path = str(zip_candidates[0]) if zip_candidates else None

        extracted_root = os.path.join(dataset_path, "SeizeIT2_preprocessed_windows")
        extracted_has_data = False
        if os.path.exists(extracted_root):
            extracted_has_data = bool(list(Path(extracted_root).rglob("*_windows.npz")))
        elif os.path.exists(dataset_path):
            extracted_has_data = bool(list(Path(dataset_path).rglob("*_windows.npz")))

        if not extracted_has_data:
            if zip_path is None or (not os.path.exists(zip_path)):
                npz_candidates = list(input_root.rglob("*_windows.npz")) if input_root.exists() else []
                if npz_candidates:
                    return str(npz_candidates[0].parent)
                print("Error: Could not find SeizeIT2_preprocessed_windows.zip under /kaggle/input")
                return None

            print(f"Unzipping dataset to {dataset_path} ...")
            if os.path.exists(dataset_path):
                shutil.rmtree(dataset_path)
            os.makedirs(dataset_path, exist_ok=True)

            local_zip_path = "/kaggle/working/SeizeIT2_preprocessed_windows.zip"
            try:
                shutil.copy2(zip_path, local_zip_path)
            except Exception as e:
                print(f"Error copying zip to /kaggle/working: {e}")
                return None

            result = subprocess.run(
                ["unzip", "-o", "-q", local_zip_path, "-d", dataset_path],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("Unzip complete!")
            else:
                print("Unzip failed!")
                if result.stderr:
                    print(result.stderr[:2000])
                return None
        else:
            print(f"Dataset already extracted at {dataset_path}")

        if os.path.exists(extracted_root):
            return extracted_root
        return dataset_path

    dataset_path = "/content/data"
    
    try:
        from google.colab import drive # type: ignore
        # Mount Drive if not already mounted
        if not os.path.exists('/content/drive'):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
        else:
            print("Google Drive already mounted.")
            
        # Path to the zip file in Google Drive
        # User said: "home" name "SeizeIT2_preprocessed_windows.zip"
        zip_path = "/content/drive/MyDrive/SeizeIT2_preprocessed_windows.zip"
        
        if not os.path.exists(zip_path):
            print(f"Error: Zip file not found at: {zip_path}")
            print("Please check that the file 'SeizeIT2_preprocessed_windows.zip' is in your 'My Drive' root folder.")
            return None
            
        extracted_root = os.path.join(dataset_path, "SeizeIT2_preprocessed_windows")
        extracted_has_data = False
        if os.path.exists(extracted_root):
            extracted_has_data = bool(list(Path(extracted_root).rglob("*_windows.npz")))
        elif os.path.exists(dataset_path):
            extracted_has_data = bool(list(Path(dataset_path).rglob("*_windows.npz")))

        if not extracted_has_data:
            print(f"Unzipping dataset to {dataset_path} (Local SSD for speed)...")
            print("This may take a few minutes...")

            total, used, free = shutil.disk_usage("/")
            print(f"Local SSD Free Space: {free / (2**30):.2f} GB")

            if os.path.exists(dataset_path):
                shutil.rmtree(dataset_path)
            os.makedirs(dataset_path, exist_ok=True)

            local_zip_path = "/content/SeizeIT2_preprocessed_windows.zip"
            try:
                print("Copying zip to local disk...")
                shutil.copy2(zip_path, local_zip_path)
            except Exception as e:
                print(f"Error copying zip to local disk: {e}")
                return None

            result = subprocess.run(
                ["unzip", "-o", "-q", local_zip_path, "-d", dataset_path],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("Unzip complete!")
            else:
                print("Unzip failed!")
                if result.stderr:
                    print(result.stderr[:2000])
                return None
        else:
            print(f"Dataset already extracted at {dataset_path}")

        # Check Google Drive Free Space
        total, used, free = shutil.disk_usage("/content/drive")
        free_gb = free / (2**30)
        print(f"Google Drive Free Space: {free_gb:.2f} GB")
        if free_gb < 2.0:
            print("WARNING: Your Google Drive has less than 2GB free!")
            print("   Checkpoints might fail to save. Please free up some space.")
            
        if os.path.exists(extracted_root):
            return extracted_root
        return dataset_path
        
    except ImportError:
        print("Not running in Google Colab.")
        return None

def check_system_specs():
    # Check RAM
    from psutil import virtual_memory # type: ignore
    ram_gb = virtual_memory().total / 1e9
    print(f"System RAM: {ram_gb:.2f} GB")
    
    if ram_gb < 25:
        print("Tip: You are using Standard RAM. With Colab Pro+, switch to 'High-RAM' in Runtime > Change runtime type.")
        print("   High-RAM allows loading the entire dataset into memory for maximum speed.")
    else:
        print("High-RAM Runtime detected! We will load all data into RAM for faster training.")
        
    # Check CPU Cores
    import multiprocessing
    cores = multiprocessing.cpu_count()
    print(f"CPU Cores: {cores}")
    return ram_gb, cores

# Check for GPU
def get_device():
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {gpu_name}")
        if "A100" in gpu_name:
            print("A100 GPU Detected. Training will be extremely fast.")
        elif "T4" in gpu_name:
            print("T4 GPU Detected. Good balance of speed and cost.")
        return torch.device("cuda")
    
    print("\n" + "!"*50)
    print("GPU NOT DETECTED!")
    print("In Colab, go to Runtime > Change runtime type > T4 GPU")
    print("!"*50 + "\n")
    return torch.device("cpu")

# --- Model Architecture ---
# CNN-LSTM Hybrid Model
class CNNLSTMSeizureNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=3):
        super(CNNLSTMSeizureNet, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, 
                            batch_first=True, bidirectional=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(64*2, num_classes)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.adaptive_avg_pool1d(x, 256)
        
        x = x.permute(0, 2, 1) # Prepare for LSTM (Batch, Seq, Feature)
        _, (h_n, _) = self.lstm(x)
        
        # Use last hidden state
        x = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(x)

# --- Dataset Class ---
class SeizeIT2Dataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.file_info = []
        self.total_samples = 0
        self.preload_data = {} # Cache for full dataset
        
        # Do NOT preload everything to RAM. Kaggle 30GB RAM is not enough for 378 files uncompressed.
        # We will use the optimized grouped sampler with direct file reading.
        self.use_preload = False
        
        print(f"Reading metadata for {len(file_paths)} files...")
        for i, f in enumerate(file_paths):
            try:
                # Only read metadata (shape), not the actual data
                with np.load(f, mmap_mode="r") as data:
                    n = data["y"].shape[0]
                
                self.file_info.append((f, n, self.total_samples))
                self.total_samples += n
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
        self.start_indices = [x[2] for x in self.file_info]
        self.open_files = {} # Not used if preloaded
        self.last_accessed_path = None

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        import bisect
        # Find the file containing this index
        file_idx = bisect.bisect_right(self.start_indices, idx) - 1
        path, count, start = self.file_info[file_idx]
        local_idx = idx - start
        
        str_path = str(path)
        
        # Standard Loading Strategy (Optimized):
        if self.last_accessed_path != str_path:
            # We must load the new file.
            self.open_files.clear()
            try:
                # np.load into memory directly rather than using mmap to avoid disk seeks during __getitem__
                loaded_npz = np.load(str_path)
                self.open_files[str_path] = {
                    "X": loaded_npz["X"][:], # Load fully to RAM to avoid repeated disk reads
                    "y": loaded_npz["y"][:]
                }
                self.last_accessed_path = str_path
            except Exception as e:
                print(f"Error loading {str_path}: {e}")
                return torch.zeros(1, 15360), torch.tensor(0, dtype=torch.long)
            
        data = self.open_files[str_path]
        x_np = data["X"][local_idx]
        y_np = int(data["y"][local_idx])
        
        x = torch.from_numpy(x_np).float()
        y = torch.tensor(y_np, dtype=torch.long)
        
        # Ensure correct shape (1, 15360)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.shape[1] == 1:
            x = x.permute(1, 0)
            
        return x, y

    def __del__(self):
        if self.last_accessed_path is not None:
            prev = self.open_files.get(self.last_accessed_path)
            try:
                if prev is not None and hasattr(prev, "close"):
                    prev.close()
            except Exception:
                pass

# --- Sampler for Optimized IO ---
class GroupedShuffleSampler(Sampler):
    """
    Shuffles files, then yields all indices from that file.
    This minimizes disk seeks by keeping the same file open for many samples.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.file_indices = []
        
        # Pre-calculate indices for each file
        for i, (path, count, start) in enumerate(dataset.file_info):
            indices = list(range(start, start + count))
            self.file_indices.append(indices)

    def __iter__(self):
        # Shuffle the order of files
        file_order = list(range(len(self.file_indices)))
        random.shuffle(file_order)
        
        final_indices = []
        for file_idx in file_order:
            # Get indices for this file
            indices = self.file_indices[file_idx]
            # Shuffle indices WITHIN the file (for randomness)
            random.shuffle(indices)
            final_indices.extend(indices)
            
        return iter(final_indices)

    def __len__(self):
        return len(self.dataset)

def get_subject_id(filename):
    # Helper to extract subject ID (e.g., sub-001)
    m = re.search(r"(sub-[^_]+)", filename.name)
    if m:
        return m.group(1)
    return filename.name.split("_")[0]

def check_overlap(list1, list2, name1, name2):
    # Check for data leakage between sets
    s1 = set(list1)
    s2 = set(list2)
    overlap = s1.intersection(s2)
    if overlap:
        raise ValueError(f"Leakage detected between {name1} and {name2}!")

def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def class_counts_from_files(file_paths, num_classes=3):
    counts = np.zeros(int(num_classes), dtype=np.int64)
    total = 0
    for p in file_paths:
        try:
            with np.load(str(p), mmap_mode="r") as d:
                y = np.asarray(d["y"])
        except Exception:
            continue
        y = y.astype(np.int64).reshape(-1)
        total += int(y.size)
        if y.size:
            bc = np.bincount(y, minlength=int(num_classes))
            counts[: int(num_classes)] += bc[: int(num_classes)]
    return counts, total

def class_percentages(counts):
    total = int(np.sum(counts))
    if total <= 0:
        return [0.0 for _ in range(len(counts))]
    return [100.0 * float(c) / float(total) for c in counts]

def compute_class_weights(counts, power=0.5, clamp_max=10.0):
    counts = np.asarray(counts, dtype=np.float64)
    num_classes = int(counts.size)
    total = float(np.sum(counts))
    denom = np.maximum(counts, 1.0)
    raw = total / (float(num_classes) * denom)
    weights = np.power(raw, float(power))
    mean = float(np.mean(weights)) if weights.size else 1.0
    if mean > 0:
        weights = weights / mean
    if clamp_max is not None:
        weights = np.clip(weights, 0.0, float(clamp_max))
    return weights.astype(np.float32)

def f1_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.int64)
    num_classes = int(cm.shape[0])
    f1s = []
    recalls = []
    precisions = []
    for i in range(num_classes):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return macro_f1, precisions, recalls, f1s

# --- Training Function ---
def train_colab_model(data_dir):
    print(f"Starting Training...")
    
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    path = Path(data_dir)
    print(f"Searching for data in: {path}")
    files = sorted(list(path.rglob("*_windows.npz")))
    
    if not files:
        print("No files found.")
        print(f"Checking directory contents: {list(path.iterdir())}")
        return None, None
    
    print(f"Found {len(files)} data files!")
        
    # Split data by subject to avoid leakage
    subjects = {}
    for f in files:
        subjects[f] = get_subject_id(f)
        
    unique_subs = sorted(list(set(subjects.values())))
    random.seed(42)
    random.shuffle(unique_subs)
    
    n = len(unique_subs)
    train_subs = set(unique_subs[:int(0.75*n)])
    val_subs = set(unique_subs[int(0.75*n):int(0.9*n)])
    test_subs = set(unique_subs[int(0.9*n):])
    
    train_files = [f for f in files if subjects[f] in train_subs]
    val_files = [f for f in files if subjects[f] in val_subs]
    test_files = [f for f in files if subjects[f] in test_subs]
    
    print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    check_overlap(train_subs, val_subs, "Train", "Val")
    
    # Setup Datasets
    train_ds = SeizeIT2Dataset(train_files, transform=True)
    val_ds = SeizeIT2Dataset(val_files)
    test_ds = SeizeIT2Dataset(test_files)

    train_counts, train_total = class_counts_from_files(train_files, num_classes=3)
    val_counts, val_total = class_counts_from_files(val_files, num_classes=3)
    train_pct = class_percentages(train_counts)
    val_pct = class_percentages(val_counts)
    print(f"TRAIN class counts: {train_counts.tolist()} | pct: {[round(x, 4) for x in train_pct]}")
    print(f"VAL class counts:   {val_counts.tolist()} | pct: {[round(x, 4) for x in val_pct]}")
    
    # We will use class weights in FocalLoss instead of extreme oversampling which crashes I/O on Kaggle
    # We increase the power from 0.5 to 0.8 to give even MORE weight to minority classes
    class_weights_np = compute_class_weights(train_counts, power=0.8, clamp_max=20.0)
    print(f"Class weights: {class_weights_np.tolist()}")
    
    # Check specs to optimize workers
    ram_gb, cores = check_system_specs()
    
    if ram_gb > 25 and cores >= 4:
        workers = min(4, max(0, cores - 1))
        print(f"Colab Pro+ / High-RAM detected: using {workers} workers for data loading.")
    else:
        workers = 2 if cores >= 2 else 0
        print(f"Standard instance: using {workers} workers.")
    
    # Overwrite for Kaggle to force speedup if possible
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        workers = 0 # FORCED to 0 to avoid massive I/O bottlenecks and RAM issues in multiprocessing on Kaggle
        print(f"Kaggle detected: forcing {workers} workers to fix I/O lockup.")

    batch_size = 512 # Default bumped up for faster P100/T4 training
    if device.type == "cuda":
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_mem_gb >= 80:
                batch_size = 1024
            elif gpu_mem_gb >= 40:
                batch_size = 512
            elif gpu_mem_gb >= 16:
                batch_size = 256 # Reduced back to 256 to speed up batch formation
        except Exception:
            pass
    print(f"Train samples: {len(train_ds)} | Batch size: {batch_size} | Steps/epoch: {int(np.ceil(len(train_ds)/batch_size))}")
    
    # Fast grouped sampler that prevents disk thrashing
    train_sampler = GroupedShuffleSampler(train_ds, batch_size)
    print("GroupedShuffleSampler initialized (Fast I/O Mode).")

    dl_kwargs = {
        "num_workers": workers,
        "pin_memory": True,
    }
    # Only use persistent_workers if workers > 0 (prevents ValueError)
    if workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)
    
    # Initialize Model
    model = CNNLSTMSeizureNet(input_channels=1, num_classes=3).to(device)
    
    # Focal Loss
    class FocalLoss(nn.Module):
        def __init__(self, alpha: Union[float, torch.Tensor] = 1.0, gamma: float = 2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = (1 - pt) ** self.gamma * ce_loss
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(device=inputs.device, dtype=loss.dtype)
                w = alpha_t.gather(0, targets)
                loss = w * loss
            else:
                loss = self.alpha * loss
            return loss.mean()

    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
    # Increased gamma from 2.0 to 3.0 to strongly penalize easy examples (Normal class)
    criterion = FocalLoss(alpha=class_weights, gamma=3.0)
    
    # Reduced Learning Rate to prevent jumping to a local minimum that only predicts Normal
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # FINAL TRAINING CONFIGURATION
    num_epochs = 2 if QUICK_TRIAL else 30 # Increased to 30 for deeper learning
    best_val_loss = float('inf')
    best_val_macro_f1 = -1.0
    patience = 0
    
    # History for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    start_epoch = 0
    if os.path.exists("/content/drive/MyDrive"):
        drive_root = "/content/drive/MyDrive"
    elif os.path.exists("/kaggle/working"):
        drive_root = "/kaggle/working"
    else:
        drive_root = os.getcwd()
    latest_ckpt_path = None
    latest_step_ckpt_path = None
    if os.path.exists(drive_root):
        pattern = re.compile(r"seizure_detection_checkpoint_epoch_(\d+)\.pth$")
        pattern_step = re.compile(r"seizure_step_ckpt_epoch_(\d+)_step_(\d+)\.pth$")
        candidates = []
        step_candidates = []
        for name in os.listdir(drive_root):
            m = pattern.match(name)
            if m:
                candidates.append((int(m.group(1)), os.path.join(drive_root, name)))
            ms = pattern_step.match(name)
            if ms:
                step_candidates.append((int(ms.group(1)), int(ms.group(2)), os.path.join(drive_root, name)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, latest_ckpt_path = candidates[-1]
        if step_candidates:
            step_candidates.sort(key=lambda x: (x[0], x[1]))
            latest_step_ckpt_path = step_candidates[-1][2]

    start_step = 0
    if latest_step_ckpt_path is not None:
        try:
            ckpt = torch.load(latest_step_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt and device.type == "cuda" and ckpt["scaler_state_dict"] is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            history = ckpt.get("history", history)
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            best_val_macro_f1 = ckpt.get("best_val_macro_f1", best_val_macro_f1)
            patience = ckpt.get("patience", patience)
            start_epoch = int(ckpt.get("epoch", -1))
            start_step = int(ckpt.get("step", 0))
            if start_epoch < 0:
                start_epoch = 0
            if start_epoch > 0 or start_step > 0:
                print(f"Resuming from step checkpoint: {latest_step_ckpt_path}")
        except Exception as e:
            print(f"Could not resume from step checkpoint: {e}")
            latest_step_ckpt_path = None

    elif latest_ckpt_path is not None:
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt and device.type == "cuda" and ckpt["scaler_state_dict"] is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            history = ckpt.get("history", history)
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            best_val_macro_f1 = ckpt.get("best_val_macro_f1", best_val_macro_f1)
            patience = ckpt.get("patience", patience)
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            if start_epoch > 0:
                print(f"Resuming from checkpoint: {latest_ckpt_path}")
        except Exception as e:
            print(f"Could not resume from checkpoint: {e}")

    if start_epoch >= num_epochs:
        print(f"Already completed {start_epoch}/{num_epochs} epochs. Skipping training.")
    else:
        print(f"Starting Training Loop: epochs {start_epoch+1} to {num_epochs}")
    
    save_step_interval = 20
    for epoch in range(start_epoch, num_epochs):
        if latest_step_ckpt_path is not None and epoch == start_epoch:
            pass
        else:
            start_step = 0
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for i, (x, y) in enumerate(pbar):
            if QUICK_TRIAL and i >= 200:
                break
            if i < start_step:
                continue
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(x)
                loss = criterion(out, y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if save_step_interval and (i + 1) % save_step_interval == 0:
                step_ckpt_name = f"seizure_step_ckpt_epoch_{epoch+1}_step_{i+1}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if device.type == 'cuda' else None,
                    'epoch': epoch,
                    'step': i + 1,
                    'val_acc': None,
                    'best_val_loss': best_val_loss,
                    'patience': patience,
                    'history': history
                }, step_ckpt_name)
                try:
                    if os.path.exists(drive_root):
                        shutil.copy(step_ckpt_name, os.path.join(drive_root, step_ckpt_name))
                except Exception as e:
                    print(f"Warning: Could not save step checkpoint to Drive: {e}")
            
            total_loss += loss.item() * y.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            if i % 10 == 0:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{correct/total:.2%}'})
        
        # Calculate epoch metrics
        train_loss = total_loss / total
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_cm = torch.zeros((3, 3), dtype=torch.long, device="cpu")
        
        with torch.no_grad():
            for j, (x, y) in enumerate(tqdm(val_loader, desc="Validation")):
                if QUICK_TRIAL and j >= 200:
                    break
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    out = model(x)
                    loss = criterion(out, y)
                val_loss += loss.item() * y.size(0)
                pred = out.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                idx = (y.detach().to("cpu").long() * 3) + pred.detach().to("cpu").long()
                bc = torch.bincount(idx, minlength=9).reshape(3, 3)
                val_cm += bc
                
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        val_cm_np = val_cm.numpy()
        val_macro_f1, val_precisions, val_recalls, val_f1s = f1_from_confusion(val_cm_np)
        val_true_dist = val_cm.sum(dim=1).tolist()
        val_pred_dist = val_cm.sum(dim=0).tolist()
        
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        print(f"VAL true dist: {val_true_dist} | pred dist: {val_pred_dist}")
        print(f"VAL macro F1: {val_macro_f1:.4f} | per-class recall: {[round(x, 4) for x in val_recalls]}")
        
        scheduler.step(avg_val_loss)
        
        improved = False
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            improved = True
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            improved = True

        if improved:
            torch.save(model.state_dict(), "best_model.pth")
            try:
                if os.path.exists(drive_root):
                    shutil.copy("best_model.pth", os.path.join(drive_root, "best_model.pth"))
            except Exception as e:
                print(f"Warning: Could not save best_model.pth to Drive: {e}")
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping triggered!")
                break
        
        # Save checkpoint locally and to Drive
        checkpoint_name = f'seizure_detection_checkpoint_epoch_{epoch+1}.pth'
        print(f"Saving checkpoint for epoch {epoch+1}...")
        
        # Save locally
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if device.type == 'cuda' else None,
            'epoch': epoch,
            'val_acc': val_acc,
            'best_val_loss': best_val_loss,
            'best_val_macro_f1': best_val_macro_f1,
            'patience': patience,
            'history': history
        }, checkpoint_name)
        
        # Copy to Drive immediately to prevent data loss
        try:
            drive_ckpt_path = f"/content/drive/MyDrive/{checkpoint_name}"
            shutil.copy(checkpoint_name, drive_ckpt_path)
            print(f"Checkpoint backed up to Drive: {drive_ckpt_path}")
        except Exception as e:
            print(f"Warning: Could not save checkpoint to Drive: {e}")
                
    # --- Final Evaluation ---
    print("\n--- Final Evaluation on Test Set ---")
    best_model_path = "best_model.pth"
    if not os.path.exists(best_model_path):
        drive_best_model_path = os.path.join(drive_root, "best_model.pth")
        if os.path.exists(drive_best_model_path):
            best_model_path = drive_best_model_path
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1)
            
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(prob.cpu().numpy())
            
    results = {}
    results['accuracy'] = np.mean(np.array(all_preds) == np.array(all_labels))
    results['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
    results['classification_report'] = classification_report(all_labels, all_preds, target_names=['Normal', 'Preictal', 'Seizure'], output_dict=True)
    results['classification_report_str'] = classification_report(all_labels, all_preds, target_names=['Normal', 'Preictal', 'Seizure'])
    results['history'] = history
    
    results['sensitivity_per_class'] = []
    results['far_per_class'] = []
    results['auc_per_class'] = []
    
    cm = results['confusion_matrix']
    all_probs = np.array(all_probs)
    
    for i in range(3):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        results['sensitivity_per_class'].append(sens)
        
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        results['far_per_class'].append(1 - spec)
        
        try:
            binary_labels = (np.array(all_labels) == i).astype(int)
            auc = roc_auc_score(binary_labels, all_probs[:, i])
            results['auc_per_class'].append(auc)
        except:
            results['auc_per_class'].append(0.0)
            
    return model, results

# --- Main Execution ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Colab Pro+ Pre-Flight Check")
    print("="*50)
    
    # 0. Check Specs Immediately
    check_system_specs()
    get_device()
    
    print("="*50 + "\n")

    # 1. Setup Environment
    data_dir = setup_colab_env()
    
    if data_dir:
        # 2. Run Training
        try:
            model, res = train_colab_model(data_dir)
            
            if model is not None and res is not None:
                print("\n" + "="*60)
                print("Final Evaluation Results:")
                print("="*60)
                print(f"Test Accuracy: {res['accuracy']:.4f}")
                
                classes = ['Normal', 'Preictal', 'Seizure']
                print(f"\nSensitivity per class:")
                for i, c in enumerate(classes):
                    print(f"  {c}: {res['sensitivity_per_class'][i]:.4f}")
                    
                print(f"\nFalse Alarm Rate (FAR) per class:")
                for i, c in enumerate(classes):
                    print(f"  {c}: {res['far_per_class'][i]:.4f}")
                    
                print(f"\nAUC per class:")
                for i, c in enumerate(classes):
                    print(f"  {c}: {res['auc_per_class'][i]:.4f}")
                    
                print(f"Confusion Matrix:\n{res['confusion_matrix']}")
                
                print("\nClassification Report:")
                print(res['classification_report_str'])
                
                # --- Plotting and Saving Figures ---
                print("\nGenerating and Saving Plots...")
                
                # 1. Plot Training History (Loss & Accuracy)
                hist = res['history']
                epochs_range = range(1, len(hist['train_loss']) + 1)
                
                plt.figure(figsize=(14, 5))
                
                # Loss Plot
                plt.subplot(1, 2, 1)
                plt.plot(epochs_range, hist['train_loss'], label='Train Loss', marker='o')
                plt.plot(epochs_range, hist['val_loss'], label='Val Loss', marker='o')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                # Accuracy Plot
                plt.subplot(1, 2, 2)
                plt.plot(epochs_range, hist['train_acc'], label='Train Acc', marker='o')
                plt.plot(epochs_range, hist['val_acc'], label='Val Acc', marker='o')
                plt.title('Training and Validation Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig('training_history.png')
                print("Saved training_history.png")
                
                # 2. Plot Confusion Matrix
                import seaborn as sns
                plt.figure(figsize=(8, 6))
                cm = res['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=classes, yticklabels=classes)
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig('confusion_matrix.png')
                print("Saved confusion_matrix.png")
                
                # Save plots to Drive
                try:
                    shutil.copy('training_history.png', '/content/drive/MyDrive/training_history.png')
                    shutil.copy('confusion_matrix.png', '/content/drive/MyDrive/confusion_matrix.png')
                    print("Plots successfully saved to Google Drive.")
                except Exception as e:
                    print(f"Could not copy plots to Drive: {e}")

                print("\nSaving full results to Google Drive...")
                drive_save_path = "/content/drive/MyDrive/SeizeIT2_Model_Results.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'results': res
                }, drive_save_path)
                print(f"Saved to: {drive_save_path}")
                print("="*60 + "\n")
                
            else:
                print("\nTraining failed.")
                
        except Exception as e:
            print(f"\nUNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
