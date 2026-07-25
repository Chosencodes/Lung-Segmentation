import os, shutil, json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from src.data.dataset import load_image_label_paths, make_subject
from src.data.preprocessing import build_fold_loaders, kf, patch_size
from src.models.lightning_module import LungTumorSegmentation, build_model


def model_fingerprint():
    m = build_model()
    return [tuple(p.shape) for p in m.parameters()]


def fingerprint_matches(ckpt_dir, current_fingerprint):
    fp_file = os.path.join(ckpt_dir, "architecture.json")
    if not os.path.exists(fp_file):
        return False
    with open(fp_file) as f:
        saved = json.load(f)
    return saved == [list(s) for s in current_fingerprint]


def save_fingerprint(ckpt_dir, current_fingerprint):
    with open(os.path.join(ckpt_dir, "architecture.json"), "w") as f:
        json.dump([list(s) for s in current_fingerprint], f)


def train_all_folds(data_root, ckpt_root, log_dir):
    image_path, label_path = load_image_label_paths(data_root)
    all_subject = make_subject(image_path, label_path)

    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    current_fingerprint = model_fingerprint()
    pl.seed_everything(42, workers=True)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_subject), start=1):
        ckpt_dir = os.path.join(ckpt_root, f"fold_{fold}")
        os.makedirs(ckpt_dir, exist_ok=True)
        done_marker = os.path.join(ckpt_dir, "done.txt")

        if os.path.exists(done_marker):
            with open(done_marker) as f:
                score = float(f.read().split("=")[1])
            fold_scores.append(score)
            print(f"Fold {fold} already done ({score:.4f}) — skipping")
            continue

        print(f"Training Fold {fold}")
        tr = [all_subject[i] for i in train_idx]
        va = [all_subject[i] for i in val_idx]
        train_loader, val_loader = build_fold_loaders(tr, va)

        model = LungTumorSegmentation(lr=5e-4, patch_size=patch_size)

        checkpoint = ModelCheckpoint(
            monitor="val_dice", mode="max", save_top_k=1, save_last=True,
            dirpath=ckpt_dir, filename="best-{epoch}-{val_dice:.3f}")
        step_checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir, filename="step-checkpoint",
            every_n_train_steps=100, save_top_k=1, save_last=False)
        early_stop = EarlyStopping(monitor="val_dice", mode="max", patience=25)
        lr_mon = LearningRateMonitor(logging_interval="epoch")

        loggers = [TensorBoardLogger(save_dir=log_dir, name=f"fold{fold}"),
                   CSVLogger(save_dir=log_dir, name=f"fold{fold}_csv")]

        trainer = pl.Trainer(
            accelerator="gpu", devices=1, max_epochs=120,
            precision="32-true", gradient_clip_val=1.0,
            check_val_every_n_epoch=1,
            callbacks=[checkpoint, step_checkpoint, early_stop, lr_mon],
            logger=loggers, log_every_n_steps=5, num_sanity_val_steps=0)

        last = os.path.join(ckpt_dir, "last.ckpt")
        step_ckpt = os.path.join(ckpt_dir, "step-checkpoint.ckpt")
        resume_path = None
        candidates = [p for p in [last, step_ckpt] if os.path.exists(p)]
        if candidates:
            newest = max(candidates, key=os.path.getmtime)
            if fingerprint_matches(ckpt_dir, current_fingerprint):
                resume_path = newest
                print(f"Resuming Fold {fold} from {newest}")
            else:
                shutil.rmtree(ckpt_dir, ignore_errors=True)
                os.makedirs(ckpt_dir, exist_ok=True)

        save_fingerprint(ckpt_dir, current_fingerprint)

        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)

        best = checkpoint.best_model_score.item()
        fold_scores.append(best)
        with open(done_marker, "w") as f:
            f.write(f"best_dice={best:.4f}")
        print(f"Fold {fold} Best Dice: {best:.4f}")

    print(f"\nPer-fold: {[round(s, 4) for s in fold_scores]}")
    return fold_scores


if __name__ == "__main__":
    train_all_folds(
        data_root="data/Task06_Lung",
        ckpt_root="checkpoints",
        log_dir="logs",
    )
