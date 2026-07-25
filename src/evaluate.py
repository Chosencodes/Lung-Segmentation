import os, glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric, HausdorffDistanceMetric

from src.data.preprocessing import val_transform, patch_size, kf
from src.data.dataset import load_image_label_paths, make_subject
from src.models.lightning_module import LungTumorSegmentation

dice_metric = DiceMetric(include_background=True, reduction="none")
iou_metric = MeanIoU(include_background=True, reduction="none")
hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none")
confusion_metric = ConfusionMatrixMetric(
    include_background=True,
    metric_name=["precision", "sensitivity", "specificity"],
    reduction="none",
)


def compute_case_metrics(pred, gt, spacing=None) -> dict:
    """
    pred, gt: shape (1, 1, H, W, D), binary tensors.
    spacing: physical voxel spacing (mm), used for HD95.
    """
    pred_cpu = pred.cpu()
    gt_cpu = gt.cpu()

    dice = dice_metric(pred_cpu, gt_cpu).item()
    iou = iou_metric(pred_cpu, gt_cpu).item()
    hd95 = hd95_metric(pred_cpu, gt_cpu, spacing=spacing).item()

    confusion_metric(pred_cpu, gt_cpu)
    prec_t, rec_t, spec_t = confusion_metric.aggregate()
    confusion_metric.reset()

    return {"dice": dice, "iou": iou, "hd95": hd95,
            "precision": prec_t.item(), "recall": rec_t.item(), "specificity": spec_t.item()}


def evaluate_fold(fold, val_subjects, ckpt_dir):
    ckpt = sorted(glob.glob(f"{ckpt_dir}/*best*.ckpt"))[-1]
    print(f"Fold {fold}: loading {ckpt}")
    model = LungTumorSegmentation.load_from_checkpoint(ckpt).cuda().eval()

    rows = []
    for subject in tqdm(val_subjects, desc=f"Fold {fold} eval"):
        subject = val_transform(subject)
        image = subject["mri"]["data"].unsqueeze(0).cuda()
        mask = subject["mask"]["data"].unsqueeze(0).cuda()

        with torch.no_grad():
            logits = sliding_window_inference(
                image, patch_size, 2, model, overlap=0.75, mode="gaussian")
        pred = (torch.sigmoid(logits) > 0.5).to(torch.uint8)

        m = compute_case_metrics(pred, mask.to(torch.uint8), spacing=subject["mri"].spacing)
        m["fold"] = fold
        m["case"] = subject["mri"].path.name
        rows.append(m)

    return pd.DataFrame(rows)


def run_evaluation(data_root, ckpt_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    image_path, label_path = load_image_label_paths(data_root)
    all_subject = make_subject(image_path, label_path)

    all_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_subject)):
        fold = fold_idx + 1
        ckpt_dir = os.path.join(ckpt_root, f"fold_{fold}")
        if not os.path.exists(ckpt_dir):
            print(f"Skipping fold {fold} — no checkpoint dir")
            continue
        va = [all_subject[i] for i in val_idx]
        all_results.append(evaluate_fold(fold, va, ckpt_dir))

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(os.path.join(out_dir, "per_case_metrics.csv"), index=False)
    print(f"\nEvaluated {len(results)} cases across {results['fold'].nunique()} fold(s)")
    return results


if __name__ == "__main__":
    run_evaluation(
        data_root="data/Task06_Lung",
        ckpt_root="checkpoints",
        out_dir="results",
    )
