from pathlib import Path
import torchio as tio

def load_image_label_paths(root_dir: str):
    root = Path(root_dir) / "imagesTr"
    label = Path(root_dir) / "labelsTr"
    image_path = sorted(p for p in root.glob("lung*.nii.gz") if not p.name.startswith("."))
    label_path = sorted(p for p in label.glob("lung*.nii.gz") if not p.name.startswith("."))
    return image_path, label_path

def make_subject(image_path, label_path):
    subjects = []
    for img_path, lbl_path in zip(image_path, label_path):
        subject = tio.Subject(
            mri=tio.ScalarImage(img_path),
            mask=tio.LabelMap(lbl_path)
        )
        subjects.append(subject)
    return subjects
