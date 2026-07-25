import torchio as tio
from sklearn.model_selection import KFold

base_transform = tio.Compose([
    tio.ToCanonical(),
    tio.Resample(1.5),
    tio.Clamp(-1000, 400),
    tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(-1000, 400)),
])

train_transform = tio.Compose([
    base_transform,
    tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
    tio.RandomAffine(scales=(0.9, 1.1), degrees=15, translation=8),
    tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
    tio.RandomNoise(std=(0, 0.02), p=0.3),
    tio.RandomBlur(std=(0, 1), p=0.2),
    tio.RandomBiasField(p=0.2),
])

val_transform = base_transform

patch_size = (64, 64, 64)
kf = KFold(n_splits=3, shuffle=True, random_state=42)

sampler = tio.LabelSampler(
    patch_size=patch_size,
    label_name="mask",
    label_probabilities={0: 0.2, 1: 0.8}
)

def build_fold_loaders(train_subjects, val_subjects):
    train_ds = tio.SubjectsDataset(train_subjects, transform=train_transform)
    train_queue = tio.Queue(
        subjects_dataset=train_ds,
        max_length=128,
        samples_per_volume=48,
        sampler=sampler,
        num_workers=2,
        shuffle_subjects=True,
        shuffle_patches=True
    )
    train_loader = tio.SubjectsLoader(train_queue, batch_size=2)

    val_ds = tio.SubjectsDataset(val_subjects, transform=val_transform)
    val_loader = tio.SubjectsLoader(val_ds, batch_size=1, num_workers=2, shuffle=False)

    return train_loader, val_loader
