import torch
from torchmetrics import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    MeanSquaredError,
)
import scipy.io as scio
import numpy as np
import cv2
import scipy.io as scio
from scipy import ndimage


def process_label_tensor(label_tensor):
    # Convert the label tensor to a NumPy array
    label = label_tensor.cpu().numpy()  # Assuming label_tensor is on CPU

    # Threshold for binary mask
    threshold = 0.03 * np.max(np.abs(label))
    mask1 = np.abs(label) > threshold

    # Convert the mask to a binary image (0 or 255)
    mask1 = np.uint8(mask1) * 255

    # Create a circular structuring element with a radius of 10
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    # Erode the mask using the structuring element
    mask1 = cv2.erode(mask1, se)

    # Dilate the mask using a larger circular structuring element
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    mask1 = cv2.dilate(mask1, se)

    # Convert the NumPy array back to a PyTorch tensor
    mask1_tensor = torch.from_numpy(mask1).to(label_tensor.device)

    return mask1_tensor


def Evaluation_metrics(label, recon, mask=False, protocol="legacy"):
    if mask:
        mask = scio.loadmat("mask/photom_mask.mat")["mask"]
        mask = mask.astype(np.complex128)
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask).to(label)
        print(mask.shape)
        print(recon.shape)
        recon = torch.mul(mask, recon)

        print("make photom")
        pass

    if not torch.is_tensor(label):
        label = torch.from_numpy(label)
        recon = torch.from_numpy(recon)
    label = torch.abs(label).detach().cpu().float()
    recon = torch.abs(recon).detach().cpu().float()

    label = torch.squeeze(label)
    recon = torch.squeeze(recon)
    if label.ndim == 2:
        label = label.unsqueeze(0).unsqueeze(0)
        recon = recon.unsqueeze(0).unsqueeze(0)
    elif label.ndim == 3:
        label = label.unsqueeze(1)
        recon = recon.unsqueeze(1)
    elif label.ndim != 4:
        raise ValueError(f"Unsupported metric tensor shape: {tuple(label.shape)}")
    if label.shape != recon.shape:
        raise ValueError(
            f"Metric shape mismatch: label={tuple(label.shape)}, recon={tuple(recon.shape)}"
        )

    protocol = str(protocol).lower()
    if protocol == "legacy":
        # Preserve the metric convention used by the original project/results.
        res = StructuralSimilarityIndexMeasure()(recon, label)
        psnrs = PeakSignalNoiseRatio()(recon, label)
        recon_nmse = recon / torch.clamp(torch.max(recon), min=1e-12)
        label_nmse = label / torch.clamp(torch.max(label), min=1e-12)
        err = recon_nmse - label_nmse
        nmse = torch.sum(torch.abs(err) ** 2) / torch.clamp(
            torch.sum(torch.abs(label_nmse) ** 2), min=1e-12
        )
    elif protocol == "standard":
        # Use one reference-derived scale so amplitude errors remain visible.
        reference_scale = torch.clamp(torch.max(label), min=1e-12)
        label = label / reference_scale
        recon = recon / reference_scale

        res = StructuralSimilarityIndexMeasure(data_range=1.0)(recon, label)
        psnrs = PeakSignalNoiseRatio(data_range=1.0)(recon, label)
        err = recon - label
        nmse = torch.sum(torch.abs(err) ** 2) / torch.clamp(
            torch.sum(torch.abs(label) ** 2), min=1e-12
        )
    else:
        raise ValueError(
            f"Unsupported metric protocol {protocol!r}; "
            "expected 'legacy' or 'standard'"
        )

    return res, psnrs, nmse


def HFEN_metric(label, recon, sigma=1.5):
    if not torch.is_tensor(label):
        label = torch.from_numpy(label)
    if not torch.is_tensor(recon):
        recon = torch.from_numpy(recon)

    label_np = torch.abs(label).detach().cpu().float().numpy()
    recon_np = torch.abs(recon).detach().cpu().float().numpy()
    label_np = np.squeeze(label_np)
    recon_np = np.squeeze(recon_np)

    label_log = ndimage.gaussian_laplace(label_np, sigma=sigma)
    recon_log = ndimage.gaussian_laplace(recon_np, sigma=sigma)
    numerator = np.linalg.norm((recon_log - label_log).reshape(-1))
    denominator = np.linalg.norm(label_log.reshape(-1)) + 1e-12
    return float(numerator / denominator)
