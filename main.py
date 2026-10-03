import os
import cv2
import numpy as np
import pywt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# ============================================================
# DWT-BASED MEDICAL IMAGE FUSION
# CT + MRI IMAGE FUSION
# ============================================================

print("=" * 60)
print("       DWT-BASED MEDICAL IMAGE FUSION SYSTEM")
print("=" * 60)


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

CT_PATH = os.path.join("CT", "16003.png")
MRI_PATH = os.path.join("MRI", "16003.png")

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FUSED_PATH = os.path.join(OUTPUT_DIR, "fused_image.png")
CT_GRAY_PATH = os.path.join(OUTPUT_DIR, "ct_processed.png")
MRI_GRAY_PATH = os.path.join(OUTPUT_DIR, "mri_processed.png")


# ------------------------------------------------------------
# 2. CHECK FILES
# ------------------------------------------------------------

print("\n[1] Checking input images...")

if not os.path.exists(CT_PATH):
    print("ERROR: CT image not found!")
    print("Expected:", CT_PATH)
    raise SystemExit

if not os.path.exists(MRI_PATH):
    print("ERROR: MRI image not found!")
    print("Expected:", MRI_PATH)
    raise SystemExit

print("CT image found  :", CT_PATH)
print("MRI image found :", MRI_PATH)


# ------------------------------------------------------------
# 3. LOAD IMAGES
# ------------------------------------------------------------

print("\n[2] Loading CT and MRI images...")

ct_image = cv2.imread(CT_PATH, cv2.IMREAD_GRAYSCALE)
mri_image = cv2.imread(MRI_PATH, cv2.IMREAD_GRAYSCALE)

if ct_image is None:
    print("ERROR: Unable to read CT image.")
    raise SystemExit

if mri_image is None:
    print("ERROR: Unable to read MRI image.")
    raise SystemExit

print("CT shape  :", ct_image.shape)
print("MRI shape :", mri_image.shape)


# ------------------------------------------------------------
# 4. IMAGE PREPROCESSING
# ------------------------------------------------------------

print("\n[3] Preprocessing images...")

# Resize MRI to CT dimensions
if ct_image.shape != mri_image.shape:
    mri_image = cv2.resize(
        mri_image,
        (ct_image.shape[1], ct_image.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

# Normalize both images to 0-255
ct_image = cv2.normalize(
    ct_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

mri_image = cv2.normalize(
    mri_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite(CT_GRAY_PATH, ct_image)
cv2.imwrite(MRI_GRAY_PATH, mri_image)

print("Preprocessing completed.")
print("Final image size:", ct_image.shape)


# ------------------------------------------------------------
# 5. CONVERT TO FLOAT
# ------------------------------------------------------------

ct_float = ct_image.astype(np.float32) / 255.0
mri_float = mri_image.astype(np.float32) / 255.0


# ------------------------------------------------------------
# 6. DWT DECOMPOSITION
# ------------------------------------------------------------

print("\n[4] Performing DWT decomposition...")

wavelet = "db2"

ct_coefficients = pywt.dwt2(ct_float, wavelet)
mri_coefficients = pywt.dwt2(mri_float, wavelet)

ct_LL, (ct_LH, ct_HL, ct_HH) = ct_coefficients
mri_LL, (mri_LH, mri_HL, mri_HH) = mri_coefficients

print("DWT decomposition completed.")

print("CT LL shape :", ct_LL.shape)
print("MRI LL shape:", mri_LL.shape)


# ------------------------------------------------------------
# 7. FUSION RULE
# ------------------------------------------------------------

print("\n[5] Applying DWT fusion rule...")


def average_fusion(ct_component, mri_component):
    """
    Average fusion for low-frequency components.
    """
    return (ct_component + mri_component) / 2.0


def maximum_abs_fusion(ct_component, mri_component):
    """
    Select coefficient having greater absolute magnitude.
    Useful for preserving important high-frequency details.
    """
    mask = np.abs(ct_component) >= np.abs(mri_component)

    fused = np.where(
        mask,
        ct_component,
        mri_component
    )

    return fused


# Low-frequency component
fused_LL = average_fusion(ct_LL, mri_LL)

# High-frequency components
fused_LH = maximum_abs_fusion(ct_LH, mri_LH)
fused_HL = maximum_abs_fusion(ct_HL, mri_HL)
fused_HH = maximum_abs_fusion(ct_HH, mri_HH)

print("Fusion rule applied.")


# ------------------------------------------------------------
# 8. INVERSE DWT
# ------------------------------------------------------------

print("\n[6] Performing inverse DWT...")

fused_coefficients = (
    fused_LL,
    (fused_LH, fused_HL, fused_HH)
)

fused_image = pywt.idwt2(
    fused_coefficients,
    wavelet
)

print("Inverse DWT completed.")


# ------------------------------------------------------------
# 9. NORMALIZE FUSED IMAGE
# ------------------------------------------------------------

fused_image = np.nan_to_num(fused_image)

fused_image = np.clip(
    fused_image,
    0,
    1
)

fused_image_uint8 = (
    fused_image * 255
).astype(np.uint8)


# Make sure output has exactly the same size
fused_image_uint8 = cv2.resize(
    fused_image_uint8,
    (ct_image.shape[1], ct_image.shape[0]),
    interpolation=cv2.INTER_LINEAR
)


# ------------------------------------------------------------
# 10. IMAGE ENHANCEMENT
# ------------------------------------------------------------

print("\n[7] Enhancing fused image...")

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced_fused = clahe.apply(
    fused_image_uint8
)

# Save final image
cv2.imwrite(
    FUSED_PATH,
    enhanced_fused
)

print("Fused image saved successfully.")


# ------------------------------------------------------------
# 11. IMAGE QUALITY METRICS
# ------------------------------------------------------------

print("\n[8] Calculating image quality metrics...")


# SSIM between CT and fused image
ssim_ct = ssim(
    ct_image,
    enhanced_fused,
    data_range=255
)

# SSIM between MRI and fused image
ssim_mri = ssim(
    mri_image,
    enhanced_fused,
    data_range=255
)


# PSNR between CT and fused image
psnr_ct = psnr(
    ct_image,
    enhanced_fused,
    data_range=255
)

# PSNR between MRI and fused image
psnr_mri = psnr(
    mri_image,
    enhanced_fused,
    data_range=255
)


# ------------------------------------------------------------
# 12. INFORMATION ENTROPY
# ------------------------------------------------------------

def calculate_entropy(image):
    """
    Calculate information entropy of an image.
    Higher entropy generally indicates more information content.
    """

    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram / histogram.sum()

    histogram = histogram[
        histogram > 0
    ]

    entropy = -np.sum(
        histogram * np.log2(histogram)
    )

    return float(entropy)


entropy_ct = calculate_entropy(ct_image)
entropy_mri = calculate_entropy(mri_image)
entropy_fused = calculate_entropy(enhanced_fused)


# ------------------------------------------------------------
# 13. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("                 FUSION RESULTS")
print("=" * 60)

print("\nInput Images")
print("-" * 60)

print("CT image shape       :", ct_image.shape)
print("MRI image shape      :", mri_image.shape)

print("\nSSIM")
print("-" * 60)

print("SSIM - CT vs Fused   :", round(ssim_ct, 4))
print("SSIM - MRI vs Fused  :", round(ssim_mri, 4))

print("\nPSNR")
print("-" * 60)

print("PSNR - CT vs Fused   :", round(psnr_ct, 4), "dB")
print("PSNR - MRI vs Fused  :", round(psnr_mri, 4), "dB")

print("\nInformation Entropy")
print("-" * 60)

print("Entropy - CT         :", round(entropy_ct, 4))
print("Entropy - MRI        :", round(entropy_mri, 4))
print("Entropy - Fused      :", round(entropy_fused, 4))

print("\nOutput")
print("-" * 60)

print("Fused image           :", FUSED_PATH)

print("\n" + "=" * 60)
print("       DWT IMAGE FUSION COMPLETED SUCCESSFULLY")
print("=" * 60)


# ------------------------------------------------------------
# 14. SHOW IMAGES
# ------------------------------------------------------------

cv2.imshow(
    "CT Image",
    ct_image
)

cv2.imshow(
    "MRI Image",
    mri_image
)

cv2.imshow(
    "DWT Fused Image",
    enhanced_fused
)

print("\nPress any key inside an image window to close it.")

cv2.waitKey(0)
cv2.destroyAllWindows()