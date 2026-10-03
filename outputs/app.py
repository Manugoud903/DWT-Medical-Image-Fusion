from flask import Flask, render_template, request, send_file, send_from_directory
import cv2
import numpy as np
import pywt
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


app = Flask(__name__)

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# -----------------------------
# Image Normalization
# -----------------------------
def normalize(image):
    return cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)


# -----------------------------
# DWT Image Fusion
# -----------------------------
def fuse_images(ct, mri):

    # Resize MRI to CT dimensions
    mri = cv2.resize(
        mri,
        (ct.shape[1], ct.shape[0])
    )

    # Normalize images
    ct = normalize(ct)
    mri = normalize(mri)

    # Perform 2D DWT
    ct_coeff = pywt.dwt2(
        ct.astype(float),
        "db2"
    )

    mri_coeff = pywt.dwt2(
        mri.astype(float),
        "db2"
    )

    # Separate coefficients
    ct_LL, (ct_LH, ct_HL, ct_HH) = ct_coeff
    mri_LL, (mri_LH, mri_HL, mri_HH) = mri_coeff

    # Low-frequency fusion
    fused_LL = (
        ct_LL + mri_LL
    ) / 2

    # High-frequency fusion
    fused_LH = np.where(
        np.abs(ct_LH) >= np.abs(mri_LH),
        ct_LH,
        mri_LH
    )

    fused_HL = np.where(
        np.abs(ct_HL) >= np.abs(mri_HL),
        ct_HL,
        mri_HL
    )

    fused_HH = np.where(
        np.abs(ct_HH) >= np.abs(mri_HH),
        ct_HH,
        mri_HH
    )

    # Inverse DWT
    fused = pywt.idwt2(
        (
            fused_LL,
            (
                fused_LH,
                fused_HL,
                fused_HH
            )
        ),
        "db2"
    )

    # Convert to valid image range
    fused = np.clip(
        fused,
        0,
        255
    ).astype(np.uint8)

    return ct, mri, fused


# -----------------------------
# Entropy Calculation
# -----------------------------
def calculate_entropy(image):

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


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template(
        "index.html"
    )


# -----------------------------
# Fusion Processing
# -----------------------------
@app.route("/fuse", methods=["POST"])
def fuse():

    ct_file = request.files.get("ct")
    mri_file = request.files.get("mri")

    # Check uploads
    if not ct_file or not mri_file:
        return "Please upload both CT and MRI images."

    # Read CT image
    ct_array = np.frombuffer(
        ct_file.read(),
        np.uint8
    )

    # Read MRI image
    mri_array = np.frombuffer(
        mri_file.read(),
        np.uint8
    )

    # Decode images
    ct = cv2.imdecode(
        ct_array,
        cv2.IMREAD_GRAYSCALE
    )

    mri = cv2.imdecode(
        mri_array,
        cv2.IMREAD_GRAYSCALE
    )

    # Check image validity
    if ct is None or mri is None:
        return "Invalid image file."

    # Perform DWT fusion
    ct, mri, fused = fuse_images(
        ct,
        mri
    )

    # Save fused image
    fused_path = os.path.join(
        OUTPUT_FOLDER,
        "fused_image.png"
    )

    cv2.imwrite(
        fused_path,
        fused
    )

    # Save processed CT
    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "ct_display.png"
        ),
        ct
    )

    # Save processed MRI
    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "mri_display.png"
        ),
        mri
    )

    # -----------------------------
    # Quality Metrics
    # -----------------------------

    ssim_ct = ssim(
        ct,
        fused,
        data_range=255
    )

    ssim_mri = ssim(
        mri,
        fused,
        data_range=255
    )

    psnr_ct = psnr(
        ct,
        fused,
        data_range=255
    )

    psnr_mri = psnr(
        mri,
        fused,
        data_range=255
    )

    entropy = calculate_entropy(
        fused
    )

    # Display result page
    return render_template(
        "result.html",

        ssim_ct=round(
            ssim_ct,
            4
        ),

        ssim_mri=round(
            ssim_mri,
            4
        ),

        psnr_ct=round(
            psnr_ct,
            2
        ),

        psnr_mri=round(
            psnr_mri,
            2
        ),

        entropy=round(
            entropy,
            4
        )
    )


# -----------------------------
# Serve Output Images
# -----------------------------
@app.route("/outputs/<path:filename>")
def output_file(filename):

    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


# -----------------------------
# Download Fused Image
# -----------------------------
@app.route("/download")
def download():

    return send_file(
        os.path.join(
            OUTPUT_FOLDER,
            "fused_image.png"
        ),

        as_attachment=True,

        download_name="DWT_Fused_Medical_Image.png"
    )


# -----------------------------
# Run Application
# -----------------------------
if __name__ == "__main__":

    print()
    print("====================================")
    print("     DWT MEDICAL IMAGE FUSION APP")
    print("====================================")
    print("Open: http://127.0.0.1:5000")
    print("====================================")
    print()

    app.run(
        debug=True
    )