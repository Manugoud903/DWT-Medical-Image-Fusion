# DWT Medical Image Fusion

DEPLOYED ON RENDER :
https://dwt-medical-image-fusion.onrender.com/
DWT-based CT and MRI medical image fusion using Python, Flask, and PyWavelets.

DWT-Based Medical Image Fusion

Project Overview

DWT-Based Medical Image Fusion is a web application that combines CT and MRI medical images into a single fused image using the Discrete Wavelet Transform (DWT).

The project aims to combine useful information from both images to produce a fused image that can be analyzed more easily.

Problem Statement

CT and MRI images provide different types of medical information. Viewing them separately may make it difficult to observe all relevant details together. Medical image fusion combines information from both images into one image.

Objectives

- To combine CT and MRI images.
- To implement image fusion using DWT.
- To preserve important image details.
- To calculate image quality metrics.
- To develop a user-friendly web application.

Methodology

The project follows these steps:

1. Upload CT and MRI images.
2. Preprocess the input images.
3. Apply the Discrete Wavelet Transform (DWT).
4. Separate the images into wavelet coefficients.
5. Fuse the corresponding coefficients.
6. Apply the inverse DWT.
7. Display the fused image.

DWT Components

DWT separates an image into four frequency sub-bands:

- LL: Low-frequency information.
- LH: Horizontal detail information.
- HL: Vertical detail information.
- HH: Diagonal detail information.

Fusion Technique

- Wavelet: Daubechies 2 (db2).
- LL coefficients: Average the corresponding coefficients.
- LH, HL, HH coefficients: Select the coefficient with the larger absolute value.
- Reconstruction: Apply the inverse DWT to obtain the fused image.

Note: Confirm these fusion rules against the actual project source code.

Technologies Used

- Python
- Flask
- NumPy
- OpenCV
- PyWavelets
- HTML
- CSS
- Render.com

Features

- Upload CT and MRI images.
- Fuse images using DWT.
- Display the fused output.
- Evaluate image quality using metrics.
- Access the application through a web browser.

Evaluation Metrics

1. PSNR

Peak Signal-to-Noise Ratio measures image reconstruction quality.

2. SSIM

Structural Similarity Index measures the similarity of image structures.

3. Entropy

Entropy measures the information content of an image.

Actual metric values should be added after running the project and evaluating its output.

How to Run Locally

Step 1: Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

Step 2: Open the project folder

cd YOUR_PROJECT_FOLDER

Step 3: Install dependencies

pip install -r requirements.txt

Step 4: Run the application

python app.py

Step 5: Open in your browser

Visit:

http://127.0.0.1:5000

Make sure the commands and filenames match your project.

Deployment

The application is deployed on Render.com.

Live Demo: YOUR_RENDER_DEPLOYMENT_URL

Future Enhancements

- Improve image preprocessing.
- Support additional medical image formats.
- Compare different wavelet techniques.
- Improve the user interface.
- Add more image quality metrics.

Applications

- Medical image processing research.
- Image enhancement and analysis.
- Academic learning and experimentation.

Limitations

This project is developed for educational and research purposes. The fused images are not intended to replace professional medical interpretation or clinical diagnosis.

Author

Name: YOUR NAME
Department: Computer Science and Engineering (AI & ML)
Project: DWT-Based Medical Image Fusion

License

This project is intended for educational and research purposes.
