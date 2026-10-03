import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import pywt
import os
import math


class DWTMedicalFusionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("DWT-Based Medical Image Fusion")
        self.root.geometry("1200x750")
        self.root.configure(bg="#101820")

        self.ct_path = None
        self.mri_path = None
        self.fused_image = None

        self.ct_photo = None
        self.mri_photo = None
        self.fused_photo = None

        self.create_interface()

    # ---------------------------------------------------------
    # USER INTERFACE
    # ---------------------------------------------------------

    def create_interface(self):

        title = tk.Label(
            self.root,
            text="DWT-BASED MEDICAL IMAGE FUSION",
            font=("Arial", 24, "bold"),
            fg="#00e5ff",
            bg="#101820"
        )
        title.pack(pady=15)

        subtitle = tk.Label(
            self.root,
            text="Discrete Wavelet Transform • Multi-Scale Decomposition • Medical Image Analysis",
            font=("Arial", 11),
            fg="white",
            bg="#101820"
        )
        subtitle.pack()

        main_frame = tk.Frame(
            self.root,
            bg="#101820"
        )
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # -----------------------------------------------------
        # CONTROL PANEL
        # -----------------------------------------------------

        control_frame = tk.Frame(
            main_frame,
            bg="#182631",
            width=250
        )
        control_frame.pack(
            side="left",
            fill="y",
            padx=(0, 15)
        )

        tk.Label(
            control_frame,
            text="CONTROL PANEL",
            font=("Arial", 17, "bold"),
            fg="#00e5ff",
            bg="#182631"
        ).pack(pady=20)

        tk.Label(
            control_frame,
            text="INPUT IMAGES",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#182631"
        ).pack(pady=10)

        self.ct_button = tk.Button(
            control_frame,
            text="Select CT Image",
            command=self.select_ct,
            font=("Arial", 11, "bold"),
            bg="#243b4a",
            fg="white",
            width=22,
            height=2
        )
        self.ct_button.pack(pady=8)

        self.mri_button = tk.Button(
            control_frame,
            text="Select MRI Image",
            command=self.select_mri,
            font=("Arial", 11, "bold"),
            bg="#243b4a",
            fg="white",
            width=22,
            height=2
        )
        self.mri_button.pack(pady=8)

        self.ct_status = tk.Label(
            control_frame,
            text="CT: Not selected",
            fg="white",
            bg="#182631",
            wraplength=220
        )
        self.ct_status.pack(pady=5)

        self.mri_status = tk.Label(
            control_frame,
            text="MRI: Not selected",
            fg="white",
            bg="#182631",
            wraplength=220
        )
        self.mri_status.pack(pady=5)

        tk.Label(
            control_frame,
            text="FUSION PROCESS",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#182631"
        ).pack(pady=(25, 10))

        self.fusion_button = tk.Button(
            control_frame,
            text="PERFORM DWT FUSION",
            command=self.perform_fusion,
            font=("Arial", 11, "bold"),
            bg="#00c8d7",
            fg="white",
            width=22,
            height=2
        )
        self.fusion_button.pack(pady=8)

        tk.Label(
            control_frame,
            text="OUTPUT",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#182631"
        ).pack(pady=(25, 10))

        self.save_button = tk.Button(
            control_frame,
            text="Save Fused Image",
            command=self.save_fused_image,
            font=("Arial", 11, "bold"),
            bg="#243b4a",
            fg="white",
            width=22,
            height=2,
            state="disabled"
        )
        self.save_button.pack(pady=8)

        # -----------------------------------------------------
        # IMAGE AREA
        # -----------------------------------------------------

        image_frame = tk.Frame(
            main_frame,
            bg="#101820"
        )
        image_frame.pack(
            side="left",
            fill="both",
            expand=True
        )

        # CT
        ct_frame = tk.Frame(
            image_frame,
            bg="#182631"
        )
        ct_frame.grid(
            row=0,
            column=0,
            padx=8,
            pady=8,
            sticky="nsew"
        )

        tk.Label(
            ct_frame,
            text="CT IMAGE",
            font=("Arial", 14, "bold"),
            fg="#00e5ff",
            bg="#182631"
        ).pack(pady=8)

        self.ct_label = tk.Label(
            ct_frame,
            text="No image",
            fg="white",
            bg="#182631",
            width=30,
            height=15
        )
        self.ct_label.pack(
            padx=10,
            pady=10
        )

        # MRI
        mri_frame = tk.Frame(
            image_frame,
            bg="#182631"
        )
        mri_frame.grid(
            row=0,
            column=1,
            padx=8,
            pady=8,
            sticky="nsew"
        )

        tk.Label(
            mri_frame,
            text="MRI IMAGE",
            font=("Arial", 14, "bold"),
            fg="#00e5ff",
            bg="#182631"
        ).pack(pady=8)

        self.mri_label = tk.Label(
            mri_frame,
            text="No image",
            fg="white",
            bg="#182631",
            width=30,
            height=15
        )
        self.mri_label.pack(
            padx=10,
            pady=10
        )

        # FUSED
        fused_frame = tk.Frame(
            image_frame,
            bg="#182631"
        )
        fused_frame.grid(
            row=0,
            column=2,
            padx=8,
            pady=8,
            sticky="nsew"
        )

        tk.Label(
            fused_frame,
            text="FUSED IMAGE",
            font=("Arial", 14, "bold"),
            fg="#00e5ff",
            bg="#182631"
        ).pack(pady=8)

        self.fused_label = tk.Label(
            fused_frame,
            text="No image",
            fg="white",
            bg="#182631",
            width=30,
            height=15
        )
        self.fused_label.pack(
            padx=10,
            pady=10
        )

        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_columnconfigure(1, weight=1)
        image_frame.grid_columnconfigure(2, weight=1)
        image_frame.grid_rowconfigure(0, weight=1)

        # -----------------------------------------------------
        # METRICS
        # -----------------------------------------------------

        metrics_frame = tk.Frame(
            self.root,
            bg="#182631"
        )
        metrics_frame.pack(
            fill="x",
            padx=20,
            pady=10
        )

        tk.Label(
            metrics_frame,
            text="IMAGE QUALITY METRICS",
            font=("Arial", 15, "bold"),
            fg="#00e5ff",
            bg="#182631"
        ).pack(pady=8)

        metric_values = tk.Frame(
            metrics_frame,
            bg="#182631"
        )
        metric_values.pack(pady=5)

        self.ssim_value = self.create_metric(
            metric_values,
            "SSIM"
        )

        self.psnr_value = self.create_metric(
            metric_values,
            "PSNR"
        )

        self.entropy_value = self.create_metric(
            metric_values,
            "ENTROPY"
        )

        self.size_value = self.create_metric(
            metric_values,
            "IMAGE SIZE"
        )

    def create_metric(self, parent, name):

        frame = tk.Frame(
            parent,
            bg="#182631",
            width=180
        )
        frame.pack(
            side="left",
            padx=25
        )

        tk.Label(
            frame,
            text=name,
            font=("Arial", 11, "bold"),
            fg="white",
            bg="#182631"
        ).pack()

        value = tk.Label(
            frame,
            text="—",
            font=("Arial", 12, "bold"),
            fg="#00e5ff",
            bg="#182631"
        )
        value.pack(pady=4)

        return value

    # ---------------------------------------------------------
    # SELECT CT
    # ---------------------------------------------------------

    def select_ct(self):

        path = filedialog.askopenfilename(
            title="Select CT Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )

        if path:

            self.ct_path = path

            self.ct_status.config(
                text="CT: " + os.path.basename(path)
            )

            try:
                image = Image.open(path).convert("L")

                self.display_image(
                    image,
                    self.ct_label,
                    "ct"
                )

            except Exception as e:

                messagebox.showerror(
                    "CT Image Error",
                    "Unable to open CT image.\n\n" + str(e)
                )

    # ---------------------------------------------------------
    # SELECT MRI
    # ---------------------------------------------------------

    def select_mri(self):

        path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )

        if path:

            self.mri_path = path

            self.mri_status.config(
                text="MRI: " + os.path.basename(path)
            )

            try:
                image = Image.open(path).convert("L")

                self.display_image(
                    image,
                    self.mri_label,
                    "mri"
                )

            except Exception as e:

                messagebox.showerror(
                    "MRI Image Error",
                    "Unable to open MRI image.\n\n" + str(e)
                )

    # ---------------------------------------------------------
    # DISPLAY IMAGE
    # ---------------------------------------------------------

    def display_image(self, image, label, image_type):

        try:

            display_image = image.copy()

            display_image.thumbnail(
                (330, 330),
                Image.Resampling.LANCZOS
            )

            photo = ImageTk.PhotoImage(
                display_image
            )

            label.config(
                image=photo,
                text=""
            )

            label.image = photo

            if image_type == "ct":
                self.ct_photo = photo

            elif image_type == "mri":
                self.mri_photo = photo

            elif image_type == "fused":
                self.fused_photo = photo

        except Exception as e:

            messagebox.showerror(
                "Display Error",
                str(e)
            )

    # ---------------------------------------------------------
    # DWT FUSION
    # ---------------------------------------------------------

    def perform_fusion(self):

        if self.ct_path is None or self.mri_path is None:

            messagebox.showwarning(
                "Missing Images",
                "Please select both CT and MRI images first."
            )

            return

        try:

            # Read images
            ct_image = Image.open(
                self.ct_path
            ).convert("L")

            mri_image = Image.open(
                self.mri_path
            ).convert("L")

            # Use CT size as reference
            target_size = ct_image.size

            mri_image = mri_image.resize(
                target_size,
                Image.Resampling.LANCZOS
            )

            # Convert to numpy
            ct = np.array(
                ct_image,
                dtype=np.float32
            )

            mri = np.array(
                mri_image,
                dtype=np.float32
            )

            # Normalize
            ct = self.normalize_image(ct)
            mri = self.normalize_image(mri)

            # -------------------------------------------------
            # DWT DECOMPOSITION
            # -------------------------------------------------

            ct_coeff = pywt.dwt2(
                ct,
                "haar"
            )

            mri_coeff = pywt.dwt2(
                mri,
                "haar"
            )

            ct_LL, (ct_LH, ct_HL, ct_HH) = ct_coeff
            mri_LL, (mri_LH, mri_HL, mri_HH) = mri_coeff

            # -------------------------------------------------
            # FUSION RULE
            # -------------------------------------------------

            fused_LL = (
                0.5 * ct_LL +
                0.5 * mri_LL
            )

            fused_LH = self.select_max(
                ct_LH,
                mri_LH
            )

            fused_HL = self.select_max(
                ct_HL,
                mri_HL
            )

            fused_HH = self.select_max(
                ct_HH,
                mri_HH
            )

            # -------------------------------------------------
            # INVERSE DWT
            # -------------------------------------------------

            fused = pywt.idwt2(
                (
                    fused_LL,
                    (
                        fused_LH,
                        fused_HL,
                        fused_HH
                    )
                ),
                "haar"
            )

            # Normalize result
            fused = self.normalize_image(
                fused
            )

            fused_uint8 = np.uint8(
                np.clip(
                    fused * 255,
                    0,
                    255
                )
            )

            self.fused_image = Image.fromarray(
                fused_uint8
            )

            # Display fused image
            self.display_image(
                self.fused_image,
                self.fused_label,
                "fused"
            )

            # Calculate metrics
            self.calculate_metrics(
                ct,
                mri,
                fused
            )

            self.save_button.config(
                state="normal"
            )

            messagebox.showinfo(
                "Fusion Completed",
                "DWT-based image fusion completed successfully!"
            )

        except Exception as e:

            messagebox.showerror(
                "Fusion Error",
                "An error occurred during fusion:\n\n"
                + str(e)
            )

    # ---------------------------------------------------------
    # NORMALIZE IMAGE
    # ---------------------------------------------------------

    def normalize_image(self, image):

        image = np.nan_to_num(
            image
        )

        minimum = np.min(image)
        maximum = np.max(image)

        if maximum == minimum:
            return np.zeros_like(
                image,
                dtype=np.float32
            )

        return (
            image - minimum
        ) / (
            maximum - minimum
        )

    # ---------------------------------------------------------
    # SELECT MAXIMUM ABSOLUTE COEFFICIENT
    # ---------------------------------------------------------

    def select_max(self, image1, image2):

        mask = (
            np.abs(image1)
            >=
            np.abs(image2)
        )

        return np.where(
            mask,
            image1,
            image2
        )

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    def calculate_metrics(
        self,
        ct,
        mri,
        fused
    ):

        try:

            # SSIM
            ssim_ct = self.simple_ssim(
                ct,
                fused
            )

            ssim_mri = self.simple_ssim(
                mri,
                fused
            )

            ssim = (
                ssim_ct +
                ssim_mri
            ) / 2

            self.ssim_value.config(
                text=f"{ssim:.4f}"
            )

            # PSNR
            psnr_ct = self.calculate_psnr(
                ct,
                fused
            )

            psnr_mri = self.calculate_psnr(
                mri,
                fused
            )

            psnr = (
                psnr_ct +
                psnr_mri
            ) / 2

            self.psnr_value.config(
                text=f"{psnr:.2f} dB"
            )

            # Entropy
            entropy = self.calculate_entropy(
                fused
            )

            self.entropy_value.config(
                text=f"{entropy:.4f}"
            )

            # Image size
            height, width = fused.shape

            self.size_value.config(
                text=f"{width} × {height}"
            )

        except Exception as e:

            self.ssim_value.config(
                text="N/A"
            )

            self.psnr_value.config(
                text="N/A"
            )

            self.entropy_value.config(
                text="N/A"
            )

            self.size_value.config(
                text="N/A"
            )

    # ---------------------------------------------------------
    # SIMPLE SSIM
    # ---------------------------------------------------------

    def simple_ssim(self, image1, image2):

        image1 = image1.astype(
            np.float64
        )

        image2 = image2.astype(
            np.float64
        )

        mean1 = np.mean(image1)
        mean2 = np.mean(image2)

        variance1 = np.var(image1)
        variance2 = np.var(image2)

        covariance = np.mean(
            (
                image1 - mean1
            ) *
            (
                image2 - mean2
            )
        )

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        numerator = (
            (2 * mean1 * mean2 + c1) *
            (2 * covariance + c2)
        )

        denominator = (
            (mean1 ** 2 + mean2 ** 2 + c1) *
            (variance1 + variance2 + c2)
        )

        if denominator == 0:
            return 0

        return numerator / denominator

    # ---------------------------------------------------------
    # PSNR
    # ---------------------------------------------------------

    def calculate_psnr(
        self,
        original,
        processed
    ):

        mse = np.mean(
            (
                original -
                processed
            ) ** 2
        )

        if mse == 0:
            return 100

        return 10 * math.log10(
            1 / mse
        )

    # ---------------------------------------------------------
    # ENTROPY
    # ---------------------------------------------------------

    def calculate_entropy(
        self,
        image
    ):

        image_uint8 = np.uint8(
            np.clip(
                image * 255,
                0,
                255
            )
        )

        histogram = np.bincount(
            image_uint8.flatten(),
            minlength=256
        )

        probability = (
            histogram /
            np.sum(histogram)
        )

        probability = probability[
            probability > 0
        ]

        entropy = -np.sum(
            probability *
            np.log2(probability)
        )

        return entropy

    # ---------------------------------------------------------
    # SAVE FUSED IMAGE
    # ---------------------------------------------------------

    def save_fused_image(self):

        if self.fused_image is None:

            messagebox.showwarning(
                "No Fused Image",
                "Please perform fusion first."
            )

            return

        path = filedialog.asksaveasfilename(
            title="Save Fused Image",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("All Files", "*.*")
            ]
        )

        if path:

            try:

                self.fused_image.save(
                    path
                )

                messagebox.showinfo(
                    "Saved",
                    "Fused image saved successfully!"
                )

            except Exception as e:

                messagebox.showerror(
                    "Save Error",
                    str(e)
                )


# =============================================================
# MAIN PROGRAM
# =============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = DWTMedicalFusionApp(
        root
    )

    root.mainloop()