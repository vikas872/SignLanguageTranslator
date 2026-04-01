# Real-Time Sign Language Translator 🤟

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)

A Computer Vision project that translates American Sign Language (ASL) alphabets in real-time using a laptop webcam. Built with a custom Convolutional Neural Network (CNN) in PyTorch and OpenCV.

## 🚀 Features
*   **Custom CNN Architecture:** Trained on the Sign Language MNIST dataset for high-speed CPU inference.
*   **Real-Time Processing:** Uses `cv2` to capture frames, isolate the user's hand using a Region of Interest (ROI) box, and perform live prediction at 30+ FPS.
*   **End-to-End ML Pipeline:** Includes automatic dataset downloading using `kagglehub`, dataset loading, augmentation, training, and inference scripts.

## 🛠️ Setup Instructions

1.  **Clone the repository (or navigate to the folder)**
2.  **Create a Virtual Environment & Install Dependencies:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run Training (Optional):**
    Pre-trained weights (`asl_cnn_model.pth`) are included, but you can retrain the model from scratch by executing:
    ```bash
    python train.py
    ```
    *Note: This script requires an active internet connection to automatically download the Kaggle ASL MNIST dataset. It will train the model for 5 epochs and save the new weights.* 

4.  **Run Real-Time Inference (How to Use):**
    To start the webcam and begin translating, run the following command:
    ```bash
    python inference.py
    ```
    **Usage Steps:**
    *   A window will open displaying your webcam feed.
    *   Place your right hand completely inside the **green bounding box** visible on the screen.
    *   Make an ASL alphabet sign (A-Y, excluding J as it requires motion).
    *   The model's live prediction will appear in green text right above the box.
    *   Press the `q` key on your keyboard to safely close the camera and quit the application.

## 📂 Project Structure
*   `dataset.py`: PyTorch `Dataset` wrapper that uses `kagglehub` to fetch and parse the CSV pixel data into PIL Images.
*   `model.py`: The lightweight `ASLNet` CNN architecture.
*   `train.py`: The training loop with data augmentations to improve robustness.
*   `inference.py`: Live OpenCV webcam inference pipeline.
*   `requirements.txt`: Python dependencies.
