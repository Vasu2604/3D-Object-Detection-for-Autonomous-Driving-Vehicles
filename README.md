# Final Project: 3D Object Detection for Intelligent Autonomous Vehicles

## 📖 Project Overview

This repository contains the source code and documentation for the **Final Project** of the **Intelligent Autonomous Vehicles** course. The primary objective of this project is to implement robust 3D object detection suited for autonomous driving scenarios.

We utilized the **MMDetection3D** framework and specifically adapted the **CenterPoint** architecture. While we leveraged a pre-trained model provided by the original authors to establish a strong baseline, we performed significant modifications and fine-tuning to tailor the system to our specific project requirements and the **nuScenes** dataset.

## 🚀 Key Features

*   **Advanced 3D Detection**: Implementation of the Voxel-based CenterPoint model, known for its balance of speed and accuracy in detecting objects in 3D space.
*   **Custom Adaptation**: The model has been adapted from the original pre-trained weights. We successfully loaded these weights and adjusted the configuration to optimize performance for our specific detection tasks (cars, pedestrians, etc.).
*   **Interactive Visualization**: A custom-built inference script (`demo_inference.py`) that provides dual-perspective visualizations:
    *   **Bird's Eye View (BEV)**: A clear, top-down map of the surroundings showing object positions and orientations.
    *   **3D Perspective View**: An immersive 3D point cloud rendering that visualizes the spatial relationship of detected objects.
*   **Real-World Applicability**: The system is designed to process LiDAR point cloud data, mimicking real-world autonomous vehicle sensor inputs.

## 🛠️ Technical Implementation

### Model Architecture
We selected **CenterPoint** as our core architecture due to its center-based approach to 3D detection, which simplifies the regression task compared to anchor-based methods.

### Modifications
Starting with the official pre-trained model, we:
1.  Analyzed the model structure to ensure compatibility with our specific version of MMDetection3D.
2.  Fine-tuned the inference pipeline to handle our data inputs correctly.
3.  Developed a specialized visualization tool to better interpret the model's output for demonstration purposes.

## 💻 Installation & Setup

To replicate our results or run the demo, please follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Vasu2604/3D-Object-Detection-for-Autonomous-Driving-Vehicles.git
    cd 3D-Object-Detection-for-Autonomous-Driving-Vehicles
    ```

2.  **Environment Setup**
    This project relies on `MMDetection3D`. Please refer to the [official installation guide](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) for detailed instructions on setting up the environment with PyTorch and CUDA.

3.  **Install Dependencies**
    ```bash
    pip install -v -e .
    ```

## 🏃‍♂️ Running the Demo

We have provided a streamlined script to demonstrate the model's capabilities.

1.  **Navigate to the Source Directory**
    ```bash
    cd mmdetection3d
    ```

2.  **Execute Inference**
    ```bash
    python demo_inference.py
    ```

### Expected Output
The script will process a sample LiDAR point cloud and prompt the model to detect objects. The output will be saved as a high-resolution image (`demo_output/real_model_inference.png`) displaying both the Bird's Eye View and the 3D Perspective of the scene.

## 📁 Repository Structure

*   `configs/`: Configuration files defining the model architecture and training parameters.
*   `mmdet3d/`: Core codebase containing the model definitions and detection logic.
*   `tools/`: Utility scripts for training, testing, and file handling.
*   `demo/`: Contains demonstration assets.
*   `data/`: Directory structure for dataset organization.

## 🙏 Acknowledgements

We would like to thank the authors of **MMDetection3D** and **CenterPoint** for making their code and pre-trained models open source. Their work provided the foundation upon which this final project was built.
