# Point Cloud Adversarial Examples by Optimizing Point Intensity Values

### Authors:
- Kondreddy Rohith Sai Reddy, University of Florida
- Satya Aakash Chowdary Obellaneni, University of Florida
- Kushi Vardhan Reddy Pasham, University of Florida
- Vivek Reddy Gangula, University of Florida

## 1. Introduction
This project focuses on securing point cloud-based object detection systems, particularly those used in autonomous vehicles. The research investigates the vulnerabilities of these systems to intensity-based adversarial attacks, where the intensity values of points in 3D point clouds are manipulated to mislead detection models.

## 2. Background and Related Work
### Point Cloud-based Object Detection:
- Point cloud-based object detection is essential for various applications like autonomous vehicles, drones, virtual reality, and mapping.
- Traditional methods manipulate spatial components of point clouds to create adversarial examples. This research explores novel intensity-based manipulations.

### Adversarial Attacks:
- Adversarial attacks are designed to cause incorrect model outputs by manipulating input data. In point cloud-based systems, these can cause models to misinterpret or overlook critical objects.

## 3. Methodology
### Threat Model
- The research identifies vulnerabilities in LiDAR data analysis for autonomous vehicles. It assumes attackers can alter point cloud data and possess black-box access to detection models.

### Data Preparation
- The KITTI dataset, containing LiDAR-generated 3D point clouds, is used. Each data point includes spatial coordinates (x, y, z) and an intensity value.

### Initial Model Training
- The PointRCNN model, tailored for 3D object detection, is trained using the KITTI dataset. It segments the point cloud into foreground and background and refines object proposals based on spatial and intensity attributes.

### Object Selection using Open3D
- Open3D is used to identify and isolate target objects within the point clouds. This enables analysis of how PointRCNN responds to localized adversarial changes.

### Intensity Manipulation and Evaluation
- The Iterative Gradient Method manipulates the intensity values of selected objects to create adversarial examples. The modified point cloud is re-evaluated using the PointRCNN model to assess the impact on confidence levels.

## 4. Experimental Results
- Detection accuracy varies across different distances and iterations. For example, accuracy for cyclists, pedestrians, and cars decreases with distance. Despite iterative attacks, the model's confidence remains relatively stable.

## 5. Analysis and Discussion
- Intensity-based attacks had minimal impact on the model's confidence scores and classification outcomes. The research highlights the resilience of deep neural networks against such attacks and suggests future work to incorporate both spatial and intensity perturbations for more effective adversarial evaluations.

## References
1. Yulong Cao et al. Adversarial sensor attack on lidar-based perception in autonomous driving. 2019.
2. Xiaozhi Chen et al. Multi-view 3d object detection network for autonomous driving. 2017.
3. Loic Landrieu et al. Large-scale point cloud semantic segmentation with superpoint graphs. 2018.
4. Xinke Li et al. Pointba: Towards backdoor attacks in 3d point cloud. 2021.
5. Daniel Liu et al. Extending adversarial attacks and defenses to deep 3d point cloud classifiers. 2019.
6. Charles R. Qi et al. Frustum pointnets for 3d object detection from rgb-d data. 2018.
7. Charles R. Qi et al. Pointnet: Deep learning on point sets for 3d classification and segmentation. 2017.
8. Shaoshuai Shi et al. Pointrcnn: 3d object proposal generation and detection from point cloud. 2019.
9. Chong Xiang et al. Generating 3d adversarial point clouds. 2019.
10. Hengshuang Zhao et al. Pointweb: Enhancing local neighborhood features for point cloud processing. 2019.
11. Yin Zhou et al. Voxelnet: End-to-end learning for point cloud based 3d object detection. 2018.

## Structure of the Repository
project-root/
│
├── README.md # This file
├── data/ # Directory containing the KITTI dataset
├── models/ # Trained models and model checkpoints
├── src/ # Source code for the project
│ ├── data_preparation.py # Script for data preparation
│ ├── train.py # Script for model training
│ ├── attack.py # Script for intensity manipulation
│ └── evaluate.py # Script for evaluating model performance
└── results/ # Directory to store results and visualizations


## How to Run the Project
1. **Data Preparation:**
   - Download the KITTI dataset and place it in the `data/` directory.
   - Run `data_preparation.py` to preprocess the data.

2. **Model Training:**
   - Train the PointRCNN model using `train.py`.

3. **Adversarial Attack:**
   - Use `attack.py` to generate adversarial examples by manipulating intensity values.

4. **Evaluation:**
   - Run `evaluate.py` to assess the impact of adversarial attacks on the model's performance.

## Dependencies
- Python 3.8+
- Open3D
- PyTorch
- NumPy
- KITTI Dataset
- Other dependencies specified in `requirements.txt`

## Acknowledgements
We thank the University of Florida for providing resources and support for this research.
