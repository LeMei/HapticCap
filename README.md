# 📌 HapticCap: A Multimodal Dataset and Task for Understanding User Experience of Vibration Haptic Signals

Arxiv: https://arxiv.org/pdf/2507.13318? (Findings of EMNLP2025) 

---

## 📖 Introduction
**HapticCap** is a multimodal dataset and benchmark task designed for understanding **user experience of vibration-based haptic signals**.  
It provides a new resource for research at the intersection of **haptics, text, and multimodal learning**.

---

## 📂 Dataset
- **Modality**: Vibration haptic signals, paired with textual annotations  
- **Textual Annotations**:  
  - Sensory: It refers to physical attributes (e.g.,intensity of tapping).
  - Emotional: It refers to emotional denotes affective impressions (e.g., the mood of a scene).
  - Associative: It indicates real-world familiar experiences (e.g., buzzing of a bee, a heartbeat).
 
- **Format**: Signals stored as time-series data; annotations in JSON with haptic signal ID  
- **Scale**:

<p align="center">
  <img width="304" height="334" alt="image" src="https://github.com/user-attachments/assets/9debe339-c83e-4e3d-b173-99d004b0d1b6" />
</p>

---

Google drive:

Haptic Vibration Signals: <https://drive.google.com/drive/folders/1xylMC-EFswTc3adcc6rAzyFsXLSmVweg?usp=drive_link>

Human Descriptions: https: <https://drive.google.com/drive/folders/1ovlIbfJecXAq0TbItmrRl5dVV7OCCQzB?usp=drive_link>

or find the data in: <https://huggingface.co/datasets/GuiminHu/HapticCap>


## 🧩 Tasks
- Haptic-caption retrieval: Its objective is to retrieve the textual descriptions of three categories that correspond to a given haptic signal, using the
haptic signal as the query and the descriptions as the target documents.

- Training, Valid Test set:

<https://drive.google.com/drive/folders/1PfM2fjIHFDx1PtWADJHwo3TM2SRbp2tL?usp=drive_link>

## 🧩 Models
We design supervised contrastive learning framework that aims to pull the clusters of points belonging to the same class together in an embedding space and simultaneously
pushes apart clusters of samples from different classes. 

<p align="center">
<img width="610" height="244" alt="image" src="https://github.com/user-attachments/assets/781be7dc-3674-401d-8d41-86f07ab1b205" />
</p>

---

## 🚀 Citation
If you find this dataset useful for your research, please cite our paper:

```bibtex
@article{hu2025hapticcap,
title={Hapticcap: A multimodal dataset and task for understanding user experience of vibration haptic signals},
author={Hu, Guimin and Hershcovich, Daniel and Seifi, Hasti},
journal={arXiv preprint arXiv:2507.13318},
year={2025}
}
```

  







