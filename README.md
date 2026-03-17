# Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration

<p align="center">
  <a href="https://scholar.google.com/citations?hl=en&user=46m3WScAAAAJ">Yuetong Liu</a>,
  <a href="https://scholar.google.com/citations?hl=en&user=SdJX4nAAAAAJ">Yunqiu Xu</a>,
  <a href="https://scholar.google.com/citations?hl=en&user=CX6eDWUAAAAJ">Yang Wei</a>,
  <a href="https://scholar.google.com/citations?hl=en&user=1Ezgfw8AAAAJ">Xiuli Bi</a>,
  <a href="https://ieeexplore.ieee.org/author/37586970100">Bin Xiao</a>
</p>


 
<p align="center">
  <a href="https://arxiv.org/abs/2505.16479">
    <img src="https://img.shields.io/badge/Paper-Arxiv-red" alt="Paper">
  </a>
  
  <a href="https://henlyta.github.io/ClearNight/index.html">
    <img src="https://img.shields.io/badge/Project-Webpage-blue" alt="Project">
  </a>
  
  <a href="https://huggingface.co/datasets/YuetongLiu/AllWeatherNight">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow" alt="Dataset">
  </a>
</p>


## 📝 Abstract

Restoring nighttime images affected by multiple adverse weather conditions is a practical yet under-explored research problem, as multiple weather degradations usually coexist in the real world alongside various lighting effects at night. This paper first explores the challenging multi-weather nighttime image restoration task, where various types of weather degradations are intertwined with flare effects. To support the research, we contribute the AllWeatherNight dataset, featuring large-scale nighttime images with diverse compositional degradations. By employing illumination-aware degradation generation, our dataset significantly enhances the realism of synthetic degradations in nighttime scenes, providing a more reliable benchmark for model training and evaluation. Additionally, we propose ClearNight, a unified nighttime image restoration framework, which effectively removes complex degradations in one go. Specifically, ClearNight extracts Retinex-based dual priors and explicitly guides the network to focus on uneven illumination regions and intrinsic texture contents respectively, thereby enhancing restoration effectiveness in nighttime scenarios. Moreover, to more effectively model the common and unique characteristics of multiple weather degradations, ClearNight performs weather-aware dynamic specificity and commonality collaboration that adaptively allocates optimal sub-networks associated with specific weather types. Comprehensive experiments on both synthetic and real-world images demonstrate the necessity of the AllWeatherNight dataset and the superior performance of ClearNight.

<p align="center">
  <img src="https://github.com/henlyta/ClearNight/blob/page/static/image/frame.png?raw=True" width="100%">
</p>


## 🚀 Getting Started

### 1. 🗃️ Dataset Preparation
Download our **AllWeatherNight** dataset from [Hugging Face](https://huggingface.co/datasets/YuetongLiu/AllWeatherNight). 
Please organize the dataset directory as follows:

```text
data/
├── train/
│   ├── snow/       # Snow-related degraded input images (includes multiple weather)
│   ├── snow_gt/    # Snow-related ground truth (clean) images
│   ├── rain/       # Rain-related degraded input images (includes multiple weather)
│   ├── rain_gt/    # Rain-related ground truth (clean) images
│   ├── drop/       # Raindrop-related degraded input images (includes multiple weather)
│   └── drop_gt/    # Raindrop-related ground truth (clean) images
└── test/
    ├── snow/       # Snow-related test input images (snow_train_test)
    ├── snow_gt/    # Snow-related test ground truth images
    ├── rain/       # Rain-related test input images (rain_train_test)
    ├── rain_gt/    # Rain-related test ground truth images
    ├── drop/       # Raindrop-related test input images (drop_train_test)
    └── drop_gt/    # Raindrop-related test ground truth images

```

### 2. 📁 Pre-trained Weights
To support the training process (perceptual loss and depth guidance), download the following weights and place them in the `./loss/` folder:
* `vgg16-397923af.pth`
* `encoder.pth` (from ADDS-DepthNet ICCV 2021)
* `depth.pth` (from ADDS-DepthNet ICCV 2021)

### 3. 🏋️‍♂️ Training
To start training the **ClearNight** framework with Retinex decomposition:
```bash
python training_ClearNight.py --Retinex_decomp True
```

### 4. 🧪 Testing
To evaluate the model performance on the test set:
```bash
python testing_ClearNight.py --Retinex_decomp True
```

## 📖 Citation
If you find our work helpful for your research, please cite:

```bibtex
@inproceedings{aaai2026clearnight,
  title={Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration},
  author={Liu, Yuetong and Xu, Yunqiu and Wei, Yang and Bi, Xiuli and Xiao, Bin},
  booktitle={AAAI},
  year={2026}
}
```


## 📬 Contact
If you have any questions, please contact **Yuetong Liu** at [d230201022@stu.cqupt.edu.cn](mailto:d230201022@stu.cqupt.edu.cn).

  

