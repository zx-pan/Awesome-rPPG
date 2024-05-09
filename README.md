# Awesome rPPG

This is a collection of research papers for **rPPG**.
And the repository will be continuously updated to track the frontier of rPPG area.

Welcome to follow and star!

## Table of Contents

- [Awesome rPPG](#awesome-rppg)
  - [Table of Contents](#table-of-contents)
  - [Overview of rPPG](#overview-of-rppg)
  - [Papers](#papers)
    - [Survey](#survey)
    - [Traditional Methods](#traditional-methods)
    - [Supervised Learning](#supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
    - [Datasets](#datasets)
    - [Benchmark](#benchmark)

## Overview of rPPG
The non-contact measurement of heart rate (HR) and heart rate variability (HRV) has been a hot topic in the field of computer vision and signal processing. The rPPG technology is based on the principle of photoplethysmography (PPG), which uses the light absorption characteristics of blood to measure the heart rate. The rPPG technology can be used in various scenarios, such as health monitoring, emotion recognition, and human-computer interaction.

## Papers

### Survey

- [Video-Based Heart Rate Measurement: Recent Advances and Future Prospects](https://ieeexplore.ieee.org/document/8552414)
  - Xun Chen et al, IEEE Transactions on Instrumentation and Measurement (TIM), 2019

- [Camera Measurement of Physiological Vital Signs](https://arxiv.org/pdf/2111.11547)
  - Daniel McDuff et al, ACM Computing Surveys, 2021

- [Remote photoplethysmography for heart rate measurement: A review](https://www.sciencedirect.com/science/article/pii/S1746809423010418)
  - Hanguang Xiao et al, Biomedical Signal Processing and Control, 2023

### Traditional Methods

- [Remote plethysmographic imaging using ambient light](https://pdfs.semanticscholar.org/7cb4/46d61a72f76e774b696515c55c92c7aa32b6.pdf?_gl=1*1q7hzyz*_ga*NTEzMzk5OTY3LjE2ODYxMDg1MjE.*_ga_H7P4ZT52H5*MTY4NjEwODUyMC4xLjAuMTY4NjEwODUyMS41OS4wLjA) (GREEN) 
  - Wim Verkruysse et al, Optics Express, 2008

- [Advancements in Noncontact, Multiparameter Physiological Measurements Using a Webcam](https://ieeexplore.ieee.org/document/5599853) (ICA)
  - Ming-Zher Poh et al, IEEE Transactions on Biomedical Engineering (TBME), 2010

- [Measuring pulse rate with a webcam—a non-contact method for evaluating cardiac activity](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7ad15b6fecdb9b2ad49be5bf26efafe22c9a8945) (PCA)
  - Magdalena Lewandowska et al, 2011 federated conference on computer science and
information systems (FedCSIS), 2011

- [Robust Pulse Rate From Chrominance-Based rPPG](https://ieeexplore.ieee.org/document/6523142) (CHROM)
  - Gerard de Haan and Vincent Jeanne, IEEE Transactions on Biomedical Engineering (TBME), 2013

- [Improved motion robustness of remote-PPG by using the blood volume pulse signature](https://iopscience.iop.org/article/10.1088/0967-3334/35/9/1913) (PBV)
  - G de Haan et al, Physiological Measurement, 2014

- [A Novel Algorithm for Remote Photoplethysmography: Spatial Subspace Rotation](https://ieeexplore.ieee.org/document/7355301) (2SR)
  - Wenjin Wang et al, IEEE Transactions on Biomedical Engineering (TBME), 2015

- [Algorithmic Principles of Remote PPG](https://ieeexplore.ieee.org/document/7565547) (POS)
  - Wenjin Wang et al, IEEE Transactions on Biomedical Engineering (TBME), 2016

- [Local Group Invariance for Heart Rate Estimation from Face Videos in the Wild](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w27/Pilz_Local_Group_Invariance_CVPR_2018_paper.pdf) (LGI)
  - Christian S. Pilz et al, CVPR Workshop, 2018

- [Face2PPG: An Unsupervised Pipeline for Blood Volume Pulse Extraction From Faces](https://ieeexplore.ieee.org/document/10227326) (OMIT)
  - Constantino Álvarez Casado and Miguel Bordallo López, IEEE Journal of Biomedical and Health Informatics (JBHI), 2023

### Supervised Learning
- [Visual heart rate estimation with convolutional neural network](https://cmp.felk.cvut.cz/~spetlrad/ecg-fitness/visual-heart-rate.pdf) (HR-CNN)
  - Radim Špetlík et al, In Proceedings of British Machine Vision Conference (BMVC), 2018
  - [code](https://github.com/radimspetlik/hr-cnn)

- [DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks](https://openaccess.thecvf.com/content_ECCV_2018/papers/Weixuan_Chen_DeepPhys_Video-Based_Physiological_ECCV_2018_paper.pdf) (DeepPhys)
  - Weixuan Chen and Daniel McDuff, ECCV, 2018

- [RhythmNet: End-to-end Heart Rate Estimation from Face via Spatial-temporal Representation](https://arxiv.org/pdf/1910.11515) (RhythmNet)
  - Xuesong Niu et al, arXiv, 2019
  - [code](https://github.com/AnweshCR7/RhythmNet)

- [Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks](https://bmvc2019.org/wp-content/uploads/papers/0186-paper.pdf) (PhysNet)
  - Zitong Yu et al, BMVC, 2019
  - [code](https://github.com/ZitongYu/PhysNet)

- [PulseGAN: Learning to generate realistic pulse waveforms in remote photoplethysmography](https://arxiv.org/pdf/2006.02699) (PulseGAN)
  - Rencheng Song et al, arXiv, 2020
  - [code](https://github.com/miki998/PulseGAN?tab=readme-ov-file)

- [Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement](https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf) (TS-CAN)
  - Xin Liu et al, NeurIPS, 2020
  - [code](https://github.com/xliucs/MTTS-CAN)

- [Dual-GAN: Joint BVP and Noise Modeling for Remote Physiological Measurement](https://openaccess.thecvf.com/content/CVPR2021/papers/Lu_Dual-GAN_Joint_BVP_and_Noise_Modeling_for_Remote_Physiological_Measurement_CVPR_2021_paper.pdf) (Dual-GAN)
  - Hao Lu et al, CVPR, 2021

- [PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_PhysFormer_Facial_Video-Based_Physiological_Measurement_With_Temporal_Difference_Transformer_CVPR_2022_paper.pdf) (PhysFormer)
  - Zitong Yu et al, CVPR, 2022
  - [code](https://github.com/ZitongYu/PhysFormer)

- [PhysFormer++: Facial Video-based Physiological Measurement with SlowFast Temporal Difference Transformer](https://arxiv.org/pdf/2302.03548) (PhysFormer++)
  - Zitong Yu et al, International Journal of Computer Vision (IJCV), 2023

- [EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf) (EfficientPhys)
  - Xin Liu et al, WACV, 2023
  - [code](https://github.com/anonymous0paper/EfficientPhys)

- [rPPG-MAE: Self-supervised Pre-training with Masked Autoencoders for Remote Physiological Measurement](https://arxiv.org/pdf/2306.02301) (rPPG-MAE)
  - Xin Liu et al, arXiv, 2023
  - [code](https://github.com/linuxsino/rppg-mae)

- [BigSmall: Efficient Multi-Task Learning for Disparate Spatial and Temporal Physiological Measurement](https://arxiv.org/pdf/2303.11573) (BigSmall)
  - Girish Narayanswamy et al, WACV, 2024

- [RhythmFormer: Extracting rPPG Signals Based on Hierarchical Temporal Periodic Transformer](https://arxiv.org/pdf/2402.12788) (RhythmFormer)
  - Bochao Zou et al, arXiv, 2024
  - [code](https://github.com/zizheng-guo/RhythmFormer)

- [RhythmMamba: Fast Remote Physiological Measurement with Arbitrary Length Videos](https://arxiv.org/pdf/2404.06483) (RhythmMamba)
  - Bochao Zou et al, arXiv, 2024

  
### Unsupervised Learning / Self-supervised Learning
- [The Way to my Heart is through Contrastive Learning: Remote Photoplethysmography from Unlabelled Video](https://openaccess.thecvf.com/content/ICCV2021/papers/Gideon_The_Way_to_My_Heart_Is_Through_Contrastive_Learning_Remote_ICCV_2021_paper.pdf)
  - John Gideon and Simon Stent, ICCV, 2021
  - [code](https://github.com/ToyotaResearchInstitute/RemotePPG)

- [Contrast-Phys: Unsupervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720488.pdf) (Contrast-Phys)
  - Zhaodong Sun and Xiaobai Li, ECCV, 2022
  - [code](https://github.com/zhaodongsun/contrast-phys)

- [Contrast-Phys+: Unsupervised and Weakly-supervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast](https://ieeexplore.ieee.org/document/10440521) (Contrast-Phys+)
  - Zhaodong Sun and Xiaobai Li, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2024
  - [code](https://github.com/zhaodongsun/contrast-phys)

- [Facial Video-Based Remote Physiological Measurement via Self-Supervised Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10193817)
  - Zijie Yue et al, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023

- [Non-Contrastive Unsupervised Learning of Physiological Signals from Video](https://openaccess.thecvf.com/content/CVPR2023/papers/Speth_Non-Contrastive_Unsupervised_Learning_of_Physiological_Signals_From_Video_CVPR_2023_paper.pdf) (SiNC)
  - Jeremy Speth et al, CVPR, 2023
  - [code](https://github.com/CVRL/SiNC-rPPG)

- [SiNC+: Adaptive Camera-Based Vitals with Unsupervised Learning of Periodic Signals](https://arxiv.org/pdf/2404.13449) (SiNC+)
  - Jeremy Speth et al, arXiv, 2024

### Datasets
- [DEAP: A Database for Emotion Analysis ;Using Physiological Signals](https://ieeexplore.ieee.org/document/5871728) (DEAP)
  - Sander Koelstra et al, IEEE Transactions on Affective Computing (TAFFC), 2011
  - [download](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)

- [Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6926392) (PURE)
  - Ronny Stricker et al, IEEE International Workshop on Robot and Human Communication (ROMAN), 2014
  - [download](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)

- [Multimodal Spontaneous Emotion Corpus for Human Behavior Analysis](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Multimodal_Spontaneous_Emotion_CVPR_2016_paper.pdf) (MMSE-HR)
  - Zheng Zhang et al, CVPR, 2016
  - [download](https://binghamton.technologypublisher.com/tech/MMSE-HR_dataset_(Multimodal_Spontaneous_Expression-Heart_Rate_dataset))

- [A Reproducible Study on Remote Heart Rate Measurement](https://arxiv.org/pdf/1709.00962) (COHFACE)
  - G. Heusch et al, arXiv, 2017
  - [download](https://www.idiap.ch/en/scientific-research/data/cohface)

- [Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters](https://www.sciencedirect.com/science/article/pii/S0167865517303860) (UBFC-RPPG)
  - Serge Bobbia et al, Pattern Recognition Letters, 2017
  - [download](https://sites.google.com/view/ybenezeth/ubfcrppg)

- [VIPL-HR: A Multi-modal Database for Pulse Estimation from Less-constrained Face Video](https://arxiv.org/pdf/1810.04927v2) (VIPL-HR)
  - Xuesong Niu et al, ACCV, 2018
  - [download](http://vipl.ict.ac.cn/database.php)

- [The OBF Database: A Large Face Video Database for Remote Physiological Signal Measurement and Atrial Fibrillation Detection](https://ieeexplore.ieee.org/document/8373836) (OBF)
  - Xiaobai Li et al, 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018), 2018

- [Near-Infrared Imaging Photoplethysmography During Driving](https://ieeexplore.ieee.org/document/9275394) (MR-NIRP)
  - Ewa M. Nowara et al, IEEE Transactions on Intelligent Transportation Systems (TITS), 2020
  - [download](https://computationalimaging.rice.edu/mr-nirp-dataset/)

- [Evaluation of biases in remote photoplethysmography methods](https://www.nature.com/articles/s41746-021-00462-z) (CMU PPG)
  - Ananyananda Dasari et al,  npj digital medicine, 2021
  - [download](https://github.com/AiPEX-Lab/rppg_biases?tab=readme-ov-file)

- [UBFC-Phys: A Multimodal Database For Psychophysiological Studies Of Social Stress](https://ieeexplore.ieee.org/document/9346017) (UBFC-Phys)
  - Rita Meziati Sabour et al, IEEE Transactions on Affective Computing (TAFFC), 2021
  - [download](https://sites.google.com/view/ybenezeth/ubfc-phys)

- [Deception Detection and Remote Physiological Monitoring: A Dataset and Baseline Experimental Results](https://arxiv.org/pdf/2106.06583) (DDPM)
  - Jeremy Speth et al, IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM), 2021
  - [download](https://cvrl.nd.edu/projects/data/#deception-detection-and-%20physiological-monitoringddpm)

- [SCAMPS: Synthetics for Camera Measurement of Physiological Signals](https://proceedings.neurips.cc/paper_files/paper/2022/file/1838feeb71c4b4ea524d0df2f7074245-Paper-Datasets_and_Benchmarks.pdf) (SCAMPS)
  - Daniel McDuff et al, arXiv, 2022
  - [download](https://github.com/danmcduff/scampsdataset)

- [Synthetic Generation of Face Videos with Plethysmograph Physiology](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Synthetic_Generation_of_Face_Videos_With_Plethysmograph_Physiology_CVPR_2022_paper.pdf) (UCLA-rPPG, UCLA-synthetic)
  - Zhen Wang et al, CVPR, 2022
  - [download](http://visual.ee.ucla.edu/rppg_avatars.htm/)

- [MMPD: Multi-Domain Mobile Video Physiology Dataset](https://arxiv.org/pdf/2302.03840) (MMPD)
  - Jiankai Tang et al, EMBC, 2023
  - [download](https://github.com/McJackTang/MMPD_rPPG_dataset?tab=readme-ov-file)

- [ReactioNet: Learning High-order Facial Behavior from Universal Stimulus-Reaction by Dyadic Relation Reasoning](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_ReactioNet_Learning_High-Order_Facial_Behavior_from_Universal_Stimulus-Reaction_by_Dyadic_ICCV_2023_paper.pdf) (BP4D+)
  - Xiaotian Li et al, ICCV, 2023
  - [download](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)

- [iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels](https://doi.org/10.3390/electronics13071334) (iBVP)
  - Jitesh Joshi and Youngjun Cho, Electronics, 2024
  - [download](https://github.com/PhysiologicAILab/iBVP-Dataset?tab=readme-ov-file)





### Benchmark
- [iPhys: An Open Non-Contact Imaging-Based Physiological Measurement Toolbox](https://arxiv.org/pdf/1901.04366) (Matlab Toolbox)
  - Daniel McDuff and Ethan Blackford, arXiv, 2019
  - [code](https://github.com/danmcduff/iphys-toolbox?tab=readme-ov-file)

- [Evaluation of biases in remote photoplethysmography methods](https://www.nature.com/articles/s41746-021-00462-z) (Matlab Toolbox)
  - Ananyananda Dasari et al, npj Digital Medicine, 2021
  - [code](https://github.com/partofthestars/PPGI-Toolbox?tab=readme-ov-file)

- [pyVHR: a Python framework for remote photoplethysmography](https://peerj.com/articles/cs-929/) (Python Toolbox)
  - Giuseppe Boccignone et al, PeerJ Computer Science, 2022
  - [code](https://github.com/phuselab/pyVHR?tab=readme-ov-file)

- [rPPG-Toolbox: Deep Remote PPG Toolbox](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d7d0d548a6317407e02230f15ce75817-Abstract-Datasets_and_Benchmarks.html) (Python Toolbox)
  - Xin Liu et al, NeurIPS 2023 
  - [code](https://github.com/ubicomplab/rPPG-Toolbox)

- [Remote Bio-Sensing: Open Source Benchmark Framework for Fair Evaluation of rPPG](https://arxiv.org/pdf/2307.12644) (Python Toolbox)
  - Dae-Yeol Kim et al, arXiv, 2023
  - [code](https://github.com/remotebiosensing/rppg?tab=readme-ov-file)

