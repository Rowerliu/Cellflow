# Cellflow

**Cellflow: Advancing pathological image augmentation from spatial views to temporal trajectories**<br/>
(In submission)<br/>
[Zeyu Liu](https://github.com/Rowerliu), Tianyi Zhang, Yufang He, Bo Wen, Haoran Guo, Peng Zhang, Chenbin Ma, Shangqing Lyu, Yunlu Feng, Yu Zhao, Yueming Jin, Dachun Zhao, Guanglei Zhang<br/>
[GitHub](https://github.com/Rowerliu/Cellflow)


## 📇 Overview
Deep learning has advanced pathological image analysis but remains constrained by limited annotated data, 
especially for fine-grained and rare conditions. While data augmentation alleviates this issue, 
existing methods are restricted to spatial manipulations that lack biological plausibility and overlook the 
temporal dynamics of pathological state transition. To address this gap, we propose **Cellflow**, 
the first temporal-aware generative framework for pathological image augmentation. 
Cellflow models pathological transition as smooth trajectories on a biological image manifold, 
generating intermediate states via a stair-based diffusion bridge with classifier-guided probability-flow 
ordinary differential equations. This design produces biologically plausible sequences that capture both cellular details 
and tissue-level architecture. Evaluated on 7 diverse datasets across organs, staining modalities, and diagnostic tasks, 
Cellflow consistently outperforms 6 spatial augmentation methods and 4 state-of-the-art generative models, 
yielding improved classification performance, higher image fidelity, and preservation of temporal coherence. 
Quantitative cellularity analysis provides additional validation of the biological authenticity of transition sequences. 
By introducing temporal modeling into pathological data augmentation, Cellflow establishes a paradigm shift from 
spatial manipulations to biologically grounded temporal trajectories that advances robust model training, 
rare disease exploration, and educational simulation in computational pathology.


## 🗃️ Usage
### Step1: Prepare data
Take [MHIST](https://bmirds.github.io/MHIST) data for example:<br/>
```
MHIST
│
├── HP
│   ├── HP_0001.png
│   ├── ...
│   └── HP_9999.png<br/>
│
└── SSA
    ├── SSA_0001.png
    ├── ...
    └── SSA_9999.png
```

### Step2: Diffusion model training
1. Prepare your data<br/>
2. Follow the [guided-diffusion](https://github.com/openai/guided-diffusion)<br/>
3. Take MHIST data for example:<br/>
   'Class_cond = True', 'num_classes = 2'<br/>

### Step 3: Progressive data generation
Assign the path of trained models, and then generate intermediate images<br/>
```bash
python main_transflow.py 
--result_dir='./result' 
--model_dir='./diffusion_model'
--classifier_dir='./classifier_model'
--data_dir=r'./data'
--num_classes=2
--out_channels=2
--timestep_respacing='ddim100'
--amount=10 
```


## 📍 Acknowledgements
This implementation is based on / inspired by:<br/>
[openai/guided-diffusion](https://github.com/openai/guided-diffusion)<br/>
[openai/improved-diffusion](https://github.com/openai/improved-diffusion)<br/>
[suxuann/ddib](https://github.com/suxuann/ddib)


## 🗄️ Enviroments
A suitable [conda](https://conda.io/) environment named `cellflow` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate cellflow
```


## 🗃️ Materials
### The datasets are listed here:

| Dataset     | Paper                                                                                                                                                                                                                                                                           | Datalink                                                 |
|:------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------|
| CPIA        | [CPIA dataset: a large-scale comprehensive pathological image analysis dataset for self-supervised learning pre-training](https://www.sciencedirect.com/science/article/pii/S1746809425006597)                                                                                  | [link](https://github.com/zhanglab2021/CPIA_Dataset)     |
| pRCC        | [Instance-based vision transformer for subtyping of papillary renal cell carcinoma in histopathological image](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_29)                                                                                                  | [link](https://dataset.chenli.group/home/prcc-subtyping) |
| MHIST       | [A Petri Dish for Histopathology Image Analysis](http://arxiv.org/abs/2101.12355)                                                                                                                                                                                               | [link](https://bmirds.github.io/MHIST)                   |
| Gleason2019 | [Automatic grading of prostate cancer in digitized histopathology images: Learning from multiple experts](https://www.sciencedirect.com/science/article/pii/S1361841518307497)                                                                                                  | [link](https://gleason2019.grand-challenge.org)          |
| Chaoyang    | [Hard Sample Aware Noise Robust Learning for Histopathology Image Classification](https://ieeexplore.ieee.org/document/9344937)                                                                                                                                                 | [link](https://github.com/bupt-ai-cz/HSA-NRL)            |
| AML         | [Human-level recognition of blast cells in acute myeloid leukaemia with convolutional neural networks](https://www.nature.com/articles/s42256-019-0101-9)                                                                                                                       | [link](https://doi.org/10.7937/tcia.2019.36f5o9ld)       |
| BCI         | [BCI: Breast cancer immunohistochemical image generation through pyramid pix2pix](https://ieeexplore.ieee.org/document/9857332)                                                                                                                                                 | [link](https://bupt-ai-cz.github.io/BCI)                 |
| BreastPathQ | [SPIE-AAPM-NCI BreastPathQ challenge: an image analysis challenge for quantitative tumor cellularity assessment in breast cancer histology images following neoadjuvant treatment](https://www.spiedigitallibrary.org/journalArticle/Download?fullDOI=10.1117/1.JMI.8.3.034501) | [link](https://breastpathq.grand-challenge.org/)         |


### The comparison spatial augmentation methods are listed here:

| Method      | Paper                                                                                                                            | Codelink                                                  |
|:------------|:---------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------|
| Cutout      | [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)                         | [link](https://github.com/uoguelph-mlrg/Cutout)           |
| Mixup       | [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)                                                    | [link](https://github.com/facebookresearch/mixup-cifar10) |
| CutMix      | [CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features](https://arxiv.org/abs/1905.04899)        | [link](https://github.com/clovaai/CutMix-PyTorch)         |
| FMix        | [FMix: Enhancing Mixed Sample Data Augmentation](https://arxiv.org/abs/2002.12047)                                               | [link](https://github.com/ecs-vlc/FMix)                   |
| ResizeMix   | [ResizeMix: Mixing Data with Preserved Object Information and True Labels](https://arxiv.org/abs/2012.11101)                     | Null                                                      |
| SaliencyMix | [SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization](https://arxiv.org/abs/2006.01791)          | [link](https://github.com/SaliencyMix/SaliencyMix)        |


### The comparison generative methods are listed here:

| Method      | Paper                                                                                                     | Codelink                                                |
|:------------|:----------------------------------------------------------------------------------------------------------|:--------------------------------------------------------|
| StyleGAN    | [Stylegan-xl: Scaling stylegan to large diverse datasets](https://arxiv.org/abs/2202.00273)               | [link](https://github.com/autonomousvision/stylegan-xl) |
| ADM         | [Diffusion models beat gans on image synthesis](https://arxiv.org/abs/2105.05233)                         | [link](https://github.com/openai/guided-diffusion)      |
| LDM         | [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)          | [link](https://github.com/CompVis/latent-diffusion)     |
| DiT         | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)                           | [link](https://github.com/facebookresearch/DiT)         |


### Employ [Unpuzzle](https://github.com/Puzzle-Logic/UnPuzzle) framework for classification backbones:

| Architecture           | Model                                                              |
|:-----------------------|:-------------------------------------------------------------------|
| CNN-based              | VGG16, VGG19, ResNet, Inception, Xception, MobileNet, EfficientNet |
| ViT-based              | ViT, SwinT                                                         |
| Hybrid-based           | Conformer, CrossFormer                                             |
| Foundation Model-based | Virchow2, UNI                                                      |
