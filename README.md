<p align="center">
  <table align="center">
    <tr>
      <td>
        <img src="documents/MambaHoME_logo.png" alt="HoME Logo" width="160">
      </td>
      <td>
        <h1 align="center">
          Mamba Goes HoME:<br>
          Hierarchical Soft Mixture-of-Experts<br>
          for 3D Medical Image Segmentation
        </h1>
      </td>
    </tr>
  </table>
</p>


![visitors](https://visitor-badge.laobi.icu/badge?page_id=gmum/MambaHoME&left_color=%2363C7E6&right_color=%23CEE75F)
[![GitHub stars](https://img.shields.io/github/stars/gmum/MambaHoME.svg?style=social)](https://github.com/gmum/MambaHoME/stargazers)
[![Cite](https://img.shields.io/badge/Cite-This%20Repo-blue)](https://arxiv.org/abs/2507.06363)

<p align="center">
  <img src="documents/mri_prediction.gif" width="800" />
</p>

<p align="center">
  Qualitative segmentation results. From left input slice, ground truth, and Mamba-HoME prediction.
</p>


## TL;DR
We introduce <strong>Hierarchical Soft Mixture-of-Experts (HoME)</strong>, a two-level token-routing layer for efficient long-context modeling, specifically designed for 3D medical image segmentation. Built on the Mamba Selective State Space Model (SSM) backbone, HoME enhances sequential modeling through adaptive expert routing. In the first level, a Soft Mixture-of-Experts (SMoE) layer partitions input sequences into local groups, routing tokens to specialized per-group experts for localized feature extraction. The second level aggregates these outputs through a global SMoE layer, enabling cross-group information fusion and global context refinement. This hierarchical design, combining local expert routing with global expert refinement, enhances generalizability and segmentation performance, surpassing state-of-the-art results across datasets from the three most widely used 3D medical imaging modalities and varying data qualities.

## Paper

<b>Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation</b> <br/>
[Szymon Płotka](https://scholar.google.com/citations?user=g9sWRN0AAAAJ)<sup>1,2</sup>*, [Gizem Mert](https://scholar.google.com/citations?user=tTPBMfsAAAAJ)<sup>3</sup>, [Maciej Chrabaszcz](https://scholar.google.com/citations?user=qdUVcecAAAAJ)<sup>4,5</sup>, [Ewa Szczurek](https://scholar.google.com/citations?user=hltmGf0AAAAJ)<sup>1,3</sup>, [Arkadiusz Sitek](https://scholar.google.com/citations?user=3QheHgMAAAAJ)<sup>6,7</sup><br/>
<sup>1</sup>University of Warsaw, <sup>2</sup>Jagiellonian University, <sup>3</sup>Helmholtz Munich, <sup>4</sup>Warsaw University of Technology, <sup>5</sup>NASK, <sup>6</sup>Massachusetts General Hospital, <sup>7</sup>Harvard Medical School<br/>
Advances in Neural Information Processing Systems (NeurIPS) 2025 <br/><br/>
<a href='https://arxiv.org/pdf/2507.06363'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a>

## Datasets

For pre-training, training, and evaluation of Mamba-HoME, we use the following datasets:

<table style="border-collapse: collapse; width: 100%; border: none;">
  <thead>
    <tr style="border: none; background-color: #fff;">
      <th style="text-align: left; padding: 8px; border: 1px solid #ddd; font-weight: bold;">Dataset</th>
      <th style="text-align: left; padding: 8px; border: 1px solid #ddd; font-weight: bold;">Source</th>
      <th style="text-align: left; padding: 8px; border: 1px solid #ddd; font-weight: bold;">Modality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">AbdomenAtlas 1.1</td>
      <td style="padding: 8px; border: 1px solid #ddd;">
        <a href="https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini"><strong>Download</strong></a><br>
      </td>
      <td style="padding: 8px; border: 1px solid #ddd;">CT</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">TotalSegmentator MRI</td>
      <td style="padding: 8px; border: 1px solid #ddd;">
        <a href="https://zenodo.org/records/14710732"><strong>Download</strong></a><br>
      </td>
      <td style="padding: 8px; border: 1px solid #ddd;">MRI</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">PANORAMA</td>
      <td style="padding: 8px; border: 1px solid #ddd;">
        <a href="https://panorama.grand-challenge.org/datasets-imaging-labels/"><strong>Download</strong></a><br>
      </td>
      <td style="padding: 8px; border: 1px solid #ddd;">CT</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">AMOS</td>
      <td style="padding: 8px; border: 1px solid #ddd;">
        <a href="https://zenodo.org/records/7262581"><strong>Download</strong></a><br>
      </td>
      <td style="padding: 8px; border: 1px solid #ddd;">CT / MRI</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">FeTA 2022</td>
      <td style="padding: 8px; border: 1px solid #ddd;">
        <a href="https://feta.grand-challenge.org/feta-2022/"><strong>Download</strong></a><br>
      </td>
      <td style="padding: 8px; border: 1px solid #ddd;">MRI</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">MVSeg</td>
      <td style="padding: 8px; border: 1px solid #ddd;">
        <a href="https://www.synapse.org/Synapse:syn51186045/wiki/622044"><strong>Download</strong></a><br>
      </td>
      <td style="padding: 8px; border: 1px solid #ddd;">3D US</td>
    </tr>
  </tbody>
</table>


## Installation


#### Step 1. Install Anaconda on Linux

<details>
<summary style="margin-left: 25px;">[Optional] Install Anaconda on Linux</summary>
<div style="margin-left: 25px;">
    
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p ./anaconda3
./anaconda3/bin/conda init
source ~/.bashrc
```
</div>
</details>

#### Step 2. Create a new virtual environment:

```bash
cd MambaHoME
conda create -n mambaHoME python=3.11
conda activate mambaHoME
```

> [!IMPORTANT]  
> Before installing all dependencies, please install the Mamba packages first as provided in the official repository.

#### Step 3. Install the **Mamba SSM** package and its dependencies using the following steps:

```bash
pip install "causal-conv1d>=1.4.0"

# Install the core Mamba SSM package only
pip install mamba-ssm

# Install Mamba SSM along with causal Conv1d
pip install "mamba-ssm[causal-conv1d]"

# Install Mamba SSM with developer dependencies
pip install "mamba-ssm[dev]"

# (Optional) Troubleshooting: Install with no build isolation
pip install . --no-build-isolation
```

#### Step 4. Install all dependencies

```bash
pip install -r requirements.txt
```

## Usage

> [!NOTE]
> We trained our proposed Mamba-HoME and other state-of-the-art methods using 32-bit floating-point (float32) precision.

### Step 1. Data pre-processing

The first step in our pipeline is resampling all datasets to a standardized voxel spacing.

```bash
python convert_spacing.py \
    --input_path /path/to/raw_dataset \
    --output_path /path/to/preprocessed_dataset \
    --spacing 0.8 0.8 3.0
```

### Step 2. Training

The next step is to train the MambaHoME model using the prepared dataset.

```bash
python train.py \
    --data_train ../dataset/train --data_val ../dataset/val \
    --batch_size 2 --classes 3 --epochs 800 --lr 1e-4 --weight_decay 1e-4 \
    --optimizer AdamW --scheduler CALR --patch_size 128 128 128 --feature_size 48 \
    --use_checkpoint False --num_workers 12 --pin_memory True --use_pretrained False \
    --load_checkpoint False --checkpoint_name "MambaHoME" --model MambaHoME \
    --parallel True --num_devices 4 --strategy ddp
```

### Step 3. Inference

After training, the best-performing checkpoint can be used for inference on test datasets.

```bash
python inference.py \
    --data_path ../dataset/test \
    --model_path ../checkpoints/MambaHoME_best.pth \
    --output_dir ../inference_results \
    --roi_size 192 192 48 \
    --sw_batch_size 4 \
    --num_workers 4 \
    --overlap 0.5 \
    --gpu
```

### Requirements

* Linux
* CUDA 11.6+
* Python 3.11+
* PyTorch 2.4+
* MONAI 1.3.0+

## Citation

If you use the code or methods in this repository, please cite:


```bibtex
@article{plotka2025mamba,
  title={Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation},
  author={Płotka, Szymon and Mert, Gizem and Chrabaszcz, Maciej and Szczurek, Ewa and Sitek, Arkadiusz},
  journal={arXiv preprint arXiv:2507.06363},
  year={2025}
}
```
## Related work

Our method builds upon the following works and their official implementations:
- [Mamba: Linear-Time Sequence Modeling with Selective SSMs (Gu & Dao, 2024)](https://arxiv.org/abs/2312.00752)
- [From Sparse to Soft Mixtures of Experts (Puigcerver et al., 2024)](https://arxiv.org/abs/2308.00951)
- [SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation (Xing et al., 2024)](https://arxiv.org/abs/2401.13560)


## License

This project is released under the [MIT License](https://opensource.org/license/mit).

## Acknowledgments

We acknowledge the use of the HPC cluster at Helmholtz Munich for the computational resources used in this study.

