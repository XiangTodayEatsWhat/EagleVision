
<h1>EagleVision: Object-level Attribute Multimodal LLM for Remote Sensing</h1>

<a href="https://huggingface.co/liarzone" target="_blank"><img alt="Checkpoint" src="https://img.shields.io/badge/🤗 Hugging Face Models-2980b9?color=2980b9" /></a>
<a href="https://huggingface.co/datasets/liarzone/EVAttrs-95K" target="_blank"><img alt="Data" src="https://img.shields.io/badge/🤗 Hugging Face Datasets-8e44ad?color=8e44ad" /></a>

[Hongxiang Jiang](https://github.com/XiangTodayEatsWhat), [Jihao Yin*](https://scholar.google.com/citations?hl=zh-CN&user=-1deilUAAAAJ), [Qixiong Wang](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=rlr_izYAAAAJ&gmla=AOv-ny8tmsrvdtwgmZ2UZCW5feX8-dM2W91AhdS0RFQBIw9OAvapFIwnxCQbme-Vz27q4vaoHf9IdWysFcVCSCUqrLKakoy3YkpNxlQ927QpajJLjN9n1PNG), [Jiaqi Feng](https://github.com/fengjiaqi927/), [Guo Chen](https://github.com/arbani369)
<p align="center">
<img src="assets/logo.png" style="width: 200px" align=center>
</p>

----

<!-- <div align=center>

![Static Badge](https://img.shields.io/badge/Chat-Rex-red) [![arXiv preprint](https://img.shields.io/badge/arxiv_2411.18363-blue%253Flog%253Darxiv
)](https://arxiv.org/abs/2411.18363)  [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FIDEA-Research%2FChatRex&count_bg=%2379C83D&title_bg=%23F4A6A6&icon=waze.svg&icon_color=%23E7E7E7&title=VISITORS&edge_flat=false)](https://hits.seeyoufarm.com)

</div> -->
# Contents
- [Contents](#contents)
- [1. Introduction](#1-introduction)
- [2. Installation](#2-installation)
- [3. Usage](#3-usage)
  - [3.1 Quick Start](#31-quick-start)
    - [3.1.1 Download Pretrained Models](#311-download-pretrained-models)
    - [3.1.2 Inference](#312-inference)
  - [3.2 Training & Testing](#32-training--testing)
    - [3.2.1 Data Preparation](#321-data-preparation)
    - [3.2.2 Training](#322-training)
    - [3.2.3 Testing](#323-testing)
- [4. Gradio Demo](#4-gradio-demo)
- [BibTeX](#bibtex)

# News
- 🔥🔥 [2025-3-29] EVAttrs-95K dataset is now available at [https://huggingface.co/datasets/liarzone/EVAttrs-95K](https://huggingface.co/datasets/liarzone/EVAttrs-95K).

----

# 1. Introduction

**EagleVision outperforms in object attribute understanding, covering various attributes of all detected objects.**


<div align=center>
  <img src="assets/introduction.png" width=800 >
</div>


EagleVision is a Multimodal Large Language Model (MLLM) tailored for remote sensing that excels in object detection and attribute comprehension. EagleVision achieves state-of-the-art performance on both fine-grained object detection and object attribute understanding tasks, highlighting the mutual promotion between detection and understanding capabilities in MLLMs. 

----
# 2. Installation


```bash
conda create -n EagleVision python=3.10
conda activate EagleVision 

# install torch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# install mmengine mmcv mmdet mmrotate
pip install -U openmim
mim install mmengine==0.10.5
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install mmdet==3.3.0
pip install -v -e .

# install flash-attn
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# install other requirements
pip install transformers==4.47.1 peft==0.4.0 numpy==1.26.3 imageio==2.36.0 einops==0.6.1 timm==1.0.9 deepspeed==0.16.1 sentencepiece==0.2.0 protobuf==4.25.4
```
----

# 3. Usage
Use our provided pre-trained EagleVision to perform both object detection and object attribute understanding tasks, test on both tasks separately, or train your own EagleVision!

<div align=center>
  <img src="assets/EagleVision.png" width=800 >
</div>


## 3.1 Quick Start

### 3.1.1 Download Pretrained Models
We provide model checkpoints for EagleVision models. You can download these pre-trained models from the following links:

<table>
	<tr>
		<th>Model Name</th> 
		<th>Params</th> 
		<th>LLM</th> 
		<th>Huggingface Link</th> 
	</tr>
	<tr>
		<td>EagleVision-1B</td>
		<td>0.64 B</td>
		<td><a href="https://huggingface.co/Qwen/Qwen2-0.5B-Instruct">🤗 Qwen2-0.5B-Instruct</a></td>
		<td><a href="https://huggingface.co/liarzone/EagleVision-1B">🤗 EagleVision-1B</a></td>
	</tr>
	<tr>
		<td>EagleVision-2B</td>
		<td>1.92 B</td>
		<td><a href="https://huggingface.co/internlm/internlm2-chat-1_8b">🤗 internlm2-chat-1-8b</a></td>
		<td><a href="https://huggingface.co/liarzone/EagleVision-2B">🤗 EagleVision-2B</a></td>
	</tr>
	<tr>
		<td>EagleVision-4B</td>
		<td>3.86 B</td>
		<td><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">🤗 Phi&#8209;3&#8209;mini&#8209;128k&#8209;instruct</a></td>
		<td><a href="https://huggingface.co/liarzone/EagleVision-4B">🤗 EagleVision-4B</a></td>
	</tr>
	<tr>
		<td>EagleVision-7B</td>
		<td>7.77 B</td>
		<td><a href="https://huggingface.co/internlm/internlm2_5-7b-chat">🤗 internlm2_5-7b-chat</a></td>
		<td><a href="https://huggingface.co/liarzone/EagleVision-7B">🤗 EagleVision-7B</a></td>
	</tr>
</table>


### 3.1.2 Inference
We provide a simple method to quickly call any EagleVision model. It supports two modes:

- Object detection only, which introduces no additional computational overhead.
- Joint object detection and object attribute understanding.
```bash
# object detection
python tools/infer.py demo/demo1.png ${model_path}/EagleVision-7B/SHIPRS/mp_rank_00_model_states.pt configs/EagleVision/EagleVision_7B-shiprsimagenet.py --score-thr 0.5

# object detection and object attribute understanding
python tools/infer.py demo/demo1.png ${model_path}/EagleVision-7B/SHIPRS/mp_rank_00_model_states.pt configs/EagleVision/EagleVision_7B-shiprsimagenet.py --score-thr 0.5 --with-attribute
```

<details close>

<summary><strong>Example Output</strong></summary>

The visualization of the detection is like:

<div align=center>
  <img src="assets/ans.png" width=600 >
</div>

The attribute output is like:

```bash
<0> This object belongs to the "YuTing LL" category. Its ship-visibility is clear, ship-purpose is military, ship-motion is stationary, ship-capacity is medium, ship-load-status is unloaded, ship-cargo-status is no cargo, ship-mooring-status is moored, hull-color is gray, hull-size is large, hull-shadow is visible, hull-outline is sharp, superstructure-color is gray, superstructure-size is medium, superstructure-height is moderate, superstructure-position is central, paint-condition is good, bow-design is sharp, stern-design is flat, deck-utilization is moderate, deck-condition is good, deck-obstacles is minimal, deck-color is gray, deck-structure is flat, deck-accessories is helicopter landing pad, container-count is 0, machinery-presence is visible, location is dockside, weather-condition is clear, water-color is dark blue, water-turbulence is calm, unique-attributes is helicopter landing pad on deck. <end>
<1> This object belongs to the "Cargo" category. Its ship-visibility is partially visible, ship-purpose is cargo transport, ship-motion is stationary, ship-capacity is large, ship-load-status is loaded, ship-cargo-status is secured, ship-mooring-status is moored, hull-color is dark gray, hull-size is large, hull-shadow is visible, hull-outline is clear, superstructure-color is light gray, superstructure-size is medium, superstructure-height is low, superstructure-position is central, paint-condition is worn, bow-design is rounded, stern-design is flat, deck-utilization is high, deck-condition is worn, deck-obstacles is minimal, deck-color is dark gray, deck-structure is flat, deck-accessories is minimal, container-presence is yes, container-count is multiple, container-color is varied, container-layout is stacked, container-alignment is aligned, container-densities is dense, container-type is standard, machinery-presence is yes, location is waterway, weather-condition is clear, water-color is dark blue, water-turbulence is low, unique-attributes is large cargo ship with multiple containers. <end>
```

</details>

## 3.2 Training & Testing

### 3.2.1 Data Preparation

- We have released the complete EVAttrs-95K dataset. Download the dataset from [Hugging Face](https://huggingface.co/datasets/liarzone/EVAttrs-95K).

- Download [SHIPRSImageNet](https://github.com/zzndream/ShipRSImageNet), [MAR20](https://gcheng-nwpu.github.io/) and [FAIR1M](https://www.gaofen-challenge.com/benchmark) dataset and create the file structure as follows:
```
└── data
    └── SHIPRSImageNet
        └── train
		└── images
		└── annfiles
        	└── labelXml
        └── val
        	└── ...
	└── EVAttrs-95K-ShipRSImageNet-train.json
    	└── EVAttrs-95K-ShipRSImageNet-val.json
    └── MAR20
        └── train
        	└── images
		└── annfiles
        	└── labelXml
        └── test
        	└── ...
	└── EVAttrs-95K-MAR20-train.json
	└── EVAttrs-95K-MAR20-test.json
    └── FAIR1M
        └── train
        	└── images
        	└── labelXml
        └── ...
        └── split_ss
        	└── train
        		└── images
        		└── annfiles
		└── ...
	└── EVAttrs-95K-FAIR1M-train.json
    	└── EVAttrs-95K-FAIR1M-val.json

```
- Note: **images** and **labelXml** refer to the original dataset's images and annotations. **annfiles** must follow the DOTA annotation format with additional attribute information according to the EVAttrs-95K, as follows:

```
432.0 482.0 405.0 503.0 398.0 494.0 425.0 473.0 </cls_name>Other Ship</cls_name> 0 ${attribute_dict}
```

### 3.2.2 Training
Run the following command to train EagleVision! 🤗
```bash
# batch_size = (4 GPUs) x (2 samples per GPU) = 8
NCCL_DEBUG=INFO NCCL_P2P_LEVEL=NVL NCCL_DEBUG_FILE=./nccl.log CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port ${master_port} tools/train.py ${config_file}
```

### 3.2.3 Testing
Testing the performance of Eaglevision in object detection:

```bash
# For MAR20 and SHIPRSImageNet, the results could be displayed, while for FAIR1M, online evaluation is required.
NCCL_DEBUG=INFO NCCL_P2P_LEVEL=NVL NCCL_DEBUG_FILE=./nccl.log CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port  ${master_port} tools/test.py ${config_file} ${model_path} --task 1
```

Testing the performance of Eaglevision in object attribute understanding:

```bash
# generate inference results to ${work_dir}/val/Task2.json
NCCL_DEBUG=INFO NCCL_P2P_LEVEL=NVL NCCL_DEBUG_FILE=./nccl.log CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port  ${master_port} tools/test.py ${config_file} ${model_path} --task 2

# EVbench evaluation preparation
pip install openai==1.3.5 validators==0.34.0 sty==1.0.6 portalocker==3.0.0 omegaconf==2.3.0 httpx==0.24.0
cd VLMEvalKit
pip install -e .

# EVbench evaluation for different datasets
python tools/benchmark/EVBench_evaluation.py --root-dir ${work_dir}/val/Task2.json --openai-key ${openai-key} --predefined-attributes ${SHIPRS/MAR20/FAIR1M}
```

----


# 4. Gradio Demo
We've also built a Gradio demo for EagleVision — here’s how you can try it out:
```bash
# environment preparation
pip install gradio

# run the demo
python demo/app.py
```
<div align=center>
  <img src="assets/gradio_demo.png" width=600 >
</div>

----

# BibTeX

```
```
