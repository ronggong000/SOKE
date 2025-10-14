# Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator
Official implementation for the ICCV 2025 paper, [Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator](https://arxiv.org/pdf/2411.17799).


## Introduction
Sign language is a visual language that encompasses all linguistic features of natural languages and serves as the primary communication method for the deaf and hard-of-hearing communities. Although many studies have successfully adapted pretrained language models (LMs) for sign language translation (sign-to-text), the reverse task—sign language generation (text-to-sign)—remains largely unexplored. In this work, we introduce a multilingual sign language model, Signs as Tokens (SOKE), which can generate 3D sign avatars autoregressively from text inputs using a pretrained LM. To align sign language with the LM, we leverage a decoupled tokenizer that discretizes continuous signs into token sequences representing various body parts. During decoding, unlike existing approaches that flatten all part-wise tokens into a single sequence and predict one token at a time, we propose a multi-head decoding method capable of predicting multiple tokens simultaneously. This approach improves inference efficiency while maintaining effective information fusion across different body parts. To further ease the generation process, we propose a retrieval-enhanced SLG approach, which incorporates external sign dictionaries to provide accurate word-level signs as auxiliary conditions, significantly improving the precision of generated signs. 



## Environment
Please run 
```
conda create python=3.10 --name soke
conda activate soke
pip install -r requirements.txt
```


## Data
### Continuous Sign Language Datasets
How2Sign: [raw videos](https://how2sign.github.io/) and [split files](https://drive.google.com/drive/folders/1sPhBwmiWCXLZSHtM3fpotbz3BDgoYmco?usp=sharing).

CSL-Daily: [raw videos](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/) and [split files](https://drive.google.com/drive/folders/17uPm6r5_DQ9CIYZonfwQLpw1XI8LeNEr?usp=drive_link). 

Phoenix-2014T: [raw videos](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) and [split files](https://drive.google.com/drive/folders/1Z2zjOH5wvwT7x_F6IycWAN-nh2wgJOx1?usp=sharing).

SMPL-X Poses can be downloaded from the project [homepage](https://2000zrl.github.io/soke/).


## Models
### Human Models
Please download human models (mano, smpl, smplh, and smplx) from [here](https://drive.google.com/file/d/1YIXddvvBJPQVRuKON2Xc9EEDXikRTteo/view?usp=sharing) and unzip them into deps/smpl_models. 

Download t2m evaluators via `sh prepare/download_t2m_evaluators.sh`.

Down t5 models via `sh prepare/prepare_t5.sh`. Note that this aims to avoid errors caused by the default config.

### Language Model
We use mBart-large-cc25, which can be downloaded [here](https://drive.google.com/drive/folders/1GnaHrI0PC4ZRr-GK3FS2GXcQwlrpA5Gi?usp=sharing). Put the chekpoint into `deps/mbart-h2s-csl-phoenix`


## Decoupled Tokenizer
### Training
```
python -m train --cfg configs/deto.yaml --nodebug
```

### Inference
```
python -m test --cfg configs/deto.yaml --nodebug
```
We also provide the [mean](https://drive.google.com/file/d/1NH-eVtS0nNjMjCwae-A1ii5sxj44C3bo/view?usp=sharing) and the [std](https://drive.google.com/file/d/1FHHWS0GPM2s6S2PB2JHv4ufdEbzezuKW/view?usp=sharing) of the SMPL-X poses. The checkpoint of the tokenizer is available [here](https://drive.google.com/file/d/18HdPeXwz4O6LY4FZMC5BZ9rja4pcUTFk/view?usp=sharing).


## Autoregressive Multilingual Generator
### Training
```
python -m get_motion_code --cfg configs/soke.yaml --nodebug
python -m train --cfg configs/soke.yaml --nodebug  #Note that please first update the path of the tokenizer's checkpoint.
```

### Inference
```
python -m test --cfg configs/soke.yaml --task t2m  #you can set SAVE_PREDICTIONS in the config file to True if you want to save them.
```

## Visualizations
Simple visualizations if meshes can be done by running
```
python -m vis_mesh --cfg=configs/soke.yaml --demo_dataset=csl
```
For colorful visualiations, please refer to the configurations of [BlenderToolbox](https://github.com/HTDerekLiu/BlenderToolbox), and run
```
python vis_blender.py
```

## Acknowledgements
We sincerely thank the open-sourced codes of these works where our code is based on: [MotionGPT](https://github.com/OpenMotionLab/MotionGPT/), [ProgressiveTransformer](https://github.com/BenSaunders27/ProgressiveTransformersSLP), [WiLoR](https://github.com/rolpotamias/WiLoR), and [OSX](https://github.com/IDEA-Research/OSX/). 

Please contact [r.zuo@imperial.ac.uk](mailto:r.zuo@imperial.ac.uk) for further questions.


## Citations
```
@inproceedings{zuo2025soke,
    title={Signs as Tokens: A Retrieval-Enhanced Multilingual Sign Language Generator},
    author={Zuo, Ronglai and Potamias, Rolandos Alexandros and Ververas, Evangelos and Deng, Jiankang and Zafeiriou, Stefanos},
    booktitle={ICCV},
    year={2025}
}
```