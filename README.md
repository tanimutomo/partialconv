# PyTorch Partial Convolution Inpainting
This is the Re-implementation of "Image Inpainting for Irregular Holes Using Partial Convolutions".

This is **NOT** an official implementation by the author.

## Result
![result](./figs/pconv_result.png)
From top to bottom, The input image, The mask image, The raw output, The outpu with ground truth except for mask area, and the ground truth image.


## Setup
### Clone this repo
```
git clone https://github.com/tanimutomo/partialconv.git
```

### Install requirements
The required libraries are written in [`./requirements.txt`](./requirements.txt).  


## Quick Run
### Download the pretrained model
You can donwload the [pretrained model](https://drive.google.com/file/d/1sooo-BLSNRUGWG_AB-lxh7xHgJ2bS29a/view?usp=sharing) which is the same as the model output the above images.  
Please put the pretrained model file to `partialconv/`.

### Run
- Quick Run
```
python predict.py
```

- You can specify an image / a mask / a model paths
```
python predict.py --img <image_path> --mask <mask_path> --model <model_path>
```

## Train and Test
- Copy the config yaml file.
```
cd partialconv/
cp default-config.yml config.yml
```
- Customize `config.yml`
- Run `main.py`


## Information

### [Original Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf)

- Title
  - Image Inpainting for Irregular Holes Using Partial Convolutions
- Author
  - Guilin Liu / Fitsum A. Reda / Kevin J. Shih / Ting-Chun Wang / Andrew Tao / Bryan Catanzaro
- Affiliation
  - NVIDIA Corporation
- Official Implementation
  - https://github.com/NVIDIA/partialconv
- Other Information
  - [http://masc.cs.gmu.edu/wiki/partialconv](http://masc.cs.gmu.edu/wiki/partialconv)

### Reference Codes

I refered the following repositories for this implementation. This implementation integrate the strong points of these refered codes.

- https://github.com/NVIDIA/partialconv
- https://github.com/MathiasGruber/PConv-Keras
- https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/


