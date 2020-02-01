# PyTorch Partial Convolution Inpainting
This is the Re-implementation of "Image Inpainting for Irregular Holes Using Partial Convolutions".

This is **NOT** an official implementation by the author.

## Result
![result](./figs/pconv_result.png)
From top to bottom, The input image, The mask image, The raw output, The outpu with ground truth except for mask area, and the ground truth image.


## Installation
### Clone this repo
```
git clone https://github.com/tanimutomo/partialconv.git
```

### Setup the environment
#### Local
The required libraries are written in `partialconv/docker/cpu_requirements.txt`.  

#### Docker
- Install [Docker](https://www.docker.com/) and docker compose.

- Build the container
```
docker-compose -f ./docker/docker-compose-{cpu/gpu}.yml build
```

#### Kronos (Recommend)
[kronos](https://github.com/d-hacks/kronos) is the environment for machine learning. kronos is the docker container based environment. And you can run the code more easily by using this.
Please try to use kronos!

#### Usage
- Install kronos
```
pip install kronos-ml
```

- Build the environment
```
kronos build
```
or if you want to run on gpu, please type:
```
kronos build --gpu
```


## Quick Run
### Download the pretrained model
You can donwload the [pretrained model](https://drive.google.com/file/d/1sooo-BLSNRUGWG_AB-lxh7xHgJ2bS29a/view?usp=sharing) which is the same as the model output the above images.  
Please put the pretrained model file to `partialconv/`.

### Run
- Local setup
```
python predict.py
```

- Docker
```
docker-compose -f ./docker/docker-compose-{cpu/gpu}.yml run experiment  python3 predict.py
```

- kronos
If you want to run on gpu, please add `--gpu` option.
```
kronos run predict.py (--gpu)
```

- Specify the image, mask and model paths
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


