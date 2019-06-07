# [WIP] PyTorch Partial Convolution Inpainting
This is the Re-implementation of "Image Inpainting for Irregular Holes Using Partial Convolutions".

This is **NOT** an official implementation by the author.

## Result
![result](pconv_result.png)
From top to bottom, The input image, The mask image, The raw output, The outpu with ground truth except for mask area, and the ground truth image.


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


