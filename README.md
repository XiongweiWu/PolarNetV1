# PolarNetV1

## Installation

You need first install cornernet to obtain the function of corner pooling. I will write an unified installation script and clean the code soon.

## Environment

- cuda-10.2
- pytorch-1.4.0
- torchvision-0.5.0
- python-3.6.9

## Models

*All ResNe(x)t based models are trained with 16 images in a mini-batch and frozen batch normalization (i.e., consistent with models in [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)).* \
*I re-implement the model and re-train all the models, and thus the results may be a bit different with the ones reported* \
*I mainly follow the setting of FCOS.

Model | Multi-scale training | FPS | AP (minival) | AP (test) |  Link
--- |:---:|:---:|:---:|:---:|:---:
PolarNet_R50_1x         | No | - | 39.6 | -| [download](https://drive.google.com/file/d/10dblLJyOlMLhMzBv65OC0sK0TTDXrIWF/view?usp=sharing)
PolarNet_R101_2x        | Yes | 6.7 | 43.7 | 44.1|[download](https://drive.google.com/file/d/1uPF_z2NdQ260N1dI9RbmcV7n6MqEC3EA/view?usp=sharing)
PolarNet_dcnv2_X101_2x | Yes | 4.3 | 47.3 |47.8| [download](https://drive.google.com/file/d/1rx8wwqbNO0xDU0lXtUSH6bLy4dyNqQpp/view?usp=sharing)

