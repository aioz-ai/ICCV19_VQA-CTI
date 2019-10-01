# Compact Trilinear Interaction for Visual Question Answering- CTI

This repository is the implementation of **Compact Trilinear Interaction for Visual Question Answering**. 

In Visual Question Answering (VQA), answers have a great correlation with question meaning and visual contents. Thus, to selectively utilize image, question and answer information, we propose a novel trilinear interaction model which simultaneously learns high level associations between these three inputs. In addition, to overcome the interaction complexity, we introduce a multimodal tensor-based PARALIND decomposition which efficiently parameterizes trilinear interaction between the three inputs. Moreover, knowledge distillation is applied in Free-form Opened-ended VQA. It is not only for reducing the computational cost and required memory but also for transferring knowledge from trilinear interactionmodel to bilinear interaction model. The extensive experiments on benchmarking datasets TDIUC, VQA-2.0, and Visual7W show that the proposed compact trilinear interaction model achieves state-of-the-art results on all three datasets.

For free-form opened-ended VQA task, our proposal achieved **67.4** on [VQA-2.0 dataset](https://visualqa.org/download.html) and **87.0** on [TDIUC dataset](https://kushalkafle.com/projects/tdiuc) in VQA accuracy metric. 

For multiple choice VQA task, our proposal achieved **72.3** on [Visual7W dataset](https://github.com/yukezhu/visual7w-toolkit) in MC-VQA accuracy metric proposed in [here](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhu_Visual7W_Grounded_Question_CVPR_2016_paper.pdf). 

In this repo, we provide pretrained models and all the code needed to reproduce the experiments from our [ICCV 2019 paper](http://arxiv.xxx).

This repository is based on and inspired by @Jin-Hwa Kim's [work](https://github.com/jnhwkim/ban-vqa). We sincerely thank for their sharing of the codes.

## Summary

* [The proposed model](#the-proposed-model)
* [Prerequisites](#prerequisites)
* [Preprocessing](#preprocessing)
* [Training phase](#training-phase)
* [Testing phase](#testing-phase)
* [Citation](#citation)
* [License](#license)
* [Acknowledgment](#acknowledgment)
* [More information](#more-information)

## The proposed model
### 1. For free-form opened-ended VQA task
<img src="misc/Free-form-Tri-attn.png" width="900" height="600">

### 2. For multiple choice VQA task
<img src="misc/Multiple-Choice-VQA.png" width="950" height="600">

## Prerequisites

Please install dependence package by run following command:
```
pip install -r requirements.txt
```
## Preprocessing

All data should be downloaded via [link](https://vision.aioz.io/d/965325537ca642a3a822/). The downloaded file should be extracted to `data_vqa/`, `data_TDIUC/`, and `data_v7w/` directory for `VQA`, `TDIUC`, `V7W dataset` respectively.

## Training phase
### 1. For free-form opened-ended task in VQA-2.0 dataset

To train the model using train set of **VQA 2.0 dataset**, please follow:
```
$ python3 main_OE.py --model ban --batch_size 256 --gamma 2 --distillation --T 5 --alpha 0.005 --output saved_models/VQA2.0/ban
```
To train the model using both train set and validation set  of **VQA 2.0 dataset**, please follow:
```
$ python3 main_OE.py --model ban --batch_size 256 --gamma 2 --distillation --use_both --T 5 --alpha 0.005 --output saved_models/VQA2.0/ban_trainval
```
### 2. For free-form opened-ended task in TDIUC
The model can be trained via:
```
$ python3 main_OE.py --model ban --batch_size 256 --gamma 2 --use_TDIUC --TDIUC_dir data_TDIUC --distillation --T 3 --alpha 0.3 --output saved_models/TDIUC/ban
```

**Note:** The teacher logit file for **VQA 2.0** and **TDIUC dataset** should be downloaded via [link](https://vision.aioz.io/d/9951e206d5ae4c4a97a1/) and extracted to `data_vqa/` and `data_TDIUC/` directory respectively to make the knowledge distillation learning feasible.

### 3. For multiple choice task
We provide the training method for CTI model in multiple choice VQA task on Visual7W dataset.
```
$ python3 main_MC.py --model cti --batch_size 64 --gamma 2 --output saved_models/V7W/cti
```
**Note:** The training scores will be printed every epoch. Trained model are stored at  `save_models\`

## Testing phase
We also provide the pretrained models reported as the best single models in the paper over different dataset.

### 1. For the pretrained model tested on VQA-2.0 dataset. 
Please download the [link](https://vision.aioz.io/d/b3a905767b334284b33d/) and move to `saved_models/VQA2.0/`. The trained model can be tested in VQA test-dev set via: 
```
$ python3 test_OE.py --model ban --batch_size 256 --input saved_models/VQA2.0/ban
```
#### Submit predictions on EvalAI

For VQA 2.0 dataset, it is not possible to automaticaly compute the accuracies on the testset. You need to submit a json file on the [EvalAI platform](https://evalai.cloudcv.org/web/challenges/challenge-page/163/submission). The evaluation step on the testset creates the json file that contains the prediction of the model on the full testset. For instance: `results/test2015_banc1024_epoch12.json`. To get the accuracies on testdev or test sets, you must submit this file.

### 2. For the pretrained model tested on TDIUC dataset.
Please download the [link](https://vision.aioz.io/d/a4b48764dd4f485b9b7b/) and move to `saved_models/TDIUC/`. The trained model can be tested in TDIUC validation set via: 
```
$ python3 test_OE.py --model ban --batch_size 256 --use_TDIUC --TDIUC_dir data_TDIUC --input saved_models/TDIUC/ban
```

### 3. For the pretrained model tested on Visual7W dataset`.
Please download the [link](https://vision.aioz.io/d/54b8e39ab7b343a7997e/) and move to `saved_models/V7W/`. The trained model can be tested in VQA test set via: 
```
$ python3 test_MC.py --model cti --batch_size 64 --input saved_models/V7W/cti --epoch 10
```

**Note:** The result json file can be found in the directory `results/`.

## Citation

If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:

```
@inproceedings{aioz_cti_iccv19,
  author={Tuong Do, Thanh-Toan Do, Huy Tran, Erman Tjiputra, Quang D. Tran},
  title={Compact Trilinear Interaction for Visual Question Answering},
  booktitle = {ICCV},
  year={2019}
}
```
## License
MIT License

## Acknowledgment

Special thanks to the authors of [VQA2](https://visualqa.org/download.html), [TDIUC](https://kushalkafle.com/projects/tdiuc), and [Visual7W](https://github.com/yukezhu/visual7w-toolkit), the datasets used in this research project.

## More information
AIOZ AI Homepage: https://ai.aioz.io
