# cross-domain

Implement of `Cross-Domain Image Captioning via Cross-Modal Retrieval`  <!-- linebreak -->
`and Model Adaptation`

### Requirements:
- Python 3.8
- torch 1.7.0
- fairseq 0.10.1
- tqdm 

### Usage:
  * extract features for training: run `src/utils/extract_image_feature.py`
  * train image captioning model: `src/main/caption/train_image.py`
  * may need to put the content of [coco-caption](https://github.com/tylin/coco-caption) in `src/coco-caption` and change java path in `src/config.py` for evaluation
  The working directory is `src`
  
#### Citation
  ```
  @article{zhao@cross,
  author={Wentian Zhao and Xinxiao Wu and Jiebo Luo},
  title={Cross-Domain Image Captioning via Cross-Modal Retrieval and Model Adaptation}, 
  journal={IEEE Transactions on Image Processing}, 
  year={2021},
  volume={30},
  pages={1180-1192},
  publisher={IEEE}
  }
  ```
