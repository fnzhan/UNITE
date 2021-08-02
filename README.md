# UNITE
Unbalanced Feature Transport for Exemplar-based Image Translation (CVPR 2021)  <br>

## Preparation
Clone the Synchronized-BatchNorm-PyTorch repository.
```
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

**Pretrained VGG model** Download from [here](https://drive.google.com/file/d/1fp7DAiXdf0Ay-jANb8f0RHYLTRyjNv4m/view?usp=sharing), move it to `models/`. This model is used to calculate training loss.

## Pretrained Models
Pre-trained models will be released soon with the extended version.

### Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{zhan2021unite,
  title={Unbalanced Feature Transport for Exemplar-based Image Translation},
  author={Zhan, Fangneng and Yu, Yingchen and Cui, Kaiwen and Zhang, Gongjie and Lu, Shijian and Pan, Jianxiong and Zhang, Changgong and Ma, Feiying and Xie, Xuansong and Miao, Chunyan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
