# MCTrack
> Official code for RGB-T multi-object tracker: MCTrack, TCSVT, 2025
>
> [**Multi-Stage Cross-Modality Feature Interaction
for RGB-Thermal Multi-Object Tracking**](https://ieeexplore.ieee.org/document/11175188),            
> Jianbo Ma; Hui Luo; Shuaicheng Niu; Peilin Zhao; Yunfeng Liu; Yuxing We

## 🗼 Pipeline of MCTrack

![](demo/MOT.png)

## 💁 Get Started

### Environment preparation

> git clone https://github.com/ydhcg-BoBo/MCTrack.git

0. create a new conda environment. 

    ~~~
    conda create --name MCTrack python=3.7
    conda activate MCTrack
    ~~~

1. Install PyTorch:

    ~~~
    conda install pytorch torchvision -c pytorch
    ~~~
    
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):

    ~~~
    pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    ~~~

4. Install the requirements

    ~~~
    pip install -r requirements.txt
    ~~~
    
5. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/)).

    ~~~
    cd $PFTrack_ROOT/src/lib/model/networks/
    # git clone https://github.com/CharlesShang/DCNv2/ # clone if it is not automatically downloaded by `--recursive`.
    cd DCNv2
    ./make.sh
    ~~~

### Trained Models
| Dataset    |Link                                                                                    |
|------------|----------------------------------------------------------------------------------------|
| [VTMOT](https://github.com/wqw123wqw/PFTrack) |[BaiduDrive](https://pan.baidu.com/s/1NyQaVxJiC0Uhigws-c3ISw?pwd=2026) (password: 2026) |
| [UniRTL](https://github.com/Liamzh0331/Unismot) |[BaiduDrive](https://pan.baidu.com/s/1fNTRUa6NBuH31CRUgPyRNQ?pwd=2026) (password: 2026) |

### Train
```
sh experimets/train.sh
```

### Test
```
sh experimets/track.sh
```

## 📚 Citation
>If you find this code useful, please star the project and consider citing:
```bibtex
@ARTICLE{MCTrack,
  author={Ma, Jianbo and Luo, Hui and Niu, Shuaicheng and Zhao, Peilin and Liu, Yunfeng and Wei, Yuxing and Zhang, Jianlin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Multi-Stage Cross-Modality Feature Interaction for RGB-Thermal Multi-Object Tracking}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Transformers;Object detection;Videos;Object tracking;Trajectory;Robustness;Decoding;Data mining;Training;Multi-stage fusion;cross-modality features;RGB-Thermal object tracking;multi-object tracking},
  doi={10.1109/TCSVT.2025.3612499}}

```

## 🙏 Acknowledgement
A large part of the code is borrowed from [PFTrack](https://github.com/wqw123wqw/PFTrack).
Thanks for their wonderful works.

>If you have any questions related to the paper and the code please contact me.
> 