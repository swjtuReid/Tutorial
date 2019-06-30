# 环境
* OS: Ubuntu
* Framework: Pytorch

# 依赖
你可以使用pip install 安装所有的依赖
* Python >= 3.5
* PyTorch >= 0.4.0
* TorchVision
* Matplotlib
* Argparse
* Sklearn
* Pillow
* Numpy
* Scipy
* Tqdm

# 运行相关
## pytorch使用：
1. 定义网络结构（model/base.py: Class Base）
2. 网络前向计算（model/base.py: def forward())
3. 使用损失函数优化网络（loss/\__init__.py)
4. 反向传播更新参数 （trainer.py: def train():loss.backward())

## 训练
### 开始训练
运行命令 sh train.sh
### 参数设置
详细的参数设置见option.py，一些重要的参数设置如下：
* datadir：训练集路径, save: 模型保存路径
* batchid * batchimage == batchsize：训练时的batchsize
* batchtest：测试时的batchsize
* test_every: 每训练test_every个周期测试一次模型
* lr: 初始学习率，lr_decay:学习率衰减周期
* 如果你使用GPU，CUDA_VISIBLE_DEVICES指定GPU，例如：

```
CUDA_VISIBLE_DEVICES=0 python3 main.py --datadir ../Market-1501/ --batchid 8 --batchimage 4 --batchtest 16 --test_every 50 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy --margin 1.2 --save example 1 --nGPU 1  --lr 2e-4 --optimizer ADAM --random_erasing --reset --amsgrad
```
* 如果你使用CPU，添加cpu参数，例如

```
python3 main.py --datadir ../Market-1501/ --batchid 8 --batchimage 4 --cpu --batchtest 16 --test_every 50 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy --margin 1.2 --save example 1 --lr 2e-4 --optimizer ADAM --random_erasing --reset --amsgrad
```

