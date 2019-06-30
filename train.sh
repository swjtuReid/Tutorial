
#使用GPU训练
CUDA_VISIBLE_DEVICES=0 python3 main.py --datadir ../Market-1501-v15.09.15/ --batchid 8 --batchimage 4--batchtest 32 --test_every 50 --epochs 350 --decay_type step_150_250_320 --loss 1*CrossEntropy --margin 1.2 --save example --nGPU 1  --lr 2e-4 --optimizer ADAM --random_erasing --reset --amsgrad

#使用CPU训练
python3 main.py --datadir ../Market-1501-v15.09.15/ --cpu --batchid 2 --batchimage 4 --batchtest 8 --test_every 50 --epochs 350 --decay_type step_150_250_320 --loss 1*CrossEntropy --margin 1.2 --save example --nGPU 1  --lr 2e-4 --optimizer ADAM --random_erasing --reset --amsgrad

#测试某个保存模型
