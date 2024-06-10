## Pretrain ##
# SupBYOL
python main.py supbyol \
 -a resnet50 --framework 'supbyol' --warmup_epoch 0 --lr 0.4 \
 --dim 256 --hid-dim 4096 --K 8192 --m 0.996 --m-cos --T 0.0 \
 --fix-pred-lr --num_positive 0 --alpha 0.5 \
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 --dir '[your imagenet-folder with train and val folders]'

# SupSiam
 python main.py supsiam \
 -a resnet50 --framework 'supsiam' --warmup_epoch 40 --lr 0.4 \
 --dim 2048 --hid-dim 512 --K 8192 --m 0.0 --T 0.0 \
 --fix-pred-lr --num_positive 0 --alpha 0.5 \
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 --dir '[your imagenet-folder with train and val folders]'

# SupMoCo
python main.py supmoco \
 -a resnet50 --framework 'supmoco' --warmup_epoch 0 --lr 0.2 \
 --dim 128 --hid-dim 0 --K 8192 --m 0.999 --T 0.07 \
 --num_positive 0 --alpha 1.0 \
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 --dir '[your imagenet-folder with train and val folders]'

# SupCon
python main.py supcon \
 -a resnet50 --framework 'supcon' --warmup_epoch 0 --lr 0.3 \
 --dim 128 --hid-dim 0 --K 0 --m 0.0 --T 0.1 \
 --num_positive 0 --alpha 1.0 \
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 --dir '[your imagenet-folder with train and val folders]'

# BYOL
python main.py byol \
 -a resnet50 --framework 'byol' --warmup_epoch 0 --lr 0.4 \
 --dim 256 --hid-dim 4096 --K 0 --m 0.996 --m-cos --T 0.0 \
 --fix-pred-lr --num_positive 0 --alpha 1.0 \
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 --dir '[your imagenet-folder with train and val folders]'


# SimSiam
python main.py simsiam \
 -a resnet50 --framework 'simsiam' --warmup_epoch 40 --lr 0.4 \
 --dim 2048 --hid-dim 512 --K 0 --m 0.0 --T 0.0 \
 --fix-pred-lr --num_positive 0 --alpha 1.0 \
 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 --dir '[your imagenet-folder with train and val folders]'

##########################################################################################################

# Linear Evaluation with SupBYOL
python linear.py supbyol/imagenet100 \
 -a resnet50 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
 --dir '[your imagenet-folder with train and val folders]' \
 --pretrained '[path where pretrained model is saved]'

# Transfer Learning via Linear Evaluation with SupBYOL
for data in CIFAR10 CIFAR100 dtd food101 mit67 sun397
do
    python transfer.py supbyol/${data} -a resnet50 --data ${data} --metric top1 \
    --dir '[your dataset folder]' --pretrained '[path where pretrained model is saved]'
done

for data in caltech101 cub200 dog flowers102 pets
do
    python transfer.py supbyol/${data} -a resnet50 --data ${data} --metric class-avg \
    --dir '[your dataset folder]' --pretrained '[path where pretrained model is saved]'
done

# Few-shot Classification with SupBYOL
for K in 1 5
do
    for data in aircraft birds fc100 flowers texture omniglot traffic fungi
    do
        python fewshot.py supbyol/fewshot/5way-${K}shot/${data} -a resnet50 \
         --data ${data} --N 5 --K ${K} \
         --dir '[your dataset folder]' --pretrained '[path where pretrained model is saved]'
    done
done



