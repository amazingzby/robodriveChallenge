#vovnet occ v1
#./tools/dist_train.sh ./projects/configs/vovnet/vovnet_occ_v1.py 2 ./work_dirs/vovnet_occ
#vovnet occ v2
#./tools/dist_train.sh ./projects/configs/vovnet/vovnet_occ_v2.py 2 ./work_dirs/vovnet_occ_v2
#resnet occ v1 H,W增大为200
./tools/dist_train.sh ./projects/configs/resnet101/resnet101_occ_v1.py 2 ./work_dirs/resnet101
#resnet occ v2 Frozen stage为2 + Focal Loss
#./tools/dist_train.sh ./projects/configs/resnet101/resnet101_occ_v2.py 2 ./work_dirs/resnet101_v2
#resnet occ v3
#./tools/dist_train.sh ./projects/configs/resnet101/resnet101_occ_v3.py 2 ./work_dirs/resnet101_v3
#resnet occ v1 plus
#./tools/dist_train.sh ./projects/configs/resnet101/resnet101_occ_v1_plus.py 2 ./work_dirs/resnet101_occ_v1_plus
