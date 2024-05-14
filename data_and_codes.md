# RoboDrive Challenge Track3数据说明

## 1.Label文件生成

代码位置：tools/phase2_pkl.py

**需要修改的参数**：

filePath1，roboDrive sample_scenes.pkl路径；

filePath2，robodrive_infos_temporal_test.pkl路径(https://github.com/robodrive-24/toolkit/tree/main/track-3)；

evalFilePath,生成的robodrive 索引文件。

**代码说明**：

主要内容为将nuscenes图片路径改为roboDrive phase2图片路径和token

token方式为**场景key + 样本key**，用"-"分割符隔开，方便后面解析。

```
for cam_key in sample['cams'].keys():
    sample['cams'][cam_key]['data_path'] = 'data/robodrive-phase2/'+key+'/samples/'+cam_key+"/"+sample['cams'][cam_key]['data_path'].split("/")[-1]
sample['token'] = key+"-"+sample['token']
```

## 2.代码执行

`./tools/dist_submit.sh /data/models/work_dirs/resnet101/resnet101_occ_v1.py /data/models/work_dirs/resnet101/latest.pth 1`

dist_submit.sh内容：

`python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/submit.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --eval bbox`

submit.py提交代码，位置：SurroundOcc_eval/tools/submit.py

大概在237行，由于评估方法基本不变，因此**pkl文件设置在代码里**覆盖掉config文件，在编程完成后基本没有变动。

` pklPath="robodrive-phase2.pkl"
 data_cfg = cfg.data.test
 data_cfg['ann_file'] = pklPath`

提交结果实现在SurroundOcc_eval/projects/mmdet3d_plugin/surroundocc/apis/test_submit.py,基于api的test.py修改

主要修改custom_multi_gpu_test函数：

custom_multi_gpu_test不再依赖其他方法，推理完成后直接保存结果，对结果key解析为场景key + 样本key(cur_key,cur_token = sample_token.split("-"))

    model.eval()
    
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    npyData = dict()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
    
            result = model(return_loss=False, rescale=True, **data)
            #robodrive result
            sample_token = data['img_metas'].data[0][0]['sample_idx']
            print()
            print(i,sample_token)
            cur_key,cur_token = sample_token.split("-")
            cur_data = result['prediction'].cpu().numpy().astype(np.uint8)
            if cur_key not in npyData:
                npyData[cur_key] = dict()
            npyData[cur_key][cur_token] = cur_data
    with open(savedPath,"wb") as f:
        pickle.dump(npyData,f)

## 3.其他说明

Occ voxel size为200x200x16时代码与Surround Occ官方项目无法适配，原因是原版代码是从25->50->100->200,上采样固定次数，在修改后会变为50->100->200->400。主要实现在SurroundOcc_eval/projects/mmdet3d_plugin/surroundocc/dense_heads/occ_head.py

if i not in self.out_indices:

修改为：

if (i not in self.out_indices) and (i < len(self.deblocks) - 2):

适配本项目的所有模型。