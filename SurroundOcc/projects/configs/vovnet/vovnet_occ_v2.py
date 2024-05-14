_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
occ_size = [200, 200, 16]
use_semantic = True


#img_norm_cfg = dict(
#    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False)

class_names =  ['barrier','bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade','vegetation']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

#_dim_ = [128, 256, 512]
#_ffn_dim_ = [256, 512, 1024]
#volume_h_ = [100, 50, 25]
#volume_w_ = [100, 50, 25]
#volume_z_ = [8, 4, 2]
#_num_points_ = [2, 4, 8]
#_num_layers_ = [1, 3, 6]

_dim_ = [128, 256, 512]
_ffn_dim_ = [256, 512, 1024]
volume_h_ = [200, 100, 50]
volume_w_ = [200, 100, 50]
volume_z_ = [16, 8, 4]
_num_points_ = [2, 4, 8]
_num_layers_ = [1, 3, 6]

model = dict(
    type='SurroundOcc',
    use_grid_mask=True,
    use_semantic=use_semantic,
    img_backbone=dict(
       type='VoVNet',
       spec_name='V-99-eSE',
       norm_eval=True,
       frozen_stages=1,
       input_ch=3,
       out_features=['stage3', 'stage4', 'stage5']),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 768, 1024],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='OccHead',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=17,
        #num_classes=18,
        #conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64],
        #conv_output=[256, _dim_[1], 128, _dim_[0], 64, 64, 32],
        #out_indices=[0, 2, 4, 6],
        #upsample_strides=[1,2,1,2,1,2,1],
        conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0]],#[512,256,256,128,128]
        conv_output=[256, _dim_[1], 128, _dim_[0], 64],#[256,256,128,128,64]
        out_indices=[0, 2, 4],#在0，2，4层输出
        upsample_strides=[1,2,1,2,1],#1,3层上采样
        embed_dims=_dim_,
        img_channels=[512, 512, 512],
        use_semantic=use_semantic,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            encoder=dict(
                type='OccEncoder',
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=1),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
),
)

dataset_type = 'CustomNuScenesOccDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadOccupancy', use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'gt_occ'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancy', use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img','gt_occ'])
]

find_unused_parameters = True
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file='data/nuscenes_infos_val.pkl',
             pipeline=test_pipeline,  
             occ_size=occ_size,
             pc_range=point_cloud_range,
             use_semantic=use_semantic,
             classes=class_names,
             modality=input_modality),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file='data/nuscenes_infos_val.pkl',
              pipeline=test_pipeline, 
              occ_size=occ_size,
              pc_range=point_cloud_range,
              use_semantic=use_semantic,
              classes=class_names,
              modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

#optimizer = dict(type='AdamW', lr=2e-5, weight_decay=1e-2)
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    #lr=2e-5,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=30, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-4)
total_epochs = 15
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/dd3d_det_final.pth'
#load_from = 'work_dirs/surroundocc_v1/latest.pth'
#load_from = 'ckpts/surroundocc_org.pth'
#load_from = '/data/models/work_dirs/surroundocc/epoch_2.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
#fp16 = dict(loss_scale='dynamic')
fp16 = dict(loss_scale=512.0)
