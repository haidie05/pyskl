model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=10, in_channels=256))
dataset_type = 'PoseDataset'
ann_file = 'E:/IndoorActionDataset/indoor_action_yolov8pose.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='PoseDataset',
            ann_file='E:/IndoorActionDataset/indoor_action_yolov8pose.pkl',
            pipeline=[
                dict(type='PreNormalize2D'),
                dict(type='GenSkeFeat', dataset='coco', feats=['j']),
                dict(type='UniformSample', clip_len=100),
                dict(type='PoseDecode'),
                dict(type='FormatGCNInput', num_person=1),
                dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['keypoint'])
            ],
            split='train')),
    val=dict(
        type='PoseDataset',
        ann_file='E:/IndoorActionDataset/indoor_action_yolov8pose.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(type='GenSkeFeat', dataset='coco', feats=['j']),
            dict(type='UniformSample', clip_len=100, num_clips=1),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=1),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='validation'),
    test=dict(
        type='PoseDataset',
        ann_file='E:/IndoorActionDataset/indoor_action_yolov8pose.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(type='GenSkeFeat', dataset='coco', feats=['j']),
            dict(type='UniformSample', clip_len=100, num_clips=10),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=1),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='test'))
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 24
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/stgcn++/stgcn++_indoor_yolov8pose/j'
gpu_ids = [0]
resume_from = './work_dirs/stgcn++/stgcn++_indoor_yolov8pose/j\latest.pth'
