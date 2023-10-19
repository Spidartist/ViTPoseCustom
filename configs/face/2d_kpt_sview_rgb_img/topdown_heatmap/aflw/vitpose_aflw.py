_base_ = [
    '../../../../body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py'
]

channel_cfg = dict(
    num_output_channels=19,
    dataset_joints=19,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
    ])

data_root = 'data/aflw'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='FaceAFLWDataset',
        ann_file=f'{data_root}/annotations/face_landmarks_aflw_train.json',
        img_prefix=f'{data_root}/images/'),
    val=dict(
        type='FaceAFLWDataset',
        ann_file=f'{data_root}/annotations/face_landmarks_aflw_test.json',
        img_prefix=f'{data_root}/images/'),
    test=dict(
        type='FaceAFLWDataset',
        ann_file=f'{data_root}/annotations/face_landmarks_aflw_test.json',
        img_prefix=f'{data_root}/images/')
)
