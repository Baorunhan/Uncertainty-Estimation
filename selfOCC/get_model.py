import pickle
import torch
import sys
import os
from selfmodel.Net import Net
from selfmodel.CustomeFPN import CustomFPN
from selfmodel.ResNet import *
from selfmodel.view_transformer import LSSViewTransformer
from selfmodel.bev_encoder_FPN import bev_encoder_with_FPN
from selfmodel.bev_occ_head import BEVOCCHead2D
 
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
# from structure.model.backbones import ResNet
# from structure.model.fpn import CustomFPN

if __name__ == '__main__':
    with open('dummy_data.pkl', 'rb') as f:  # with语句自动关闭文件，更安全
        data_input = pickle.load(f)
        
    data = {}    
    data.update({"img_inputs":[data_input['img_inputs'][0][0].cuda(),
                        data_input['img_inputs'][0][1].cuda(),
                        data_input['img_inputs'][0][2].cuda(),
                        data_input['img_inputs'][0][3].cuda(),
                        data_input['img_inputs'][0][4].cuda(),
                        data_input['img_inputs'][0][5].cuda(),
                        data_input['img_inputs'][0][6].cuda()]})
    
    data.update({"img_metas": data_input['img_metas'][0]})
    data_input['points'][0].data[0][0] = data_input['points'][0].data[0][0].cuda()
    data.update({"points": data_input['points'][0]})
    ### 前馈 网络 ###
    
    grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
    }
    
    model_predict = Net(
        training=True,
        #主干提取网络
        backbone=resnet18(),
        #FPN金字塔
        neck=CustomFPN(in_channels=[256, 512],
        out_channels=256,
        num_outs=1,
        start_level=0,
        end_level=-1,
        out_ids=[0],
        add_extra_convs=False,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='relu'),
        upsample_cfg=dict(mode='nearest')),
        #视角转换
        img_view_transformer=LSSViewTransformer(
        out_channels=64,
        in_channels=256,                    
        input_size=(256, 704), 
        grid_config=grid_config,
        collapse_z=True,
        downsample=16
        ),
        #bev主干网络
        img_bev_encoder=bev_encoder_with_FPN(
        numC_input=64,
        in_channels=64 * 2 + 64 * 8,
        num_channels=[64 * 2, 64 * 4, 64 * 8],
        num_layer=[2, 2, 2]
        ),
        #占据头
        occ_head=BEVOCCHead2D(
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False
        )
        ).to('cuda')
    result = model_predict(**data)
