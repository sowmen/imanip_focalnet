import torch
from torch import nn
from .srm_kernel import setup_srm_layer
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

from .encoder.focalnet import FocalNet
from .decoder.lawin import LAWINHead
from .decoder.mmseg_layers import resize


class FocalWin(nn.Module):
    def __init__(self, num_classes=1):
        super(FocalWin, self).__init__()
        
        
        self.srm_conv = setup_srm_layer(3)
        
        self.bayer_conv = nn.Conv2d(3, out_channels=3, kernel_size=5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.bayer_conv.weight)
        
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(16, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        nn.init.xavier_uniform_(self.rgb_conv[0].weight)
        nn.init.xavier_uniform_(self.rgb_conv[1].weight)

        
        self.ela_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        nn.init.xavier_uniform_(self.ela_net[0].weight)
        nn.init.xavier_uniform_(self.ela_net[1].weight)


        self.backbone = FocalNet(in_chans=54, embed_dim=96, depths=(2,2,18,2), focal_levels=(2,2,2,2), focal_windows=(9,9,9,9),drop_path_rate=0.3)
        print(self.backbone.load_state_dict(self.load_focalnet_dict(), strict=False))

        self.avgpool = SelectAdaptivePool2d(pool_type="avg", flatten=True)
        self.classifier = nn.Linear(768, 1)
        nn.init.xavier_uniform_(self.classifier.weight)

        decode_params = dict(
                in_channels=[96, 192, 384, 768],
                dropout_ratio=0.1,
                norm_cfg=None,
                conv_cfg=None,
                act_cfg=dict(type='ReLU'),
                align_corners=False,
                concat_fuse=True,
                depth=1,
                decoder_params=dict(embed_dim=512, in_dim=512, reduction=2, proj_type='conv', use_scale=True, mixing=True),
        )
        self.decoder = LAWINHead(num_classes=num_classes, **decode_params)
        self.decoder.load_state_dict(self.load_lawin_dict(), strict=False)
    
    def load_focalnet_dict(self):
        state_dict = torch.load('/content/drive/MyDrive/imanip/weights/focalnet_small_srf_upernet_160k.pth')['state_dict']
        encoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone'):
                encoder_dict[k[9:]] = v
        del encoder_dict['patch_embed.proj.weight']
        return encoder_dict
    
    def load_lawin_dict(self):
        lawin_state_dict = torch.load('/content/drive/MyDrive/imanip/weights/lawin_B2_ade20k.pth')['state_dict']
        decoder_dict = {}
        for k, v in lawin_state_dict.items():
            if k.startswith('decode'):
                decoder_dict[k[12:]] = v

        delete_keys = ['linear_c4.proj.weight', 'linear_c3.proj.weight', 'linear_c2.proj.weight', 'linear_pred.weight', 'linear_pred.bias', 'linear_c1.proj.weight']
        for x in delete_keys:
            del decoder_dict[x]

        return decoder_dict


    def forward(self, im, ela):
        _, _, h, w = im.shape

        x1 = self.srm_conv(im)
        x2 = self.bayer_conv(im)
        x3 = self.rgb_conv(im)
        x_ela = self.ela_net(ela)


        _merged_input = torch.cat([x1, x2, x3, x_ela], dim=1)
        
        stage_outputs = self.backbone(_merged_input)
        reduced_feat = self.avgpool(stage_outputs[-1])
        class_tensor = self.classifier(reduced_feat)

        out_mask_tensor = self.decoder(stage_outputs)
        out_mask_tensor = resize(out_mask_tensor, (h,w))
        
        return class_tensor, out_mask_tensor