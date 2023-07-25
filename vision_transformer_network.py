import torch
import torch.nn as nn
from Dropout_new import DropPath
from functools import partial

#定义一个模型初始的Patch Embedding
class Patch_Embedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channel=3, embed_dim=768, norm_Layer=None):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        stride = patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        #第一步为完成Patch Embedding的映射从224*224*3通过卷积操作到14*14*768,相当于映射
        self.proj = nn.Conv2d(in_channel,embed_dim,stride=stride,kernel_size=patch_size)
        #第二步Flatten展开操作，直接使用flatten函数
        #第三步进行归一化结束，为下面的一个额外添加的分类的class token准备.
        self.norm = norm_Layer(embed_dim) if norm_Layer else nn.Identity()

        def forward(self,x):
            B,C,H,W = x.shape#(B--Batchsize--也就是一次处理多少张图像,C--RGB图像--3，H，W--图像高，宽（224，224）
            x = self.proj(x)#（B， C--768， H--14，W--14）其中的14可以通过卷积计算公式可得
            x = x.flatten(2)#展开HW完成合并操作(B, C, HW)
            x = x.transpose(1,2)#换个位置（B,HW,C）为了后面的注意力机制的计算
            x = self.norm(x)#对x进行归一化(B,HW,C）---（B,num_patches,C)---(B,196,768)
        #至此Patch_Embedding结束
            return x



class Attention_Score(nn.Module):
    def __init__(self,
                 dim,#数据总维度
                 num_heads=8,#注意力头数，通俗来说就是将X到W（注意力里的权重）到q之后直接划分成几个小q就是几头注意力
                 qkv_bias=False,#进行线性投影时候的偏置
                 attn_drop_ratio=0.,#Dropout的失效率，一种有效的正则化和防止神经元共同适应的办法
                 proj_drop_ratio=0.,
                 qk_scale=None):
        super(Attention_Score,self).__init__()
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self,x):
        B, N, C = x.shape#经过外添加的分类的class token后,num_patches+1--（B,num_patches+1,C)--(B,197,768)
        x = self.qkv(x)#准备从x中映射出三个QKV出来（B,3*（num_patches+1）,C)--(B,3*197,768)
        x = x.reshape(B, N, 3, self.num_heads, C//self.num_heads)#重新建立结构(B,197,3,8,96)
        qkv = x.permute(2, 0, 3, 1, 4)#(3,B,8,197,96)
        q, k, v = qkv[0], qkv[1], qkv[2] #x中映射出三个QKV

        #注意力计算公式操作
        attn = (q @ k.transpose(-2,-1)) * self.scale#（B,8,197,96)*（B,8,96,197)=(B,8,197,197)
        #这里就可以看出为什么之前要换位置，因为我们多维相乘，我们只进行最后两维，因此我们将num_heads=8,放在倒数第三列
        #这样也能保证不同的head之间不会进行交叉计算，保证稳定
        attn = attn.softmax(dim=-1)#最后一维其中 就是按行进行softmax--(B,8,197,197)
        attn = self.attn_drop(attn)#(B,8,197,197)

        x = (attn @ v)#继续计算(B,8,197,197)*（B,8,197,96)=（B,8,197,96)
        x = x.transpose(1, 2)#1，2维换个位置（B,197,8,96)
        x = x.reshape(B, N, C)#重新变回x初始样子--(B,197,768)
        x = self.proj(x)#在进行一次线性映射--(B,197,768)
        x = self.proj_drop(x)#Dropout有效的正则化和防止神经元共同适应
        return x


class Mlp(nn.Module):
     def __init__(self, in_feature, hidden_features=None, out_features=None, act_layer=nn.GELU,drop=0.):
         super().__init__()
         self.fc1 = nn.Linear(in_feature,hidden_features)
         self.act = act_layer()
         self.fc2 = nn.Linear(hidden_features,out_features)
         self.drop = nn.Dropout(drop)
#定义一个熟悉的Mlp结果
     def forward(self,x):
         x = self.fc1(x)
         x = self.act(x)
         x = self.drop(x)
         x = self.fc2(x)
         x = self.drop(x)
         return x

class Vision_Transformer_Block(nn.Module):
    def __init__(self,
                 dim,  # 数据总维度
                 num_heads=8,  # 注意力头数，通俗来说就是将X到W（注意力里的权重）到q之后直接划分成几个小q就是几头注意力
                 qkv_bias=False, # 进行线性投影时候的偏置
                 qk_scale=None,
                 drop_ratio=0.,#注意力计算模块中的Dropout的失效率--后
                 attn_drop_ratio=0.,  # #注意力计算模块中的Dropout的失效率--前
                 Dropout_ratio=0.,
                 mlp_ratio=4,
                 act_layer=nn.GELU,
                 norm_layer=nn.Linear):
        super(Vision_Transformer_Block,self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_Score(
                 dim,#数据总维度
                 num_heads=num_heads,#注意力头数，通俗来说就是将X到W（注意力里的权重）到q之后直接划分成几个小q就是几头注意力
                 qkv_bias=qkv_bias,#进行线性投影时候的偏置
                 attn_drop_ratio=attn_drop_ratio,#Dropout的失效率，一种有效的正则化和防止神经元共同适应的办法
                 proj_drop_ratio=drop_ratio,
                 qk_scale =qk_scale)
        # 作者说这里用这个nn.dropout效果要好,虽然这里并没有使用Dropout_ratio=0.，可以进行对比实验
        self.drop_path = DropPath(Dropout_ratio) if Dropout_ratio  else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_features = int(dim * mlp_ratio)#中间的hidden_features设计为4*dim
        self.Mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_features, act_layer=act_layer, drop=drop_ratio)

        def forward(self, x):
            x = self.drop_path(self.attn(self.norm1(x))) + x#残差连接
            x = self.drop_path(self.mlp(self.norm2(x)))+ x
            return x


class Vision_Transformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channel=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=Patch_Embedding, norm_layer=None,
                 act_layer=None):
        super(Vision_Transformer,self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_channel=in_channel, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches#N---14*14=196

        self.cls_token = nn.Parameter(torch.zero(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Vision_Transformer_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes)

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)


    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token,x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x

#初始化nn.Conv2d,nn.Linear,nn.LayerNorm
def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    代码来自bilibili up主：
    vedio:https://www.bilibili.com/video/BV1AL411W7dT/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=bbb6fff63daa8014a7dbb0710681db68
    """
    model = Vision_Transformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model

















