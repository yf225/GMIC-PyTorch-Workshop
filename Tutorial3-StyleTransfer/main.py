# coding:utf8

import torch
import torchvision as tv

from torch.utils import data
from transformer_net import TransformerNet
import utils
from PackedVGG import Vgg16
from torch.nn import functional as F
import os
import argparse

from visdom import Visdom

vis = Visdom()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--stylize', action='store_true')
parser.add_argument('--style-path', type=str, help='path to style image') # 风格图片存放路径
parser.add_argument('--model-path', type=str, help='path to pretrained model') # 预训练模型的路径
parser.add_argument('--content-path', type=str, help='path to image to be transformed') # 需要进行分割迁移的图片
parser.add_argument('--result-path', type=str, help='path to transformed image') # 风格迁移结果的保存路径
args = parser.parse_args()


def train():
    image_size = 256
    batch_size = 8
    data_root = 'data/'  # 数据集存放路径：data/coco/a.jpg
    lr = 1e-3  # 学习率
    epoches = 2  # 训练epoch数
    content_weight = 1e5  # content_loss 的权重
    style_weight = 1e10  # style_loss的权重

    # 数据加载
    transfroms = tv.transforms.Compose([
        tv.transforms.Scale(image_size),
        tv.transforms.CenterCrop(image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 255)
    ])
    dataset = tv.datasets.ImageFolder(data_root, transfroms)
    dataloader = data.DataLoader(dataset, batch_size)

    # 转换网络
    transformer = TransformerNet()
    if args.model_path:
        transformer.load_state_dict(torch.load(args.model_path, map_location=lambda _s, _: _s))

    # 损失网络 Vgg16
    vgg = Vgg16().eval()

    # 优化器
    optimizer = torch.optim.Adam(transformer.parameters(), lr)

    # 风格图片的gram矩阵
    style.requires_grad = False
    features_style = vgg(style)
    gram_style = [utils.gram_matrix(y.data) for y in features_style]

    for epoch in range(epoches):
        for ii, (x, _) in enumerate(dataloader):

            # 训练
            optimizer.zero_grad()
            y = transformer(x)
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            features_y = vgg(y)
            features_x = vgg(x)

            # content loss
            content_loss = content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)

            # style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = utils.gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

        # 保存模型
        torch.save(transformer.state_dict(), 'checkpoints/%s_style.pth' % epoch)


def stylize():
    # 图片处理
    content_image = tv.datasets.folder.default_loader(args.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    content_image.requires_grad = False

    # 模型
    style_model = TransformerNet().eval()
    style_model.load_state_dict(torch.load(args.model_path, map_location=lambda _s, _: _s))

    # 风格迁移与保存
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    output_image = (output_data / 255).clamp(min=0, max=1)
    tv.utils.save_image(output_image, args.result_path)

    # 在Visdom中显示风格图片
    style = utils.get_style_data(args.style_path)
    style_image = (style[0] * 0.225 + 0.45).clamp(min=0, max=1)
    vis.image(
        style_image,
        win='style',
        opts=dict(title='风格图片')
        )

    # 在Visdom中显示迁移前图片
    print(content_image.shape)
    vis.image(
        content_image.cpu().data.squeeze(0),
        win='original',
        opts=dict(title='原图片')
        )

    # 在Visdom中显示迁移后图片
    print(output_image.shape)
    vis.image(
        output_image,
        win='transformed',
        opts=dict(title='迁移后图片')
        )

if __name__ == '__main__':
    if args.train:
        train()
    elif args.stylize:
        stylize()
