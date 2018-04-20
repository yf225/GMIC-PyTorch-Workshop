# coding:utf8
import sys, os
import torch as t
from data import get_data
from model import PoetryModel
from torch import nn
import tqdm
from torchnet import meter


class Config(object):
    pickle_path = 'tang.npz'  # 预处理好的二进制文件
    author = None  # 只学习某位作者的诗歌
    constrain = None  # 长度限制
    category = 'poet.tang'  # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3
    weight_decay = 1e-4
    epoch = 20
    batch_size = 128
    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    max_gen_len = 200  # 生成诗歌最长长度
    debug_file = '/tmp/debugp'
    model_path = None  # 预训练模型路径
    theme_words = '细雨鱼儿出，微风燕子斜。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '闲云潭影日悠悠'  # 诗歌开始
    acrostic = False  # 是否是藏头诗
    model_prefix = 'checkpoints/tang'  # 模型保存路径


opt = Config()


def generate(model, start_words, ix2word, word2ix, theme_words=None):
    """
    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：

    """
    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    hidden = None

    if theme_words:
        for word in theme_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results


def gen_acrostic(model, start_words, ix2word, word2ix, theme_words=None):
    """
    生成藏头诗
    start_words : u'深度学习'
    生成：
    深木通中岳，青苔半日脂。
    度山分地险，逆浪到南巴。
    学道兵犹毒，当时燕不移。
    习根通古岸，开镜出清羸。
    """
    results = []
    start_word_len = len(start_words)
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    hidden = None

    index = 0  # 用来指示已经生成了多少句藏头诗
    # 上一个词
    pre_word = '<START>'

    if theme_words:
        for word in theme_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        if (pre_word in {u'。', u'！', '<START>'}):
            # 如果遇到句号，藏头的词送进去生成

            if index == start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:
                # 把藏头的词作为输入送入模型
                w = start_words[index]
                index += 1
                input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            input = input.data.new([word2ix[w]]).view(1, 1)
        results.append(w)
        pre_word = w
    return results


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)

    # 获取数据
    data, word2ix, ix2word = get_data(opt)
    data = t.from_numpy(data)
    dataloader = t.utils.data.DataLoader(data,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=1)

    # 模型定义
    model = PoetryModel(len(word2ix), 128, 256)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))

    loss_meter = meter.AverageValueMeter()

    for epoch in range(opt.epoch):
        loss_meter.reset()
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):

            # 训练
            data_ = data_.long().transpose(1, 0).contiguous()
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

        print("Epoch: ", epoch, ", Loss: ", loss_meter.value()[0])

        t.save(model.state_dict(), '%s_loss_%s_epoch_%s.pth' % (opt.model_prefix, loss_meter.value()[0], epoch))


def gen(**kwargs):
    """
    提供命令行接口，用以生成相应的诗
    """

    for k, v in kwargs.items():
        setattr(opt, k, v)
    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), 128, 256);
    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            theme_words = opt.theme_words if opt.theme_words else None
        else:
            start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            theme_words = opt.theme_words.encode('ascii', 'surrogateescape').decode(
                'utf8') if opt.theme_words else None
    else:
        start_words = opt.start_words.decode('utf8')
        theme_words = opt.theme_words.decode('utf8') if opt.theme_words else None

    start_words = start_words.replace(',', u'，') \
        .replace('.', u'。') \
        .replace('?', u'？')

    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model, start_words, ix2word, word2ix, theme_words)
    print(''.join(result))


if __name__ == '__main__':
    import fire
    fire.Fire()
