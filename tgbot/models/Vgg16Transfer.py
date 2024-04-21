import asyncio
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from PIL import Image

class VGG16(nn.Module):
    def __init__(self, pool='max'):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, layers):
        out = {}
        out['relu1_1'] = F.relu(self.conv1_1(x))
        out['relu1_2'] = F.relu(self.conv1_2(out['relu1_1']))
        out['pool1'] = self.pool1(out['relu1_2'])
        out['relu2_1'] = F.relu(self.conv2_1(out['pool1']))
        out['relu2_2'] = F.relu(self.conv2_2(out['relu2_1']))
        out['pool2'] = self.pool2(out['relu2_2'])
        out['relu3_1'] = F.relu(self.conv3_1(out['pool2']))
        out['relu3_2'] = F.relu(self.conv3_2(out['relu3_1']))
        out['relu3_3'] = F.relu(self.conv3_3(out['relu3_2']))
        out['relu3_4'] = F.relu(self.conv3_4(out['relu3_3']))
        out['pool3'] = self.pool3(out['relu3_4'])
        out['relu4_1'] = F.relu(self.conv4_1(out['pool3']))
        out['relu4_2'] = F.relu(self.conv4_2(out['relu4_1']))
        out['relu4_3'] = F.relu(self.conv4_3(out['relu4_2']))
        out['relu4_4'] = F.relu(self.conv4_4(out['relu4_3']))
        out['pool4'] = self.pool4(out['relu4_4'])
        out['relu5_1'] = F.relu(self.conv5_1(out['pool4']))
        out['relu5_2'] = F.relu(self.conv5_2(out['relu5_1']))
        out['relu5_3'] = F.relu(self.conv5_3(out['relu5_2']))
        out['relu5_4'] = F.relu(self.conv5_4(out['relu5_3']))
        out['pool5'] = self.pool5(out['relu5_4'])
        return [out[key] for key in layers]

class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return out

def model_def():
    vgg = VGG16()
    vgg.load_state_dict(torch.load(Path('tgbot', 'models', 'data', 'vgg_conv.pth')))
    for param in vgg.parameters():
        param.requires_grad = False

    if torch.cuda.is_available():
        vgg.cuda()

    vgg.eval()
    return vgg


async def generate_styled_image(style_image_path = Path('tgbot', 'models', 'data', 'style1.png'),
                          content_image_path = Path( 'tgbot', 'models', 'data', 'content1.png')):
    # Загрузка изображений
    SIZE_IMAGE = 512  # Замените на ваш размер изображения

    to_mean_tensor = transforms.Compose([
        transforms.Resize(SIZE_IMAGE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                             std=[1, 1, 1]),
        transforms.Lambda(lambda x: x.mul_(255)),
    ])

    to_unmean_tensor = transforms.Compose([
        transforms.Lambda(lambda x: x.div_(255)),
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
                             std=[1, 1, 1]),
        transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),
    ])
    to_image = transforms.Compose([transforms.ToPILImage()])
    normalize_image = lambda t: to_image(torch.clamp(to_unmean_tensor(t), min=0, max=1))


    style_img = Image.open(style_image_path)
    content_img = Image.open(content_image_path)

    # Перевод изображений в тензоры
    imgs = [style_img, content_img]
    imgs_torch = [to_mean_tensor(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
    style_image, content_image = imgs_torch
    opt_img = Variable(content_image.data.clone(), requires_grad=True)

    # Создание модели переноса стиля
    style_transfer_model = model_def()

    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    content_layers = ['relu4_2']
    loss_layers = style_layers + content_layers
    losses = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    if torch.cuda.is_available():
        losses = [loss.cuda() for loss in losses]
    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    content_weights = [1e0]
    weights = style_weights + content_weights
    style_targets = [GramMatrix()(A).detach() for A in style_transfer_model(style_image, style_layers)]
    content_targets = [A.detach() for A in style_transfer_model(content_image, content_layers)]
    targets = style_targets + content_targets

    # Запуск оптимизации
    epochs = 10
    # async def step_opt():
    #     opt.zero_grad()
    #     out_layers = style_transfer_model(opt_img, loss_layers)
    #     layer_losses = []
    #     for j, out in enumerate(out_layers):
    #         layer_losses.append(weights[j] * losses[j](out, targets[j]))
    #     loss = sum(layer_losses)
    #     loss.backward()
    #     return loss
    #
    # for i in range(0, epochs + 1):
    #     loss = await opt.step(step_opt)

    generated_image = await asyncio.to_thread(generate_styled_image_in_thread, opt_img, style_transfer_model,
                                              loss_layers, weights, targets, epochs, losses)

    # Получение сгенерированного изображения
    generated_image = generated_image.squeeze()
    generated_image = to_unmean_tensor(generated_image.cpu())
    generated_image = to_image(generated_image)

    generated_image.save(Path('tgbot', 'models','data', 'generated_image.png'))


    return generated_image

def generate_styled_image_in_thread(opt_img, style_transfer_model, loss_layers, weights, targets, epochs, losses):
    # Здесь происходит блокирующая операция, которую мы переместили в отдельный поток
    opt = optim.LBFGS([opt_img])

    def step_opt():
        opt.zero_grad()
        out_layers = style_transfer_model(opt_img, loss_layers)
        layer_losses = []
        for j, out in enumerate(out_layers):
            layer_losses.append(weights[j] * losses[j](out, targets[j]))
        loss = sum(layer_losses)
        loss.backward()
        return loss

    for i in range(0, epochs + 1):
        loss = opt.step(step_opt)

    return opt_img.data.clone()
# # Пример использования функции

style_image_path = Path( 'data', 'style1.png')
content_image_path = Path( 'data', 'content1.png')


result_image = generate_styled_image(style_image_path, content_image_path)
