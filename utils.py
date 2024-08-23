import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import cv2
from tqdm import tqdm

VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]


def generate_region_masks(np_array):
    max_region = np_array.max()
    region_masks = []

    for region in range(max_region + 1):
        mask = (np_array == region).astype(np.float32)
        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.unsqueeze(0).expand(3, -1, -1)
        mask_tensor = mask_tensor.unsqueeze(0)
        region_masks.append(mask_tensor)

    return region_masks


def load_rgb(img_path):
    im = cv2.imread(img_path)
    height, width, _ = im.shape

    if height > width:
        new_width = 224
        new_height = int(height * (224 / width))
    else:
        new_height = 224
        new_width = int(width * (224 / height))

    im = cv2.resize(im, (new_width, new_height))
    start_x = (new_width - 224) // 2
    start_y = (new_height - 224) // 2
    im = im[start_y:start_y + 224, start_x:start_x + 224]

    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im_rgb, im


def load_image(path):
    img = io.imread(path)
    img = img / 255.0
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = transform.resize(crop_img, (224, 224), mode='reflect')

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD)
    ])

    resized_img = preprocess(resized_img).unsqueeze(0)
    return resized_img


def gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    return torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def coalition_perturbation(img, region_masks, model, device, w_weight=0.0, conf_weight=100.0, continuity_weight=1.0, c_weight=1.0,
                       max_iter=500, batch_size=32, step_size=1e-2, max_w=0.1):
    img = img.float().to(device)
    img.requires_grad_(False)

    region_masks = [mask.float().to(device) for mask in region_masks]

    w = torch.zeros_like(img, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=step_size)

    with torch.no_grad():
        prob = model(img)
        cls = torch.argmax(prob).item()

    res = []
    for itr in tqdm(range(max_iter)):
        optimizer.zero_grad()
        noise = torch.randn((batch_size, 3, 224, 224)).to(device)
        noised_image = img + noise * w
        output = model(noised_image)

        loss1 = torch.mean(w)
        loss2 = torch.mean(torch.clamp(output - output[:, cls].unsqueeze(1), min=0.0))

        kernel_size = 5
        sigma = 1.0
        padding = kernel_size // 2
        kernel = gaussian_kernel(kernel_size, sigma).to(device)
        loss3 = 0.0
        loss4 = 0.0

        if continuity_weight > 0.0:
            for mask in region_masks:
                masked_map = w * mask
                masked_map = masked_map.sum(dim=1, keepdim=True)
                conv = F.conv2d(masked_map, kernel, padding=padding)
                conv = torch.abs(conv)
                consistency = torch.mean(conv)
                continuity_loss = -consistency
                loss3 += continuity_loss

        if c_weight > 0.0:
            laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                               [1, -4, 1],
                                               [0, 1, 0]]]], dtype=torch.float32).to(device)
            laplacian_kernel = laplacian_kernel.expand(3, 1, 3, 3)

            for mask in region_masks:
                grad_w = F.conv2d(w, laplacian_kernel, padding=1, groups=3)
                c_loss = torch.mean((grad_w * mask) ** 2)
                loss4 += c_loss

        loss = -w_weight * loss1 + conf_weight * loss2 + continuity_weight * loss3 + c_weight * loss4

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            w.clamp_(0.0, max_w)

        if itr % 10 == 0:
            res.append((loss.item(), loss1.item(), loss2.item(), w.detach().clone().cpu().numpy()))

    return cls, prob.cpu().numpy(), w.detach().cpu().numpy(), res