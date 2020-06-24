# -*- coding: utf-8 -*-
"""
| **@created on:** 6/23/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
|
|
| **Sphinx Documentation Status:**
"""

import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.style.use("presentation")

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

SAVE_DIR = 'results/'
os.system(f"rm -rf {SAVE_DIR}/*")


def get_image(img):
    if len(img.shape) == 4:
        img = np.transpose(img[0], (1, 2, 0))
        return np.uint8(255 * img)
    else:
        return np.uint8(255 * img)


def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)


def save(mask, img, blurred):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    cam = 1.0 * heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    cv2.imwrite(f"{SAVE_DIR}/res-perturbated.png", np.uint8(255 * perturbated))
    cv2.imwrite(f"{SAVE_DIR}/res-heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite(f"{SAVE_DIR}/res-mask.png", np.uint8(255 * mask))
    cv2.imwrite(f"{SAVE_DIR}/res-cam.png", np.uint8(255 * cam))


def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def load_model():
    model = models.vgg19(pretrained=True)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False

    return model


if __name__ == '__main__':
    # Hyper parameters.
    # TBD: Use argparse
    tv_beta = 3
    learning_rate = 0.1
    max_iterations = 500
    l1_coeff = 0.01
    tv_coeff = 0.2

    model = load_model()
    original_img = cv2.imread("examples/flute.jpg", 1)
    original_img = cv2.resize(original_img, (224, 224))
    img = np.float32(original_img) / 255
    blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
    cv2.imwrite("results/blur_1.png", get_image(blurred_img1))
    blurred_img2 = np.float32(cv2.medianBlur(original_img, 11)) / 255
    cv2.imwrite("results/blur_2.png", get_image(blurred_img2))
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
    cv2.imwrite("results/total_blur.png", get_image(blurred_img_numpy))
    mask_init = np.ones((28, 28), dtype=np.float32)
    cv2.imwrite("results/init_mask.png", get_image(mask_init))
    # Convert to torch variables
    img = preprocess_image(img)
    blurred_img = preprocess_image(blurred_img2)
    mask = numpy_to_torch(mask_init)

    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224))

    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    target = torch.nn.Softmax()(model(img))
    category = np.argmax(target.cpu().data.numpy())
    print("Category with highest probability", category)
    print("Optimizing.. ")
    c1l, c2l, c3l, lossl, categl = [], [], [], [], []
    for i in range(max_iterations):
        CUR_DIR = SAVE_DIR + f'/epoch_{i}/'
        os.system(f"mkdir -p {CUR_DIR}")
        upsampled_mask = upsample(mask)

        cv2.imwrite(f"{CUR_DIR}/1-upsampled_mask.png", get_image(upsampled_mask.detach().numpy()))
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = \
            upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
                                  upsampled_mask.size(3))

        cv2.imwrite(f"{CUR_DIR}/2-upsampled_mask_rgb.png", get_image(upsampled_mask.detach().numpy()))

        # Use the mask to perturbated the input image.
        neg_upsampled = (1 - upsampled_mask)
        cv2.imwrite(f"{CUR_DIR}/3-neg_upsampled.png", get_image(neg_upsampled.detach().numpy()))
        perturbation = blurred_img.mul(neg_upsampled)
        cv2.imwrite(f"{CUR_DIR}/3-perturbation.png", get_image(perturbation.detach().numpy()))
        perturbated_input = img.mul(upsampled_mask) + perturbation
        cv2.imwrite(f"{CUR_DIR}/4-perturbed_inp.png", get_image(perturbated_input.detach().numpy()))

        noise = np.zeros((224, 224, 3), dtype=np.float32)
        cv2.randn(noise, 0, 0.2)
        noise = numpy_to_torch(noise)
        cv2.imwrite(f"{CUR_DIR}/5-noise.png", get_image(noise.detach().numpy()))
        perturbated_input = perturbated_input + noise
        cv2.imwrite(f"{CUR_DIR}/6-perturbed_input+noise.png", get_image(perturbated_input.detach().numpy()))
        outputs = torch.nn.Softmax(dim=-1)(model(perturbated_input))
        c1 = l1_coeff * torch.mean(torch.abs(1 - mask))
        c2 = tv_coeff * tv_norm(mask, tv_beta)
        c3 = outputs[0, category]

        loss = c1 + c2 + c3
        c1l.append(c1)
        c2l.append(c2)
        c3l.append(c3)
        lossl.append(loss)
        categl.append(np.argmax(outputs.detach().numpy()))
        print(
            f"Iter: {i}/{max_iterations} | ({i / max_iterations * 100:.2f}%) | ~Mask: {c1l[-1]:.4f} "
            f"| TV Norm: {c2l[-1]:.4f} | Class Confidence: {c3l[-1]:.4f} | Total Loss: {lossl[-1]:.4f} | Categ: {categl[-1]}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)
        cv2.imwrite(f"{CUR_DIR}/7-clamped_mask.png", get_image(mask.detach().numpy()))
    upsampled_mask = upsample(mask)

    plt.subplot(2,1,1)
    plt.suptitle(f"Original Category: {category}")
    plt.plot(c1l, label="~Mask")
    plt.plot(c2l, label="TV Norm")
    plt.plot(c3l, label="Class Confidence")
    plt.plot(lossl, label="Total Loss")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(categl, label="Category")
    plt.ylabel("Epochs")
    plt.legend()

    plt.savefig(f"{SAVE_DIR}/loss.png")
    save(upsampled_mask, original_img, blurred_img_numpy)
