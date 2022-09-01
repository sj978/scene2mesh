import copy
import os
import random
from collections import namedtuple

import numpy as np
from pathlib import Path

import torchvision
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from render import Renderer
from mesh import Mesh
from Normalization import MeshNormalizer
from neural_style_field import NeuralStyleField
import clip
import kaolin.ops.mesh
import kaolin as kal

from utils import device


# Hyper-parameters
seed = 12345678
background = None  # background = torch.tensor([1, 1, 1]).to(device)
n_aug = 1
n_displ_aug = 4
normalize_mean = (0.48145466, 0.4578275, 0.40821073)  # default is clip value
normalize_std = (0.26862954, 0.26130258, 0.27577711)  # default is clip value
mincrop_ratio = 0.1  # if object based method is applied, crop_ratio function is needed
maxcrop_ratio = 0.1  # if object based method is applied, crop_ratio function is needed
sigma = 5.0  # gaussian distribution's std in positional encoding  # 실험이 필요한 세팅
depth = 4  # depth of MLP
width = 256  # width of MLP
encoding = 'gaussian'
color_depth = 2  # depth of color MLP
displ_depth = 2  # depth of displacement MLP
displ_ratio = 0.1  # limitation of displacement value
color_activation = 'tanh'  # MLP activation function (or 'clamp')  # 실험이 필요한 세팅
displ_activation = 'tanh'  # MLP activation function (or 'clamp')  # 실험이 필요한 세팅
n_iter = 1500
learning_rate = 0.0005
lr_decay = 0.9
lr_decay_step = 100
lr_plateau = False  # 실험이 필요한 세팅
loss_check = None
clipavg = "view"  # 실험  # 실험이 필요한 세팅이 필요한 세팅
split_displ_loss = False  # 실험이 필요한 세팅
split_color_loss = False  # 실험이 필요한 세팅
geoloss = True
# renderer parameters
n_views = 5
frontview_center = [0., 0.]
frontview_std = 8.
# In demo, only people set these parameters in reverse.
progressive = True  # 실험이 필요한 세팅
symmetry = False  # 실험이 필요한 세팅
standardize = False  # 실험이 필요한 세팅
# content & style
obj = 'shoe'
is_prompt = True  # text = True / image = False
text = 'cactus'  # text
text_connect = 'made of'  # 실험이 필요한 세팅
displ_obj = 'shoe'  # text
displ_text = 'cactus'  # text
image_dir = 'data/style/in1.jpg'  # image


# Path setting
obj_path = 'data/source_meshes/{}.obj'.format(obj)
# text (is_prompt == True)
prompt = 'an image of a {} {} {}'.format(obj, text_connect, text)
displ_prompt = 'an image of a {} {} {}'.format(displ_obj, text_connect, displ_text)
# image (is_prompt == False)
image_prompt, _ = os.path.splitext(os.path.basename(image_dir))  # if image target is used with image prompt, fix if.

if is_prompt:
    if prompt == displ_prompt:
        output_dir = 'results/test/text/{}/{}'.format(obj, text)
    else:
        output_dir = 'results/test/text/{}-{}/{}-{}/'.format(obj, text, displ_obj, displ_text)
else:
    output_dir = 'results/test/image/{}'.format(image_prompt)

Path(output_dir).mkdir(parents=True, exist_ok=True)


# Constrain all sources of randomness
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Transform settings
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(normalize_mean, normalize_std)
])

global_to_full_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(1, 1)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    transforms.Normalize(normalize_mean, normalize_std)
])

local_to_full_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(maxcrop_ratio, maxcrop_ratio)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    transforms.Normalize(normalize_mean, normalize_std)
])

local_to_displ_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(mincrop_ratio, mincrop_ratio)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    transforms.Normalize(normalize_mean, normalize_std)
])


# MLP settings
mlp = NeuralStyleField(sigma, depth, width, encoding, color_depth, displ_depth,
                       displ_ratio, color_activation, displ_activation,
                       niter=n_iter, progressive_encoding=progressive, input_dim=3, exclude=0).to(device)
mlp.reset_weights()
optim = torch.optim.Adam(mlp.parameters(), learning_rate)
activate_schedular = lr_decay < 1 and lr_decay_step > 0 and not lr_plateau
if activate_schedular:
    lr_schedular = torch.optim.lr_scheduler.StepLR(optim, step_size=lr_decay_step, gamma=lr_decay)
losses = []


# CLIP settings

clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

if is_prompt:
    prompt_token = clip.tokenize([prompt]).to(device)
    encoded_text = clip_model.encode_text(prompt_token)
    displ_encoded = encoded_text
    print("target image's clip encode shape is", encoded_text.shape)
    if prompt != displ_prompt:
        dis_prompt_token = clip.tokenize([displ_prompt]).to(device)
        displ_encoded = clip_model.encode_text(dis_prompt_token)
else:
    img = Image.open(image_dir)
    img = preprocess(img).to(device)
    encoded_image = clip_model.encode_image(img.unsqueeze(0))
    displ_encoded = encoded_image
    print("target image's clip encode shape is", encoded_image.shape)


# Initialize renderer&mesh
render = Renderer()
mesh = Mesh(obj_path)
MeshNormalizer(mesh)()

prior_color = torch.full(size=(mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device)

vertices = copy.deepcopy(mesh.vertices)
network_input = copy.deepcopy(vertices)
print(network_input)

if symmetry:
    network_input[:, 2] = torch.abs(network_input[:, 2])  # vertex 의 z 값에 절대값 씌우기
if standardize:
    network_input = (network_input - torch.mean(network_input, dim=0))/torch.std(network_input, dim=0)


# Optimization
def run_train():
    for i in tqdm(range(n_iter)):
        loss = 0.0
        displ_loss = 0.0
        optim.zero_grad()
        sampled_mesh = mesh
        update_mesh(mlp, network_input, prior_color, sampled_mesh, vertices)
        rendered_images, elev, azim = render.render_front_views(sampled_mesh, num_views=n_views,
                                                                center_azim=frontview_center[0],
                                                                center_elev=frontview_center[1],
                                                                std=frontview_std,
                                                                return_views=True,
                                                                background=background
                                                                )

        # loss for global to full
        assert n_aug > 0, "augmentation number must be greater than 0"
        if n_aug > 0:
            for _ in range(n_aug):
                augmented_image = global_to_full_transform(rendered_images)
                encoded_renders = clip_model.encode_image(augmented_image)
                if is_prompt:
                    if clipavg == "view":
                        if encoded_text.shape[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(encoded_text, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            encoded_text)
                    else:
                        loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))

                else:
                    if clipavg == "view":
                        if encoded_image.shape[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(encoded_image, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            encoded_image)
                    else:
                        loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_image))
            if split_displ_loss:
                for param in mlp.mlp_normal.parameters():
                    param.requires_grad = False
            loss.backward(retain_graph=True)

        # loss for local to full
        assert n_displ_aug > 0, "displacement augmentation number must be greater than 0"
        if n_displ_aug > 0:
            for _ in range(n_displ_aug):
                augmented_image = local_to_full_transform(rendered_images)
                encoded_renders = clip_model.encode_image(augmented_image)
                if is_prompt:
                    if clipavg == "view":
                        if displ_encoded.shape[0] > 1:
                            displ_loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                  torch.mean(displ_encoded, dim=0), dim=0)
                        else:
                            displ_loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                  displ_encoded)
                    else:
                        displ_loss -= torch.mean(torch.cosine_similarity(encoded_renders, displ_encoded))

                else:
                    if clipavg == "view":
                        if encoded_image.shape[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(encoded_image, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            encoded_image)
                    else:
                        loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_image))
            if split_displ_loss:
                for param in mlp.mlp_normal.parameters():
                    param.requires_grad = True
            if split_color_loss:
                for param in mlp.mlp_rgb.parameters():
                    param.requires_grad = False
            if is_prompt:
                displ_loss.backward(retain_graph=True)

        # loss for local to displacement
        if geoloss:
            default_color = torch.zeros(len(mesh.vertices), 3).to(device)
            default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)
            sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                                   sampled_mesh.faces)
            geo_renders, elev, azim = render.render_front_views(sampled_mesh, num_views=n_views,
                                                                center_azim=frontview_center[0],
                                                                center_elev=frontview_center[1],
                                                                std=frontview_std,
                                                                return_views=True,
                                                                background=background
                                                                )

            if n_displ_aug > 0:
                displ_loss = 0.0
                for _ in range(n_displ_aug):
                    augmented_image = local_to_displ_transform(geo_renders)
                    encoded_renders = clip_model.encode_image(augmented_image)
                    if displ_encoded.shape[0] > 1:
                        displ_loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                              torch.mean(displ_encoded, dim=0), dim=0)
                    else:
                        displ_loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                              displ_encoded)

                    if not is_prompt:
                        if encoded_image[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(encoded_image, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            encoded_image)

                displ_loss.backward(retain_graph=True)
        optim.step()

        for param in mlp.mlp_normal.parameters():
            param.requires_grad = True
        for param in mlp.mlp_rgb.parameters():
            param.requires_grad = True

        if activate_schedular:
            lr_schedular.step()

        with torch.no_grad():
            losses.append(loss.item())

        if i % 100 == 0:
            report_process(lr_plateau, output_dir, i, loss, loss_check, losses, rendered_images)

    export_final_results(frontview_center, obj_path, output_dir, losses, mesh, mlp, network_input, vertices)


def update_mesh(mlp, network_input, prior_color, sampled_mesh, vertices):
    pred_rgb, pred_normal = mlp(network_input)
    sampled_mesh.face_attributes = prior_color + kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces)
    sampled_mesh.vertices = vertices + sampled_mesh.vertex_normals * pred_normal
    MeshNormalizer(sampled_mesh)()


def report_process(lr_plateau, dir, i, loss, loss_check, losses, rendered_images):
    print('iter: {} loss: {}'.format(i, loss.item()))
    torchvision.utils.save_image(rendered_images, os.path.join(dir, 'iter_{}.jpg'.format(i)))
    if lr_plateau and loss_check is not None:
        new_loss_check = np.mean(losses[-100:])
        # If avg loss increased or plateaued then reduce LR
        if new_loss_check >= loss_check:
            for g in torch.optim.param_groups:
                g['lr'] *= 0.5
        loss_check = new_loss_check

    elif lr_plateau and loss_check is None and len(losses) >= 100:
        loss_check = np.mean(losses[-100:])


def export_final_results(frontview_center, obj_path, dir, losses, mesh, mlp, network_input, vertices):
    with torch.no_grad():
        pred_rgb, pred_normal = mlp(network_input)
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        torch.save(pred_rgb, os.path.join(dir, f"colors_final.pt"))
        torch.save(pred_normal, os.path.join(dir, f"normals_final.pt"))

        base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
        final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        mesh.vertices = vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal

        objbase, extension = os.path.splitext(os.path.basename(obj_path))
        mesh.export(os.path.join(dir, f"{objbase}_final.obj"), color=final_color)

        # Run renders
        save_rendered_results(frontview_center, dir, final_color, mesh)

        # Save final losses
        torch.save(torch.tensor(losses), os.path.join(dir, "losses.pt"))


def save_rendered_results(frontview_center, dir, final_color, mesh):
    default_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5, device=device)
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                   mesh.faces.to(device))
    kal_render = Renderer(
        camera=kal.render.camera.generate_perspective_projection(np.pi / 4, 1280 / 720).to(device),
        dim=(1280, 720))
    MeshNormalizer(mesh)()
    img, mask = kal_render.render_single_view(mesh, frontview_center[1], frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"init_cluster.png"))
    MeshNormalizer(mesh)()
    # Vertex colorings
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(final_color.unsqueeze(0).to(device),
                                                                   mesh.faces.to(device))
    img, mask = kal_render.render_single_view(mesh, frontview_center[1], frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"final_cluster.png"))


run_train()
