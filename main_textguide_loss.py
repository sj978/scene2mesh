import time

import clip
from tqdm import tqdm
import kaolin.ops.mesh
import kaolin as kal
import torch
from neural_style_field import NeuralStyleField
from render import Renderer
from mesh import Mesh
from Normalization import MeshNormalizer
import numpy as np
import random
import copy
import torchvision
import os
from PIL import Image
import argparse
from pathlib import Path
from torchvision import transforms
import re
from matplotlib import pyplot as plt

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("cuda is available")
else:
    device = torch.device("cpu")


# Set util functions
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


def export_final_results(obj_path, save_render, frontview_center, dir, losses, mesh, mlp, network_input, vertices):
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
        if save_render:
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


def update_mesh(mlp, network_input, prior_color, sampled_mesh, vertices):
    pred_rgb, pred_normal = mlp(network_input)
    sampled_mesh.face_attributes = prior_color + kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces)
    sampled_mesh.vertices = vertices + sampled_mesh.vertex_normals * pred_normal
    MeshNormalizer(sampled_mesh)()


# Constrain all sources of randomness
seed = 1234567
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Set transformations
crop_ratio = 0.1
clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                       (0.26862954, 0.26130258, 0.27577711))
# CLIP transform (preprocess: + CenterCrop(224, 224) + ToTensor())
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    clip_normalizer
])
# Augmentation transform (Global to Full)
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(1, 1)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    clip_normalizer
])
# Augmentations for normal network (Local to Full) - Set various crop methods
normaugment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(crop_ratio, crop_ratio)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    clip_normalizer
])
# Displacement-only augmentations (Local to Displ)
displacement_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(crop_ratio, crop_ratio)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    clip_normalizer
])

# Set data directory
obj_list = ['shoe', 'horse', 'lamp']
obj_parent_dir = "data/source_meshes"
scene_parent_dir = "data/target_scenes"
output_parent_dir = "results/text_guide_loss"
total_file_num = len(obj_list) * len(os.listdir(scene_parent_dir))

# Run main
start = time.time()
check_num = 0
for obj in obj_list:
    obj_path = os.path.join(obj_parent_dir, obj + '.obj')
    obj_text = obj
    for scene in os.listdir(scene_parent_dir):
        check_num += 1
        # Set file path
        scene_path = os.path.join(scene_parent_dir, scene)
        scene_prompt, _ = os.path.splitext(os.path.basename(scene_path))
        scene_list = re.findall('[a-zA-Z]+', scene_prompt)
        scene_text = ' '.join(scene_list)
        print("{}/{}th\nobj: {}, scene: {}".format(check_num, total_file_num, obj_text, scene_text))

        # Set guidance text option
        '''
        A: scene text
        B: rendered text
        C: scene+rendered text
        Base: (rendered, scene) loss
        
        guide_option = 0: Base + (rendered, A) loss
        guide_option = 1: Base + (rendered, B) loss
        guide_option = 2: Base + (rendered, C) loss
        '''
        for guide_option in range(3):
            print("guidance text option :", guide_option)
            output_dir = os.path.join(output_parent_dir, str(guide_option) + '_' + obj + '_' + scene_prompt)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Set MLP
            n_iter = 2000
            sigma = 5.0
            depth = 4
            width = 256
            colordepth = 2
            normdepth = 2
            normratio = 0.1
            clamp = "tanh"
            normclamp = "tanh"
            exclude = 0
            input_dim = 3

            mlp = NeuralStyleField(sigma, depth, width, 'gaussian', colordepth, normdepth,
                                   normratio, clamp, normclamp, niter=n_iter,
                                   progressive_encoding=True, input_dim=input_dim, exclude=exclude).to(device)
            mlp.reset_weights()

            # Set optimizer
            learning_rate = 0.0005
            decay = 0.0
            lr_decay = 0.9
            decay_step = 100
            lr_plateau = False
            optim = torch.optim.Adam(mlp.parameters(), learning_rate, weight_decay=decay)
            activate_scheduler = lr_decay < 1 and decay_step > 0 and not lr_plateau
            if activate_scheduler:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=decay_step, gamma=lr_decay)

            # Set CLIP
            text_guide = None
            clip_m = 'ViT-B/32'
            jit = False
            clip_model, preprocess = clip.load(clip_m, device, jit=jit)
            img = Image.open(scene_path)
            img = preprocess(img).to(device)
            clip_scene = clip_model.encode_image(img.unsqueeze(0))

            if guide_option == 0:
                text_guide = "an image of a {}".format(scene_text)
            elif guide_option == 1:
                text_guide = "an image of a {}".format(obj_text)
            elif guide_option == 2:
                text_guide = "an image of a {} in {}".format(obj_text, scene_text)
            print("text guidance: '{}'".format(text_guide))
            guide_token = clip.tokenize([text_guide]).to(device)
            clip_text = clip_model.encode_text(guide_token)
            with open(os.path.join(output_dir, text_guide), "w") as f:
                f.write("")

            # Set main instances
            symmetry = False
            standardize = False

            render = Renderer()
            mesh = Mesh(obj_path)
            MeshNormalizer(mesh)()

            prior_color = torch.full(size=(mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device)

            vertices = copy.deepcopy(mesh.vertices)
            network_input = copy.deepcopy(vertices)

            if symmetry:
                network_input[:, 2] = torch.abs(network_input[:, 2])

            if standardize:
                # Each channel into z-score
                network_input = (network_input - torch.mean(network_input, dim=0)) / torch.std(network_input, dim=0)

            # Set render parameters
            n_views = 5
            frontview_center = [0.5, 0.6283]
            frontview_std = 4
            background = None  # white color: torch.tensor([1, 1, 1]).to(device)

            # Set render parameters
            n_augs = 1
            n_normaugs = 4
            splitnormloss = False
            splitcolorloss = False
            geoloss = True
            cropdecay = 1.0
            loss_check = None
            losses = []
            guide_weight = 0.8

            # Calculate loss and optimization
            for i in tqdm(range(n_iter)):
                optim.zero_grad()
                sampled_mesh = mesh
                update_mesh(mlp, network_input, prior_color, sampled_mesh, vertices)
                rendered_images, elev, azim = render.render_front_views(sampled_mesh, num_views=n_views,
                                                                        show=False,
                                                                        center_azim=frontview_center[0],
                                                                        center_elev=frontview_center[1],
                                                                        std=frontview_std,
                                                                        return_views=True,
                                                                        background=background)

                loss = 0.0
                for _ in range(n_augs):
                    augmented_image = augment_transform(rendered_images)
                    clip_render = clip_model.encode_image(augmented_image)

                    loss -= (1.0 - guide_weight) * torch.cosine_similarity(torch.mean(clip_render, dim=0, keepdim=True), clip_scene)
                    # loss -= torch.mean(torch.cosine_similarity(clip_render, clip_scene))
                    loss -= guide_weight * torch.cosine_similarity(torch.mean(clip_render, dim=0, keepdim=True), clip_text)
                    # loss -= torch.mean(torch.cosine_similarity(clip_render, clip_text))

                    loss.backward(retain_graph=True)

                if splitnormloss:
                    for param in mlp.mlp_normal.parameters():
                        param.requires_grad = False

                # optim.step()

                normloss = 0.0
                for _ in range(n_normaugs):
                    augmented_image = normaugment_transform(rendered_images)
                    clip_render = clip_model.encode_image(augmented_image)

                    normloss -= (1.0 - guide_weight) * torch.cosine_similarity(torch.mean(clip_render, dim=0, keepdim=True), clip_scene)
                    # normloss -= torch.mean(torch.cosine_similarity(clip_render, clip_scene))
                    normloss -= guide_weight * torch.cosine_similarity(torch.mean(clip_render, dim=0, keepdim=True), clip_text)
                    # normloss -= torch.mean(torch.cosine_similarity(clip_render, clip_text))

                if splitnormloss:
                    for param in mlp.mlp_normal.parameters():
                        param.requires_grad = True
                if splitcolorloss:
                    for param in mlp.mlp_rgb.parameters():
                        param.requires_grad = False
                normloss.backward(retain_graph=True)

                # Also run separate loss on the uncolored displacements
                if geoloss:
                    default_color = torch.zeros(len(mesh.vertices), 3).to(device)
                    default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)
                    sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                                           sampled_mesh.faces)
                    geo_renders, elev, azim = render.render_front_views(sampled_mesh, num_views=n_views,
                                                                        show=False,
                                                                        center_azim=frontview_center[0],
                                                                        center_elev=frontview_center[1],
                                                                        std=frontview_std,
                                                                        return_views=True,
                                                                        background=background)
                    if n_normaugs > 0:
                        normloss = 0.0
                        ### avgview != aug
                        for _ in range(n_normaugs):
                            augmented_image = displacement_transform(geo_renders)
                            clip_render = clip_model.encode_image(augmented_image)

                            normloss -= (1.0 - guide_weight) * torch.cosine_similarity(torch.mean(clip_render, dim=0, keepdim=True), clip_scene)
                            # normloss -= torch.mean(torch.cosine_similarity(clip_render, clip_scene))
                            normloss -= guide_weight * torch.cosine_similarity(torch.mean(clip_render, dim=0, keepdim=True), clip_text)
                            # normloss -= torch.mean(torch.cosine_similarity(clip_render, clip_text))

                        normloss.backward(retain_graph=True)
                optim.step()

                for param in mlp.mlp_normal.parameters():
                    param.requires_grad = True
                for param in mlp.mlp_rgb.parameters():
                    param.requires_grad = True  # ?

                if activate_scheduler:
                    lr_scheduler.step()

                with torch.no_grad():
                    losses.append(loss.item())

                if i % 100 == 0:
                    report_process(lr_plateau, output_dir, i, loss, loss_check, losses, rendered_images)

            save_render = True
            export_final_results(obj_path, save_render, frontview_center, output_dir, losses, mesh, mlp, network_input,
                                 vertices)

end = time.time()
print("total run time : ", end - start)
