import os
import torch
import datetime
import argparse
import copy
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)

from tensorboardX import SummaryWriter
import torchvision.transforms.functional as TF

from torchtyping import TensorType
import tinycudann as tcnn

from configs import get_args
from model import TCNNModel
from dataset import TextureDataset
from configs import Config


class Trainer:

    def __init__(self, params: argparse.Namespace) -> None:

        # init required variables, e.g. dataset, configs and models
        configs = Config(params)
        dataset = TextureDataset(configs)
        configs.num_channels = dataset.num_channels
        configs.num_lods = dataset.num_lods
        model = TCNNModel(configs)

        # init required 
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.max_iter = configs.max_iter
        self.trained_iter = 0
        self.eval_interval = configs.eval_interval
        self.save_interval = configs.save_interval
        self.quantize = configs.quantize
        self.quantize_bits = configs.quantize_bits
        self.save_bits = configs.save_bits

        # save and log
        self.save_dir = configs.save_dir
        self.start_time = datetime.datetime.now()
        self.end_time = 0
        self.duration_time = 0
        self.save_path = os.path.join(self.save_dir, self.start_time.strftime(r"%y_%m_%d_%H_%M_%S"))
        self.log_path = os.path.join(self.save_path, "tensorboard")
        self.model_path = os.path.join(self.save_path, "models")
        self.media_path = os.path.join(self.save_path, "media")
        self.writer = SummaryWriter(log_dir=self.log_path)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.media_path, exist_ok=True)  
        if configs.load_iter != 0:
            self.infer_path = os.path.join(configs.load_dir, "infer")
            os.makedirs(self.infer_path, exist_ok=True)  

        # data config
        self.num_lods = dataset.num_lods
        self.texture_height = dataset.texture_height
        self.texture_width = dataset.texture_width

        # loss weights
        self.output_loss_weights = configs.output_loss_weights

        self.sample_probabilities = self.generate_probabilities()      

        # losses
        self.L2_loss = torch.nn.MSELoss(reduction="none")
        
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(return_full_image=True).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

        self.configs = configs
        self.model = model
        self.dataset = dataset


    def train(self) -> None:

        for curr_iter in range(self.trained_iter, self.max_iter):

            self.model.optimizer.zero_grad()

            # generate random indexs
            ys = torch.randint(0, self.texture_height, [self.batch_size, 1]).to(self.device)
            xs = torch.randint(0, self.texture_width, [self.batch_size, 1]).to(self.device)
            # care for the sample probabilty
            # lods = torch.randint(0, self.num_lods, [self.batch_size, 1]).to(self.device)
            lods = self.sample_probabilities.multinomial(num_samples=self.batch_size, replacement=True)
            #lods = torch.zeros(lods.shape).to(self.device).to(torch.int32)
            lods = lods.unsqueeze(1)
            # lods = torch.randint(0, 1, size=(self.batch_size, 1)).to(self.device)
            batch_index = torch.cat([ys, xs, lods], dim=1)

            # get data
            gt_texture = self.dataset(batch_index)  # [batch_size, num_channels]
            gt_texture = gt_texture.to(torch.float16)

            # xys -> uvs
            # shift the sample position from [0, 1, ..., 1023] -> [0.5, 1.5, ..., 1023.5]
            # uvs = ((xys + 0.5) / lod_scale) / (texture_weight / lod_scale)
            us = (xs + 0.5) / self.texture_height 
            vs = (ys + 0.5) / self.texture_width
            lods = lods / (self.num_lods - 1)
            batch_input = torch.cat([us, vs, lods], dim=1)
            # predict
            predict_texture = self.model(batch_input)  # [batch_size, num_channels]

            # loss
            loss = self.L2_loss(gt_texture, predict_texture)
            # custom loss weights for different channels(albedo.rgb, normal.xyz, roughness, ao, displacement)
            loss_weights = torch.tensor(self.output_loss_weights).to(self.device)
            loss = loss.mean(dim=0) * loss_weights
            loss = loss.sum()

            self.writer.add_scalar('Loss/train', loss.item(), curr_iter)

            # optimize
            loss.backward()
            self.model.optimizer.step()
            self.model.scheduler.step(metrics=loss.item())

            # print(self.model.optimizer.param_groups[0]['lr'], self.model.optimizer.param_groups[1]['lr'])

            self.model.clamp_value()

            # eval
            if curr_iter % self.eval_interval == 0:
                self.eval(curr_iter)
                # print(self.model.optimizer.param_groups[0]['lr'], self.model.optimizer.param_groups[1]['lr'])
            
            if curr_iter % self.save_interval == 0:
                self.model.save(curr_iter, self.model_path)
                self.end_time = datetime.datetime.now()
                self.duration_time = self.end_time - self.start_time
                print(self.duration_time)
            
            if curr_iter % 10000 == 0:
                torch.cuda.empty_cache()
                tcnn.free_temporary_memory()

    @torch.no_grad()
    def eval(self, curr_iter) -> None:
        
        psnr_list = []
        ssim_list = []
        lpips_list = []

        os.makedirs(os.path.join(self.media_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.media_path, "normal"), exist_ok=True)
        os.makedirs(os.path.join(self.media_path, "roughness"), exist_ok=True)
        os.makedirs(os.path.join(self.media_path, "occlusion"), exist_ok=True)
        os.makedirs(os.path.join(self.media_path, "displacement"), exist_ok=True)

        quant_model = copy.deepcopy(self.model)
        quant_model.simulate_quantize()

        #for lod in range(self.num_lods - 4):
        for lod in [0]:

            lod_height = self.texture_height // (2 ** lod)
            lod_width = self.texture_width // (2 ** lod)
            x_coords, y_coords = torch.meshgrid(torch.arange(lod_height), torch.arange(lod_width), indexing='xy')
            u_coords = (x_coords + 0.5) / lod_height
            v_coords = (y_coords + 0.5) / lod_width 
            lod_coords = torch.ones_like(x_coords) * lod / (self.num_lods - 1)
            # eval_input = torch.stack([u_coords, v_coords, lod_coords, x_coords, y_coords], dim=2).to(self.device)
            # eval_input = eval_input.reshape([-1, 5])
            eval_input = torch.stack([u_coords, v_coords, lod_coords], dim=2).to(self.device)
            eval_input = eval_input.reshape([-1, 3])

            predicted_image = quant_model(eval_input)
            predicted_image = predicted_image.reshape([lod_height, lod_width, -1])
            predicted_image = torch.clamp(predicted_image, min=0, max=1)
            gt_image = self.dataset.lod_cache[lod, :lod_height, :lod_width, :]

            predicted_image = predicted_image.permute(2, 0, 1)[None, ...]  # [B, C, H, W]
            gt_image = gt_image.permute(2, 0, 1)[None, ...]

            predicted_rgb = predicted_image[:, 0:3, :, :]
            gt_rgb = gt_image[:, 0:3, :, :]
                        
            psnr_value = self.psnr(predicted_rgb, gt_rgb)
            psnr_list.append(psnr_value.item())
            self.writer.add_scalar(f'PSNR_LOD{int(lod)}/train', psnr_value.item(), curr_iter)

            ssim_value, ssim_images = self.ssim(predicted_rgb, gt_rgb)
            ssim_list.append(ssim_value.item())
            self.writer.add_scalar(f'SSIM_LOD{int(lod)}/train', ssim_value.item(), curr_iter)

            if lod_height >= 128 and lod_width >= 128:
                lpips_value = self.lpips(predicted_rgb, gt_rgb)
                lpips_list.append(lpips_value.item())
                self.writer.add_scalar(f'LPIPS_LOD{int(lod)}/train', lpips_value.item(), curr_iter)

            if curr_iter % (self.eval_interval * 2) == 0:
                save_image = torch.cat([predicted_image, gt_image], dim=3).squeeze()

                # Use dataset-provided channel mapping if available
                channel_slices = getattr(self.dataset, 'channel_slices', {})

                # Diffuse (RGB)
                if 'diffuse' in channel_slices:
                    s, e = channel_slices['diffuse']
                    rgb_image = save_image[s:e, ...]
                    # convert from linear to sRGB for visualization
                    rgb_image = torch.clamp(rgb_image, 0.0, 1.0)
                    rgb_image = torch.pow(rgb_image, 1.0 / 2.2)
                    rgb_path = os.path.join(self.media_path, "rgb", f"{curr_iter}_{int(lod)}.png")
                    TF.to_pil_image(rgb_image).save(rgb_path)

                # Normal (RGB)
                if 'normal' in channel_slices:
                    s, e = channel_slices['normal']
                    normal_image = save_image[s:e, ...]
                    # visualize normals by re-normalizing vectors: [0,1]→[-1,1]→unit→[0,1]
                    n = normal_image * 2.0 - 1.0
                    norm = torch.sqrt(torch.clamp((n ** 2).sum(dim=0, keepdim=True), min=1e-8))
                    n = n / norm
                    normal_vis = torch.clamp((n + 1.0) * 0.5, 0.0, 1.0)
                    normal_path = os.path.join(self.media_path, "normal", f"{curr_iter}_{int(lod)}.png")
                    TF.to_pil_image(normal_vis).save(normal_path)

                # Roughness (1ch)
                if 'roughness' in channel_slices:
                    s, e = channel_slices['roughness']
                    rough_image = save_image[s:e, ...]
                    rough_path = os.path.join(self.media_path, "roughness", f"{curr_iter}_{int(lod)}.png")
                    TF.to_pil_image(rough_image).save(rough_path)

                # Occlusion (1ch)
                if 'occlusion' in channel_slices:
                    s, e = channel_slices['occlusion']
                    ao_image = save_image[s:e, ...]
                    ao_path = os.path.join(self.media_path, "occlusion", f"{curr_iter}_{int(lod)}.png")
                    TF.to_pil_image(ao_image).save(ao_path)

                # Displacement (1ch)
                if 'displacement' in channel_slices:
                    s, e = channel_slices['displacement']
                    disp_image = save_image[s:e, ...]
                    disp_path = os.path.join(self.media_path, "displacement", f"{curr_iter}_{int(lod)}.png")
                    TF.to_pil_image(disp_image).save(disp_path)
        
        psnr_aver = torch.tensor(psnr_list).mean()
        ssim_aver = torch.tensor(ssim_list).mean()
        lpips_aver = torch.tensor(lpips_list).mean()
        self.writer.add_scalar('PSNR/train', psnr_aver.item(), curr_iter)
        self.writer.add_scalar('SSIM/train', ssim_aver.item(), curr_iter)
        self.writer.add_scalar('LPIPS/train', lpips_aver.item(), curr_iter)
    
        print(f"Iter:{curr_iter}, PSNR:{psnr_aver.item():.4f}, SSIM:{ssim_aver.item():.4f}, LPIPS:{lpips_aver.item():.4f}")

    @torch.no_grad()
    def infer(self) -> None:

        psnr_list = []
        ssim_list = []
        lpips_list = []

        os.makedirs(os.path.join(self.infer_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.infer_path, "normal"), exist_ok=True)
        os.makedirs(os.path.join(self.infer_path, "roughness"), exist_ok=True)
        os.makedirs(os.path.join(self.infer_path, "occlusion"), exist_ok=True)
        os.makedirs(os.path.join(self.infer_path, "displacement"), exist_ok=True)

        metrics = f"LOD PSNR SSIM LPIPS\n"

        for lod in range(self.num_lods - 4):

            lod_height = self.texture_height // (2 ** lod)
            lod_width = self.texture_width // (2 ** lod)
            x_coords, y_coords = torch.meshgrid(
                torch.arange(lod_height), torch.arange(lod_width), 
            indexing='xy')
            u_coords = (x_coords + 0.5) / lod_height
            v_coords = (y_coords + 0.5) / lod_width 
            lod_coords = torch.ones_like(x_coords) * lod / (self.num_lods - 1)
            eval_input = torch.stack([u_coords, v_coords, lod_coords, x_coords, y_coords], dim=2).to(self.device)
            eval_input = eval_input.reshape([-1, 5])

            predicted_image = self.model(eval_input)
            predicted_image = predicted_image.reshape([lod_height, lod_width, -1])
            predicted_image = torch.clamp(predicted_image, min=0, max=1)
            gt_image = self.dataset.lod_cache[lod, :lod_height, :lod_width, :]

            predicted_image = predicted_image.permute(2, 0, 1)[None, ...]  # [B, C, H, W]
            gt_image = gt_image.permute(2, 0, 1)[None, ...]

            predicted_rgb = predicted_image[:, 0:3, :, :]
            gt_rgb = gt_image[:, 0:3, :, :]
                        
            psnr_value = self.psnr(predicted_rgb, gt_rgb)
            psnr_list.append(psnr_value.item())

            ssim_value, ssim_images = self.ssim(predicted_rgb, gt_rgb)
            ssim_list.append(ssim_value.item())

            lpips_value = 0
            if lod_height >= 128 and lod_width >= 128:
                lpips_value = self.lpips(predicted_rgb, gt_rgb)
                lpips_list.append(lpips_value.item())

            save_image = torch.cat([predicted_image, gt_image], dim=3).squeeze()

            channel_slices = getattr(self.dataset, 'channel_slices', {})

            if 'diffuse' in channel_slices:
                s, e = channel_slices['diffuse']
                rgb_image = save_image[s:e, ...]
                # convert from linear to sRGB for visualization
                rgb_image = torch.clamp(rgb_image, 0.0, 1.0)
                rgb_image = torch.pow(rgb_image, 1.0 / 2.2)
                rgb_path = os.path.join(self.infer_path, "rgb", f"LOD_{int(lod)}.png")
                TF.to_pil_image(rgb_image).save(rgb_path)

            if 'normal' in channel_slices:
                s, e = channel_slices['normal']
                normal_image = save_image[s:e, ...]
                # visualize normals by re-normalizing vectors: [0,1]→[-1,1]→unit→[0,1]
                n = normal_image * 2.0 - 1.0
                norm = torch.sqrt(torch.clamp((n ** 2).sum(dim=0, keepdim=True), min=1e-8))
                n = n / norm
                normal_vis = torch.clamp((n + 1.0) * 0.5, 0.0, 1.0)
                normal_path = os.path.join(self.infer_path, "normal", f"LOD_{int(lod)}.png")
                TF.to_pil_image(normal_vis).save(normal_path)

            if 'roughness' in channel_slices:
                s, e = channel_slices['roughness']
                rough_image = save_image[s:e, ...]
                rough_path = os.path.join(self.infer_path, "roughness", f"LOD_{int(lod)}.png")
                TF.to_pil_image(rough_image).save(rough_path)

            if 'occlusion' in channel_slices:
                s, e = channel_slices['occlusion']
                ao_image = save_image[s:e, ...]
                ao_path = os.path.join(self.infer_path, "occlusion", f"LOD_{int(lod)}.png")
                TF.to_pil_image(ao_image).save(ao_path)

            if 'displacement' in channel_slices:
                s, e = channel_slices['displacement']
                disp_image = save_image[s:e, ...]
                disp_path = os.path.join(self.infer_path, "displacement", f"LOD_{int(lod)}.png")
                TF.to_pil_image(disp_image).save(disp_path)

            metrics += f"LOD_{int(lod)} {psnr_value:.4f} {ssim_value:.4f} {lpips_value:.4f}\n"
        
        psnr_aver = torch.tensor(psnr_list).mean()
        ssim_aver = torch.tensor(ssim_list).mean()
        lpips_aver = torch.tensor(lpips_list).mean()
        metrics += f"AVER {psnr_aver} {ssim_aver} {lpips_aver}\n"
        with open(os.path.join(self.infer_path, "metrics.txt"), "w+") as file:
            file.writelines(metrics)
    
    @torch.no_grad()
    def generate_probabilities(self) -> TensorType["num_lods"]:
        
        probabilities = []

        # here we need to generate a sample posibility
        current_prob = 1.0
        for i in range(0, self.num_lods):
            prob = current_prob / 4.0  # original is current_prob / 2**2
            probabilities.append(max(prob, 0.05))  # min probability is greater than 5%
            current_prob = prob
        probabilities = torch.tensor(probabilities).to(self.device)
        probabilities /= probabilities.sum()
        # print(probabilities)

        return probabilities


if __name__ == "__main__":

    params = get_args()

    trainer = Trainer(params)

    if params.mode == "train":
        trainer.train()
    elif params.mode == "infer":
        trainer.infer()
    else:
        raise ValueError("Error mode.")