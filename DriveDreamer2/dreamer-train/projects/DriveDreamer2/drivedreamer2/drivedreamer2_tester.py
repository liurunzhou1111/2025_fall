import copy
import os
import math
import shutil
import time
import imageio
import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import DDIMScheduler, UniPCMultistepScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,AutoencoderKLTemporalDecoder

from dreamer_datasets import DefaultCollator, load_dataset, DefaultSampler,CLIPTextTransform
from dreamer_models import DriveDreamer2Pipeline
from dreamer_train import Tester
from . import drivedreamer2_transforms
from .drivedreamer2_utils import GLIGEN_WEIGHT_NAME, VideoSampler, VideoCollator, draw_mv_video,draw_mv_video_v2
from PIL import Image
from dreamer_datasets import ImageVisualizer, image_utils

CAM_NAMES = ['CAM_FRONT',
'CAM_FRONT_LEFT',
'CAM_FRONT_RIGHT',
'CAM_BACK',
'CAM_BACK_LEFT',
'CAM_BACK_RIGHT']
from torchvision import transforms

def extract_view_components(videos, batch_dict, cam_num=6, frame_num=8):
    """
    Extract ground truth, annotation, and generated images for each view separately.
    
    Args:
        videos: List of PIL Images (generated images), each image has width = single_view_width * cam_num
               OR a list containing a list of PIL Images (to handle nested lists)
        batch_dict: Dictionary containing 'image', 'image_hdmap', 'image_box'
        cam_num: Number of camera views (default: 6)
        frame_num: Number of frames (default: 8)
    
    Returns:
        dict: {
            'gt': [view_0_gt, view_1_gt, ..., view_5_gt],  # List of lists of PIL Images
            'ann': [view_0_ann, view_1_ann, ..., view_5_ann],  # List of lists of PIL Images
            'gen': [view_0_gen, view_1_gen, ..., view_5_gen]  # List of lists of PIL Images
        }
    """
    # Handle nested list case: if first element is a list, unwrap it
    if len(videos) > 0 and isinstance(videos[0], list):
        gen_images = videos[0]
    else:
        gen_images = videos
    
    if len(gen_images) == 0:
        return {'gt': [[] for _ in range(cam_num)], 
                'ann': [[] for _ in range(cam_num)], 
                'gen': [[] for _ in range(cam_num)]}
    
    # Check if gen_images[0] is a PIL Image
    if not hasattr(gen_images[0], 'size'):
        raise ValueError(f"Expected PIL Image, but got {type(gen_images[0])}. gen_images should be a list of PIL Images.")
    
    width, height = gen_images[0].size
    single_width = width // cam_num
    img_size = (single_width, height)
    
    # Extract components for each view (similar to draw_mv_video_v2)
    gt_videos = [[] for _ in range(cam_num)]
    ann_videos = [[] for _ in range(cam_num)]
    gen_videos = [[] for _ in range(cam_num)]
    
    for i in range(frame_num):
        for cam_idx in range(cam_num):
            # Ground truth image
            if i + frame_num * cam_idx < len(batch_dict['image']):
                gt_img = batch_dict['image'][i + frame_num * cam_idx]
                if isinstance(gt_img, Image.Image):
                    gt_img = gt_img.resize(img_size)
                gt_videos[cam_idx].append(gt_img)
            
            # Annotation image (HD map + boxes)
            if 'image_hdmap' in batch_dict and i + frame_num * cam_idx < len(batch_dict['image_hdmap']):
                # Create a base image for visualization
                if gt_videos[cam_idx] and len(gt_videos[cam_idx]) > 0:
                    base_img = np.array(gt_videos[cam_idx][-1])
                else:
                    base_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
                image = ImageVisualizer(base_img)
                
                image_hdmap = batch_dict['image_hdmap'][i + frame_num * cam_idx]
                if isinstance(image_hdmap, Image.Image):
                    image_hdmap = image_hdmap.resize(img_size)
                image_hdmap = np.array(image_hdmap)[:, :, ::-1]
                image.draw_seg(image_hdmap, scale=1.0)
                
                # Add boxes if available
                canvas_box = batch_dict.get('image_box', None)
                if canvas_box is not None and i + frame_num * cam_idx < len(canvas_box):
                    box = canvas_box[i + frame_num * cam_idx]
                    if hasattr(box, 'device') and box.device != 'cpu':
                        box = box.cpu()
                    if hasattr(box, 'sum'):
                        box = box.sum(axis=0)
                        image_box = Image.fromarray(box.numpy()).resize(img_size)
                        image_box = np.array(image_box)
                        image.add_boxes(image_box)
                
                ann_videos[cam_idx].append(image.get_image())
            
            # Generated image
            if i < len(gen_images):
                left = cam_idx * single_width
                right = (cam_idx + 1) * single_width
                gen_frame = gen_images[i].crop((left, 0, right, height))
                gen_videos[cam_idx].append(gen_frame)
    
    return {'gt': gt_videos, 'ann': ann_videos, 'gen': gen_videos}

def split_multiview_video(videos, cam_num=6):
    """
    Split a multiview video (with all views concatenated horizontally) into separate videos for each view.
    
    Args:
        videos: List of PIL Images, each image has width = single_view_width * cam_num
        cam_num: Number of camera views (default: 6)
    
    Returns:
        List of lists: [view_0_videos, view_1_videos, ..., view_5_videos]
    """
    if len(videos) == 0:
        return [[] for _ in range(cam_num)]
    
    # Get dimensions
    total_width, height = videos[0].size
    single_width = total_width // cam_num
    
    # Split each frame
    split_videos = [[] for _ in range(cam_num)]
    for frame in videos:
        for cam_idx in range(cam_num):
            left = cam_idx * single_width
            right = (cam_idx + 1) * single_width
            view_frame = frame.crop((left, 0, right, height))
            split_videos[cam_idx].append(view_frame)
    
    return split_videos

class DriveDreamer2_Tester(Tester):
    def get_dataloaders(self, data_config):
        self.data_config = data_config
        dataset = load_dataset(data_config.data_or_config)
        transform = getattr(drivedreamer2_transforms, data_config.transform.pop('type'))(**data_config.transform)
        dataset.set_transform(transform)

        self.fps=data_config.fps
        self.cam_num = data_config.cam_num
        self.frame_num = data_config.frame_num
        batch_size_per_gpu = self.frame_num * self.cam_num
        cam_names = data_config.get('cam_names',None)
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=VideoSampler(
                    dataset, 
                    batch_size=batch_size_per_gpu, 
                    frame_num=self.frame_num, 
                    cam_num=self.cam_num,
                    video_split_rate=data_config.get('video_split_rate',1),
                    hz_factor=data_config.hz_factor,
                    mv_video=data_config.is_multiview, 
                    view=data_config.view,
                    shuffle=data_config.shuffle,
                    logger=self.logger),
            collate_fn=VideoCollator(
                frame_num=self.frame_num,
                img_mask_type=data_config.img_mask_type,
                img_mask_num=data_config.img_mask_num) 
                if 'Video' in data_config.type else DefaultCollator(),
                batch_size=batch_size_per_gpu,
                num_workers=data_config.num_workers,)
       
       
        return dataloader
    
    def get_models(self, model_config):
        local_files_only = model_config.get('local_files_only', True)
        pipeline_name = model_config.pipeline_name
        text_encoder_pretrained = model_config.get('text_encoder_pretrained',None)
        variant = 'fp16' if self.mixed_precision == 'fp16' else None
        if pipeline_name == 'DriveDreamer2Pipeline':
            model=DriveDreamer2Pipeline.from_pretrained(
                model_config.pretrained,
                torch_dtype=self.dtype,
                variant=variant,
                local_files_only=local_files_only,
                safety_checker=None,
            )
            if text_encoder_pretrained is None:
                assert False
            self.text_encoder = CLIPTextTransform(
                model_path=text_encoder_pretrained,
                device=self.device,
                dtype=self.dtype,
                local_files_only=local_files_only,
            )
            setattr(model, 'frame_num',8)
            # setattr(model, 'cam_num', self.cam_num)
            # model.load_clipTextTransformer()
        else:
            assert False
        
        self.mode = model_config.get('mode','img_cond')
        
        assert self.mode in ['img_cond','video_cond','wo_img']

        self.num_inf_steps = model_config.get('num_inf_steps', 50)
        
        weight_path = model_config.get('weight_path', None)
        if weight_path is None:
            checkpoint = self.get_checkpoint()
            weight_path = os.path.join(checkpoint, GLIGEN_WEIGHT_NAME)
        elif os.path.isdir(weight_path):
            weight_path = os.path.join(weight_path, GLIGEN_WEIGHT_NAME)
        
        assert weight_path is not None
        self.logger.info('load from {}'.format(weight_path))
        model.load_weights(weight_path)
        model.to(self.device)
        return model

    def test(self):
        if self.is_main_process:
            save_dir = self.kwargs.get('save_dir', None)
            os.makedirs(save_dir,exist_ok=True)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
            idx = 0
            prompts = [
                ['realistic autonomous driving scene, panoramic videos from different perspectives.' ],
                ['rainy, realistic autonomous driving scene, panoramic videos from different perspectives.'],
                ['night, realistic autonomous driving scene, panoramic videos from different perspectives.'],
            ]
            for batch_dict in self.dataloader:
                grounding_downsampler_input = batch_dict.get('grounding_downsampler_input', None)
                grounding_downsampler_input = grounding_downsampler_input.reshape(self.cam_num,self.frame_num,*grounding_downsampler_input.shape[1:]).permute(1,2,3,0,4).flatten(3,4)
                box_downsampler_input = batch_dict.get('box_downsampler_input',None)
                box_downsampler_input = box_downsampler_input.reshape(self.cam_num,self.frame_num,*box_downsampler_input.shape[1:]).permute(1,2,3,0,4).flatten(3,4)
                img_cond = batch_dict.get('input_image',None)
                img_cond = img_cond.reshape(self.cam_num,self.frame_num,*img_cond.shape[1:]).permute(1,2,3,0,4).flatten(3,4)
                input_dict = {
                    'grounding_downsampler_input': grounding_downsampler_input,
                    'box_downsampler_input': box_downsampler_input}
                
                if self.mode == 'img_cond':
                    input_dict.update({
                        'img_cond':img_cond[:1],
                    })
                elif self.mode =='video_cond':
                    input_dict.update({
                        'video_cond':img_cond,
                    })
                
                prompt_embed = batch_dict.get('prompt_embeds',None)
                if prompt_embed is None:
                    videos = []
                    for this_prompt in prompts:
                        this_prompt_embed = self.text_encoder(this_prompt,mode='after_pool',to_numpy=False)[:,None]
                        images = self.model(
                            this_prompt_embed,
                            scheduled_sampling_beta=1.0,
                            input_dict=copy.deepcopy(input_dict),
                            height=batch_dict['height'][0],
                            width=batch_dict['width'][0]*6,
                            generator=generator,
                            min_guidance_scale=self.kwargs.get('min_guidance_scale', 1),
                            max_guidance_scale=self.kwargs.get('max_guidance_scale', 7.5),
                            num_inference_steps=self.num_inf_steps,
                            num_frames=self.frame_num,
                            first_frame=True,
                        )
                        
                        images=images.frames[0]
                        videos.append(images)

                    if save_dir is not None:
                        # Extract and save components for each video
                        for video_idx, video in enumerate(videos):
                            # Create subfolder for each video
                            video_subdir = os.path.join(save_dir, '{:06d}_{}'.format(idx, video_idx))
                            os.makedirs(video_subdir, exist_ok=True)
                            
                            # Extract components (gt, ann, gen) for each view
                            components = extract_view_components(video, batch_dict, cam_num=self.cam_num, frame_num=self.frame_num)
                            
                            for cam_idx in range(self.cam_num):
                                cam_name = CAM_NAMES[cam_idx] if cam_idx < len(CAM_NAMES) else f'CAM_{cam_idx}'
                                cam_subdir = os.path.join(video_subdir, cam_name)
                                os.makedirs(cam_subdir, exist_ok=True)
                                
                                # Save ground truth video
                                if components['gt'][cam_idx]:
                                    gt_path = os.path.join(cam_subdir, 'gt.mp4')
                                    imageio.mimsave(gt_path, components['gt'][cam_idx], fps=self.fps)
                                
                                # Save annotation video
                                if components['ann'][cam_idx]:
                                    ann_path = os.path.join(cam_subdir, 'ann.mp4')
                                    imageio.mimsave(ann_path, components['ann'][cam_idx], fps=self.fps)
                                
                                # Save generated video
                                if components['gen'][cam_idx]:
                                    gen_path = os.path.join(cam_subdir, 'gen.mp4')
                                    imageio.mimsave(gen_path, components['gen'][cam_idx], fps=self.fps)
                        idx += 1
                else:
                    images = self.model(
                                prompt_embed[0:1].half(),
                                scheduled_sampling_beta=1.0,
                                input_dict=copy.deepcopy(input_dict),
                                height=batch_dict['height'][0],
                                width=batch_dict['width'][0]*6,
                                generator=generator,
                                min_guidance_scale=self.kwargs.get('min_guidance_scale',1),
                                max_guidance_scale=self.kwargs.get('max_guidance_scale', 7.5),
                                num_inference_steps=self.num_inf_steps,
                                num_frames=self.frame_num,
                                first_frame=True,
                            )
                    images=images.frames[0]
                    if save_dir is not None:
                        # Create subfolder for this video
                        video_subdir = os.path.join(save_dir, '{:06d}'.format(idx))
                        os.makedirs(video_subdir, exist_ok=True)
                        
                        # Extract components (gt, ann, gen) for each view
                        components = extract_view_components(images, batch_dict, cam_num=self.cam_num, frame_num=self.frame_num)
                        
                        for cam_idx in range(self.cam_num):
                            cam_name = CAM_NAMES[cam_idx] if cam_idx < len(CAM_NAMES) else f'CAM_{cam_idx}'
                            cam_subdir = os.path.join(video_subdir, cam_name)
                            os.makedirs(cam_subdir, exist_ok=True)
                            
                            # Save ground truth video
                            if components['gt'][cam_idx]:
                                gt_path = os.path.join(cam_subdir, 'gt.mp4')
                                imageio.mimsave(gt_path, components['gt'][cam_idx], fps=self.fps)
                            
                            # Save annotation video
                            if components['ann'][cam_idx]:
                                ann_path = os.path.join(cam_subdir, 'ann.mp4')
                                imageio.mimsave(ann_path, components['ann'][cam_idx], fps=self.fps)
                            
                            # Save generated video
                            if components['gen'][cam_idx]:
                                gen_path = os.path.join(cam_subdir, 'gen.mp4')
                                imageio.mimsave(gen_path, components['gen'][cam_idx], fps=self.fps)
                    idx += 1     
        self.accelerator.wait_for_everyone()
