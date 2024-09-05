import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import torchvision

device = 'cuda'

pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', cache_dir='/fs/nexus-scratch/vatsalb/huggingface', torch_dtype=torch.float16, variant='fp16')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to(device)

with torch.no_grad():
  video_frames = pipe(f'A man playing basketball', num_inference_steps=25).frames

torchvision.io.write_video(f'./diffusion/samples/step{0:03d}.mp4', video_frames, 8)