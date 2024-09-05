import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import torchvision

LEARNING_RATE = 4e-7 # 1e-7 too low, 5e-7 too high past step 100 (after warmup)
LR_WARMUP_STEPS = 100
MAX_TRAIN_STEPS = 1000
BATCH_SIZE = 1

INITIALIZER_TOKEN = 'man'
NUM_VECTORS=1

device = 'cuda'
# weight_dtype=torch.float16

pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', cache_dir='/fs/nexus-scratch/vatsalb/huggingface', torch_dtype=torch.float16, variant='fp16')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

tokenizer = pipe.tokenizer
noise_scheduler = pipe.scheduler
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet

unet = unet.to(device)
vae = vae.to(device)
text_encoder = text_encoder.to(device)
video_frames = pipe('A man doing lunges', num_inference_steps=25).frames

from videoio import videosave, videoread
import numpy as np
videosave("out.mp4", np.array(video_frames), fps=8.0)

# import cv2
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'h264'), 8.0, (256, 256))
# for frame in video_frames:
#   out.write(frame)

# out.release()

# module load ffmpeg/6.0