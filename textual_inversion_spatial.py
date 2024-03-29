import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import torchvision
import numpy as np
from videoio import videosave
import wandb

with open('wandbkey') as f:
  key = f.read()

wandb.login(key=key)
run = wandb.init(
  project='ModelScope Textual Inversion', 
  job_type='training', 
  anonymous='allow'
)

LEARNING_RATE = 1e-7
LR_WARMUP_STEPS = 100
MAX_TRAIN_STEPS = 1000
BATCH_SIZE = 1

INITIALIZER_TOKEN = 'man'
NUM_VECTORS = 5

device = 'cuda'

pipe = DiffusionPipeline.from_pretrained('damo-vilab/text-to-video-ms-1.7b', cache_dir='/fs/nexus-scratch/vatsalb/huggingface', torch_dtype=torch.float16, variant='fp16')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

tokenizer = pipe.tokenizer
noise_scheduler = pipe.scheduler
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet

placeholder_token = '<S*>'
placeholder_tokens = []

# copy placeholder token N times (to learn N embeddings)
for i in range(NUM_VECTORS):
  placeholder_tokens.append(f'{placeholder_token}{i}')

# add N tokens into tokenizer
num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
if num_added_tokens != NUM_VECTORS:
  raise ValueError('Tokenizer already contains token')

# get initializer token id (to use the reference embedding as starting point)
token_ids = tokenizer.encode(INITIALIZER_TOKEN, add_special_tokens=False)
if len(token_ids) > 1:
  raise ValueError('Initializer token must be a single token')

initializer_token_id = token_ids[0]
placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

# resize text encoder embeddings to fit new tokens, and copy over initializer token embedding
text_encoder.resize_token_embeddings(len(tokenizer))
token_embeds = text_encoder.get_input_embeddings().weight.data
with torch.no_grad():
  for token_id in placeholder_token_ids:
    token_embeds[token_id] = token_embeds[initializer_token_id].clone()

# freeze all but text encoder embedding layer
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.text_model.encoder.requires_grad_(False)
text_encoder.text_model.final_layer_norm.requires_grad_(False)
text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

unet = unet.to(device)
vae = vae.to(device)
text_encoder = text_encoder.to(device)

# optimizer and scheduler
optimizer = torch.optim.AdamW(
  text_encoder.get_input_embeddings().parameters(),
  lr=LEARNING_RATE,
  betas=(0.9, 0.999),
  weight_decay=1e-2,
  eps=1e-6,
)

lr_scheduler = get_scheduler(
  'constant',
  optimizer=optimizer,
  num_warmup_steps=LR_WARMUP_STEPS,
  num_training_steps=MAX_TRAIN_STEPS,
  num_cycles=1,
)

placeholder_token_string = ' '.join(placeholder_tokens)
input_ids = tokenizer(
  f'A {placeholder_token_string} doing lunges',
  padding="max_length",
  truncation=True,
  max_length=tokenizer.model_max_length,
  return_tensors="pt"
).input_ids.to(device)

video, audio, fps = torchvision.io.read_video('/fs/vulcan-datasets/UCF101/videos/Lunges/v_Lunges_g01_c06.avi', pts_unit='sec')
transform = torchvision.transforms.Resize(256)

# order F, C, H, W
# 25 fps, pick 48 frames (~2s), every 3 to get 16 frames (8fps)
# crop to make it square, then reduce to 256x256 and normalize # 1, 16, 3, 128, 128
# batch = transform(video.permute(0, 3, 1, 2)[73:121:3, :, :, -240:]).to(device).half()
batch = transform(video.permute(0, 3, 1, 2)[73:121:3, :, :, -240:]).to(device).half()/127.5 - 1.0
# batch = transform(video.permute(0, 3, 1, 2)[73:121:3, :, :, -240:]).to(device).half()/255.0 
# batch = (batch - 0.5) / 0.5

progress_bar = tqdm(
  range(MAX_TRAIN_STEPS),
  initial=0,
  desc='Steps'
)
# keep original embeddings for reference (only want placeholder token embeddings to change)
orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

losses = []

for i in range(1, MAX_TRAIN_STEPS+1):
  text_encoder.train()
  
  latents = vae.encode(batch).latent_dist.sample().detach() # 16, 4, 16, 16
  latents = latents * vae.config.scaling_factor
  
  noise = torch.randn_like(latents).to(device)
  bsz = latents.shape[0]
  bsz=1
  
  timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
  timesteps = timesteps.long()
  
  noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
  
  encoder_hidden_states = text_encoder(input_ids)[0].to(device) # 1, 77, 1024
  
  model_pred = unet(noisy_latents.unsqueeze(0).permute(0, 2, 1, 3, 4), timesteps, encoder_hidden_states).sample
  
  target = noise.unsqueeze(0).permute(0, 2, 1, 3, 4)
  
  loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction='mean')
  losses.append(loss.detach().item())
  
  loss.backward()
  
  optimizer.step()
  lr_scheduler.step()
  optimizer.zero_grad()
  
  # dont update any embedding weights besides the placeholder token
  index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
  index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
  
  with torch.no_grad():
    text_encoder.get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]
  
  if i % 100 == 0:
    text_encoder.eval()
    with torch.no_grad():
      video_frames = pipe(f'A {placeholder_token_string} walking', num_inference_steps=25).frames
      videosave(f'./diffusion/samples/step{i:03d}.mp4', np.array(video_frames), fps=8.0)

    wandb.log({'step': i, 'loss': sum(losses)/len(losses), 'sample_walking': wandb.Video(f'./diffusion/samples/step{i:03d}.mp4')})
    losses = []
  
  progress_bar.update(1)


# pipe.save_pretrained('/fs/nexus-scratch/vatsalb/modelscope_textual_inversion')

# module load Python3/3.11.2 ffmpeg/6.0
# srun --qos=default --gres=gpu:rtxa6000:1 --mem=8gb python3 diffusion/textual_inversion_spatial.py 