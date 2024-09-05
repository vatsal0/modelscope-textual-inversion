import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import torchvision
import numpy as np
import wandb
import random

with open('wandbkey') as f:
  key = f.read()

wandb.login(key=key)
run = wandb.init(
  project='ModelScope Textual Inversion', 
  job_type='training', 
  anonymous='allow'
)

LEARNING_RATE = 1e-4
LR_WARMUP_STEPS = 100
MAX_TRAIN_STEPS = 3000
BATCH_SIZE = 1 # not used
NUM_FRAMES = 16

LEARNING_RATE = LEARNING_RATE * BATCH_SIZE / NUM_FRAMES

INITIALIZER_TOKEN = 'man'
NUM_VECTORS = 5

print(f'{LEARNING_RATE=} {NUM_VECTORS=}')

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

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

video, audio, fps = torchvision.io.read_video('ironman.gif', pts_unit='sec')
transform = torchvision.transforms.Resize((256, 256))
video = transform(video.permute(0,3,1,2))
video = video.to(device).half()/127.5 - 1.0
batch = video

progress_bar = tqdm(
  range(MAX_TRAIN_STEPS),
  initial=0,
  desc='Steps'
)
# keep original embeddings for reference (only want placeholder token embeddings to change)
orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

losses = []

text_encoder.eval()
with torch.no_grad():
  video_frames = pipe(f'A {placeholder_token_string} playing basketball', num_inference_steps=25).frames
  output_video = np.stack([frame.transpose(2, 0, 1) for frame in video_frames])

wandb.log({'step': 0, 'sample_basketball': wandb.Video(output_video, fps=8)})
torchvision.io.write_video(f'./diffusion/samples/step{0:03d}.mp4', output_video.transpose(0,2,3,1), 8)
  

for i in range(1, MAX_TRAIN_STEPS+1):
  text_encoder.train()
  
  latents = vae.encode(batch).latent_dist.sample().detach() # 16, 4, 16, 16
  latents = latents * vae.config.scaling_factor

  latents = latents.unsqueeze(1)
  latents = latents.permute(0, 2, 1, 3, 4)
  new_bsz = latents.shape[0] // NUM_FRAMES
  latents = latents[ : new_bsz * NUM_FRAMES].view(new_bsz, NUM_FRAMES, 4, 1, 32, 32).permute(0, 2, 1, 3, 4, 5).view(new_bsz, 4, NUM_FRAMES, 32, 32)
  
  noise = torch.randn_like(latents).to(device)
  bsz = latents.shape[0]
  
  timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
  timesteps = timesteps.long()
  
  noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
  
  text = [random.choice(imagenet_templates_small).format(placeholder_token_string) for _ in range(bsz)] # .replace('photo', 'video')
  input_ids = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=tokenizer.model_max_length,
    return_tensors="pt"
  ).input_ids.to(device)
  encoder_hidden_states = text_encoder(input_ids)[0]
  encoder_hidden_states = encoder_hidden_states.to(device)
  
  model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, bypass_all_temporal=True).sample
  
  target = noise
  
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
      video_frames = pipe(f'A {placeholder_token_string} playing basketball', num_inference_steps=25, bypass_all_temporal=False).frames
      output_video = np.stack([frame.transpose(2, 0, 1) for frame in video_frames])

    wandb.log({'step': i, 'loss': sum(losses)/len(losses), 'sample_basketball': wandb.Video(output_video, fps=8)})
    torchvision.io.write_video(f'./diffusion/samples/step{i:03d}.mp4', output_video.transpose(0,2,3,1), 8)
    losses = []
  
  progress_bar.update(1)


# pipe.save_pretrained('/fs/nexus-scratch/vatsalb/modelscope_textual_inversion')

# module load Python3/3.11.2 ffmpeg/6.0
# srun --qos=default --gres=gpu:rtxa6000:1 --mem=8gb python3 diffusion/textual_inversion_spatial.py 
# diffusers 0.18.0