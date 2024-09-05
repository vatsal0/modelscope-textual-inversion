timestamps, fps = torchvision.io.read_video_timestamps('/fs/vulcan-datasets/UCF101/videos/Lunges/v_Lunges_g01_c06.avi', pts_unit='sec')
video, audio, fps = torchvision.io.read_video('/fs/vulcan-datasets/UCF101/videos/Lunges/v_Lunges_g01_c06.avi')

for i in range(video.size(0)):
  torchvision.utils.save_image(transform(video[i].permute(2,0,1)[:,:,-240:]/255), f'frames/{i}.jpg')

input = torch.randn((1, 3, 128, 128)).half().to(device) # 1, 3, 128, 128
encoding = pipe.vae.encoder(input) # 1, 8, 16, 16
latent = torch.randn(1,4,16,16).half().to(device) * 10
decoding = pipe.vae.decoder(latent) # range -1,1
