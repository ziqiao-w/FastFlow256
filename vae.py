from diffusers import AutoencoderKL
from PIL import Image
import  torch
import torchvision.transforms as T


vae = AutoencoderKL.from_pretrained("/home/nus-wzq/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493").to(torch.float32)


def encode_img(input_img):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        # Transform the image to a tensor and normalize it
    transform = T.Compose([
        # T.Resize((256, 256)),
        T.ToTensor()
    ])
    input_img = transform(input_img)
    print(input_img.mean())
    print(input_img.std())
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img*2 - 1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()



def decode_img(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu()
    # image = T.Resize(original_size)(image.squeeze())ss
    return T.ToPILImage()(image.squeeze())

if __name__ == '__main__':
    # Load an example image
    input_img = Image.open("/home/nus-wzq/.cache/kagglehub/datasets/balraj98/edges2shoes-dataset/versions/1/val/1_AB.jpg")
    original_size = input_img.size
    w, h = input_img.size
    w2 = int(w / 2)
    A = input_img.crop((0, 0, w2, h))
    B = input_img.crop((w2, 0, w, h))
    A = T.Resize((64, 64))(A)
    B = T.Resize((64, 64))(B)
    B.save("B.jpg")
    original_size = A.size
    print('original_size',original_size)
    A = T.ToTensor()(A)
    B = T.ToTensor()(B)
    input_img = T.ToPILImage()(B)

    latents = encode_img(T.ToPILImage()(B-A))
    
    reconstructed_img = decode_img(latents)
    reconstructed_img = T.ToTensor()(reconstructed_img) + A
    reconstructed_img = T.ToPILImage()(reconstructed_img)
    # Save the reconstructed image
    reconstructed_img.save("reconstructed_example2.jpg")
    # Concatenate the original and reconstructed images
    concatenated_img = Image.new('RGB', (original_size[0] * 2, original_size[1]))
    # concatenated_img.paste(input_img, (0, 0))
    
    
    concatenated_img.paste(input_img, (0, 0))
    concatenated_img.paste(reconstructed_img, (original_size[0], 0))
    # Save the concatenated image
    concatenated_img.save("concatenated_example2.jpg")
