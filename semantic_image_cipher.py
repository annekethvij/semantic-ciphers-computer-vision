import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL,StableDiffusionImg2ImgPipeline
from transformers import CLIPVisionModel, CLIPProcessor
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class SemanticImageCipher:
    def __init__(self, sd_model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Using device: {self.device}")
        self.vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae").to(self.device)
        self.sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(sd_model_id,torch_dtype=torch.float16).to(self.device)
        self.sd_pipeline.enable_attention_slicing()
        self.face_detector = MTCNN(keep_all=True, device=self.device)
        self.face_recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def encode_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        faces, probs = self.face_detector.detect(image)
        face_masks = torch.zeros((1, 4, 64, 64), device=self.device)
        if faces is not None:
            for box in faces:
                x1, y1, x2, y2 = box.tolist()
                x1, y1, x2, y2 = int(x1/8), int(y1/8), int(x2/8), int(y2/8)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(63, x2), min(63, y2)
                face_masks[0, :, y1:y2, x1:x2] = 1.0
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample() * 0.18215
        return latents, face_masks, image

    def apply_semantic_cipher(self, latents, face_masks, noise_level=0.8, content_preservation=0.7):
        structured_noise = torch.randn_like(latents)
        smoothed_noise = torch.nn.functional.avg_pool2d(structured_noise, kernel_size=3, stride=1, padding=1)
        face_noise = smoothed_noise * face_masks * noise_level
        content_factor = 1.0 - content_preservation
        content_noise = smoothed_noise * (1 - face_masks) * content_factor * 0.3
        ciphered_latents = latents.clone()
        face_regions = latents * face_masks
        face_mean = torch.mean(face_regions[face_regions != 0]) if torch.sum(face_masks) > 0 else 0
        face_std = torch.std(face_regions[face_regions != 0]) if torch.sum(face_masks) > 0 else 1
        normalized_face_regions = (face_regions - face_mean) / (face_std + 1e-6)
        transformed_face_regions = normalized_face_regions + face_noise
        transformed_face_regions = transformed_face_regions * face_std + face_mean
        ciphered_latents = (latents * (1 - face_masks)) + transformed_face_regions + content_noise
        return ciphered_latents

    def decode_image(self, latents, guidance_scale=7.5, num_inference_steps=50, scene_preservation=0.7):
        with torch.no_grad():
            direct_decoded = self.vae.decode(latents / 0.18215).sample
            direct_decoded = (direct_decoded / 2 + 0.5).clamp(0, 1)
        direct_decoded_np = direct_decoded.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
        direct_decoded_pil = Image.fromarray((direct_decoded_np * 255).round().astype("uint8"))
        #self.sd_pipeline.scheduler.set_timesteps(num_inference_steps)
        #t_identity = int(self.sd_pipeline.scheduler.timesteps[0] * 0.8)
        face_prompt = "a photograph of a person with a changed face, same pose, same background, same clothes"
        diffusion_result = self.sd_pipeline(
            prompt=face_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            image=direct_decoded_pil,
            strength=1.0 - scene_preservation,
        ).images[0]
        return diffusion_result

    def save_latents(self, latents, output_path):
        torch.save(latents, output_path)
        return output_path

    def load_latents(self, input_path):
        return torch.load(input_path, map_location=self.device)

    def measure_identity_preservation(self, original_image, reconstructed_image):
        if isinstance(original_image, str):
            original_image = Image.open(original_image).convert("RGB")
        if isinstance(reconstructed_image, str):
            reconstructed_image = Image.open(reconstructed_image).convert("RGB")
        orig_faces, _ = self.face_detector.detect(original_image)
        recon_faces, _ = self.face_detector.detect(reconstructed_image)
        if orig_faces is None or recon_faces is None:
            return 0.0
        if len(orig_faces) != len(recon_faces):
            return 0.2
        orig_embeddings = []
        for box in orig_faces:
            face = original_image.crop((box[0], box[1], box[2], box[3]))
            face_tensor = self.transform(face).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.face_recognizer(face_tensor)
            orig_embeddings.append(embedding)
        recon_embeddings = []
        for box in recon_faces:
            face = reconstructed_image.crop((box[0], box[1], box[2], box[3]))
            face_tensor = self.transform(face).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.face_recognizer(face_tensor)
            recon_embeddings.append(embedding)
        similarities = []
        for i in range(len(orig_embeddings)):
            for j in range(len(recon_embeddings)):
                sim = F.cosine_similarity(orig_embeddings[i], recon_embeddings[j]).item()
                similarities.append(sim)
        return np.mean(similarities) if similarities else 0.0

    def measure_semantic_similarity(self, original_image, reconstructed_image):
        if isinstance(original_image, str):
            original_image = Image.open(original_image).convert("RGB")
        if isinstance(reconstructed_image, str):
            reconstructed_image = Image.open(reconstructed_image).convert("RGB")
        inputs1 = self.clip_processor(images=original_image, return_tensors="pt").to(self.device)
        inputs2 = self.clip_processor(images=reconstructed_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding1 = self.clip_model(**inputs1).pooler_output
            embedding2 = self.clip_model(**inputs2).pooler_output
        similarity = F.cosine_similarity(embedding1, embedding2).item()
        return similarity

    def process_image(self, image_path, output_dir="output", noise_level=0.8, content_preservation=0.7, scene_preservation=0.7, save_intermediates=False):
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        latents, face_masks, original_image = self.encode_image(image_path)
        if save_intermediates:
          plt.imshow(np.array(original_image))
          plt.title("Original Image")
          plt.axis('off')
          plt.savefig(os.path.join(output_dir, f"{base_name}_step1_original.png"))
          plt.close()
        
        ciphered_latents = self.apply_semantic_cipher(latents, face_masks, noise_level, content_preservation)
        
        if save_intermediates:
          # Optional: You can visualize the ciphered latent as an image if a method exists
          plt.imshow(latents[0].detach().cpu().numpy()[0], cmap='viridis')
          plt.title("Latents After Cipher (Visualization)")
          plt.axis('off')
          plt.savefig(os.path.join(output_dir, f"{base_name}_step2_latents_ciphered.png"))
          plt.close()

        latent_path = os.path.join(output_dir, f"{base_name}_latents.pt")
        self.save_latents(ciphered_latents, latent_path)
        if save_intermediates:
            face_mask_viz = face_masks.cpu().numpy()[0, 0] * 255
            face_mask_path = os.path.join(output_dir, f"{base_name}_face_mask.png")
            Image.fromarray(face_mask_viz.astype(np.uint8)).save(face_mask_path)
            plt.imshow(face_mask_viz, cmap='gray')
            plt.title("Face Mask")
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"{base_name}_step3_face_mask_viz.png"))
            plt.close()
        reconstructed_image = self.decode_image(ciphered_latents, scene_preservation=scene_preservation)
        reconstructed_path = os.path.join(output_dir, f"{base_name}_reconstructed.png")
        reconstructed_image.save(reconstructed_path)
        if save_intermediates:
            plt.imshow(np.array(reconstructed_image))
            plt.title("Reconstructed Image")
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"{base_name}_step4_reconstructed.png"))
            plt.close()
        identity_score = self.measure_identity_preservation(original_image, reconstructed_image)
        semantic_score = self.measure_semantic_similarity(original_image, reconstructed_image)
        original_size = os.path.getsize(image_path)
        latent_size = os.path.getsize(latent_path)
        compression_ratio = original_size / latent_size
        if save_intermediates:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(np.array(original_image))
            ax[0].set_title("Original Image")
            ax[0].axis('off')
            ax[1].imshow(np.array(reconstructed_image))
            ax[1].set_title("Reconstructed Image")
            ax[1].axis('off')
            fig.suptitle(f"Identity Score: {identity_score:.3f} (lower = better privacy)\nSemantic Score: {semantic_score:.3f} (higher = better content)")
            viz_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            plt.tight_layout()
            plt.savefig(viz_path)
            plt.close()
        results = {
            "original_image": image_path,
            "latent_representation": latent_path,
            "reconstructed_image": reconstructed_path,
            "identity_preservation": identity_score,
            "semantic_similarity": semantic_score,
            "original_size_bytes": original_size,
            "latent_size_bytes": latent_size,
            "compression_ratio": compression_ratio
        }
        return results

    def batch_process(self, image_dir, output_dir="output", **kwargs):
        results = []
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                try:
                    result = self.process_image(image_path, output_dir, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        return results
    
    

def main():
    cipher = SemanticImageCipher(device="cuda" if torch.cuda.is_available() else "cpu")
    result = cipher.process_image("Inputs\pexels-cottonbro-5739122.jpg", output_dir="output", noise_level=0.8, content_preservation=0.7, save_intermediates=True)
    print("Results:")
    print(f"Original image: {result['original_image']}")
    print(f"Latent representation: {result['latent_representation']}")
    print(f"Reconstructed image: {result['reconstructed_image']}")
    print(f"Identity preservation: {result['identity_preservation']:.3f} (lower = better privacy)")
    print(f"Semantic similarity: {result['semantic_similarity']:.3f} (higher = better content)")
    print(f"Compression ratio: {result['compression_ratio']:.2f}x")
    loaded_latents = cipher.load_latents(result['latent_representation'])
    decoded_image = cipher.decode_image(loaded_latents)
    decoded_image.save("output/decoded_from_saved.png")

if __name__ == "__main__":
    main()