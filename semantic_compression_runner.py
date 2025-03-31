import torch
from semantic_image_cipher import SemanticImageCipher
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def run_example():
    input_path = "/projects/pexels-cottonbro-5739122.jpg"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing Semantic Image Cipher...")
    cipher = SemanticImageCipher(device="cuda" if torch.cuda.is_available() else "cpu")
    
    privacy_levels = [0.5, 0.8, 1.0]
    results = []
    
    for level in privacy_levels:
        print(f"Processing with privacy level: {level}")
        result = cipher.process_image(
            input_path,
            output_dir=output_dir,
            noise_level=level,
            content_preservation=0.8,
            save_intermediates=False
        )
        results.append(result)
    
    original_img = Image.open(input_path)
    
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(5 * (len(results) + 1), 5))
    
    axes[0].imshow(np.array(original_img))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    for i, (result, level) in enumerate(zip(results, privacy_levels)):
        recon_img = Image.open(result['reconstructed_image'])
        axes[i + 1].imshow(np.array(recon_img))
        

        orig_size_kb = result['original_size_bytes'] / 1024
        latent_size_kb = result['latent_size_bytes'] / 1024
        
        if orig_size_kb > 1024:
            orig_size_str = f"{orig_size_kb/1024:.2f} MB"
        else:
            orig_size_str = f"{orig_size_kb:.2f} KB"
            
        if latent_size_kb > 1024:
            latent_size_str = f"{latent_size_kb/1024:.2f} MB"
        else:
            latent_size_str = f"{latent_size_kb:.2f} KB"
        

        compression_ratio = result['original_size_bytes'] / result['latent_size_bytes']
        
        axes[i + 1].set_title(f"Privacy Level: {level}\n"
                             f"ID Score: {result['identity_preservation']:.2f}\n"
                             f"Semantic Score: {result['semantic_similarity']:.2f}\n"
                             f"Orig: {orig_size_str} → Latent: {latent_size_str}\n"
                             f"Compression: {compression_ratio:.1f}x")
        axes[i + 1].axis('off')
    
    plt.tight_layout(pad=4.0)
    plt.savefig(os.path.join(output_dir, "privacy_comparison.png"))
    print(f"Comparison saved to {os.path.join(output_dir, 'privacy_comparison.png')}")
    
    print("Testing latent loading and decoding...")
    latent_path = results[1]['latent_representation']
    loaded_latents = cipher.load_latents(latent_path)
    decoded_image = cipher.decode_image(loaded_latents)
    decoded_path = os.path.join(output_dir, "decoded_from_latents.png")
    decoded_image.save(decoded_path)
    print(f"Decoded image saved to {decoded_path}")
    
    print("\nMetrics for different privacy levels:")
    for i, (result, level) in enumerate(zip(results, privacy_levels)):
        orig_size_kb = result['original_size_bytes'] / 1024
        latent_size_kb = result['latent_size_bytes'] / 1024
        
        if orig_size_kb > 1024:
            orig_size_str = f"{orig_size_kb/1024:.2f} MB"
        else:
            orig_size_str = f"{orig_size_kb:.2f} KB"
            
        if latent_size_kb > 1024:
            latent_size_str = f"{latent_size_kb/1024:.2f} MB"
        else:
            latent_size_str = f"{latent_size_kb:.2f} KB"
            
        compression_ratio = result['original_size_bytes'] / result['latent_size_bytes']
        
        print(f"\nPrivacy Level {level}:")
        print(f"  Identity Score: {result['identity_preservation']:.3f} (lower = better privacy)")
        print(f"  Semantic Score: {result['semantic_similarity']:.3f} (higher = better content)")
        print(f"  Original Size: {orig_size_str}, Latent Size: {latent_size_str}")
        print(f"  Compression Ratio: {compression_ratio:.2f}x")

if __name__ == "__main__":
    run_example()
