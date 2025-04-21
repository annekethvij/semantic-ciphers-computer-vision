from semantic_image_cipher import SemanticImageCipher
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def format_file_size(size_bytes):
    size_kb = size_bytes / 1024
    if size_kb > 1024:
        return f"{size_kb/1024:.2f} MB"
    else:
        return f"{size_kb:.2f} KB"

def create_comparison_plot(original_img, results, param_values, param_name, output_dir, filename):
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(5 * (len(results) + 1), 5))
    
    axes[0].imshow(np.array(original_img))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    for i, (result, param_val) in enumerate(zip(results, param_values)):
        recon_img = Image.open(result['reconstructed_image'])
        axes[i + 1].imshow(np.array(recon_img))
        
        orig_size_str = format_file_size(result['original_size_bytes'])
        latent_size_str = format_file_size(result['latent_size_bytes'])
        compression_ratio = result['original_size_bytes'] / result['latent_size_bytes']
        
        axes[i + 1].set_title(f"{param_name}: {param_val}\n"
                            f"ID Score: {result['identity_preservation']:.2f}\n"
                            f"Semantic Score: {result['semantic_similarity']:.2f}\n"
                            f"Orig: {orig_size_str} → Latent: {latent_size_str}\n"
                            f"Compression: {compression_ratio:.1f}x")
        axes[i + 1].axis('off')
    
    plt.tight_layout(pad=4.0)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    return output_path

def print_analysis_results(param_values, results, param_name):
    print(f"\n{param_name} Analysis:")
    for param_val, result in zip(param_values, results):
        orig_size_str = format_file_size(result['original_size_bytes'])
        latent_size_str = format_file_size(result['latent_size_bytes'])
        compression_ratio = result['original_size_bytes'] / result['latent_size_bytes']
        
        print(f"  {param_name} {param_val}:")
        print(f"    Identity Score: {result['identity_preservation']:.3f} (lower = better privacy)")
        print(f"    Semantic Score: {result['semantic_similarity']:.3f} (higher = better content)")
        print(f"    Original Size: {orig_size_str}, Latent Size: {latent_size_str}")
        print(f"    Compression Ratio: {compression_ratio:.2f}x")

def run_advanced_example():
    input_image = "/home/ubuntu/projects/pexels-cottonbro-5739122.jpg"
    output_dir = "results_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing Semantic Image Cipher...")
    cipher = SemanticImageCipher(device="cuda" if torch.cuda.is_available() else "cpu")
    
    scene_preservation_levels = [0.1, 0.5, 0.8]
    results = []
    noise_level = 0.8
    content_preservation = 0.7
    
    for scene_pres in scene_preservation_levels:
        print(f"Processing with scene preservation level: {scene_pres}")
        result = cipher.process_image(
            input_image,
            output_dir=output_dir,
            noise_level=noise_level,
            content_preservation=content_preservation,
            scene_preservation=scene_pres,
            save_intermediates=True
        )
        results.append(result)
    
    original_img = Image.open(input_image)
    
    comparison_path = create_comparison_plot(
        original_img,
        results,
        scene_preservation_levels,
        "Scene Preservation",
        output_dir,
        "scene_preservation_comparison.png"
    )
    print(f"Comparison saved to {comparison_path}")
    
    noise_levels = [0.3, 0.6, 0.9]
    scene_preservation = 0.7
    noise_results = []
    
    for noise in noise_levels:
        print(f"Processing with noise level: {noise}")
        result = cipher.process_image(
            input_image,
            output_dir=output_dir,
            noise_level=noise,
            content_preservation=content_preservation,
            scene_preservation=scene_preservation
        )
        noise_results.append(result)
    
    noise_comparison_path = create_comparison_plot(
        original_img,
        noise_results,
        noise_levels,
        "Noise Level",
        output_dir,
        "noise_level_comparison.png"
    )
    print(f"Noise comparison saved to {noise_comparison_path}")
    
    print("\nSummary of results:")
    print_analysis_results(scene_preservation_levels, results, "Scene Preservation")
    print_analysis_results(noise_levels, noise_results, "Noise Level")

if __name__ == "__main__":
    run_advanced_example()