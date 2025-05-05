import os
import csv
from datetime import datetime
from semantic_image_cipher import SemanticImageCipher
import torch

def ensure_directory(path):
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)
    return path

def main():    
    input_dir = "inputs"
    output_dir = "eval_out"
    results_dir = "Evaluation"
    
    ensure_directory(input_dir)
    ensure_directory(output_dir)
    ensure_directory(results_dir)
    
    csv_path = os.path.join(results_dir, "evaluation_results.csv")
    
    # Initialize the cipher
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cipher = SemanticImageCipher(device=device)
    
    # Prepare CSV file
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = [
            'image_name',
            'original_path',
            'reconstructed_path',
            'identity_score',
            'semantic_score',
            'original_size_bytes',
            'latent_size_bytes',
            'compression_ratio'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each image in input directory
        for filename in sorted(os.listdir(input_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image_path = os.path.join(input_dir, filename)
                    print(f"\nProcessing {filename}...")
                    
                    # Process image with cipher
                    result = cipher.process_image(
                        image_path,
                        output_dir=output_dir
                    )
                    
                    # Write results to CSV
                    writer.writerow({
                        'image_name': filename,
                        'original_path': image_path,
                        'reconstructed_path': result['reconstructed_image'],
                        'identity_score': result['identity_preservation'],
                        'semantic_score': result['semantic_similarity'],
                        'original_size_bytes': result['original_size_bytes'],
                        'latent_size_bytes': result['latent_size_bytes'],
                        'compression_ratio': result['compression_ratio']
                    })
                    
                    # Print results to console
                    print(f"Reconstructed image saved to: {result['reconstructed_image']}")
                    # print(f"Identity preservation score: {result['identity_preservation']:.3f}")
                    # print(f"Semantic similarity score: {result['semantic_similarity']:.3f}")
                    # print(f"Compression ratio: {result['compression_ratio']:.2f}x")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    
    print(f"\nProcessing complete! Results saved to {csv_path}")

if __name__ == "__main__":
    main()