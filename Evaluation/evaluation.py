# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from PIL import Image
# import pandas as pd
# from scipy import stats
# import torch
# from semantic_image_cipher import SemanticImageCipher

# class ImageCipherEvaluator:
#     def __init__(self, original_dir, compressed_dir, output_dir="eval_results"):
#         """
#         Initialize the evaluator with directories containing original and compressed images.
        
#         Args:
#             original_dir (str): Path to directory containing original images
#             compressed_dir (str): Path to directory containing compressed/reconstructed images
#             output_dir (str): Directory to save evaluation graphs
#         """
#         self.original_dir = original_dir
#         self.compressed_dir = compressed_dir
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)
        
#         self.cipher = SemanticImageCipher(device="cuda" if torch.cuda.is_available() else "cpu")
#         self.image_pairs = self._get_image_pairs()
#         self.results = []
        
#     def _get_image_pairs(self):
#         """Match original and compressed image filenames"""
#         original_files = sorted([f for f in os.listdir(self.original_dir) 
#                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
#         compressed_files = sorted([f for f in os.listdir(self.compressed_dir) 
#                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
#         if len(original_files) != len(compressed_files):
#             print(f"Warning: Different number of files in directories ({len(original_files)} vs {len(compressed_files)})")
        
#         return [
#             (os.path.join(self.original_dir, orig), os.path.join(self.compressed_dir, comp))
#             for orig, comp in zip(original_files, compressed_files)
#     ]
    
#     def evaluate_all(self):
#         """Evaluate all image pairs and store results"""
#         for orig_path, comp_path in self.image_pairs:
#             result = self.evaluate_single_pair(orig_path, comp_path)
#             self.results.append(result)
        
#         # Save all graphs
#         self.plot_compression_vs_quality()
#         self.plot_rpp_space()
#         self.plot_individual_metrics()
        
#         return pd.DataFrame(self.results)
    
#     def evaluate_single_pair(self, original_path, compressed_path):
#         """Evaluate a single image pair"""
#         original_img = Image.open(original_path).convert('RGB')
#         compressed_img = Image.open(compressed_path).convert('RGB')
        
#         original_size = os.path.getsize(original_path)
#         compressed_size = os.path.getsize(compressed_path)
#         compression_ratio = original_size / compressed_size
#         bpp = (compressed_size * 8) / (original_img.size[0] * original_img.size[1])
        
#         identity_score = self.cipher.measure_identity_preservation(original_img, compressed_img)
#         semantic_score = self.cipher.measure_semantic_similarity(original_img, compressed_img)
        
#         return {
#             'original_path': original_path,
#             'compressed_path': compressed_path,
#             'original_size': original_size,
#             'compressed_size': compressed_size,
#             'compression_ratio': compression_ratio,
#             'bpp': bpp,
#             'identity_score': identity_score,
#             'semantic_score': semantic_score
#         }
    
#     def plot_compression_vs_quality(self):
#         """Plot compression vs quality metrics"""
#         df = pd.DataFrame(self.results)
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
#         # Compression vs Identity Preservation
#         ax1.scatter(df['compression_ratio'], df['identity_score'], alpha=0.7)
#         self._add_trendline(ax1, df['compression_ratio'], df['identity_score'])
#         ax1.set_xlabel('Compression Ratio')
#         ax1.set_ylabel('Identity Preservation Score')
#         ax1.set_title('Compression vs Identity Preservation')
#         ax1.grid(True)
        
#         # Compression vs Semantic Similarity
#         ax2.scatter(df['compression_ratio'], df['semantic_score'], alpha=0.7)
#         self._add_trendline(ax2, df['compression_ratio'], df['semantic_score'])
#         ax2.set_xlabel('Compression Ratio')
#         ax2.set_ylabel('Semantic Similarity Score')
#         ax2.set_title('Compression vs Semantic Similarity')
#         ax2.grid(True)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir, 'compression_vs_quality.png'))
#         plt.close()
    
#     def plot_rpp_space(self):
#         """Create 3D RPP plot"""
#         df = pd.DataFrame(self.results)
        
#         # Create theoretical reference curve
#         theor_bpp = np.linspace(0.1, 2, 50)
#         theor_ssim = 1 - np.exp(-theor_bpp*2)
#         theor_cr = 24 / theor_bpp
        
#         fig = plt.figure(figsize=(12, 8))
#         ax = fig.add_subplot(111, projection='3d')
        
#         # Plot theoretical relationship
#         ax.plot(theor_bpp, theor_ssim, 1/theor_cr, 'b-', alpha=0.3, label='Theoretical Frontier')
        
#         # Plot our results
#         for _, row in df.iterrows():
#             ax.scatter(
#                 row['bpp'], 
#                 row['semantic_score'], 
#                 1/row['compression_ratio'], 
#                 c='r', 
#                 s=50,
#                 alpha=0.7
#             )
        
#         # Plot average point
#         ax.scatter(
#             [df['bpp'].mean()], 
#             [df['semantic_score'].mean()], 
#             [1/df['compression_ratio'].mean()], 
#             c='g', 
#             s=200, 
#             marker='*', 
#             label='Average'
#         )
        
#         ax.set_xlabel('Bits/Pixel (Rate)')
#         ax.set_ylabel('Semantic Score (Perception)')
#         ax.set_zlabel('1/Compression Ratio (Parsimony)')
#         ax.set_title('Rate-Perception-Parsimony Space')
#         ax.legend()
#         ax.view_init(elev=20, azim=45)
        
#         plt.savefig(os.path.join(self.output_dir, 'rpp_space.png'))
#         plt.close()
    
#     def plot_individual_metrics(self):
#         """Plot histograms of individual metrics"""
#         df = pd.DataFrame(self.results)
        
#         fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
#         # Compression Ratio
#         axes[0,0].hist(df['compression_ratio'], bins=20, color='skyblue', edgecolor='black')
#         axes[0,0].set_title('Compression Ratio Distribution')
#         axes[0,0].set_xlabel('Compression Ratio')
#         axes[0,0].set_ylabel('Count')
        
#         # Bits per pixel
#         axes[0,1].hist(df['bpp'], bins=20, color='salmon', edgecolor='black')
#         axes[0,1].set_title('Bits Per Pixel Distribution')
#         axes[0,1].set_xlabel('Bits Per Pixel')
#         axes[0,1].set_ylabel('Count')
        
#         # Identity Score
#         axes[1,0].hist(df['identity_score'], bins=20, color='lightgreen', edgecolor='black')
#         axes[1,0].set_title('Identity Preservation Score Distribution')
#         axes[1,0].set_xlabel('Identity Score')
#         axes[1,0].set_ylabel('Count')
        
#         # Semantic Score
#         axes[1,1].hist(df['semantic_score'], bins=20, color='gold', edgecolor='black')
#         axes[1,1].set_title('Semantic Similarity Score Distribution')
#         axes[1,1].set_xlabel('Semantic Score')
#         axes[1,1].set_ylabel('Count')
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.output_dir, 'metric_distributions.png'))
#         plt.close()
    
#     def _add_trendline(self, ax, x, y):
#         """Add trendline to a plot"""
#         slope, intercept, r_value, _, _ = stats.linregress(x, y)
#         line = slope * x + intercept
#         ax.plot(x, line, 'r-', label=f'R²={r_value**2:.2f}')
#         ax.legend()

# def main():
#     # Configure these paths
#     original_dir = "original_image"
#     compressed_dir = "compressed_image"
#     output_dir = "evaluation_graphs"
    
#     evaluator = ImageCipherEvaluator(original_dir, compressed_dir, output_dir)
#     results_df = evaluator.evaluate_all()
    
#     # Save results to CSV
#     results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
#     print(f"Evaluation complete. Graphs saved to {output_dir}")

# if __name__ == "__main__":
#     main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load and prepare data
df = pd.read_csv('evaluation_results.csv')
df['privacy_score'] = 1 - df['identity_score']
df['size_mb'] = df['original_size_bytes'] / (1024 * 1024)  # Convert to MB

# ✅ Force all image sizes to be at least 2 MB
df['size_mb'] = df['size_mb'].apply(lambda x: max(x, 2.0))

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot with compression ratio as color
sc = ax.scatter(
    df['privacy_score'],      # X-axis: Privacy score (1 - identity_score)
    df['semantic_score'],     # Y-axis: Semantic similarity score
    df['size_mb'],            # Z-axis: Adjusted size (MB)
    c=df['compression_ratio'],
    cmap='RdYlGn',
    s=60,
    alpha=0.9,
    edgecolor='w',
    linewidth=0.5,
    depthshade=True
)

# Add colorbar
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('Compression Ratio', rotation=270, labelpad=15)

# Label axes
ax.set_xlabel('\nPrivacy Score\n(1 - identity_score)', linespacing=3.2, fontsize=11, labelpad=12)
ax.set_ylabel('\nSemantic\nSimilarity Score', linespacing=3.2, fontsize=11, labelpad=12)
ax.set_zlabel('\nImage Size (MB, min 2MB)', linespacing=3.2, fontsize=11, labelpad=12)

# Set title and viewing angle
ax.set_title('Privacy-Semantics trade-off', fontsize=20)
ax.view_init(elev=20, azim=135)

# Grid and tick styling
ax.xaxis._axinfo["grid"].update(linewidth=0.5, linestyle='--', alpha=0.4)
ax.yaxis._axinfo["grid"].update(linewidth=0.5, linestyle='--', alpha=0.4)
ax.zaxis._axinfo["grid"].update(linewidth=0.5, linestyle='--', alpha=0.4)
ax.tick_params(axis='both', which='major', pad=8)

plt.tight_layout()
plt.savefig('privacy_semantics_size_adjusted_3d.png', bbox_inches='tight', dpi=150, transparent=False)
plt.close()

print("Generated 3D plot with image sizes forced to minimum 2MB: 'privacy_semantics_size_adjusted_3d.png'")

