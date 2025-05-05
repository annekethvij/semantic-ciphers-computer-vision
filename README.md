# Semantic Image Cipher

A privacy-preserving image transformation applied research tool that protects personal identities while maintaining scene context.

## Overview

This project implements a "Semantic Image Cipher" that uses deep learning to transform images in a way that:
- Obscures identifiable facial features
- Preserves scene context, clothing, pose, and activity
- Creates a compact latent representation that can be stored and shared

Built using Stable Diffusion's latent space, this tool offers an alternative to traditional image anonymization methods like blurring or pixelation by producing more natural-looking results.

For the full detailed architecture diagram, see [semantic-ciphers-architecture](docs/improved_architecture_cv.mmd).

## Features

- **Privacy Protection**: Transforms facial features to prevent recognition
- **Context Preservation**: Maintains the setting, activity, and general content of the image
- **Adjustable Privacy Levels**: Control the balance between privacy and content preservation
- **Quantitative Evaluation**: Measures both identity protection and semantic preservation
- **Compact Representation**: Stores images in an efficient latent format

## Installation

```bash
# Clone the repository
git clone https://github.com/annekethvij/semantic-ciphers-computer-vision.git

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from semantic_image_cipher import SemanticImageCipher

# Initialize the cipher
cipher = SemanticImageCipher()

# Process an image
result = cipher.process_image(
    "path/to/image.jpg",
    noise_level=0.8,         # Higher = more privacy
    content_preservation=0.7, # Higher = better scene preservation
    scene_preservation=0.7    # Higher = better overall structure
)

print(f"Reconstructed image saved to: {result['reconstructed_image']}")
print(f"Identity preservation score: {result['identity_preservation']}")
print(f"Semantic similarity score: {result['semantic_similarity']}")
```

## Parameters

- **noise_level** (0-1): Controls how aggressively facial features are transformed
- **content_preservation** (0-1): Controls latent space modification in non-facial areas
- **scene_preservation** (0-1): Controls blending between direct decoding and diffusion

## Example Results

For the full detailed results, see [results](results/).

## Storing and Loading Ciphers

```python
# Save the latent representation
latent_path = cipher.save_latents(latents, "person_latents.pt")

# Load and decode later
loaded_latents = cipher.load_latents("person_latents.pt")
decoded_image = cipher.decode_image(loaded_latents)
decoded_image.save("decoded_person.png")
```

## Evaluation Metrics

The system measures two key aspects:

1. **Identity Preservation Score**: How well identity is preserved (lower is better for privacy)
2. **Semantic Similarity Score**: How well scene context is preserved (higher is better)

## Limitations

- Requires GPU for reasonable processing speed
- Works best with clear facial images
- May sometimes alter clothing details while preserving overall appearance
- Cannot guarantee mathematical privacy like formal encryption

## Methodology

This implementation is based on the following approach:

1. **Semantic Feature Extraction**: 
   - Uses Stable Diffusion's VAE encoder to convert images to latent space
   - Applies face detection to create masks for identity-specific regions

2. **Selective Transformation**:
   - Applies structured noise to facial regions
   - Preserves non-identity features with minimal modification

3. **Hybrid Reconstruction**:
   - Combines direct decoding with controlled diffusion process
   - Uses intelligent blending based on face detection

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Diffusers library
- FaceNet PyTorch
- CLIP

## Image Credits

This project uses free images from [Pexels](https://www.pexels.com/), a platform that provides free stock photos and videos for personal and commercial use.

Example images used in testing and demonstrations include:
- [Man in Black Polo Shirt Holding Tennis Racket](https://www.pexels.com/photo/man-in-black-polo-shirt-holding-tennis-racket-5739122/)

### Pexels License

All photos from Pexels are free to use under their license, which allows:
- Free use for commercial and non-commercial purposes
- No attribution required (though always appreciated)
- Modification of the photos as needed

The full license can be found on the [Pexels License page](https://www.pexels.com/license/).


## Citation and Publication Policy

If you use this project in your research, please cite it as:

```
Vij, A., Pandiyan, M.N., Nehrudass, P.  
Department of Computer Science  
University of Southern California  
Los Angeles, CA 90007  
[anneketh, maruthup, nehrudas]@usc.edu
```

# Publication Rights

1. Any publications, papers, presentations, or other academic/research works that use or are based on this codebase MUST:
   - Obtain explicit written approval from repo owners/collaboratos
   - Acknowledge the use of this codebase with proper citation

2. These requirements apply to:
   - Direct use of the code
   - Derivative works
   - Results obtained using this codebase
   - Methodologies or algorithms implemented in this codebase

3. Failure to comply with these terms constitutes both:
   - A violation of the software license
   - Academic misconduct/ethics violation
  
Copyright (c) 2025

All rights reserved.

No part of this software, including but not limited to the source code and documentation, may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the copyright holder.

For permission requests, please contact the repository owner.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---
