from semantic_image_cipher import SemanticImageCipher

# Initialize the cipher
cipher = SemanticImageCipher()

# Process an image
result = cipher.process_image(
    "Inputs\pexels-cottonbro-5739122.jpg"

)

print(f"Reconstructed image saved to: {result['reconstructed_image']}")
print(f"Identity preservation score: {result['identity_preservation']}")
print(f"Semantic similarity score: {result['semantic_similarity']}")