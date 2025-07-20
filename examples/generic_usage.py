#!/usr/bin/env python3
"""
Example usage of the GenericStreamDiffusion pipeline.

This example demonstrates how to use the generic pipeline to automatically
detect and load the appropriate StreamDiffusion implementation for any model.
"""

import torch
from streamdiffusion import GenericStreamDiffusion

def main():
    # Configuration
    model_id = "runwayml/stable-diffusion-v1-5"  # Can be SD or SDXL
    t_index_list = [32, 45]  # Timestep indices
    
    print(f"Loading model: {model_id}")
    print("The GenericStreamDiffusion will automatically detect if this is SD or SDXL...")
    
    # Create the pipeline - it will auto-detect SD vs SDXL
    stream = GenericStreamDiffusion(
        model_id_or_path=model_id,
        t_index_list=t_index_list,
        width=512,
        height=512,
        acceleration="none",  # Use "tensorrt" for faster inference
        mode="txt2img",
        use_tiny_vae=True,
        use_lcm_lora=True,
    )
    
    print(f"Pipeline created: {type(stream).__name__}")
    
    # Text-to-image generation
    prompt = "A beautiful landscape with mountains and a lake"
    print(f"Generating image with prompt: '{prompt}'")
    
    # Generate image
    image = stream(prompt=prompt)
    
    # Save the result
    if hasattr(image, 'save'):
        image.save("generated_image.png")
        print("Image saved as 'generated_image.png'")
    
    print("Generation complete!")


def example_with_sdxl():
    """Example specifically for SDXL models."""
    print("\n" + "="*50)
    print("SDXL Example")
    print("="*50)
    
    # SDXL model example
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    t_index_list = [32, 45]
    
    print(f"Loading SDXL model: {model_id}")
    
    stream = GenericStreamDiffusion(
        model_id_or_path=model_id,
        t_index_list=t_index_list,
        width=1024,  # SDXL typically uses 1024x1024
        height=1024,
        acceleration="none",
        mode="txt2img",
        use_tiny_vae=True,
        use_lcm_lora=True,
        lcm_lora_id="latent-consistency/lcm-lora-sdxl",  # SDXL-specific LCM LoRA
    )
    
    print(f"SDXL Pipeline created: {type(stream).__name__}")
    
    # The API is exactly the same!
    prompt = "A futuristic city with flying cars, highly detailed"
    image = stream(prompt=prompt)
    
    if hasattr(image, 'save'):
        image.save("sdxl_generated_image.png")
        print("SDXL image saved as 'sdxl_generated_image.png'")


def show_supported_pipelines():
    """Show all supported pipeline types."""
    print("\n" + "="*50)
    print("Supported Pipeline Types")
    print("="*50)
    
    pipelines = GenericStreamDiffusion.list_supported_pipelines()
    for i, pipeline in enumerate(pipelines, 1):
        print(f"{i}. {pipeline}")


if __name__ == "__main__":
    # Show supported pipelines
    show_supported_pipelines()
    
    # Note: Uncomment the examples below to run them
    # These require actual model downloads and GPU memory
    
    print("\n# Uncomment the lines below to run actual examples:")
    print("# main()  # SD 1.5 example")
    print("# example_with_sdxl()  # SDXL example")
    
    print("\nExamples are commented out to avoid downloading large models.")
    print("Uncomment them when you're ready to test with real models!")