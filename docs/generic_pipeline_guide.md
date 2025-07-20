# Generic StreamDiffusion Pipeline Guide

## Overview

The Generic StreamDiffusion Pipeline provides a unified interface for real-time diffusion model inference, automatically detecting and optimizing for different model architectures (SD, SDXL, etc.).

## Key Features

- **Automatic Model Detection**: Detects whether a model is SD 1.5, SDXL, or other architectures
- **Unified API**: Same interface for all supported models
- **Optimized Performance**: Architecture-specific optimizations under the hood
- **Easy Migration**: Drop-in replacement for existing StreamDiffusion code

## Quick Start

### Basic Usage

```python
from streamdiffusion import GenericStreamDiffusion

# Works with any supported model - auto-detects architecture
stream = GenericStreamDiffusion(
    model_id_or_path="runwayml/stable-diffusion-v1-5",  # or any SDXL model
    t_index_list=[32, 45],
    width=512,
    height=512,
    mode="txt2img",
    acceleration="tensorrt",  # or "none", "xformers"
)

# Generate image
image = stream(prompt="A beautiful landscape")
image.save("output.png")
```

### SDXL Example

```python
# SDXL automatically detected and optimized
stream = GenericStreamDiffusion(
    model_id_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    t_index_list=[32, 45],
    width=1024,  # SDXL typical resolution
    height=1024,
    mode="txt2img",
    lcm_lora_id="latent-consistency/lcm-lora-sdxl",  # SDXL-specific LoRA
)

image = stream(prompt="Futuristic cityscape, highly detailed")
```

### img2img Mode

```python
from PIL import Image

stream = GenericStreamDiffusion(
    model_id_or_path="your-model",
    t_index_list=[32, 45],
    mode="img2img",  # Switch to img2img mode
)

# Load input image
input_image = Image.open("input.jpg")

# Transform the image
output_image = stream(
    image=input_image,
    prompt="Transform this into a painting"
)
```

## Architecture Details

### Supported Models

| Pipeline Type | Example Models | Notes |
|---------------|----------------|-------|
| StableDiffusion | runwayml/stable-diffusion-v1-5 | SD 1.5 and similar |
| StableDiffusionXL | stabilityai/stable-diffusion-xl-base-1.0 | SDXL and variants |

### Automatic Detection

The pipeline automatically detects model architecture by:
1. Loading model configuration
2. Checking pipeline class type
3. Selecting appropriate StreamDiffusion wrapper
4. Applying architecture-specific optimizations

### Performance Optimizations

#### SDXL-Specific Features
- **Dual Text Encoders**: Proper handling of both CLIP text encoders
- **Pooled Embeddings**: Automatic conditioning with pooled text embeddings
- **Time IDs**: Proper micro-conditioning for resolution/cropping
- **TinyVAE**: SDXL-specific TinyVAE (`madebyollin/taesdxl`)

#### SD 1.5-Specific Features  
- **Single Text Encoder**: Optimized for single CLIP encoder
- **Standard VAE**: Regular TinyVAE (`madebyollin/taesd`)
- **Simplified Conditioning**: No additional time embeddings

## Configuration Options

### Essential Parameters

```python
GenericStreamDiffusion(
    model_id_or_path="path/to/model",     # Model path or HF repo
    t_index_list=[32, 45],                # Timestep indices for sampling
    width=512,                            # Output width
    height=512,                           # Output height
    mode="txt2img",                       # "txt2img" or "img2img"
    
    # Performance
    acceleration="tensorrt",              # "none", "xformers", "tensorrt"
    use_tiny_vae=True,                    # Faster VAE
    use_denoising_batch=True,             # Batch denoising steps
    
    # Quality/Speed Trade-off
    frame_buffer_size=1,                  # Larger = smoother but slower
    cfg_type="self",                      # "none", "self", "full", "initialize"
    
    # LoRA Support
    use_lcm_lora=True,                    # Use LCM LoRA for speed
    lcm_lora_id=None,                     # Custom LCM LoRA (auto-detected)
    lora_dict={"style_lora": 0.8},        # Additional LoRAs
)
```

### Advanced Configuration

```python
# Multi-GPU setup
stream = GenericStreamDiffusion(
    model_id_or_path="model",
    device_ids=[0, 1],                    # Use multiple GPUs
    t_index_list=[32, 45],
)

# Memory optimization
stream = GenericStreamDiffusion(
    model_id_or_path="model", 
    dtype=torch.float16,                  # Lower precision
    use_tiny_vae=True,                    # Smaller VAE
    enable_similar_image_filter=True,     # Skip similar frames
    t_index_list=[32, 45],
)

# Safety filtering
stream = GenericStreamDiffusion(
    model_id_or_path="model",
    use_safety_checker=True,              # Enable NSFW filtering
    t_index_list=[32, 45],
)
```

## Migration Guide

### From Individual Pipelines

**Old Code:**
```python
from streamdiffusion.pipelines.stream_sdxl_pipeline import StreamSDXLPipeline

# Had to manually choose the right pipeline
stream = StreamSDXLPipeline(...)
```

**New Code:**
```python
from streamdiffusion import GenericStreamDiffusion

# Automatically detects and uses the right pipeline
stream = GenericStreamDiffusion(...)
```

### From Base StreamDiffusion

**Old Code:**
```python
from streamdiffusion import StreamDiffusion
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("model")
stream = StreamDiffusion(pipe=pipe, ...)
```

**New Code:**
```python
from streamdiffusion import GenericStreamDiffusion

# Much simpler - handles pipeline loading internally
stream = GenericStreamDiffusion(model_id_or_path="model", ...)
```

## Performance Tips

### For Maximum Speed
```python
stream = GenericStreamDiffusion(
    model_id_or_path="model",
    t_index_list=[32],                    # Single timestep
    acceleration="tensorrt",              # TensorRT acceleration
    use_tiny_vae=True,                    # Fast VAE
    use_lcm_lora=True,                    # LCM for few steps
    cfg_type="none",                      # No CFG overhead
    dtype=torch.float16,                  # Half precision
    t_index_list=[32],
)
```

### For Maximum Quality
```python
stream = GenericStreamDiffusion(
    model_id_or_path="model",
    t_index_list=[20, 25, 30, 35, 40],    # More timesteps
    acceleration="xformers",              # Memory efficient
    use_tiny_vae=False,                   # Full VAE quality
    cfg_type="full",                      # Full CFG
    width=1024,                           # Higher resolution
    height=1024,
    t_index_list=[20, 25, 30, 35, 40],
)
```

## Troubleshooting

### Common Issues

**Model Not Found**
```python
# Ensure model path is correct
stream = GenericStreamDiffusion(
    model_id_or_path="runwayml/stable-diffusion-v1-5",  # Full repo name
    # ...
)
```

**Out of Memory**
```python
# Reduce memory usage
stream = GenericStreamDiffusion(
    model_id_or_path="model",
    dtype=torch.float16,
    use_tiny_vae=True,
    width=512,  # Smaller resolution
    height=512,
    frame_buffer_size=1,
    # ...
)
```

**Slow Performance**
```python
# Enable acceleration
stream = GenericStreamDiffusion(
    model_id_or_path="model",
    acceleration="tensorrt",  # or "xformers"
    use_lcm_lora=True,
    t_index_list=[32],  # Fewer steps
    # ...
)
```

### Debug Mode

```python
# Check what pipeline was detected
pipeline_types = GenericStreamDiffusion.list_supported_pipelines()
print(f"Supported: {pipeline_types}")

# Manual pipeline selection (if auto-detection fails)
from streamdiffusion import StreamSDXLPipeline
stream = StreamSDXLPipeline(...)  # Use specific pipeline directly
```

## Extension

### Adding New Pipeline Types

```python
from streamdiffusion import GenericStreamDiffusion
from your_custom_pipeline import CustomStreamPipeline

# Register new pipeline type
GenericStreamDiffusion.register_pipeline(
    "YourCustomPipeline", 
    CustomStreamPipeline
)

# Now works automatically
stream = GenericStreamDiffusion(model_id_or_path="custom-model")
```

## Examples

See `examples/generic_usage.py` for complete working examples with different model types and configurations.