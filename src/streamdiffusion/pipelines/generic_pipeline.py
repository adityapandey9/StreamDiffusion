import os
from pathlib import Path
from typing import List, Literal, Optional, Union, Dict, Type
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from .stream_sd_pipeline import StreamSDPipeline
from .stream_sdxl_pipeline import StreamSDXLPipeline


class GenericStreamDiffusion:
    """
    Generic StreamDiffusion pipeline that automatically detects and loads
    the appropriate pipeline (SD or SDXL) based on the model architecture.
    """
    
    PIPELINE_MAPPING = {
        "StableDiffusionPipeline": StreamSDPipeline,
        "StableDiffusionXLPipeline": StreamSDXLPipeline,
    }
    
    def __new__(
        cls,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
    ):
        """
        Automatically detects the appropriate pipeline and creates the correct instance.
        
        Parameters are the same as the individual pipeline classes.
        """
        pipeline_class = cls._detect_pipeline_type(model_id_or_path)
        
        return pipeline_class(
            model_id_or_path=model_id_or_path,
            t_index_list=t_index_list,
            lora_dict=lora_dict,
            mode=mode,
            output_type=output_type,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            device=device,
            dtype=dtype,
            frame_buffer_size=frame_buffer_size,
            width=width,
            height=height,
            warmup=warmup,
            acceleration=acceleration,
            do_add_noise=do_add_noise,
            device_ids=device_ids,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            enable_similar_image_filter=enable_similar_image_filter,
            similar_image_filter_threshold=similar_image_filter_threshold,
            similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
            use_denoising_batch=use_denoising_batch,
            cfg_type=cfg_type,
            seed=seed,
            use_safety_checker=use_safety_checker,
            engine_dir=engine_dir,
        )
    
    @classmethod
    def _detect_pipeline_type(cls, model_id_or_path: str) -> Type:
        """
        Detects the pipeline type based on the model configuration.
        
        Returns the appropriate pipeline class.
        """
        try:
            # Try to load model config to detect type
            from diffusers import AutoPipelineForText2Image
            
            # Load pipeline temporarily to check its type
            temp_pipe = AutoPipelineForText2Image.from_pretrained(
                model_id_or_path, 
                torch_dtype=torch.float16,
                device_map=None  # Don't load to device yet
            )
            
            pipeline_class_name = temp_pipe.__class__.__name__
            
            # Clean up temporary pipeline
            del temp_pipe
            torch.cuda.empty_cache()
            
            if pipeline_class_name in cls.PIPELINE_MAPPING:
                return cls.PIPELINE_MAPPING[pipeline_class_name]
            else:
                # Default to SD pipeline for unknown types
                print(f"Unknown pipeline type: {pipeline_class_name}, defaulting to SD pipeline")
                return StreamSDPipeline
                
        except Exception as e:
            print(f"Error detecting pipeline type: {e}, defaulting to SD pipeline")
            return StreamSDPipeline
    
    @classmethod
    def list_supported_pipelines(cls) -> List[str]:
        """Returns a list of supported pipeline types."""
        return list(cls.PIPELINE_MAPPING.keys())
    
    @classmethod
    def register_pipeline(cls, pipeline_name: str, pipeline_class: Type):
        """
        Register a new pipeline type.
        
        Args:
            pipeline_name: Name of the diffusion pipeline class
            pipeline_class: StreamDiffusion wrapper class
        """
        cls.PIPELINE_MAPPING[pipeline_name] = pipeline_class