#!/usr/bin/env python3
"""
Simple test to verify the code structure and basic functionality without heavy dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_class_structure():
    """Test basic class structure without importing heavy dependencies."""
    print("Testing basic class structure...")
    
    # Test that we can read and parse the files
    pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'streamdiffusion', 'pipeline.py')
    sdxl_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'streamdiffusion', 'pipelines', 'stream_sdxl_pipeline.py')
    generic_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'streamdiffusion', 'pipelines', 'generic_pipeline.py')
    
    # Check files exist
    assert os.path.exists(pipeline_path), "Base pipeline.py missing"
    assert os.path.exists(sdxl_path), "SDXL pipeline missing"
    assert os.path.exists(generic_path), "Generic pipeline missing"
    
    # Check key methods exist in base class
    with open(pipeline_path, 'r') as f:
        content = f.read()
        assert 'def get_added_cond_kwargs' in content, "Missing get_added_cond_kwargs method"
        assert 'def use_embeded_prompt' in content, "Missing use_embeded_prompt method"
        assert 'def _get_add_time_ids' in content, "Missing _get_add_time_ids method"
        assert 'def _setup_sdxl_conditioning' in content, "Missing _setup_sdxl_conditioning method"
    
    # Check SDXL pipeline has proper inheritance
    with open(sdxl_path, 'r') as f:
        content = f.read()
        assert 'super().__init__(' in content, "SDXL pipeline missing proper inheritance"
        assert 'text_encoder_2' in content, "SDXL pipeline missing text_encoder_2"
    
    # Check generic pipeline structure
    with open(generic_path, 'r') as f:
        content = f.read()
        assert 'class GenericStreamDiffusion' in content, "Missing GenericStreamDiffusion class"
        assert 'PIPELINE_MAPPING' in content, "Missing pipeline mapping"
        assert '_detect_pipeline_type' in content, "Missing pipeline detection"
    
    print("✓ Class structure test passed")


def test_inheritance_fixes():
    """Test that inheritance issues are fixed."""
    print("Testing inheritance fixes...")
    
    # Read SDXL pipeline
    sdxl_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'streamdiffusion', 'pipelines', 'stream_sdxl_pipeline.py')
    with open(sdxl_path, 'r') as f:
        content = f.read()
    
    # Should NOT have duplicate methods that are now in base class
    assert 'def _get_add_time_ids(' not in content or content.count('def _get_add_time_ids(') == 0, "SDXL has duplicate _get_add_time_ids"
    assert 'def get_added_cond_kwargs(' not in content or content.count('def get_added_cond_kwargs(') == 0, "SDXL has duplicate get_added_cond_kwargs"
    
    # Should use super().__init__
    assert 'super().__init__(' in content, "SDXL doesn't use proper inheritance"
    
    print("✓ Inheritance fixes test passed")


def test_code_duplication_removed():
    """Test that code duplication was removed."""
    print("Testing code duplication removal...")
    
    base_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'streamdiffusion', 'pipeline.py')
    sd_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'streamdiffusion', 'pipelines', 'stream_sd_pipeline.py')
    sdxl_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'streamdiffusion', 'pipelines', 'stream_sdxl_pipeline.py')
    
    # Check that long initialization blocks were replaced with super().__init__
    with open(sd_path, 'r') as f:
        sd_content = f.read()
    
    with open(sdxl_path, 'r') as f:
        sdxl_content = f.read()
    
    # Should have super().__init__ calls
    assert 'super().__init__(' in sd_content, "SD pipeline missing super init"
    assert 'super().__init__(' in sdxl_content, "SDXL pipeline missing super init"
    
    # Should NOT have long initialization blocks
    assert 'self.device = pipe.device\n        self.dtype = self.dtype\n        self.generator = None' not in sd_content, "SD still has duplicate initialization"
    assert 'self.device = pipe.device\n        self.dtype = self.dtype\n        self.generator = None' not in sdxl_content, "SDXL still has duplicate initialization"
    
    print("✓ Code duplication removal test passed")


def run_tests():
    """Run all simple tests."""
    print("Running simple structure tests...\n")
    
    test_class_structure()
    test_inheritance_fixes()
    test_code_duplication_removed()
    
    print("\n✓ All simple tests passed!")
    print("Note: Full integration tests require proper environment setup.")


if __name__ == "__main__":
    run_tests()