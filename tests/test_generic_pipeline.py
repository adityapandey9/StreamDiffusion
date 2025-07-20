#!/usr/bin/env python3
"""
Basic test script for the GenericStreamDiffusion pipeline.
This script tests the pipeline detection and initialization without requiring heavy models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from streamdiffusion.pipelines.generic_pipeline import GenericStreamDiffusion


def test_pipeline_mapping():
    """Test that pipeline mapping contains expected entries."""
    print("Testing pipeline mapping...")
    
    expected_pipelines = [
        "StableDiffusionPipeline", 
        "StableDiffusionXLPipeline"
    ]
    
    supported = GenericStreamDiffusion.list_supported_pipelines()
    print(f"Supported pipelines: {supported}")
    
    for pipeline in expected_pipelines:
        assert pipeline in supported, f"Missing pipeline: {pipeline}"
    
    print("✓ Pipeline mapping test passed")


def test_pipeline_registration():
    """Test pipeline registration functionality."""
    print("Testing pipeline registration...")
    
    class MockPipeline:
        pass
    
    original_count = len(GenericStreamDiffusion.list_supported_pipelines())
    GenericStreamDiffusion.register_pipeline("TestPipeline", MockPipeline)
    
    new_count = len(GenericStreamDiffusion.list_supported_pipelines())
    assert new_count == original_count + 1, "Pipeline registration failed"
    assert "TestPipeline" in GenericStreamDiffusion.list_supported_pipelines()
    
    print("✓ Pipeline registration test passed")


def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from streamdiffusion import GenericStreamDiffusion, StreamSDPipeline, StreamSDXLPipeline
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        raise


def run_tests():
    """Run all tests."""
    print("Running GenericStreamDiffusion tests...\n")
    
    test_imports()
    test_pipeline_mapping()
    test_pipeline_registration()
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_tests()