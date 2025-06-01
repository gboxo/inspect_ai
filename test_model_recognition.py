#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

from inspect_ai.model import get_model

print("Testing vllm_router model recognition...")

try:
    # Test if the model is recognized (this should fail with a ValueError about missing models, not an unknown model error)
    model = get_model('vllm_router/test')
    print("SUCCESS: vllm_router model is recognized!")
except ValueError as e:
    if "models must be" in str(e) or "At least one model must be specified" in str(e):
        print("SUCCESS: vllm_router model is recognized! (Expected ValueError about missing models parameter)")
    else:
        print(f"ERROR: Unexpected ValueError: {e}")
except Exception as e:
    if "Unknown model" in str(e) or "not found" in str(e):
        print(f"ERROR: vllm_router model is NOT recognized: {e}")
    else:
        print(f"ERROR: Unexpected error: {e}")

# Test with proper parameters
try:
    print("\nTesting with proper models parameter...")
    model = get_model('vllm_router/test', models=['dummy1', 'dummy2'])
    print("SUCCESS: vllm_router model created with models parameter!")
except Exception as e:
    print(f"Expected error (no actual model files): {e}") 