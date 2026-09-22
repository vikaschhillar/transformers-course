"""Quick check that the course environment works (Chapter 0 setup)."""
import sys

import torch
import transformers
from transformers import pipeline

print(f"Python        {sys.version.split()[0]}")
print(f"transformers  {transformers.__version__}")
print(f"torch         {torch.__version__}")
print(f"Apple GPU (MPS) available: {torch.backends.mps.is_available()}")

classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))
