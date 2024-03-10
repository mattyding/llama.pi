This dir contains files for processing and evaluating the llama2 models. The files are largely borrowed from karpathy's llama2.c implementation, with modifications for our experiments.

Files:
- model.py:     Transformer model definition in Python (unchanged)
- convert.py:    Convert weights to binary format, perform int4 and int8 quantization (int4 added by us)
- profiler.c:    Memory profiler to estimate RAM requirements (entirely new)

---
Instructions for serializing models:
1. Download Llama2 model weights from [Meta](https://llama.meta.com/llama-downloads/).
2. Convert weights to binary format and quantize to int4 or int8 using convert.py.
```
mkdir ../models/
python export.py ../models/7b-fp32.bin --version 1 --meta-llama=/path/to/weights/dir
python export.py ../models/7b-q80.bin --version 2 --meta-llama=/path/to/weights/dir
python export.py ../models/7b-q40.bin --version 3 --meta-llama=/path/to/weights/dir
```