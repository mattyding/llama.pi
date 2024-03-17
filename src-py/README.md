Instructions for serializing models:
1. Download Llama2 model weights from [Meta](https://llama.meta.com/llama-downloads/).
2. Convert weights to binary format and quantize to int4 or int8 using convert.py.

For instance, from this directory,
```
mkdir ../models
python convert.py --model=/path/to/model --out_dir=../models/7b_q80/ 
```