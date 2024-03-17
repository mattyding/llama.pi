author: matthew ding

w24 final course project for {cs140e, cs224n}

we attempt to run llama2-7b inference on the raspberry pi zero [[paper](paper.pdf)]

This repository is organized as follows:
- `/src` holds the r/pi source code. the starter code for the pi is contained in `/libpi`. `/src/llama` contains the main code for our project. `/src/sanity/` contains a simple sanity check to make sure your Pi setup is working correctly. However, we stopped working in this repo after we determined that it was probably not feasible to run the model on the Pi Zero, so the more up-to-date code is in `/src-osx`.
- `/src-osx` contains implementations of the same code to run on our Mac, for testing and benchmarking purposes
- `/src-py` contains Python code. Importantly `convert.py` serializes and quantizes the models. Data analysis is performed in Jupyter notebooks starting with `exp-*`. 

File breakdown of the `/src` and `/src-osx` directories:
- The main function is `run.c` (or `run-seg.c` and `runq-seg.c` in `/src`). This is the entry point for the program. It performs inference for a given prompt. 
- The forward pass is implemented in `forward.c` and `forwardq.c`. 
- `algo.*` and `fileutils.*` contain implementations of some helpful utils. `quant.*` contains helper functions for quantization. `prune.*` contains helper functions for pruning. `tokenizer.c` and `sampler.c` contain the tokenizer and sampler, respectively. Memory and CPU profilers are implemented in `mprof.c` and `tprof.c`. 

To modify our code, download the llama2 weights, serialize/quantize using `/src-py/convert.py`, set up the r/pi as described in the [cs140e repo](https://github.com/dddrrreee/cs140e-24win/tree/main/labs/0-pi-setup), and then I believe the code should work as is.

Attribution: this code heavily used karpathy's [llama2.c](https://github.com/karpathy/llama2.c) repo as a starting point. Our implementation of the Llama2 forward pass is a modification of his, and we use the same code for serializing and quantizing the model.

Our contributions include rewriting the code to work with our r/pi starter code and further modifying it with various memory optimizations. These are described in the paper. We also implement methods of profiling the code and run several experiments to measure cpu and memory performance.