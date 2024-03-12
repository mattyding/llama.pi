# modification of export.py that splits into multiple files
import os
import struct
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import ModelArgs, Transformer


# --- utilities


def serialize_fp32(file, tensor):
    """writes one fp32 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


def serialize_int8(file, tensor):
    """writes one int8 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f"{len(d)}b", *d)
    file.write(b)


def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float()  # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:, None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr


# --- export fns
def fp32_export(model, out_dir, group_size=64):
    """
    Export FP32 model weights into .bin files to be read from C. Each layer is exported into a separate file.
       That is:
       - all tensors are exported in fp32
    File format:
    - header: 320 bytes
        - magic: uint32 of "ak32" in ASCII
        - version: int
        - params: 7 ints
            - dim
            - hidden_dim
            - n_layers
            - n_heads
            - n_kv_heads
            - vocab_size
            - max_seq_len
        - shared_classifier: int
    """
    version = 1
    magic = 0x616B3332

    os.makedirs(out_dir, exist_ok=True)
    # write config file
    with open(out_dir + "/config.bin", "wb+") as f:
        # 36-byte header
        # 1) write magic
        f.write(struct.pack("I", magic))
        # 2) write version, which will be int
        f.write(struct.pack("i", version))
        # 3) write the params, which will be 7 ints
        p = model.params
        hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack(
            "iiiiiii",
            p.dim,
            hidden_dim,
            p.n_layers,
            p.n_heads,
            n_kv_heads,
            p.vocab_size,
            p.max_seq_len,
        )
        f.write(header)

    # write token embedding table
    with open(out_dir + "/tok_emb.bin", "wb+") as f:
        print("writing tok_emb.bin")
        # 4-byte header: magic
        f.write(struct.pack("I", magic))
        # token embedding table
        serialize_fp32(f, model.tok_embeddings.weight)

    # write the weights for each layer into a separate .bin file
    for i, layer in enumerate(model.layers):
        print(f"writing layer {i}.bin")
        with open(f"{out_dir}/layer{i}.bin", "wb+") as f:
            # 8-byte header
            # 1) magic
            f.write(struct.pack("I", magic))
            # 2) layer id: int
            f.write(struct.pack("i", i))

            # model weights
            serialize_fp32(f, layer.attention_norm.weight)
            serialize_fp32(f, layer.ffn_norm.weight)
            serialize_fp32(f, layer.attention.wq.weight)
            serialize_fp32(f, layer.attention.wk.weight)
            serialize_fp32(f, layer.attention.wv.weight)
            serialize_fp32(f, layer.attention.wo.weight)
            serialize_fp32(f, layer.feed_forward.w1.weight)
            serialize_fp32(f, layer.feed_forward.w2.weight)
            serialize_fp32(f, layer.feed_forward.w3.weight)

    # write the output weights
    with open(f"{out_dir}/output.bin", "wb+") as f:
        # 4-byte header: magic
        f.write(struct.pack("I", magic))
        serialize_fp32(f, model.output.weight)

    # write final rmsnorm weights
    with open(f"{out_dir}/norm.bin", "wb+") as f:
        print("writing norm.bin")
        # 4-byte header: magic
        f.write(struct.pack("I", magic))
        serialize_fp32(f, model.norm.weight)


def q80_export(model, out_dir, group_size=64):
    """
    Export Q8_0 model weights into .bin files to be read from C. Each layer is exported into a separate file.
       That is:
       - quantize all weights to symmetric int8, in range [-127, 127]
       - all other tensors (the rmsnorm params) are kept and exported in fp32
       - quantization is done in groups of group_size to reduce the effects of any outliers
    File format:
    - header: 352 bytes
        - magic: uint32 of "0x80" in ASCII
        - version: int
        - params: 7 ints
            - dim
            - hidden_dim
            - n_layers
            - n_heads
            - n_kv_heads
            - vocab_size
            - max_seq_len
        - shared_classifier: int
        - group_size: int
    """
    version = 2
    magic = 0x616B3830

    # let's first do some validation for this export type
    while model.params.dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)

    os.makedirs(out_dir, exist_ok=True)
    # write config file
    with open(out_dir + "/config.bin", "wb+") as f:
        # 44-byte header
        # 1) write magic, which will be uint32 of "ak80" in ASCII
        f.write(struct.pack("I", magic))
        # 2) write version, which will be int
        f.write(struct.pack("i", version))
        # 3) write the params, which will be 7 ints
        p = model.params
        hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack(
            "iiiiiii",
            p.dim,
            hidden_dim,
            p.n_layers,
            p.n_heads,
            n_kv_heads,
            p.vocab_size,
            p.max_seq_len,
        )
        f.write(header)
        # 4) write some other flags
        f.write(struct.pack("i", int(shared_classifier)))
        f.write(struct.pack("i", group_size))  # group size used for quantization

    # write token embedding table
    with open(out_dir + "/tok_emb.bin", "wb+") as f:
        print("writing tok_emb.bin")
        # 4-byte header: magic
        f.write(struct.pack("I", magic))
        # 2) write the token embedding table
        serialize_fp32(f, model.tok_embeddings.weight)

    ew = []

    # write the weights for each layer into a separate .bin file
    for i, layer in enumerate(model.layers):
        print(f"writing layer {i}.bin")
        # quantize weights
        q80_wq, scale_wq, maxerr_wq = quantize_q80(
            layer.attention.wq.weight, group_size
        )
        q80_wk, scale_wk, maxerr_wk = quantize_q80(
            layer.attention.wk.weight, group_size
        )
        q80_wv, scale_wv, maxerr_wv = quantize_q80(
            layer.attention.wv.weight, group_size
        )
        q80_wo, scale_wo, maxerr_wo = quantize_q80(
            layer.attention.wo.weight, group_size
        )
        q80_ff1, scale_ff1, maxerr_ff1 = quantize_q80(
            layer.feed_forward.w1.weight, group_size
        )
        q80_ff2, scale_ff2, maxerr_ff2 = quantize_q80(
            layer.feed_forward.w2.weight, group_size
        )
        q80_ff3, scale_ff3, maxerr_ff3 = quantize_q80(
            layer.feed_forward.w3.weight, group_size
        )
        # write to file
        with open(f"{out_dir}/layer{i}.bin", "wb+") as f:
            # 8-byte header
            # 1) magic
            f.write(struct.pack("I", magic))
            # 2) layer id: int
            f.write(struct.pack("i", i))

            # fp32 weights
            serialize_fp32(f, layer.attention_norm.weight)
            serialize_fp32(f, layer.ffn_norm.weight)
            # q80 weights (int8) and scale (fp32)
            serialize_int8(f, q80_wq)
            serialize_fp32(f, scale_wq)
            serialize_int8(f, q80_wk)
            serialize_fp32(f, scale_wk)
            serialize_int8(f, q80_wv)
            serialize_fp32(f, scale_wv)
            serialize_int8(f, q80_wo)
            serialize_fp32(f, scale_wo)
            serialize_int8(f, q80_ff1)
            serialize_fp32(f, scale_ff1)
            serialize_int8(f, q80_ff2)
            serialize_fp32(f, scale_ff2)
            serialize_int8(f, q80_ff3)
            serialize_fp32(f, scale_ff3)

            # append max errors
            ew.append((maxerr_wq, layer.attention.wq.weight.shape))
            ew.append((maxerr_wk, layer.attention.wk.weight.shape))
            ew.append((maxerr_wv, layer.attention.wv.weight.shape))
            ew.append((maxerr_wo, layer.attention.wo.weight.shape))
            ew.append((maxerr_ff1, layer.feed_forward.w1.weight.shape))
            ew.append((maxerr_ff2, layer.feed_forward.w2.weight.shape))
            ew.append((maxerr_ff3, layer.feed_forward.w3.weight.shape))

            # logging
            # print(
            #     f"Layer {i}: max errors: wq={maxerr_wq}, wk={maxerr_wk}, wv={maxerr_wv}, wo={maxerr_wo}, ff1={maxerr_ff1}, ff2={maxerr_ff2}, ff3={maxerr_ff3}"
            # )

    # write the output weights
    if not shared_classifier:
        q80_output, scale_output, maxerr_output = quantize_q80(
            model.output.weight, group_size
        )
        with open(f"{out_dir}/output.bin", "wb+") as f:
            # 4-byte header: magic
            f.write(struct.pack("I", magic))

            serialize_fp32(f, model.output.weight)
            serialize_int8(f, q80_output)
            serialize_fp32(f, scale_output)
            ew.append((maxerr_output, model.output.weight.shape))

    # write final rmsnorm weights
    with open(f"{out_dir}/norm.bin", "wb+") as f:
        print("writing norm.bin")
        # 4-byte header: magic
        f.write(struct.pack("I", magic))
        serialize_fp32(f, model.norm.weight)

    # print the highest error across all weights, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")


# --- load model


def load_meta_model(model_path):
    params_path = os.path.join(model_path, "params.json")
    with open(params_path) as f:
        params = json.load(f)
        print(params)

    model_paths = sorted(list(Path(model_path).glob("consolidated.*.pth")))
    models = [torch.load(p, map_location="cpu") for p in model_paths]

    def concat_weights(models):
        state_dict = {}
        for name in list(models[0]):
            tensors = [model[name] for model in models]
            if len(tensors) == 1 or len(tensors[0].shape) == 1:
                state_dict[name] = tensors[0]
                continue
            is_axis_1 = (
                name.startswith("tok_embeddings.")
                or name.endswith(".attention.wo.weight")
                or name.endswith(".feed_forward.w2.weight")
            )
            axis = 1 if is_axis_1 else 0
            state_dict[name] = torch.cat(tensors, dim=axis)
            for model in models:
                del model[name]
        return state_dict

    state_dict = concat_weights(models)
    del models

    # set ModelArgs
    config = ModelArgs()
    config.dim = params["dim"]
    config.n_layers = params["n_layers"]
    config.n_heads = params["n_heads"]
    config.n_kv_heads = params.get("n_kv_heads") or params["n_heads"]
    config.multiple_of = params["multiple_of"]
    config.norm_eps = params["norm_eps"]

    config.vocab_size = state_dict["tok_embeddings.weight"].shape[0]
    config.max_seq_len = 2048

    # create a new Transformer object and set weights
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(state_dict["tok_embeddings.weight"])
    model.norm.weight = nn.Parameter(state_dict["norm.weight"])

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(
            state_dict[f"layers.{i}.attention_norm.weight"]
        )
        layer.attention.wq.weight = nn.Parameter(
            state_dict[f"layers.{i}.attention.wq.weight"]
        )
        layer.attention.wk.weight = nn.Parameter(
            state_dict[f"layers.{i}.attention.wk.weight"]
        )
        layer.attention.wv.weight = nn.Parameter(
            state_dict[f"layers.{i}.attention.wv.weight"]
        )
        layer.attention.wo.weight = nn.Parameter(
            state_dict[f"layers.{i}.attention.wo.weight"]
        )
        layer.ffn_norm.weight = nn.Parameter(state_dict[f"layers.{i}.ffn_norm.weight"])
        layer.feed_forward.w1.weight = nn.Parameter(
            state_dict[f"layers.{i}.feed_forward.w1.weight"]
        )
        layer.feed_forward.w2.weight = nn.Parameter(
            state_dict[f"layers.{i}.feed_forward.w2.weight"]
        )
        layer.feed_forward.w3.weight = nn.Parameter(
            state_dict[f"layers.{i}.feed_forward.w3.weight"]
        )

    # final classifier
    model.output.weight = nn.Parameter(state_dict["output.weight"])
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--out_dir", type=str, help="the output dir")
    parser.add_argument("--version", default=2, type=int, help="v2 == q80")
    parser.add_argument(
        "--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32"
    )
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model = load_meta_model(args.model)

    if model is None:
        parser.error("Can't load input model!")

    # export
    if args.version == 2:
        q80_export(model, args.out_dir)
