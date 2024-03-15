""""
prune llama ff layers and save as bitvectors
the way we run this experiment is that we prune the model in pytorch and save a bitvec of nonzero weights
then in our forwardq.c call, we load the bitvec and zero out all the weights that are zero in the bitvec
This allows us to simulate the inference/performance of a pruned model without actually implementing our serializing/deserialization methods, which is a lot of work but feasible in practice.
"""

import argparse
import struct
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from convert import serialize_fp32, load_meta_model

DEBUG = 1


def serialize_bitvec(file, tensor):
    """
    Serialize a float tensor into a bitvector.
    """
    # Convert the tensor to a ByteTensor of 0s and 1s
    bitvec = tensor.round().clamp(0, 1).to(torch.uint8).view(-1)
    bytes_data = struct.pack(f'{"B" * len(bitvec)}', *bitvec.tolist())
    if DEBUG:
        print("num nonzero", bitvec.sum().item(), "num total", len(bitvec))

    file.write(bytes_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--out_dir", type=str, help="the output dir")
    args = parser.parse_args()

    model = load_meta_model(args.model)

    if model is None:
        parser.error("Can't load input model!")

    hidden_dim = 11008
    dim = 4096
    n_layers = 32

    w1_layers = [nn.Linear(hidden_dim, dim) for l in range(n_layers)]
    w2_layers = [nn.Linear(dim, hidden_dim) for l in range(n_layers)]
    w3_layers = [nn.Linear(hidden_dim, dim) for l in range(n_layers)]

    for i, layer in enumerate(model.layers):
        w1_layers[i].weight.data = layer.feed_forward.w1.weight.data.clone()
        w2_layers[i].weight.data = layer.feed_forward.w2.weight.data.clone()
        w3_layers[i].weight.data = layer.feed_forward.w3.weight.data.clone()

    # make a perm copy of w1_weights

    # prune the weights
    # prune.ln_structured is a bit diff. we don't use
    for prune_fn in [prune.l1_unstructured]:
        for percent in [0.5]:
            print("pruning", prune_fn.__name__, percent)
            for w in w1_layers:
                prune_fn(w, name="weight", amount=percent)
            for w in w2_layers:
                prune_fn(w, name="weight", amount=percent)
            for w in w3_layers:
                prune_fn(w, name="weight", amount=percent)

            # restore the weights
            for w in w1_layers:
                prune.remove(w, name="weight")
            for w in w2_layers:
                prune.remove(w, name="weight")
            for w in w3_layers:
                prune.remove(w, name="weight")

            print("saving file", prune_fn.__name__, percent)
            # save the bitvec
            with open(
                f"{args.out_dir}/bitvec_{prune_fn.__name__}_{percent}.bin", "wb+"
            ) as f:
                for w in w1_layers:
                    serialize_bitvec(f, w.weight)
                for w in w2_layers:
                    serialize_bitvec(f, w.weight)
                for w in w3_layers:
                    serialize_bitvec(f, w.weight)
            print("save complete")

            # restore the weights
            for i, layer in enumerate(model.layers):
                w1_layers[i].weight.data = layer.feed_forward.w1.weight.data.clone()
                w2_layers[i].weight.data = layer.feed_forward.w2.weight.data.clone()
                w3_layers[i].weight.data = layer.feed_forward.w3.weight.data.clone()
