#!/usr/bin/env python3
from run import run


def main():
    params = {}
    params["framework"] = "pytorch"
    params["problem"] = "resnet50"
    params["mode"] = "training"
    for batch_size in range(1, 256):
        params["gpus"] = "0"
        params["backend"] = "native"
        params["batch_size"] = batch_size
        prob_size = params["batch_size"] * 4
        params["problem_size"] = f"{prob_size}"
        print(batch_size)
        run(params)
        params.pop("gpus")
        params["backend"] = "DNNL"
        run(params)


if __name__ == "__main__":
    main()
