# import pickle
from functools import lru_cache

import numpy as np
import torch
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_embedder_ensembele(image: np.ndarray):
    triton_client = get_client()
    input_image = InferInput(
        name="INPUTS", shape=image.shape, datatype=np_to_triton_dtype(image.dtype)
    )
    input_image.set_data_from_numpy(image, binary_data=True)

    infer_output = InferRequestedOutput("OUTPUTS", binary_data=True)
    query_response = triton_client.infer(
        "onnx-resnet", [input_image], outputs=[infer_output]
    )
    logits = query_response.as_numpy("OUTPUTS")[0]
    return logits


def main():
    images = []
    logits = torch.tensor(
        [call_triton_embedder_ensembele(image).argmax() for image in images]
    )
    for logit in logits:
        print(logit)


if __name__ == "__main__":
    main()
