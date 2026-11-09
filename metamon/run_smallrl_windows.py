"""Run Metamon's SmallRL evaluator without FlashAttention on Windows.

Pass the normal ``python -m metamon.rl.evaluate`` arguments to this script.
Set METAMON_CACHE_DIR in the shell before running it.
"""

import os

# AMAGO's VanillaAttention uses torch.compile internally. Its CPU compiler is
# unavailable in a standard Windows Python installation.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import runpy

import amago.nets.traj_encoders  # register the Gin configurable
from amago.nets.transformer import VanillaAttention
import gin


if __name__ == "__main__":
    gin.bind_parameter(
        "amago.nets.traj_encoders.TformerTrajEncoder.attention_type",
        VanillaAttention,
    )
    runpy.run_module("metamon.rl.evaluate", run_name="__main__")
