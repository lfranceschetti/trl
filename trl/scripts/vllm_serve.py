# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ctypes
import logging
import os
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence
import time

import torch
import torch.distributed as dist
import numpy as np
import sys

from trl import TrlParser
from trl.import_utils import is_fastapi_available, is_pydantic_available, is_uvicorn_available, is_vllm_available


if is_fastapi_available():
    from fastapi import BackgroundTasks, FastAPI


if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn

# When no GPU is available, importing 'Worker' from 'vllm.worker.worker' fails with the following error:
# libcuda.so.1: cannot open shared object file: No such file or directory.
# To prevent this error, we check if libcuda is available before importing 'Worker'. While this check is not
# crucial—since vLLM and TRL are not intended to run without a GPU—it helps avoid errors when running CI on a machine
# without a GPU.
try:
    ctypes.CDLL("libcuda.so.1")
except OSError:
    libcuda_available = False
else:
    libcuda_available = True

if is_vllm_available() and libcuda_available:
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.worker.worker import Worker
else:
    Worker = object

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
update_sequence_counter = 0

# Create a queue and synchronization primitives for ordered parameter updates
param_update_queue = deque()
queue_lock = asyncio.Lock()
update_in_progress = False


class WeightSyncWorker(Worker):
    """
    A vLLM worker that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """

    def __init__(self, *args, **kwargs):
        if not is_vllm_available():
            raise ImportError(
                "vLLM is required to use the WeightSyncWorker. Please install it using `pip install vllm`."
            )

        super().__init__(*args, **kwargs)

        # The following attributes are initialized when `init_communicator` method is called.
        self.pynccl_comm = None  # Communicator for weight updates
        self.client_rank = None  # Source rank for broadcasting updated weights

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to
        communicate with vLLM workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
        """
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)

        # Initialize the NCCL-based communicator for weight synchronization.
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: torch.dtype, shape: Sequence[int], sequence_num: int) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`torch.dtype`):
                Data type of the weight tensor (e.g., `torch.float32`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
            sequence_num (`int`):
                Sequence number to track the order of updates.
        """
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        try:
            logger.info(f"SEQUENCE: Starting update #{sequence_num} for param {name}")

            size_mb = np.prod(shape) * torch.tensor([], dtype=dtype).element_size() / (1024**2)
            free_memory = torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)
            free_memory_gb = free_memory / (1024**3)
            logger.info(f"Allocating tensor of shape {shape}, size {size_mb:.2f} MB, free memory: {free_memory_gb:.2f} GB")

            logger.info(f"SEQUENCE: Allocating memory for update #{sequence_num}")

            # Allocate memory for the incoming weight tensor on the correct device.
            weight = torch.empty(shape, dtype=dtype, device=self.device)

            logger.info(f"SEQUENCE: Entering broadcast for update #{sequence_num}")

            # Use NCCL to broadcast the updated weights from the client (src) to all workers.
            self.pynccl_comm.broadcast(weight, src=self.client_rank, stream=torch.cuda.current_stream())
            
            logger.info(f"SEQUENCE: Entering barrier for update #{sequence_num}")
            
            # Use a timeout to prevent indefinite hangs
            barrier_timeout = 30.0  # 30 seconds timeout
            barrier_start = time.time()
            
            try:
                self.pynccl_comm.group.barrier()
            except Exception as e:
                if time.time() - barrier_start > barrier_timeout:
                    logger.error(f"Barrier timeout after {barrier_timeout}s for update #{sequence_num}")
                    raise TimeoutError(f"Barrier operation timed out after {barrier_timeout}s") from e
                raise

            # Update the model weights
            self.model_runner.model.load_weights(weights=[(name, weight)])
            
            logger.info(f"SEQUENCE: Completed update #{sequence_num} for param {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error during update #{sequence_num} for param {name}: {e}")
            return False

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None

    def log_memory_usage(self):
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            logger.info(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    def log_group_state(self, group):
        logger.info(f"Group size: {group.size()}, rank: {group.rank()}, is_initialized: {dist.is_initialized()}")


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
    """

    model: str = field(metadata={"help": "Model name or path to load the model from."})
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )


async def process_param_update_queue(llm):
    """
    Process parameter updates from the queue in strict order.
    """
    global update_in_progress
    
    async with queue_lock:
        update_in_progress = True
    
    try:
        while True:
            # Check if queue is empty
            async with queue_lock:
                if not param_update_queue:
                    update_in_progress = False
                    break
                
                # Get next update from queue (but don't remove it yet)
                name, dtype, shape, sequence_num = param_update_queue[0]
            
            logger.info(f"SEQUENCE: Processing queued update #{sequence_num} for param {name}")
            
            try:
                # Process update synchronously
                result = llm.collective_rpc("update_named_param", args=(name, dtype, shape, sequence_num))
                
                # Only remove from queue after successful completion
                async with queue_lock:
                    param_update_queue.popleft()
                
                logger.info(f"SEQUENCE: Successfully completed update #{sequence_num} for param {name}")
                
            except Exception as e:
                logger.error(f"Error processing update #{sequence_num} for {name}: {e}")
                # Remove the failed update to prevent blocking the queue
                async with queue_lock:
                    param_update_queue.popleft()
    
    except Exception as e:
        logger.error(f"Error in queue processor: {e}")
        async with queue_lock:
            update_in_progress = False


def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )

    if not is_vllm_available():
        raise ImportError("vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`.")

    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        dtype=script_args.dtype,
        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        # This is particularly useful here because we generate completions from the same prompts.
        enable_prefix_caching=script_args.enable_prefix_caching,
        max_model_len=script_args.max_model_len,
        worker_cls="trl.scripts.vllm_serve.WeightSyncWorker",
    )

    app = FastAPI()

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @app.get("/get_tensor_parallel_size/")
    async def get_tensor_parallel_size():
        """
        Retrieves the tensor parallel size from the LLM engine.

        Returns:
            `dict`:
                A dictionary containing the tensor parallel size.

        Example response:
        ```json
        {"tensor_parallel_size": 8}
        ```
        """
        return {"tensor_parallel_size": llm.llm_engine.parallel_config.tensor_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts.

        Args:
            request (`GenerateRequest`):
                - `prompts` (list of `str`): A list of prompts (text strings) for the model to generate completions.

        Returns:
            `GenerateResponse`:
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.

        Example request:
        ```json
        {"prompts": ["Hello world", "What is AI?"]}
        ```

        Example response:
        ```json
        {"completion_ids": [[101, 102, 103], [201, 202, 203]]}
        ```
        """

        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        # Sampling parameters
        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
        )
        all_outputs = llm.generate(request.prompts, sampling_params=sampling_params)
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        return {"completion_ids": completion_ids}

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest, background_tasks: BackgroundTasks):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server
        workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        background_tasks.add_task(
            llm.collective_rpc,
            "init_communicator",
            args=(request.host, request.port, script_args.tensor_parallel_size + 1),
        )
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        """
        Updates the model weights with the provided tensor.
        Uses a queue system to ensure parameters are processed in order.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight tensor.
        """
        global update_sequence_counter, update_in_progress
        update_sequence_counter += 1
        sequence_num = update_sequence_counter
        
        logger.info(f"SEQUENCE: Received HTTP request #{sequence_num} for param {request.name}")
        
        # Parse dtype
        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        
        # Add to queue and process synchronously
        async with queue_lock:
            param_update_queue.append((request.name, dtype, request.shape, sequence_num))
            
            # Start the queue processor if it's not already running
            if not update_in_progress:
                asyncio.create_task(process_param_update_queue(llm))
        
        return {"message": f"Request #{sequence_num} queued for param {request.name}"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        success = llm.llm_engine.reset_prefix_cache()
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        """
        Closes the weight update group and cleans up associated resources.
        """
        llm.collective_rpc("close_communicator")
        return {"message": "Request received, closing communicator"}

    # Start the server
    uvicorn.run(app, host=script_args.host, port=script_args.port)

    dist.destroy_process_group()


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)