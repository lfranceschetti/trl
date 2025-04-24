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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download



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
    from trl.extras.stateless_group import CustomStatelessProcessGroup
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
        pg = CustomStatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)

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

            size_mb = np.prod(shape) * torch.tensor([], dtype=dtype).element_size() / (1024**2)
            free_memory = torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)
            free_memory_gb = free_memory / (1024**3)

            # Allocate memory for the incoming weight tensor on the correct device.
            weight = torch.empty(shape, dtype=dtype, device=self.device)

            # Use NCCL to broadcast the updated weights from the client (src) to all workers.
            self.pynccl_comm.broadcast(weight, src=self.client_rank, stream=torch.cuda.current_stream())
            
            # Use a timeout to prevent indefinite hangs
            self.pynccl_comm.group.barrier()

            # Update the model weights
            self.model_runner.model.load_weights(weights=[(name, weight)])
            
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

    conversation_turns : int = field(
        default=5,
        metadata={"help": "Number of turns in the conversation."},
    )

    starting_agent: bool = field(
        default=True,
        metadata={"help": "Whether this agent should start the conversation. If False, the partner agent will start."},
    )

    partner_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the partner on"},
    )
    partner_port: str = field(
        default=8001,
        metadata={"help": "Host address to run the partner on"},
    )

    adapter: str = field(
        default=None,
        metadata={"help": "Adapter to use for the model."},
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
    quantization: str = field(
        default=None,
        metadata={
            "help": "Quantization to use for vLLM generation. If set to 'auto', the quantization will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
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
    ),
    enable_weight_sync: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to enable weight synchronization between the server and the partner."
        },
    ),
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )


def main(script_args: ScriptArguments):


    logger.info(f"Starting vLLM server with arguments: {script_args}")

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

    logger.info(f"Loading model: {script_args.model}")

    enable_lora = True if script_args.adapter else False
    try:

        #Temporary fix, https://github.com/vllm-project/vllm/issues/12967 should be set dynamically
        llm = LLM(
            model=script_args.model,
            revision=script_args.revision,
            tensor_parallel_size=script_args.tensor_parallel_size,
            gpu_memory_utilization=script_args.gpu_memory_utilization,
            dtype=script_args.dtype,
            enforce_eager=True,
            # enable_lora=enable_lora,
            # max_lora_rank=16,
            enable_prefix_caching=script_args.enable_prefix_caching,
            max_model_len=script_args.max_model_len,
            worker_cls="trl.scripts.vllm_serve.WeightSyncWorker" if script_args.enable_weight_sync else "auto",
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)

        logger.info(f"Loading tokenizer from: {script_args.model}")


    if script_args.adapter:
        logger.info(f"Loading adapter from: {script_args.adapter}")
        lora_path = snapshot_download(script_args.adapter)
        logger.info(f"Adapter downloaded to: {lora_path}")
        lora_request = LoRARequest("adapter", 1, lora_path)
    else:
        lora_request = None

        
    try:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model)

        logger.info("Setting tokenizer pad token to EOS token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
        raise

    app = FastAPI()

    # Add memory management endpoints
    @app.post("/clear_cache/")
    async def clear_cache():
        """
        Clears the KV cache and prefix cache to free up memory.
        """
        try:
            llm.llm_engine.reset_prefix_cache()
            # Clear CUDA cache
            torch.cuda.empty_cache()
            return {"status": "success", "message": "Cache cleared successfully"}
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return {"status": "error", "message": str(e)}

    @app.get("/memory_status/")
    async def memory_status():
        """
        Returns current memory usage statistics.
        """
        try:
            memory_stats = {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**3,  # GB
            }
            return {"status": "success", "memory_stats": memory_stats}
        except Exception as e:
            logger.error(f"Failed to get memory status: {e}")
            return {"status": "error", "message": str(e)}

    # Add reconnection handling
    @app.post("/reconnect/")
    async def reconnect():
        """
        Attempts to reinitialize the model and communicator.
        """
        try:
            # Close existing communicator if any
            if hasattr(llm, 'collective_rpc'):
                llm.collective_rpc("close_communicator")
            
            # Reinitialize the model
            llm.llm_engine.reset()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            return {"status": "success", "message": "Reconnected successfully"}
        except Exception as e:
            logger.error(f"Failed to reconnect: {e}")
            return {"status": "error", "message": str(e)}

    def tokenize_messages(messages, tokenizer, sampled_h, starting_agent):
        """
        Convert messages to token IDs and attention masks.
        
        Args:
            messages: List of message dictionaries
            tokenizer: Tokenizer to use
            
        Returns:
            Tuple of (token_ids, attention_mask, assistant_mask)
        """
        MAX_LENGTH = 4096
        
        if sampled_h is not None and starting_agent:
            messages = messages[:2*sampled_h+2]
        elif sampled_h is not None and not starting_agent:
            messages = messages[:2*sampled_h+3]

        # Apply chat template to get the full tokenized conversation
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        
        # Create properly sized tensors
        token_ids = torch.full((MAX_LENGTH,), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((MAX_LENGTH,), dtype=torch.long)
        assistant_mask = torch.zeros((MAX_LENGTH,), dtype=torch.long)

        # Fill tensors with actual content up to min(tokens.size(1), MAX_LENGTH)
        length = min(len(tokens), MAX_LENGTH)
        token_ids[:length] = torch.tensor(tokens[:length])
        attention_mask[:length] = 1
        
        # Define the start and end sequences
        start_sequence = [128006, 78191, 128007, 271]
        end_token = 128009

        # Find all start and end positions
        i = 0
        while i < length - len(start_sequence) + 1:
            if token_ids[i:i+len(start_sequence)].tolist() == start_sequence:
                start_pos = i
                # Find the end position for this start
                for j in range(start_pos + len(start_sequence), length):
                    if token_ids[j] == end_token:
                        end_pos = j + 1  # Include the end token
                        # Update the assistant mask for this occurrence
                        assistant_mask[start_pos:end_pos] = 1
                        i = end_pos  # Move i to the end of this occurrence
                        break
            i += 1

        if sampled_h is not None:
            #If there is a sampled_h I want to only keep the last set of 1s in the assistant mask (only the last assistant message e.g
            # [0,0,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0]
            # ->
            # [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0]
            indices = np.where(assistant_mask == 1)[0]
            if len(indices) > 0:
                # Get start and end of the last contiguous block
                last_end = indices[-1]
                last_start = last_end
                while last_start > 0 and assistant_mask[last_start - 1] == 1:
                    last_start -= 1

                # Zero everything except the last block
                assistant_mask[:] = 0
                assistant_mask[last_start:last_end + 1] = 1

        return token_ids, attention_mask, assistant_mask

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    async def interact_with_partner(session, conversations, request, script_args):
        """
        Separate function to handle interaction with the partner model.
        
        Returns:
            user_messages: List of user messages from the partner model
            user_tokens_list: List of token tensors from the partner model
        """

        partner_response = session.post(
            f"http://{script_args.partner_host}:{script_args.partner_port}/generate/",
            json={
                "convos": conversations,
                "prompts_2": request.prompts_2,
                "n": request.n,
                "repetition_penalty": request.repetition_penalty,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "min_p": request.min_p,
                "max_tokens": request.max_tokens,
                "guided_decoding_regex": request.guided_decoding_regex
            }
        )
        
        partner_data = partner_response.json()
        
        user_messages = []
        user_tokens_list = []
        
        # Update conversations with partner's responses
        for i, output in enumerate(partner_data["responses"]):
            user_msg = {"role": "user", "content": output}
            user_messages.append(user_msg)
            
            # Tokenize the user message
            user_tokens = tokenizer.apply_chat_template(
                [user_msg],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            user_tokens_list.append(user_tokens)
        
        return user_messages, user_tokens_list

                
    

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
        prompts_2: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None
        starting_agent: Optional[bool] = None  # Override the script argument if provided
        sampled_h: Optional[int] = None

    class GenerateResponse(BaseModel):
        conversations: list[list[dict]]
        token_ids: list[list[int]]
        attention_masks: list[list[int]]
        assistant_masks: list[list[int]]



    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions and directly tracks token IDs and masks for efficiency.
        """
        import requests
        session = requests.Session()
        
        try:
            # Check memory status before generation
            memory_stats = await memory_status()
            logger.info(f"Memory status: {memory_stats}")
                
               
            
            # Initialize conversation histories and token tracking for each prompt
            conversations = []
            all_token_ids = []
            all_attention_masks = []
            all_assistant_masks = []
            max_length = 4096
            
            # Determine which agent starts the conversation
            starting_agent = request.starting_agent if request.starting_agent is not None else script_args.starting_agent
            logger.info(f"Starting agent: {starting_agent}. True means current model starts, False means partner model starts.")
            
            # Initialize with system prompts
            for prompt in request.prompts:
                conversations.append([{"role": "system", "content": prompt}])
                
                # Tokenize the initial prompt
            
            # Configure sampling parameters
            sampling_params = SamplingParams(
                n=request.n,
                repetition_penalty=request.repetition_penalty,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                max_tokens=request.max_tokens,
                guided_decoding=request.guided_decoding_regex and 
                    GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
            )


            sampled_h = getattr(request, 'sampled_h', None)
            original_number_of_prompts = len(request.prompts)

            print("Generating with sampled_h: ", sampled_h)
            
            for turn in range(script_args.conversation_turns):

                if sampled_h is not None and turn < sampled_h:
                    conversations = [conversations[0]]
                    print(f"Turn {turn}/{script_args.conversation_turns}: Only one conversation")
                elif sampled_h is not None and turn == sampled_h:
                    base_conversation = conversations[0].copy()  # Create a copy of the conversation history
                    conversations = [base_conversation.copy() for _ in range(original_number_of_prompts)]
                    print(f"Turn {turn}/{script_args.conversation_turns}: {original_number_of_prompts} conversations")
                else:
                    print(f"Turn {turn}/{script_args.conversation_turns}: Nothing changed, still {len(conversations)} conversations")

                try:
                    # If starting_agent is False and this is the first turn, let the partner go first
                    if not starting_agent and turn == 0:
                        # Partner model goes first
                        logger.info(f"Turn {turn}: Partner model starts the conversation (starting_agent=False)")
                        user_messages, user_tokens_list = await interact_with_partner(session, conversations, request, script_args)
                        
                        # Update all conversations with user messages
                        for i in range(len(conversations)):
                            if i < len(user_messages):
                                conversations[i].append(user_messages[i])
                                

                    # Generate responses for all conversations
                    formatted_conversations = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in conversations]

                    # if lora_request:
                    #     all_outputs = llm.generate(formatted_conversations, sampling_params=sampling_params, lora_request=lora_request)
                    # else:
                    all_outputs = llm.generate(formatted_conversations, sampling_params=sampling_params)
                    
                    # Update conversations and token tracking
                    for i, output in enumerate(all_outputs):
                        # Get the text response
                        this_model_response = output.outputs[0].text
                        conversations[i].append({"role": "assistant", "content": this_model_response})
                        
               
                    
                    # Clear cache periodically
                    if turn % 5 == 0:  # Clear cache every 5 turns
                        await clear_cache()

                    if turn == script_args.conversation_turns - 1 and not starting_agent:
                        pass
                    else:
                        logger.info(f"Turn {turn}: Getting response from partner model")
                        user_messages, user_tokens_list = await interact_with_partner(
                            session, conversations, request, script_args)
                        
                        # Update all conversations with user messages
                        for i in range(len(conversations)):
                            if i < len(user_messages):
                                conversations[i].append(user_messages[i])
                       
                except Exception as e:
                    logger.error(f"Error during generation at turn {turn}: {e}")
                    # Attempt to reconnect and retry once
           
            
            for convo in conversations:
                token_ids, attention_mask, assistant_mask = tokenize_messages(convo, tokenizer, sampled_h, starting_agent)
                all_token_ids.append(token_ids)
                all_attention_masks.append(attention_mask)
                all_assistant_masks.append(assistant_mask)


            #Show decoded assistant responses
            for i, output in enumerate(conversations):
                # print(f"Conversation {i+1}:")
                # for j, msg in enumerate(output):
                #     print(f"Turn {j+1} - {msg['role']}: {msg['content']}")

                token = all_token_ids[i]
                attention_mask = all_attention_masks[i]
                assistant_mask = all_assistant_masks[i]

                applied_mask = assistant_mask * token

                # Decode the token ids
                decoded_tokens = tokenizer.decode(applied_mask)
                # print(f"Decoded tokens: {decoded_tokens}")

                

            # Final cleanup
            await clear_cache()
            
            # Return both conversations and tokenized data
            return {
                "conversations": conversations,
                "token_ids": [t.tolist() for t in all_token_ids],
                "attention_masks": [m.tolist() for m in all_attention_masks],
                "assistant_masks": [m.tolist() for m in all_assistant_masks]
            }
            
        except Exception as e:
            logger.error(f"Error in generate endpoint: {e}")
            # Attempt to reconnect before raising the error
            try:
                await reconnect()
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect after error: {reconnect_error}")
            raise



       

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

        if not script_args.enable_weight_sync:
            return {"message": "Weight sync is disabled, skipping communicator initialization"}
        
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
    async def update_named_param(request: UpdateWeightsRequest, background_tasks: BackgroundTasks):
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
        if not script_args.enable_weight_sync:
            return {"message": "Weight sync is disabled, skipping weight update"}
        
        global update_sequence_counter, update_in_progress
        update_sequence_counter += 1
        sequence_num = update_sequence_counter
        
        # Parse dtype
        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        
        background_tasks.add_task(llm.collective_rpc, "update_named_param", args=(request.name, dtype, request.shape, sequence_num))

        
        return {"message": f"Request #{sequence_num} queued for param {request.name}"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        if not script_args.enable_prefix_caching:
            return {"message": "Prefix caching is disabled, skipping prefix cache reset"}
        
        success = llm.llm_engine.reset_prefix_cache()
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        """
        Closes the weight update group and cleans up associated resources.
        """
        if not script_args.enable_weight_sync:
            return {"message": "Weight sync is disabled, skipping communicator closing"}
        
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
