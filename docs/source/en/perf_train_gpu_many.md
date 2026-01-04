<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Parallelism methods

Multi-GPU setups are effective for accelerating training and fitting large models in memory that otherwise wouldn't fit on a single GPU. It relies on parallelizing the workload across GPUs. There are several types of parallelism such as data parallelism, tensor parallelism, pipeline parallelism, and model parallelism. Each type of parallelism splits the workload differently, whether it's the data or the model.

This guide will discuss the various parallelism methods, combining them, and choosing an appropriate strategy for your setup. For more details about distributed training, refer to the [Accelerate](https://hf.co/docs/accelerate/index) documentation.

For a comprehensive guide on scaling large language models, check out the [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook), which provides detailed strategies and best practices for training at scale.

## Scalability strategy

Use the [Model Memory Calculator](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) to calculate how much memory a model requires. Then refer to the table below to select a strategy based on your setup.

| setup | scenario | strategy |
|---|---|---|
| single node/multi-GPU | fits on single GPU | DistributedDataParallel or ZeRO |
|  | doesn't fit on single GPU | PipelineParallel, ZeRO or TensorParallel |
|  | largest model layer doesn't fit | TensorParallel or ZeRO |
| multi-node/multi-GPU | fast inter-node connectivity (NVLink or NVSwitch) | ZeRO or 3D parallelism (PipelineParallel, TensorParallel, DataParallel) |
|  | slow inter-node connectivity | ZeRO or 3D parallelism (PipelineParallel, TensorParallel, DataParallel) |

## Data parallelism

Data parallelism evenly distributes data across multiple GPUs. Each GPU holds a copy of the model and concurrently processes their portion of the data. At the end, the results from each GPU are synchronized and combined.

Data parallelism significantly reduces training time by processing data in parallel, and it is scalable to the number of GPUs available. However, synchronizing results from each GPU can add overhead.

There are two types of data parallelism, DataParallel (DP) and DistributedDataParallel (DDP).

### DataParallel

[DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) supports distributed training on a *single machine* with multiple GPUs.

1. The default GPU, `GPU 0`, reads a batch of data and sends a mini batch of it to the other GPUs.
2. An up-to-date model is replicated from `GPU 0` to the other GPUs.
3. A `forward` pass is performed on each GPU and their outputs are sent to `GPU 0` to compute the loss.
4. The loss is distributed from `GPU 0` to the other GPUs for the `backward` pass.
5. The gradients from each GPU are sent back to `GPU 0` and averaged.

### DistributedDataParallel

[DistributedDataParallel](https://pytorch.org/docs/main/notes/ddp.html) supports distributed training across *multiple machines* with multiple GPUs.

1. The main process replicates the model from the default GPU, `GPU 0`, to each GPU.
2. Each GPU directly processes a mini batch of data.
3. The local gradients are averaged across all GPUs during the `backward` pass.

DDP is recommended because it reduces communication overhead between GPUs, efficiently utilizes each GPU, and scales to more than one machine.

### ZeRO data parallelism

[Zero Redundancy Optimizer](https://www.deepspeed.ai/tutorials/zero/) is a more memory efficient type of data parallelism. It significantly improves memory efficiency by partitioning parameters, gradients, and optimizer states across data parallel processes to reduce memory usage. There are three ZeRO stages:

- Stage 1 partitions the optimizer states
- Stage 2 partitions the optimizer and gradient states
- Stage 3 partitions the optimizer, gradient, and parameters

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png"/>
</div>

## Model parallelism

Model parallelism distributes a model across multiple GPUs. There are several ways to split a model, but the typical method distributes the model layers across GPUs. On the `forward` pass, the first GPU processes a batch of data and passes it to the next group of layers on the next GPU. For the `backward` pass, the data is sent backward from the final layer to the first layer.

Model parallelism is a useful strategy for training models that are too large to fit into the memory of a single GPU. However, GPU utilization is unbalanced because only one GPU is active at a time. Passing results between GPUs also adds communication overhead and it can be a bottleneck.

## Pipeline parallelism

Pipeline parallelism is conceptually very similar to model parallelism, but it's more efficient because it reduces the amount of idle GPU time. Instead of waiting for each GPU to finish processing a batch of data, pipeline parallelism creates *micro-batches* of data. As soon as one micro-batch is finished, it is passed to the next GPU. This way, each GPU can concurrently process part of the data without waiting for the other GPU to completely finish processing a mini batch of data.

Pipeline parallelism shares the same advantages as model parallelism, but it optimizes GPU utilization and reduces idle time. But pipeline parallelism can be more complex because models may need to be rewritten as a sequence of [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) modules and it also isn't possible to completely reduce idle time because the last `forward` pass must also wait for the `backward` pass to finish.

## Tensor parallelism

Tensor parallelism distributes large tensor computations across multiple GPUs. The tensors are sliced horizontally or vertically and each slice is processed by a separate GPU. Each GPU performs its calculations on its tensor slice and the results are synchronized at the end to reconstruct the final result.

Tensor parallelism is effective for training large models that don't fit into the memory of a single GPU. It is also faster and more efficient because each GPU can process its tensor slice in parallel, and it can be combined with other parallelism methods. Like other parallelism methods though, tensor parallelism adds communication overhead between GPUs.

Refer to the [Tensor parallelism](./perf_infer_gpu_multi) guide to learn how to use it for inference.

### Tensor Parallelism with Trainer (TP-only)

Tensor Parallelism (TP) can also be used during **training** with the `Trainer` API. However, because TP is often demonstrated together with other strategies (such as data parallelism, pipeline parallelism, or FSDP), it can be difficult to understand how to enable **TP-only training** in isolation.

This section provides a concise overview of how Tensor Parallelism fits into the `Trainer` workflow, focusing on the minimal configuration needed to enable TP without combining it with other parallelism strategies.

#### Key components involved

Using Tensor Parallelism with `Trainer` relies on three main components:

- **`device_mesh`**  
  Defines the set of devices that participate in tensor parallelism. For TP-only training, this is typically a 1D mesh spanning multiple GPUs on a single node.

- **`tp_plan`**  
  Specifies how individual tensors inside model layers (for example, linear projection weights) are partitioned across the tensor-parallel devices. This is passed when initializing the model.

- **`parallelism_config` (in `TrainingArguments`)**  
  Informs the `Trainer` which parallelism strategies to enable. When configured for TP-only training, this enables tensor parallelism without activating FSDP or other strategies.

In practice:
- `device_mesh` determines **which devices participate**
- `tp_plan` determines **what tensors are split**
- `parallelism_config` determines **how the Trainer applies tensor parallelism**

#### How TP-only training fits into `Trainer`

In a TP-only setup:

1. The model is initialized with a tensor parallel plan and a device mesh.
2. `TrainingArguments` are configured with a tensor-parallel-only `parallelism_config`.
3. The `Trainer` API itself remains unchanged.
4. Tensor parallel communication and synchronization are handled internally during the forward and backward passes.

This allows Tensor Parallelism to be enabled while keeping the standard `Trainer` training loop intact.

#### Minimal TP-only configuration (illustrative)

The example below illustrates the **configuration flow** required for TP-only training with `Trainer`. It is intentionally minimal and focuses on how the components are connected, rather than on full training details.

```python
# Illustrative example (simplified)

# 1. Define the device mesh for tensor parallelism
device_mesh = ...

# 2. Load the model with a tensor parallel plan
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    tp_plan=...,
    device_mesh=device_mesh,
)

# 3. Configure TrainingArguments with TP-only parallelism
training_args = TrainingArguments(
    output_dir="...",
    parallelism_config={
        "tensor_parallel": {
            "enabled": True,
        }
    },
)

# 4. Use Trainer as usual
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=...,
)

trainer.train()
```

## Hybrid parallelism

Parallelism methods can be combined to achieve even greater memory savings and more efficiently train models with billions of parameters.

### Data parallelism and pipeline parallelism

Data and pipeline parallelism distributes the data across GPUs and divides each mini batch of data into micro-batches to achieve pipeline parallelism.

Each data parallel rank treats the process as if there were only one GPU instead of two, but GPUs 0 and 1 can offload micro-batches of data to GPUs 2 and 3 and reduce idle time.

This approach optimizes parallel data processing by reducing idle GPU utilization.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero-dp-pp.png"/>
</div>

### ZeRO data parallelism, pipeline parallelism, and model parallelism (3D parallelism)

Data, pipeline and model parallelism combine to form [3D parallelism](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/) to optimize memory and compute efficiency.

Memory efficiency is achieved by splitting the model across GPUs and also dividing it into stages to create a pipeline. This allows GPUs to work in parallel on micro-batches of data, reducing the memory usage of the model, optimizer, and activations.

Compute efficiency is enabled by ZeRO data parallelism where each GPU only stores a slice of the model, optimizer, and activations. This allows higher communication bandwidth between data parallel nodes because communication can occur independently or in parallel with the other pipeline stages.

This approach is scalable to extremely large models with trillions of parameters.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png"/>
</div>
