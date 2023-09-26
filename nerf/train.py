from datetime import datetime
from typing import Optional

from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss
from torch.profiler import profile, ProfilerActivity
from torch.profiler.profiler import tensorboard_trace_handler
from .config import get_device, RUNS_DIR, DTYPE, DataFormat
from .dataset import get_datasets, DatasetType
from .model import HashNeRFSmall, HashNeRF, Render
from .encoders import HashEncoder, SphericalEncoder
from .losses import total_variation_loss, mse2psnr
from .test import test_epoch


def train(dataset_path: str,
          format: DataFormat,
          batch_size: int = 2 * 1024,
          chunk_size: int = 1024,
          num_samples: int = 64,
          epochs: int = 3,
          learning_rate: float = 1e-3,
          learning_rate_decay: float = 0.1,
          finest_resolution: int = 1024,
          entropy_loss_weight: float = 1e-10,
          tv_loss_weight: float = 1e-6,
          perturb: bool = True,
          profiler: bool = False,
          experiment_dir: str = None) -> None:
    """
    Train the model
    :param dataset_path: the path to the dataset
    :param format: the format of the dataset
    :param batch_size: the batch size
    :param chunk_size: the chunk size to process the data in parallel (decrease to avoid out of
    memory errors)
    :param num_samples: number of samples per ray
    :param epochs: number of total epochs
    :param learning_rate: the initial learning rate
    :param learning_rate_decay: the learning rate decay per epoch
    :param finest_resolution: the finest resolution of the hash map
    :param entropy_loss_weight: the weight of the entropy loss
    :param tv_loss_weight: the weight of the total variation loss
    :param perturb: if True, the samples will be perturbed (added random noise)
    :param profiler: if True, the profiler will be used
    :param experiment_dir: the directory where to store the logs and the models
    """
    assert chunk_size % num_samples == 0, "Chunk size must be a multiple of num_samples"
    dtype = DTYPE
    if experiment_dir is None:
        experiment_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = RUNS_DIR.joinpath(experiment_dir)
    writer = SummaryWriter(experiment_dir.joinpath("logs/train"))
    writer_val = SummaryWriter(experiment_dir.joinpath("logs/val"))
    data_loaders, bounding_box = get_datasets(dataset_path,
                                                 format=format,
                                                 batch_size=batch_size)
    train_dataloader = data_loaders[DatasetType.TRAIN]
    val_dataloader = data_loaders[DatasetType.VAL]

    device = get_device()
    embedder_points = HashEncoder(bounding_box, finest_resolution=finest_resolution).to(device)
    embedder_directions = SphericalEncoder().to(device)
    # nerf = NeRFSmall(embedder_points.out_dim, embedder_directions.out_dim).to(device)
    nerf = HashNeRF(embedder_points.out_dim, embedder_directions.out_dim).to(device)
    render = Render(embedder_points, embedder_directions, nerf)

    # optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.RAdam([
        {'params': nerf.parameters(), 'weight_decay': 1e-6},
        {'params': embedder_points.parameters(), 'eps': 1e-15},
    ], lr=learning_rate, betas=(0.9, 0.99))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)

    prof = None
    if profiler:
        def _trace_handler(prof):
            print("Profiling...")
            tensorboard_trace_handler(dir_name=experiment_dir, use_gzip=True)(prof)
            for order in ["self_cpu_time_total", "cpu_time_total",
                          "self_cuda_time_total", "cuda_time_total"]:
                print(order)
                output = prof.key_averages().table(sort_by=order, row_limit=15)
                print(output)

        prof = profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(skip_first=0, wait=2, warmup=2, active=2, repeat=1),
            on_trace_ready=_trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True)
        print("Profiler ON", torch.profiler.supported_activities())

    progress_bar = tqdm(total=epochs, leave=True, position=0)
    best_score = test_epoch(dataloader=val_dataloader, render=render,
                            num_samples=num_samples, chunk_size=chunk_size,
                            summary_writer=writer_val, device=device, dtype=dtype,
                            current_epoch=0, global_step=0, tqdm_position=1)
    progress_bar.set_description(f"Epoch")
    if profiler:
        prof.start()
    size_dataset = len(train_dataloader.dataset)
    for e in range(epochs):
        epoch(dataloader=train_dataloader, current_epoch=e, render=render, num_samples=num_samples,
              chunk_size=chunk_size, optimizer=optimizer, lr_scheduler=lr_scheduler,
              entropy_loss_weight=entropy_loss_weight, tv_loss_weight=tv_loss_weight,
              perturb=perturb, summary_writer=writer, device=device, dtype=dtype, profiler=prof)
        progress_bar.update(1)
        lr_scheduler.step()
        score = test_epoch(dataloader=val_dataloader, render=render,
                           num_samples=num_samples, chunk_size=chunk_size,
                           summary_writer=writer_val, device=device, dtype=dtype,
                           current_epoch=e + 1, global_step=(e + 1) * size_dataset, tqdm_position=1)
        if (best_score is None) or (score < best_score):
            best_score = score
            print(f"New best score: {best_score}")
            torch.save({
                'global_step': (e + 1) * size_dataset,
                'nerf_state_dict': nerf.state_dict(),
                'embedder_points_state_dict': embedder_points.state_dict(),
                'embedder_directions_state_dict': embedder_directions.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, experiment_dir.joinpath(f"model_{e + 1}_{best_score:.4f}.pt"))
    if profiler:
        prof.stop()
    writer.close()


def epoch(dataloader: torch.utils.data.DataLoader,
          current_epoch: int,
          render: Render,
          num_samples: int,
          chunk_size: int,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
          entropy_loss_weight: float,
          tv_loss_weight: float,
          perturb: bool,
          summary_writer: SummaryWriter,
          device: torch.device,
          dtype: torch.dtype,
          profiler: Optional[profile] = None) -> None:
    """
    Train for one epoch
    :param dataloader: the dataloader
    :param current_epoch: the current epoch
    :param render: the render of the scene
    :param num_samples: number of samples per ray
    :param chunk_size: chunk size to process the data in parallel (decrease to avoid out of
    memory errors)
    :param optimizer: the optimizer
    :param lr_scheduler: the learning rate scheduler
    :param entropy_loss_weight: the weight of the entropy loss
    :param tv_loss_weight:
    :param perturb: if True, the samples will be perturbed (added random noise)
    :param summary_writer: the summary writter to store the logs
    :param device: the device to use
    :param dtype: the data type to use
    :param profiler: the profiler to use, if any
    """
    # model = model.train()
    near = dataloader.dataset.near
    far = dataloader.dataset.far

    total_steps = len(dataloader)
    total_examples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    progress_bar = tqdm(total=total_steps, leave=False, position=1)
    for batch, (rays_origin, rays_direction, rgb) in enumerate(dataloader):

        rays_origin = rays_origin.to(dtype).to(device)
        rays_direction = rays_direction.to(dtype).to(device)
        rgb = rgb.to(dtype).to(device)

        predicted_rgb, depth, accumulated, entropy = \
            render.render_rays(rays_origin, rays_direction, near, far,
                               num_samples, chunk_size, perturb)

        optimizer.zero_grad()

        loss_mse = mse_loss(rgb, predicted_rgb)
        loss_entropy = entropy.sum()
        if tv_loss_weight > 0.0:
            hash_embedder = render.embedder_points
            loss_embedding = sum(total_variation_loss(hash_embedder.embeddings,
                                                      hash_embedder.base_resolution,
                                                      hash_embedder.finest_resolution,
                                                      i, hash_embedder.log2_hashmap_size,
                                                      hash_embedder.n_levels)
                                 for i in range(hash_embedder.n_levels))
        else:
            loss_embedding = torch.tensor(0.0, dtype=dtype, device=device)

        loss = loss_mse + entropy_loss_weight * loss_entropy + tv_loss_weight * loss_embedding

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Logging
        psnr = mse2psnr(loss_mse.item())
        step = rgb.shape[0] + batch * batch_size + total_examples * current_epoch
        summary_writer.add_scalar('loss/total', loss.item(), global_step=step)
        summary_writer.add_scalar('loss/mse', loss_mse.item(), global_step=step)
        summary_writer.add_scalar('loss/entropy', loss_entropy.item(), global_step=step)
        summary_writer.add_scalar('loss/embedding', loss_embedding.item(), global_step=step)
        summary_writer.add_scalar('metric/psnr', psnr, global_step=step)

        # update progress bar
        progress_bar.update(1)
        logs = {"loss": loss.detach().item(),
                "mse": loss_mse.detach().item(),
                "entropy": loss_entropy.detach().item(),
                "embedding": loss_embedding.detach().item(),
                "psnr": psnr,
                "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        if profiler is not None:
            profiler.step()

