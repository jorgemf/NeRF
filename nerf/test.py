import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss
import torchvision
from .dataset import get_datasets, DatasetType
from .config import get_device, RUNS_DIR, DTYPE, DataFormat
from .losses import total_variation_loss, mse2psnr
from .model import Render


def test(dataset_path: str,
         format: DataFormat,
         experiment_dir: str,
         chunk_size: int = 30*1024,
         num_samples: int = 64,
         profiler: bool = False):
    assert chunk_size % num_samples == 0, "Chunk size must be a multiple of num_samples"
    dtype = DTYPE
    experiment_dir = RUNS_DIR.joinpath(experiment_dir)
    writer = SummaryWriter(experiment_dir.joinpath("logs/test"))
    dataloaders, bounding_box = get_datasets(dataset_path, format=format, batch_size=1)
    test_dataloader = dataloaders[DatasetType.TEST]
    model =  None # TODO load...
    test_epoch(test_dataloader, model, num_samples, chunk_size, writer, get_device(), dtype, 0, 0,
               profiler=profiler)



def test_epoch(dataloader: torch.utils.data.DataLoader,
               render: Render,
               num_samples: int,
               chunk_size: int,
               summary_writer: SummaryWriter,
               device: torch.device,
               dtype: torch.dtype,
               current_epoch: int,
               global_step: int,
               profiler=None,
               tqdm_position: int = 0) -> float:
    """
    Test an epoch of the model
    :param dataloader: the dataloader to test with
    :param render: the render of the scene
    :param num_samples: number of samples per ray
    :param chunk_size: chunk size to process the data in parallel (decrease to avoid out of
    memory errors)
    :param summary_writer: the summary writter to store the logs
    :param device: the device to use
    :param dtype: the data type to use
    :param current_epoch: the current epoch
    :param global_step: the global step
    :param profiler: the profiler to use, if any
    :param tqdm_position: the position of the tqdm progress bar
    :return: the mean mse of the epoch
    """
    if global_step == 0:
        return 10000
    # model = model.eval()
    near = dataloader.dataset.near
    far = dataloader.dataset.far

    total_images = len(dataloader)
    progress_bar = tqdm(total=total_images, leave=False, position=tqdm_position)
    progress_bar.set_description(f"Image")
    mses = []
    entropies = []
    embeddings = []
    images = []
    depths = []
    accumulateds = []
    with torch.no_grad():
        for batch, (rays_origen, rays_direction, rgb) in enumerate(dataloader):
            rays_origen = rays_origen.view(-1, 3).to(dtype).to(device)
            rays_direction = rays_direction.view(-1, 3).to(dtype).to(device)
            predicted_rgb, depth, accumulated, entropy = \
                render.render_rays(rays_origen, rays_direction, near, far, num_samples, chunk_size, False)
            predicted_rgb = predicted_rgb.view(*rgb.shape[1:])

            predicted_image = predicted_rgb.permute(2, 0, 1).cpu()
            images.append(predicted_image)
            depth = (depth - near) / (far - near) # normalize depth
            depth_image = depth.view(*rgb.shape[1:3]).repeat(3, 1, 1).cpu()
            depths.append(depth_image)
            acumulated_image = accumulated.view(*rgb.shape[1:3]).repeat(3, 1, 1).cpu()
            accumulateds.append(acumulated_image)

            rgb = rgb.view(predicted_rgb.shape).to(dtype).to(device)
            loss_mse = mse_loss(rgb, predicted_rgb)
            loss_entropy = entropy.sum()
            hash_embedder = render.embedder_points
            loss_embedding = sum(total_variation_loss(hash_embedder.embeddings,
                                                          hash_embedder.base_resolution,
                                                          hash_embedder.finest_resolution,
                                                          i, hash_embedder.log2_hashmap_size,
                                                          hash_embedder.n_levels)
                                     for i in range(hash_embedder.n_levels))
            # update progress bar
            mses.append(loss_mse.detach().item())
            entropies.append(loss_entropy.detach().item())
            embeddings.append(loss_embedding.detach().item())
            progress_bar.update(1)
            psnr = mse2psnr(loss_mse.item())
            logs = {"mse": loss_mse.detach().item(),
                    "entropy": loss_entropy.detach().item(),
                    "embedding": loss_embedding.detach().item(),
                    "psnr": psnr}
            progress_bar.set_postfix(**logs)
            if profiler is not None:
                profiler.step()

    # Logging
    mean_mse = sum(mses)/total_images
    mean_entropy = sum(entropies)/total_images
    mean_embedding = sum(embeddings)/total_images
    psnr = mse2psnr(mean_mse)
    summary_writer.add_scalar('loss/mse', mean_mse, global_step=global_step)
    summary_writer.add_scalar('loss/entropy', mean_entropy, global_step=global_step)
    summary_writer.add_scalar('loss/embedding', mean_embedding, global_step=global_step)
    summary_writer.add_scalar('loss/psnr', psnr, global_step=global_step)
    for name, data in zip(["render", "depth", "accumulated"], [images, depths, accumulateds]):
        data_grid = torchvision.utils.make_grid(data)
        summary_writer.add_image(f"images/{name}", data_grid, global_step=global_step)
        log_dir = summary_writer.log_dir
        filename = f"{name}_{current_epoch}_{mean_mse:.3f}.png"
        torchvision.utils.save_image(data_grid, log_dir.joinpath(filename))

    return mean_mse
