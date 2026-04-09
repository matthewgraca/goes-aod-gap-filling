import os
import torch.nn
import torchmetrics
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import utils
from model import Unet3p
from datautils import DataCollection, DataSource
from constants import *


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 train_loss: torchmetrics.Metric,
                 val_loss: torchmetrics.Metric,
                 save_every: int,
                 snapshot_path: str) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.train_loss = train_loss.to(self.local_rank)
        self.val_loss = val_loss.to(self.local_rank)
        self.train_losses = []
        self.val_losses = []
        self.save_every = save_every
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self.__load_snapshot(snapshot_path)
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def __load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.train_losses = snapshot["TRAIN_LOSSES"]
        self.val_losses = snapshot["VAL_LOSSES"]
        print(f"Resuming training from snapshot at epoch {self.epochs_run}")

    def __run_batch(self, x, y, weights, train=True):
        yhat = self.model(x)
        y[y == FILL_VALUE] = yhat[y == FILL_VALUE]
        if train:
            self.optimizer.zero_grad()
            loss = self.train_loss(yhat=yhat, y=y, weights=weights, preds=yhat, target=y)
            loss.backward()
            self.optimizer.step()
        self.val_loss.update(yhat=yhat, y=y, weights=weights, preds=yhat, target=y)

    def __run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batch size: {b_sz} Steps: {len(self.train_data)}")
        # training stage
        self.model.train()
        for step, (x, y, weights, _, _, _) in enumerate(self.train_data):
            self.__run_batch(x.float().to(self.local_rank),
                             y.float().to(self.local_rank),
                             weights.float().to(self.local_rank))

        # validation stage
        self.model.eval()
        for step, (x, y, weights, _, _, _) in enumerate(self.val_data):
            self.__run_batch(x.float().to(self.local_rank),
                             y.float().to(self.local_rank),
                             weights.float().to(self.local_rank),
                             train=False)

        # save losses and update training curve plot
        if self.global_rank == 0:
            self.train_losses.append(self.train_loss.compute())
            self.val_losses.append(self.val_loss.compute())
            self.train_loss.reset()
            self.val_loss.reset()
            plt.figure(figsize=(9, 6))
            epochs = np.arange(1, self.epochs_run + 1)
            plt.plot(epochs, self.train_losses, label="train")
            plt.plot(epochs, self.val_losses, label="val")
            plt.title("AOD Gap-Filler Training Curve", fontsize=20)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.savefig("training_curve.png")

    def __save_checkpoint(self, epoch):
        snapshot = {"MODEL_STATE": self.model.module.state_dict(),
                    "OPTIMIZER_STATE": self.optimizer.state_dict(),
                    "EPOCHS_RUN": epoch,
                    "TRAIN_LOSSES": self.train_losses,
                    "VAL_LOSSES": self.val_losses}
        torch.save(snapshot, "snapshot.pt")
        print(f"Epoch {epoch} | Training checkpoint saved at {epoch}.pt")

    def train(self, max_epoch: int):
        for epoch in range(self.epochs_run, max_epoch):
            self.__run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self.__save_checkpoint(epoch)


def load_train_objs():
    dataset = DataCollection(datasources=[
        DataSource(name="frp", lags=[1, 2]),
        DataSource(name="Density", is_cat=True, cat_levels=[np.nan, 1, 2, 3], lags=[1, 2]),
        DataSource(name="merra_aod", log_transform=True, lags=[0, 1, 2]),
        DataSource(name="aod_avg", log_transform=True, lags=[0, 1, 2]),
        DataSource(name="pressure_msl", lags=[0, 1, 2]),
        DataSource(name="pressure_surf", lags=[0, 1, 2]),
        DataSource(name="temp_surf", lags=[0, 1, 2]),
        DataSource(name="temp_2m", lags=[0, 1, 2]),
        DataSource(name="dwt_2m", lags=[0, 1, 2]),
        DataSource(name="orography", lags=[0, 1, 2]),
        DataSource(name="rh_2m", lags=[0, 1, 2]),
        DataSource(name="u_10m", lags=[0, 1, 2]),
        DataSource(name="v_10m", lags=[0, 1, 2]),
        DataSource(name="land_mask", lags=[0, 1, 2]),
        DataSource(name="vegetation", lags=[0, 1, 2]),
        DataSource(name="pblh", lags=[0, 1, 2]),
        DataSource(name="Aerosol_Type_Land_Ocean", is_cat=True, cat_levels=[np.nan, 0, 1, 2, 3, 4, 5, 6, 7],
                   lags=[1, 2]),
        DataSource(name="Algorithm_Flag_Ocean", is_cat=True, cat_levels=[np.nan, 0, 1, 2], lags=[1, 2]),
        DataSource(name="Algorithm_Flag_Land", is_cat=True, cat_levels=[np.nan, 0, 1, 2], lags=[1, 2])
    ],
        daily_path=DAILY_DIR,
        stack_path=STACK_DIR,
        sample_path=SAMPLE_DIR,
        mins_path="mins.pkl",
        maxs_path="maxs.pkl",
        hist_path="hist.pkl"
    )
    dataset.set_bin_edges(EDGES)
    dataset.scan_samples()
    train_size = int(len(dataset) * DATASET_SPLIT)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    model = Unet3p(IN_CHANNELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    training_loss = ALPHA * utils.WeightedMSELoss() \
                    + BETA * (1 - MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0))
    validation_loss = ALPHA * utils.WeightedMSELoss() \
                      + BETA * (1 - MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0))

    return train_set, val_set, model, training_loss, validation_loss, optimizer


def prepare_dataloader(dataset: Dataset):
    return DataLoader(dataset,
                      batch_size=BATCH_SIZE,
                      pin_memory=True,
                      shuffle=False,
                      sampler=DistributedSampler(dataset))


def ddp_setup():
    init_process_group(backend="nccl")


def main(total_epochs: int, save_every: int, snapshot_path: str = "snapshot.pt"):
    torch.cuda.empty_cache()
    ddp_setup()
    train_set, val_set, model, train_loss, val_loss, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_set)
    val_data = prepare_dataloader(val_set)
    t = Trainer(model, train_data, val_data, optimizer, train_loss, val_loss, save_every, snapshot_path)
    t.train(total_epochs)
    destroy_process_group()
