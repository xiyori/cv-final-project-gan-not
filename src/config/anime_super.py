import math

from . import anime
from .anime import DataConfig, ModelConfig, dataclass


@dataclass
class TrainConfig(anime.TrainConfig):
    num_epochs = 200

    dis_loss_coef = 0.
    min_dis_loss = math.inf

    dis_lr = 0.

    gen_grad_clip_threshold = 1.

    @property
    def run_name(self) -> str:
        return f"anime_super_mae{self.l1_coef}" \
               f"_edge{self.edge_coef}_vgg{self.vgg_coef}" \
               f"_genlr{self.gen_lr}_batch{self.train_batch}"
