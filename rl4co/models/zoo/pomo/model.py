from typing import Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)
torch.set_printoptions(profile="full")

class POMO(REINFORCE):
    """POMO Model for neural combinatorial optimization based on REINFORCE
    Based on Kwon et al. (2020) http://arxiv.org/abs/2010.16011.

    Note:
        If no policy kwargs is passed, we use the Attention Model policy with the following arguments:
        Differently to the base class:
        - `num_encoder_layers=6` (instead of 3)
        - `normalization="instance"` (instead of "batch")
        - `use_graph_context=False` (instead of True)
        The latter is due to the fact that the paper does not use the graph context in the policy, which seems to be
        helpful in overfitting to the training graph size.

    Args:
        env: TorchRL Environment
        policy: Policy to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        baseline: Baseline to use for the algorithm. Note that POMO only supports shared baseline,
            so we will throw an error if anything else is passed.
        num_augment: Number of augmentations (used only for validation and test)
        augment_fn: Function to use for augmentation, defaulting to dihedral8
        first_aug_identity: Whether to include the identity augmentation in the first position
        feats: List of features to augment
        num_starts: Number of starts for multi-start. If None, use the number of available actions
        pip_decoder: Whether to use Predictive Improvement Decoder (PIP-D) 
        decision_boundary: Decision boundary for PI mask prediction
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        policy_kwargs={},
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: Union[str, callable] = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        num_starts: int = None,
        pip_decoder: bool = False,
        decision_boundary: float = 0.5,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        if policy is None:
            policy_kwargs_with_defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            policy_kwargs_with_defaults.update(policy_kwargs)
            
            # Add PIP decoder parameters to policy
            if pip_decoder:
                policy_kwargs_with_defaults["pip_decoder"] = True
                policy_kwargs_with_defaults["decision_boundary"] = decision_boundary
            
            policy = AttentionModelPolicy(
                env_name=env.name, **policy_kwargs_with_defaults
            )

        assert baseline in ["shared", "best_shared", "topk_shared", "exponential"], "POMO only supports shared, best_shared, topk_shared, or exponential baseline"

        # Initialize with the shared baseline
        super(POMO, self).__init__(env, policy, baseline, **kwargs)

        self.num_starts = num_starts
        self.num_augment = num_augment
        if self.num_augment > 1:
            self.augment = StateAugmentation(
                num_augment=self.num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
        else:
            self.augment = None

        # Add `_multistart` to decode type for train, val and test in policy
        for phase in ["train", "val", "test"]:
            self.set_decode_type_multistart(phase)
        
        # PIP decoder settings
        self.pip_decoder = pip_decoder
        self.decision_boundary = decision_boundary
        
        # 初始化最佳 PIP-D 相关状态
        # self.best_pip_state = None
        # self.current_best_val_reward = -float('inf') # 或者根据实际奖励范围设定初始值
        self.pip_loss_weight = 0.3 # 辅助损失的权重，可以设为超参数

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start

        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
            # 训练时使用当前的 PIP-D 参数
            if self.pip_decoder and hasattr(self.policy.decoder, 'pointer_pip'):
                pass # 默认使用模型当前的参数
        elif phase == "val" or phase == "test":
            #  # 验证和测试时，如果有保存的最佳参数，则加载最佳参数
            #  if self.pip_decoder and self.best_pip_state is not None and hasattr(self.policy.decoder, 'pointer_pip'):
            #      log.info("加载最佳 PIP-D 参数进行评估...")
            #      self.policy.decoder.pointer_pip.load_state_dict(self.best_pip_state['pointer_pip'])
            #      self.policy.decoder.project_node_embeddings_pip.load_state_dict(self.best_pip_state['project_node_embeddings_pip'])

             if n_aug > 1:
                 td = self.augment(td)
        else: # 其他阶段（例如 'predict'）
             if n_aug > 1:
                 td = self.augment(td)

        # Evaluate policy
        out = self.policy(
            td, self.env, phase=phase, num_starts=n_start, return_actions=True
        )

        # 从策略输出中获取辅助损失
        pip_aux_loss = out.get("pip_aux_loss", torch.tensor(0.0, device=out["reward"].device))

        # Make sure path exists in the output dictionary
        if "path" not in out:
            # Create path from actions if it doesn't exist
            if "actions" in out:
                out["path"] = out["actions"]
            else:
                # Fallback to set path same as reward shape
                out["path"] = out["reward"].clone()
                
        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, n_start))
        path = unbatchify(out["path"], (n_aug, n_start))
        
        # Training phase
        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, n_start))
            # 将辅助损失传递给 calculate_loss
            self.calculate_loss(td, batch, out, reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward,"max_path":gather_by_index(path,max_idxs,dim=max_idxs.dim())})
            # 训练后恢复模型的 PIP-D 参数（如果之前加载了最佳参数）
            if (phase == "val" or phase == "test") and self.pip_decoder and self.best_pip_state is not None:
                 # 这一步通常不需要，因为训练是在另一个步骤进行的，
                 # 但如果shared_step可能被用于训练后的评估，则可能需要恢复
                 pass # 通常在 on_validation_epoch_end 中处理参数恢复

        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward,"max_path":gather_by_index(path,max_idxs,dim=max_idxs.dim())})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (n_aug, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions
                   
            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                path_=gather_by_index(path,max_idxs,dim=max_idxs.dim()) if n_start>1 else path
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward,"max_aug_path":gather_by_index(path_,max_idxs,dim=max_idxs.dim())})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})
                    zero_index = torch.nonzero(out["best_aug_actions"])[:, -1].max() + 2 # 可能出问题
                    solution= out["best_aug_actions"][:, :zero_index]
                    log.info(solution)
            else:
                log.info(actions) # actions 可能很大



        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        
 
        return {"loss": out.get("loss", None), **metrics}

