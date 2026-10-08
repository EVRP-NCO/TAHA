import abc
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union
import math
import torch.nn as nn
import torch
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)
#torch.set_printoptions(profile="full")


class ConstructiveEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for the encoder of constructive models"""

    @abc.abstractmethod
    def forward(self, td: TensorDict) -> Tuple[Any, Tensor]:
        """Forward pass for the encoder

        Args:
            td: TensorDict containing the input data

        Returns:
            Tuple containing:
              - latent representation (any type)
              - initial embeddings (from feature space to embedding space)
        """
        raise NotImplementedError("Implement me in subclass!")


class ConstructiveDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base decoder model for constructive models. The decoder is responsible for generating the logits for the action"""

    @abc.abstractmethod
    def forward(
        self, td: TensorDict, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Obtain logits for current action to the next ones

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder. Can be any type
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the logits and the action mask
        """
        raise NotImplementedError("Implement me in subclass!")

    def pre_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[TensorDict, Any, RL4COEnvBase]:
        """By default, we don't need to do anything here.

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder
            env: Environment for decoding
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the updated hidden state, TensorDict, and environment
        """
        return td, env, hidden


class NoEncoder(ConstructiveEncoder):
    """Default encoder decoder-only models, i.e. autoregressive models that re-encode all the state at each decoding step."""

    def forward(self, td: TensorDict) -> Tuple[Tensor, Tensor]:
        """Return Nones for the hidden state and initial embeddings"""
        return None, None


class ConstructivePolicy(nn.Module):
    """
    Base class for constructive policies. Constructive policies take as input and instance and output a solution (sequence of actions).
    "Constructive" means that a solution is created from scratch by the model.

    The structure follows roughly the following steps:
        1. Create a hidden state from the encoder
        2. Initialize decoding strategy (such as greedy, sampling, etc.)
        3. (Optional) Apply adapter to enhance embeddings and add biases
        4. Decode the action given the hidden state and the environment state at the current step
        5. Update the environment state with the action. Repeat 3-5 until all sequences are done
        6. Obtain log likelihood, rewards etc.

    Architecture:
        The policy supports three independent components:
        - Encoder: Transforms input to hidden representations
        - Adapter (optional): Middle layer that enhances embeddings and applies constraint-based biases
        - Decoder: Generates action logits from hidden state
        
        Data flow: Encoder -> Adapter (optional) -> Decoder

    Note that an encoder is not strictly needed (see :class:`NoEncoder`).). A decoder however is always needed either in the form of a
    network or a function. An adapter is optional and provides constraint-aware enhancements.

    Note:
        There are major differences between this decoding and most RL problems. The most important one is
        that reward may not defined for partial solutions, hence we have to wait for the environment to reach a terminal
        state before we can compute the reward with `env.get_reward()`.

    Warning:
        We suppose environments in the `done` state are still available for sampling. This is because in NCO we need to
        wait for all the environments to reach a terminal state before we can stop the decoding process. This is in
        contrast with the TorchRL framework (at the moment) where the `env.rollout` function automatically resets.
        You may follow tighter integration with TorchRL here: https://github.com/ai4co/rl4co/issues/72.

    Args:
        encoder: Encoder to use
        decoder: Decoder to use
        adapter: Optional adapter module to sit between encoder and decoder
        env_name: Environment name to solve (used for automatically instantiating networks)
        temperature: Temperature for the softmax during decoding
        tanh_clipping: Clipping value for the tanh activation (see Bello et al. 2016) during decoding
        mask_logits: Whether to mask the logits or not during decoding
        train_decode_type: Decoding strategy for training
        val_decode_type: Decoding strategy for validation
        test_decode_type: Decoding strategy for testing
    """

    def __init__(
        self,
        encoder: Union[ConstructiveEncoder, Callable],
        decoder: Union[ConstructiveDecoder, Callable],
        env_name: str = "tsp",
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        adapter: Optional[nn.Module] = None,
        **unused_kw,
    ):
        super(ConstructivePolicy, self).__init__()

        # if len(unused_kw) > 0:
        #     log.error(f"Found {len(unused_kw)} unused kwargs: {unused_kw}")

        self.env_name = env_name

        # Encoder, Adapter, and Decoder - three independent components
        if encoder is None:
            log.warning("`None` was provided as encoder. Using `NoEncoder`.")
            encoder = NoEncoder()
        self.encoder = encoder
        self.adapter = adapter  # Adapter is now an independent middle layer
        self.decoder = decoder

        # Decoding strategies
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        # return_tw=True,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        # Enable attention collection during testing/evaluation
        if phase in ["test", "val"] and hasattr(self.encoder, 'forward'):
            try:
                # Try to get attention weights during testing
                encoder_output = self.encoder(td, return_attention=True)
                if len(encoder_output) == 3:
                    hidden, init_embeds, attention_weights = encoder_output
                   # log.info(f"成功收集到注意力权重，层数: {len(attention_weights) if attention_weights else 0}")
                else:
                    hidden, init_embeds = encoder_output
            except Exception as e:
                #log.warning(f"无法收集注意力权重: {e}，使用标准forward")
                hidden, init_embeds = self.encoder(td)
        else:
            hidden, init_embeds = self.encoder(td)
        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        # IMPORTANT: Call decoder/adapter hook BEFORE decode_strategy to avoid td expansion issues
        if self.adapter is not None:
            # Adapter's pre_decoder_hook will set env and forward to decoder
            # Pass num_starts=0 here since td is not yet expanded
            td, env, hidden = self.adapter.pre_decoder_hook(td, env, hidden, num_starts=0)
        else:
            td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts=0)
        
        # NOW expand td for multi-start decoding
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Initialize log likelihood and entropy
        log_likelihood = torch.zeros(
            *td.batch_size, 1, device=td.device, dtype=torch.float32
        )
        entropy = torch.zeros(*td.batch_size, device=td.device, dtype=torch.float32)

        # Initialize accumulated auxiliary loss (e.g., for PIP-D)
        accumulated_aux_loss = torch.tensor(0.0, device=td.device)

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            if self.__module__ == "rl4co.models.zoo.energy.policy":
                td.set("current_node_energy", td["current_node_energy"]+td["demand"][0][0][step])
            elif self.__module__ == "rl4co.models.zoo.energy.policy":
                self.decoder.now = self.route_feature_mul(
                    decode_strategy.actions, td.clone(), hidden.node_embeddings.clone()
                )
                
            # Data flow: Encoder -> Adapter -> Decoder
            # If adapter is available, it calls decoder internally and enhances the output
            # Otherwise, call decoder directly
            if self.adapter is not None:
                # Pass tour actions to adapter
                if hasattr(self.adapter, 'tour_actions'):
                    self.adapter.tour_actions = decode_strategy.actions
                # Adapter internally calls decoder and adds biases/enhancements
                decoder_output = self.adapter(td, hidden, num_starts)
            else:
                # Direct decoder call without adapter
                decoder_output = self.decoder(td, hidden, num_starts)

            # Handle dict or tuple return from decoder
            if isinstance(decoder_output, dict):
                logits = decoder_output["logits"]
                mask = decoder_output["mask"]
                # Get auxiliary loss if present and add to accumulator
                step_aux_loss = decoder_output.get("pip_aux_loss", torch.tensor(0.0, device=logits.device))
                accumulated_aux_loss = accumulated_aux_loss + step_aux_loss
            elif isinstance(decoder_output, tuple) and len(decoder_output) == 2:
                # If decoder returns a tuple (legacy or non-PIP case), assume no aux loss
                logits, mask = decoder_output
                step_aux_loss = torch.tensor(0.0, device=logits.device) # Ensure aux loss is zero
                # Note: accumulated_aux_loss does not increase here
            else:
                raise ValueError(f"Decoder output must be a dict or a tuple of (logits, mask), got {type(decoder_output)}")

            if mask is None:
                mask = td.get("action_mask", None)
            if mask is None:
                raise ValueError("Energy adapter requires a valid action mask")

            # 适配器已在解码器包装器内部处理能量/约束偏置，无需此处再叠加

            # Decode action based on decoding strategy
            # The decode_strategy.step updates the td in place and stores logp/actions internally
            td = decode_strategy.step(logits, mask, td, action=actions[..., step] if actions is not None else None)

            # Update environment state using the action selected by the strategy
            td = env.step(td)["next"]

            # Check termination condition
            if step > max_steps:
                log.error(
                    f"Max steps {max_steps} reached. This may indicate an infinite loop. "
                    f"Check the environment step function or set a higher max_steps."
                )
                td["done"] = torch.ones_like(td["done"])

            step += 1

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "path": td["path"],
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }
        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        # Aggregate results
        outdict["pip_aux_loss"] = accumulated_aux_loss / step

        return outdict

    def route_feature(self, tour, td, embeddings):
        """
        路径特征提取函数 - 为您的UnifiedVRP环境定制
        
        这个函数的作用是：
        1. 从当前路径中提取节点嵌入特征
        2. 结合车辆状态信息（容量、能量、时间等）
        3. 生成一个综合的路径上下文表示，用于指导后续决策
        
        参数说明：
        - tour: 当前已访问的路径节点序列
        - td: TensorDict，包含环境状态信息
        - embeddings: 节点嵌入表示
        
        返回：
        - context: 路径上下文特征，用于指导下一步动作选择
        """
        
        batch_size, _, embed_dim = embeddings.size()
        
        # 提取约束标志
        constraint_energy = td.get("constraint_energy", torch.ones_like(td["current_node"], dtype=torch.bool))
        constraint_time_windows = td.get("constraint_time_windows", torch.ones_like(td["current_node"], dtype=torch.bool))
        constraint_backhaul = td.get("constraint_backhaul", torch.zeros_like(td["current_node"], dtype=torch.bool))
        
        if len(tour) != 0:
            # 从路径中提取节点嵌入
            batch_tour = torch.stack(tour, -1)  # [batch_size, tour_len]
            tour_con = torch.gather(
                embeddings,  # [batch_size, graph_size, embed_dim]
                1,
                (batch_tour.clone())[..., None]
                .contiguous()  # [batch_size, tour_len]
                .expand(batch_size, batch_tour.size(-1), embed_dim),
            ).view(
                batch_size, batch_tour.size(-1), embed_dim
            )  # [batch_size, tour_len, embed_dim]
            
            # 使用最大池化提取路径中最显著的特征
            mean_tour = torch.max(tour_con, dim=1)[0]
            
            # 构建车辆状态上下文 - 根据约束动态调整
            veh_context_features = []
            
            # 容量特征 - 总是包含
            veh_context_features.append(td["used_capacity"])
            
            # 能量相关特征 - 仅在能量约束启用时包含
            if constraint_energy.any():
                veh_context_features.append(td["used_length"] / 3.0)
            else:
                # 能量约束禁用时，用零填充保持维度一致
                veh_context_features.append(torch.zeros_like(td["used_length"]))
            
            # 时间相关特征 - 仅在时间窗口约束启用时包含
            if constraint_time_windows.any():
                veh_context_features.append(td["current_time"] / 9.0)
            else:
                # 时间窗口约束禁用时，用零填充
                veh_context_features.append(torch.zeros_like(td["current_time"]))
            
            # 回程约束特征 - 仅在回程约束启用时包含
            if constraint_backhaul.any():
                # 这里可以添加回程相关的特征，比如剩余回程需求等
                backhaul_feature = torch.zeros_like(td["used_capacity"])
                veh_context_features.append(backhaul_feature)
            
            veh_context = torch.cat(veh_context_features, -1)
            
        else:
            # 空路径时的初始化
            mean_tour = torch.zeros([batch_size, embed_dim]).float().cuda()
            
            # 构建初始车辆状态上下文
            veh_context_features = [
                td["used_capacity"],
                torch.zeros_like(td["used_length"]),  # 初始能量使用为0
                torch.zeros_like(td["current_time"])   # 初始时间为0
            ]
            
            if constraint_backhaul.any():
                veh_context_features.append(torch.zeros_like(td["used_capacity"]))
                
            veh_context = torch.cat(veh_context_features, -1)
        
        # 通过解码器的投影层处理车辆状态
        veh_context1 = self.decoder.tour(veh_context)
        
        # 合并路径特征和车辆状态特征
        cat_context = torch.cat((veh_context1, mean_tour), -1).view(
            batch_size, embed_dim * 2
        )
        
        # 最终投影到上下文空间
        context = self.decoder.FF_tour(cat_context)

        return context
    
    # 已删除 route_feature_clh

    def route_feature_mul(self, tour, td, embeddings):

        batch_size, _, embed_dim = embeddings.size()
        if len(tour) != 0:
            batch_tour = torch.stack(tour, -1).view(
                batch_size, -1
            )  # [batch_size, tour_len]
            tour_con = torch.gather(
                embeddings,  # [batch_size, graph_size, embed_dim]
                1,
                (batch_tour.clone())[..., None]
                .contiguous()  # [batch_size, tour_len]
                .expand(batch_size, batch_tour.size(-1), embed_dim),
            ).view(
                batch_size, -1, embed_dim
            )  # [batch_size, tour_len, embed_dim]
            mean_tour = torch.mean(tour_con, dim=1)
            veh_context = torch.cat(
                (
                    td["used_capacity"],
                    td["used_length"] / 3,
                    td["current_time"] / 9,
                ),
                -1,
            )
        else:
            mean_tour = torch.zeros([batch_size, embed_dim]).float().cuda()
            veh_context = torch.cat(
                (
                    td["used_capacity"],
                    td["used_length"] / 3,
                    td["current_time"] / 9,
                ),
                -1,
            )
        veh_context1 = self.decoder.tour(veh_context)
        cat_context = torch.cat((veh_context1, mean_tour), -1)
        context = self.decoder.FF_tour(cat_context)

        return context.view(batch_size, -1, embed_dim)


class AdapterConstructivePolicy(ConstructivePolicy):
    """
    ConstructivePolicy with Adapter support for encoder-adapter-decoder architecture.
    
    This class extends ConstructivePolicy with an independent adapter layer that:
    - Sits between encoder and decoder as a separate component
    - Computes route context from trajectory
    - Applies energy/constraint biases to logits
    - Enhances embeddings with constraint information
    
    Args:
        encoder: Encoder module
        decoder: Decoder module (NOT wrapped, remains independent)
        adapter: Adapter module (independent middle layer). If None and use_adapter=True, 
                 creates default Adapter
        env_name: Environment name
        use_adapter: Whether to use adapter (default: True)
        adapter_bias_scale: Scaling factor for energy bias (default: 1.0)
        adapter_temperature: Temperature for feasibility computation (default: 1.0)
        **kwargs: Additional arguments passed to ConstructivePolicy
    """
    
    def __init__(
        self,
        encoder: Union[ConstructiveEncoder, Callable],
        decoder: Union[ConstructiveDecoder, Callable],
        adapter: Optional[nn.Module] = None,
        env_name: str = "tsp",
        use_adapter: bool = True,
        adapter_bias_scale: float = 1.0,
        adapter_temperature: float = 1.0,
        **kwargs,
    ):
        # Import here to avoid circular dependency
        from rl4co.models.zoo.am.adapter import Adapter
        
        # Create adapter if requested and not provided
        if use_adapter and adapter is None:
            adapter = Adapter(
                base_decoder=decoder,
                bias_scale=adapter_bias_scale,
                temperature=adapter_temperature
            )
        
        # Call parent constructor with independent components
        super(AdapterConstructivePolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            adapter=adapter,  # Pass adapter as independent component
            env_name=env_name,
            **kwargs
        )
        
        self.use_adapter = use_adapter
