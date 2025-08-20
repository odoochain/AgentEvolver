"""
Advantage分析器框架 - 从checkpoint进行真实推理分析
完全复用训练代码，避免mock实现
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.utils.dataset.rl_dataset import collate_fn
from typing import Tuple, List, Dict, Optional
import ray
from hydra import initialize_config_dir, compose
from verl.utils.fs import copy_to_local
from verl.utils import hf_processor, hf_tokenizer
import json
import time
from datetime import datetime

# 导入已有的组件
from beyondagent.main_ppo import create_rl_dataset, create_rl_sampler
from beyondagent.module.trainer.ba_ray_trainer import parse_reward_from_dataproto
from beyondagent.module.env_manager.env_manager import ParallelEnvManager
from beyondagent.module.trainer.ba_async_llm_server_manager import BaAsyncLLMServerManager
from beyondagent.schema.task import Task
from beyondagent.module.advantage_assignment.parallel_semantic_assignment import ParallelSemanticProcessor


class AdvantageAnalyzer:
    """优势分析器 - 从checkpoint进行完整的推理分析"""
    
    def __init__(self, checkpoint_path: str, config_path: str, analysis_config: dict = None):
        """
        Args:
            checkpoint_path: checkpoint目录路径 (包含actor子目录)
            config_path: 配置文件路径
            analysis_config: 分析配置
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.analysis_config = analysis_config or {
            'batch_size': 8,                    # 小batch避免OOM
            'num_samples': 32,                  # 总分析样本数
            'save_raw_data': True,              # 是否保存原始数据
            'enable_semantic_eval': False,      # 是否启用语义评估
            'skip_env_interaction': False,      # 是否跳过环境交互(用于调试)
        }
        
        # 核心组件
        self.config = None
        self.tokenizer = None
        self.processor = None
        self.dataset = None
        self.env_manager = None
        self.async_rollout_manager = None
        self.semantic_processor = None
        
        print(f"Initializing AdvantageAnalyzer for checkpoint: {checkpoint_path}")
    
    def setup(self):
        """设置分析环境"""
        print("Setting up analysis environment...")
        
        # 1. 加载配置和tokenizer
        self._load_config()
        self._load_tokenizer_and_processor()
        
        # 2. 创建数据集
        self._create_dataset()
        
        # 3. 初始化Ray环境
        self._setup_ray_environment()
        
        # 4. 创建推理组件
        self._create_inference_components()
        
        # 5. 创建语义评估组件(如果启用)
        if self.analysis_config['enable_semantic_eval']:
            self._create_semantic_processor()
        
        print("✓ Setup completed successfully!")
    
    def _load_config(self):
        cfg_dir = str(self.config_path.parent)
        cfg_name = self.config_path.stem

        with initialize_config_dir(version_base=None, config_dir=cfg_dir):
            extra = self.analysis_config.get("hydra_overrides", [])  # ← 接收从 shell 转发的 overrides
            self.config = compose(
                config_name=cfg_name,
                overrides=[
                    "hydra.searchpath=[pkg://verl.trainer.config]",
                    f"data.train_batch_size={self.analysis_config['batch_size']}",
                    "trainer.total_epochs=1",
                    "trainer.save_freq=999999",
                    "trainer.test_freq=999999",
                    f"semantic_advantage.enable={self.analysis_config['enable_semantic_eval']}",
                    "experience_maker.enable_summarizer=false",
                    "experience_maker.enable_context_generator=false",
                    *extra,  # ← ★ 关键：把 shell 传来的 Hydra overrides 生效
                ],
            )

        # 指向 checkpoint 的 actor
        self.config.actor_rollout_ref.model.path = str(self.checkpoint_path / "actor")
        OmegaConf.resolve(self.config)

        # 建议打印一下最终生效的数据路径，方便你确认覆盖成功
        print(f"✓ Using train_files: {self.config.data.train_files}")
        print(f"✓ Using val_files  : {self.config.data.val_files}")

    def _load_tokenizer_and_processor(self):
        """加载tokenizer和processor"""
        local_model_path = copy_to_local(
            self.config.actor_rollout_ref.model.path,
            use_shm=self.config.actor_rollout_ref.model.get('use_shm', False)
        )
        
        trust_remote_code = self.config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_model_path, use_fast=True)
        
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Tokenizer loaded from: {local_model_path}")
        print(f"✓ Vocab size: {self.tokenizer.vocab_size}")
        print(f"✓ Pad token: {self.tokenizer.pad_token}")

    def _create_dataset(self):
        """创建数据集"""
        self.dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor
        )
        
        # 限制数据集大小用于分析
        total_samples = len(self.dataset)
        analysis_samples = min(self.analysis_config['num_samples'], total_samples)
        
        print(f"✓ Dataset created: {total_samples} total samples")
        print(f"✓ Analysis will use: {analysis_samples} samples")
        print(f"✓ Dataset type: {type(self.dataset).__name__}")

    def _setup_ray_environment(self):
        """设置Ray环境"""
        if not ray.is_initialized():
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN", 
                        "VLLM_LOGGING_LEVEL": "WARN",
                        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                        "VLLM_USE_V1": "1"
                    }
                },
                num_cpus=self.config.ray_init.get('num_cpus', 8),
            )
            print("✓ Ray initialized for analysis")
        else:
            print("✓ Ray already initialized")

    def _create_inference_components(self):
        """创建推理相关组件"""
        # 创建最小化的trainer来初始化推理组件
        # 复用训练代码的初始化逻辑，但只初始化必要组件
        
        print("Creating inference components...")
        
        # 1. 创建resource pool和worker groups (简化版)
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
        
        # 简化的resource pool配置
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [self.config.trainer.n_gpus_per_node] * self.config.trainer.nnodes,
        }
        mapping = {Role.ActorRollout: global_pool_id}
        
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, 
            mapping=mapping
        )
        resource_pool_manager.create_resource_pool()
        
        # 2. 创建actor rollout worker group
        resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = ray.remote(AsyncActorRolloutRefWorker)
        
        from verl.single_controller.ray import RayClassWithInitArgs
        actor_rollout_with_init = RayClassWithInitArgs(
            cls=actor_rollout_cls,
            config=self.config.actor_rollout_ref,
            role="actor_rollout"
        )
        
        ray_worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=actor_rollout_with_init,
            device_name=self.config.trainer.device
        )
        
        # 初始化worker
        ray_worker_group.init_model()
        print("✓ Actor rollout worker group created")
        
        # 3. 创建异步rollout管理器
        self.async_rollout_manager = BaAsyncLLMServerManager(
            config=self.config,
            worker_group=ray_worker_group
        )
        print("✓ Async rollout manager created")
        
        # 4. 创建环境管理器
        self.env_manager = ParallelEnvManager(
            config=self.config,
            async_rollout_manager=self.async_rollout_manager,
            max_parallel=self.analysis_config['batch_size']  # 限制并行数
        )
        print("✓ Environment manager created")
        
        # 保存resource manager用于后续清理
        self._resource_pool_manager = resource_pool_manager

    def _create_semantic_processor(self):
        """创建语义评估处理器"""
        semantic_config = self.config.semantic_advantage
        
        self.semantic_processor = ParallelSemanticProcessor(
            max_concurrent=semantic_config.concurrent,
            model_name=semantic_config.model,
            evaluation_type=semantic_config.evaluation_type,
            api_max_retries=getattr(semantic_config, 'api_max_retries', 200)
        )
        print(f"✓ Semantic processor created: {semantic_config.model}")
        print(f"✓ Max concurrent: {semantic_config.concurrent}")
        print(f"✓ Evaluation type: {semantic_config.evaluation_type}")
    
    def analyze_advantages(self, save_dir: str = None) -> Dict:
        """运行advantage分析的主函数"""
        print("="*70)
        print("STARTING ADVANTAGE ANALYSIS")
        print("="*70)
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集所有分析数据
        all_advantages = {
            'original_grpo': [],
            'after_semantic': [],
            'after_normalization': []
        }
        all_metadata = []
        
        # 分批处理数据
        num_batches = self._calculate_num_batches()
        
        for batch_idx in range(num_batches):
            print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")
            
            # Step 1: 获取输入数据
            input_batch = self._get_input_batch(batch_idx)
            if input_batch is None:
                continue
            
            # Step 2: 运行推理获取trajectories
            trajectories = self._run_inference_batch(input_batch)
            
            # Step 3: 转换为DataProto
            inference_dataproto = self._convert_to_dataproto(trajectories)
            
            # Step 4: 计算三阶段advantages
            stage_advantages = self._compute_staged_advantages(inference_dataproto)
            
            # Step 5: 收集数据
            self._collect_batch_results(stage_advantages, trajectories, all_advantages, all_metadata)
            
            print(f"✓ Batch {batch_idx + 1} completed")
        
        # 生成分析结果
        analysis_results = self._generate_analysis_results(all_advantages, all_metadata)
        
        # 保存和可视化
        if save_dir:
            self._save_results(analysis_results, all_advantages, save_dir)
        
        return analysis_results
    
    def _calculate_num_batches(self) -> int:
        """计算需要处理的batch数量"""
        batch_size = self.analysis_config['batch_size']
        num_samples = min(self.analysis_config['num_samples'], len(self.dataset))
        return (num_samples + batch_size - 1) // batch_size
    
    def _get_input_batch(self, batch_idx: int) -> Optional[DataProto]:
        """获取指定batch的输入数据"""
        batch_size = self.analysis_config['batch_size']
        num_samples = min(self.analysis_config['num_samples'], len(self.dataset))
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        if start_idx >= num_samples:
            return None
        
        print(f"  Loading samples {start_idx} to {end_idx-1}")
        
        # 创建sample indices并获取数据
        indices = list(range(start_idx, end_idx))
        items = [self.dataset[i] for i in indices]
        
        # 使用训练时相同的collate function
        batch = collate_fn(items)
        
        print(f"  Batch created: {len(batch)} samples")
        return batch

    def _run_inference_batch(self, input_batch: DataProto) -> List:
        """运行推理获取trajectories"""
        if self.analysis_config['skip_env_interaction']:
            print("  Using simplified mode (skip env interaction)")
            return self._construct_trajectories_from_batch(input_batch)
        else:
            print("  Running real inference with environment interaction")
            return self._run_real_inference(input_batch)

    def _construct_trajectories_from_batch(self, input_batch: DataProto) -> List:
        """从输入batch构造trajectories(简化模式)"""
        from beyondagent.schema.trajectory import Trajectory
        
        trajectories = []
        
        for i in range(len(input_batch)):
            # 构造基本的trajectory结构
            if "messages" in input_batch.non_tensor_batch:
                # 如果有现成的messages，直接使用
                if isinstance(input_batch.non_tensor_batch["messages"][i], dict):
                    messages = input_batch.non_tensor_batch["messages"][i]["messages"]
                else:
                    messages = input_batch.non_tensor_batch["messages"][i]
            else:
                # 否则从raw_prompt构造
                user_content = input_batch.non_tensor_batch["raw_prompt"][i]
                messages = [{"role": "user", "content": user_content}]
                
                # 如果有reference答案，也加上
                if "reference" in input_batch.non_tensor_batch:
                    messages.append({
                        "role": "assistant", 
                        "content": input_batch.non_tensor_batch["reference"][i]
                    })
            
            trajectory = Trajectory(
                data_id=str(i),
                rollout_id="0",
                steps=messages,
                is_terminated=True  # 简化模式下认为都已终止
            )
            
            # 设置reward（如果有的话）
            if "reward_scores" in input_batch.non_tensor_batch:
                try:
                    reward_data = input_batch.non_tensor_batch["reward_scores"][i]
                    if isinstance(reward_data, dict):
                        trajectory.reward.outcome = float(reward_data.get("outcome", 1.0))
                    else:
                        trajectory.reward.outcome = float(reward_data)
                except:
                    trajectory.reward.outcome = 1.0  # 默认正向reward
            else:
                trajectory.reward.outcome = 1.0
            
            trajectory.reward.description = "Outcome 1 = success, 0 = failure."
            trajectories.append(trajectory)
        
        print(f"    Constructed {len(trajectories)} trajectories from batch")
        return trajectories

    def _run_real_inference(self, input_batch: DataProto) -> List:
        """运行真实推理(完整模式)"""
        # 启动异步rollout服务
        self.async_rollout_manager.wake_up()
        
        try:
            # 构造Task列表
            tasks = []
            for i in range(len(input_batch)):
                task = Task(
                    task_id=input_batch.non_tensor_batch["extras"][i]["task_id"],
                    query=input_batch.non_tensor_batch["raw_prompt"][i],
                    env_type=self.config.env_service.env_type
                )
                tasks.append(task)
            
            print(f"    Created {len(tasks)} tasks for inference")
            
            # 运行rollout (复用训练代码)
            trajectories = self.env_manager.rollout(
                tasks=tasks,
                mode="sample",  # 使用采样模式
                epoch=f"analysis_{int(time.time())}"
            )
            
            print(f"    Completed rollout, got {len(trajectories)} trajectories")
            
            # 统计一些基本信息
            terminated_count = sum(1 for t in trajectories if t.is_terminated)
            avg_steps = np.mean([len(t.steps) for t in trajectories])
            avg_reward = np.mean([t.reward.outcome for t in trajectories])
            
            print(f"    Terminated: {terminated_count}/{len(trajectories)}")
            print(f"    Avg steps: {avg_steps:.1f}")
            print(f"    Avg reward: {avg_reward:.3f}")
            
            return trajectories
            
        finally:
            # 关闭异步rollout服务
            self.async_rollout_manager.sleep()

    def _convert_to_dataproto(self, trajectories: List) -> DataProto:
        """转换trajectories为DataProto"""
        print("    Converting trajectories to DataProto...")
        
        # 直接复用训练代码的转换逻辑
        dataproto = self.env_manager.to_dataproto(trajectories)
        
        print(f"    DataProto created:")
        print(f"      Batch size: {len(dataproto)}")
        print(f"      Prompt length: {dataproto.batch['prompts'].shape[1]}")
        print(f"      Response length: {dataproto.batch['responses'].shape[1]}")
        
        return dataproto

    def _compute_staged_advantages(self, dataproto: DataProto) -> Dict[str, torch.Tensor]:
        """计算三阶段的advantages"""
        print("    Computing staged advantages...")
        
        # Stage 1: 准备数据和计算原始GRPO advantage
        batch = dataproto
        
        # 1.1 计算token_level_rewards
        reward_tensor = parse_reward_from_dataproto(batch, return_dict=False)
        batch.batch["token_level_rewards"] = reward_tensor
        
        # 1.2 计算response_mask
        prompt_length = batch.batch["prompts"].shape[1]
        batch.batch["response_mask"] = batch.batch["attention_mask"][:, prompt_length:]
        
        # # 1.3 Mock critic values (分析时不需要真实critic)
        # bs, response_len = batch.batch["responses"].shape
        # values = torch.zeros((bs, response_len + 1), dtype=torch.float32)
        # batch.batch["values"] = values
        
        # 1.4 计算原始GRPO advantages
        original_batch = self._clone_dataproto(batch)
        compute_advantage(
            original_batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
            config=self.config.algorithm,
        )
        original_advantages = original_batch.batch["advantages"].clone()
        
        print(f"      Original advantages computed: shape={original_advantages.shape}")
        print(f"      Original stats: mean={original_advantages[original_advantages!=0].mean():.4f}, "
              f"std={original_advantages[original_advantages!=0].std():.4f}")
        
        # Stage 2: 语义评估后的advantages (如果启用)
        semantic_batch = self._clone_dataproto(original_batch)
        if self.analysis_config['enable_semantic_eval'] and self.semantic_processor:
            print("      Applying semantic evaluation...")
            
            # 应用语义评估
            semantic_config = self.config.semantic_advantage
            response_length = semantic_batch.batch["responses"].size(1)
            
            # 选择mask类型
            mask_type = semantic_config.mask_type
            if mask_type == "response_mask":
                selected_mask = semantic_batch.batch["response_mask"]
            elif mask_type == "loss_mask":
                selected_mask = semantic_batch.batch["loss_mask"][:, -response_length:]
            else:
                selected_mask = semantic_batch.batch["response_mask"]
            
            # 运行语义评估和mask应用
            semantic_stats = self.semantic_processor.process_batch_sync(
                tokenizer=self.tokenizer,
                batch=semantic_batch,
                consistent_scale=semantic_config.consistent_scale,
                pos_unconsistent_scale=semantic_config.pos_unconsistent_scale,
                neg_unconsistent_scale=semantic_config.neg_unconsistent_scale,
                mask_tensor=selected_mask,
                save_dir=None,  # 分析时不保存详细记录
                global_step=None,
                epoch="analysis"
            )
            
            print(f"      Semantic evaluation completed: {semantic_stats['mask_stats']['good_steps']} good steps, "
                  f"{semantic_stats['mask_stats']['bad_steps']} bad steps")
        
        semantic_advantages = semantic_batch.batch["advantages"].clone()
        
        print(f"      Semantic advantages: mean={semantic_advantages[semantic_advantages!=0].mean():.4f}, "
              f"std={semantic_advantages[semantic_advantages!=0].std():.4f}")
        
        # Stage 3: 归一化后的advantages
        normalized_batch = self._clone_dataproto(semantic_batch)
        if self.config.semantic_advantage.adv_norm.enable:
            print("      Applying advantage normalization...")
            self._apply_advantage_normalization(normalized_batch)
        
        normalized_advantages = normalized_batch.batch["advantages"].clone()
        
        print(f"      Normalized advantages: mean={normalized_advantages[normalized_advantages!=0].mean():.4f}, "
              f"std={normalized_advantages[normalized_advantages!=0].std():.4f}")
        
        return {
            'original_grpo': original_advantages,
            'after_semantic': semantic_advantages,
            'after_normalization': normalized_advantages
        }
    
    def _clone_dataproto(self, dataproto: DataProto) -> DataProto:
        """深度复制DataProto"""
        return DataProto(
            batch={k: v.clone() if isinstance(v, torch.Tensor) else v 
                  for k, v in dataproto.batch.items()},
            non_tensor_batch={k: v.copy() if hasattr(v, 'copy') else v 
                            for k, v in dataproto.non_tensor_batch.items()}
        )
    
    def _apply_advantage_normalization(self, batch: DataProto):
        """应用advantage归一化"""
        config = self.config.semantic_advantage.adv_norm
        advantages = batch.batch["advantages"]
        response_mask = batch.batch["response_mask"]
        
        nonzero_mask = response_mask & (advantages != 0)
        
        if config.level == "batch":
            # Batch级别归一化
            if nonzero_mask.any():
                nonzero_advs = advantages[nonzero_mask]
                median = torch.median(nonzero_advs)
                std = nonzero_advs.std(unbiased=False).clamp_min(1e-8)
                advantages[nonzero_mask] = (advantages[nonzero_mask] - median) / std
        
        elif config.level == "group":
            # Group级别归一化
            group_size = config.group_size or self.config.actor_rollout_ref.rollout.n
            bs = advantages.shape[0]
            
            for g_start in range(0, bs, group_size):
                g_end = min(g_start + group_size, bs)
                g_mask = nonzero_mask[g_start:g_end]
                
                if g_mask.any():
                    g_advs = advantages[g_start:g_end][g_mask]
                    median = torch.median(g_advs)
                    std = g_advs.std(unbiased=False).clamp_min(1e-8)
                    advantages[g_start:g_end][g_mask] = (advantages[g_start:g_end][g_mask] - median) / std
        
        batch.batch["advantages"] = advantages

    def _collect_batch_results(self, stage_advantages: Dict, trajectories: List, 
                              all_advantages: Dict, all_metadata: List):
        """收集batch结果"""
        # 提取有效的advantage值并添加到总结果中
        for stage, advs in stage_advantages.items():
            valid_advs = advs[advs != 0].cpu().numpy().tolist()
            all_advantages[stage].extend(valid_advs)
        
        # 收集trajectory元数据
        for i, traj in enumerate(trajectories):
            metadata = {
                'trajectory_id': f"{traj.data_id}_{traj.rollout_id}",
                'num_steps': len(traj.steps),
                'is_terminated': traj.is_terminated,
                'final_reward': traj.reward.outcome,
                'response_length': len(traj.steps) if traj.steps else 0
            }
            all_metadata.append(metadata)
        
        print(f"    Collected {len(trajectories)} trajectory metadata")
        for stage in stage_advantages:
            stage_count = len(stage_advantages[stage][stage_advantages[stage] != 0])
            print(f"    {stage}: {stage_count} valid advantages")

    def _generate_analysis_results(self, all_advantages: Dict, all_metadata: List) -> Dict:
        """生成分析结果"""
        results = {
            'summary': {},
            'detailed_stats': {},
            'metadata_stats': {},
            'transformation_analysis': {},
            'timestamp': datetime.now().isoformat(),
            'config_info': {
                'checkpoint_path': str(self.checkpoint_path),
                'analysis_config': self.analysis_config,
                'num_samples_analyzed': len(all_metadata)
            }
        }
        
        # 统计每个stage的advantage分布
        for stage, advs in all_advantages.items():
            if len(advs) > 0:
                advs_array = np.array(advs)
                results['detailed_stats'][stage] = {
                    'count': len(advs),
                    'mean': float(np.mean(advs_array)),
                    'std': float(np.std(advs_array)),
                    'median': float(np.median(advs_array)),
                    'min': float(np.min(advs_array)),
                    'max': float(np.max(advs_array)),
                    'q25': float(np.percentile(advs_array, 25)),
                    'q75': float(np.percentile(advs_array, 75)),
                    'positive_ratio': float(np.mean(advs_array > 0)),
                    'negative_ratio': float(np.mean(advs_array < 0)),
                    'zero_ratio': float(np.mean(np.abs(advs_array) < 1e-8))
                }
            else:
                results['detailed_stats'][stage] = {'count': 0, 'error': 'No valid advantages'}
        
        # 计算stage之间的变化
        stages = ['original_grpo', 'after_semantic', 'after_normalization']
        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]
            
            if (current_stage in results['detailed_stats'] and 
                next_stage in results['detailed_stats'] and
                results['detailed_stats'][current_stage]['count'] > 0 and
                results['detailed_stats'][next_stage]['count'] > 0):
                
                current_stats = results['detailed_stats'][current_stage]
                next_stats = results['detailed_stats'][next_stage]
                
                transformation_key = f"{current_stage}_to_{next_stage}"
                results['transformation_analysis'][transformation_key] = {
                    'mean_change': next_stats['mean'] - current_stats['mean'],
                    'std_change': next_stats['std'] - current_stats['std'],
                    'median_change': next_stats['median'] - current_stats['median'],
                    'range_change': (next_stats['max'] - next_stats['min']) - (current_stats['max'] - current_stats['min']),
                    'positive_ratio_change': next_stats['positive_ratio'] - current_stats['positive_ratio']
                }
        
        # 元数据统计
        if all_metadata:
            final_rewards = [m['final_reward'] for m in all_metadata]
            response_lengths = [m['response_length'] for m in all_metadata]
            termination_rate = np.mean([m['is_terminated'] for m in all_metadata])
            
            results['metadata_stats'] = {
                'total_samples': len(all_metadata),
                'termination_rate': float(termination_rate),
                'avg_final_reward': float(np.mean(final_rewards)),
                'std_final_reward': float(np.std(final_rewards)),
                'avg_response_length': float(np.mean(response_lengths)),
                'std_response_length': float(np.std(response_lengths)),
                'reward_distribution': {
                    'mean': float(np.mean(final_rewards)),
                    'std': float(np.std(final_rewards)),
                    'median': float(np.median(final_rewards)),
                    'success_rate': float(np.mean(np.array(final_rewards) > 0)),
                    'min': float(np.min(final_rewards)),
                    'max': float(np.max(final_rewards))
                }
            }
        
        # 生成简要总结
        if len(all_advantages.get('original_grpo', [])) > 0:
            orig_mean = results['detailed_stats']['original_grpo']['mean']
            final_mean = results['detailed_stats']['after_normalization']['mean']
            results['summary'] = {
                'original_mean_advantage': orig_mean,
                'final_mean_advantage': final_mean,
                'overall_transformation': final_mean - orig_mean,
                'semantic_eval_enabled': self.analysis_config['enable_semantic_eval'],
                'samples_processed': len(all_metadata),
                'stages_computed': len([s for s in stages if s in results['detailed_stats']])
            }
        
        return results

    def _save_results(self, analysis_results: Dict, all_advantages: Dict, save_dir: Path):
        """保存结果和可视化"""
        print(f"Saving results to {save_dir}")
        
        # 1. 保存JSON结果
        results_file = save_dir / "advantage_analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"✓ Analysis results saved to: {results_file}")
        
        # 2. 保存原始advantage数据
        if self.analysis_config['save_raw_data']:
            raw_data_file = save_dir / "raw_advantages.json"
            # 转换为可序列化格式
            raw_data = {stage: advs for stage, advs in all_advantages.items()}
            with open(raw_data_file, 'w') as f:
                json.dump(raw_data, f, indent=2)
            print(f"✓ Raw advantage data saved to: {raw_data_file}")
        
        # 3. 创建可视化
        self._create_visualizations(all_advantages, save_dir)
        
        # 4. 生成分析报告
        self._generate_text_report(analysis_results, save_dir)
    
    def _create_visualizations(self, all_advantages: Dict, save_dir: Path):
        """创建可视化图表"""
        print("Creating visualizations...")
        
        # 提取有效数据
        stages = ['original_grpo', 'after_semantic', 'after_normalization']
        stage_data = []
        stage_labels = []
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        for stage in stages:
            if stage in all_advantages and len(all_advantages[stage]) > 0:
                stage_data.append(np.array(all_advantages[stage]))
                stage_labels.append(stage.replace('_', ' ').title())
        
        if not stage_data:
            print("No data to visualize")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Advantage Analysis - {self.checkpoint_path.name}', 
                    fontsize=16, fontweight='bold')
        
        # 上排：直方图
        for i, (data, label, color) in enumerate(zip(stage_data, stage_labels, colors)):
            if i >= 3:
                break
            ax = axes[0, i]
            ax.hist(data, bins=50, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
            ax.set_title(f'{label}\n(n={len(data)})', fontweight='bold')
            ax.set_xlabel('Advantage Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            stats_text = f'μ={np.mean(data):.3f}\nσ={np.std(data):.3f}\nmed={np.median(data):.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 下排：对比图
        # 箱线图
        ax_box = axes[1, 0]
        bp = ax_box.boxplot(stage_data, labels=stage_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax_box.set_title('Box Plot Comparison', fontweight='bold')
        ax_box.set_ylabel('Advantage Value')
        ax_box.grid(True, alpha=0.3)
        ax_box.tick_params(axis='x', rotation=45)
        
        # CDF图
        ax_cdf = axes[1, 1]
        for data, label, color in zip(stage_data, stage_labels, colors):
            sorted_data = np.sort(data)
            y = np.arange(1, len(data) + 1) / len(data)
            ax_cdf.plot(sorted_data, y, label=label, color=color, linewidth=2)
        ax_cdf.set_title('Cumulative Distribution', fontweight='bold')
        ax_cdf.set_xlabel('Advantage Value')
        ax_cdf.set_ylabel('Cumulative Probability')
        ax_cdf.legend()
        ax_cdf.grid(True, alpha=0.3)
        
        # 统计总结
        ax_stats = axes[1, 2]
        ax_stats.axis('off')
        
        stats_text = "Statistical Summary:\n\n"
        for data, label in zip(stage_data, stage_labels):
            stats_text += f"{label}:\n"
            stats_text += f"  Mean: {np.mean(data):.4f}\n"
            stats_text += f"  Std:  {np.std(data):.4f}\n"
            stats_text += f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]\n"
            stats_text += f"  Pos%: {100*np.mean(data > 0):.1f}%\n\n"
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = save_dir / "advantage_distributions.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {plot_file}")
        plt.close()
    
    def _generate_text_report(self, analysis_results: Dict, save_dir: Path):
        """生成文本分析报告"""
        report_file = save_dir / "analysis_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ADVANTAGE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Analysis Time: {analysis_results['timestamp']}\n")
            f.write(f"Samples Analyzed: {analysis_results['config_info']['num_samples_analyzed']}\n")
            f.write(f"Semantic Evaluation: {'Enabled' if self.analysis_config['enable_semantic_eval'] else 'Disabled'}\n\n")
            
            # Summary
            if 'summary' in analysis_results:
                summary = analysis_results['summary']
                f.write("SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Original Mean Advantage: {summary.get('original_mean_advantage', 'N/A'):.4f}\n")
                f.write(f"Final Mean Advantage: {summary.get('final_mean_advantage', 'N/A'):.4f}\n")
                f.write(f"Overall Transformation: {summary.get('overall_transformation', 'N/A'):.4f}\n")
                f.write(f"Stages Computed: {summary.get('stages_computed', 'N/A')}\n\n")
            
            # Detailed stats
            f.write("DETAILED STATISTICS\n")
            f.write("-" * 30 + "\n")
            for stage, stats in analysis_results['detailed_stats'].items():
                f.write(f"\n{stage.upper().replace('_', ' ')}:\n")
                if 'count' in stats and stats['count'] > 0:
                    f.write(f"  Count: {stats['count']}\n")
                    f.write(f"  Mean: {stats['mean']:.6f}\n")
                    f.write(f"  Std: {stats['std']:.6f}\n")
                    f.write(f"  Median: {stats['median']:.6f}\n")
                    f.write(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                    f.write(f"  Positive%: {100*stats['positive_ratio']:.1f}%\n")
                else:
                    f.write("  No valid data\n")
            
            # Transformation analysis
            if 'transformation_analysis' in analysis_results:
                f.write("\n\nTRANSFORMATION ANALYSIS\n")
                f.write("-" * 30 + "\n")
                for transform, changes in analysis_results['transformation_analysis'].items():
                    f.write(f"\n{transform.replace('_', ' ').title()}:\n")
                    f.write(f"  Mean Change: {changes['mean_change']:+.6f}\n")
                    f.write(f"  Std Change: {changes['std_change']:+.6f}\n")
                    f.write(f"  Median Change: {changes['median_change']:+.6f}\n")
                    f.write(f"  Range Change: {changes['range_change']:+.6f}\n")
            
            # Metadata stats
            if 'metadata_stats' in analysis_results:
                meta = analysis_results['metadata_stats']
                f.write("\n\nTRAJECTORY METADATA\n")
                f.write("-" * 25 + "\n")
                f.write(f"Total Samples: {meta['total_samples']}\n")
                f.write(f"Termination Rate: {100*meta['termination_rate']:.1f}%\n")
                f.write(f"Avg Final Reward: {meta['avg_final_reward']:.4f} ± {meta['std_final_reward']:.4f}\n")
                f.write(f"Avg Response Length: {meta['avg_response_length']:.1f} ± {meta['std_response_length']:.1f}\n")
                f.write(f"Success Rate: {100*meta['reward_distribution']['success_rate']:.1f}%\n")
        
        print(f"✓ Text report saved to: {report_file}")

    def cleanup(self):
        """清理资源"""
        print("Cleaning up resources...")
        
        try:
            # 清理异步rollout管理器
            if hasattr(self, 'async_rollout_manager') and self.async_rollout_manager:
                self.async_rollout_manager.sleep()
                print("✓ Async rollout manager stopped")
            
            # 清理Ray资源
            if hasattr(self, '_resource_pool_manager'):
                # 注意：这里可能需要根据你的Ray清理逻辑调整
                print("✓ Ray resource pool cleaned")
            
            print("✓ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Warning during cleanup: {e}")


# 工具函数
def analyze_single_checkpoint(checkpoint_path: str, 
                             config_path: str,
                             save_dir: str = None,
                             analysis_config: dict = None) -> Dict:
    """分析单个checkpoint"""
    
    # 验证输入路径
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")
    
    # 检查checkpoint结构
    actor_path = checkpoint_path / "actor"
    if not actor_path.exists():
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
    
    # 设置默认分析配置
    default_config = {
        'batch_size': 4,
        'num_samples': 16,
        'save_raw_data': True,
        'enable_semantic_eval': False,
        'skip_env_interaction': False,
    }
    
    if analysis_config:
        default_config.update(analysis_config)
    analysis_config = default_config
    
    print(f"Starting analysis with config: {analysis_config}")
    
    analyzer = AdvantageAnalyzer(str(checkpoint_path), str(config_path), analysis_config)
    
    try:
        analyzer.setup()
        
        if save_dir is None:
            save_dir = f"./analysis_results/{checkpoint_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = analyzer.analyze_advantages(save_dir)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Results saved to: {save_dir}")
        if 'summary' in results:
            summary = results['summary']
            print(f"Samples processed: {summary.get('samples_processed', 'N/A')}")
            print(f"Original mean advantage: {summary.get('original_mean_advantage', 'N/A'):.4f}")
            print(f"Final mean advantage: {summary.get('final_mean_advantage', 'N/A'):.4f}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        analyzer.cleanup()


def compare_multiple_checkpoints(checkpoint_paths: List[str],
                                config_path: str, 
                                save_dir: str = None,
                                analysis_config: dict = None) -> Dict:
    """比较多个checkpoints"""
    if save_dir is None:
        save_dir = f"./analysis_results/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"\n{'='*70}")
        print(f"Analyzing checkpoint {i+1}/{len(checkpoint_paths)}: {ckpt_path}")
        print(f"{'='*70}")
        
        try:
            # 为每个checkpoint创建子目录
            ckpt_name = Path(ckpt_path).name
            ckpt_save_dir = save_dir / ckpt_name
            
            result = analyze_single_checkpoint(
                ckpt_path, config_path, 
                save_dir=str(ckpt_save_dir),
                analysis_config=analysis_config
            )
            all_results[ckpt_name] = result
            
        except Exception as e:
            print(f"❌ Error analyzing {ckpt_path}: {e}")
            all_results[Path(ckpt_path).name] = {'error': str(e)}
    
    # 生成比较报告
    comparison_file = save_dir / "checkpoint_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 生成比较可视化
    _create_comparison_visualization(all_results, save_dir)
    
    print(f"\n✓ Comparison results saved to: {comparison_file}")
    return all_results


def _create_comparison_visualization(all_results: Dict, save_dir: Path):
    """创建checkpoints比较可视化"""
    # 提取各checkpoint的关键指标
    checkpoint_names = []
    original_means = []
    final_means = []
    transformations = []
    
    for ckpt_name, result in all_results.items():
        if 'error' in result:
            continue
        
        if 'summary' in result:
            summary = result['summary']
            checkpoint_names.append(ckpt_name)
            original_means.append(summary.get('original_mean_advantage', 0))
            final_means.append(summary.get('final_mean_advantage', 0))
            transformations.append(summary.get('overall_transformation', 0))
    
    if not checkpoint_names:
        print("No valid results to visualize")
        return
    
    # 创建比较图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Checkpoint Comparison', fontsize=16, fontweight='bold')
    
    # 原始advantage均值
    axes[0, 0].bar(checkpoint_names, original_means, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Original Mean Advantages')
    axes[0, 0].set_ylabel('Advantage Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 最终advantage均值
    axes[0, 1].bar(checkpoint_names, final_means, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Final Mean Advantages')
    axes[0, 1].set_ylabel('Advantage Value')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 变换幅度
    axes[1, 0].bar(checkpoint_names, transformations, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Overall Transformation')
    axes[1, 0].set_ylabel('Advantage Change')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 原始vs最终散点图
    axes[1, 1].scatter(original_means, final_means, s=100, alpha=0.7)
    for i, name in enumerate(checkpoint_names):
        axes[1, 1].annotate(name, (original_means[i], final_means[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 1].set_xlabel('Original Mean Advantage')
    axes[1, 1].set_ylabel('Final Mean Advantage')
    axes[1, 1].set_title('Original vs Final')
    axes[1, 1].grid(True, alpha=0.3)
    # 添加y=x参考线
    lims = [
        np.min([axes[1, 1].get_xlim(), axes[1, 1].get_ylim()]),
        np.max([axes[1, 1].get_xlim(), axes[1, 1].get_ylim()]),
    ]
    axes[1, 1].plot(lims, lims, 'k-', alpha=0.5, zorder=0)
    
    plt.tight_layout()
    
    # 保存比较图表
    plot_file = save_dir / "checkpoint_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison visualization saved to: {plot_file}")
    plt.close()

