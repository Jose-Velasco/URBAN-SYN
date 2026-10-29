import torch
import os
import numpy as np
from pathlib import Path
from nyc_workflow.data import GravityTrajectoryDataset, load_gravity_data, split_trajectory_indices
from mobilitygpt.model import GPT, get_lora_model
from mobilitygpt.trainer import Trainer
from finetuners.supervised_finetuning import SupervisedFinetuner
from finetuners.dpo_finetuning import DPOFinetuner
from finetuners.ppo_finetuning import PPOFinetuner
from mobilitygpt.ppo_policy_trainer import Agent
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

def initialize_model(config, dataset, od_list):
    from datasets.data_utils import od_pair_to_adjacency_matrix
    # Update model config with dataset properties
    config.model.vocab_size = dataset.get_vocab_size()
    config.model.block_size = dataset.get_block_size()

    # Create adjacency matrix if needed
    if config.model.use_adjacency:
        adj_matrix = od_pair_to_adjacency_matrix(od_list, config.system.device)
    else:
        adj_matrix = None

    # Initialize model with updated config
    model = GPT(config.model, adj_matrix=adj_matrix)

    if config.model.use_lora:
        model = get_lora_model(model)

    # Load pretrained model for all finetuning stages
    if config.training.mode != 'pretrain':
        if not config.model.load_path:
            raise ValueError(f"Pretrained model path must be provided for {config.training.mode} finetuning")
        path = f"{config.system.work_dir}/{config.model.load_path}.pt"
        model.load_state_dict(torch.load(path))
        print(f"Loaded pretrained model from {config.model.load_path}")

    return model.to(config.system.device), adj_matrix


def train_base_model(model, dataset, config, adj_matrix=None):
    """Train the base model"""
    print("Starting base model training")
    
    # Weighted indices represent trajectories; the original dataset remains
    # available to callers for generation and all other training modes.
    if config.training.mode == 'pretrain' and config.training.gravity_sampling:
        dataset = GravityTrajectoryDataset(dataset)

    # Create samplers
    train_sampler, val_sampler = create_samplers(dataset, config)
    
    # Initialize trainer
    trainer = Trainer(
        config=config.training,
        model=model,
        train_dataset=dataset,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        DP=config.training.dp_training,
        eps=config.training.dp_epsilon
    )
    
    # Add callbacks
    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: "
                  f"train loss {trainer.loss.item():.5f}")
        if trainer.iter_num % 500 == 0:
            save_checkpoint(trainer.model, config, "model_pretrain.pt")
            
    def validation_end_callback(trainer):
        print(f"Validation loss: {trainer.val_loss:.5f}")
    
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.add_callback('validation_end', validation_end_callback)
    
    # Run training
    trainer.run()
    
    # Save final model
    save_checkpoint(model, config, "model_pretrain.pt")

def train_supervised(model, dataset, config):
    """Train with supervised finetuning"""
    print("Starting supervised finetuning")
    supervised_finetuner = SupervisedFinetuner(
        config=config,
        model=model,
        dataset=dataset,
        dp_epsilon=config.training.dp_epsilon
    )
    supervised_finetuner.train()
    save_checkpoint(model, config, "model_supervised.pt")

def train_ppo(model, dataset, config):
    """Train with PPO finetuning"""
    print("Starting PPO training")
    
    # Create reward model with same config as base model
    reward_model = GPT(config.model)
    # Try to load supervised model as reward model, if not exists train one
    supervised_path = f"{config.system.work_dir}/model_supervised.pt"
    
    if os.path.exists(supervised_path):
        print("Loading existing supervised model as reward model...")
        reward_model.load_state_dict(torch.load(supervised_path))
    else:
        print("No supervised model found. Performing supervised finetuning for reward model first...")
        # Create a copy of model for supervised training
        reward_model.load_state_dict(model.state_dict())
        supervised_finetuner = SupervisedFinetuner(
            config=config,
            model=reward_model,
            dataset=dataset,
            dp_epsilon=config.training.dp_epsilon
        )
        supervised_finetuner.train()
        save_checkpoint(reward_model, config, "model_supervised.pt")
    # Set reward model mode after loading
    reward_model.reward_model = True
    reward_model.eval()
    
    # Create policy and reference models
    policy_model = Agent(model, trainable=True)

    # Create new instance for reference model
    ref_model_base = GPT(config.model, adj_matrix=model.adj_matrix)
    ref_model_base.load_state_dict(model.state_dict())
    ref_model_base = ref_model_base.to(config.system.device)
    ref_model = Agent(ref_model_base, trainable=False)
    
    # PPO training
    ppo_finetuner = PPOFinetuner(
        config=config,
        policy_model=policy_model,
        ref_model=ref_model,
        dataset=dataset,
        reward_model=reward_model,
        gravity_sampling=config.training.gravity_sampling,
        prompt_size=config.training.prompt_size
    )
    ppo_finetuner.train()
    save_checkpoint(policy_model.model, config, "model_ppo.pt")

def train_dpo(model, dataset, config):
    """Train with DPO finetuning"""
    print("Starting DPO training")

    # Create reward model with same config as base model
    reference_model = GPT(config.model)
    # Try to load supervised model as reward model, if not exists train one
    supervised_path = f"{config.system.work_dir}/model_supervised.pt"
    
    if os.path.exists(supervised_path):
        print("Loading existing supervised model as reward model...")
        reference_model.load_state_dict(torch.load(supervised_path))
    else:
        print("No supervised model found. Performing supervised finetuning for reward model first...")
        # Create a copy of model for supervised training
        reference_model.load_state_dict(model.state_dict())
        supervised_finetuner = SupervisedFinetuner(
            config=config,
            model=reference_model,
            dataset=dataset,
            dp_epsilon=config.training.dp_epsilon
        )
        supervised_finetuner.train()
        save_checkpoint(reference_model, config, "model_supervised.pt")
    # Set reward model mode after loading
    reference_model.eval()


    dpo_finetuner = DPOFinetuner(
        config=config,
        model=model,
        reference_model=reference_model,
        dataset=dataset,
    )
    
    # Train
    dpo_finetuner.train()
    save_checkpoint(model, config, "model_dpo.pt")
    
def create_samplers(dataset, config):
    """Create train and validation samplers"""
    if config.training.mode == 'pretrain' and config.training.gravity_sampling:
        return create_gravity_samplers(dataset, config)
    return create_random_samplers(dataset, config)

def create_gravity_samplers(dataset, config):
    """Sample whole training trajectories proportionally; validate uniformly.

    Keep weights in the full dataset index space, with zero held-out weights.
    The epoch draw count is the number of training trajectory rows. Grouped
    validation fractions apply to original parents, rather than split sections.
    """
    if config.training.mode != 'pretrain' or not isinstance(dataset, GravityTrajectoryDataset):
        raise ValueError("Gravity sampling requires the pretraining trajectory dataset adapter")
    name = config.data.dataset
    directory = Path(getattr(config.data, 'root', '.')) / f"{name}-Taxi"
    gravity, parents = load_gravity_data(
        dataset, directory / f"{name}_Taxi_trajectory_train_w_gravity.csv",
        directory / f"{name}_Taxi_trajectory_train.csv",
    )
    groups = parents if parents is not None else list(range(len(dataset)))
    train_indices, val_indices = split_trajectory_indices(
        groups, config.training.validation_split, config.training.shuffle_dataset,
        config.training.random_seed,
    )
    if not train_indices:
        raise ValueError("Gravity pretraining split has no training trajectories")
    weights = gravity.clone()
    weights[val_indices] = 0
    maximum = weights.max().item()
    if maximum <= 0:
        raise ValueError("Gravity pretraining split has no positive total training weight")
    # Scaling by the largest training value preserves ratios, including equal
    # positive weights, without overflowing a raw-weight sum or shifting zeros.
    weights /= maximum
    if torch.any((gravity[train_indices] > 0) & (weights[train_indices] == 0)):
        raise ValueError("Gravity weight dynamic range is too large for proportional sampling")
    generator = torch.Generator().manual_seed(config.training.random_seed)
    return (WeightedRandomSampler(weights, len(train_indices), replacement=True, generator=generator),
            SubsetRandomSampler(val_indices))

def create_random_samplers(dataset, config):
    """Create samplers with random sampling"""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.training.validation_split * dataset_size))
    
    if config.training.shuffle_dataset:
        np.random.seed(config.training.random_seed)
        np.random.shuffle(indices)
        
    train_indices, val_indices = indices[split:], indices[:split]
    
    return (SubsetRandomSampler(train_indices),
            SubsetRandomSampler(val_indices))

def save_checkpoint(model, config, filename):
    """Save model checkpoint"""
    ckpt_path = os.path.join(config.system.work_dir, filename)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
