#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from copy import deepcopy
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
import math
from tools.early_stop import EarlyStopping
from tools.tool import get_data_path, get_adj_matrix, get_data_nodes
from model.STPDformerFlow import PDFormerFlow

# Monkey patch for transformer compatibility
def patch_transformer():
    """Patch transformer modules to handle older PyTorch versions"""
    # Patch TransformerEncoderLayer
    if not hasattr(nn.TransformerEncoderLayer, '_old_forward'):
        nn.TransformerEncoderLayer._old_forward = nn.TransformerEncoderLayer.forward
        
        def new_forward(self, src, src_mask=None, src_key_padding_mask=None):
            # Convert to (seq_len, batch, features) if needed
            if src.dim() == 3 and src.size(0) > 1:
                src = src.permute(1, 0, 2)
            output = self._old_forward(src, src_mask, src_key_padding_mask)
            if src.dim() == 3 and src.size(0) > 1:
                output = output.permute(1, 0, 2)
            return output
            
        nn.TransformerEncoderLayer.forward = new_forward
    
    # Patch TransformerEncoder
    if not hasattr(nn.TransformerEncoder, '_old_forward'):
        nn.TransformerEncoder._old_forward = nn.TransformerEncoder.forward
        
        def new_forward(self, src, mask=None, src_key_padding_mask=None):
            # Convert to (seq_len, batch, features) if needed
            if src.dim() == 3 and src.size(0) > 1:
                src = src.permute(1, 0, 2)
            output = self._old_forward(src, mask, src_key_padding_mask)
            if src.dim() == 3 and src.size(0) > 1:
                output = output.permute(1, 0, 2)
            return output
            
        nn.TransformerEncoder.forward = new_forward

# Apply patch before model creation
patch_transformer()

def split_dataset(data, train_rate=0.7, val_rate=0.1, seq_len=12, predict_len=1, wise='matrix'):
    """
    Split the dataset into training, validation, and test sets.
    Args:
        data: numpy array of shape (time_steps, num_nodes, num_nodes) for OD flows
        train_rate: ratio of training data
        val_rate: ratio of validation data
        seq_len: input sequence length (history)
        predict_len: output sequence length (prediction)
        wise: splitting strategy, 'matrix' meaning by time steps
    Returns:
        train_x, train_y, val_x, val_y, test_x, test_y, max_data
    """
    # Normalize the data
    max_data = np.max(data)
    data = data / max_data
    
    # Total time steps
    total_steps = data.shape[0]
    
    # Create sequences
    sequences = []
    labels = []
    
    for i in range(total_steps - seq_len - predict_len + 1):
        sequences.append(data[i:i+seq_len])
        labels.append(data[i+seq_len:i+seq_len+predict_len])
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # Add feature dimension
    sequences = sequences[..., np.newaxis]  # Shape: (samples, seq_len, num_nodes, num_nodes, 1)
    labels = labels[..., np.newaxis]  # Shape: (samples, predict_len, num_nodes, num_nodes, 1)
    
    # Flatten the spatial dimensions (OD matrix to flow vector)
    sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], -1, sequences.shape[-1])
    labels = labels.reshape(labels.shape[0], labels.shape[1], -1, labels.shape[-1])
    
    # Split dataset
    num_samples = sequences.shape[0]
    num_train = int(num_samples * train_rate)
    num_val = int(num_samples * val_rate)
    num_test = num_samples - num_train - num_val
    
    # Split indices
    train_indices = np.arange(0, num_train)
    val_indices = np.arange(num_train, num_train + num_val)
    test_indices = np.arange(num_train + num_val, num_samples)
    
    # Create splits
    train_x = sequences[train_indices]
    train_y = labels[train_indices]
    val_x = sequences[val_indices]
    val_y = labels[val_indices]
    test_x = sequences[test_indices]
    test_y = labels[test_indices]
    
    return train_x, train_y, val_x, val_y, test_x, test_y, max_data
    
    
def get_model(model_name, args):
    """Model selection function with PDFormerFlow added"""
    if model_name == 'PDFormerFlow':
        # Create dummy data_feature required by PDFormerFlow
        num_flows = args.num_flows
        num_nodes = int(math.sqrt(num_flows))  # Assuming square OD matrix
        
        # Generate origin-destination mappings
        origins = np.repeat(np.arange(num_nodes), num_nodes)
        destinations = np.tile(np.arange(num_nodes), num_nodes)
        
        # Create dummy node features (using one-hot encoding)
        node_features = np.eye(num_nodes)
        
        # Create dummy distance and similarity matrices
        flow_dists = np.random.uniform(1, 100, (num_flows, num_flows))
        np.fill_diagonal(flow_dists, 0)
        
        # Create DTW matrix (pattern similarity)
        flow_dtw_matrix = np.random.rand(num_flows, num_flows)
        np.fill_diagonal(flow_dtw_matrix, 1)
        
        # Create pattern keys (random initialization)
        flow_pattern_keys = np.random.randn(num_flows, args.s_attn_size, 1)
        
        # Create data_feature dictionary
        data_feature = {
            'num_flows': num_flows,
            'origins': origins,
            'destinations': destinations,
            'node_features': node_features,
            'flow_dists': flow_dists,
            'flow_dtw_matrix': flow_dtw_matrix,
            'flow_pattern_keys': flow_pattern_keys,
            'scaler': None,
            'feature_dim': 1,  # Only flow value as feature
            'ext_dim': 0,      # No external features
            'num_batches': args.batch_size,
        }
        
        # Create config dictionary from args
        config = {
            'embed_dim': args.dim_model,
            'skip_dim': args.skip_dim,
            'geo_num_heads': args.geo_num_heads,
            'sem_num_heads': args.sem_num_heads,
            't_num_heads': args.t_num_heads,
            'mlp_ratio': args.mlp_ratio,
            'qkv_bias': True,
            'drop': args.dropout,
            'attn_drop': args.attn_drop,
            'drop_path': args.drop_path,
            's_attn_size': args.s_attn_size,
            't_attn_size': args.t_attn_size,
            'enc_depth': args.encoder_layers,
            'type_ln': args.type_ln,
            'type_short_path': args.type_short_path,
            'output_dim': 1,
            'input_window': args.seq_len,
            'output_window': args.pre_len,
            'add_time_in_day': False,
            'add_day_in_week': False,
            'device': args.device,
            'huber_delta': 1.0,
            'quan_delta': 0.25,
            'far_mask_delta': args.far_mask_delta,
            'dtw_delta': args.dtw_delta,
            'use_curriculum_learning': args.use_curriculum,
            # Adjust step_size to be smaller than epochs
            'step_size': min(args.step_size, args.epochs // 2),
            'max_epoch': args.epochs,
            'task_level': 0,
        }
        
        # Create and return the model
        model = PDFormerFlow(config, data_feature)
        return model
        
  
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_loss_func(loss_name):
    """Loss function selection"""
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'huber':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='PDFormerFlow', help='train model name')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--dataset', default='abilene', help='choose dataset', 
                    choices=['geant', 'abilene','nobel','germany'])
parser.add_argument('--gpu', type=int, default=0, help='use -1/0/1 chose cpu/gpu:0/gpu:1', choices=[-1, 0, 1])
parser.add_argument('--batch_size', '--bs', type=int, default=64, help='batch_size')
parser.add_argument('--learning_rate', '--lr', type=float, default=0.0001, help='learning_rate')
parser.add_argument('--seq_len', type=int, default=12, help='input history length')
parser.add_argument('--pre_len', type=int, default=3, help='prediction length')
parser.add_argument('--dim_model', type=int, default=64, help='dimension of embedding vector')
parser.add_argument('--num_flows', type=int, default=144, help='number of OD flows')
parser.add_argument('--dim_attn', type=int, default=32, help='dimension of attention')
parser.add_argument('--num_heads', type=int, default=1, help='attention heads')
parser.add_argument('--train_rate', type=float, default=0.7, help='training data ratio')
parser.add_argument('--rnn_layers', type=int, default=3, help='rnn layers')
parser.add_argument('--encoder_layers', type=int, default=3, help='encoder layers')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--attn_drop', type=float, default=0.1, help='attention dropout rate')
parser.add_argument('--drop_path', type=float, default=0.1, help='drop path rate')
parser.add_argument('--early_stop', type=int, default=10, help='early stop patient epochs')
parser.add_argument('--loss', default='mse', help='loss function', choices=['mse','mae','huber'])
parser.add_argument('--l2_loss', type=float, default=0, help='L2 regularization weight')
parser.add_argument('--rounds', type=int, default=2, help='number of experiment rounds')
parser.add_argument('--skip_dim', type=int, default=256, help='skip connection dimension')
parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP expansion ratio')
parser.add_argument('--geo_num_heads', type=int, default=4, help='number of geometric attention heads')
parser.add_argument('--sem_num_heads', type=int, default=2, help='number of semantic attention heads')
parser.add_argument('--t_num_heads', type=int, default=2, help='number of temporal attention heads')
parser.add_argument('--s_attn_size', type=int, default=3, help='spatial attention size')
parser.add_argument('--t_attn_size', type=int, default=3, help='temporal attention size')
parser.add_argument('--type_ln', default='pre', choices=['pre', 'post'], help='layer normalization type')
parser.add_argument('--type_short_path', default='dist', choices=['dist', 'hop'], help='short path type')
parser.add_argument('--far_mask_delta', type=float, default=5.0, help='distance threshold for masking')
parser.add_argument('--dtw_delta', type=int, default=5, help='top-k similar flows for semantic attention')
parser.add_argument('--use_curriculum', type=int, default=1, choices=[0, 1], help='use curriculum learning')
parser.add_argument('--step_size', type=int, default=1000, help='curriculum learning step size')

args = parser.parse_args()

# Set device
if args.gpu == -1 or not torch.cuda.is_available():
    device = 'cpu'
else:
    device = f'cuda:{args.gpu}'
args.device = device

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if device.startswith('cuda'):
    torch.cuda.manual_seed_all(args.seed)

# Dataset parameters
dataset = args.dataset
fea_path = get_data_path(dataset)
num_nodes = get_data_nodes(dataset)
m_adj = np.load(get_adj_matrix(dataset))
num_flows = num_nodes * num_nodes
args.num_flows = num_flows
args.m_adj = m_adj
args.num_nodes = num_nodes

# Training parameters
epochs = args.epochs
batch_size = args.batch_size
lr = args.learning_rate
seq_len = args.seq_len
pre_len = args.pre_len
train_rate = args.train_rate
rounds = args.rounds

# Adjust step_size if needed
if args.step_size > epochs:
    print(f"Warning: step_size ({args.step_size}) is larger than epochs ({epochs}). Adjusting to {epochs//2}")
    args.step_size = epochs // 2

# Result containers
ALL_TEST_MSE = []
ALL_TEST_MAE = []
ALL_PRE_TIME = []

for r in range(rounds):
    print(f'=== Round {r+1}/{rounds} ===')
    early_stop = EarlyStopping(patience=args.early_stop, verbose=True)
    
    # Load and split dataset
    data = np.load(fea_path)
    train_x, train_y, val_x, val_y, test_x, test_y, max_data = split_dataset(
        data, 
        train_rate=train_rate,
        val_rate=0.1,
        seq_len=seq_len,
        predict_len=pre_len,
        wise='matrix'
    )
    
    # Convert to tensors
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    val_x = torch.from_numpy(val_x).float()
    val_y = torch.from_numpy(val_y).float()
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).float()

    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=4)

    # Initialize model
    model = get_model(args.model, args)
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2_loss)
    criterion = get_loss_func(args.loss)
    
    print(f'Model: {args.model}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print(f'Device: {device}')
    print(args)

    # Training variables
    best_val_mse = float('inf')
    best_model_dict = None
    train_losses = []
    val_mses = []
    val_maes = []
    
    # Training loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
    
        # Training phase
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
            # Create batch with future_time placeholder
            batch = {'X': x.to(device)}
            y = y.to(device)
        
            # Forward pass
            y_hat = model(batch)
            loss = criterion(y_hat, y)
        
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            epoch_train_loss += loss.item()
    
        
        # Average training loss
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_y = []
        all_val_y_hat = []
        
        with torch.no_grad():
            for x, y in val_loader:
                batch = {'X': x.to(device)}
                y = y.to(device)
                y_hat = model(batch)
                loss = criterion(y_hat, y)
                val_loss += loss.item()
                
                # Collect predictions
                all_val_y.append(y.cpu().numpy())
                all_val_y_hat.append(y_hat.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_y = np.concatenate(all_val_y)
        val_y_hat = np.concatenate(all_val_y_hat)
        
        # Reshape for metric calculation
        val_y = val_y.reshape(-1, num_flows)
        val_y_hat = val_y_hat.reshape(-1, num_flows)
        
        val_mse = mean_squared_error(val_y, val_y_hat)
        val_mae = mean_absolute_error(val_y, val_y_hat)
        
        val_mses.append(val_mse)
        val_maes.append(val_mae)
        
        # Check for best model
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model_dict = deepcopy(model.state_dict())
            print(f'[Epoch {epoch}] New best validation MSE: {best_val_mse:.6f}')
        
        # Early stopping check
        early_stop(val_mse)
        if early_stop.early_stop:
            print(f'Early stopping at epoch {epoch}')
            break
        
        print(f'Epoch {epoch}: '
              f'Train Loss: {avg_train_loss:.6f}, '
              f'Val MSE: {val_mse:.6f}, '
              f'Val MAE: {val_mae:.6f}')
    
    # Load best model for testing
    model.load_state_dict(best_model_dict)
    model.eval()
    
    # Testing phase
    # Testing phase
    test_start_time = time.time()
    all_test_y = []
    all_test_y_hat = []
    inference_times = []

    with torch.no_grad():
        for x, y in test_loader:
            # Create test batch with future_time placeholder
            batch = {'X': x.to(device)}
            y = y.to(device)
            
            # Measure inference time
            start_infer = time.time()
            y_hat = model(batch)
            infer_time = time.time() - start_infer
            
            # Collect results
            all_test_y.append(y.cpu().numpy())
            all_test_y_hat.append(y_hat.cpu().numpy())
            inference_times.append(infer_time / y.size(0))  # Per sample time
        
        # Calculate test metrics
        test_y = np.concatenate(all_test_y)
        test_y_hat = np.concatenate(all_test_y_hat)
        
        # Reshape for metric calculation
        test_y = test_y.reshape(-1, num_flows)
        test_y_hat = test_y_hat.reshape(-1, num_flows)
        
        test_mse = mean_squared_error(test_y, test_y_hat)
        test_mae = mean_absolute_error(test_y, test_y_hat)
        avg_infer_time = np.mean(inference_times)
        
        # Store results
        ALL_TEST_MSE.append(test_mse)
        ALL_TEST_MAE.append(test_mae)
        ALL_PRE_TIME.append(avg_infer_time)
        
        print(f'[Round {r+1}] Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}, '
              f'Inference Time: {avg_infer_time:.6f} sec/sample')

# Final results
print('\n' + '='*50)
print('FINAL RESULTS ACROSS ALL ROUNDS:')
print(f'Average Test MSE: {np.mean(ALL_TEST_MSE):.6f} +/- {np.std(ALL_TEST_MSE):.6f}')
print(f'Average Test MAE: {np.mean(ALL_TEST_MAE):.6f} +/- {np.std(ALL_TEST_MAE):.6f}')
print(f'Average Inference Time: {np.mean(ALL_PRE_TIME):.6f} +/- {np.std(ALL_PRE_TIME):.6f} sec/sample')
print('='*50)