from pathlib import Path
import sys
import pandas as pd
import numpy as np

from model import LSTM_TrajGAN
from utlis import train_file_parse_args

if __name__ == '__main__':
    # n_epochs = int(sys.argv[1])
    # n_batch_size = int(sys.argv[2])
    # n_sample_interval = int(sys.argv[3])
    args  = train_file_parse_args()

    train_csv = args.train_csv
    test_csv = args.test_csv
    train_npy = args.train_npy
    output_dir = args.output_dir

    epochs = args.epochs
    batch_size = args.batch_size
    save_params_rate = args.save_params_rate
    
    latent_dim = 100
    max_length = 144
    
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {"lat_lon":2,"day":7,"hour":24,"category":10,"mask":1}
    
    # tr = pd.read_csv('data/train_latlon.csv')
    # te = pd.read_csv('data/test_latlon.csv')
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    
    lat_centroid = (tr['lat'].sum() + te['lat'].sum())/(len(tr)+len(te))
    lon_centroid = (tr['lon'].sum() + te['lon'].sum())/(len(tr)+len(te))
    
    scale_factor=max(max(abs(tr['lat'].max() - lat_centroid),
                         abs(te['lat'].max() - lat_centroid),
                         abs(tr['lat'].min() - lat_centroid),
                         abs(te['lat'].min() - lat_centroid),
                        ),
                     max(abs(tr['lon'].max() - lon_centroid),
                         abs(te['lon'].max() - lon_centroid),
                         abs(tr['lon'].min() - lon_centroid),
                         abs(te['lon'].min() - lon_centroid),
                        ))
    
    gan = LSTM_TrajGAN(latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor)
    
    # gan.train(epochs=n_epochs, batch_size=n_batch_size, sample_interval=n_sample_interval)
    gan.train(epochs=epochs, batch_size=batch_size, sample_interval=save_params_rate, train_npy=train_npy, output_dir=output_dir)