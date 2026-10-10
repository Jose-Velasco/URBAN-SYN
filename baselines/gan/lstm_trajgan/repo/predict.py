from pathlib import Path

import sys
import pandas as pd
import numpy as np

from model import LSTM_TrajGAN
from utlis import get_max_trajectory_length, predict_file_parse_args, load_max_length

from keras.preprocessing.sequence import pad_sequences

if __name__ == '__main__':
    # n_epochs = int(sys.argv[1])
    args = predict_file_parse_args()
    
    latent_dim = 100
    # paper is explicit that trajectories are padded to the length of the longest trajectory in the dataset,
    # using zero pre-padding, and that padded points are masked during training/inference.
    # max_length = 144
    # max_length = load_max_length(
    #     args.generator_weights_dir
    # )
    # print(f"Using cached maximum trajectory length: {max_length}")

    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {"lat_lon":2,"day":7,"hour":24,"category":10,"mask":1}
    
    # tr = pd.read_csv('data/train_latlon.csv')
    # te = pd.read_csv('data/test_latlon.csv')

    tr = pd.read_csv(args.train_csv)
    te = pd.read_csv(args.test_csv)

    if args.max_length is not None:
        if args.max_length <= 0:
            raise ValueError(
                f"--max_length must be positive, got {args.max_length}."
            )

        max_length = args.max_length
        print(f"Using provided max_length: {max_length}")
    else:
        max_length = get_max_trajectory_length(
            train_df=tr,
            test_df=te,
        )
        print(f"Derived max_length from dataset: {max_length}")
    
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
    
    # Test data
    # x_test = np.load('data/final_test.npy',allow_pickle=True)
    x_test = np.load(
        args.test_npy,
        allow_pickle=True,
    )
    
    # final_train.npy: 5 arrays needed for training
    # final_test.npy: 7 arrays needed for prediction/reconstruction
    # https://github.com/GeoDS/LSTM-TrajGAN/issues/5
    x_test = [x_test[0],x_test[1],x_test[2],x_test[3],x_test[4],x_test[5].reshape(-1,1),x_test[6].reshape(-1,1)]
    X_test = [pad_sequences(f, max_length, padding='pre', dtype='float64') for f in x_test[:5]]
    
    # Add random noise to the data
    # 1027 comes from the original authors' test-set size
    # noise = np.random.normal(0, 1, (1027, 100))
    num_trajectories = X_test[0].shape[0]

    if num_trajectories == 0:
        raise ValueError("No trajectories found in the test dataset.")

    noise = np.random.normal(
        0,
        1,
        (num_trajectories, latent_dim),
    )

    X_test.append(noise)
    
    # Load params for the generator
    # gan.generator.load_weights('training_params/G_model_' + str(n_epochs) + '.h5') # params/G_model_2000.h5

    generator_weights = (
        args.generator_weights_dir
        / "G_model_{}.h5".format(args.load_checkpoint_epochs)
    )
    gan.generator.load_weights(generator_weights)

    
    # Make predictions
    prediction = gan.generator.predict(X_test)
    
    traj_attr_concat_list = []
    for attributes in prediction:
        traj_attr_list = []
        idx = 0
        for row in attributes:
            if row.shape == (max_length, 2):
                traj_attr_list.append(row[max_length-x_test[6][idx][0]:])
            else:
                traj_attr_list.append(np.argmax(row[max_length-x_test[6][idx][0]:],axis=1).reshape(x_test[6][idx][0],1))
            idx += 1
        traj_attr_concat = np.concatenate(traj_attr_list)
        traj_attr_concat_list.append(traj_attr_concat)
    traj_data = np.concatenate(traj_attr_concat_list,axis=1)
    
    # df_test = pd.read_csv('data/dev_test_encoded_final.csv')
    df_test = pd.read_csv(
        args.encoded_test_csv
    )

    label = np.array(df_test['label']).reshape(-1,1)
    tid = np.array(df_test['tid']).reshape(-1,1)
    traj_data = np.concatenate([label,tid,traj_data],axis=1)
    df_traj_fin = pd.DataFrame(traj_data)
    
    df_traj_fin.columns = ['label','tid','lat','lon','day', 'hour', 'category','mask']
    
    # Convert location deviation to longtitude and latitude
    df_traj_fin['lat'] = df_traj_fin['lat'] + gan.lat_centroid
    df_traj_fin['lon'] = df_traj_fin['lon'] + gan.lon_centroid
    
    del df_traj_fin['mask']
    
    df_traj_fin['tid'] = df_traj_fin['tid'].astype(np.int32)
    df_traj_fin['day'] = df_traj_fin['day'].astype(np.int32)
    df_traj_fin['hour'] = df_traj_fin['hour'].astype(np.int32)
    df_traj_fin['category'] = df_traj_fin['category'].astype(np.int32)
    df_traj_fin['label'] = df_traj_fin['label'].astype(np.int32)

    output_csv = Path(args.output_csv)

    # Ensure the output directory exists before writing the generated trajectories.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Save synthetic trajectory data
    # df_traj_fin.to_csv('results/syn_traj_test.csv',index=False)
    df_traj_fin.to_csv(
        output_csv,
        index=False,
    )
    
    
    
    
    
    
    
    
    
    
    