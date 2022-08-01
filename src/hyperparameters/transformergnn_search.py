import os
os.environ["SLURM_JOB_NAME"] = "bash"
import pytorch_lightning as pl
from train_transformergnn import get_data, Model
from src.hyperparameters.search import ihm_TuneReportCallback, los_TuneReportCallback, main_tune
from src.args import init_transformergnn_args, add_tune_params, add_configs, get_transformer_out_dim

def main_train(config):
    dataset, train_loader, subgraph_loader = get_data(config)

    # define model
    model = Model(config, dataset, train_loader, subgraph_loader)

    trcb = [ihm_TuneReportCallback()] if config['task'] == 'ihm' else [los_TuneReportCallback()]
    config['transformer_outdim'], config['transformer_last_ts_dim'] = get_transformer_out_dim(config)
    config['gnn_indim'] = config['transformer_outdim']


    trainer = pl.Trainer(
        gpus=config['gpus'],
        progress_bar_refresh_rate=0,
        weights_summary=None,
        max_epochs=config['epochs'],
        distributed_backend='dp',
        precision=16 if config['use_amp'] else 32,
        default_root_dir=config['log_path'],
        deterministic=True,
        callbacks=trcb
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = init_transformergnn_args()
    parser = add_tune_params(parser)
    config = parser.parse_args()
    config.model = 'transformergnn'
    config = add_configs(config)
    main_tune(main_train, config)