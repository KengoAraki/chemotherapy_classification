import logging
import os
import sys
import yaml
import joblib
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eval import plot_confusion_matrix, eval_metrics
from src.util import fix_seed
from src_multi.dataset import WSI_multi, get_files_dic
from src_multi.model import MultiscaleNet
from src_multi.eval import eval_net_test

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    fix_seed(0)
    # config_path = './config/config_src_LEV012.yaml'
    config_path = '../config/config_src_LEV012.yaml'

    with open(config_path) as file:
        config = yaml.safe_load(file.read())

    weight_dir = config['test']['weight_dir']
    weight_list = [weight_dir + name for name in config['test']['weight_names']]

    logging.basicConfig(
        level=logging.INFO,
        filename=f"{config['test']['output_dir']}{config['test']['target_data']}.txt",
        format='%(levelname)s: %(message)s'
    )

    for cv_num in range(config['main']['cv']):
        logging.info(f"== CV{cv_num} ==")
        weight_path = weight_list[cv_num]
        project_prefix = f"cv{cv_num}_"

        project = (
            project_prefix
            + config['main']['model']
            + "_" + config['main']['optim']
            + "_batch" + str(config['main']['batch_size'])
            + "_shape" + str(config['main']['shape'])
            + "_lev" + str(config['main']['levels']))
        logging.info(f"{project}\n")

        criterion = nn.CrossEntropyLoss()

        logging.info("Loading model {}".format(weight_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')

        net = MultiscaleNet(
            base_fe=config['main']['model'],
            num_class=len(config['main']['classes']))
        net.to(device=device)
        net.load_state_dict(
            torch.load(weight_path, map_location=device))

        wsis = joblib.load(
            config['dataset']['jb_dir']
            + f"cv{cv_num}_"
            + f"{config['test']['target_data']}_"
            + f"{config['main']['facility']}_wsi.jb"
        )

        files_dic = get_files_dic(
            wsis=wsis,
            levels=config['main']['levels'],
            classes=config['main']['classes'],
            imgs_dir=config['dataset']['imgs_dir'])

        dataset = WSI_multi(
            file_list_0=files_dic[0],
            file_list_1=files_dic[1],
            file_list_2=files_dic[2],
            classes=config['main']['classes'],
            shape=tuple(config['main']['shape']),
            transform={'Resize': True, 'HFlip': False, 'VFlip': False}
        )

        loader = DataLoader(
            dataset, batch_size=config['main']['batch_size'],
            shuffle=False, num_workers=2, pin_memory=True)
        val_loss, cm = eval_net_test(
            net, loader, criterion, device,
            get_miss=config['test']['get_miss'],
            save_dir=config['test']['output_dir'])

        logging.info(
            f"\n cm ({config['test']['target_data']}):\n{np.array2string(cm, separator=',')}\n"
        )
        val_metrics = eval_metrics(cm)
        logging.info('===== eval metrics =====')
        logging.info(f"\n Accuracy ({config['test']['target_data']}):  {val_metrics['accuracy']}")
        logging.info(f"\n Precision ({config['test']['target_data']}): {val_metrics['precision']}")
        logging.info(f"\n Recall ({config['test']['target_data']}):    {val_metrics['recall']}")
        logging.info(f"\n F1 ({config['test']['target_data']}):        {val_metrics['f1']}")
        logging.info(f"\n mIoU ({config['test']['target_data']}):      {val_metrics['mIoU']}")

        # Not-Normalized
        cm_plt = plot_confusion_matrix(
            cm, config['main']['classes'], normalize=False)
        cm_plt.savefig(
            config['test']['output_dir']
            + project
            + f"_{config['test']['target_data']}_nn-confmatrix.png"
        )
        plt.clf()
        plt.close()

        # Normalized
        cm_plt = plot_confusion_matrix(
            cm, config['main']['classes'], normalize=True)
        cm_plt.savefig(
            config['test']['output_dir']
            + project
            + f"_{config['test']['target_data']}_confmatrix.png"
        )
        plt.clf()
        plt.close()
