from models import GuideGateDTI
import time
from network.tools import set_seed, mkdir
from configs import get_cfg_defaults
from dataloader import preparedataset
from trainer import Trainer
import torch
import argparse
import warnings, os
from datetime import datetime


cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available(): 
    torch.cuda.set_device(cuda_id)  
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device index: {torch.cuda.current_device()}")  
    print(f"Current device name: {torch.cuda.get_device_name(device)}")


parser = argparse.ArgumentParser(description="GuideGateDTI for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='DrugBank',choices=["human","bindingdb","celegans","biosnap","DrugBank","Davis","KIBA"])
parser.add_argument('--type', type=str,default="random", choices=["cluster","cold","random","E2","E3","E4"],help='cluster,cold or random')

args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    warnings.filterwarnings("ignore", message="invalid value encountered "
    "in divide")
    output_path = os.path.join(cfg.RESULT.OUTPUT_DIR, args.data)
    cfg.RESULT.OUTPUT_DIR = output_path
    mkdir(output_path)

    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    dataFolder = f'../dataset/{args.data}/'

    # """load data"""
    training_generator , val_generator , test_generator,_,_= preparedataset(cfg.SOLVER.BATCH_SIZE,args.type,args.data)
    all_results = []
    seed = cfg.SOLVER.SEED  
    # for seed in seeds:
    print(f"\n===== Running with seed {seed} =====")
    set_seed(seed)
    seed_output_path = os.path.join(output_path, f"seed_{seed}")
    cfg.RESULT.OUTPUT_DIR = seed_output_path
    mkdir(seed_output_path)
    model = GuideGateDTI(device=device, **cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, **cfg)
    result = trainer.train(args.type)
    all_results.append(result)
    seed_model_file = os.path.join(seed_output_path, "model_architecture.txt")
    with open(seed_model_file, "w") as wf:
        wf.write(str(model))

    return all_results

if __name__ == '__main__':
    print(f"start: {datetime.now()}")
    start_time = time.time()
    """ train """
    result = main()
    """"""
    end_time = time.time()
    total_time_seconds = end_time - start_time
    hours = total_time_seconds // 3600
    minutes = (total_time_seconds % 3600) // 60
    seconds = total_time_seconds % 60
    print("Total running time of the model: {} hours {} minutes {} seconds".format(int(hours), int(minutes),int(seconds)))
    print(f"end: {datetime.now()}")