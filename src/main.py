import os, sys, pickle, argparse, yaml,torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from models import Encoder, NNBlock, NormalizingFlow, BimodalGMM
from train import ModelTrainer, ModelPreTrainer 
from utils import MyCallBack, SSLDataset
from collections import OrderedDict

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#########################################################################################################################################
# PARSE ARGS
#########################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--subdir', type=str, required=True, action="store", help="checkpoint directory")
parser.add_argument('--config', type=str, required=False, action="store", default="main", help="config file")
parser.add_argument('-c', '--train_checkpoint', type=str, required=False, action="store", help="checkpoint file(.ckpt) path")
parser.add_argument('-d', '--parent_dir', type=str, required=False, action="store", default="/home/anup/AFP3_PB", help="parent working directory")
parser.add_argument('-g', '--device', type=int, required=False, action="store", nargs='+', default=[0], help="gpu device")
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

if args.train_checkpoint is not None:
    cfg = pickle.load(open(os.path.join(args.parent_dir, "checkpoints", args.train_checkpoint.split("checkpoints")[1].split("/")[1], "params.pkl"), "rb"))
    # cfg = pickle.load(open(os.path.join(args.parent_dir, "d256/checkpoints_d256", args.train_checkpoint.split("checkpoints")[1].split("/")[1], "params.pkl"), "rb"))
    cfg['checkpoint'] = args.train_checkpoint
    cfg['seed'] = 29
    cfg['lr'] = 0.00005
    print("Loading saved hyperparams...")

else:
    with open("../config/"+args.config+".yaml") as f:
        print(f"Reading config file: {args.config}...")
        cfg = yaml.load(f, Loader=yaml.FullLoader)

pl.seed_everything(cfg['seed'], workers=True)

#########################################################################################################################################
# PREPARE DATASET
#########################################################################################################################################
train_dataset = SSLDataset(audiopath=cfg['train_clean'], noisepath=cfg['train_noise'], rirpath=cfg['train_rir'], fs=cfg['fs'], seglen=cfg['seglen'],
                 power_thresh=cfg['powerthresh'], audiofeat=cfg['audiofeat'], audiofeat_params=cfg['audiofeat_params'], max_offset=cfg['max_offset'], 
                 snr_range=cfg['snr_range'],specaug=cfg['specaug'], distort_probs=cfg['train_distort_probs'])

valid_dataset = SSLDataset(audiopath=cfg['valid_clean'], noisepath=cfg['valid_noise'], rirpath=cfg['valid_rir'], fs=cfg['fs'], seglen=cfg['seglen'],
                 power_thresh=cfg['powerthresh'], audiofeat=cfg['audiofeat'], audiofeat_params=cfg['audiofeat_params'], max_offset=cfg['max_offset'], 
                 snr_range=cfg['snr_range'], distort_probs=cfg['valid_distort_probs'])

print(f"##########\nTotal files -->\nTraining files: {len(train_dataset)}\nValidation files: {len(valid_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=int(cfg['batchsize']), shuffle=True, drop_last=True, num_workers=cfg['load_workers'], pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=int(cfg['batchsize']), shuffle=False, drop_last=True, num_workers=cfg['load_workers'], pin_memory=True)


#########################################################################################################################################
# TRAINING 
#########################################################################################################################################

#models
encoder = Encoder(inp_dims=cfg['patch_emb_dim'], patch_size=cfg['patch_size'], nhead=cfg['nhead'], dim_feedforward=cfg['dim_feedforward'], num_layers=cfg['num_layers'],
                 activation=cfg['encoder_activation'], projection_dims=cfg['projection_dims'], concat_position=cfg['concat_position'])
nnblock = NNBlock(inp_dims=cfg['projection_dims']['out'], bits=cfg['bits'], activation=cfg['activation'], factor=cfg['factor'])
base = BimodalGMM(num_feats=cfg['bits'])
nflows = NormalizingFlow(num_layers=cfg['nf_nblocks'], nfeats=cfg['bits'], mlp_units=cfg['nf_mlp_units'], q0=base)

# save experiment parameters
# dic_path = os.path.join("../d256/checkpoints_d256", args.subdir)
dic_path = os.path.join("../checkpoints", args.subdir)
if os.path.isdir(dic_path) is False:
    os.makedirs(dic_path)
pickle.dump(cfg, open(os.path.join(dic_path, "params.pkl"), "wb"))

# NF training
pretrained_model = ModelPreTrainer.load_from_checkpoint(ckpt_path)
train_module = ModelTrainer(pretrained_model=pretrained_model, nflows=nflows, lambd=cfg['lambda'], temp=cfg['temp'], optimizer=cfg['optimizer'], lr=cfg['lr'], wt_decay=cfg['weight_decay'],)

# callbacks
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(filename='{epoch}-{valid_loss:.2f}-{train_loss:.2f}', monitor="valid_loss", save_top_k=1, save_last=True)
custom_callback = MyCallBack()

#logger
version = "_".join(["temp:"+str(cfg['temp']), "bsz:"+str(cfg['batchsize']), "lr:"+str(cfg['lr']), "seg:"+str(cfg['seglen']), "emb:"+str(cfg['patch_emb_dim'])])
logger = TensorBoardLogger(save_dir=os.path.join(args.parent_dir, "checkpoints"), name=args.subdir, version=version, default_hp_metric=False)
# logger = TensorBoardLogger(save_dir=os.path.join(args.parent_dir, "d256/checkpoints_d256"), name=args.subdir, version=version, default_hp_metric=False)

trainer = Trainer(accelerator="gpu", callbacks=[lr_monitor, checkpoint_callback, custom_callback], gpus=args.device, logger=logger, deterministic=False, max_epochs=4000, log_every_n_steps=48)#, gradient_clip_val=10, gradient_clip_algorithm="value") 

if args.train_checkpoint is not None:
    print(f"Training resumes from checkpoint: {cfg['checkpoint']}")
    trainer.fit(train_module, train_dataloader, valid_dataloader,ckpt_path=cfg['checkpoint'])
else: 
    print("training from scratch begins")
    trainer.fit(train_module, train_dataloader, valid_dataloader)
