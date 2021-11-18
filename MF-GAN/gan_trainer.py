#/*******

#author              : xiongkun
#email               : xk18@mails.tsinghua.edu.cn
#homepage            : https://github.com/2742195759
#last modified time  : 10/17/2021

#*******/


import logging#{{{
import os
import calculate_kldiv
import pickle as pkl
import sys
from collections import OrderedDict
import torch
import tqdm
from colorama import Fore, Style
import cvpods.data.transforms as T
from cvpods.engine.base_runner import RunnerBase
from cvpods.checkpoint import Checkpointer
from cvpods.data import build_test_loader, build_train_loader
from cvpods.evaluation import (
    DatasetEvaluator, inference_on_dataset,
    print_csv_format, verify_results
)
from cvpods.modeling.nn_utils.precise_bn import get_bn_modules
from cvpods.solver import build_lr_scheduler, build_optimizer
from cvpods.utils import (
    CommonMetricPrinter, JSONWriter, PathManager,
    TensorboardXWriter, collect_env_info, comm,
    seed_all_rng, setup_logger, VisdomWriter
)
from cvpods.engine import DefaultRunner, default_argument_parser, default_setup, hooks, launch
from cvpods.evaluation import ClassificationEvaluator
from cvpods.utils import comm
import cvpods.model_zoo as model_zoo
from config import clean_cfg, noise_cfg, global_cfg
from sklearn.metrics import roc_auc_score
import copy
import sys
import numpy as np
import show_distribution
import common

#}}}

if True:  # "INIT"/*{{{*/
    setup_logger("/home/data/GAN/gan-based/gan_trainer_log.txt")
    gen = common.Generator()
    dis = common.Discriminator()
    clean_dataloader = build_train_loader(clean_cfg)
    noise_dataloader = build_train_loader(noise_cfg)#/*}}}*/

class Cvpack2DataloaderWrapper: #/*{{{*/
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self.sampler = dataloader.sampler

    def __iter__(self):
        self.next_batch = []
        self.next_idx   = 0
        self.iter = iter(self.dataloader)
        return self

    def __next__(self):
        while (self.next_idx + 1 > len(self.next_batch)):
            data = next(self.iter)
            self.next_batch = self.process(data)
            self.next_idx = 0
        new_data = self.next_batch[self.next_idx]
        self.next_idx += 1
        return new_data
    
    def __len__(self):
        return len(self.dataloader)

    def process(self, data):
        raise NotImplementedError("Please implement the process function.")

class GenDataloaderWrapper(Cvpack2DataloaderWrapper):
    def __init__(self, dataloader, dis_model):
        Cvpack2DataloaderWrapper.__init__(self, dataloader)
        self.dis_model = dis_model

    def process(self, data) : 
        self.dis_model.eval()
        scores = self.dis_model(data).reshape([-1]).tolist()
        for i, s in enumerate(scores):
            data[i]['dis_gt'] = s
        return [data]

class DisDataloaderWrapper(Cvpack2DataloaderWrapper):
    def __init__(self, dataloader):
        Cvpack2DataloaderWrapper.__init__(self, dataloader)

    def process(self, data) : 
        ret = []
        batch_size = len(data)
        for d in data:
            d['is_clean'] = 1
        #assert batch_size % 2 == 0, "Use DisDataloaderWrapper must ensure batch_size is even"
        i = 0
        ret.append(
            data[i*batch_size:(i+1)*batch_size] 
            + self.fake[i*batch_size:(i+1)*batch_size])
        assert (ret[0].__len__() == 2 * batch_size)
        return ret

    def __len__(self):
        return len(self.dataloader)

    def insert_fake(self, fake):
        self.fake = fake
        self.noise_size = len(self.fake)
#}}}

class GenTrainer(DefaultRunner): #{{{
    def __init__(self, cfg, model):
        super().__init__(cfg, lambda x: model)

    @classmethod
    def build_train_loader(cls, cfg):
        data_loader = build_train_loader(cfg)
        return GenDataloaderWrapper(data_loader, dis)

    def train(self):
        super(GenTrainer, self).train()

    def test(self, a, b): 
        pass
#}}}

class DisTrainer(DefaultRunner): #{{{
    def __init__(self, cfg, model ):
        super().__init__(cfg, lambda x: model)

    def train(self):
        super(DisTrainer, self).train()

    @classmethod
    def build_train_loader(cls, cfg):
        data_loader = build_train_loader(cfg)
        return DisDataloaderWrapper(data_loader)

    def test(self, a, b): 
        pass
#}}}

class GANTrainer(RunnerBase):  #{{{
    def __init__(self, cfg): #{{{
        super(GANTrainer, self).__init__()
        self.step = 0
        self.setup(cfg)
        self.register_hooks(self.build_hooks(cfg))#}}}
    def build_hooks(self, cfg):#{{{
        ret = []
        if hasattr(cfg, 'VISDOM') and cfg.VISDOM.TURN_ON == True:
            logger = logging.getLogger("Visdom")
            logger.info('Host:' + str(cfg.VISDOM.HOST), 'Port:' + str(cfg.VISDOM.PORT))
            ret.append(hooks.PeriodicWriter([VisdomWriter(cfg.VISDOM.HOST, cfg.VISDOM.PORT, 1, cfg.VISDOM.KEY_LIST, cfg.VISDOM.ENV_PREFIX)], 1))
        return ret
#}}}
    def cal_reweight(self):#{{{
        show_distribution.main(gen)#}}}
    def _gen_fake_data_by_method(self, method, logits, size):#{{{
        if method == 'sample':
            return self._gen_fake_data_sample(size, logits)
        elif method == 'topk': 
            return self._gen_fake_data_topk(size, logits)#}}}
    def _gen_fake_data_sample(self, sample_size, logits):#{{{
        """ return idxs: list like
            param: logits shape = N, 
        """
        logits = logits.reshape([-1])
        length= logits.shape[0]
        prob = torch.functional.F.softmax(logits, dim=-1)
        indices = np.random.choice(length, (sample_size, ), replace=True, p=prob.numpy())
        print ("Sample Number:", len(set(list(indices))))
        return indices#}}}
    def _gen_fake_data_topk(self, sample_size, logits):#{{{
        """ return idxs: list(int) like
            param: logits shape = N, 
        """
        indices = (torch.topk(logits[:,0], sample_size)[1]).tolist()
        return indices#}}}
    def _gen_fake_data(self, method, size=None):#{{{
        """ check if the selected sampler is 
        """
        output = []
        data_tmp = []
        clean_size = size
        if size == None:
            clean_size = noise_cfg.DATASETS.CLEAN_NUM

        print ( "Inserting fake samples" )
        gen.eval()
        for data in tqdm.tqdm(noise_dataloader): 
            output.append(gen(data).reshape([-1,1]))
            data_tmp.extend(data)
        output = torch.cat(output, dim=0)

        indices = self._gen_fake_data_by_method(method, output, clean_size)
        data_fake = []
        choise_clean_rate = 0.0
        for idx in indices: 
            data_tmp[idx]['is_clean'] = 0  # 0 is noise, 1 is clean
            data_fake.append(data_tmp[idx])
        return data_fake#}}}
    def setup(self, cfg): #{{{
        cfg.link_log()
        self.start_iter = cfg.GAN.START_ITER
        self.max_iter   = cfg.GAN.MAX_ITER
        self.epoch_dis = cfg.GAN.EPOCH_DIS
        self.epoch_gen = cfg.GAN.EPOCH_GEN 
        self.interval = cfg.GAN.HISTOGRAM_INTERVAL 
        if cfg.GAN.RESUME:
            print ("Resuming")
            self.resume = True
        else:
            print ("From Stratch")
            self.resume = False
        if self.resume: self.resume_model()
#}}}
    def resume_model(self):#{{{
        self.dis_trainer = DisTrainer(copy.deepcopy(clean_cfg), dis)
        self.gen_trainer = GenTrainer(copy.deepcopy(noise_cfg), gen)
        self.dis_trainer.resume_or_load(False)  # just load the weigths from MODEL.WEIGHTS""
        self.gen_trainer.resume_or_load(False)#}}}
    def eval_dis_model(self, dataloader, prefix='noise', mode='eval'):#{{{
        if mode == 'eval':
            dis.eval()
        else : 
            dis.train()
        hit = 0.0
        tot = 0.0
        clean_tot = 0
        losses = 0.0
        neg_nr = 0.0
        outputs = []
        for batch in tqdm.tqdm(dataloader): 
            output = dis(batch).reshape([-1]).cpu().detach()
            neg_nr += (output < 0.5).sum()
            outputs.append(output)
            label  = torch.as_tensor([ b['is_clean'] for b in batch ]).float()
            clean_tot += label.sum().cpu()
            losses += torch.nn.BCELoss(reduction='sum')(output, label)
            for idx, item in enumerate(output.tolist()): 
                if item > 0.5 and batch[idx]['is_clean'] == 1: hit += 1
                if item <=0.5 and batch[idx]['is_clean'] == 0: hit += 1
            tot += len(batch)
        outputs = torch.cat(outputs[:5], dim=-1)
        print ("NegNumber    : ", neg_nr)
        print ("Discriminator: ", hit / tot)
        print ("Clean Rate   : ", clean_tot / tot)
        print ("Losses       : ", losses / tot)
        if prefix : 
            self.storage.put_scalar("dis-" + prefix + '-loss', losses / tot)
            if self.step % self.interval == 0:
                self.storage.put_tensor(outputs, 'histogram', {
                        'title': 'dis-'+prefix+'-'+str(self.step), 
                        'bincount': 200,
                    })
#}}}
    def run_step(self):#{{{
        """
        Run just a iteration inside the EventStorage
        """
        self.step += 1
        print ("#" * 80)
        print ("Current Step:", self.step)

        print ("######### Train Discriminative model")
        fake_data = self._gen_fake_data(global_cfg.GAN.FAKE_DATA_METHOD)
        for _ in range(self.epoch_dis): 
            self.dis_trainer = DisTrainer(copy.deepcopy(clean_cfg), dis)
            self.dis_trainer.data_loader.insert_fake(fake_data)
            dis.train()
            self.dis_trainer.train()  # train dis model with clean data and fake data from generative model

        print ("######### Train Generative model")
        for _ in range(self.epoch_gen): 
            self.gen_trainer = GenTrainer(copy.deepcopy(noise_cfg), gen)
            gen.train()
            self.gen_trainer.train()

        self.cal_reweight()
        kl = calculate_kldiv.main(False)
        self.storage.put_scalar("KLDIV", kl)
        print ("\n\nKLDIV\n\n", kl)

#}}}
    def start(self):#{{{
        self.train(self.start_iter, 1, self.max_iter)
        #self.baseline_task.train()#}}}
#}}}

# main function{{{
gan_trainer = GANTrainer(copy.deepcopy(global_cfg))
gan_trainer.start()
#}}}
