# -*- coding: utf-8 -*
import json, datetime, sys, os, pickle, argparse, time, tqdm, random
import platform

from torch.utils.tensorboard import SummaryWriter
import torch as t


if platform.system() == 'Linux':
    sys.path.append('/data/code/ZSL/C2F_ZSL_CVPR/')

# print(sys.path)

from data.episode_data import ZSL_Episode_Img_Dataset
from model.c2f_awa2 import Fine_Align, Coarse_Align
from utils.tool import *
from engine.train import do_train

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
t.autograd.set_detect_anomaly(True)
t.set_default_tensor_type(t.cuda.FloatTensor)



def prepare_work(opt):
    
    if opt.train_stage == 'Fine':
        comment = f'{opt.GMIX}_gama={opt.gama}_walk_num={opt.walk_num}_temp_{opt.temp}_pkl_data={opt.pkl_data.split("/")[-1]}'
    else:
        comment = f'coarse_iter={opt.coarse_iter_num}_lr={opt.coarse_lr}_decay={opt.coarse_opt_decay}'
    # dataset_name
    writer_dir_path = os.path.join('./tensor_writter', opt.dataset_name)
    os.makedirs(writer_dir_path) if not os.path.exists(writer_dir_path) else print('{} have been created.'.format(writer_dir_path))
    # key_word
    writer_dir_path = os.path.join(writer_dir_path, cfg.key_word)
    os.makedirs(writer_dir_path) if not os.path.exists(writer_dir_path) else print('{} have been created.'.format(writer_dir_path))
    # train_stage
    writer_dir_path = os.path.join(writer_dir_path, opt.train_stage)
    os.makedirs(writer_dir_path) if not os.path.exists(writer_dir_path) else print('{} have been created.'.format(writer_dir_path))
    # comment
    writer_dir_path = os.path.join(writer_dir_path, comment)
    os.makedirs(writer_dir_path) if not os.path.exists(writer_dir_path) else print('{} have been created.'.format(writer_dir_path))
    logger = logger_(file_path=writer_dir_path, cfg=cfg)          # make log file: ../log/dataset_name/key_word/train_stage/...
    print('Tensor_Writter file: {}'.format(writer_dir_path))
    
    Writer = SummaryWriter(log_dir=writer_dir_path)

    return Writer, logger



def main(opt, pkl_data):

    Writer, logger = prepare_work(opt)

    HU = H_U_Update()

    coarse_info = 'coarse_lr:{}, coarse_decay:{}, coarse_iter_num:{}, coarse_way_shot:{}, seed:{}, pkl_data_path:{}'.format(cfg.coarse_lr, cfg.coarse_opt_decay, cfg.coarse_iter_num, [cfg.ways, cfg.shots], cfg.seed, pkl_data.split('/')[-1])
    fine_info =      'fine_lr:{},   fine_decay:{},   fine_iter_num:{},   fine_way_shot:{}, hyper_parameters:::beta:{}, walk_num:{}, sim_gama:{}, sparse_softmax_temp:{}'.\
        format(cfg.fine_lr, cfg.fine_opt_decay, cfg.fine_iter_num, [cfg.sub_ways, cfg.sub_shots], [cfg.beta1, cfg.beta2], cfg.walk_num, cfg.gama, cfg.temp)

    print(coarse_info, '\n', fine_info)
    logger.info(coarse_info)
    logger.info(fine_info)
    logger.info(opt)

    init_seeds(opt.seed)

    # prepare data
    data_loader = ZSL_Episode_Img_Dataset(cfg=opt)
    train_loader = data_loader.train_dataloader
    test_s_loader = data_loader.test_seen_dataloader
    test_u_loader = data_loader.test_unseen_dataloader

    train_tmp_loader = data_loader.train_tmp_dataloader                       # prepare for traning the fine_branch

    vision_dim = opt.VISION_DIM                                               # 2048
    attr_dim = opt.attr_num                                                   # 312, 85, 102, 1024

    seen_att = t.from_numpy(data_loader.seen_attr).float().to(opt.device)
    unseen_att = t.from_numpy(data_loader.unseen_attr).float().to(opt.device)

    seen_class = data_loader.seen_class_big_idx
    unseen_class = data_loader.unseen_class_big_idx

    coarse_align_model = Coarse_Align(args=opt,
                                      seen_att=seen_att,
                                      unseen_att=unseen_att,
                                      vision_dim=vision_dim,
                                      attr_dim=attr_dim).to(opt.device).train()

    fine_align_model = Fine_Align(args=opt,
                                  seen_class=seen_class,
                                  seen_att=seen_att,
                                  unseen_att=unseen_att,
                                  vision_dim=vision_dim,
                                  attr_dim=attr_dim,
                                  h_dim=1600,
                                  Class_Center=None,
                                  V2S_W=None).to(opt.device).train()


    # set optimizer
    coarse_align_model_para = filter(lambda  p: id(p), coarse_align_model.parameters())
    optimizer_coarse = t.optim.SGD(coarse_align_model_para, lr=opt.coarse_lr, weight_decay=opt.coarse_opt_decay, momentum=opt.coarse_momentum)

    S2V_Para = []
    for k, v in fine_align_model.named_parameters():
        S2V_Para.append(v)

    optimizer_fine = t.optim.Adam(S2V_Para, lr=opt.fine_lr, weight_decay=opt.fine_opt_decay)

    # inductive  training
    if not opt.transductive:

        if opt.train_stage == 'coarse':
            for epoch in (range(opt.coarse_epoch_num)):
                coarse_align_model = coarse_align_model.train()
                coarse_align_model = do_train(log=logger,
                                              HU=HU,
                                              writer=Writer,
                                              cfg=opt,
                                              model=coarse_align_model,
                                              epoch=epoch,
                                              optimizer=optimizer_coarse,
                                              seen_cls_idx=seen_class,
                                              unseen_cls_idx=unseen_class,
                                              train_dataloader=train_loader,
                                              test_seen_dataloader=test_s_loader,
                                              test_unseen_dataloader=test_u_loader,
                                              seen_class_indices_inside_test=data_loader.seen_class_indices_inside_test,
                                              unseen_class_indices_inside_test=data_loader.unseen_class_indices_inside_test)

                coarse_align_model.eval()

                if not os.path.exists(opt.coarse_output_pkl_dir):
                    os.makedirs(opt.coarse_output_pkl_dir)

                file_path = save_mat_data(cfg=opt,
                                          coarse_epoch=epoch,
                                          model=coarse_align_model,
                                          tr=train_tmp_loader,
                                          ts=test_s_loader,
                                          tu=test_u_loader,
                                          save_pkl_path=opt.coarse_output_pkl_dir)

        else:
            # prepare the pkl_dataset
            attr = data_loader.raw_attr
            train_mix_up_dataset, \
            test_seen_dataset, \
            test_unseen_dataset, \
            new_seen_class_indices_inside_test, \
            new_unseen_class_indices_inside_test, \
            seen_b_classes,\
            Class_Center = \
                make_mat_dataloader(cfg=opt,
                                    attr=attr,
                                    save_pkl_path=pkl_data,
                                    maml=False)
            fine_align_model.Class_Center = Class_Center.cuda()
            # print(t.load('/data/code/C2F_ZSL/saved_ck_model_coarse_mix/AWA2/mix_train_backbone/AWA2_coarse_backbone_11_lr_0.0005_decay_0.001_way_8_shot_2_iter_100.pt')['model'].keys())
            fine_align_model.V2S_W = t.load('/data/code/C2F_ZSL/saved_ck_model_coarse_mix/AWA2/mix_train_backbone/AWA2_coarse_backbone_11_lr_0.0005_decay_0.001_way_8_shot_2_iter_100.pt')['model']['V2S.weight']

            # breakpoint()

            for epoch in range(opt.fine_epoch_num):
                fine_align_model.train()

                fine_align_model = do_train(log=logger,
                                            HU=HU,
                                            writer=Writer,
                                            cfg=opt,
                                            model=fine_align_model,
                                            epoch=epoch,
                                            optimizer=optimizer_fine,
                                            seen_cls_idx=seen_class,
                                            unseen_cls_idx=unseen_class,
                                            train_dataloader=train_mix_up_dataset,
                                            test_seen_dataloader=test_seen_dataset,
                                            test_unseen_dataloader=test_unseen_dataset,
                                            seen_class_indices_inside_test=new_seen_class_indices_inside_test,
                                            unseen_class_indices_inside_test=new_unseen_class_indices_inside_test)

            info = f'current best::::::ZSL_U: {100.0*HU.U_MAX:.2f}:::::::::GZSL_H: {100.0*HU.H_MAX:.2f}::::::::GZSL_H_Epoch: {HU.Epoch}'
            print(info)
            logger.info(info)
    else:
        # transductive  training
        # initialize the seen_dataset to warm_up the training of s2v fine_align model
        # print('------------------', opt.train_stage)
        attr = data_loader.raw_attr

        train_mix_up_dataset_list = []
        train_mix_up_dataset, test_seen_dataset, test_unseen_dataset, \
        new_seen_class_indices_inside_test, \
        new_unseen_class_indices_inside_test, \
        seen_b_classes = make_mat_dataloader(cfg=opt,
                                             attr=attr,
                                             save_pkl_path=pkl_data,
                                             maml=False)
        train_mix_up_dataset_list.append(train_mix_up_dataset)
        # new hyper_parameters prepared for .yaml file
        # warm_up_epoch = 60 # 10 # 40
        # epoch_interval = 20 # 10  # 20

        for epoch in range(opt.fine_epoch_num):
            fine_align_model.train()
            if epoch >= cfg.warm_epoch and epoch % cfg.epoch_interval == 0:
                # if use pseudo labels to improve the  visual mebeddings,
                # then update the coarse training dataloader and update backbone
                # if opt.update_feature:
                #     train_loader = data_loader.forward()
                #     coarse_align_model = do_train(log=logger,
                #                                   HU=HU,
                #                                   writer=Writer,
                #                                   cfg=opt,
                #                                   model=coarse_align_model,
                #                                   epoch=epoch,
                #                                   optimizer=optimizer_coarse,
                #                                   seen_cls_idx=seen_class,
                #                                   unseen_cls_idx=unseen_class,
                #                                   train_dataloader=train_loader,
                #                                   test_seen_dataloader=test_s_loader,
                #                                   test_unseen_dataloader=test_u_loader,
                #                                   seen_class_indices_inside_test=data_loader.seen_class_indices_inside_test,
                #                                   unseen_class_indices_inside_test=data_loader.unseen_class_indices_inside_test)
                #     file_path = save_mat_data(cfg=opt,
                #                               coarse_epoch=epoch,
                #                               model=coarse_align_model,
                #                               tr=train_tmp_loader,
                #                               ts=test_s_loader,
                #                               tu=test_u_loader,
                #                               save_pkl_path=opt.coarse_output_pkl_dir)
                # else:
                file_path = pkl_data  # if do not update the visual backbone, then still the inductive learnt visual embeddings

                # model_dict = t.load('/data/code/C2F_ZSL/saved_model_final/CUB/for_tsne_visualize/Fine/best_s2v_pkl_data_CUB_0_data_ck.pkl_gama_1.0_temp_1.0_walk_num_6.pt')['model']

                train_mix_up_dataset_list = update_reliable_unseen_data(
                    cfg=opt,
                    saved_pkl_path=file_path,
                    s2v_model=fine_align_model,
                    all_attr=attr,
                    seen_class_id=data_loader.seen_class_big_idx,
                    unseen_class_id=data_loader.unseen_class_big_idx,
                    reliable_filter='ratio',
                    unseen_class_indices_inside_test=new_unseen_class_indices_inside_test
                )(model_dict=fine_align_model.state_dict())


                fine_align_model = do_train(log=logger,
                                            HU=HU,
                                            writer=Writer,
                                            cfg=opt,
                                            model=fine_align_model,
                                            epoch=epoch,
                                            optimizer=optimizer_fine,
                                            seen_cls_idx=seen_class,
                                            unseen_cls_idx=unseen_class,
                                            train_dataloader=train_mix_up_dataset_list,
                                            test_seen_dataloader=test_seen_dataset,
                                            test_unseen_dataloader=test_unseen_dataset,
                                            seen_class_indices_inside_test=new_seen_class_indices_inside_test,
                                            unseen_class_indices_inside_test=new_unseen_class_indices_inside_test)


            else:
                fine_align_model = do_train(log=logger,
                                            HU=HU,
                                            writer=Writer,
                                            cfg=opt,
                                            model=fine_align_model,
                                            epoch=epoch,
                                            optimizer=optimizer_fine,
                                            seen_cls_idx=seen_class,
                                            unseen_cls_idx=unseen_class,
                                            train_dataloader=train_mix_up_dataset_list,
                                            test_seen_dataloader=test_seen_dataset,
                                            test_unseen_dataloader=test_unseen_dataset,
                                            seen_class_indices_inside_test=new_seen_class_indices_inside_test,
                                            unseen_class_indices_inside_test=new_unseen_class_indices_inside_test)

        info = f'current best::::::ZSL_U: {100.0 * HU.U_MAX:.2f}:::::::::GZSL_H: {100.0 * HU.H_MAX:.2f}::::::::GZSL_H_Epoch: {HU.Epoch}'
        print(info)
        logger.info(info)



if __name__ == "__main__":

    t.multiprocessing.set_start_method('spawn')

    if platform.system() == 'Linux':
        cfg = parse_args(config_file='/data/code/LPL/config/awa2_hyp_para.yaml')


    if platform.system() == 'Linux':
        # main(opt=cfg, pkl_data='AWA2_448_4_data_lr_5e-4_decay_1e-3_iter_100_ck.pkl')
        main(opt=cfg, pkl_data='V2S_CLS_AWA2_448_11_lr_0.0005_decay_0.001_data.pkl')
