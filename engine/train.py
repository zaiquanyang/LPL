import torch
from utils.tool import *
import tqdm
import scipy.io as sio


def do_train(log,
             HU,
             writer,
             cfg,
             model,
             epoch,
             optimizer,
             seen_cls_idx,
             unseen_cls_idx,
             train_dataloader,
             test_seen_dataloader,
             test_unseen_dataloader,
             seen_class_indices_inside_test,
             unseen_class_indices_inside_test):

    
    model.train()

    start = time.time()
    if cfg.train_stage == 'Fine':
        for i in (range(cfg.fine_iter_num)):
            
            
            img_batch, att_batch, meta_label_batch, label_batch = train_dataloader[0].__getitem__(i)
               
                
            real_feat = img_batch
            real_attr = att_batch
            real_meta_label = meta_label_batch
            real_label = label_batch

            real_feat = real_feat.to(cfg.device)
            real_attr = real_attr.to(cfg.device)
            real_meta_label = real_meta_label.to(cfg.device)
            real_label = real_label.to(cfg.device)
            # att_batch = F.normalize(att_batch, p=2, dim=-1, eps=1e-12)
            seen_loss, CL_loss = model(x_fea=real_feat, x_attr=real_attr, x_meta_label=real_meta_label, x_label=real_label, epoch=epoch)

            loss = seen_loss + CL_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # writer.add_scalar('fine_loss/total_loss', loss.item(), epoch * cfg.fine_iter_num + i)
            writer.add_scalar('Train/fine_seen_loss', seen_loss.item(), epoch * cfg.fine_iter_num + i)
            writer.add_scalar('Train/CL_loss', CL_loss.item(), epoch * cfg.fine_iter_num + i)

            # if len(train_dataloader) > 1:
            #     img_batch, att_batch, meta_label_batch, label_batch = train_dataloader[1].__getitem__(i)
            #     real_feat = img_batch
            #     real_attr = att_batch
            #     real_meta_label = meta_label_batch
            #     real_label = label_batch
            #
            #     real_feat = real_feat.to(cfg.device)
            #     real_attr = real_attr.to(cfg.device)
            #     real_meta_label = real_meta_label.to(cfg.device)
            #     real_label = real_label
            #     # att_batch = F.normalize(att_batch, p=2, dim=-1, eps=1e-12)
            #     seen_loss = model(x_fea=real_feat, x_attr=real_attr, x_meta_label=real_meta_label,
            #                       x_label=real_label, show=True if i == 0 else False, epoch=0)
            #
            #     loss = seen_loss
            #
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()


    else:

        for i, batch in tqdm.tqdm(enumerate(train_dataloader)):

            img_batch, att_batch, label_batch = batch[0], batch[1], batch[2]

            img_batch = img_batch.to(cfg.device)
            label_batch = label_batch.to(cfg.device)
            att_batch = att_batch.to(cfg.device)

            # patch_batch, patch_label = split_patch(img_batch, label_batch)

            s2v_loss = model(x_img=img_batch, x_attr=att_batch, x_label=label_batch)
            # grads
            # x_4 [16, 2048, 14, 14]
            # x_3 [16, 2048, 28, 28]
            # x_2 [16, 2048, 56, 56]
            # x_1 [16, 2048, 112, 112]
            # x_0 [16, 2048, 224, 224]


            writer.add_scalar('v2s_loss', s2v_loss.item(), epoch * cfg.coarse_iter_num + i)

            loss_1 = s2v_loss
            optimizer.zero_grad()
            loss_1.backward()
            for name, parms in model.named_parameters():      # V2S.0.weight
                # print(name)
                if name == 'backbone.layer4.2.conv3.weight':
                # if name == 'S2V.0.weight':
                # if name == 'S2V.weight':
                #     print(parms.size())
                #     print(parms.mean(dim=1).norm(p=2).item()*10)
                #     print(parms.mean(dim=1).std().item()*10)
                    writer.add_scalar('v2s_w_mean', parms.mean(dim=1).norm(p=2).item()*10, epoch * cfg.coarse_iter_num + i)
                    writer.add_scalar('v2s_w_std', parms.mean(dim=1).std().item()*10, epoch * cfg.coarse_iter_num + i)
            # breakpoint()
            optimizer.step()

            # breakpoint()

            # patch_v2s_loss, patch_kl_loss = model(x_img=patch_batch, x_attr=att_batch, x_label=patch_label, patch=True)  # 这里输入的att_batch其实大小是不对的，但是无妨训练
            #
            # writer.add_scalar('patch_v2s_loss', patch_v2s_loss.item(), epoch * cfg.coarse_iter_num + i)
            # writer.add_scalar('patch_kl_loss', patch_kl_loss.item(), epoch * cfg.coarse_iter_num + i)
            #
            # loss_2 = patch_v2s_loss - cfg.Lambda * patch_kl_loss
            # optimizer.zero_grad()
            # loss_2.backward()
            # optimizer.step()

            # if i % (cfg.coarse_iter_num / 300) == 0:
            #     # info = f'Cls_loss: {v2s_loss:.4f}:::Adv_loss: {adv_loss:.4f}'
            #     # print(info)
            #     print(grads['x_fea'].mean(dim=0).squeeze().norm(p=2)*10000)
            #     print(grads['pre_attr'].mean(dim=0).squeeze().std()*10000)
            #     # log.info(info)

    end = time.time()
    # H, U = HU.H_MAX, HU.U_MAX
    # evaluate the model !!!!
    if cfg.key_word=='label_finetune' and cfg.train_stage=='coarse':
        return model
    GH, ZU, GS, GU, ZS= evaluate(log=log,
             cfg=cfg,
             tr_time=start-end,
             epoch=epoch,
             model=model,
             seen_cls_idx=seen_cls_idx,
             unseen_cls_idx=unseen_cls_idx,
             ts_dataloader=test_seen_dataloader,
             tu_dataloader=test_unseen_dataloader,
             seen_class_indices_inside_test=seen_class_indices_inside_test,
             unseen_class_indices_inside_test=unseen_class_indices_inside_test)

    writer.add_scalar('Test/GZSL_H', GH, epoch)
    writer.add_scalar('Test/GZSL_S', GS, epoch)
    writer.add_scalar('Test/GZSL_U', GU, epoch)
    writer.add_scalar('Test/ZSL_U', ZU, epoch)
    writer.add_scalar('Test/ZSL_S', ZS, epoch)


    if cfg.train_stage == 'coarse':
        if not os.path.exists(cfg.save_model_dir):
            os.makedirs(cfg.save_model_dir)
        dir_ = os.path.join(cfg.save_model_dir, cfg.dataset_name)
        if not os.path.exists(dir_): os.makedirs(dir_)
        dir_ = os.path.join(dir_, cfg.key_word)
        if not os.path.exists(dir_): os.makedirs(dir_)
        dir_ = os.path.join(dir_, cfg.train_stage)
        if not os.path.exists(dir_): os.makedirs(dir_)
        f_name = os.path.join(dir_, '{}_lr_{}_decay_{}_way_{}_shot_{}_iter_{}.pt'.format(
            epoch, cfg.coarse_lr, cfg.coarse_opt_decay, cfg.ways, cfg.shots, cfg.coarse_iter_num))
        save_model(main_epoch=epoch, sub_epoch=epoch, cfg=cfg, model=model, U=ZU, f_name=f_name)
        time.sleep(5)
        print('save to file {}'.format(f_name))


    if cfg.train_stage == 'Fine':
        if not os.path.exists(cfg.save_model_dir):
            os.makedirs(cfg.save_model_dir)
        dir_ = os.path.join(cfg.save_model_dir, cfg.dataset_name)
        if not os.path.exists(dir_): os.makedirs(dir_)
        dir_ = os.path.join(dir_, cfg.key_word)
        if not os.path.exists(dir_): os.makedirs(dir_)
        dir_ = os.path.join(dir_, cfg.train_stage)
        if not os.path.exists(dir_): os.makedirs(dir_)
        f_name = os.path.join(dir_, 'best_s2v_pkl_data_{}_gama_{}_temp_{}_walk_num_{}.pt'.format(cfg.pkl_data.split('/')[-1],
                                                                         cfg.gama, cfg.temp, cfg.walk_num))
        if GH > HU.H_MAX:
            save_model(main_epoch=epoch, sub_epoch=epoch, cfg=cfg, model=model, U=ZU, f_name=f_name)
            time.sleep(2)

    HU.Update(GH, ZU, epoch)

    return model