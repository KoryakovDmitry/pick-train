import os
import cv2
import numpy as np
from numpy import inf

import torch
import randomcolor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.util import iob_index_to_str, text_index_to_str
from utils.entities_list import Entities_list
from utils import inf_loop
from utils.metrics import MetricTracker, SpanBasedF1MetricTracker
from logger import TensorboardWriter
from utils.class_utils import iob_labels_vocab_cls
from utils.util import iob_tags_to_union_iob_tags
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans


class Trainer:
    """
    Trainer class
    """

    def __init__(self, model, optimizer, config, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, max_len_step=None):
        '''
        :param model:
        :param optimizer:
        :param config:
        :param data_loader:
        :param valid_data_loader:
        :param test_data_loader:
        :param lr_scheduler:
        :param max_len_step:  controls number of batches(steps) in each epoch.
        '''
        self.config = config
        self.distributed = config['distributed']
        if self.distributed:
            self.local_master = (config['local_rank'] == 0)
            self.global_master = (dist.get_rank() == 0)
        else:
            self.local_master = True
            self.global_master = True
        self.logger = config.get_logger('trainer', config['trainer']['log_verbosity']) if self.local_master else None

        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(config['local_rank'], config['local_world_size'])
        self.model = model.to(self.device)

        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        monitor_open = cfg_trainer['monitor_open']
        if monitor_open:
            self.monitor = cfg_trainer.get('monitor', 'off')
        else:
            self.monitor = 'off'

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.monitor_mode = 'off'
            self.monitor_best = 0
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = inf if self.monitor_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            self.early_stop = inf if self.early_stop == -1 else self.early_stop

        self.start_epoch = 1

        if self.local_master:
            self.checkpoint_dir = config.save_dir
            # setup visualization writer instance
            self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        # load checkpoint for resume training
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        # load checkpoint following load to multi-gpu, avoid 'module.' prefix
        if self.config['trainer']['sync_batch_norm'] and self.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        if self.distributed:
            self.model = DDP(self.model, device_ids=self.device_ids, output_device=self.device_ids[0],
                             find_unused_parameters=True)

        self.data_loader = data_loader
        if max_len_step is None:  # max length of iteration step of every epoch
            # epoch-based training
            self.len_step = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_step = max_len_step

        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        log_step = self.config['trainer']['log_step_interval']
        self.log_step = log_step if log_step != -1 and 0 < log_step < self.len_step else int(
            np.sqrt(data_loader.batch_size))

        val_step_interval = self.config['trainer']['val_step_interval']
        # self.val_step_interval = val_step_interval if val_step_interval!= -1 and 0 < val_step_interval < self.len_step\
        #                                             else int(np.sqrt(data_loader.batch_size))
        self.val_step_interval = val_step_interval

        self.gl_loss_lambda = self.config['trainer']['gl_loss_lambda']

        self.loss_metrics = MetricTracker('loss', 'gl_loss', 'crf_loss',
                                          writer=self.writer if self.local_master else None)
        other = [lbl + '_' + metric for lbl in Entities_list + ['overall'] for metric in ('mEP', 'mER', 'mEF', 'mEA')]
        self.f1_metrics = MetricTracker(*other,
                                        writer=self.writer if self.local_master else None)

        self.valid_f1_metrics = SpanBasedF1MetricTracker(iob_labels_vocab_cls)
        self.train_f1_metrics = SpanBasedF1MetricTracker(iob_labels_vocab_cls)
        self.test_f1_metrics = SpanBasedF1MetricTracker(iob_labels_vocab_cls)
        self.colors_cat = {"neg": [0, 0, 0]}
        rand_color = randomcolor.RandomColor(seed=42)
        colors_add = rand_color.generate(count=len(Entities_list), format_='Arrayrgb')
        for cat, col in zip(Entities_list, colors_add):
            self.colors_cat[cat] = col

        self.step_test = 0
        self.step_val = 0

    def train(self):
        """
        Full training logic, including train and validation.
        """

        if self.distributed:
            dist.barrier()  # Syncing machines before training

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            # ensure distribute worker sample different data,
            # set different random seed by passing epoch to sampler
            if self.distributed:
                self.data_loader.sampler.set_epoch(epoch)
            result_dict = self._train_epoch(epoch)

            # print logged informations to the screen
            if self.do_validation:
                val_result_dict = result_dict['val_result_dict']
                val_res = SpanBasedF1MetricTracker.dict2str(val_result_dict)
            else:
                val_res = ''
            # every epoch log information
            self.logger_info('[Epoch Validation] Epoch:[{}/{}] Total Loss: {:.6f} '
                             'GL_Loss: {:.6f} CRF_Loss: {:.6f} \n{}'.
                             format(epoch, self.epochs, result_dict['loss'],
                                    result_dict['gl_loss'] * self.gl_loss_lambda,
                                    result_dict['crf_loss'], val_res))

            # evaluate model performance according to configured metric, check early stop, and
            # save best checkpoint as model_best
            best = False
            if self.monitor_mode != 'off' and self.do_validation:
                best, not_improved_count = self._is_best_monitor_metric(best, not_improved_count, val_result_dict)
                if not_improved_count > self.early_stop:
                    self.logger_info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _is_best_monitor_metric(self, best, not_improved_count, val_result_dict):
        '''
        monitor metric
        :param best:
        :param not_improved_count:
        :param val_result_dict:
        :return:
        '''
        entity_name, metric = self.monitor_metric.split('-')
        val_monitor_metric_res = val_result_dict[entity_name][metric]
        try:
            # check whether model performance improved or not, according to specified metric(monitor_metric)
            improved = (self.monitor_mode == 'min' and val_monitor_metric_res <= self.monitor_best) or \
                       (self.monitor_mode == 'max' and val_monitor_metric_res >= self.monitor_best)
        except KeyError:
            self.logger_warning("Warning: Metric '{}' is not found. "
                                "Model performance monitoring is disabled.".format(self.monitor_metric))
            self.monitor_mode = 'off'
            improved = False
        if improved:
            self.monitor_best = val_monitor_metric_res
            not_improved_count = 0
            best = True
        else:
            not_improved_count += 1
        return best, not_improved_count

    def _train_epoch(self, epoch):
        '''
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log dict that contains average loss and metric in this epoch.
        '''
        self.model.train()
        self.loss_metrics.reset()
        self.f1_metrics.reset()
        self.train_f1_metrics.reset()
        # step iteration start
        for step_idx, input_data_item in enumerate(self.data_loader):
            step_idx += 1
            for key, input_value in input_data_item.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(self.device, non_blocking=True)
            if self.config['trainer']['anomaly_detection']:
                # This mode will increase the runtime and should only be enabled for debugging
                with torch.autograd.detect_anomaly():
                    self.optimizer.zero_grad()
                    # model forward
                    output = self.model(**input_data_item)
                    # calculate loss
                    gl_loss = output['gl_loss']
                    crf_loss = output['crf_loss']
                    logits = output['logits'].detach().clone()
                    new_mask = output['new_mask'].detach().clone()
                    total_loss = torch.sum(crf_loss) + self.gl_loss_lambda * torch.sum(gl_loss)
                    # backward
                    total_loss.backward()
                    # self.average_gradients(self.model)
                    self.optimizer.step()
            else:
                self.optimizer.zero_grad()
                # model forward
                output = self.model(**input_data_item)
                # calculate loss
                gl_loss = output['gl_loss']
                crf_loss = output['crf_loss']
                logits = output['logits'].detach().clone()
                new_mask = output['new_mask'].detach().clone()
                total_loss = torch.sum(crf_loss) + self.gl_loss_lambda * torch.sum(gl_loss)
                # backward
                total_loss.backward()
                # self.average_gradients(self.model)
                self.optimizer.step()

            # Use a barrier() to make sure that all process have finished forward and backward
            if self.distributed:
                dist.barrier()
                #  obtain the sum of all total_loss at all processes
                dist.all_reduce(total_loss, op=dist.reduce_op.SUM)

                size = dist.get_world_size()
            else:
                size = 1

            if hasattr(self.model, 'module'):
                #  List[(List[int], torch.Tensor)] contain the tag indices of the maximum likelihood tag sequence.
                #  and the score of the viterbi path.
                best_paths = self.model.module.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,
                                                                              logits_batch_first=True)
            else:
                best_paths = self.model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,
                                                                       logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)

            # calculate and update f1 metrics
            # (B, N*T, out_dim)
            predicted_tags_hard_prob = logits * 0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    predicted_tags_hard_prob[i, j, tag_id] = 1

            golden_tags = input_data_item['iob_tags_label']
            mask = input_data_item['mask']
            union_iob_tags = iob_tags_to_union_iob_tags(golden_tags, mask)

            if self.distributed:
                dist.barrier()
            self.train_f1_metrics.update(predicted_tags_hard_prob.long(), union_iob_tags, new_mask)

            gl_loss /= size  # averages gl_loss across the whole world
            crf_loss /= size  # averages crf_loss across the whole world

            # calculate average loss across the batch size
            avg_gl_loss = torch.mean(gl_loss)
            avg_crf_loss = torch.mean(crf_loss)
            avg_loss = avg_crf_loss + self.gl_loss_lambda * avg_gl_loss

            # update metrics
            step = (epoch - 1) * self.len_step + step_idx - 1
            self.writer.set_step(step) if self.local_master else None
            self.loss_metrics.update('loss', avg_loss.item())
            self.loss_metrics.update('gl_loss', avg_gl_loss.item() * self.gl_loss_lambda)
            self.loss_metrics.update('crf_loss', avg_crf_loss.item())
            f1_result_dict = self.train_f1_metrics.result()
            for lbl in f1_result_dict.keys():
                for metric in ('mEP', 'mER', 'mEF', 'mEA'):
                    k = lbl + '_' + metric
                    self.f1_metrics.update(k, f1_result_dict[lbl][metric])

            # log messages
            if step_idx % self.log_step == 0:
                self.logger_info('Train Epoch:[{}/{}] Step:[{}/{}] Total Loss: {:.6f} GL_Loss: {:.6f} CRF_Loss: {:.6f}'.
                                 format(epoch, self.epochs, step_idx, self.len_step,
                                        avg_loss.item(), avg_gl_loss.item() * self.gl_loss_lambda, avg_crf_loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            # do validation after val_step_interval iteration
            if self.do_validation and step_idx % self.val_step_interval == 0:
                val_result_dict = self._valid_epoch()
                self.logger_info('[Step Validation] Epoch:[{}/{}] Step:[{}/{}]  \n{}'.
                                 format(epoch, self.epochs, step_idx, self.len_step,
                                        SpanBasedF1MetricTracker.dict2str(val_result_dict)))
                # check if best metric, if true, then save as model_best checkpoint.
                best, not_improved_count = self._is_best_monitor_metric(False, 0, val_result_dict)
                if best:
                    self._save_checkpoint(epoch=epoch, iter=step, save_best=best)

            # decide whether continue iter
            if step_idx == self.len_step + 1:
                break

        # step iteration end

        # {'loss': avg_loss, 'gl_loss': avg_gl_loss, 'crf_loss': avg_crf_loss}
        log = self.loss_metrics.result()

        # do validation after training an epoch
        if self.do_validation:
            val_result_dict = self._valid_epoch()
            log['val_result_dict'] = val_result_dict

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def plot_test_imgs(self):
        self.model.eval()
        self.loss_metrics.reset()
        self.f1_metrics.reset()
        self.test_f1_metrics.reset()

        with torch.no_grad():
            for step_idx, input_data_item in enumerate(self.test_data_loader):
                for key, input_value in input_data_item.items():
                    if input_value is not None and isinstance(input_value, torch.Tensor):
                        input_data_item[key] = input_value.to(self.device, non_blocking=True)

                output = self.model(**input_data_item)
                logits = output['logits']  # (B, N*T, out_dim)
                new_mask = output['new_mask']

                gl_loss = output['gl_loss']
                crf_loss = output['crf_loss']

                if hasattr(self.model, 'module'):
                    #  List[(List[int], torch.Tensor)] contain the tag indices of the maximum likelihood tag sequence.
                    #  and the score of the viterbi path.
                    best_paths = self.model.module.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,
                                                                                  logits_batch_first=True)
                else:
                    best_paths = self.model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,
                                                                           logits_batch_first=True)
                predicted_tags = []
                for path, score in best_paths:
                    predicted_tags.append(path)

                text_segments = input_data_item['text_segments'].cpu().numpy()  # (B, num_boxes, T)
                boxes_coords = input_data_item['boxes_unresized']
                filenames = input_data_item['filenames']
                image_unresized = input_data_item['whole_image_unresized']
                batch_idxs_box = np.array([np.repeat(np.arange(i.shape[0]), i.shape[1]) for i in text_segments])
                text_segments_with_box = np.concatenate([text_segments.reshape(-1)[np.newaxis, :], batch_idxs_box])[
                                         np.newaxis, :]
                mask = input_data_item['mask']
                golden_tags = input_data_item['iob_tags_label']

                # List[(List[int], torch.Tensor)]
                if hasattr(self.model, 'module'):
                    #  List[(List[int], torch.Tensor)] contain the tag indices of the maximum likelihood tag sequence.
                    #  and the score of the viterbi path.
                    best_paths = self.model.module.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,
                                                                                  logits_batch_first=True)
                else:
                    best_paths = self.model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,

                                                                           logits_batch_first=True)
                predicted_tags = []
                for path, score in best_paths:
                    predicted_tags.append(path)

                self.writer.set_step(self.step_test, 'test') \
                    if self.local_master else None
                self.step_test += 1

                avg_gl_loss = torch.mean(gl_loss)
                avg_crf_loss = torch.mean(crf_loss)
                avg_loss = avg_crf_loss + self.gl_loss_lambda * avg_gl_loss
                self.loss_metrics.update('loss', avg_loss.item())
                self.loss_metrics.update('gl_loss', avg_gl_loss.item() * self.gl_loss_lambda)
                self.loss_metrics.update('crf_loss', avg_crf_loss.item())

                # calculate and update f1 metrics
                # (B, N*T, out_dim)
                predicted_tags_hard_prob = logits * 0
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        predicted_tags_hard_prob[i, j, tag_id] = 1

                union_iob_tags = iob_tags_to_union_iob_tags(golden_tags, mask)

                if self.distributed:
                    dist.barrier()
                self.test_f1_metrics.update(predicted_tags_hard_prob.long(), union_iob_tags, new_mask)

                # add histogram of model parameters to the tensorboard
                # for name, p in self.model.named_parameters():
                #     self.writer.add_histogram(name, p, bins='auto')

                f1_result_dict = self.test_f1_metrics.result()
                for lbl in f1_result_dict.keys():
                    for metric in ('mEP', 'mER', 'mEF', 'mEA'):
                        k = lbl + '_' + metric
                        self.f1_metrics.update(k, f1_result_dict[lbl][metric])

                # convert iob index to iob string
                decoded_tags_list = iob_index_to_str(predicted_tags)
                # union text as a sequence and convert index to string
                decoded_texts_list, union_boxes = text_index_to_str(text_segments_with_box, mask, self.device)

                boxes_coords = np.array(boxes_coords)

                for decoded_tags, decoded_texts, image, boxes, union_box, filename in zip(decoded_tags_list,
                                                                                          decoded_texts_list,
                                                                                          image_unresized,
                                                                                          boxes_coords,
                                                                                          union_boxes,
                                                                                          filenames,
                                                                                          ):

                    text = 'NER: \n'
                    # List[ Tuple[str, Tuple[int, int]] ]
                    spans = bio_tags_to_spans(decoded_tags, [])
                    spans = sorted(spans, key=lambda x: x[1][0])
                    entity_names = spans[0] if len(spans) != 0 else []
                    box_idx_pred = []
                    for entity_name, range_tuple in spans:
                        t = ''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1])
                        box_idx = set((map(lambda x: int(x), union_box[range_tuple[0]:range_tuple[1] + 1])))
                        box_idx_pred.append(tuple(box_idx)[0])
                        text += f'{entity_name}, box_idxs: {box_idx}, \ntext: \n{t}\n\n'

                    img_pred = image.copy()

                    for margin, (cat, color) in enumerate(self.colors_cat.items()):
                        img_pred = cv2.putText(img_pred, cat,
                                               org=(10, int(margin * 20 + 10)),
                                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=0.5,
                                               color=color,
                                               lineType=1)

                    for box_idx in range(boxes.shape[0]):
                        box = boxes[box_idx].astype(np.int32).reshape(-1, 2)
                        key = "neg"
                        rect = cv2.boundingRect(box)
                        x, y, w, h = rect
                        img_pred = cv2.putText(img_pred, str(box_idx),
                                               org=(int(x + w + 10), int(y + h + 10)),
                                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                               fontScale=0.5,
                                               color=(0, 0, 255),
                                               lineType=1)
                        if box_idx in box_idx_pred:
                            ent_idx = box_idx_pred.index(box_idx)
                            key = entity_names[ent_idx]
                        img_pred = cv2.polylines(img_pred, [box], True, color=self.colors_cat[key], thickness=2)

                    self.writer.add_image(os.path.basename(filename), img_pred[:, :, ::-1], self.step_test,
                                          dataformats="HWC")
                    self.writer.add_text(os.path.basename(filename), text, self.step_test)

    def _valid_epoch(self):
        '''
         Validate after training an epoch or regular step, this is a time-consuming procedure if validation data is big.
        :param step: Integer, current training epoch.
        :return: A dict that contains information about validation
        '''

        self.model.eval()
        self.loss_metrics.reset()
        self.f1_metrics.reset()
        self.valid_f1_metrics.reset()
        with torch.no_grad():
            for step_idx, input_data_item in enumerate(self.valid_data_loader):
                for key, input_value in input_data_item.items():
                    if input_value is not None and isinstance(input_value, torch.Tensor):
                        input_data_item[key] = input_value.to(self.device, non_blocking=True)

                output = self.model(**input_data_item)
                logits = output['logits']
                new_mask = output['new_mask']

                gl_loss = output['gl_loss']
                crf_loss = output['crf_loss']

                if hasattr(self.model, 'module'):
                    #  List[(List[int], torch.Tensor)] contain the tag indices of the maximum likelihood tag sequence.
                    #  and the score of the viterbi path.
                    best_paths = self.model.module.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,
                                                                                  logits_batch_first=True)
                else:
                    best_paths = self.model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask,
                                                                           logits_batch_first=True)
                predicted_tags = []
                for path, score in best_paths:
                    predicted_tags.append(path)

                self.writer.set_step(self.step_val, 'valid') \
                    if self.local_master else None
                self.step_val += 1

                avg_gl_loss = torch.mean(gl_loss)
                avg_crf_loss = torch.mean(crf_loss)
                avg_loss = avg_crf_loss + self.gl_loss_lambda * avg_gl_loss
                self.loss_metrics.update('loss', avg_loss.item())
                self.loss_metrics.update('gl_loss', avg_gl_loss.item() * self.gl_loss_lambda)
                self.loss_metrics.update('crf_loss', avg_crf_loss.item())

                # calculate and update f1 metrics
                # (B, N*T, out_dim)
                predicted_tags_hard_prob = logits * 0
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        predicted_tags_hard_prob[i, j, tag_id] = 1

                golden_tags = input_data_item['iob_tags_label']
                mask = input_data_item['mask']
                union_iob_tags = iob_tags_to_union_iob_tags(golden_tags, mask)

                if self.distributed:
                    dist.barrier()
                self.valid_f1_metrics.update(predicted_tags_hard_prob.long(), union_iob_tags, new_mask)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        f1_result_dict = self.valid_f1_metrics.result()
        for lbl in f1_result_dict.keys():
            for metric in ('mEP', 'mER', 'mEF', 'mEA'):
                k = lbl + '_' + metric
                self.f1_metrics.update(k, f1_result_dict[lbl][metric])

        # test imgs
        self.plot_test_imgs()

        # rollback to train mode
        self.model.train()

        return f1_result_dict

    def average_gradients(self, model):
        '''
        Gradient averaging
        :param model:
        :return:
        '''
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

    def logger_info(self, msg):
        self.logger.info(msg) if self.local_master else None

    def logger_warning(self, msg):
        self.logger.warning(msg) if self.local_master else None

    def _prepare_device(self, local_rank, local_world_size):
        '''
        setup GPU device if available, move model into configured device
        :param local_rank:
        :param local_world_size:
        :return:
        '''
        if self.distributed:
            ngpu_per_process = torch.cuda.device_count() // local_world_size
            device_ids = list(range(local_rank * ngpu_per_process, (local_rank + 1) * ngpu_per_process))

            if torch.cuda.is_available() and local_rank != -1:
                torch.cuda.set_device(device_ids[0])  # device_ids[0] =local_rank if local_world_size = n_gpu per node
                device = 'cuda'
                self.logger_info(
                    f"[Process {os.getpid()}] world_size = {dist.get_world_size()}, "
                    + f"rank = {dist.get_rank()}, n_gpu/process = {ngpu_per_process}, device_ids = {device_ids}"
                )
            else:
                self.logger_warning('Training will be using CPU!')
                device = 'cpu'
            device = torch.device(device)
            return device, device_ids
        else:
            n_gpu = torch.cuda.device_count()
            n_gpu_use = local_world_size
            if n_gpu_use > 0 and n_gpu == 0:
                self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                    "training will be performed on CPU.")
                n_gpu_use = 0
            if n_gpu_use > n_gpu:
                self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                    "on this machine.".format(n_gpu_use, n_gpu))
                n_gpu_use = n_gpu

            list_ids = list(range(n_gpu_use))
            if n_gpu_use > 0:
                torch.cuda.set_device(list_ids[0])  # only use first available gpu as devices
                self.logger_warning(f'Training is using GPU {list_ids[0]}!')
                device = 'cuda'
            else:
                self.logger_warning('Training is using CPU!')
                device = 'cpu'
            device = torch.device(device)
            return device, list_ids

    def _save_checkpoint(self, epoch, iter=None, save_best=False):
        '''
        Saving checkpoints
        :param epoch:  current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        :return:
        '''
        iter = iter if iter else epoch
        # only local master process do save model
        if not self.local_master:
            return

        if hasattr(self.model, 'module'):
            arch = type(self.model.module).__name__
            state_dict = self.model.module.state_dict()
        else:
            arch = type(self.model).__name__
            state_dict = self.model.state_dict()
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best_eph{}.pth'.format(epoch))
            torch.save(state, best_path)
            self.logger_info(f"Saving current best: model_best_eph{epoch}.pth ...")
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger_info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        '''
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        :return:
        '''
        resume_path = str(resume_path)
        self.logger_info("Loading checkpoint: {} ...".format(resume_path))
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % self.config['local_rank']}
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model_arch'] != self.config['model_arch']:
            self.logger_warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger_warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger_info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
