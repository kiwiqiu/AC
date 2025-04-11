import os
import platform
import time
from datetime import timedelta
from unbalanced_loss.focal_loss import MultiFocalLoss
from unbalanced_loss.loss_PT import DiceLoss
import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from tqdm import tqdm
from loguru import logger
from visualdl import LogWriter

from macls.data_utils.collate_fn import collate_fn
from macls.data_utils.featurizer import AudioFeaturizer
from macls.data_utils.reader import MAClsDataset
from macls.metric.metrics import accuracy
from macls.models import build_model
from macls.optimizer import build_optimizer, build_lr_scheduler
from macls.utils.checkpoint import load_pretrained, load_checkpoint, save_checkpoint
from macls.utils.utils import dict_to_object, plot_confusion_matrix, print_arguments

# VERSION 1
# VERSION 2
class MAClsTrainer(object):
    def __init__(self, configs, use_gpu=True, data_augment_configs=None):
        """ macls集成工具类

        :param configs: 配置字典
        :param use_gpu: 是否使用GPU训练模型
        :param data_augment_configs: 数据增强配置字典或者其文件路径
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        self.use_gpu = use_gpu
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.audio_featurizer = None
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.amp_scaler = None
        # 添加蒸馏参数（应在配置文件中定义）
        self.temperature = self.configs.train_conf.distillation.temperature
        self.alpha = self.configs.train_conf.distillation.alpha
        self.beta = self.configs.train_conf.distillation.beta
        print("self.beta type",type(self.beta))
        print("self.beta temperature",type(self.temperature))
        print("self.alpha type",type(self.alpha))
        # 读取数据增强配置文件
        if isinstance(data_augment_configs, str):
            with open(data_augment_configs, 'r', encoding='utf-8') as f:
                data_augment_configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=data_augment_configs, title='数据增强配置')
        self.data_augment_configs = dict_to_object(data_augment_configs)
        # 获取分类标签
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]
        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')
        if self.configs.preprocess_conf.get('use_hf_model', False):
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('使用HuggingFace模型不支持多线程进行特征提取，已自动关闭！')
        self.max_step, self.train_step = None, None
        self.train_loss, self.train_acc = None, None
        self.train_eta_sec = None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        self.stop_train, self.stop_eval = False, False

    def __setup_dataloader(self, is_train=False):
        """ 获取数据加载器

        :param is_train: 是否获取训练数据
        """
        # 获取特征器
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                use_hf_model=self.configs.preprocess_conf.get('use_hf_model', False),
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))

        dataset_args = self.configs.dataset_conf.get('dataset', {})
        data_loader_args = self.configs.dataset_conf.get('dataLoader', {})
        if is_train:
            self.train_dataset = MAClsDataset(data_list_path=self.configs.dataset_conf.train_list,
                                              audio_featurizer=self.audio_featurizer,
                                              aug_conf=self.data_augment_configs,
                                              mode='train',
                                              **dataset_args)
            # 设置支持多卡训练
            train_sampler = RandomSampler(self.train_dataset)
            if torch.cuda.device_count() > 1:
                # 设置支持多卡训练
                train_sampler = DistributedSampler(dataset=self.train_dataset)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=collate_fn,
                                           sampler=train_sampler,
                                           **data_loader_args)
        # 获取测试数据
        data_loader_args.drop_last = False
        dataset_args.max_duration = self.configs.dataset_conf.eval_conf.max_duration
        data_loader_args.batch_size = self.configs.dataset_conf.eval_conf.batch_size
        self.test_dataset = MAClsDataset(data_list_path=self.configs.dataset_conf.test_list,
                                         audio_featurizer=self.audio_featurizer,
                                         mode='eval',
                                         **dataset_args)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      **data_loader_args)

    def extract_features(self, save_dir='dataset/features', max_duration=100):
        """ 提取特征保存文件

        :param save_dir: 保存路径
        :param max_duration: 提取特征的最大时长，避免过长显存不足，单位秒
        """
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                use_hf_model=self.configs.preprocess_conf.get('use_hf_model', False),
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        dataset_args = self.configs.dataset_conf.get('dataset', {})
        dataset_args.max_duration = max_duration
        data_loader_args = self.configs.dataset_conf.get('dataLoader', {})
        data_loader_args.drop_last = False   # 确保完整数据
        # data_loader_args['shuffle'] = False  # 强制关闭随机排序
        for data_list in [self.configs.dataset_conf.train_list, self.configs.dataset_conf.test_list]:
            test_dataset = MAClsDataset(data_list_path=data_list,
                                        audio_featurizer=self.audio_featurizer,
                                        mode='extract_feature',
                                        **dataset_args)
            all_data_paths = test_dataset.data_paths # 关键点：获取完整路径列表
            test_loader = DataLoader(dataset=test_dataset,
                                     collate_fn=collate_fn,
                                     shuffle=False,
                                     **data_loader_args)
            # 在处理前验证路径数量匹配
            total_samples = len(test_dataset)
            if len(all_data_paths) != total_samples:
                raise RuntimeError("数据路径列表与样本数量不匹配")

            # 新的音频特征数据命名，和之前的名字一样确保唯一性，之前的命名方式以时间戳，可能会出现重复覆盖的错误，导致预处理后的数据量减少
            current_idx = 0
            save_data_list = data_list.replace('.txt', '_features.txt')
            with open(save_data_list, 'w', encoding='utf-8') as f:
                for batch in tqdm(test_loader):
                    features, labels, input_lens = batch
                    actual_batch_size = len(features)  # 动态获取实际批次大小

                    # 获取对应原始路径段
                    batch_paths = all_data_paths[current_idx: current_idx + actual_batch_size]
                    current_idx += actual_batch_size

                    # 处理每个样本
                    for i in range(actual_batch_size):
                        # 生成唯一文件名（基于原始路径）
                        original_name = os.path.splitext(
                            os.path.basename(batch_paths[i]))[0]
                        save_path = os.path.join(
                            save_dir,
                            str(labels[i].item()),
                            f"{original_name}.npy"
                        ).replace('\\', '/')

                        # 保存文件与记录路径
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        np.save(save_path, features[i].numpy()[:input_lens[i]])
                        f.write(f"{save_path}\t{labels[i].item()}\n")
            logger.info(f'{data_list}列表中的数据已提取特征完成，新列表为：{save_data_list}')
            logger.info(f"当前批次尺寸: {actual_batch_size} | 累计处理: {current_idx}/{total_samples}")

            ### 旧的加载数据特征
            # save_data_list = data_list.replace('.txt', '_features.txt')
            # with open(save_data_list, 'w', encoding='utf-8') as f:
            #     for features, labels, input_lens in tqdm(test_loader):
            #         for i in range(len(features)):
            #             feature, label, input_len = features[i], labels[i], input_lens[i]
            #             feature = feature.numpy()[:input_len]
            #             label = int(label)
            #             save_path = os.path.join(save_dir, str(label),
            #                                      f'{int(time.time() * 1000)}.npy').replace('\\', '/')
            #             os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #             np.save(save_path, feature)
            #             f.write(f'{save_path}\t{label}\n')
            # logger.info(f'{data_list}列表中的数据已提取特征完成，新列表为：{save_data_list}')

    def __setup_model(self, input_size, is_train=False):
        """ 获取模型

        :param input_size: 模型输入特征大小
        :param is_train: 是否获取训练模型
        """
        # 自动获取列表数量
        if self.configs.model_conf.model_args.get('num_class', None) is None:
            self.configs.model_conf.model_args.num_class = len(self.class_labels)
        # 获取模型 build_model
        self.model = build_model(input_size=input_size, configs=self.configs)
        # 打印模型信息，98是长度，这个取决于输入的音频长度
        # self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        summary(self.model, input_size=(1, 98, input_size))
        # 使用Pytorch2.0的编译器
        if self.configs.train_conf.use_compile and torch.__version__ >= "2" and platform.system().lower() == 'windows':
            self.model = torch.compile(self.model, mode="reduce-overhead")
        # print(self.model)
        # 获取损失函数
        # 原始交叉熵CE
        label_smoothing = self.configs.train_conf.get('label_smoothing', 0.0)
        # self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # focal loss， V1:alpha=1,V2:alpha=[1,1,1,2,2,2,1,3],V3:alpha=[1,1,1,1,1,3,1,4],V4:alpha=[2,1,1,1,1,4,1,5],V5:alpha=[1,1,1,1,1,2,1]
        self.loss = MultiFocalLoss(num_class=self.configs.model_conf.model_args.num_class, gamma=2, reduction='mean')

        # dice loss
        # self.loss = DiceLoss(reduction='mean')

        # 分层损失策略：仅深层使用Focal Loss，中间层用标准CE
        self.focal_loss = MultiFocalLoss(num_class=self.configs.model_conf.model_args.num_class, gamma=2, reduction='mean')
        self.ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if is_train:
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.GradScaler(init_scale=1024)
            # 获取优化方法
            self.optimizer = build_optimizer(params=self.model.parameters(), configs=self.configs)
            # 学习率衰减函数
            self.scheduler = build_lr_scheduler(optimizer=self.optimizer, step_per_epoch=len(self.train_loader),
                                                configs=self.configs)

    def kd_loss(self, student_output, teacher_output):
        """知识蒸馏损失"""
        soft_teacher = torch.softmax(teacher_output / self.temperature, dim=1)
        log_student = torch.log_softmax(student_output / self.temperature, dim=1)
        loss_kd = -torch.mean(torch.sum(soft_teacher * log_student, dim=1))
        return loss_kd
        # return -(soft_teacher * log_student).sum(dim=1).mean() * (self.temperature ** 2)

    def feature_loss(self, student_feature, teacher_feature):
        """特征对齐损失"""
        return torch.mean(torch.square(student_feature - teacher_feature.detach()))

    # 课程学习策略：初期侧重分类，后期逐步增加蒸馏
    def get_current_weight(self, epoch, max_epoch, base_weight):
        return base_weight * (epoch / max_epoch) ** 2  # 二次曲线增长

    def __train_epoch(self, epoch_id, local_rank, writer, nranks=0):
        """训练一个epoch

        :param epoch_id: 当前epoch
        :param local_rank: 当前显卡id
        :param writer: VisualDL对象
        :param nranks: 所使用显卡的数量
        """
        train_times, accuracies, loss_sum = [], [], []
        # 初始化各层指标容器
        metrics = {
            'final_acc': [],
            'mid1_acc': [], 'mid2_acc': [], 'mid3_acc': [],
            'kd_loss': [], 'feature_loss': []
        }
        start = time.time()
        for batch_id, (features, label, input_len) in enumerate(self.train_loader):
            if self.stop_train: break
            if nranks > 1:
                features = features.to(local_rank)
                label = label.to(local_rank).long()
            else:
                features = features.to(self.device)
                label = label.to(self.device).long()
            # 执行模型计算，是否开启自动混合精度
            with torch.autocast('cuda', enabled=self.configs.train_conf.enable_amp):
                # 获取所有中间输出和特征
                output, middle1_out, middle2_out, middle3_out, \
                    finnal_fea, middle1_fea, middle2_fea, middle3_fea = self.model(features)
            # 计算损失值
            # 计算所有分类损失
            # los = self.loss(output, label)
            final_loss = self.focal_loss(output, label)
            middle1_loss = self.ce_loss(middle1_out, label)
            middle2_loss = self.ce_loss(middle2_out, label)
            middle3_loss = self.ce_loss(middle3_out, label)
            # cls_loss = final_loss + 0.3 * middle1_loss + 0.3 * middle2_loss + 0.3 * middle3_loss
            cls_loss = final_loss + middle1_loss + middle2_loss + middle3_loss
            # 计算蒸馏损失
            loss1_kd = self.kd_loss(middle1_out, output)
            loss2_kd = self.kd_loss(middle2_out, output)
            loss3_kd = self.kd_loss(middle3_out, output)
            loss_kd = loss1_kd + loss2_kd + loss3_kd
            # 计算特征对齐损失
            loss_fea1 = self.feature_loss(middle1_fea, finnal_fea)
            loss_fea2 = self.feature_loss(middle2_fea, finnal_fea)
            loss_fea3 = self.feature_loss(middle3_fea, finnal_fea)
            loss_fea = loss_fea1 + loss_fea2 + loss_fea3
            # 组合总损失v1
            total_loss = cls_loss * (1 - self.alpha) + \
                          loss_kd * self.alpha + \
                          loss_fea * self.beta
            # 组合总损失v2 线性衰减分类损失权重
            # max_epoch = self.configs.train_conf.max_epoch
            # initial_alpha = self.alpha
            # current_alpha = initial_alpha * (1 - epoch_id / max_epoch)
            # total_loss = cls_loss * (1 - current_alpha) + loss_kd * current_alpha + loss_fea * self.beta
            # 组合总损失v3
            # max_epoch= self.configs.train_conf.max_epoch
            # initial_alpha = self.alpha
            # initial_beta = self.beta
            # current_alpha = self.get_current_weight(epoch_id, max_epoch, initial_alpha)
            # current_beta = self.get_current_weight(epoch_id, max_epoch, initial_beta)
            # total_loss = cls_loss * (1-current_alpha) + current_alpha * loss_kd + current_beta * loss_fea
            # 是否开启自动混合精度,反向传播
            if self.configs.train_conf.enable_amp:
                # loss缩放，乘以系数loss_scaling
                scaled = self.amp_scaler.scale(total_loss)
                scaled.backward()
            else:
                total_loss.backward()
            # 是否开启自动混合精度，优化步骤
            if self.configs.train_conf.enable_amp:
                self.amp_scaler.unscale_(self.optimizer)
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

            # 计算各个batch的准确率
            batch_metrics = {
                'final_acc': accuracy(output, label),
                'mid1_acc': accuracy(middle1_out, label),
                'mid2_acc': accuracy(middle2_out, label),
                'mid3_acc': accuracy(middle3_out, label),
                'kd_loss': loss_kd.item(),
                'feature_loss': loss_fea.item()
            }
            # 整合所有batch的acc和loss
            for k, v in batch_metrics.items():
                metrics[k].append(v)
            loss_sum.append(total_loss.item())

            train_times.append((time.time() - start) * 1000)
            self.train_step += 1

            # 多卡训练只使用一个进程打印，log_interval=10，每10个batch进行打印
            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:
                batch_id = batch_id + 1
                # 所有acc的平均值
                avg_metrics = {k: np.mean(v[-self.configs.train_conf.log_interval:])
                               for k, v in metrics.items()}
                # 计算每秒训练数据量
                train_speed = self.configs.dataset_conf.dataLoader.batch_size / (
                        sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                self.train_eta_sec = (sum(train_times) / len(train_times)) * (self.max_step - self.train_step) / 1000
                eta_str = str(timedelta(seconds=int(self.train_eta_sec)))

                # 构造日志信息
                self.train_loss = sum(loss_sum) / len(loss_sum)
                self.train_acc = avg_metrics['final_acc']

                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {self.train_loss:.5f}, '
                            f'accuracy: {self.train_acc:.2%}, '
                            f"mid acc:{avg_metrics['mid1_acc']:.2%}/{avg_metrics['mid2_acc']:.2%}/{avg_metrics['mid3_acc']:.2%}"
                            f'KD loss: {avg_metrics["kd_loss"]:.3f}, '
                            f'Fea loss: {avg_metrics["feature_loss"]:.3f}, '
                            f'learning rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                # 记录到visualdl
                writer.add_scalar('Train/Loss', self.train_loss, self.train_log_step)
                writer.add_scalar('Train/Accuracy', self.train_acc, self.train_log_step)
                writer.add_scalars('Train/Accuracy', {
                    'final': avg_metrics['final_acc'],
                    'mid1': avg_metrics['mid1_acc'],
                    'mid2': avg_metrics['mid2_acc'],
                    'mid3': avg_metrics['mid3_acc']
                }, self.train_step)
                writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], self.train_log_step)
                writer.add_scalars('Train/SelfDistill', {
                    'kd_loss': avg_metrics['kd_loss'],
                    'feature_loss': avg_metrics['feature_loss']
                }, self.train_step)
                # 清空所有列表
                train_times, accuracies, loss_sum = [], [], []
                for k in metrics:
                    metrics[k].clear()
                self.train_log_step += 1
            start = time.time()
            self.scheduler.step()

    def train(self,
              save_model_path='models/',
              log_dir='log/',
              resume_model=None,
              pretrained_model=None):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param log_dir: 保存VisualDL日志文件的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        """
        # 创建结果保存路径
        result_dir = os.path.join(log_dir, "elevator_results")
        os.makedirs(result_dir, exist_ok=True)
        result_csv = os.path.join(result_dir, 'Elevator_ResNetSE_SD_totalLossV2.1_epoch60_lr3e-4_alpha0.8_dataset3.3.csv')
        # 初始化 CSV 表头（如果文件不存在）
        if not os.path.exists(result_csv):
            with open(result_csv, 'w', encoding='utf-8') as f:
                f.write("epoch,loss,final_acc,mid1_acc,mid2_acc,mid3_acc\n")
        # 获取有多少张显卡训练
        nranks = torch.cuda.device_count()
        local_rank = 0
        writer = None
        if local_rank == 0:
            # 日志记录器
            writer = LogWriter(logdir=log_dir)

        if nranks > 1 and self.use_gpu:
            # 初始化NCCL环境
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ["LOCAL_RANK"])

        # 获取数据
        self.__setup_dataloader(is_train=True)
        # 获取模型
        self.__setup_model(input_size=self.audio_featurizer.feature_dim, is_train=True)
        # 加载预训练模型
        self.model = load_pretrained(model=self.model, pretrained_model=pretrained_model)
        # 加载恢复模型
        self.model, self.optimizer, self.amp_scaler, self.scheduler, last_epoch, best_acc = \
            load_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                            amp_scaler=self.amp_scaler, scheduler=self.scheduler, step_epoch=len(self.train_loader),
                            save_model_path=save_model_path, resume_model=resume_model)

        # 支持多卡训练
        if nranks > 1 and self.use_gpu:
            self.model.to(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        logger.info('训练数据：{}'.format(len(self.train_dataset)))

        self.train_loss, self.train_acc = None, None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], last_epoch)
        # 最大步数
        self.max_step = len(self.train_loader) * self.configs.train_conf.max_epoch
        self.train_step = max(last_epoch, 0) * len(self.train_loader)
        # 开始训练
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            if self.stop_train: break
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            self.__train_epoch(epoch_id=epoch_id, local_rank=local_rank, writer=writer, nranks=nranks)
            # 多卡训练只使用一个进程执行评估和保存模型，多卡训练时仅主进程验证：通过 local_rank == 0 避免重复计算。
            if local_rank == 0:
                if self.stop_eval: continue
                logger.info('=' * 70)
                # 每个epoch结束后验证一次
                self.eval_loss, self.eval_acc, precision, recall, f1, mid1_acc, mid2_acc, mid3_acc = self.evaluate()
                logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, final_accuracy: {:.2%}\n'
                            'mid1_accuracy: {:.2%}, mid2_accuracy: {:.2%}, mid3_accuracy: {:.2%}'
                            .format(
                    epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), self.eval_loss, self.eval_acc, mid1_acc, mid2_acc, mid3_acc))
                logger.info('=' * 70)

                writer.add_scalar('Test/Accuracy', self.eval_acc, self.test_log_step)
                writer.add_scalar('Test/Loss', self.eval_loss, self.test_log_step)
                writer.add_scalars('Test/Mid_Accuracy', {'mid1': mid1_acc,'mid2': mid2_acc,'mid3': mid3_acc }, self.test_log_step)

                # 保存到本地 CSV 文件
                with open(result_csv, 'a', encoding='utf-8') as f:
                    f.write(
                        f"{epoch_id},"
                        f"{self.eval_loss:.5f},"
                        f"{self.eval_acc:.5f},"
                        f"{mid1_acc:.5f},"
                        f"{mid2_acc:.5f},"
                        f"{mid3_acc:.5f}\n"
                    )

                self.test_log_step += 1
                self.model.train()
                # 保存最优模型
                if self.eval_acc >= best_acc:
                    best_acc = self.eval_acc
                    save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                    amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                                    accuracy=self.eval_acc, best_model=True)
                # 保存模型
                save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                                accuracy=self.eval_acc)

    def evaluate(self, resume_model=None, save_matrix_path=None):
        """
        评估模型
        :param resume_model: 所使用的模型
        :param save_matrix_path: 保存混合矩阵的路径
        :return: 评估结果
        """
        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = torch.load(resume_model)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            eval_model = self.model.module
        else:
            eval_model = self.model
            # 初始化指标容器（新增中间层）
        metrics = {
            'final_acc': [],
            'mid1_acc': [],
            'mid2_acc': [],
            'mid3_acc': []
        }
        accuracies, losses, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (features, label, input_lens) in enumerate(tqdm(self.test_loader)):
                if self.stop_eval: break
                features = features.to(self.device)
                label = label.to(self.device).long()
                output, middle1_out, middle2_out, middle3_out, \
                    finnal_fea, middle1_fea, middle2_fea, middle3_fea = eval_model(features)
                los = self.loss(output, label)
                # 计算各层准确率
                final_acc = accuracy(output, label)
                mid1_acc = accuracy(middle1_out, label)
                mid2_acc = accuracy(middle2_out, label)
                mid3_acc = accuracy(middle3_out, label)
                accuracies.append(final_acc)
                metrics['final_acc'].append(final_acc)
                metrics['mid1_acc'].append(mid1_acc)
                metrics['mid2_acc'].append(mid2_acc)
                metrics['mid3_acc'].append(mid3_acc)
                # 模型预测标签,(仍基于最终输出)
                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                pred = np.argmax(output, axis=1)
                preds.extend(pred.tolist())
                # 真实标签
                labels.extend(label.tolist())
                losses.append(los.data.cpu().numpy())
        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        loss = float(sum(losses) / len(losses)) if len(losses) > 0 else -1
        acc = float(sum(accuracies) / len(accuracies)) if len(accuracies) > 0 else -1
        # acc = {k: np.mean(v) for k, v in metrics.items()}
        # print("所有的真实标签:", labels)
        # print("所有的预测标签:", preds)
        # print("按照文件的所有真实标签：", all_labels)
        # 保存混淆矩阵

        # precision
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
        print("precision,recall,f1-score",precision,recall,f1)
        if save_matrix_path is not None:
            try:
                cm = confusion_matrix(labels, preds)
                print("Confusion matrix:",cm)
                plot_confusion_matrix(cm=cm, save_path=os.path.join(save_matrix_path, f'{int(time.time())}.png'),
                                      class_labels=self.class_labels)
            except Exception as e:
                logger.error(f'保存混淆矩阵失败：{e}')
        self.model.train()
        logger.info(f'Evaluate | Loss: {loss:.5f}, '
                    f'Final Acc: {avg_metrics["final_acc"]:.2%}, '
                    f'Mid1 Acc: {avg_metrics["mid1_acc"]:.2%}, '
                    f'Mid2 Acc: {avg_metrics["mid2_acc"]:.2%}, '
                    f'Mid3 Acc: {avg_metrics["mid3_acc"]:.2%}, ')

        return loss, acc, precision, recall, f1, avg_metrics['mid1_acc'], avg_metrics['mid2_acc'], avg_metrics['mid3_acc']


    def export(self, save_model_path='models/', resume_model='models/EcapaTdnn_Fbank/best_model/'):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :return:
        """
        self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        # 加载预训练模型
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pth')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model_state_dict = torch.load(resume_model)
        self.model.load_state_dict(model_state_dict)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
        self.model.eval()
        # 获取静态模型
        infer_model = self.model.export()
        infer_model_path = os.path.join(save_model_path,
                                        f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                        'inference.pth')
        os.makedirs(os.path.dirname(infer_model_path), exist_ok=True)
        torch.jit.save(infer_model, infer_model_path)
        logger.info("预测模型已保存：{}".format(infer_model_path))
