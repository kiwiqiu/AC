import argparse
import functools
import time

from macls.trainer import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',              str,    'configs/panns.yml',        '配置文件') #改
add_arg('data_augment_configs', str,    'configs/augmentation.yml', '数据增强配置文件')
add_arg("local_rank",           int,    0,                          '多卡训练需要的参数')
add_arg("use_gpu",              bool,   True,                       '是否使用GPU训练')
add_arg('save_model_path',      str,    'models/',                  '模型保存的路径')
add_arg('log_dir',              str,    'log/',                     '保存VisualDL日志文件的路径')
add_arg('resume_model',         str,    None,                       '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model',     str,    None,                       '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = MAClsTrainer(configs=args.configs,
                       use_gpu=args.use_gpu,
                       data_augment_configs=args.data_augment_configs)
start = time.time()
trainer.train(save_model_path=args.save_model_path,
              log_dir=args.log_dir,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)
end = time.time()

# 将评估结果写入txt文件 改
with open('output/MelSpectrogram_results/PANNS_CNN6_results.txt', 'a') as f:
    f.write('模型：PANNS_CNN6，池化：SAP，训练消耗时间：{}s\n'.format(int(end - start)))


