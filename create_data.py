import os
import pandas as pd
# import chardet
from sklearn.model_selection import StratifiedShuffleSplit
import random
from pathlib import Path

# 生成数据列表
def get_data_list(audio_path, list_path):
    sound_sum = 0
    audios = os.listdir(audio_path)
    os.makedirs(list_path, exist_ok=True)
    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8')
    f_label = open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8')

    for i in range(len(audios)):
        f_label.write(f'{audios[i]}\n')
        sounds = os.listdir(os.path.join(audio_path, audios[i]))
        for sound in sounds:
            sound_path = os.path.join(audio_path, audios[i], sound).replace('\\', '/')
            if sound_sum % 10 == 0:
                f_test.write(f'{sound_path}\t{i}\n')
            else:
                f_train.write(f'{sound_path}\t{i}\n')
            sound_sum += 1
        print(f"Audio：{i + 1}/{len(audios)}")
    f_label.close()
    f_test.close()
    f_train.close()

# 下载数据方式，执行：./tools/download_3dspeaker_data.sh
# 生成生成方言数据列表
def get_language_identification_data_list(audio_path, list_path):
    labels_dict = {0: 'Standard Mandarin', 3: 'Southwestern Mandarin', 6: 'Central Plains Mandarin',
                   4: 'JiangHuai Mandarin', 2: 'Wu dialect', 8: 'Gan dialect', 9: 'Jin dialect',
                   11: 'LiaoJiao Mandarin', 12: 'JiLu Mandarin', 10: 'Min dialect', 7: 'Yue dialect',
                   5: 'Hakka dialect', 1: 'Xiang dialect', 13: 'Northern Mandarin'}

    with open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8') as f:
        train_dir = os.path.join(audio_path, 'train')
        for root,  dirs, files in os.walk(train_dir):
            for file in files:
                if not file.endswith('.wav'): continue
                label = int(file.split('_')[-1].replace('.wav', '')[-2:])
                file = os.path.join(root, file)
                f.write(f'{file}\t{label}\n')

    with open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8') as f:
        test_dir = os.path.join(audio_path, 'test')
        for root,  dirs, files in os.walk(test_dir):
            for file in files:
                if not file.endswith('.wav'): continue
                label = int(file.split('_')[-1].replace('.wav', '')[-2:])
                file = os.path.join(root, file)
                f.write(f'{file}\t{label}\n')

    with open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(labels_dict)):
            f.write(f'{labels_dict[i]}\n')


# 创建UrbanSound8K数据列表
def create_UrbanSound8K_list(audio_path, metadata_path, list_path):
    sound_sum = 0

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8')
    f_label = open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8')

    with open(metadata_path) as f:
        lines = f.readlines()

    labels = {}
    for i, line in enumerate(lines):
        if i == 0:continue
        data = line.replace('\n', '').split(',')
        class_id = int(data[6])
        if class_id not in labels.keys():
            labels[class_id] = data[-1]
        sound_path = os.path.join(audio_path, f'fold{data[5]}', data[0]).replace('\\', '/')
        if sound_sum % 10 == 0:
            f_test.write(f'{sound_path}\t{data[6]}\n')
        else:
            f_train.write(f'{sound_path}\t{data[6]}\n')
        sound_sum += 1
    for i in range(len(labels)):
        f_label.write(f'{labels[i]}\n')
    f_label.close()
    f_test.close()
    f_train.close()

# 电梯数据集列表csv编码不唯一，需要chardet辅助自行判断
# def detect_encoding(file_path):
#     with open(file_path, 'rb') as f:
#         result = chardet.detect(f.read())
#     return result['encoding']

# 创建电梯音频数据列表
# def create_filtered_df_from_all_target_meta_file(target_labels, meta_df_path, target_meta_file):
#     save_columns = ['filename', 'event', 'type']
#
#     # 检测编码并读取文件
#     encoding = detect_encoding(meta_df_path)
#     df = pd.read_csv(meta_df_path, encoding=encoding, low_memory=False)
#     merge = any('-' in label for label in target_labels)
#
#     # 首先选取单事件音频段，去除重叠的音频段
#     df = df[(df['event'].isin(
#         ['blank', 'regular_talking', 'elec_device', 'lift_noise', 'buzzer', 'loud_talking', 'object_sound', 'scream',
#          'screech']))]
#     filtered_df = df
#
#     if not merge:  # 类别没有合并的情况
#         filtered_df = df[df['event'].isin(target_labels)]
#     else:  # 对于类别合并的情况单独处理
#         # 输入的形式
#         # ['regular_talking-elec_device', 'lift_noise-object_sound', 'loud_talking-scream', 'blank', 'buzzer']
#         for tl in target_labels:
#             if '-' in tl:
#                 labels = tl.split('-')
#                 # 将event赋值为 合并的label
#                 to_replace = {}
#                 for l in labels:
#                     to_replace[l] = tl
#                 filtered_df.replace(to_replace, inplace=True)
#         filtered_df = filtered_df[filtered_df['event'].isin(target_labels)]
#
#     filtered_df = filtered_df.reset_index(drop=True)
#     filtered_df = filtered_df[save_columns]
#     filtered_df.to_csv(target_meta_file, index=False, encoding='utf-8-sig')
#
#     return filtered_df

def create_elevator_audio_list(audio_root, metadata_path, output_dir, test_size=0.2, random_seed=42):
    """
       生成电梯音频数据集的训练/测试列表及标签列表

       参数:
           audio_root    : 音频文件根目录 (如: "D:/elevator_audio/")
           metadata_path : 元数据文件路径 (如: "metadata.csv")
           output_dir    : 输出文件目录 (如: "dataset_lists/")
           test_size     : 测试集比例 (默认0.2)
           random_seed   : 随机种子 (默认42)
       """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取元数据 (假设为CSV格式，列分隔符为\t)
    df = pd.read_csv(metadata_path, sep=',')

    # 生成标签映射
    labels = df['event'].unique().tolist()
    label_to_id = {label: idx for idx, label in enumerate(sorted(labels))}

    # 添加完整文件路径
    df['full_path'] = df['filename'].apply(lambda x: os.path.join(audio_root, x.strip()))

    # 分层划分数据集,样本存在严重的数据不平衡的问题，进行分层抽样，保持训练/测试集的类别分布一致，避免数据偏差。
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    for train_idx, test_idx in split.split(df, df['event']):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

    # 写入训练列表文件 (txt格式)
    with open(os.path.join(output_dir, 'train_list.txt'), 'w', encoding='utf-8') as f:
        for _, row in train_df.iterrows():
            line = f"{row['full_path']}\t{label_to_id[row['event']]}\n"
            f.write(line)

    # 写入测试列表文件 (txt格式)
    with open(os.path.join(output_dir, 'test_list.txt'), 'w', encoding='utf-8') as f:
        for _, row in test_df.iterrows():
            line = f"{row['full_path']}\t{label_to_id[row['event']]}\n"
            f.write(line)

    # 写入标签列表文件 (txt格式)
    with open(os.path.join(output_dir, 'label_list.txt'), 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(f"{label}\n")

# 数据集减少至每个类别400，直接在特征数据加划分训练测试集
def split_feature_dataset(
        feature_root: str,
        output_dir: str,
        train_ratio: float = 0.8,
        seed: int = 42
):
    """
    在预处理特征文件上执行分层抽样划分
    :param feature_root: 特征根目录（结构：features/类别/*.npy）
    :param output_dir: 输出目录，用于保存划分文件
    :param train_ratio: 训练集比例
    :param seed: 随机种子
    """
    # 初始化路径和随机种子
    project_root = Path.cwd()  # 项目根目录 D:/qiulingyan/ACP-master/
    feature_path = (project_root / feature_root).resolve()
    output_path = (project_root / output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # 获取类别映射
    class_dirs = sorted([d for d in feature_path.iterdir() if d.is_dir()])
    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}

    # 存储分割结果
    train_data = []
    test_data = []

    # 遍历每个类别目录
    for class_dir in class_dirs:
        # 收集所有特征文件
        all_files = list(class_dir.glob("*.npy"))
        random.shuffle(all_files)

        # 分割数据
        split_idx = int(len(all_files) * train_ratio)
        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]

        # 相对于项目根目录的路径（dataset/features/class_x/file.npy）
        for f in train_files:
            rel_path = f.relative_to(project_root)  # 相对于features的父目录
            train_data.append((str(rel_path), class_to_idx[class_dir.name]))

        for f in test_files:
            rel_path = f.relative_to(project_root)
            test_data.append((str(rel_path), class_to_idx[class_dir.name]))

    # 打乱整体顺序（保持分层结构）
    random.shuffle(train_data)
    random.shuffle(test_data)

    # 写入文件
    def write_txt(data, filename):
        with (output_path / filename).open('w') as f:
            for path, label in data:
                f.write(f"{path}\t{label}\n")

    write_txt(train_data, "train_list_features.txt")
    write_txt(test_data, "test_list_features.txt")

    # # 保存类别映射
    # with (output_path / "classes.txt").open('w') as f:
    #     for cls, idx in class_to_idx.items():
    #         f.write(f"{idx} {cls}\n")


if __name__ == "__main__":
    split_feature_dataset(
        feature_root="dataset/features",  # 特征根目录路径
        output_dir="dataset/",  # 输出目录路径
        train_ratio=0.8,
        seed=42
    )

# if __name__ == '__main__':
#     # get_data_list('dataset/audio', 'dataset')
#     # 生成生成方言数据列表
#     # get_language_identification_data_list(audio_path='dataset/language',
#     #                                       list_path='dataset/')
#     # 创建UrbanSound8K数据列表
#     # create_UrbanSound8K_list(audio_path='dataset/UrbanSound8K/audio',
#     #                          metadata_path='dataset/UrbanSound8K/metadata/UrbanSound8K.csv',
#     #                          list_path='dataset')
#
#     # 1.创建电梯音频数据列表
#     target_labels = ['regular_talking', 'elec_device', 'blank', 'lift_noise', 'object_sound', 'loud_talking', 'scream',
#                      'buzzer']
#     meta_file = 'D:/qiulingyan/ACP-master/dataset/elevator_audio/quiet_filtered_meta_df.csv'
#     target_meta_file = 'D:/qiulingyan/ACP-master/dataset/elevator_audio/no_target_meta_data.csv'
#     print("当前工作目录:", os.getcwd())
#     print("文件是否存在:", os.path.exists(meta_file))
#
#     # 从总数据的csv，选取单事件音频target_labels中8种事件，没有screach和other这两种事件
#     # meta_df = create_filtered_df_from_all_target_meta_file(target_labels, meta_file, target_meta_file)
#
#     # 2.划分训练和测试集
#     audio_root = "dataset/elevator_audio/"  # 替换为实际音频根目录
#     # metadata_path = target_meta_file # 元数据文件路径
#     output_dir = "D:/qiulingyan/ACP-master/dataset/"  # 输出目录
#
#     create_elevator_audio_list(
#         audio_root=audio_root,
#         metadata_path=target_meta_file,
#         output_dir=output_dir,
#         test_size=0.2,
#         random_seed=42
#     )