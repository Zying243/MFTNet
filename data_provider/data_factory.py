from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, generated_time_series_data, Dataset_Solar
from torch.utils.data import DataLoader



data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'weather': Dataset_Custom,
    'generated_time_series_data':generated_time_series_data,
    '1wind_dataset': Dataset_Custom,
    'appo': Dataset_Custom,
}


def data_provider(args, flag):
    # 确保 args.data 在 data_dict 中
    if args.data not in data_dict:
        raise ValueError(f"无效的数据类型 '{args.data}'。可用选项: {list(data_dict.keys())}")

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    # 根据 flag 设置参数
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred  # 确保 Dataset_Pred 是有效的数据集类
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # 初始化数据集
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )

    # 创建数据加载器
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader

