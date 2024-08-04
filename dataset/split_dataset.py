import json
import random

def split_dataset(dataset_info_path, random_seed=0, train_rate=0.8):
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)

    dataset_info = pick_dataset(dataset_info)
    
    random.seed(random_seed)
    random.shuffle(dataset_info)

    train_ratio = train_rate
    val_ratio = (1 - train_ratio) * 0.5

    total_samples = len(dataset_info)
    train_samples = int(train_ratio * total_samples)
    val_samples = int(val_ratio * total_samples)

    train_dataset = dataset_info[:train_samples]
    val_dataset = dataset_info[train_samples:train_samples+val_samples]
    test_dataset = dataset_info[train_samples+val_samples:]

    with open('train_dataset_info.json', 'w') as f:
        json.dump(train_dataset, f, indent=4)

    with open('validation_dataset_info.json', 'w') as f:
        json.dump(val_dataset, f, indent=4)

    with open('test_dataset_info.json', 'w') as f:
        json.dump(test_dataset, f, indent=4)
    return None

def pick_dataset(dataset_info):
    target_id_list = ['0373', '0143', '0346', '0166', '0566', '0126', '0473', '0456', '0146', '0356', '0363', '0133', '0553', '0376', '0343', '0477']
    new_dataset_info = []
    for data_info in dataset_info:
        date = data_info['meta_data']['date']
        if date == '20230623' or date == '20230627':
            new_annotation = []
            for annotation in data_info['annotations']:
                if annotation['penguin_id'] in target_id_list:
                    new_annotation.append(annotation)
                else:
                    annotation['penguin_id'] = '0000'
                    new_annotation.append(annotation)
            data_info['annotations'] = new_annotation
            if len(new_annotation) > 0:
                new_dataset_info.append(data_info)
    return new_dataset_info