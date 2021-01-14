import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import RobustScaler
import dgl


def build_graph(relations_list, relations_data_list):
    relations_data_dic = {}
    i = 0
    for each in relations_list:
        relations_data_dic[each] = relations_data_list[i]
        i += 1
    graph = dgl.heterograph(
        relations_data_dic
    )

    print('Node types:', graph.ntypes)
    print('Edge types:', graph.etypes)
    print('Canonical edge types:', graph.canonical_etypes)
    for each in graph.canonical_etypes:
        print('graph number edges--' + str(each) + ':', graph.number_of_edges(each))
    for each in graph.ntypes:
        print('graph number nodes--' + str(each) + ':', graph.number_of_nodes(each))
    return graph

# ('user', 'collect', 'note')


def build_labels(pos_label_path, neg_label_path, label_column, pos_train_ratio, neg_train_ratio):
    pos_df = pq.ParquetDataset(pos_label_path, filesystem=s3).read().to_pandas()
    neg_df = pq.ParquetDataset(neg_label_path, filesystem=s3).read().to_pandas()

    pos_u_list = np.array(pos_df[pos_df['node_type'] == label_column]['no'])  # 获取某个点类型的pos标签
    neg_u_list = np.array(neg_df[neg_df['node_type'] == label_column]['no'])  # 获取某个点类型的neg标签
    print("neg_list:", np.max(neg_u_list))
    print("pos_list:", np.max(pos_u_list))
    neg_split_pt = int(neg_u_list.shape[0] * neg_train_ratio) - 1
    pos_split_pt = int(pos_u_list.shape[0] * pos_train_ratio) - 1

    print("positive_samples_num:", pos_u_list.shape[0])
    print("negative_samples_num:", neg_u_list.shape[0])
    print("positive_train_samples_num:", pos_split_pt)
    print("negative_train_samples_num:", neg_split_pt)
    shuffled_neg = np.random.permutation(neg_u_list)
    train_neg_list = shuffled_neg[:neg_split_pt]
    valid_neg_list = shuffled_neg[neg_split_pt:]

    shuffled_pos = np.random.permutation(pos_u_list)
    train_pos_list = shuffled_pos[:pos_split_pt]
    valid_pos_list = shuffled_pos[pos_split_pt:]

    train_idx = np.concatenate([train_neg_list, train_pos_list])
    valid_idx = np.concatenate([valid_neg_list, valid_pos_list])

    neg_labels = np.zeros(neg_u_list.shape)
    pos_labels = np.ones(pos_u_list.shape)

    train_labels = np.concatenate([neg_labels[:neg_split_pt], pos_labels[:pos_split_pt]])
    valid_labels = np.concatenate([neg_labels[neg_split_pt:], pos_labels[pos_split_pt:]])

    train_idx = torch.from_numpy(train_idx).long()
    valid_idx = torch.from_numpy(valid_idx).long()

    train_labels = torch.Tensor(train_labels).long()
    valid_labels = torch.Tensor(valid_labels).long()

    return train_idx, valid_idx, train_labels, valid_labels


def node_feature_handle(df,categorical_variables_list,numerical_variables_list):
    df = df[categorical_variables_list + numerical_variables_list ]
    for each in categorical_variables_list:
        print(each)
        features_encoder = LabelBinarizer()
        features_encoder.fit(df[each])
        transformed = features_encoder.transform(df[each])
        ohe_df = pd.DataFrame(transformed)
        df = pd.concat([df, ohe_df], axis=1)
#   categorical_variables_list.append('no')
    df = df.drop(categorical_variables_list, axis=1)
    df.info()
    scaler = RobustScaler()
    df[numerical_variables_list] = scaler.fit_transform(df[numerical_variables_list])
    torch_tensor = torch.tensor(df.values)
    return torch_tensor