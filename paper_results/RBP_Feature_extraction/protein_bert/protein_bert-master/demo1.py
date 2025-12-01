import os
import pandas as pd
from proteinbert import load_pretrained_model, get_feature_representation


BENCHMARKS_DIR = './protein_benchmarks'
# BENCHMARK_NAME = 'signalP_binary'
BENCHMARK_NAME = '37'
# BENCHMARK_NAME = '172'
# BENCHMARK_NAME = 'rbp_ts'

#输入对应RBP的序列
test_set_file_path = os.path.join(BENCHMARKS_DIR, '%s.test.csv' % BENCHMARK_NAME)

test_set = pd.read_csv(test_set_file_path).dropna()
test_set['label'] = test_set['label'].astype(float)

print('test set records: ', len(test_set))

# Loading the pre-trained model
pretrained_model_generator, input_encoder = load_pretrained_model()
model_generator = pretrained_model_generator

# Get the representations on the test-set
local_representations, global_representations = get_feature_representation(model_generator, input_encoder, test_set['seq'],
                                                                           start_seq_len=2048, start_batch_size=32)

# local_representations, global_representations = get_feature_representation(model_generator, input_encoder, test_set['seq'],
#                                                                            start_seq_len=4096, start_batch_size=32)

print('local representation: ', local_representations.shape)
print('global representation: ', global_representations.shape)

