import numpy as np
import pandas as pd

# from .shared_utils.util import log
from .tokenization import ADDED_TOKENS_PER_SEQ

class OutputType:
    
    def __init__(self, is_seq, output_type):
        self.is_seq = is_seq
        self.output_type = output_type
        self.is_numeric = (output_type == 'numeric')
        self.is_binary = (output_type == 'binary')
        self.is_categorical = (output_type == 'categorical')
        
    def __str__(self):
        if self.is_seq:
            return '%s sequence' % self.output_type
        else:
            return 'global %s' % self.output_type
            
class OutputSpec:

    def __init__(self, output_type, unique_labels = None):
        
        if output_type.is_numeric:
            assert unique_labels is None
        elif output_type.is_binary:
            if unique_labels is None:
                unique_labels = [0, 1]
            else:
                assert unique_labels == [0, 1]
        elif output_type.is_categorical:
            assert unique_labels is not None
        else:
            raise ValueError('Unexpected output type: %s' % output_type)
        
        self.output_type = output_type
        self.unique_labels = unique_labels
        
        if unique_labels is not None:
            self.n_unique_labels = len(unique_labels)
            
def finetune(model_generator, input_encoder, output_spec, train_seqs, train_raw_Y, valid_seqs = None, valid_raw_Y = None, seq_len = 512, batch_size = 32, \
        max_epochs_per_stage = 40, lr = None, begin_with_frozen_pretrained_layers = True, lr_with_frozen_pretrained_layers = None, n_final_epochs = 1, \
        final_seq_len = 1024, final_lr = None, callbacks = []):
        
    encoded_train_set, encoded_valid_set = encode_train_and_valid_sets(train_seqs, train_raw_Y, valid_seqs, valid_raw_Y, input_encoder, output_spec, seq_len)
        
    if begin_with_frozen_pretrained_layers:
        #log('Training with frozen pretrained layers...')
        model_generator.train(encoded_train_set, encoded_valid_set, seq_len, batch_size, max_epochs_per_stage, lr = lr_with_frozen_pretrained_layers, \
                callbacks = callbacks, freeze_pretrained_layers = True)
     
    #log('Training the entire fine-tuned model...')
    model_generator.train(encoded_train_set, encoded_valid_set, seq_len, batch_size, max_epochs_per_stage, lr = lr, callbacks = callbacks, \
            freeze_pretrained_layers = False)
                
    if n_final_epochs > 0:
        #log('Training on final epochs of sequence length %d...' % final_seq_len)
        final_batch_size = max(int(batch_size / (final_seq_len / seq_len)), 1)
        encoded_train_set, encoded_valid_set = encode_train_and_valid_sets(train_seqs, train_raw_Y, valid_seqs, valid_raw_Y, input_encoder, output_spec, final_seq_len)
        model_generator.train(encoded_train_set, encoded_valid_set, final_seq_len, final_batch_size, n_final_epochs, lr = final_lr, callbacks = callbacks, \
                freeze_pretrained_layers = False)
                
    model_generator.optimizer_weights = None

def get_feature_representation(model_generator, input_encoder, seqs, start_seq_len=512, start_batch_size=32, increase_factor=2):
# def get_feature_representation(model_generator, input_encoder, seqs, start_seq_len=512, start_batch_size=32, increase_factor=2):
# def get_feature_representation(model_generator, input_encoder, seqs, start_seq_len=857, start_batch_size=32, increase_factor=2):

    local_representations = []
    global_representations = []
    assert model_generator.optimizer_weights is None
    dataset = pd.DataFrame({'seq': seqs})
    for len_matching_dataset, seq_len, batch_size in split_dataset_by_len(dataset, start_seq_len = start_seq_len, start_batch_size = start_batch_size, \
            increase_factor = increase_factor):
        # len_matching_dataset = 37
        X = encode_dataset(len_matching_dataset['seq'], input_encoder, seq_len=seq_len, needs_filtering=False)

        model = model_generator.create_model(seq_len)
        y_pred = model.predict(X, batch_size=batch_size)

        local_representations.append(y_pred[2])
        global_representations.append(y_pred[3])
        a = local_representations[0]
        b = global_representations[0]

    np.savetxt("global_representation_172.txt", b)
    with open('local_representations_172.txt', 'w') as outfile:
        for slice_2d in a:
            np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')

    # return np.array(local_representations)[0], np.array(global_representations)[0]
    return a, b
    # return a, b

def get_evaluation_results(y_true, y_pred, output_spec, return_confusion_matrix = False):

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
            
    results = {}
    results['# records'] = len(y_true)
            
    if output_spec.output_type.is_numeric:
        results['Spearman\'s rank correlation'] = spearmanr(y_true, y_pred)[0]
        confusion_matrix = None
    else:
    
        str_unique_labels = list(map(str, output_spec.unique_labels))
        
        if output_spec.output_type.is_binary:
            
            y_pred_classes = (y_pred >= 0.5)
            
            if len(np.unique(y_true)) == 2:
                results['AUC'] = roc_auc_score(y_true, y_pred)
            else:
                results['AUC'] = np.nan
        elif output_spec.output_type.is_categorical:
            y_pred_classes = y_pred.argmax(axis = -1)
            results['Accuracy'] = accuracy_score(y_true, y_pred_classes)
        else:
            raise ValueError('Unexpected output type: %s' % output_spec.output_type)
                    
        confusion_matrix = pd.DataFrame(confusion_matrix(y_true, y_pred_classes, labels = np.arange(output_spec.n_unique_labels)), index = str_unique_labels, \
                    columns = str_unique_labels)
         
    if return_confusion_matrix:
        return results, confusion_matrix
    else:
        return results
        
def encode_train_and_valid_sets(train_seqs, train_raw_Y, valid_seqs, valid_raw_Y, input_encoder, output_spec, seq_len):
    
    encoded_train_set = encode_dataset(train_seqs, train_raw_Y, input_encoder, output_spec, seq_len = seq_len, needs_filtering = True, \
            dataset_name = 'Training set')
    
    if valid_seqs is None and valid_raw_Y is None:
        encoded_valid_set = None
    else:
        encoded_valid_set = encode_dataset(valid_seqs, valid_raw_Y, input_encoder, output_spec, seq_len = seq_len, needs_filtering = True, \
                dataset_name = 'Validation set')

    return encoded_train_set, encoded_valid_set
        
def encode_dataset(seqs, input_encoder, seq_len = 512, needs_filtering = True, dataset_name = 'Dataset', verbose = True):

    if needs_filtering:
        dataset = pd.DataFrame({'seq': seqs})
        dataset = filter_dataset_by_len(dataset, seq_len = seq_len, dataset_name = dataset_name, verbose = verbose)
        seqs = dataset['seq']

    X = input_encoder.encode_X(seqs, seq_len)

    return X

def encode_Y(raw_Y, output_spec, seq_len = 512):
    if output_spec.output_type.is_seq:
        return encode_seq_Y(raw_Y, seq_len, output_spec.output_type.is_binary, output_spec.unique_labels)
    elif output_spec.output_type.is_categorical:
        return encode_categorical_Y(raw_Y, output_spec.unique_labels), np.ones(len(raw_Y))
    elif output_spec.output_type.is_numeric or output_spec.output_type.is_binary:
        return raw_Y.values.astype(float), np.ones(len(raw_Y))
    else:
        raise ValueError('Unexpected output type: %s' % output_spec.output_type)

def encode_seq_Y(seqs, seq_len, is_binary, unique_labels):

    label_to_index = {str(label): i for i, label in enumerate(unique_labels)}

    Y = np.zeros((len(seqs), seq_len), dtype = int)
    sample_weigths = np.zeros((len(seqs), seq_len))
    
    for i, seq in enumerate(seqs):
        
        for j, label in enumerate(seq):
            # +1 to account for the <START> token at the beginning.
            Y[i, j + 1] = label_to_index[label]
            
        sample_weigths[i, 1:(len(seq) + 1)] = 1
        
    if is_binary:
        Y = np.expand_dims(Y, axis = -1)
        sample_weigths = np.expand_dims(sample_weigths, axis = -1)
    
    return Y, sample_weigths
    
def encode_categorical_Y(labels, unique_labels):
    
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    Y = np.zeros(len(labels), dtype = int)
    
    for i, label in enumerate(labels):
        Y[i] = label_to_index[label]
        
    return Y
    
def filter_dataset_by_len(dataset, seq_len = 512, seq_col_name = 'seq', dataset_name = 'Dataset', verbose = True):

    max_allowed_input_seq_len = seq_len - ADDED_TOKENS_PER_SEQ
    filtered_dataset = dataset[dataset[seq_col_name].str.len() <= max_allowed_input_seq_len]
    n_removed_records = len(dataset) - len(filtered_dataset)
    '''
    if verbose:
        log('%s: Filtered out %d of %d (%.1f%%) records of lengths exceeding %d.' % (dataset_name, n_removed_records, len(dataset), 100 * n_removed_records / len(dataset), \
                max_allowed_input_seq_len))
    '''
    return filtered_dataset
    
def split_dataset_by_len(dataset, seq_col_name = 'seq', start_seq_len = 512, start_batch_size = 32, increase_factor = 2):

    seq_len = start_seq_len
    batch_size = start_batch_size
    
    while len(dataset) > 0:
        max_allowed_input_seq_len = seq_len - ADDED_TOKENS_PER_SEQ
        len_mask = (dataset[seq_col_name].str.len() <= max_allowed_input_seq_len)
        # len_matching_dataset = dataset
        len_matching_dataset = dataset[len_mask]
        yield len_matching_dataset, seq_len, batch_size
        dataset = dataset[~len_mask]
        seq_len *= increase_factor
        batch_size = max(batch_size // increase_factor, 1)
