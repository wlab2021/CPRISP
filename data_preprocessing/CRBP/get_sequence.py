import numpy as np
import collections

# coden_dict = {'GCU': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,  # alanine<A>
#               'UGU': 1, 'UGC': 1,  # systeine<C>
#               'GAU': 2, 'GAC': 2,  # aspartic acid<D>
#               'GAA': 3, 'GAG': 3,  # glutamic acid<E>
#               'UUU': 4, 'UUC': 4,  # phenylanaline<F>
#               'GGU': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,  # glycine<G>
#               'CAU': 6, 'CAC': 6,  # histidine<H>
#               'AUU': 7, 'AUC': 7, 'AUA': 7,  # isoleucine<I>
#               'AAA': 8, 'AAG': 8,  # lycine<K>
#               'UUA': 9, 'UUG': 9, 'CUU': 9, 'CUC': 9, 'CUA': 9, 'CUG': 9,  # leucine<L>
#               'AUG': 10,  # methionine<M>
#               'AAU': 11, 'AAC': 11,  # asparagine<N>
#               'CCU': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,  # proline<P>
#               'CAA': 13, 'CAG': 13,  # glutamine<Q>
#               'CGU': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,  # arginine<R>
#               'UCU': 15, 'UCC': 15, 'UCA': 15, 'UCG': 15, 'AGU': 15, 'AGC': 15,  # serine<S>
#               'ACU': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,  # threonine<T>
#               'GUU': 17, 'GUC': 17, 'GUA': 17, 'GUG': 17,  # valine<V>
#               'UGG': 18,  # tryptophan<W>
#               'UAU': 19, 'UAC': 19,  # tyrosine(Y)
#               'UAA': 20, 'UAG': 20, 'UGA': 20,  # STOP code
#               }
# # the amino acid code adapting 21-dimensional vector (5 amino acid and 1 STOP code)
#
#
# def coden(seq):
#     vectors = np.zeros((len(seq) - 2, 21))
#     for i in range(len(seq) - 2):
#         vectors[i][coden_dict[seq[i:i + 3].replace('T', 'U')]] = 1
#     return vectors.tolist()
#
# #
# def dealwithSequence(protein):
#     dataX = []
#     dataY = []
#     with open('../../datasets/circRNA-RBP/' + protein + '/positive') as f:
#         for line in f:
#             if '>' not in line:
#                 dataX.append(coden(line.strip()))
#                 dataY.append([0, 1])
#     with open('../../datasets/circRNA-RBP/' +  protein + '/negative') as f:
#         for line in f:
#             if '>' not in line:
#                 dataX.append(coden(line.strip()))
#                 dataY.append([1, 0])
#     indexes = np.random.choice(len(dataY), len(dataY), replace=False)
#     dataX = np.array(dataX)[indexes]
#     dataY = np.array(dataY)[indexes]
#     return dataX, dataY, indexes


#********************************************************************************
def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 1
    for i in range(0, end):
        n = i
        ch0 = chars[n%base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 2
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 3
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 4
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        n = n // base
        ch3 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_5_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 5
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        n = n // base
        ch3 = chars[n % base]
        n = n // base
        ch4 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def get_6_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 6
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        n = n // base
        ch2 = chars[n % base]
        n = n // base
        ch3 = chars[n % base]
        n = n // base
        ch4 = chars[n % base]
        n = n // base
        ch5 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4 + ch5)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index


def frequency(seq,kmer,coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i: i + k]
        kmer_value = coden_dict[kmer.replace('T', 'U')]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict


def coden(seq,kmer,tris):
    coden_dict = tris
    freq_dict = frequency(seq, kmer, coden_dict)
    vectors = np.zeros((101, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i: i + kmer].replace('T', 'U')]]
        vectors[i][coden_dict[seq[i: i + kmer].replace('T', 'U')]] = value / 100
    return vectors


def dealwithdata1(protein):
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    # tris4 = get_4_trids()
    # tris6 = get_6_trids()
    dataX = []
    dataY = []
    with open(r'datasets/circRNA-RBP/' + protein + '/positive') as f:
        for line in f:
            if '>' not in line:
                kmer1 = coden(line.strip(), 1, tris1)
                kmer2 = coden(line.strip(), 2, tris2)
                kmer3 = coden(line.strip(), 3, tris3)
                # kmer4 = coden(line.strip(), 4, tris4)
                Kmer = np.hstack((kmer1, kmer2, kmer3))
                dataX.append(Kmer.tolist())
                dataY.append(1)
    with open(r'datasets/circRNA-RBP/' + protein + '/negative') as f:
        for line in f:
            if '>' not in line:
                kmer1 = coden(line.strip(), 1, tris1)
                kmer2 = coden(line.strip(), 2, tris2)
                kmer3 = coden(line.strip(), 3, tris3)
                # kmer4 = coden(line.strip(), 4, tris4)
                Kmer = np.hstack((kmer1, kmer2, kmer3))
                dataX.append(Kmer.tolist())
                dataY.append(0)

    dataX = np.array(dataX)
    dataY = np.array(dataY)

    return dataX, dataY
