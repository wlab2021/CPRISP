import argparse
import numpy as np


coden_dict = {'GCU': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,  # alanine<A>
              'UGU': 1, 'UGC': 1,  # systeine<C>
              'GAU': 2, 'GAC': 2,  # aspartic acid<D>
              'GAA': 3, 'GAG': 3,  # glutamic acid<E>
              'UUU': 4, 'UUC': 4,  # phenylanaline<F>
              'GGU': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,  # glycine<G>
              'CAU': 6, 'CAC': 6,  # histidine<H>
              'AUU': 7, 'AUC': 7, 'AUA': 7,  # isoleucine<I>
              'AAA': 8, 'AAG': 8,  # lycine<K>
              'UUA': 9, 'UUG': 9, 'CUU': 9, 'CUC': 9, 'CUA': 9, 'CUG': 9,  # leucine<L>
              'AUG': 10,  # methionine<M>
              'AAU': 11, 'AAC': 11,  # asparagine<N>
              'CCU': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,  # proline<P>
              'CAA': 13, 'CAG': 13,  # glutamine<Q>
              'CGU': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,  # arginine<R>
              'UCU': 15, 'UCC': 15, 'UCA': 15, 'UCG': 15, 'AGU': 15, 'AGC': 15,  # serine<S>
              'ACU': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,  # threonine<T>
              'GUU': 17, 'GUC': 17, 'GUA': 17, 'GUG': 17,  # valine<V>
              'UGG': 18,  # tryptophan<W>
              'UAU': 19, 'UAC': 19,  # tyrosine(Y)
              'UAA': 20, 'UAG': 20, 'UGA': 20,  # STOP code
              }
# the amino acid code adapting 21-dimensional vector (5 amino acid and 1.txt STOP code)


def coden(seq):
    seq2 = seq + seq[0: 2]
    # vectors = np.zeros((len(seq) - 2, 21))
    vectors = np.zeros((len(seq2) - 2, 21))
    for i in range(len(seq2) - 2):
            vectors[i][coden_dict[seq2[i: i + 3].replace('T', 'U')]] = 1
            # print(seq2[i: i + 3])
            # print(seq2)
            # print(seq)

    return vectors.tolist()


def Pseudo_amino_acid(protein):
    dataX = []
    dataY = []
    with open(r'datasets/circRNA-RBP/' + protein + '/positive') as f:
        for line in f:
            if '>' not in line:
                dataX.append(coden(line.strip()))
                dataY.append([0, 1])
    with open(r'datasets/circRNA-RBP/' + protein + '/negative') as f:
        for line in f:
            if '>' not in line:
                dataX.append(coden(line.strip()))
                dataY.append([1, 0])
    indexes = np.random.choice(len(dataY), len(dataY), replace=False)
    dataX = np.array(dataX)[indexes]
    # dataY = np.array(dataY)[indexes]
    # return dataX, dataY, indexes
    return dataX


coden_dict3 = {'GCU': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,  # alanine<A>
               'UGU': 1, 'UGC': 1,  # systeine<C>
               'GAU': 2, 'GAC': 2,  # aspartic acid<D>
               'GAA': 3, 'GAG': 3,  # glutamic acid<E>
               'UUU': 4, 'UUC': 4,  # phenylanaline<F>
               'GGU': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,  # glycine<G>
               'CAU': 6, 'CAC': 6,  # histidine<H>
               'AUU': 7, 'AUC': 7, 'AUA': 7,  # isoleucine<I>
               'AAA': 8, 'AAG': 8,  # lycine<K>
               'UUA': 9, 'UUG': 9, 'CUU': 9, 'CUC': 9, 'CUA': 9, 'CUG': 9,  # leucine<L>
               'AUG': 10,  # methionine<M>
               'AAU': 11, 'AAC': 11,  # asparagine<N>
               'CCU': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,  # proline<P>
               'CAA': 13, 'CAG': 13,  # glutamine<Q>
               'CGU': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,  # arginine<R>
               'UCU': 15, 'UCC': 15, 'UCA': 15, 'UCG': 15, 'AGU': 15, 'AGC': 15,  # serine<S>
               'ACU': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,  # threonine<T>
               'GUU': 17, 'GUC': 17, 'GUA': 17, 'GUG': 17,  # valine<V>
               'UGG': 18,  # tryptophan<W>
               'UAU': 19, 'UAC': 19,  # tyrosine(Y)
               'UAA': 20, 'UAG': 20, 'UGA': 20,  # STOP code
               }
# the amino acid code adapting 21-dimensional vector (5 amino acid and 1.txt STOP code)


def coden3(seq, D):
    vectors = np.zeros((400, 10))
    for j in range(10):
        seq2 = seq + seq[0:(j + 2) * 3 - 1]
        for i in range(len(seq2) - 5 - j * 3):
            a = coden_dict3[seq2[i: i + 3].replace('T', 'U')]
            b = coden_dict3[seq2[i + 3 + j * 3: i + 6 + j * 3].replace('T', 'U')]
            if a != 20 and b != 20:
                vectors[D[str(a) + str(b)]][j] += 1

    return vectors.tolist()


def Pseudo_dipeptide(protein):
    D = {}
    f = 0
    for i in range(20):
        for j in range(20):
            D[str(i) + str(j)] = f
            f += 1

    dataX = []
    dataY = []
    with open(r'datasets/circRNA-RBP/' + protein + '/positive') as f:
        for line in f:
            if ('>' not in line):
                dataX.append(coden3(line.strip(), D))
                dataY.append([0, 1])
    with open(r'datasets/circRNA-RBP/' + protein + '/negative') as f:
        for line in f:
            if '>' not in line:
                dataX.append(coden3(line.strip(), D))
                dataY.append([1, 0])
    indexes = np.random.choice(len(dataY), len(dataY), replace=False)
    dataX = np.array(dataX)[indexes]
    return dataX


coden_dict4 = {'AF': 0, 'AT': 1, 'AI': 2, 'AH': 3, 'AM': 4, 'AS': 5, 'CF': 6, 'CT': 7, 'CI': 8, 'CH': 9, 'CM': 10,
               'CS': 11, 'GF': 12, 'GT': 13, 'GI': 14, 'GH': 15, 'GM': 16, 'GS': 17, 'UF': 18, 'UT': 19, 'UI': 20,
               'UH': 21, 'UM': 22, 'US': 23, }


def coden4(useful, ignore):
    vectors = np.zeros((len(useful), 24))
    for i in range(len(useful)):
        vectors[i][coden_dict4[useful[i] + ignore[i]]] = 1
    return vectors.tolist()


# def main(parser):
#     protein = parser.RBPID
#     dataX1, dataY, indexes = Pseudo_amino_acid(protein)
#     print(dataX1)  ##892,99(101-2),21

#     dataX2 = Pseudo_dipeptide(protein, indexes)
#     print(dataX2) ##892,400,10
