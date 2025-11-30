# CPRISP: an interpretable deep learning framework for predicting circRNA-RBP interactions and guiding therapeutic discovery in neurodegeneration

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#Usage)
- [License](#license)

# Overview
Accurate identification of circular RNA (circRNA)-protein interactions is essential for understanding post-transcriptional regulation and its disruption in complex diseases such as neurodegeneration. Here, we present CPRISP, a deep learning framework for predicting circRNA-RNA-binding protein (RBP) interaction sites by integratively modeling sequence and structural characteristics of circRNAs along with protein-specific features. CPRISP extracts multi-view circRNA representations using k-mer frequency, physicochemical encodings, secondary structure profiles, and pretrained embeddings, while simultaneously modeling RBP sequence signals to capture both local and global binding preferences. Through hierarchical residual encoding and adaptive feature fusion, the model achieves robust and generalizable predictions across diverse RBPs. Comprehensive evaluation on benchmark datasets demonstrates that CPRISP outperforms state-of-the-art methods in predictive accuracy and cross-cell line generalization. Model interpretation highlights the importance of conserved single-stranded motifs in binding site recognition. Applying CPRISP to 172 RBPs, we identify disease-relevant regulatory targets and pathways associated with Alzheimer’s disease, Parkinson’s disease, and amyotrophic lateral sclerosis. Functional enrichment and regulatory network analysis reveal convergent disruption of nucleocytoplasmic transport, lysosomal signaling, and GTPase activity across diseases. Furthermore, transcriptome-based compound screening uncovers repositionable therapeutic candidates, including metformin, N-acetylcysteine, and ambroxol, which target CPRISP-predicted RBP–gene axes. Together, these results establish CPRISP as a scalable and interpretable platform for decoding circRNA-RBP regulatory networks and guiding therapeutic discovery in neurodegenerative disorders.
![CPRISP](https://github.com/wlab2021/CPRISP/blob/main/CPRISP.svg)  

# System Requirements

## Hardware environment
+ Physical Memory: 503GB.   
+ GPU: NVIDIA-GeForce-RTX-4090.

## Software requirements
### OS Requirements
The package development version is tested on Linux operating systems. The developmental version of the package has been tested on the following systems:

+ Linux: (Ubuntu 20.04.3 LTS)

### Python Dependencies
`CPRISP` mainly depends on the Python scientific stack.
The recommended requirements for DSRNAFold are specified as follows:
```
python: 3.8
numpy: 1.22.4
pytorch：2.1.2
```
For specific setting, please see <a href="https://github.com/wlab2021/CPRISP/blob/main/requirements.txt">requirements</a> or <a href="https://github.com/wlab2021/CPRISP/blob/main/CPRISP.yml">yml</a>.

# Installation Guide:

Here is how you can create a conda environment with the specified versions of Python and required packages:
```
$ conda create -n CPRISP python=3.8
$ conda activate CPRISP
$ pip install -r requirements.txt
```
You can also create the required environment for xxx by using the provided environment.yml file:
```
$ conda env create -f CPRISP.yml 
```
+ **Installation time**: ~28m, depending on download speed.

# Usage

Due to limited GitHub space, the original data and the circRNA2Vec model required for training can be obtained via this link: <a href="https://drive.google.com/drive/folders/1OSetKafjPcpxeEIIH2i3Q_ifDMfPjG4V?usp=drive_link">Dataset and Model</a>. 

Please place the dataset in the datasets folder and the circRNA2Vec model in the circRNA2Vec folder.

Then, you can train a model on a specific RBP dataset using the following command (taking WTAP as an example):
```
$ python main.py --datasets WTAP --epoch 30 --is_train
```
*Runtime (NVIDIA-A40 GPU, epoch = 30, taking WTAP as an example):*
+ *Data preprocessing: ~13s*
+ *Model training: ~19m14s*
+ *Total time: ~19m27s*

*Outputs:：*
+ *Training loss and progress are displayed automatically.*
+ *Model checkpoints and logs are saved under:./experiments_logs/WTAP_Exp1*

Of course, you can also train on multiple RBP datasets simultaneously.
```
$ python main.py --datasets AGO1 AGO2 AGO3 --epoch 30 --is_train
```

After training, you can validate the model by using (taking WTAP as an example):

```
$ python main.py --datasets WTAP
```
*Runtime (NVIDIA-A40 GPU, taking WTAP as an example):*
+ *Testing runtime: ~16s*

*Outputs:：*
+ *Prediction results are printed automatically in the console.*


# License
This project is covered under the **MIT License**.



