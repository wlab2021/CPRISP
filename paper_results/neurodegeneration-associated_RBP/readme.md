### Data Processing Workflow



#### 1\. Model Execution



###### After running the model, three .txt files are generated in the scan folder.



#### 2\. Conversion to Tables



###### Convert the three .txt files into tabular format and save them in the disdata folder.



#### 3\. Data Filtering



###### Process the tabular data with disdul.py to obtain filtered\_data\_\* results.



#### 4\. Gene ID Conversion and Functional Enrichment



###### Process filtered\_data\_\* with the id\_tras.R script in RStudio to generate \*\_common\_gene files, and then perform functional enrichment using GO.R:

###### ① GO enrichment

###### ② KEGG enrichment




### Cys Visualization
#### 1\. Gene Symbol Conversion for Network Analysis



###### Based on the KEGG enrichment results, convert gene IDs into gene symbols using the functions within GO.R to prepare them for downstream network construction.

#### 2\. Network Construction in NetworkAnalyst



###### Upload the gene symbol list to NetworkAnalyst, perform network analysis (e.g., PPI network), and download the resulting network files.
###### During this step, also export the node-level scores (e.g., degree, betweenness, centrality) for each gene.


#### 3\. Cytoscape Visualization



###### Import the JSON-format network file into Cytoscape, and simultaneously load the exported node score table.
###### Merge node attributes and visualize the network by mapping node scores to visual features (size, color, etc.).
###### Generate the final network visualization for reporting and interpretation.


#### Note: For the figures presented in the article, the Cytoscape network files **tf.cys** and **mirna.cys** can be opened directly in Cytoscape to reproduce the visualizations.



