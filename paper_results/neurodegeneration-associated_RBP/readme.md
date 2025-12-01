### Data Processing Workflow



#### 1\. Model Execution



###### After running the model, three .txt files are generated in the scan folder.



#### 2\. Conversion to Tables



###### Convert the three .txt files into tabular format and save them in the disdata folder.



#### 3\. Data Filtering



###### Process the tabular data with disdul to obtain filtered\_data\_\* results.



#### 4\. Gene ID Conversion and Functional Enrichment



###### Process filtered\_data\_\* with the id\_tras.R script in RStudio to generate \*\_common\_gene files, and then perform functional enrichment using GO.R:

###### ① GO enrichment

###### ② KEGG enrichment

