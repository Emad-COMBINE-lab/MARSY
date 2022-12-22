- data.csv
This file holds the drug combinations, the cell line and the synergy scores. Using this file, the drug and cell line names will be extracted.

- 75_cell_lines_gene_expression.csv
This file holds the gene expression values for the cell lines. Each cell line has expression of  4639 genes. See the corresponding journal manuscript for details of data pre-processing.

- MCF7_drugs_gene_expression.csv
This file holds the gene expression drug signatures obtained using MCF7 cell line. Each drug is represented using a vector of length 978.

- PC3_drugs_gene_expression.csv
This file holds the gene expression drug signatures obtained using PC3 cell line. Each drug is represented using a vector of length 978.


In total, we will have 8551 expression values for each sample (978 + 978 + 978 + 978 + 4639).

------------------------------------------------------------------------------------------------

We have also uploaded the train and test folds for the leave-triple-out (lto) and the leave-pair-out (lpo) in the form of indices. The indices correspond to the indices of the data.csv file.

Tr = Training
Tst = Test
The number corresponds to the fold number.

The folder 'lpo_folds' has 10 files.

The folder 'lto_folds' has 1 file
For lto, the five-fold cross-validation will be done by splitting the data using an 80:20 split to create folds.
Note: The data in 'lto_folds/Tpl_Folds_Sig.txt' only has even indices and half the total number of samples. So, to perform cross-validation using this, the index needs to be incremented once. This incremented odd index will be included with the respective train/test set. 
This has been done to cater to the flipped drugs augmented data and to make sure there is no data leakage.