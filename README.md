
Machine Learning and Statistical Benchmarking of 39 Classifiers for the Physicochemical Classification of the 20 Standard Amino Acids from a Unified Molecular Descriptor Set, with Explicit Null-Control Validation
Samuel Kakraba1,2,3,*, Edmund Fosu Agyemang1, Kamyar Sharifi4, Desmond Yemeh3 and Helena Okyere5
1Department of Biostatistics and Data Science, Celia Scott Weatherhead School of Public Health and Tropical Medicine, Tulane University, New Orleans, LA, USA
2Connolly Alexander Institute for Data Science, Tulane University, New Orleans, LA, United States
3Tulane Center for Aging, School of Medicine, Tulane University, New Orleans, LA, United States
4Biomedical Science Graduate Program, School of Medicine, Tulane University, New Orleans, LA, United States
5Department of Chemistry, School of Science and Engineering, Tulane University, New Orleans, LA, United States
*Corresponding author: Samuel Kakraba, Department of Biostatistics and Data Science, Celia Scott Weatherhead School of Public Health and Tropical Medicine, Tulane University, 1440 Canal Street, New Orleans, LA 70112, USA. Email: skakraba@tulane.edu; Tel: 504-988-2475






 Description of files


(i) the unified descriptor table analyzed here, derived from 8000molecular_descriptors_to_be_used.xlsx (with zero-variance columns removed) and merged with 22_molecular_descriptors_graph_theory.xlsx, together with a dictionary mapping column identifiers d1–d6977 to the original descriptor names;
(ii) the descriptor-generation script that produced the 16,382-column source table ([script file name, version or commit]; RDKit [version]; Python [version]);
(iii) the Python workflow (Python 3.13, scikit-learn 1.9.1, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, matplotlib 3.11.2), comprising amino_acid_pipeline_v4.py (LOOCV benchmark and null controls), amino_acid_unsupervised_v4.py (clustering, validation, univariate, and descriptive analyses), and make_figures_v4.py (all figures);
(iv) the parallel R workflow, amino_acid_workflow.R (packages cluster and e1071) [101]; and
(v) the full 39 × 17 metrics tables for both feature sets, out-of-fold predictions, and the 100-seed null-control results.
The synthetic null control is fully specified by its generating code, numpy.random.RandomState(42).normal(0, 1, size=(20, 6977)), and can be regenerated independently of the distributed data. The additional null-control replicates (Section 2.3) are generated in the same way using seeds 0–99.
