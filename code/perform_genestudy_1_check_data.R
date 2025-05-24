# ==============================
# File: data_checks.R
# ==============================

# Author: Chanyoung Ko
# Date: 2025-04-18
# Description: Check phenotype and genotype data structure for SKAT analysis

base_path <- "/home/data/2025_asd_wgs"
setwd(base_path)

cat("Files in main directory:\n")
print(list.files())

pheno_files <- list.files(pattern = "participant")
cat("\nPhenotype files found:\n")
print(pheno_files)

if(length(pheno_files) > 0) {
  pheno_file <- file.path(base_path, pheno_files[1])
  cat("\nReading file:", pheno_file, "\n")
  pheno_data <- read.csv(pheno_file)
  cat("\nStructure of phenotype data:\n")
  str(pheno_data)
  cat("\nFirst few rows:\n")
  print(head(pheno_data))
  if("cluster" %in% colnames(pheno_data)) {
    cat("\nUnique clusters:\n")
    print(unique(pheno_data$cluster))
    cat("\nSamples per cluster:\n")
    print(table(pheno_data$cluster))
  }
  if("id" %in% colnames(pheno_data)) {
    cat("\nSample ID format check:\n")
    print(head(pheno_data$id))
    cat("Are IDs lowercase?", all(tolower(pheno_data$id) == pheno_data$id), "\n")
  }
}

if(dir.exists(file.path(base_path, "snp_matrices"))) {
  snp_files <- list.files(file.path(base_path, "snp_matrices"), pattern = "\\.csv$")
  cat("\nFound", length(snp_files), "gene files\n")
  if(length(snp_files) > 0) {
    sample_file <- file.path(base_path, "snp_matrices", snp_files[1])
    gene_data <- read.csv(sample_file, nrows = 5)
    cat("\nSample gene file:", sample_file, "\n")
    str(gene_data)
    print(head(colnames(gene_data), 10))
    print(head(rownames(gene_data)))
    print(head(gene_data[, 1]))
  }
}