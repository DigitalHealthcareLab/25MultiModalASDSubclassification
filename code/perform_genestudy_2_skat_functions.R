# ==============================
# File: skat_batch_functions.R
# ==============================

library(SKAT)

analyze_gene_batch <- function(start_index, end_index, cluster_to_test, pheno_file, base_path) {
  gene_files <- list.files(file.path(base_path, "snp_matrices"), pattern = "\\.csv$", full.names = TRUE)
  end_index <- min(end_index, length(gene_files))
  gene_files <- gene_files[start_index:end_index]

  pheno_data <- read.csv(file.path(base_path, pheno_file))
  pheno_data$Sample_ID <- toupper(pheno_data$id)
  pheno_data <- na.omit(pheno_data[, c("brain_integration", "brain_segregation", "sex", "cluster", "Sample_ID", "age", "fsiq", "vci")])
  pheno_data$target_status <- ifelse(pheno_data$cluster == cluster_to_test, 1, 0)

  null_model <- SKAT_Null_Model(target_status ~ brain_integration + brain_segregation + age + fsiq + vci, data = pheno_data, out_type = "D")

  results <- data.frame(Cluster = numeric(), Gene = character(), P_value = numeric(), N_variants = numeric(), N_samples = numeric(), stringsAsFactors = FALSE)

  for (file in gene_files) {
    gene_name <- sub("\\.csv$", "", basename(file))
    genotype_data <- read.csv(file)
    common_samples <- intersect(pheno_data$Sample_ID, genotype_data$Sample)
    if(length(common_samples) < 5) next

    phenotype_subset <- pheno_data[pheno_data$Sample_ID %in% common_samples, ]
    genotype_subset <- genotype_data[genotype_data$Sample %in% common_samples, ]
    Z <- as.matrix(genotype_subset[, -which(colnames(genotype_subset) == "Sample")])
    Z_ordered <- Z[match(phenotype_subset$Sample_ID, genotype_subset$Sample), , drop = FALSE]
    if(ncol(Z_ordered) == 0) next

    skat_result <- tryCatch(SKAT(Z_ordered, null_model, method = "davies"), error = function(e) NULL)
    if(!is.null(skat_result)) {
      results <- rbind(results, data.frame(
        Cluster = cluster_to_test,
        Gene = gene_name,
        P_value = skat_result$p.value,
        N_variants = ncol(Z_ordered),
        N_samples = nrow(Z_ordered),
        stringsAsFactors = FALSE))
    }
  }
  return(results)
}