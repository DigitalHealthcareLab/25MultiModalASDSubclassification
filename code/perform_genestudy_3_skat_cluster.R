# ==============================
# File: cluster_analysis_main.R
# ==============================

source("skat_batch_functions.R")

run_complete_analysis <- function(base_path, pheno_file) {
  gene_files <- list.files(file.path(base_path, "snp_matrices"), pattern = "\\.csv$")
  total_genes <- length(gene_files)
  batch_size <- 500
  num_batches <- ceiling(total_genes / batch_size)

  for (cluster in 1:4) {
    for (batch in 1:num_batches) {
      start_index <- (batch - 1) * batch_size + 1
      end_index <- batch * batch_size
      results <- analyze_gene_batch(start_index, end_index, cluster, pheno_file, base_path)
      result_file <- file.path(base_path, sprintf("skat_results/SKAT_cluster%d_batch_%d_%d.csv", cluster, start_index, end_index))
      write.csv(results, result_file, row.names = FALSE)
      cat(sprintf("Saved batch results: %s\n", result_file))
    }
  }
}

# run_complete_analysis("/home/data/2025_asd_wgs", "participant_clinical_brain_cluster_df.csv")