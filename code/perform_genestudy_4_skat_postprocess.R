# ==============================
# File: summary_postprocess.R
# ==============================

combine_cluster_results <- function(base_path, cluster) {
  pattern <- sprintf("SKAT_cluster%d_batch_", cluster)
  result_files <- list.files(file.path(base_path, "skat_results"), pattern = pattern, full.names = TRUE)
  all_results <- do.call(rbind, lapply(result_files, read.csv))

  all_results$FDR <- p.adjust(all_results$P_value, method = "BH")
  all_results$Bonferroni <- p.adjust(all_results$P_value, method = "bonferroni")

  output_file <- file.path(base_path, sprintf("skat_results/SKAT_cluster%d_all_results.csv", cluster))
  write.csv(all_results, output_file, row.names = FALSE)
  cat("Combined cluster results saved to:", output_file, "\n")
  return(all_results)
}

check_gene_across_clusters <- function(gene_name, base_path) {
  for(cluster in 1:4) {
    file_name <- file.path(base_path, sprintf("skat_results/SKAT_cluster%d_all_results.csv", cluster))
    if(file.exists(file_name)) {
      cluster_results <- read.csv(file_name)
      gene_results <- cluster_results[cluster_results$Gene == gene_name, ]
      if(nrow(gene_results) > 0) {
        cat("\nCluster", cluster, "results:\n")
        print(gene_results[, c("Gene", "P_value", "FDR", "Bonferroni", "N_variants", "N_samples")])
      }
    }
  }
}

create_cross_cluster_summary <- function(base_path, top_genes_by_cluster) {
  summary_table <- data.frame()
  for(cluster in names(top_genes_by_cluster)) {
    for(gene in top_genes_by_cluster[[cluster]]) {
      p_values <- sapply(1:4, function(c) {
        file <- file.path(base_path, sprintf("skat_results/SKAT_cluster%d_all_results.csv", c))
        if(file.exists(file)) {
          df <- read.csv(file)
          row <- df[df$Gene == gene, ]
          if(nrow(row) > 0) return(row$P_value[1])
        }
        return(NA)
      })
      summary_table <- rbind(summary_table, data.frame(
        Gene = gene,
        Top_Cluster = as.integer(cluster),
        P_Cluster1 = p_values[1],
        P_Cluster2 = p_values[2],
        P_Cluster3 = p_values[3],
        P_Cluster4 = p_values[4],
        stringsAsFactors = FALSE))
    }
  }
  write.csv(summary_table, file.path(base_path, "skat_results/cluster_specific_genes_publication.csv"), row.names = FALSE)
  return(summary_table)
}

annotate_top_genes <- function(summary_table, annotations_file) {
  annotations <- read.csv(annotations_file)
  annotated_results <- merge(summary_table, annotations, by = "Gene", all.x = TRUE)
  write.csv(annotated_results, "skat_results/annotated_cluster_genes.csv", row.names = FALSE)
  return(annotated_results)
}

create_top_gene_summary <- function(base_path, top_n = 20) {
  results_dir <- file.path(base_path, "skat_results")
  output_file <- file.path(results_dir, "SKAT_top_genes_summary.csv")

  top_gene_list <- list()

  for (cluster in 1:4) {
    file_path <- file.path(results_dir, sprintf("SKAT_cluster%d_all_results.csv", cluster))
    if (file.exists(file_path)) {
      df <- read.csv(file_path)
      df <- df[order(df$P_value), ]
      df$Cluster <- cluster
      top_gene_list[[as.character(cluster)]] <- head(df, top_n)
    }
  }

  top_genes_summary <- do.call(rbind, top_gene_list)
  top_genes_summary <- top_genes_summary[, c("Cluster", "Gene", "P_value", "FDR", "Bonferroni", "N_variants", "N_samples")]
  write.csv(top_genes_summary, output_file, row.names = FALSE)
  cat("✅ Top genes per cluster summary saved to:\n", output_file, "\n")
  return(top_genes_summary)
}
