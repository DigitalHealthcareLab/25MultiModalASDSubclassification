# Purpose: Wrap-up script to explore top SKAT results and publication summaries
#%%
# -----------------------------
# Setup
# -----------------------------
base_path <- "/home/data/2025_asd_wgs"
results_dir <- file.path(base_path, "skat_results")

# -----------------------------
# 1. Load Top Genes Summary
# -----------------------------
cat("========== Examining Top Genes Summary ==========\n")
summary_file <- file.path(results_dir, "SKAT_top_genes_summary.csv")
if (file.exists(summary_file)) {
  top_genes <- read.csv(summary_file)
  cat("Total number of top genes:", nrow(top_genes), "\n")
  cat("Number of genes per cluster:\n")
  print(table(top_genes$Cluster))

  for(cluster in sort(unique(top_genes$Cluster))) {
    cat("\n----- Top Genes for Cluster", cluster, "-----\n")
    cluster_genes <- top_genes[top_genes$Cluster == cluster, ]
    cluster_genes <- cluster_genes[order(cluster_genes$P_value), ]
    top_n <- min(10, nrow(cluster_genes))
    cat("Showing top", top_n, "genes:\n")
    print(cluster_genes[1:top_n, c("Gene", "P_value", "FDR", "N_variants", "N_samples")])
  }
} else {
  cat("File 'SKAT_top_genes_summary.csv' not found\n")
}

#%%
# -----------------------------
# 2. Load Cluster-Specific Genes
# -----------------------------
cat("\n\n========== Examining Cluster-Specific Genes ==========\n")
specific_file <- file.path(results_dir, "SKAT_all_cluster_specific_genes.csv")
if (file.exists(specific_file)) {
  specific_genes <- read.csv(specific_file)
  cat("Total number of cluster-specific genes:", nrow(specific_genes), "\n")
  cat("Number of specific genes per cluster:\n")
  print(table(specific_genes$Cluster))

  for(cluster in sort(unique(specific_genes$Cluster))) {
    cat("\n----- Specific Genes for Cluster", cluster, "-----\n")
    cluster_specific <- specific_genes[specific_genes$Cluster == cluster, ]
    cluster_specific <- cluster_specific[order(cluster_specific$P_value), ]
    top_n <- min(10, nrow(cluster_specific))
    cat("Showing top", top_n, "genes:\n")
    print(cluster_specific[1:top_n, c("Gene", "P_value", "FDR", "N_variants")])
  }
} else {
  cat("File 'SKAT_all_cluster_specific_genes.csv' not found\n")
}

#%%
# -----------------------------
# 3. Check One Gene Across All Clusters
# -----------------------------
check_gene_across_clusters <- function(gene_name) {
  cat("\n========== Checking gene", gene_name, "across all clusters ==========\n")
  for(cluster in 1:4) {
    file_name <- file.path(results_dir, paste0("SKAT_cluster", cluster, "_all_results.csv"))
    if(file.exists(file_name)) {
      cluster_results <- read.csv(file_name)
      gene_results <- cluster_results[cluster_results$Gene == gene_name, ]
      if(nrow(gene_results) > 0) {
        cat("Cluster", cluster, "results:\n")
        print(gene_results[, c("Gene", "P_value", "FDR", "Bonferroni", "N_variants", "N_samples")])
      } else {
        cat("Gene", gene_name, "not found in cluster", cluster, "results\n")
      }
    } else {
      cat("File", file_name, "not found\n")
    }
  }
}

#%%
# -----------------------------
# 4. Find Most Promising Genes
# -----------------------------
find_promising_genes <- function(top_genes, specific_genes, n = 10) {
  cat("Top promising genes for each cluster based on significance:\n")
  for(cluster in sort(unique(top_genes$Cluster))) {
    cat("\nCluster", cluster, "candidates:\n")
    cluster_top <- top_genes[top_genes$Cluster == cluster, ]
    if(nrow(specific_genes) > 0) {
      cluster_specific <- specific_genes[specific_genes$Cluster == cluster, ]
      common_genes <- intersect(cluster_top$Gene, cluster_specific$Gene)
      if(length(common_genes) > 0) {
        cat("Genes that are both top significant AND cluster-specific:\n")
        for(gene in common_genes) {
          gene_data <- cluster_top[cluster_top$Gene == gene, ]
          cat("  -", gene, "(p =", gene_data$P_value, ", FDR =", gene_data$FDR, ")\n")
        }
      } else {
        cat("No genes are both top significant AND cluster-specific\n")
      }
    }
    cluster_top_sorted <- cluster_top[order(cluster_top$P_value), ]
    top_n <- min(n, nrow(cluster_top_sorted))
    cat("\nTop significant genes for this cluster:\n")
    for(i in 1:top_n) {
      cat("  -", cluster_top_sorted$Gene[i], 
          "(p =", cluster_top_sorted$P_value[i], 
          ", FDR =", cluster_top_sorted$FDR[i], 
          ", variants =", cluster_top_sorted$N_variants[i], ")\n")
    }
  }
}

#%%
# Example usage:
# check_gene_across_clusters("SHANK1")
# if (exists("top_genes") && exists("specific_genes")) {
#   find_promising_genes(top_genes, specific_genes)
# }

# find_promising_genes(top_genes, specific_genes, n = 10)
