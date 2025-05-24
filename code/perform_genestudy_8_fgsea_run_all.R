## ---------------------------------------------------------------------
## MASTER SCRIPT FOR COMPLETE FGSEA ANALYSIS PIPELINE
## ---------------------------------------------------------------------

# 1) Install & load required packages
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
required_pkgs <- c(
  "fgsea","msigdbr","dplyr","ggplot2","tidyr","pheatmap",
  "stringr","RColorBrewer","openxlsx","gridExtra","UpSetR"
)
bioc_pkgs <- c("fgsea","msigdbr","org.Hs.eg.db","ReactomePA","clusterProfiler")
for(pkg in required_pkgs) if(!requireNamespace(pkg,quietly=TRUE)) install.packages(pkg)
for(pkg in bioc_pkgs)   if(!requireNamespace(pkg,quietly=TRUE)) BiocManager::install(pkg)

library(fgsea);    library(msigdbr); library(dplyr);  library(ggplot2)
library(tidyr);    library(pheatmap); library(stringr);library(RColorBrewer)
library(openxlsx); library(gridExtra);library(org.Hs.eg.db)
library(ReactomePA); library(clusterProfiler)

# —————————————————————————————————————————————————————————————
#  PROJECT CONFIGURATION
# —————————————————————————————————————————————————————————————

# where your code lives:
code_root    <- "/home/cko4/2025_asd_multimodal/code"

# where your data lives:
data_root    <- "/home/data/2025_asd_wgs"
results_dir  <- file.path(data_root, "skat_results")
output_dir   <- file.path(data_root, "fgsea_results")
data_dir     <- file.path(data_root, "data")
reference_dir<- file.path(data_root, "reference")

if(!dir.exists(output_dir)) dir.create(output_dir, recursive=TRUE)

# source core helpers
source(file.path(code_root, "perform_genestudy_7_fgsea_corefunctions.R"))
source(file.path(code_root, "perform_genestudy_7_fgsea_ppifunctions.R"))
source(file.path(code_root, "perform_genestudy_7_fgsea_sfarifunctions.R"))

# Loader for raw SFARI CSV (no args)
load_sfari_database <- function() {
  sf <- file.path(reference_dir,
    "SFARI-Gene_genes_04-03-2025release_04-21-2025export.csv")
  if(!file.exists(sf)) stop("SFARI CSV not found: ", sf)
  read.csv(sf, stringsAsFactors=FALSE)
}

# Utility to flatten list columns before CSV export
prepare_dataframe_for_csv <- function(df) {
  for(col in names(df)) if(is.list(df[[col]])) {
    df[[col]] <- sapply(df[[col]], function(x)
      if(length(x)==0) "" else paste(x, collapse=","))
  }
  df
}

# ----------------------------------------------------------------------
# Main pipeline function
# ----------------------------------------------------------------------
run_complete_analysis <- function() {
  cat("\n=== FGSEA PIPELINE START ===\n")

  # 1) SKAT results
  cat("1) Loading SKAT results...\n")
  sk <- load_cluster_specific_genes()

  # 2) All gene sets
  cat("2) Loading gene sets...\n")
  pw <- load_all_gene_sets()
  asd_gene_sets <- load_asd_gene_sets() # Store specifically for overlaps

  # 3) fGSEA per cluster
  cat("3) Running fGSEA analyses...\n")
  all_pos <- list(); all_std <- list()
  for(cl in 1:4) {
    all_pos[[cl]] <- run_fgsea_analysis(cl, pw, sk, score_type="pos")
    all_std[[cl]] <- run_fgsea_analysis(cl, pw, sk, score_type="std")
  }
  names(all_pos) <- names(all_std) <- paste0("cluster",1:4)
  results_fgsea <- all_std

  # 4) Save combined
  cat("4) Saving fGSEA results...\n")
  pos_df <- do.call(rbind, all_pos) %>% prepare_dataframe_for_csv()
  std_df <- do.call(rbind, all_std) %>% prepare_dataframe_for_csv()
  write.csv(pos_df, file.path(output_dir,
            "fgsea_all_clusters_positive_results.csv"), row.names=FALSE)
  write.csv(std_df, file.path(output_dir,
            "fgsea_all_clusters_bidirectional_results.csv"), row.names=FALSE)

  # 5) Create enhanced visualizations
  cat("5) Creating enhanced visualizations...\n")
    
  # 5.1) ASD-specific gene set heatmap
  cat("  5.1) Creating ASD gene set heatmap...\n")
  asd_heatmap <- create_asd_gene_set_heatmap(results_fgsea)
  if(!is.null(asd_heatmap)) {
    pdf(file.path(output_dir, "asd_gene_sets_heatmap.pdf"), width=10, height=8)
    grid::grid.newpage(); grid::grid.draw(asd_heatmap$gtable)
    dev.off()
  }
    
  # 5.2) Category-specific heatmaps
  cat("  5.2) Creating pathway category heatmaps...\n")
  categories <- list(
    "Synaptic_signaling" = grep("SYNAP|TRANSMITTER|VESICLE|MEMBRANE|CHANNEL", names(pw), value=TRUE),
    "Immune_signaling"   = grep("IMMUNE|CYTOKINE|LEUKOCYTE|T_CELL|B_CELL", names(pw), value=TRUE),
    "Protein_translation"= grep("PROTEIN|TRANSLATION|RIBOSOM", names(pw), value=TRUE)
  )

  for(cat_nm in names(categories)) {
    hm <- create_enrichment_heatmap(results_fgsea, cat_nm, categories[[cat_nm]])
    if(!is.null(hm)) {
      pdf(file.path(output_dir, paste0(cat_nm,"_heatmap.pdf")), width=10, height=8)
      grid::grid.newpage(); grid::grid.draw(hm$gtable)
      dev.off()
    }
  }

  # Create and save the enhanced bubble plot
  cat("  Creating enhanced bubble plot...\n")
  enhanced_bp <- create_enrichment_bubble_plot(results_fgsea, top_n=5)
  if (!is.null(enhanced_bp)) {
    ggsave(file.path(output_dir, "top_pathways_bubble_plot_enhanced.pdf"), enhanced_bp, width=12, height=10)
    cat("  Bubble plot saved successfully\n")
  } else {
    cat("  Failed to create bubble plot\n")
  }
    
  # 5.3) Gene overlap visualization 
  cat("  5.3) Creating gene set overlap visualization...\n")
  gene_overlap_plot <- create_gene_overlap_plot(sk, asd_gene_sets)
  if(!is.null(gene_overlap_plot)) {
    ggsave(file.path(output_dir, "asd_gene_set_overlap.pdf"), gene_overlap_plot, width=10, height=8)
  }
    
  # 5.4) Multi-panel figure
  cat("  5.4) Creating multi-panel figure...\n")
  multi_panel <- create_multipanel_figure(results_fgsea, asd_gene_sets, sk)
  if(!is.null(multi_panel) && all(sapply(multi_panel$panels, function(x) !is.null(x)))) {
    pdf(file.path(output_dir, "comprehensive_enrichment_analysis.pdf"), width=16, height=14)
    grid.arrange(grobs=multi_panel$panels, layout_matrix=multi_panel$layout)
    dev.off()
  }
  
  # 5.5) Create multi-panel figure
  cat("  5.5) Creating multi-panel figure...\n")
  multi_panel <- create_multipanel_figure(results_fgsea, asd_gene_sets, sk)
  pdf(file.path(output_dir, "comprehensive_enrichment_analysis.pdf"), width=16, height=14)
  grid.arrange(grobs=multi_panel$panels, layout_matrix=multi_panel$layout)
  dev.off()
  
  # 6) Compare clusters
  cat("6) Comparing enrichment across clusters...\n")
  compare_clusters_enrichment(results_fgsea)

  # 7) SFARI & ASD analyses
  cat("7) Running SFARI & ASD-specific analyses...\n")
  sfari_df <- load_sfari_database()
  asd_res  <- run_additional_analyses()

  # 8) PPI analysis
  cat("8) Running PPI network analysis...\n")
  ppi_res <- run_ppi_analysis(
    cluster_specific_genes = sk,
    asd_gene_sets          = asd_gene_sets
  )

  # 9) SFARI annotation of SKAT table
  cat("9) Annotating SKAT genes with SFARI...\n")
  gf <- load_gene_functions(cluster_specific_genes = sk)
  add_sfari_annotations(
    cluster_specific_genes = sk,
    sfari_data             = sfari_df,
    gene_functions         = gf,
    output_dir             = output_dir  # Add this parameter
  )

  # 10) Summary report
  cat("10) Generating summary report...\n")
  create_summary_report(
    results_fgsea,
    ppi_res,
    asd_res$sfari_enrichment,
    asd_res$asd_overlaps
  )

  cat("\n=== PIPELINE COMPLETE: results →", output_dir, "===\n")
}

# execute
run_complete_analysis()