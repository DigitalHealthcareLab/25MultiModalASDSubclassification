# ASD Gene Set Enrichment Heatmap Generator
# ==========================================
# This standalone script creates an ASD gene set enrichment heatmap
# similar to those in Buch et al. 2023 and Bertelsen et al. 2021.

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, tidyr, ggplot2, pheatmap, RColorBrewer, stringr)

# ---------------------------------------------------------
# CONFIGURATION - MODIFY THESE PATHS AS NEEDED
# ---------------------------------------------------------
base_path <- "/home/data/2025_asd_wgs"  # Update this to your project path
output_dir <- file.path(base_path, "fgsea_results")
figures_dir <- file.path(base_path, "figures")

# Create figures directory if it doesn't exist
dir.create(figures_dir, showWarnings = FALSE)

# ---------------------------------------------------------
# LOAD fGSEA RESULTS FROM CSV FILES
# ---------------------------------------------------------
# Function to load fGSEA results for a specific cluster
load_fgsea_results <- function(cluster_num, score_type = "pos") {
  file_path <- file.path(output_dir, 
                         sprintf("fgsea_cluster%d_%s_results.csv", 
                                 cluster_num, score_type))
  
  if (!file.exists(file_path)) {
    warning("Results file not found: ", file_path)
    return(NULL)
  }
  
  results <- read.csv(file_path)
  results$cluster <- cluster_num
  
  # Convert leadingEdge from string back to list
  if ("leadingEdge" %in% colnames(results)) {
    results$leadingEdge <- strsplit(as.character(results$leadingEdge), ",")
  }
  
  return(results)
}

# Load results for all clusters
all_results <- list()
for (cluster in 1:4) {
  results <- load_fgsea_results(cluster)
  if (!is.null(results)) {
    all_results[[paste0("cluster_", cluster)]] <- results
  }
}

# Check if we have any results
if (length(all_results) == 0) {
  stop("No fGSEA results found. Please check your file paths.")
}

# ---------------------------------------------------------
# CREATE ASD GENE SET ENRICHMENT HEATMAP
# ---------------------------------------------------------
create_asd_gene_set_heatmap <- function(all_results) {
  # Define ASD-related gene sets we want to include
  asd_sets <- c(
    "ASD_downregulated", "FMRP_interacting", "Syndromic", 
    "ASD_SPARK", "ASD_RDNV", "ASD_Grove", "ASD_upregulated"
  )
  
  # Filter results for ASD gene sets
  asd_results <- do.call(rbind, all_results) %>%
    filter(grepl(paste(asd_sets, collapse="|"), pathway))
  
  if(nrow(asd_results) == 0) {
    warning("No ASD gene set results found")
    return(NULL)
  }
  
  # Standardize pathway names to match expected sets
  asd_results <- asd_results %>%
    mutate(
      gene_set = case_when(
        grepl("ASD_downregulated", pathway) ~ "ASD_downregulated",
        grepl("FMRP_interacting", pathway) ~ "FMRP-interacting",
        grepl("Syndromic|SFARI", pathway) ~ "Syndromic (SFARI)",
        grepl("ASD_SPARK", pathway) ~ "ASD SPARK",
        grepl("ASD_RDNV", pathway) ~ "ASD RDNV",
        grepl("ASD_Grove", pathway) ~ "ASD Grove et al.",
        grepl("ASD_upregulated", pathway) ~ "ASD upregulated",
        TRUE ~ pathway
      )
    )
  
  # Create a wide format table for heatmap
  enrichment_matrix <- asd_results %>%
    mutate(
      # Calculate enrichment score as -log10(FDR) * sign(NES)
      # Ensure we have a minimum FDR to avoid infinite values
      score = -log10(pmax(padj, 1e-10)) * sign(NES)
    ) %>%
    select(gene_set, cluster, score) %>%
    # Keep only the last occurrence of each gene set for each cluster
    # (in case there are duplicates)
    group_by(gene_set, cluster) %>%
    slice_tail(n = 1) %>%
    ungroup() %>%
    # Create wide format
    pivot_wider(
      names_from = cluster,
      values_from = score,
      values_fill = 0
    ) %>%
    as.data.frame()
  
  # Set gene sets as row names
  rownames(enrichment_matrix) <- enrichment_matrix$gene_set
  enrichment_matrix$gene_set <- NULL
  
  # Ensure all clusters are represented (1-4)
  for (i in 1:4) {
    if (!paste0(i) %in% colnames(enrichment_matrix)) {
      enrichment_matrix[[paste0(i)]] <- 0
    }
  }
  
  # Reorder columns to ensure clusters are in order
  enrichment_matrix <- enrichment_matrix[, as.character(1:4)]
  
  # Ensure ASD gene sets are in the desired order
  desired_order <- c(
    "ASD_downregulated", "FMRP-interacting", "Syndromic (SFARI)", 
    "ASD SPARK", "ASD RDNV", "ASD Grove et al.", "ASD upregulated"
  )
  
  # Filter and order rows by desired_order
  enrichment_matrix <- enrichment_matrix[intersect(desired_order, rownames(enrichment_matrix)), ]
  
  # Create the heatmap
  # Use a blue-white-red color scale with bounds similar to Buch et al.
  color_breaks <- seq(-10, 10, length.out = 100)
  color_palette <- colorRampPalette(c("blue4", "blue", "white", "red", "red4"))(100)
  
  # Create heatmap with significance indicators
  # Marking cells where |score| > 1.3 (approx. FDR < 0.05)
  significant_cells <- matrix(
    ifelse(
      abs(as.matrix(enrichment_matrix)) >= 1.3,
      "*",
      ""
    ),
    nrow = nrow(enrichment_matrix)
  )
  
  # Print actual values in console for debugging
  print("Enrichment scores matrix:")
  print(enrichment_matrix)
  
  # Create heatmap with significance indicators
  heatmap <- pheatmap(
    enrichment_matrix,
    main = "ASD Gene Set Enrichment",
    color = color_palette,
    breaks = color_breaks,
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    border_color = "black",
    cellwidth = 40,
    cellheight = 25,
    fontsize_row = 10,
    fontsize_col = 12,
    angle_col = 0,
    display_numbers = significant_cells,
    number_color = "black",
    silent = TRUE
  )
  
  return(heatmap)
}

# ---------------------------------------------------------
# CREATE A SECOND VISUALIZATION - SYNAPTIC SIGNALING PATHWAYS
# ---------------------------------------------------------
create_synaptic_signaling_heatmap <- function(all_results) {
  # Define synaptic-related terms to filter pathways
  synaptic_terms <- c(
    "SYNAP", "VESICLE", "NEUROTRANSMITTER", "ION_CHANNEL", 
    "MEMBRANE_POTENTIAL", "GPCR", "GLUTAMATE", "AXON", "POSTSYNAPTIC", 
    "PRESYNAPTIC", "TRANSMISSION"
  )
  
  # Combine all results
  combined_results <- do.call(rbind, all_results)
  
  # Filter for synaptic pathways
  synaptic_results <- combined_results %>%
    filter(grepl(paste(synaptic_terms, collapse="|"), pathway, ignore.case=TRUE))
  
  if(nrow(synaptic_results) == 0) {
    warning("No synaptic signaling pathways found")
    return(NULL)
  }
  
  # Standardize pathway names for cleaner display
  synaptic_results <- synaptic_results %>%
    mutate(
      pathway_clean = case_when(
        grepl("SECRETORY_VESICLE", pathway, ignore.case=TRUE) ~ "Secretory vesicle",
        grepl("PRESYNAPTIC.*ACTIVE", pathway, ignore.case=TRUE) ~ "Presynaptic active zone",
        grepl("POSTSYNAPTIC.*MEMBRANE", pathway, ignore.case=TRUE) ~ "Postsynaptic membrane",
        grepl("ION_CHANNEL", pathway, ignore.case=TRUE) ~ "Ion channel complex",
        grepl("CHEMICAL_SYNAPTIC_TRANSMISSION", pathway, ignore.case=TRUE) ~ "Chemical synaptic transmission",
        grepl("REGULATION.*MEMBRANE_POTENTIAL", pathway, ignore.case=TRUE) ~ "Regulation of membrane potential",
        grepl("NEUROTRANSMITTER_SECRETION", pathway, ignore.case=TRUE) ~ "Neurotransmitter secretion",
        grepl("GPCR_SIGNALING", pathway, ignore.case=TRUE) ~ "GPCR signaling pathway",
        grepl("SYNAPSE_ORGANIZATION", pathway, ignore.case=TRUE) ~ "Synapse organization",
        grepl("CAMP_MEDIATED_SIGNALING", pathway, ignore.case=TRUE) ~ "cAMP-mediated signaling",
        grepl("GLUTAMATE_RECEPTOR", pathway, ignore.case=TRUE) ~ "Glutamate receptor signaling pathway",
        TRUE ~ gsub("_", " ", str_replace_all(pathway, "^.*?_", ""))
      )
    )
  
  # Select top pathways for each category
  top_synaptic_pathways <- synaptic_results %>%
    # Take the top pathway by significance for each clean name
    group_by(pathway_clean) %>%
    arrange(padj) %>%
    slice_head(n = 1) %>%
    ungroup() %>%
    # Get the top pathways overall
    arrange(padj) %>%
    slice_head(n = 12) %>%
    pull(pathway)
  
  # Filter results to keep only top pathways
  filtered_results <- synaptic_results %>%
    filter(pathway %in% top_synaptic_pathways)
  
  # Create enrichment matrix
  enrichment_matrix <- filtered_results %>%
    mutate(
      # Calculate enrichment score
      score = -log10(pmax(padj, 1e-10)) * sign(NES)
    ) %>%
    select(pathway_clean, cluster, score) %>%
    group_by(pathway_clean, cluster) %>%
    slice_tail(n = 1) %>%
    ungroup() %>%
    pivot_wider(
      names_from = cluster,
      values_from = score,
      values_fill = 0
    ) %>%
    as.data.frame()
  
  # Set pathway names as row names
  rownames(enrichment_matrix) <- enrichment_matrix$pathway_clean
  enrichment_matrix$pathway_clean <- NULL
  
  # Ensure all clusters are represented
  for (i in 1:4) {
    if (!paste0(i) %in% colnames(enrichment_matrix)) {
      enrichment_matrix[[paste0(i)]] <- 0
    }
  }
  
  # Reorder columns
  enrichment_matrix <- enrichment_matrix[, as.character(1:4)]
  
  # Create heatmap
  color_breaks <- seq(-10, 10, length.out = 100)
  color_palette <- colorRampPalette(c("blue4", "blue", "white", "red", "red4"))(100)
  
  # Mark significant results
  significant_cells <- matrix(
    ifelse(
      abs(as.matrix(enrichment_matrix)) >= 1.3,
      "*",
      ""
    ),
    nrow = nrow(enrichment_matrix)
  )
  
  # Print values for debugging
  print("Synaptic pathways enrichment scores:")
  print(enrichment_matrix)
  
  # Create the heatmap
  heatmap <- pheatmap(
    enrichment_matrix,
    main = "Synaptic Signaling Gene Sets",
    color = color_palette,
    breaks = color_breaks,
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    border_color = "black",
    cellwidth = 40,
    cellheight = 20,
    fontsize_row = 9,
    fontsize_col = 12,
    angle_col = 0,
    display_numbers = significant_cells,
    number_color = "black",
    silent = TRUE
  )
  
  return(heatmap)
}

# ---------------------------------------------------------
# CREATE IMMUNE SIGNALING HEATMAP
# ---------------------------------------------------------
create_immune_signaling_heatmap <- function(all_results) {
  # Define immune-related terms to filter pathways
  immune_terms <- c(
    "IMMUNE", "CYTOKINE", "LEUKOCYTE", "T_CELL", "B_CELL",
    "CHEMOTAXIS", "INTERLEUKIN", "INFLAMMATORY", "RESPONSE_TO_CYTOKINE"
  )
  
  # Combine all results
  combined_results <- do.call(rbind, all_results)
  
  # Filter for immune pathways
  immune_results <- combined_results %>%
    filter(grepl(paste(immune_terms, collapse="|"), pathway, ignore.case=TRUE))
  
  if(nrow(immune_results) == 0) {
    warning("No immune signaling pathways found")
    return(NULL)
  }
  
  # Standardize pathway names
  immune_results <- immune_results %>%
    mutate(
      pathway_clean = case_when(
        grepl("CYTOKINE_RECEPTOR_BINDING", pathway, ignore.case=TRUE) ~ "Cytokine receptor binding",
        grepl("CYTOKINE_ACTIVITY", pathway, ignore.case=TRUE) ~ "Cytokine activity",
        grepl("CYTOKINE_BINDING", pathway, ignore.case=TRUE) ~ "Cytokine binding",
        grepl("CYTOKINE_RECEPTOR_ACTIVITY", pathway, ignore.case=TRUE) ~ "Cytokine receptor activity",
        grepl("PROTEASE_BINDING", pathway, ignore.case=TRUE) ~ "Protease binding",
        grepl("LEUKOCYTE_CHEMOTAXIS", pathway, ignore.case=TRUE) ~ "Leukocyte chemotaxis",
        grepl("RESPONSE_TO_TOXIC_SUBSTANCE", pathway, ignore.case=TRUE) ~ "Response to toxic substance",
        grepl("CYTOKINE_PRODUCTION", pathway, ignore.case=TRUE) ~ "Cytokine production",
        grepl("T_CELL_RECEPTOR", pathway, ignore.case=TRUE) ~ "T cell receptor signaling pathway",
        grepl("RESPONSE_TO_CYTOKINE", pathway, ignore.case=TRUE) ~ "Response to cytokine",
        grepl("IMMUNE_RESPONSE_TRANSDUCTION", pathway, ignore.case=TRUE) ~ "Immune response transduction",
        grepl("IMMUNE_RESPONSE", pathway, ignore.case=TRUE) ~ "Immune response",
        TRUE ~ gsub("_", " ", str_replace_all(pathway, "^.*?_", ""))
      )
    )
  
  # Select top pathways for each category
  top_immune_pathways <- immune_results %>%
    group_by(pathway_clean) %>%
    arrange(padj) %>%
    slice_head(n = 1) %>%
    ungroup() %>%
    arrange(padj) %>%
    slice_head(n = 12) %>%
    pull(pathway)
  
  # Filter results to keep only top pathways
  filtered_results <- immune_results %>%
    filter(pathway %in% top_immune_pathways)
  
  # Create enrichment matrix
  enrichment_matrix <- filtered_results %>%
    mutate(
      score = -log10(pmax(padj, 1e-10)) * sign(NES)
    ) %>%
    select(pathway_clean, cluster, score) %>%
    group_by(pathway_clean, cluster) %>%
    slice_tail(n = 1) %>%
    ungroup() %>%
    pivot_wider(
      names_from = cluster,
      values_from = score,
      values_fill = 0
    ) %>%
    as.data.frame()
  
  # Set pathway names as row names
  rownames(enrichment_matrix) <- enrichment_matrix$pathway_clean
  enrichment_matrix$pathway_clean <- NULL
  
  # Ensure all clusters are represented
  for (i in 1:4) {
    if (!paste0(i) %in% colnames(enrichment_matrix)) {
      enrichment_matrix[[paste0(i)]] <- 0
    }
  }
  
  # Reorder columns
  enrichment_matrix <- enrichment_matrix[, as.character(1:4)]
  
  # Create heatmap
  color_breaks <- seq(-10, 10, length.out = 100)
  color_palette <- colorRampPalette(c("blue4", "blue", "white", "red", "red4"))(100)
  
  # Mark significant results
  significant_cells <- matrix(
    ifelse(
      abs(as.matrix(enrichment_matrix)) >= 1.3,
      "*",
      ""
    ),
    nrow = nrow(enrichment_matrix)
  )
  
  # Print values for debugging
  print("Immune pathways enrichment scores:")
  print(enrichment_matrix)
  
  # Create the heatmap
  heatmap <- pheatmap(
    enrichment_matrix,
    main = "Immune Signaling Gene Sets",
    color = color_palette,
    breaks = color_breaks,
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    border_color = "black",
    cellwidth = 40,
    cellheight = 20,
    fontsize_row = 9,
    fontsize_col = 12,
    angle_col = 0,
    display_numbers = significant_cells,
    number_color = "black",
    silent = TRUE
  )
  
  return(heatmap)
}

# ---------------------------------------------------------
# SAVE THE HEATMAPS
# ---------------------------------------------------------
# Generate and save the ASD gene set heatmap
asd_heatmap <- create_asd_gene_set_heatmap(all_results)
if (!is.null(asd_heatmap)) {
  pdf(file.path(figures_dir, "asd_gene_set_enrichment.pdf"), width = 10, height = 7)
  grid::grid.newpage()
  grid::grid.draw(asd_heatmap$gtable)
  dev.off()
  cat("ASD gene set heatmap saved to:", file.path(figures_dir, "asd_gene_set_enrichment.pdf"), "\n")
}

# Generate and save the synaptic signaling heatmap
synaptic_heatmap <- create_synaptic_signaling_heatmap(all_results)
if (!is.null(synaptic_heatmap)) {
  pdf(file.path(figures_dir, "synaptic_signaling_enrichment.pdf"), width = 10, height = 8)
  grid::grid.newpage()
  grid::grid.draw(synaptic_heatmap$gtable)
  dev.off()
  cat("Synaptic signaling heatmap saved to:", file.path(figures_dir, "synaptic_signaling_enrichment.pdf"), "\n")
}

# Generate and save the immune signaling heatmap
immune_heatmap <- create_immune_signaling_heatmap(all_results)
if (!is.null(immune_heatmap)) {
  pdf(file.path(figures_dir, "immune_signaling_enrichment.pdf"), width = 10, height = 8)
  grid::grid.newpage()
  grid::grid.draw(immune_heatmap$gtable)
  dev.off()
  cat("Immune signaling heatmap saved to:", file.path(figures_dir, "immune_signaling_enrichment.pdf"), "\n")
}

# Print session info for reproducibility
cat("\nSession Info:\n")
print(sessionInfo())

# ---------------------------------------------------------
#%%
# Test script for creating the enhanced bubble plot
library(dplyr)
library(ggplot2)
library(stringr)

# Function to create enhanced bubble plot
create_enrichment_bubble_plot <- function(fgsea_data, top_n = 5, category_filter = NULL) {
  # Ensure we have a proper data frame to work with
  if (is.list(fgsea_data) && !is.data.frame(fgsea_data)) {
    combined_results <- do.call(rbind, fgsea_data)
  } else {
    combined_results <- fgsea_data
  }
  
  # Add some error checking
  if (nrow(combined_results) == 0) {
    stop("No data provided for bubble plot")
  }
  
  # Ensure required columns exist
  required_cols <- c("pathway", "cluster", "padj", "NES")
  missing_cols <- setdiff(required_cols, colnames(combined_results))
  
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
  }
  
  # Filter OUT ASD-related gene sets and High Confidence
  combined_results <- combined_results %>%
    filter(!grepl("ASD|SFARI|FMRP|RDNV|High_Confidence", pathway, ignore.case = TRUE))
  
  # Filter by category if specified
  if (!is.null(category_filter)) {
    combined_results <- combined_results %>%
      filter(grepl(category_filter, pathway))
  }
  
  # Get top pathways for each cluster
  top_pathways <- combined_results %>%
    group_by(cluster) %>%
    arrange(padj) %>%
    slice_head(n = top_n) %>%
    ungroup()
  
  # Clean pathway names for better display
  top_pathways$pathway_clean <- str_replace_all(top_pathways$pathway, 
                                           "^(HALLMARK_|REACTOME_|KEGG_|GO_|C[25]\\.|GOMF_|GOBP_|GOCC_)", "")
  top_pathways$pathway_clean <- str_replace_all(top_pathways$pathway_clean, "_", " ")
  
  # Categorize the remaining pathways - improved categories with better naming
  top_pathways <- top_pathways %>%
    mutate(category = case_when(
      grepl("CARDIAC|HEART|MUSCLE|NODE|DEPOLAR|CONTRACTION", pathway, ignore.case = TRUE) ~ "Cardiac & Muscle Pathways",
      grepl("COLLAGEN|JUNCTION|WOUND|PROJECTION|ASSEMBLY", pathway, ignore.case = TRUE) ~ "Cell Structure Pathways",
      grepl("ION|CALCIUM|CHANNEL|TRANSPORTER", pathway, ignore.case = TRUE) ~ "Ion Transport Pathways",
      grepl("NEUROMUSCULAR|ELECTRON|TRANSPORT", pathway, ignore.case = TRUE) ~ "Neural Pathways",
      TRUE ~ "Other Cellular Pathways"
    ))
  
  # Create enhanced bubble plot with up/down regulation colors
  bubble_plot <- ggplot(top_pathways, aes(x = factor(cluster), y = reorder(pathway_clean, -padj))) +
    # Use circles with different colors for up/down regulation
    geom_point(aes(size = abs(NES), color = NES), shape = 16) +
    scale_color_gradient2(
      low = "blue", 
      mid = "white", 
      high = "red", 
      midpoint = 0, 
      name = "Direction & NES",
      guide = guide_colorbar(title.position = "top")
    ) +
    scale_size_continuous(name = "Magnitude of\nEnrichment", range = c(3, 10)) +
    facet_grid(category ~ ., scales = "free_y", space = "free") +
    labs(x = "Cluster", y = "",
       title = "Top enriched pathways across clusters") +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 8),
      legend.position = "right",
      panel.grid.major = element_line(color = "gray90"),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold"),
      strip.background = element_rect(fill = "gray95"),
      # Fix for cutoff labels - increased right margin substantially
      strip.text.y = element_text(margin = margin(r = 15, l = 15)),
      plot.margin = margin(10, 30, 10, 10) # Increased right margin
    )
  
  return(bubble_plot)
}

# Set up the correct paths based on your project structure
base_path <- "/home/data/2025_asd_wgs"
output_dir <- file.path(base_path, "fgsea_results")
results_file <- file.path(output_dir, "top_pathways_per_cluster.csv")

# Load data
if (file.exists(results_file)) {
  cat("Loading fGSEA results from:", results_file, "\n")
  fgsea_results <- read.csv(results_file)
  
  # Check data structure
  cat("Data structure:\n")
  str(fgsea_results)
  
  # Create the bubble plot
  bubble_plot <- create_enrichment_bubble_plot(fgsea_results)
  
  # Save the plot
  output_file <- file.path(output_dir, "functional_pathways_bubble_plot.pdf")
  ggsave(output_file, bubble_plot, width=16, height=10)
  cat("Bubble plot saved to:", output_file, "\n")
} else {
  cat("Error: Results file not found:", results_file, "\n")
}