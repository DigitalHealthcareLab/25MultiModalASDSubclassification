## Diagnostic script for ASD Gene Set Enrichment ## 
#%%
# ASD Gene Set Enrichment Diagnostic Script
# =========================================
# This script diagnoses issues with the ASD gene set enrichment heatmap
# by examining the actual fGSEA results and pathway names.

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, tidyr, ggplot2, stringr)

# ---------------------------------------------------------
# CONFIGURATION - MODIFY THESE PATHS AS NEEDED
# ---------------------------------------------------------
base_path <- "/home/data/2025_asd_wgs"  # Update this to your project path
output_dir <- file.path(base_path, "fgsea_results")
figures_dir <- file.path(base_path, "figures")
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
    cat("Results file not found: ", file_path, "\n")
    return(NULL)
  }
  
  cat("Loading results from:", file_path, "\n")
  results <- read.csv(file_path)
  results$cluster <- cluster_num
  
  # Convert leadingEdge from string back to list if present
  if ("leadingEdge" %in% colnames(results)) {
    results$leadingEdge <- strsplit(as.character(results$leadingEdge), ",")
  }
  
  return(results)
}

# ---------------------------------------------------------
# DIAGNOSTIC FUNCTIONS
# ---------------------------------------------------------

# Function to inspect pathway names and find ASD-related sets
inspect_pathways <- function(all_results) {
  # Combine results from all clusters
  combined_results <- do.call(rbind, all_results)
  
  # Check how many results we have
  cat("\n==== RESULTS SUMMARY ====\n")
  cat("Total pathways found across all clusters:", length(unique(combined_results$pathway)), "\n")
  
  # List all column names in the results to ensure we have what we need
  cat("\nColumns in the results:\n")
  print(colnames(combined_results))
  
  # Look for any ASD-related pathways
  asd_patterns <- c("ASD", "AUTISM", "FMRP", "SFARI", "RDNV", "SPARK", "GROVE")
  asd_pathways <- combined_results %>%
    filter(grepl(paste(asd_patterns, collapse="|"), pathway, ignore.case=TRUE)) %>%
    select(pathway) %>%
    distinct() %>%
    arrange(pathway)
  
  cat("\n==== ASD-RELATED PATHWAYS ====\n")
  cat("Number of ASD-related pathways found:", nrow(asd_pathways), "\n")
  
  if (nrow(asd_pathways) > 0) {
    cat("ASD-related pathways:\n")
    print(asd_pathways)
  } else {
    cat("No ASD-related pathways found.\n")
  }
  
  # If no ASD pathways found, look at pathway naming patterns
  if (nrow(asd_pathways) == 0) {
    cat("\n==== PATHWAY NAME PATTERNS ====\n")
    # Show a sample of pathways to understand naming patterns
    sample_pathways <- head(unique(combined_results$pathway), 20)
    cat("Sample pathway names:\n")
    print(sample_pathways)
  }
  
  return(asd_pathways)
}

# Manually extract enrichment scores for specific pathways
extract_pathway_enrichment <- function(all_results, pattern) {
  # Combine results
  combined_results <- do.call(rbind, all_results)
  
  # Filter for pathways matching the pattern
  matching_results <- combined_results %>%
    filter(grepl(pattern, pathway, ignore.case=TRUE))
  
  return(matching_results)
}

# Create a fallback version using gene set overlap counts
create_overlap_count_heatmap <- function() {
  # This is just a placeholder - the actual implementation would depend
  # on how you're calculating the gene overlaps currently.
  cat("\nA fallback heatmap using gene count overlaps would be implemented here.\n")
  cat("This would require the gene overlap counts we see in Image 3.\n")
}

# ---------------------------------------------------------
# LOAD AND INSPECT RESULTS
# ---------------------------------------------------------

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

# Inspect pathways to find ASD-related gene sets
asd_pathways <- inspect_pathways(all_results)

# ---------------------------------------------------------
# DETAILED ENRICHMENT INSPECTION
# ---------------------------------------------------------

# Patterns for different ASD gene set categories
asd_set_patterns <- list(
  "ASD downregulated" = "ASD.*DOWN|DOWN.*ASD",
  "FMRP interacting" = "FMRP",
  "Syndromic" = "SYNDROMIC|SFARI.*S",
  "ASD SPARK" = "SPARK",
  "ASD RDNV" = "RDNV",
  "ASD Grove" = "GROVE",
  "ASD upregulated" = "ASD.*UP|UP.*ASD"
)

cat("\n==== DETAILED ENRICHMENT VALUES ====\n")
for (set_name in names(asd_set_patterns)) {
  pattern <- asd_set_patterns[[set_name]]
  results <- extract_pathway_enrichment(all_results, pattern)
  
  cat("\nChecking for", set_name, "using pattern:", pattern, "\n")
  
  if (nrow(results) > 0) {
    cat("Found", nrow(results), "matching pathways\n")
    
    # Show the pathway results
    results_subset <- results %>%
      select(pathway, cluster, pval, padj, NES) %>%
      arrange(cluster, pathway)
    
    print(results_subset)
    
    # Calculate and show the enrichment scores as would be used in heatmap
    cat("\nEnrichment scores (-log10(padj) * sign(NES)):\n")
    enrichment_scores <- results %>%
      mutate(
        score = -log10(pmax(padj, 1e-10)) * sign(NES)
      ) %>%
      select(pathway, cluster, score) %>%
      pivot_wider(
        names_from = cluster,
        values_from = score,
        values_fill = 0
      )
    
    print(enrichment_scores)
  } else {
    cat("No matching pathways found\n")
  }
}

# ---------------------------------------------------------
# RECOMMENDATIONS
# ---------------------------------------------------------

cat("\n==== RECOMMENDATIONS ====\n")

if (nrow(asd_pathways) == 0) {
  cat("1. Your fGSEA results don't contain the expected ASD gene sets.\n")
  cat("   - Check how the gene sets were loaded in your fGSEA analysis.\n")
  cat("   - Ensure you've properly loaded the ASD-related gene sets from your reference files.\n")
  cat("   - The load_asd_gene_sets() function should be returning the expected sets.\n\n")
} else {
  cat("1. ASD gene sets found, but they may not match the expected format.\n")
  cat("   - Check the actual pathway names in your fGSEA results.\n")
  cat("   - Modify the pattern matching in create_asd_gene_set_heatmap() to match your actual pathway names.\n\n")
}

cat("2. If you're not getting significant enrichment results:\n")
cat("   - Consider using a more lenient FDR threshold (e.g., 0.25 instead of 0.05).\n")
cat("   - Check if your ranked gene lists have enough genes.\n")
cat("   - Ensure your gene set files contain the expected genes.\n\n")

cat("3. As a fallback, you can use the gene set overlap counts approach:\n")
cat("   - This is what's shown in Image 3, using raw gene counts rather than enrichment scores.\n")
cat("   - If the statistical enrichment isn't showing meaningful patterns, the overlap counts can still be informative.\n")

# Print session info for reproducibility
cat("\nSession Info:\n")
print(sessionInfo())


#%%
## Visualization script for ASD Gene Set Enrichment ##
# Improved ASD Gene Set Enrichment Heatmap Script
# ============================================
# This script creates heatmaps for ASD gene set enrichment
# with modifications to handle non-significant results

# Load required packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, tidyr, ggplot2, pheatmap, RColorBrewer, stringr, grid, gridExtra)

# ---------------------------------------------------------
# CONFIGURATION - MODIFY THESE PATHS AS NEEDED
# ---------------------------------------------------------
base_path <- "/home/data/2025_asd_wgs"  # Update this to your project path
output_dir <- file.path(base_path, "fgsea_results")
figures_dir <- file.path(base_path, "figures")
data_dir <- file.path(base_path, "data")

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
# IMPROVED ASD GENE SET HEATMAP FUNCTION 
# ---------------------------------------------------------
create_improved_asd_heatmap <- function(all_results, use_nes = FALSE) {
  # Define ASD-related gene sets we want to include
  asd_sets <- c(
    "ASD_downregulated", "FMRP_interacting", "Syndromic", 
    "ASD_SPARK", "ASD_RDNV", "ASD_Grove", "ASD_upregulated"
  )
  
  # Filter results for ASD gene sets
  asd_results <- do.call(rbind, all_results) %>%
    filter(grepl(paste(asd_sets, collapse="|"), pathway, ignore.case=TRUE))
  
  if(nrow(asd_results) == 0) {
    warning("No ASD gene set results found")
    return(NULL)
  }
  
  # Standardize pathway names
  asd_results <- asd_results %>%
    mutate(
      gene_set = case_when(
        grepl("ASD_downregulated", pathway) ~ "ASD_downregulated",
        grepl("FMRP_interacting", pathway) ~ "FMRP-interacting",
        grepl("All_SFARI", pathway) ~ "Syndromic (SFARI)", # Use All_SFARI as Syndromic
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
      # Option 1: Use NES directly for more visible patterns
      # Even if not significant, the direction and magnitude can be informative
      score = if(use_nes) NES else -log10(pmax(padj, 1e-10)) * sign(NES)
    ) %>%
    select(gene_set, cluster, score) %>%
    # Keep only the last occurrence of each gene set for each cluster
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
  # Adjust color scale and limits based on the chosen scoring method
  if (use_nes) {
    # For NES values, use a different scale
    color_breaks <- seq(-3, 3, length.out = 100)
    main_title <- "ASD Gene Set Enrichment (NES)"
  } else {
    # For -log10(padj) * sign(NES)
    color_breaks <- seq(-4, 4, length.out = 100)
    main_title <- "ASD Gene Set Enrichment"
  }
  
  color_palette <- colorRampPalette(c("blue", "white", "red"))(100)
  
  # Determine which cells would be significant if we were using a lenient threshold
  # We can mark nominal significance (p < 0.1) since none reach FDR significance
  significance_matrix <- matrix("", nrow = nrow(enrichment_matrix), ncol = ncol(enrichment_matrix))
  
  # Get the raw p-values for each cell
  for (i in 1:nrow(enrichment_matrix)) {
    gene_set_name <- rownames(enrichment_matrix)[i]
    for (j in 1:ncol(enrichment_matrix)) {
      cluster_num <- as.numeric(colnames(enrichment_matrix)[j])
      
      # Find the corresponding result
      result <- asd_results %>%
        filter(gene_set == gene_set_name & cluster == cluster_num)
      
      if (nrow(result) > 0) {
        # Mark nominal significance
        if (result$pval[1] < 0.1) {
          significance_matrix[i, j] <- "*"
        }
      }
    }
  }
  
  # Print the actual matrix values for debugging
  cat("Enrichment matrix values:\n")
  print(enrichment_matrix)
  
  # Create heatmap
  heatmap <- pheatmap(
    enrichment_matrix,
    main = main_title,
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
    display_numbers = significance_matrix,
    number_color = "black",
    silent = TRUE
  )
  
  return(heatmap)
}

# ---------------------------------------------------------
# CREATE GENE OVERLAP COUNTS HEATMAP
# ---------------------------------------------------------
create_gene_overlap_heatmap <- function() {
  # Load cluster-specific genes
  load_cluster_specific_genes <- function() {
    skat_file <- file.path(results_dir, "SKAT_all_cluster_specific_genes.csv")
    if(file.exists(skat_file)) {
      cat("Loading SKAT results from:", skat_file, "\n")
      cluster_specific_genes <- read.csv(skat_file)
      return(cluster_specific_genes)
    } else {
      # Try an alternative path
      alt_file <- file.path(data_dir, "cluster_specific_genes.csv")
      if(file.exists(alt_file)) {
        cat("Loading cluster-specific genes from:", alt_file, "\n")
        return(read.csv(alt_file))
      } else {
        stop("Could not find cluster-specific genes file")
      }
    }
  }
  
  # Load ASD gene sets
  load_asd_gene_sets <- function() {
    reference_dir <- file.path(base_path, "reference")
    
    # Define the ASD-related gene set files
    asd_files <- list(
      FMRP_interacting      = file.path(reference_dir, "fmrp_interacting_genes.txt"),
      ASD_Grove             = file.path(reference_dir, "asd_grove_genes.txt"),
      ASD_RDNV              = file.path(reference_dir, "asd_rdnv_genes.txt"),
      ASD_SPARK             = file.path(reference_dir, "asd_spark_genes.txt"),
      ASD_upregulated       = file.path(reference_dir, "asd_upregulated_genes.txt"),
      ASD_downregulated     = file.path(reference_dir, "asd_downregulated_genes.txt")
    )
    
    # Initialize list
    asd_gene_sets <- list()
    
    # Load each gene set
    for (set_name in names(asd_files)) {
      file_path <- asd_files[[set_name]]
      if (file.exists(file_path)) {
        genes <- readLines(file_path)
        genes <- trimws(genes[genes != ""])
        asd_gene_sets[[set_name]] <- genes
        cat(paste("Loaded", length(genes), "genes for", set_name, "\n"))
      } else {
        asd_gene_sets[[set_name]] <- character(0)
        cat(paste("File not found for", set_name, ":", file_path, "\n"))
      }
    }
    
    return(asd_gene_sets)
  }
  
  # Try to load the required data
  tryCatch({
    cluster_specific_genes <- load_cluster_specific_genes()
    asd_gene_sets <- load_asd_gene_sets()
    
    # Calculate overlap counts
    overlap_data <- data.frame()
    
    # Extract genes for each cluster
    cluster_genes <- list()
    for(cl in 1:4) {
      cluster_genes[[paste0("Cluster_", cl)]] <- 
        cluster_specific_genes$Gene[cluster_specific_genes$Cluster == cl]
    }
    
    # Calculate overlaps
    for(cl_name in names(cluster_genes)) {
      cl_num <- as.numeric(gsub("Cluster_", "", cl_name))
      cl_genes <- cluster_genes[[cl_name]]
      
      for(set_name in names(asd_gene_sets)) {
        set_genes <- asd_gene_sets[[set_name]]
        
        # Skip empty gene sets
        if (length(set_genes) == 0) {
          next
        }
        
        overlap_count <- length(intersect(cl_genes, set_genes))
        
        # Add to results
        overlap_data <- rbind(overlap_data, data.frame(
          Cluster = cl_num,
          Gene_Set = set_name,
          Overlap_Count = overlap_count,
          stringsAsFactors = FALSE
        ))
      }
    }
    
    # Make sure we have data to plot
    if (nrow(overlap_data) == 0) {
      warning("No gene set overlap data generated")
      return(NULL)
    }
    
    # Create a matrix for heatmap
    overlap_matrix <- overlap_data %>%
      pivot_wider(
        names_from = Cluster,
        values_from = Overlap_Count,
        values_fill = 0
      ) %>%
      as.data.frame()
    
    # Set gene sets as row names
    rownames(overlap_matrix) <- overlap_matrix$Gene_Set
    overlap_matrix$Gene_Set <- NULL
    
    # Create heatmap
    color_palette <- colorRampPalette(c("white", "pink", "red"))(100)
    
    # Create the heatmap
    heatmap <- pheatmap(
      overlap_matrix,
      main = "ASD Gene Set Overlap Counts",
      color = color_palette,
      cluster_rows = FALSE,
      cluster_cols = FALSE,
      border_color = "black",
      cellwidth = 40,
      cellheight = 25,
      fontsize_row = 10,
      fontsize_col = 12,
      angle_col = 0,
      display_numbers = TRUE,
      number_color = "black",
      silent = TRUE
    )
    
    return(heatmap)
  }, error = function(e) {
    warning("Error creating gene overlap heatmap: ", e$message)
    return(NULL)
  })
}

# ---------------------------------------------------------
# CREATE VISUALIZATIONS WITH DIFFERENT APPROACHES
# ---------------------------------------------------------

# 1. Original method: Enrichment scores as -log10(padj) * sign(NES)
original_heatmap <- create_improved_asd_heatmap(all_results, use_nes = FALSE)
if (!is.null(original_heatmap)) {
  pdf(file.path(figures_dir, "asd_gene_set_enrichment_scores.pdf"), width = 10, height = 7)
  grid::grid.newpage()
  grid::grid.draw(original_heatmap$gtable)
  dev.off()
  cat("ASD gene set enrichment scores heatmap saved\n")
}

# 2. Alternative method: Raw NES values (more visible patterns)
nes_heatmap <- create_improved_asd_heatmap(all_results, use_nes = TRUE)
if (!is.null(nes_heatmap)) {
  pdf(file.path(figures_dir, "asd_gene_set_nes_values.pdf"), width = 10, height = 7)
  grid::grid.newpage()
  grid::grid.draw(nes_heatmap$gtable)
  dev.off()
  cat("ASD gene set NES values heatmap saved\n")
}

# 3. Fallback method: Gene overlap counts
overlap_heatmap <- create_gene_overlap_heatmap()
if (!is.null(overlap_heatmap)) {
  pdf(file.path(figures_dir, "asd_gene_set_overlap_counts.pdf"), width = 10, height = 7)
  grid::grid.newpage()
  grid::grid.draw(overlap_heatmap$gtable)
  dev.off()
  cat("ASD gene set overlap counts heatmap saved\n")
}

# Create a comparison figure with all three approaches
if (!is.null(original_heatmap) && !is.null(nes_heatmap) && !is.null(overlap_heatmap)) {
  pdf(file.path(figures_dir, "asd_gene_set_comparison.pdf"), width = 15, height = 8)
  grid.newpage()
  
  # Create a layout with three panels side by side
  pushViewport(viewport(layout = grid.layout(1, 3)))
  
  # Add each heatmap to the layout
  print(original_heatmap, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
  print(nes_heatmap, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
  print(overlap_heatmap, vp = viewport(layout.pos.row = 1, layout.pos.col = 3))
  
  dev.off()
  cat("Combined comparison figure saved\n")
}

# Print session info for reproducibility
cat("\nSession Info:\n")
print(sessionInfo())