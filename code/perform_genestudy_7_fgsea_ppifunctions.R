## ---------------------------------------------------------------------
## PPI NETWORK ANALYSIS FUNCTIONS
## ---------------------------------------------------------------------

# Prepare gene list for PPI network analysis
# Using top 20 genes for PPI as per Buch et al's approach 
prepare_genes_for_ppi <- function(cluster_num, top_n_genes = 20, cluster_specific_genes = NULL) {
  cat(paste("Preparing top", top_n_genes, "genes for PPI analysis from cluster", cluster_num, "\n"))
  
  # Load data if not provided
  if(is.null(cluster_specific_genes)) {
    skat_file <- file.path(results_dir, "SKAT_all_cluster_specific_genes.csv")
    if(file.exists(skat_file)) {
      cluster_specific_genes <- read.csv(skat_file)
    } else {
      stop("SKAT results file not found: ", skat_file)
    }
  }
  
  # Get genes for this cluster and sort by p-value
  cluster_genes <- cluster_specific_genes %>%
    filter(Cluster == cluster_num) %>%
    arrange(P_value)
  
  # Get top N genes by p-value
  top_genes <- head(cluster_genes$Gene, min(top_n_genes, nrow(cluster_genes)))
  
  # Write to file for external processing with STRING or NetworkAnalyst
  output_file <- file.path(output_dir, sprintf("cluster%d_top%d_genes_for_ppi.txt", 
                                            cluster_num, top_n_genes))
  write.table(
    top_genes,
    output_file,
    row.names = FALSE, 
    col.names = FALSE, 
    quote = FALSE
  )
  
  cat(paste("  Saved", length(top_genes), "genes for cluster", 
            cluster_num, "to", output_file, "\n"))
  
  return(top_genes)
}

# Function to create and analyze PPI networks similar to Buch et al.
analyze_ppi_networks <- function(cluster_num, top_genes = NULL, sfari_genes = NULL) {
  cat(paste("Analyzing PPI network for cluster", cluster_num, "\n"))
  
  # If top genes not provided, prepare them
  if(is.null(top_genes)) {
    top_genes <- prepare_genes_for_ppi(cluster_num)
  }
  
  # Export gene list for STRING analysis
  output_file <- file.path(output_dir, sprintf("cluster%d_ppi_input.txt", cluster_num))
  write.table(top_genes, output_file, row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  cat(paste("Exported", length(top_genes), "genes for PPI analysis to:", output_file, "\n"))
  cat("To generate PPI networks similar to Buch et al., follow these steps:\n")
  cat("1. Upload gene list to STRING (https://string-db.org/) or NetworkAnalyst\n")
  cat("2. Set parameters: high confidence (0.7), no more than 50 interactions\n")
  cat("3. Export the network for visualization\n")
  cat("4. Use Cytoscape or similar tools to create final visualizations\n")
  
  # Create placeholder for network modules
  # In a real implementation, this would come from clustering analysis of the PPI network
  network_modules <- list(
    module1 = sample(top_genes, min(5, length(top_genes))),
    module2 = sample(top_genes, min(7, length(top_genes))), 
    module3 = sample(top_genes, min(6, length(top_genes)))
  )
  
  # Save modules for further analysis
  for (module_name in names(network_modules)) {
    module_file <- file.path(output_dir, 
                            sprintf("cluster%d_%s_genes.txt", cluster_num, module_name))
    write.table(network_modules[[module_name]], module_file, 
                row.names = FALSE, col.names = FALSE, quote = FALSE)
  }
  
  return(network_modules)
}

# Function to check and standardize STRING network files
validate_string_network_files <- function(cluster_num) {
  edges_file <- file.path(output_dir, sprintf("cluster%d_string_interactions.csv", cluster_num))
  nodes_file <- file.path(output_dir, sprintf("cluster%d_string_nodes.csv", cluster_num))
  
  string_files_exist <- file.exists(edges_file) && file.exists(nodes_file)
  
  if (!string_files_exist) {
    # Look for alternative file naming conventions
    alt_edges_file <- list.files(output_dir, pattern = paste0("cluster", cluster_num, ".*edges|interactions.*\\.csv|\\.tsv|\\.txt"), full.names = TRUE)
    alt_nodes_file <- list.files(output_dir, pattern = paste0("cluster", cluster_num, ".*nodes|proteins.*\\.csv|\\.tsv|\\.txt"), full.names = TRUE)
    
    if (length(alt_edges_file) > 0 && length(alt_nodes_file) > 0) {
      # Use the first matches
      edges_file_to_use <- alt_edges_file[1]
      nodes_file_to_use <- alt_nodes_file[1]
      
      # Create standardized copies with expected names
      edges_data <- read.delim(edges_file_to_use, sep = NULL, stringsAsFactors = FALSE)
      nodes_data <- read.delim(nodes_file_to_use, sep = NULL, stringsAsFactors = FALSE)
      
      # Ensure standard column names
      if ("protein1" %in% colnames(edges_data) && "protein2" %in% colnames(edges_data)) {
        colnames(edges_data)[colnames(edges_data) == "protein1"] <- "from"
        colnames(edges_data)[colnames(edges_data) == "protein2"] <- "to"
      }
      
      # Save with standardized names
      write.csv(edges_data, edges_file, row.names = FALSE)
      write.csv(nodes_data, nodes_file, row.names = FALSE)
      
      cat("Found alternative STRING network files and standardized them:\n")
      cat("  Original edges file:", edges_file_to_use, "\n")
      cat("  Original nodes file:", nodes_file_to_use, "\n")
      cat("  Standardized to:", edges_file, "and", nodes_file, "\n")
      
      return(TRUE)
    } else {
      return(FALSE)
    }
  }
  
  return(TRUE)
}

# Analyze enrichment of gene modules identified from PPI networks
analyze_ppi_modules_enrichment <- function(modules, cluster_num) {
  # This function would analyze GO/Reactome enrichment for each module
  # using ClusterProfiler or similar tools
  
  module_enrichment <- list()
  
  for (module_name in names(modules)) {
    genes <- modules[[module_name]]
    
    if (length(genes) < 3) {
      cat(paste("Module", module_name, "has too few genes for enrichment analysis\n"))
      next
    }
    
    # Placeholder for module enrichment analysis
    # In a real implementation, use clusterProfiler or ReactomePA
    cat(paste("Analyzing enrichment for module", module_name, "in cluster", cluster_num, "\n"))
    
    # Convert gene symbols to Entrez IDs for enrichment analysis
    if (requireNamespace("org.Hs.eg.db", quietly = TRUE)) {
      entrez_ids <- mapIds(org.Hs.eg.db, keys = genes, keytype = "SYMBOL", column = "ENTREZID")
      entrez_ids <- entrez_ids[!is.na(entrez_ids)]
      
      if (length(entrez_ids) >= 3) {
        # GO Enrichment Analysis
        if (requireNamespace("clusterProfiler", quietly = TRUE)) {
          # Biological Process enrichment
          go_bp <- enrichGO(
            gene = entrez_ids,
            OrgDb = org.Hs.eg.db,
            ont = "BP",
            pAdjustMethod = "BH",
            pvalueCutoff = 0.1,  # More lenient cutoff for small gene sets
            qvalueCutoff = 0.2
          )
          
          # Save results
          if (length(go_bp) > 0) {
            go_bp_file <- file.path(output_dir, 
                                  sprintf("cluster%d_%s_GO_BP.csv", cluster_num, module_name))
            write.csv(as.data.frame(go_bp), go_bp_file, row.names = FALSE)
          }
          
          # Molecular Function enrichment
          go_mf <- enrichGO(
            gene = entrez_ids,
            OrgDb = org.Hs.eg.db,
            ont = "MF",
            pAdjustMethod = "BH",
            pvalueCutoff = 0.1,
            qvalueCutoff = 0.2
          )
          
          # Save results
          if (length(go_mf) > 0) {
            go_mf_file <- file.path(output_dir, 
                                  sprintf("cluster%d_%s_GO_MF.csv", cluster_num, module_name))
            write.csv(as.data.frame(go_mf), go_mf_file, row.names = FALSE)
          }
          
          # Store results
          module_enrichment[[module_name]] <- list(
            GO_BP = go_bp,
            GO_MF = go_mf
          )
        }
        
        # Reactome Pathway Analysis
        if (requireNamespace("ReactomePA", quietly = TRUE)) {
          reactome <- enrichPathway(
            gene = entrez_ids,
            organism = "human",
            pAdjustMethod = "BH",
            pvalueCutoff = 0.1,
            qvalueCutoff = 0.2
          )
          
          # Save results
          if (length(reactome) > 0) {
            reactome_file <- file.path(output_dir, 
                                     sprintf("cluster%d_%s_Reactome.csv", cluster_num, module_name))
            write.csv(as.data.frame(reactome), reactome_file, row.names = FALSE)
            
            # Store results
            module_enrichment[[module_name]]$Reactome <- reactome
          }
        }
      } else {
        cat(paste("  Too few genes with Entrez IDs (", length(entrez_ids), 
                 ") for enrichment analysis\n"))
      }
    } else {
      cat("  org.Hs.eg.db package is required for enrichment analysis\n")
    }
  }
  
  return(module_enrichment)
}

# Function to visualize PPI modules and their enrichment results
visualize_ppi_enrichment <- function(module_enrichment, cluster_num) {
  # This function would create visualizations for the enrichment results
  # of the PPI network modules
  
  for (module_name in names(module_enrichment)) {
    cat(paste("Creating visualizations for", module_name, "in cluster", cluster_num, "\n"))
    
    # GO Biological Process dotplot
    if (!is.null(module_enrichment[[module_name]]$GO_BP) && 
        length(module_enrichment[[module_name]]$GO_BP)) {
      go_bp <- module_enrichment[[module_name]]$GO_BP
      
      if (requireNamespace("clusterProfiler", quietly = TRUE) && nrow(as.data.frame(go_bp)) > 0) {
        pdf(file.path(output_dir, 
                     sprintf("cluster%d_%s_GO_BP_dotplot.pdf", cluster_num, module_name)),
            width = 8, height = 6)
        print(dotplot(go_bp, showCategory = 10, title = paste("GO BP -", module_name)))
        dev.off()
        
        # Network plot
        pdf(file.path(output_dir, 
                     sprintf("cluster%d_%s_GO_BP_network.pdf", cluster_num, module_name)),
            width = 10, height = 8)
        print(cnetplot(go_bp, categorySize = "pvalue", foldChange = NULL))
        dev.off()
      }
    }
    
    # GO Molecular Function dotplot
    if (!is.null(module_enrichment[[module_name]]$GO_MF) && 
        length(module_enrichment[[module_name]]$GO_MF)) {
      go_mf <- module_enrichment[[module_name]]$GO_MF
      
      if (requireNamespace("clusterProfiler", quietly = TRUE) && nrow(as.data.frame(go_mf)) > 0) {
        pdf(file.path(output_dir, 
                     sprintf("cluster%d_%s_GO_MF_dotplot.pdf", cluster_num, module_name)),
            width = 8, height = 6)
        print(dotplot(go_mf, showCategory = 10, title = paste("GO MF -", module_name)))
        dev.off()
      }
    }
    
    # Reactome pathway dotplot
    if (!is.null(module_enrichment[[module_name]]$Reactome) && 
        length(module_enrichment[[module_name]]$Reactome)) {
      reactome <- module_enrichment[[module_name]]$Reactome
      
      if (requireNamespace("ReactomePA", quietly = TRUE) && nrow(as.data.frame(reactome)) > 0) {
        pdf(file.path(output_dir, 
                     sprintf("cluster%d_%s_Reactome_dotplot.pdf", cluster_num, module_name)),
            width = 9, height = 7)
        print(dotplot(reactome, showCategory = 10, 
                     title = paste("Reactome Pathways -", module_name)))
        dev.off()
      }
    }
  }
}

## ---------------------------------------------------------------------
## EXTENDED ANALYSES AND COMPARISONS 
## ---------------------------------------------------------------------

# Function to compare enrichment results across clusters
compare_clusters_enrichment <- function(all_results) {
  cat("Comparing enrichment results across clusters...\n")
  
  # First, merge all results into one dataframe
  combined_results <- bind_rows(lapply(names(all_results), function(cluster_name) {
    df <- all_results[[cluster_name]]
    if(nrow(df) > 0) {
      df$cluster <- as.numeric(gsub("cluster", "", cluster_name))
      return(df)
    }
    return(NULL)
  }))
  
  # Skip if no results available
  if(nrow(combined_results) == 0) {
    warning("No enrichment results available for comparison.")
    return(NULL)
  }
  
  # Find pathways enriched in multiple clusters
  pathway_counts <- combined_results %>%
    group_by(pathway) %>%
    summarize(
      num_clusters = n_distinct(cluster),
      avg_nes = mean(NES),
      min_padj = min(padj)
    ) %>%
    arrange(desc(num_clusters), min_padj)
  
  # Save shared pathways
  write.csv(
    pathway_counts,
    file.path(output_dir, "shared_pathways_across_clusters.csv"),
    row.names = FALSE
  )
  
  # Create comparison heatmap for top shared pathways
  top_shared <- pathway_counts %>%
    filter(num_clusters >= 2) %>%
    arrange(min_padj) %>%
    head(30) %>%
    pull(pathway)
  
  # If we have shared pathways, create comparison visualizations
  if(length(top_shared) > 0) {
    # Prepare data for heatmap
    top_pathways_data <- combined_results %>%
      filter(pathway %in% top_shared) %>%
      mutate(
        # Create a combined score similar to Buch et al.
        score = -log10(pmax(padj, 1e-4)) * sign(NES)
      ) %>%
      select(pathway, cluster, score) %>%
      pivot_wider(
        names_from = cluster,
        values_from = score,
        values_fill = 0
      )
    
    # Set pathway names as row names
    rownames(top_pathways_data) <- top_pathways_data$pathway
    top_pathways_data$pathway <- NULL
    
    # Create heatmap
    if(requireNamespace("pheatmap", quietly = TRUE)) {
      # Make sure we have at least 2 rows before clustering
      if(nrow(top_pathways_data) >= 2) {
        # Clean pathway names for display
        rownames(top_pathways_data) <- str_replace_all(
          rownames(top_pathways_data), 
          "^HALLMARK_|^REACTOME_|^KEGG_|^GO_", ""
        )
        rownames(top_pathways_data) <- str_replace_all(rownames(top_pathways_data), "_", " ")
        
        # Create PDF
        pdf(file.path(output_dir, "shared_pathways_heatmap.pdf"), width = 10, height = 12)
        pheatmap(
          as.matrix(top_pathways_data),
          main = "Shared Pathways Across Clusters",
          color = colorRampPalette(c("blue", "white", "red"))(100),
          breaks = seq(-4, 4, length.out = 101),
          cluster_rows = TRUE, 
          cluster_cols = FALSE,
          border_color = "black",
          fontsize = 8,
          cellwidth = 25,
          cellheight = 12
        )
        dev.off()
      } else {
        cat("Not enough shared pathways (minimum 2) for creating a clustered heatmap.\n")
        
        # Alternative: Create a simple heatmap without clustering
        if(nrow(top_pathways_data) > 0) {
          pdf(file.path(output_dir, "shared_pathways_simple.pdf"), width = 10, height = 6)
          pheatmap(
            as.matrix(top_pathways_data),
            main = "Shared Pathways Across Clusters",
            color = colorRampPalette(c("blue", "white", "red"))(100),
            breaks = seq(-4, 4, length.out = 101),
            cluster_rows = FALSE,  # No clustering
            cluster_cols = FALSE,
            border_color = "black",
            fontsize = 8,
            cellwidth = 25,
            cellheight = 12
          )
          dev.off()
          cat("Created simple non-clustered heatmap instead.\n")
        }
      }
    }
  } else {
    cat("No pathways were found to be shared across multiple clusters.\n")
    cat("This could be because:\n")
    cat("1. Each cluster has distinct biological pathways\n")
    cat("2. The significance threshold may be too stringent\n")
    cat("3. There may be limited overlap in the enriched pathways\n")
    
    # Create a summary table of top pathways for each cluster instead
    top_per_cluster <- combined_results %>%
      group_by(cluster) %>%
      arrange(padj) %>%
      slice_head(n = 5) %>%
      select(cluster, pathway, NES, pval, padj) %>%
      mutate(pathway = str_replace_all(pathway, "^HALLMARK_|^REACTOME_|^KEGG_|^GO_", ""),
             pathway = str_replace_all(pathway, "_", " "))
    
    # Save as alternative result
    write.csv(
      top_per_cluster,
      file.path(output_dir, "top_pathways_per_cluster.csv"),
      row.names = FALSE
    )
    
    cat("Saved top 5 pathways per cluster as an alternative.\n")
  }
  
  return(pathway_counts)
}

# Enhance visualization # 
# Use this function in create_enhanced_ppi_network
create_enhanced_ppi_network <- function(cluster_num, top_genes, asd_gene_sets) {
  # First validate network files
  if (!validate_string_network_files(cluster_num)) {
    cat("Network files not found for cluster", cluster_num, "\n")
    cat("Please run STRING analysis first and save the results in the expected format.\n")
    return(NULL)
  }

  if (!requireNamespace("igraph", quietly = TRUE)) {
    install.packages("igraph")
  }
  library(igraph)
  
  # File paths for network data (after STRING analysis)
  edges_file <- file.path(output_dir, sprintf("cluster%d_string_interactions.csv", cluster_num))
  nodes_file <- file.path(output_dir, sprintf("cluster%d_string_nodes.csv", cluster_num))
  
  if (!file.exists(edges_file) || !file.exists(nodes_file)) {
    cat("Network files not found. Please run STRING analysis first and export the results.\n")
    return(NULL)
  }
  
  # Load network data
  edges <- read.csv(edges_file)
  nodes <- read.csv(nodes_file)
  
  # Create graph
  g <- graph_from_data_frame(edges, directed = FALSE, vertices = nodes)
  
  # Calculate node degrees (for sizing)
  V(g)$degree <- degree(g)
  
  # Color nodes based on ASD gene overlap
  V(g)$color <- "gray80"  # Default color
  
  # Check for transcriptionally regulated ASD genes (upregulated)
  if ("ASD_upregulated" %in% names(asd_gene_sets)) {
    upreg_idx <- which(V(g)$name %in% asd_gene_sets$ASD_upregulated)
    if (length(upreg_idx) > 0) {
      V(g)$color[upreg_idx] <- "skyblue"
    }
  }
  
  # Check for transcriptionally regulated ASD genes (downregulated)
  if ("ASD_downregulated" %in% names(asd_gene_sets)) {
    downreg_idx <- which(V(g)$name %in% asd_gene_sets$ASD_downregulated)
    if (length(downreg_idx) > 0) {
      V(g)$color[downreg_idx] <- "blue"
    }
  }
  
  # Check for other known ASD genes (SFARI, RDNV, etc.)
  other_asd_genes <- unique(c(
    asd_gene_sets$FMRP_interacting,
    asd_gene_sets$ASD_RDNV,
    asd_gene_sets$ASD_SPARK,
    asd_gene_sets$ASD_Grove
  ))
  other_idx <- which(V(g)$name %in% other_asd_genes)
  if (length(other_idx) > 0) {
    V(g)$color[other_idx] <- "red"
  }
  
  # Size nodes by degree (connectivity)
  V(g)$size <- 3 + (V(g)$degree / max(V(g)$degree) * 10)
  
  # Save the network object
  saveRDS(g, file.path(output_dir, sprintf("cluster%d_ppi_network.rds", cluster_num)))
  
  # Plot network
  pdf(file.path(output_dir, sprintf("cluster%d_ppi_network.pdf", cluster_num)), 
      width = 10, height = 10)
  plot(g, 
       layout = layout_with_fr(g),
       vertex.label.cex = 0.7,
       vertex.label.color = "black",
       vertex.label.family = "sans",
       vertex.frame.color = "gray50",
       edge.width = 0.5,
       edge.color = "gray70",
       main = sprintf("Cluster %d PPI Network", cluster_num))
  legend("bottomright", 
         legend = c("Transcriptionally Upregulated", 
                   "Transcriptionally Downregulated", 
                   "Other ASD Risk Genes"),
         col = c("skyblue", "blue", "red"),
         pch = 19,
         cex = 0.8,
         pt.cex = 1.5)
  dev.off()
  
  return(g)
}

# Walktrap module detection implementation
detect_ppi_modules <- function(g, cluster_num) {
  if (!requireNamespace("igraph", quietly = TRUE)) {
    install.packages("igraph")
  }
  library(igraph)
  
  # Apply Walktrap algorithm to detect communities
  # Adjust steps parameter as needed (Buch et al. doesn't specify their exact setting)
  wt <- walktrap.community(g, steps = 4)
  
  # Get module memberships
  modules <- membership(wt)
  V(g)$module <- modules
  
  # Extract modules
  module_list <- list()
  for (i in 1:max(modules)) {
    module_genes <- V(g)$name[modules == i]
    if (length(module_genes) >= 3) {  # Only keep modules with at least 3 genes
      module_list[[paste0("module", i)]] <- module_genes
    }
  }
  
  # Save modules to files
  for (mod_name in names(module_list)) {
    mod_file <- file.path(output_dir, 
                         sprintf("cluster%d_%s_genes.txt", cluster_num, mod_name))
    write.table(module_list[[mod_name]], mod_file, 
                row.names = FALSE, col.names = FALSE, quote = FALSE)
  }
  
  # Visualize network with modules highlighted
  pdf(file.path(output_dir, sprintf("cluster%d_ppi_modules.pdf", cluster_num)), 
      width = 12, height = 10)
  
  # Create layout (only once for consistency)
  layout <- layout_with_fr(g)
  
  # Plot with modules
  plot(g, 
       layout = layout,
       vertex.color = modules,  # Color by module
       vertex.label.cex = 0.7,
       vertex.label.color = "black",
       vertex.label.family = "sans",
       vertex.frame.color = "gray50",
       edge.width = 0.5,
       edge.color = "gray70",
       main = sprintf("Cluster %d PPI Network Modules", cluster_num))
  
  # Create module significance test (as in Buch et al.)
  module_stats <- data.frame(
    Module = character(),
    Size = integer(),
    P_value = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (mod in unique(modules)) {
    mod_vertices <- which(modules == mod)
    if (length(mod_vertices) >= 3) {
      # Calculate within-module degrees
      subg <- induced_subgraph(g, mod_vertices)
      within_degrees <- degree(subg)
      
      # Calculate cross-module degrees
      cross_degrees <- sapply(mod_vertices, function(v) {
        sum(neighbors(g, v) %in% setdiff(V(g), mod_vertices))
      })
      
      # Wilcoxon rank-sum test as in Buch et al.
      if (length(within_degrees) > 0 && length(cross_degrees) > 0) {
        p_val <- wilcox.test(within_degrees, cross_degrees)$p.value
        
        module_stats <- rbind(module_stats, 
                              data.frame(Module = paste0("module", mod),
                                        Size = length(mod_vertices),
                                        P_value = p_val,
                                        stringsAsFactors = FALSE))
      }
    }
  }
  
  # Save module statistics
  write.csv(module_stats, 
           file.path(output_dir, sprintf("cluster%d_module_stats.csv", cluster_num)),
           row.names = FALSE)
  
  dev.off()
  
  return(list(network = g, modules = module_list, stats = module_stats))
}


## ---------------------------------------------------------------------
## MAIN FUNCTION FOR PPI NETWORK ANALYSIS
## ---------------------------------------------------------------------

run_ppi_analysis <- function(cluster_specific_genes = NULL, asd_gene_sets = NULL) {
  cat("\nStarting PPI network analysis for all clusters...\n")
  
  # Load data if not provided
  if(is.null(cluster_specific_genes)) {
    skat_file <- file.path(results_dir, "SKAT_all_cluster_specific_genes.csv")
    if(file.exists(skat_file)) {
      cluster_specific_genes <- read.csv(skat_file)
    } else {
      stop("SKAT results file not found: ", skat_file)
    }
  }
  
  # Load ASD gene sets if not provided
  if(is.null(asd_gene_sets)) {
    cat("Loading ASD gene sets for network annotation...\n")
    asd_gene_sets <- load_asd_gene_sets()
  }
  
  # Run for each cluster
  all_modules <- list()
  all_enrichment <- list()
  all_networks <- list()
  all_module_stats <- list()
  
  for (cluster in 1:4) {
    cat(paste("\n== Analyzing PPI network for Cluster", cluster, "==\n"))
    
    # Get top 20 genes for this cluster for PPI analysis (as per Buch et al.)
    top_genes <- prepare_genes_for_ppi(cluster, top_n_genes = 20, cluster_specific_genes)
    
    # Analyze PPI network
    modules <- analyze_ppi_networks(cluster, top_genes)
    all_modules[[paste0("cluster", cluster)]] <- modules
    
    # Check if STRING analysis results are available
    edges_file <- file.path(output_dir, sprintf("cluster%d_string_interactions.csv", cluster))
    nodes_file <- file.path(output_dir, sprintf("cluster%d_string_nodes.csv", cluster))
    
    if(file.exists(edges_file) && file.exists(nodes_file)) {
      cat(paste("Creating enhanced PPI network visualization for Cluster", cluster, "...\n"))
      
      # Create enhanced network visualization with ASD gene annotations
      g <- create_enhanced_ppi_network(cluster, top_genes, asd_gene_sets)
      all_networks[[paste0("cluster", cluster)]] <- g
      
      if(!is.null(g)) {
        # Apply Walktrap algorithm to detect biological modules
        cat(paste("Detecting modules using Walktrap algorithm for Cluster", cluster, "...\n"))
        module_results <- detect_ppi_modules(g, cluster)
        
        # Store detailed module information
        all_modules[[paste0("cluster", cluster)]] <- module_results$modules
        all_module_stats[[paste0("cluster", cluster)]] <- module_results$stats
        
        # Update modules for enrichment analysis
        modules <- module_results$modules
      } else {
        cat("Error creating network for Cluster", cluster, ". Using original modules.\n")
      }
    } else {
      cat("STRING analysis results not found for Cluster", cluster, ".\n")
      cat("  Expected files:", edges_file, "and", nodes_file, "\n")
      cat("  Please run STRING analysis and export the results before proceeding.\n")
      cat("  Using placeholder modules for enrichment analysis.\n")
    }
    
    # Analyze enrichment for modules
    module_enrichment <- analyze_ppi_modules_enrichment(modules, cluster)
    all_enrichment[[paste0("cluster", cluster)]] <- module_enrichment
    
    # Visualize module enrichment
    visualize_ppi_enrichment(module_enrichment, cluster)
  }
  
  # Save module results
  saveRDS(all_modules, file.path(output_dir, "ppi_modules.rds"))
  saveRDS(all_enrichment, file.path(output_dir, "ppi_module_enrichment.rds"))
  saveRDS(all_networks, file.path(output_dir, "ppi_networks.rds"))
  saveRDS(all_module_stats, file.path(output_dir, "ppi_module_stats.rds"))
  
  # Create a combined module statistics table
  combined_stats <- do.call(rbind, lapply(names(all_module_stats), function(cluster_name) {
    stats_df <- all_module_stats[[cluster_name]]
    if(!is.null(stats_df) && nrow(stats_df) > 0) {
      stats_df$Cluster <- gsub("cluster", "", cluster_name)
      return(stats_df)
    }
    return(NULL)
  }))
  
  if(!is.null(combined_stats) && nrow(combined_stats) > 0) {
    write.csv(combined_stats, file.path(output_dir, "all_clusters_module_stats.csv"), row.names = FALSE)
  }
  
  cat("\nPPI network analysis complete\n")
  cat("Saved results:\n")
  cat("  - PPI networks: ", file.path(output_dir, "ppi_networks.rds"), "\n")
  cat("  - PPI modules: ", file.path(output_dir, "ppi_modules.rds"), "\n")
  cat("  - Module statistics: ", file.path(output_dir, "ppi_module_stats.rds"), "\n")
  cat("  - Module enrichment: ", file.path(output_dir, "ppi_module_enrichment.rds"), "\n")
  
  return(list(
    modules = all_modules, 
    enrichment = all_enrichment,
    networks = all_networks,
    module_stats = all_module_stats
  ))
}