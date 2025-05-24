# ----------------------------------------------------------------
# STRING-based PPI Network Creation - Using Existing SFARI Annotations
# ----------------------------------------------------------------
# Path settings (using the same structure from your scripts)
base_path <- "/home/data/2025_asd_wgs"
results_dir <- file.path(base_path, "skat_results")
output_dir <- file.path(base_path, "fgsea_results")
data_dir <- file.path(base_path, "data")
reference_dir <- file.path(base_path, "reference")

# Install STRINGdb if needed
# if (!requireNamespace("STRINGdb", quietly = TRUE)) {
#   BiocManager::install("STRINGdb")
# }

# Load required libraries
library(STRINGdb)
library(igraph)
library(RColorBrewer)

# Increase timeout for downloads
options(timeout = 600)  # 10 minutes

# Load existing SFARI annotation data from your results
sfari_file <- file.path(output_dir, "SKAT_all_genes_SFARI_annotated.csv")
if (file.exists(sfari_file)) {
  sfari_annotated <- try(read.csv(sfari_file, header=TRUE, stringsAsFactors=FALSE))
  
  if (!inherits(sfari_annotated, "try-error")) {
    # Create a lookup table for SFARI scores by gene
    sfari_lookup <- setNames(sfari_annotated$SFARI_Score, sfari_annotated$Gene)
    cat("Loaded SFARI annotations for", length(sfari_lookup), "genes\n")
  } else {
    cat("Error reading SFARI annotation file\n")
    sfari_lookup <- NULL
  }
} else {
  cat("SFARI annotation file not found, will not highlight SFARI genes\n")
  sfari_lookup <- NULL
}

create_string_network <- function(cluster_name, score_threshold = 400) {
  # Load genes
  cluster_file <- file.path(output_dir, paste0(cluster_name, "_ppi_input.txt"))
  if (!file.exists(cluster_file)) {
    cat(cluster_name, ": Gene file not found\n")
    return(NULL)
  }
  
  genes <- readLines(cluster_file)
  genes <- trimws(genes)
  genes <- unique(genes[genes != ""])
  cat(cluster_name, ": Processing", length(genes), "genes\n")
  
  if (length(genes) < 3) {
    cat(cluster_name, ": Too few genes to create network\n")
    return(NULL)
  }
  
  # Initialize STRING database with lower score threshold for more connections
  string_db <- try(STRINGdb$new(version="11.5", species=9606, score_threshold=score_threshold))
  if (inherits(string_db, "try-error")) {
    cat("Error initializing STRING database\n")
    return(NULL)
  }
  
  # Map genes to STRING identifiers
  mapped_genes <- string_db$map(data.frame(gene=genes), "gene", removeUnmappedRows=TRUE)
  cat(cluster_name, ": Mapped", nrow(mapped_genes), "out of", length(genes), "genes\n")
  
  if (nrow(mapped_genes) < 3) {
    cat(cluster_name, ": Too few genes mapped to create network\n")
    return(NULL)
  }
  
  # Get network
  string_graph <- try(string_db$get_subnetwork(mapped_genes$STRING_id))
  
  # Check if network creation was successful
  if (inherits(string_graph, "try-error")) {
    cat(cluster_name, ": Error creating subnetwork\n")
    return(NULL)
  }
  
  # Check if network has any edges
  if (ecount(string_graph) == 0) {
    cat(cluster_name, ": Network has 0 edges, trying with lower threshold\n")
    if (score_threshold > 200) {
      # Try again with lower threshold
      return(create_string_network(cluster_name, score_threshold - 100))
    }
  }
  
  # Add SFARI annotation if available
  if (!is.null(sfari_lookup)) {
    # Map STRING IDs back to gene symbols
    node_mapping <- mapped_genes[, c("STRING_id", "gene")]
    
    # Add SFARI annotation to nodes
    V(string_graph)$sfari <- "Not in SFARI"
    V(string_graph)$color <- "lightblue"
    
    for (i in 1:vcount(string_graph)) {
      node_id <- V(string_graph)$name[i]
      gene_name <- node_mapping$gene[node_mapping$STRING_id == node_id]
      
      if (length(gene_name) > 0 && gene_name %in% names(sfari_lookup)) {
        # Safe handling of SFARI scores
        score <- sfari_lookup[gene_name]
        
        # Handle NA or empty scores
        if (is.na(score) || score == "") {
          V(string_graph)$sfari[i] <- "SFARI (Score N/A)"
          V(string_graph)$color[i] <- "gray"
          next
        }
        
        V(string_graph)$sfari[i] <- paste("SFARI", score)
        
        # Color based on SFARI score - with safe checking
        if (!is.na(score) && score == "1") {
          V(string_graph)$color[i] <- "red"        # High confidence
        } else if (!is.na(score) && score == "2") {
          V(string_graph)$color[i] <- "orange"     # Strong candidate
        } else if (!is.na(score) && score == "3") {
          V(string_graph)$color[i] <- "yellow"     # Suggestive evidence
        } else if (!is.na(score) && score == "S") {
          V(string_graph)$color[i] <- "purple"     # Syndromic
        } else {
          # For other scores
          V(string_graph)$color[i] <- "pink"
        }
      }
    }
    
    # Replace STRING IDs with gene symbols for better readability
    for (i in 1:vcount(string_graph)) {
      string_id <- V(string_graph)$name[i]
      gene_matches <- node_mapping$gene[node_mapping$STRING_id == string_id]
      if (length(gene_matches) > 0) {
        V(string_graph)$name[i] <- gene_matches[1]
      }
    }
  }
  
  # Plot and save
  pdf(file.path(output_dir, paste0(cluster_name, "_STRING_network.pdf")), width=12, height=10)
  
  # Use Fruchterman-Reingold layout for better visualization
  layout <- layout_with_fr(string_graph)
  
  plot(string_graph, 
       layout = layout,
       vertex.size = 10,
       vertex.label.cex = 0.8,
       vertex.label.dist = 0.5,
       main = paste("STRING PPI Network for", cluster_name, 
                    "\nScore threshold:", score_threshold, 
                    "- Nodes:", vcount(string_graph), 
                    "- Edges:", ecount(string_graph)))
  
  # Add legend for SFARI genes if applicable
  if (!is.null(sfari_lookup)) {
    legend("bottomright", 
           legend = c("SFARI 1 (High)", "SFARI 2 (Strong)", "SFARI 3 (Suggestive)", 
                    "SFARI S (Syndromic)", "Other SFARI", "Not in SFARI"),
           fill = c("red", "orange", "yellow", "purple", "pink", "lightblue"),
           cex = 0.8)
  }
  
  dev.off()
  
  # Save network data
  write_graph(string_graph, file.path(output_dir, paste0(cluster_name, "_STRING.graphml")), format="graphml")
  
  cat(cluster_name, ": STRING network created successfully with", 
      vcount(string_graph), "nodes and", ecount(string_graph), "edges\n")
  
  return(string_graph)
}

# Function to analyze PPI network
analyze_ppi_network <- function(string_graph, cluster_name) {
  if (is.null(string_graph)) {
    cat(cluster_name, ": No network to analyze\n")
    return(NULL)
  }
  
  # Output analysis
  cat("\nNetwork Analysis for", cluster_name, ":\n")
  cat("Number of nodes:", vcount(string_graph), "\n")
  cat("Number of edges:", ecount(string_graph), "\n")
  
  if (ecount(string_graph) > 0) {
    # Degree analysis
    degrees <- degree(string_graph, mode="all")
    high_degree_nodes <- names(sort(degrees, decreasing=TRUE)[1:min(10, length(degrees))])
    cat("Top hub genes (highest connectivity):\n")
    for (node in high_degree_nodes) {
      cat("  -", node, "(connections:", degrees[node], ")\n")
    }
    
    # Betweenness centrality
    bet <- betweenness(string_graph)
    high_bet_nodes <- names(sort(bet, decreasing=TRUE)[1:min(5, length(bet))])
    cat("Top betweenness centrality genes (pathway connectors):\n")
    for (node in high_bet_nodes) {
      cat("  -", node, "\n")
    }
    
    # SFARI gene analysis if available
    if ("sfari" %in% vertex_attr_names(string_graph)) {
      sfari_nodes <- V(string_graph)[V(string_graph)$sfari != "Not in SFARI"]
      if (length(sfari_nodes) > 0) {
        cat("SFARI genes in network:\n")
        for (i in 1:length(sfari_nodes)) {
          node_name <- V(string_graph)$name[sfari_nodes[i]]
          sfari_info <- V(string_graph)$sfari[sfari_nodes[i]]
          cat("  -", node_name, "(", sfari_info, ")\n")
        }
      } else {
        cat("No SFARI genes found in this network\n")
      }
    }
  } else {
    cat("No edges in network - can't perform connectivity analysis\n")
  }
  
  # Create a more detailed network visualization with additional metrics
  if (ecount(string_graph) > 0) {
    pdf(file.path(output_dir, paste0(cluster_name, "_STRING_network_detailed.pdf")), width=14, height=12)
    
    # Calculate node sizes based on degree centrality
    deg <- degree(string_graph)
    v_size <- 5 + 15 * (deg / max(deg))
    
    # Better layout for detailed view
    layout <- layout_with_fr(string_graph)
    
    # Plot with more details
    plot(string_graph, 
         layout = layout,
         vertex.size = v_size,
         vertex.label.cex = 0.7,
         vertex.label.dist = 0.5,
         vertex.label.color = "black",
         edge.width = 1,
         main = paste("Detailed PPI Network for", cluster_name))
    
    # Add legend
    if ("sfari" %in% vertex_attr_names(string_graph)) {
      legend("bottomright", 
             legend = c("SFARI 1 (High)", "SFARI 2 (Strong)", "SFARI 3 (Suggestive)", 
                      "SFARI S (Syndromic)", "Other SFARI", "Not in SFARI", 
                      "Node size = Connectivity"),
             fill = c("red", "orange", "yellow", "purple", "pink", "lightblue", NA),
             cex = 0.7)
    }
    
    dev.off()
  }
  
  # Write analysis to text file
  output_file <- file.path(output_dir, paste0(cluster_name, "_ppi_analysis.txt"))
  sink(output_file)
  
  cat("PPI Network Analysis for", cluster_name, "\n")
  cat("====================================\n\n")
  cat("Number of nodes:", vcount(string_graph), "\n")
  cat("Number of edges:", ecount(string_graph), "\n\n")
  
  if (ecount(string_graph) > 0) {
    # Node connectivity
    cat("Node Connectivity:\n")
    for (i in 1:vcount(string_graph)) {
      cat("  -", V(string_graph)$name[i], ":", degree(string_graph, v=i), "connections\n")
    }
    
    # SFARI gene information
    if ("sfari" %in% vertex_attr_names(string_graph)) {
      cat("\nSFARI Gene Information:\n")
      for (i in 1:vcount(string_graph)) {
        cat("  -", V(string_graph)$name[i], ":", V(string_graph)$sfari[i], "\n")
      }
    }
  }
  
  sink()
  
  return(TRUE)
}

# Create STRING networks for all clusters with a lower initial threshold
string_networks <- list()
for (cluster_num in 1:4) {
  cluster_name <- paste0("cluster", cluster_num)
  cat("\nProcessing", cluster_name, "\n")
  string_networks[[cluster_name]] <- create_string_network(cluster_name, score_threshold=400)
  
  # Analyze the created network
  if (!is.null(string_networks[[cluster_name]])) {
    analyze_ppi_network(string_networks[[cluster_name]], cluster_name)
  }
}

# Save all networks
saveRDS(string_networks, file.path(output_dir, "string_networks.rds"))
cat("\nAll STRING networks created, analyzed and saved\n")

# Generate a single summary report
cat("\nGenerating summary report of all networks\n")
sink(file.path(output_dir, "ppi_networks_summary.txt"))

cat("STRING PPI Network Analysis Summary\n")
cat("==================================\n\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

for (cluster_num in 1:4) {
  cluster_name <- paste0("cluster", cluster_num)
  net <- string_networks[[cluster_name]]
  
  cat("\n", cluster_name, "Summary:\n")
  cat("--------------------\n")
  
  if (!is.null(net)) {
    cat("Network created with", vcount(net), "nodes and", ecount(net), "edges\n")
    
    if (ecount(net) > 0) {
      # Top genes by connectivity
      deg <- degree(net)
      top_nodes <- names(sort(deg, decreasing=TRUE)[1:min(5, length(deg))])
      cat("Top connected genes:", paste(top_nodes, collapse=", "), "\n")
      
      # SFARI genes
      if ("sfari" %in% vertex_attr_names(net)) {
        sfari_nodes <- V(net)[V(net)$sfari != "Not in SFARI"]
        if (length(sfari_nodes) > 0) {
          sfari_names <- V(net)$name[sfari_nodes]
          cat("SFARI genes:", paste(sfari_names, collapse=", "), "\n")
        } else {
          cat("No SFARI genes in network\n")
        }
      }
    } else {
      cat("No interactions found between genes\n")
    }
  } else {
    cat("No network generated\n")
  }
}

sink()
cat("Summary report generated\n")