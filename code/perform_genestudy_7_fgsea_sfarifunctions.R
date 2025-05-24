## ---------------------------------------------------------------------
## SFARI GENE PROCESSING FUNCTIONS
## ---------------------------------------------------------------------
# Gene function annotation utilities (needed by run_additional_analyses and the master)
load_gene_functions <- function(function_file = NULL, cluster_specific_genes = NULL) {
  if(is.null(function_file)) {
    function_file <- file.path(data_dir, "gene_functions.csv")
  }
  if(file.exists(function_file)) {
    gene_func_data <- read.csv(function_file, stringsAsFactors = FALSE)
    gene_functions <- setNames(as.character(gene_func_data$Function),
                               gene_func_data$Gene)
    return(gene_functions)
  } else {
    if(requireNamespace("org.Hs.eg.db", quietly = TRUE) && !is.null(cluster_specific_genes)) {
      cat("Gene function file not found; building from org.Hs.eg.db...\n")
      all_genes <- unique(cluster_specific_genes$Gene)
      entrez <- mapIds(org.Hs.eg.db, keys = all_genes,
                       column = "ENTREZID", keytype = "SYMBOL", multiVals = "first")
      desc  <- mapIds(org.Hs.eg.db, keys = entrez[!is.na(entrez)],
                      column = "GENENAME", keytype = "ENTREZID", multiVals = "first")
      gf <- setNames(as.character(desc), names(desc))
      df  <- data.frame(Gene=names(gf), Function=gf, stringsAsFactors=FALSE)
      write.csv(df, file.path(output_dir,"minimal_gene_functions.csv"), row.names=FALSE)
      return(gf)
    } else {
      warning("No gene_functions.csv and org.Hs.eg.db unavailable")
      return(list())
    }
  }
}

# Load the SFARI CSV from the global reference_dir
load_sfari_database <- function() {
  sfari_csv <- file.path(reference_dir,
    "SFARI-Gene_genes_04-03-2025release_04-21-2025export.csv")
  if (!file.exists(sfari_csv)) {
    stop("SFARI CSV not found: ", sfari_csv)
  }
  df <- read.csv(sfari_csv, stringsAsFactors = FALSE)
  return(df)
}

# Helper: find the gene symbol column
find_gene_symbol_column <- function(df) {
  for (col in c("gene.symbol","gene-symbol","gene","symbol","Gene")) {
    if (col %in% names(df)) return(col)
  }
  stop("Cannot find gene symbol column in SFARI data")
}

# Helper: find the score column
find_score_column <- function(df) {
  for (col in c("gene.score","gene-score","score","Score")) {
    if (col %in% names(df)) return(col)
  }
  return(NULL)
}

# Analyze SFARI enrichment
analyze_sfari_enrichment <- function(cluster_specific_genes, sfari_data) {
  if (is.null(sfari_data) || nrow(sfari_data)==0) {
    warning("No SFARI data")
    return(NULL)
  }
  gene_col  <- find_gene_symbol_column(sfari_data)
  score_col <- find_score_column(sfari_data)

  sfari_data[[gene_col]] <- toupper(trimws(sfari_data[[gene_col]]))
  if (!is.null(score_col)) {
    sfari_data[[score_col]] <- as.character(sfari_data[[score_col]])
    sfari_data <- sfari_data[sfari_data[[score_col]] %in% c("1","2","3","S"), ]
  }
  sfari_genes <- unique(sfari_data[[gene_col]])
  cat("Using", length(sfari_genes), "SFARI genes for enrichment\n")

  res <- data.frame(Cluster=integer(), Total_Genes=integer(),
                    SFARI_Genes=integer(), SFARI_Percent=numeric(),
                    P_Value=numeric(), stringsAsFactors=FALSE)
  all_genes <- unique(toupper(trimws(cluster_specific_genes$Gene)))
  
  for (cl in sort(unique(cluster_specific_genes$Cluster))) {
    genes <- toupper(trimws(
      cluster_specific_genes$Gene[cluster_specific_genes$Cluster==cl]
    ))
    hits <- sum(genes %in% sfari_genes)
    pct  <- hits/length(genes)*100
    pval <- phyper(hits-1, length(sfari_genes),
                   length(all_genes)-length(sfari_genes),
                   length(genes), lower.tail=FALSE)
    res <- rbind(res, data.frame(
      Cluster=cl, Total_Genes=length(genes),
      SFARI_Genes=hits, SFARI_Percent=pct,
      P_Value=pval, stringsAsFactors=FALSE
    ))
  }
  if (nrow(res)>0) res$FDR <- p.adjust(res$P_Value, method="BH")
  return(res)
}

# Plot SFARI enrichment
plot_sfari_enrichment <- function(enrichment_results, output_dir) {
  if (is.null(enrichment_results) || nrow(enrichment_results)==0) {
    warning("No SFARI enrichment to plot"); return(NULL)
  }
  enrichment_results$stars <- ""
  enrichment_results$stars[enrichment_results$P_Value<0.05]  <- "*"
  enrichment_results$stars[enrichment_results$P_Value<0.01]  <- "**"
  enrichment_results$stars[enrichment_results$P_Value<0.001] <- "***"
  p <- ggplot2::ggplot(enrichment_results,
       aes(factor(Cluster), SFARI_Percent, fill=factor(Cluster))) +
    ggplot2::geom_col(color="black") +
    ggplot2::geom_text(aes(label=sprintf("%.1f%%",SFARI_Percent)),
                       vjust=-0.5) +
    ggplot2::geom_text(aes(label=stars), vjust=-2, size=6) +
    ggplot2::labs("SFARI Gene Enrichment", x="Cluster", y="Percent") +
    ggplot2::theme_minimal() +
    ggplot2::theme(legend.position="none")
  ggplot2::ggsave(file.path(output_dir,"sfari_enrichment_by_cluster.pdf"),
                  p, width=8, height=6)
  return(p)
}

# Annotate SKAT with SFARI & functions
add_sfari_annotations <- function(cluster_specific_genes,
                                  sfari_data, gene_functions, 
                                  output_dir) {
  gene_col  <- find_gene_symbol_column(sfari_data)
  score_col <- find_score_column(sfari_data)
  df <- cluster_specific_genes %>%
    dplyr::mutate(
      Is_SFARI    = toupper(trimws(Gene)) %in% toupper(trimws(sfari_data[[gene_col]])),
      SFARI_Score = NA
    )
  for (i in seq_len(nrow(df))) {
    if (df$Is_SFARI[i]) {
      g <- toupper(trimws(df$Gene[i]))
      idx <- which(toupper(trimws(sfari_data[[gene_col]]))==g)
      if (length(idx)>0 && !is.null(score_col))
        df$SFARI_Score[i] <- sfari_data[[score_col]][idx[1]]
    }
  }
  df$Function <- sapply(df$Gene,
    function(g) if (g %in% names(gene_functions))
      gene_functions[[g]] else "Function not annotated")
  write.csv(df, file.path(output_dir,"SKAT_all_genes_SFARI_annotated.csv"),
            row.names=FALSE)
  return(df)
}

# Analyze overlap between cluster-specific genes and ASD-related gene sets
analyze_asd_gene_overlap <- function(cluster_specific_genes, asd_gene_sets) {
  overlap_summary <- data.frame()

  all_clusters <- unique(cluster_specific_genes$Cluster)
  
  for (cluster in all_clusters) {
    cluster_genes <- unique(toupper(cluster_specific_genes$Gene[cluster_specific_genes$Cluster == cluster]))
    
    for (set_name in names(asd_gene_sets)) {
      if (length(asd_gene_sets[[set_name]]) == 0) next
      
      set_genes <- toupper(asd_gene_sets[[set_name]])
      overlap <- length(intersect(cluster_genes, set_genes))
      pval <- phyper(overlap - 1, 
                     m = length(set_genes), 
                     n = length(unique(toupper(cluster_specific_genes$Gene))) - length(set_genes), 
                     k = length(cluster_genes),
                     lower.tail = FALSE)
      
      overlap_summary <- rbind(overlap_summary, data.frame(
        Cluster = cluster,
        Gene_Set = set_name,
        Overlap_Count = overlap,
        Set_Size = length(set_genes),
        Cluster_Size = length(cluster_genes),
        P_Value = pval,
        stringsAsFactors = FALSE
      ))
    }
  }

  if (nrow(overlap_summary) > 0) {
    overlap_summary$FDR <- p.adjust(overlap_summary$P_Value, method = "BH")
  }

  return(overlap_summary)
}

plot_asd_overlaps <- function(asd_overlaps) {
  if (is.null(asd_overlaps) || nrow(asd_overlaps) == 0) {
    warning("No ASD gene set overlaps to plot")
    return(NULL)
  }
  
  # Add stars for significance
  asd_overlaps$stars <- ""
  asd_overlaps$stars[asd_overlaps$P_Value < 0.05]  <- "*"
  asd_overlaps$stars[asd_overlaps$P_Value < 0.01]  <- "**"
  asd_overlaps$stars[asd_overlaps$P_Value < 0.001] <- "***"
  
  # Clean labels
  asd_overlaps$Gene_Set <- gsub("_", " ", asd_overlaps$Gene_Set)
  
  # Plot
  p <- ggplot(asd_overlaps, aes(x = factor(Cluster), y = Overlap_Count, fill = Gene_Set)) +
    geom_col(position = position_dodge()) +
    geom_text(aes(label = Overlap_Count), 
              position = position_dodge(width = 0.9), vjust = -0.25, size = 3) +
    geom_text(aes(label = stars), 
              position = position_dodge(width = 0.9), vjust = -1.2, size = 4) +
    labs(
      title = "ASD Gene Set Overlap by Cluster",
      x = "Cluster",
      y = "Overlap Count",
      fill = "ASD Gene Set"
    ) +
    theme_minimal() +
    theme(legend.position = "right")
  
  ggsave(file.path(output_dir, "asd_overlap_by_cluster.pdf"), p, width = 10, height = 6)
  return(p)
}

# Run SFARI & ASD‐specific analyses
run_additional_analyses <- function() {
  cat("\nRunning SFARI & ASD-specific analyses…\n")
  # assumes load_cluster_specific_genes(), load_sfari_database(),
  # load_gene_functions(), etc. already in scope

  # 1) load SKAT genes
  cluster_specific_genes <- load_cluster_specific_genes()

  # 2) load SFARI CSV
  sfari_data <- load_sfari_database()

  # 3) gene‐function map
  gene_funcs <- load_gene_functions(cluster_specific_genes = cluster_specific_genes)

  # 4) SFARI enrichment
  sfari_enrich <- analyze_sfari_enrichment(cluster_specific_genes, sfari_data)
  if(!is.null(sfari_enrich)) {
    write.csv(
      sfari_enrich,
      file.path(output_dir, "sfari_enrichment_by_cluster.csv"),
      row.names = FALSE
    )
    plot_sfari_enrichment(sfari_enrich, output_dir)
  }

  # 5) ASD gene‐set overlaps
  asd_gene_sets <- load_asd_gene_sets()
  asd_overlaps <- analyze_asd_gene_overlap(cluster_specific_genes, asd_gene_sets)
  if(!is.null(asd_overlaps)) {
    write.csv(
      asd_overlaps,
      file.path(output_dir, "asd_gene_set_overlaps.csv"),
      row.names = FALSE
    )
    plot_asd_overlaps(asd_overlaps)
  }

  # 6) annotate full SKAT table with SFARI & functions
  annotated <- add_sfari_annotations(
    cluster_specific_genes = cluster_specific_genes,
    sfari_data             = sfari_data,
    gene_functions         = gene_funcs,
    output_dir             = output_dir
  )

  return(list(
    sfari_enrichment = sfari_enrich,
    asd_overlaps     = asd_overlaps,
    asd_gene_sets    = asd_gene_sets
  ))
}
