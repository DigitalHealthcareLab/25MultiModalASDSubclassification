'''
1. FMRP-Interacting Genes
2. Grove et al. ASD Common Variants
3. ASD Rare De Novo Variants (RDNV)
4. SPARK Gene List
5. ASD Transcriptionally Upregulated and Downregulated Genes

/home/data/2025_asd_wgs/reference/
├── asd_downregulated_genes.txt
├── asd_grove_genes.txt
├── asd_rdnv_genes.txt
├── asd_spark_genes.txt
├── asd_upregulated_genes.txt
└── fmrp_interacting_genes.txt

'''
#%%
import pandas as pd
from pathlib import Path

# Load the Excel file with the correct header row
fmrp_path = Path("/home/data/2025_asd_wgs/reference/fmrp_interacting_genes.xls")
df = pd.read_excel(fmrp_path, header=1)  # <- header row is the second row (index 1)

# Check available columns
print("Available columns:", df.columns.tolist())

# Extract gene symbols
fmrp_genes = df['Gene Symbol'].dropna().astype(str).unique().tolist()

# Save to .txt file
output_path = Path("/home/data/2025_asd_wgs/reference/fmrp_interacting_genes.txt")
with open(output_path, "w") as f:
    f.write("\n".join(fmrp_genes))

print(f"✅ Saved {len(fmrp_genes)} FMRP-interacting genes to {output_path}")

# %%
# Gene list from Grove et al. Table 10 (top 25 MAGMA gene-based results)
# Grove et al. ASD Common Variants
grove_gene_list = [
    "XRN2", "KCNV2", "KIZ", "KASL1", "MACROD2", "WAP75", "MATP", "MFHAS1", "XKR6", "MSRA", 
    "CHRNA1", "SOX7", "MYT1", "MMP12", "BLK", "MANBA", "ADTRP", "WDPCP", "PINX1", "PKM", 
    "PLEKHM1", "CHD7", "MDH1", "HDAC2", "WNT5B"
]

# Use relative path
output_path = Path("/home/data/2025_asd_wgs/reference/asd_grove_genes.txt")
with open(output_path, "w") as f:
    f.write("\n".join(grove_gene_list))

print(f"✅ Saved Grove gene list to: {output_path.resolve()}")

# # How to check if saved correctly
# with open(output_path, "r") as f:
#     lines = f.readlines()
#     print(f"✅ Read {len(lines)} lines from {output_path.resolve()}")
#     print("First 5 lines:", lines[:5])

# %%
# ASD Rare De Novo Variants (RDNV)
# Path to the Excel file you downloaded
rdnv_path = "/home/data/2025_asd_wgs/reference/satterstrom_tada_results.xlsx"

# Load the correct sheet ("102_ASD")
df_rdnv = pd.read_excel(rdnv_path, sheet_name="102_ASD")

# Strip whitespace from column names just in case
df_rdnv.columns = df_rdnv.columns.str.strip()

# Extract gene symbols
rdnv_genes = df_rdnv["gene"].dropna().unique().tolist()

# Save to txt file
output_path = "/home/data/2025_asd_wgs/reference/asd_rdnv_genes.txt"
with open(output_path, "w") as f:
    f.write("\n".join(rdnv_genes))

print(f"✅ Saved {len(rdnv_genes)} ASD RDNV genes to {output_path}")

# # Check the first 5 lines of the file
# with open(output_path, "r") as f:
#     lines = f.readlines()
#     print(f"✅ Read {len(lines)} lines from {output_path}")
#     print("First 5 lines:", lines[:5])

# %%
# Manually copy-paste the gene list from the image into a Python list
spark_genes = [
    "ACTB", "ADNP", "ADSL", "AFF2", "AHDC1", "ALDH5A1", "ANK2", "ANK3", "ANKRD11", "ARHGEF9", "ARID1B", "ARX",
    "ASH1L", "ASXL3", "ATRX", "AUTS2", "BAZ2B", "BCKDK", "BCL11A", "BRAF", "BRSK2", "CACNA1C", "CAPRIN1",
    "CASK", "CASZ1", "CDKL5", "CHAMP1", "CHD2", "CHD3", "CHD7", "CHD8", "CIC", "CNOT3", "CREBBP", "CSDE1",
    "CTCF", "CTNNB1", "CUL3", "DDX3X", "DEAF1", "DHCR7", "DLG4", "DMPK", "DNMT3A", "DSCAM", "DYRK1A", "EBF3",
    "EHMT1", "EIF3F (F232V)", "EP300", "FMR1", "FOXG1", "FOXP1", "GIGYF1", "GIGYF2", "GRIN2B", "HIVEP2",
    "HNRNPH2", "HNRNPU", "HRAS", "IQSEC2", "IRF2BPL", "KANSL1", "KCNB1", "KDM3B", "KDM6B", "KIAA2022", "KMT2A",
    "KMT2C", "KMT5B", "KRAS", "LZTR1", "MAGEL2", "MAP2K1", "MAP2K2", "MBD5", "MBOAT7", "MECP2", "MED13",
    "MEIS2", "MYT1L", "NAA15", "NBEA", "NCKAP1", "NF1", "NIPBL", "NLGN2", "NLGN3", "NR4A2", "NRAS", "NRXN1",
    "NRXN2", "NRXN3", "NSD1", "PACS1", "PCDH19", "PHF21A", "PHF3", "PHIP", "POGZ", "POMGNT1", "PPP1CB",
    "PPP2R5D", "PSMD12", "PTCHD1", "PTEN", "PTPN11", "RAI1", "RAI1", "RELN", "RERE", "RFX3", "RIMS1", "RIT1",
    "ROBO1", "RORB", "SCN1A", "SCN2A", "SETBP1", "SETD2", "SETD5", "SHANK2", "SHANK3", "SHOC2", "SLC6A1",
    "SLC9A6 (NHE6)", "SMARCC2", "SON", "SOS1", "SOS2", "SOX5", "SPAST", "SRCAP", "STXBP1", "SYNGAP1", "TANC2",
    "TAOK1", "TBCK", "TBR1", "TCF20", "TCF4", "TLK2", "TRIO", "TRIP12", "TSC1", "TSC2", "TSHZ3", "UBE3A",
    "UPF3B", "VPS13B", "WAC", "WDFY3", "ZBTB20", "ZNF292", "ZNF462"
]

# Output path
output_path = "/home/data/2025_asd_wgs/reference/asd_spark_genes.txt"

# Save one gene per line
with open(output_path, "w") as f:
    f.write("\n".join(sorted(set(spark_genes))))  # remove duplicates and sort

print(f"✅ Saved {len(set(spark_genes))} unique SPARK genes to {output_path}")

# %%
# ASD Transcriptionally Upregulated and Downregulated Genes
# Using Gandal et al. Supplementary Data 5

# Path to the Excel file
gene_level_xlsx = "/home/data/2025_asd_wgs/reference/gandal_supplementary5.xlsx"

# Load the Gene_Level sheet
df_gene = pd.read_excel(gene_level_xlsx, sheet_name="Gene_Level")

# Clean column names just in case
df_gene.columns = df_gene.columns.str.strip()

# Check available columns
assert 'WGCNA_module' in df_gene.columns, "Missing 'WGCNA_module' column"
assert 'hgnc_symbol' in df_gene.columns, "Missing 'hgnc_symbol' column"

# Filter genes in M12_tan (downregulated) and M16_lightcyan (upregulated)
downregulated = df_gene[df_gene['WGCNA_module'] == "M12_tan"]['hgnc_symbol'].dropna().unique()
upregulated = df_gene[df_gene['WGCNA_module'] == "M16_lightcyan"]['hgnc_symbol'].dropna().unique()

# Output directory
output_dir = Path("/home/data/2025_asd_wgs/reference")

# Save to .txt files
with open(output_dir / "asd_downregulated_genes.txt", "w") as f:
    f.write("\n".join(downregulated))

with open(output_dir / "asd_upregulated_genes.txt", "w") as f:
    f.write("\n".join(upregulated))

print(f"✅ Saved {len(downregulated)} downregulated genes and {len(upregulated)} upregulated genes.")

# %%
## Check the saved files ## 
import os

# Set your reference directory
reference_dir = "/home/data/2025_asd_wgs/reference"

# List of filenames to check
gene_set_files = [
    "fmrp_interacting_genes.txt",
    "asd_grove_genes.txt",
    "asd_rdnv_genes.txt",
    "asd_spark_genes.txt",
    "asd_upregulated_genes.txt",
    "asd_downregulated_genes.txt"
]

# Function to preview contents
def preview_gene_sets(files, directory, n_preview=5):
    for file in files:
        path = os.path.join(directory, file)
        print(f"\n📁 Previewing {file} ...")
        if os.path.exists(path):
            with open(path, "r") as f:
                genes = [line.strip() for line in f if line.strip()]
                print(f"✅ {len(genes)} genes found.")
                print("Top genes:", genes[:n_preview])
        else:
            print(f"❌ File not found: {path}")

# Run it
preview_gene_sets(gene_set_files, reference_dir)


# %%
