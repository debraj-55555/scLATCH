"""scLATCH preprocessing pipeline.

Usage (one-shot):
    from sclatch.preprocessing import preprocess
    adata = preprocess("raw_data/pbmc_10k", organism="human",
                       out_dir="processed/pbmc_10k")

Usage (step-by-step):
    from sclatch.preprocessing import io, qc, normalize, embed, pseudotime, export

    adata_rna, adata_atac = io.load_data("raw_data/pbmc_10k")
    adata_rna = qc.run_qc(adata_rna, organism="human")
    adata_rna = normalize.normalize(adata_rna)
    adata_rna = embed.embed(adata_rna)
    adata_rna = pseudotime.compute_pseudotime(adata_rna)
    export.export(adata_rna, "processed/pbmc_10k")
"""

from sclatch.preprocessing.pipeline import preprocess

__all__ = ["preprocess"]
