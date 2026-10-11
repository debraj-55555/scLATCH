# Priors

scLATCH reads its priors from this folder. The regulatory mask keeps a TF-to-target edge when
ChIP-Atlas reports binding of the TF within 10 kb of the target's TSS and the region is
accessible in scATAC.

| Path | Content | In repo |
|---|---|---|
| `allTFs_hg38.txt`, `allTFs_mm.txt` | SCENIC+ TF lists, human and mouse (`tf_list="scenicplus"`, default) | yes |
| `hs_hgnc_tfs.txt`, `mm_mgi_tfs.txt` | pySCENIC TF lists, human and mouse (`tf_list="pyscenic"`) | yes |
| `tss/TSS_extend_1M.txt`, `tss/mm10_gene_tss.csv` | TSS tables, hg38 and mm10 | yes |
| `chip_atlas/{hg38,mm10}/{TF}.10.tsv` | ChIP-Atlas binding within 10 kb of the TSS, one file per TF | download |

## ChIP-Atlas cache

Needed only to build a new mask. Download `scLATCH_priors_chip_atlas.tar.gz` (335 MB, 1.7 GB
unpacked) from
[Google Drive](https://drive.google.com/drive/folders/1Uj2zvCKHfi9ET7Ud1NzR39VKzNwefF-p?usp=sharing)
and extract it from the repository root:

```bash
tar -xzf scLATCH_priors_chip_atlas.tar.gz -C priors/
sha256sum -c --quiet priors/chip_atlas/SHA256SUMS    # optional integrity check
```

This gives 1,183 human files in `chip_atlas/hg38/` and 498 mouse files in `chip_atlas/mm10/`.
The cache is the ChIP-Atlas release downloaded from 30 March to 10 April 2026. Pass
`allow_download=False` (command line: `--no_download`) to build masks from this cache only;
otherwise a TF missing from the cache is downloaded from ChIP-Atlas, whose data change over
time. To keep the cache elsewhere, set `SCLATCH_CHIPATLAS_DIR` (command line: `--cache_dir`).

## Notes

- The bundled `huang_tcell` mask was built with the pySCENIC list and the `pbmc_10k` mask with
  the SCENIC+ list. Use the same `tf_list` to rebuild them.
- If you install scLATCH with pip without cloning the repository, set `SCLATCH_DATA_ROOT` to the
  folder that contains `priors/`.

## Sources

ChIP-Atlas (https://chip-atlas.org); SCENIC+ TF lists
(https://resources.aertslab.org/cistarget/tf_lists/); pySCENIC TF lists
(https://github.com/aertslab/pySCENIC).
