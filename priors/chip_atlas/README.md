# priors/chip_atlas/ : ChIP-Atlas binding cache (download)

Per-TF ChIP-Atlas binding tables go here, split by genome, as `{TF}.10.tsv`:

```
priors/chip_atlas/
├── hg38/{TF}.10.tsv     # human, 1,183 files, 988 MB
└── mm10/{TF}.10.tsv     # mouse,    498 files, 683 MB
```

These are large (about 1.7 GB total) and are not in the repo. Download
`scLATCH_priors_chip_atlas.tar.gz` from Google Drive `<DRIVE_LINK>` (placeholder) and run
`tar -xzf scLATCH_priors_chip_atlas.tar.gz -C priors/` from the repository root; the files
land here. `sclatch.build_prior` reads this location by default (override with
`SCLATCH_CHIPATLAS_DIR` or `cache_dir`). You only need this cache to rebuild a mask from
scratch; precomputed masks are provided and loaded automatically when present.
