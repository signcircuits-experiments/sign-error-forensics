# sign-circuit-anon

Anonymous release for a pre-registered mechanistic-interpretability study of
sign errors in Qwen2.5-72B-Instruct (4×4 determinants, plus an integration-by-parts
transfer domain). A small set of late MLPs (L75/L78) writes the minus sign; a
separate late component (L79) detects wrong signs but never overrides them.
Sign errors are commitment failures, not knowledge failures.

## Folder map

| Folder | Contents |
|---|---|
| `preregistration_qwen/` | `PREREGISTRATION.md` — frozen data counts, frozen instruments, 17 numeric pass bars, exclusions. Committed before held-out measurement. |
| `pipeline_release/` | The full experiment pipeline (stages s01–s09, one driver, model adapters). See `pipeline_release/PIPELINE.md`. |
| `Qwen/raw_data_4x4_DET/` | Frozen experiment_ready inputs: discovery (59 errors / 47 corrects) and held-out (147 / 141). |
| `Qwen/raw_data_IBP/` | Frozen experiment_ready inputs for the IBP domain: discovery (10 / 25) and held-out (26 / 50). |
| `Qwen/discovery_4x4_DET/` | Per-stage result JSONs from the discovery set. |
| `Qwen/heldout_4x4_DET/` | Per-stage result JSONs from the single pre-registered held-out run. |
| `Qwen/workbooks_4x4_DET/` | 10 per-stage discovery-vs-held-out Excel workbooks + `FINDINGS_det_4x4.md` (the narrative summary and verdicts: 12/17 bars confirmed, 2 misses). |
| `Qwen/discovery_IBP/` | Per-stage IBP result JSONs from the discovery set. |
| `Qwen/heldout_IBP/` | Per-stage IBP result JSONs from the single pre-registered held-out run, plus the one-shot pod runs (`s09_judge/*_L66.json`, `s06_direction_subtraction/*_a3.json`). |
| `Qwen/workbooks_IBP/` | 9 per-stage IBP discovery-vs-held-out Excel workbooks + `FINDINGS_ibp.md` (narrative summary: minus-writer machinery replicates; judge reversed at L63, null at L66 — no identified judge). |
| `predeclarations/` | Second-generation pre-declared tests, written after the original pre-registration froze and content-frozen BEFORE execution. Currently: `2026-08_ibp_L66_armA.md` (single-shot causal judge test at L66, IBP domain — executed once; outcome recorded in the file, run log alongside as `2026-08_ibp_L66_armA_run.log`). |

## Reproducing

1. `pip install -r requirements.txt`
2. Run the pipeline on the frozen inputs: see `pipeline_release/PIPELINE.md`
   (stages s01→s09; s03-derived `targets.json` files are included frozen —
   s00/s03 are excluded from held-out runs by the pre-registration).
3. Raw stage outputs → released per-case JSONs: the s05 release files are
   reproduced exactly by
   `pipeline_release/tools/derive_s05_release_json.py --verify <released file>`.
4. Every number in `FINDINGS_det_4x4.md`, `FINDINGS_ibp.md` and the workbooks
   traces to the raw JSONs in `discovery_4x4_DET/`, `heldout_4x4_DET/`,
   `discovery_IBP/` and `heldout_IBP/`.

## Integrity

`CHECKSUMS.sha256` covers the frozen input data and the released result JSONs.
Verify with:

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

## Note on commit timing

The pre-registration file was frozen (content-final) before the held-out data
was measured; repository assembly and the public commits happened afterward, so
the commit sequence (pre-registration first, results second) preserves the
ordering of the protocol rather than the original calendar dates.

## License

Code: MIT. Data (`Qwen/`): CC-BY-4.0. See `LICENSE`.
