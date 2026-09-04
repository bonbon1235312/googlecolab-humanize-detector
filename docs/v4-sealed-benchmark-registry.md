# V4 sealed benchmark registry

## RAID-derived paraphrase cohort, version 1

| Field | Value |
| --- | --- |
| Claim scope | RAID-derived unseen-source paraphrase cohort; not RAID's official hidden test split. |
| Source | [liamdugan/raid](https://huggingface.co/datasets/liamdugan/raid) |
| Dataset configuration | `raid` |
| Dataset split | `train` |
| Pinned revision | `865cac74188466cb0c3b7574a10204007b57a459` |
| Selection rule | One human `attack=none` row and one non-human `attack=paraphrase` row per `source_id`, selected by deterministic SHA-256 source-family rank. |
| Selection seed | `20260904` |
| Sealed at | `2026-09-04T18:36:15.869304+00:00` |
| Source-candidate snapshot SHA-256 | `48ad764f069d0fde425ef75edfe076a4dc9e7e96c64e30abfd3818cc9c7aaf58` |
| Metadata-manifest SHA-256 | `36f3514f104f97000c61fccb730c9b253f73e47c0ff8b3021e24e5d1e269654e` |
| Cohort size | 2,500 source-family pairs / 5,000 records |

The sealed cohort resides outside Git at `MyDrive/v4-data/raid-paraphrase-sealed-v1`. Do not open its JSONL records, use it for training or calibration, rerun selection into the same directory, or evaluate V4 models against it until model selection and calibration are frozen.

This registry deliberately contains only source metadata and hashes; it contains no benchmark text or labels.
