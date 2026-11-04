# doggo-dataset validation report

Generated: 2026-10-30T07:38:15+00:00

**Overall: PASS**

## Image count per species

| Species | Label | Expected | Found |
|---|---|---:|---:|
| dog | dog | 100 | 100 |
| wolf | non_dog | 10 | 10 |
| fox | non_dog | 10 | 10 |
| coyote | non_dog | 10 | 10 |
| dingo | non_dog | 10 | 10 |
| cat | non_dog | 10 | 10 |
| koala | non_dog | 10 | 10 |
| red_panda | non_dog | 10 | 10 |
| raccoon | non_dog | 10 | 10 |
| bear | non_dog | 10 | 10 |
| hyena | non_dog | 10 | 10 |
| **total** | | **200** | **200** |

## Checks

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | exactly_200_images | PASS | 200 image files found (expected 200) |
| 2 | dog_count_100 | PASS | 100 dog images (expected 100) |
| 3 | ten_per_non_dog_species | PASS | wolf=10, fox=10, coyote=10, dingo=10, cat=10, koala=10, red_panda=10, raccoon=10, bear=10, hyena=10 |
| 4 | all_images_decode | PASS | 200 images decoded |
| 5 | no_zero_byte_or_corrupt_files | PASS | zero-byte: 0, corrupt: 0 |
| 6 | no_duplicate_files | PASS | no exact SHA-256 duplicates |
| 7 | no_near_duplicate_files | PASS | no perceptual near-duplicates |
| 8 | manifest_matches_images | PASS | missing entries: none; orphan entries: none |
| 9 | source_and_license_traceable | PASS | 200 entries verified |
| 10 | no_unexpected_files_in_image_dirs | PASS | only expected .jpg files |
| 11 | dingo_labelled_non_dog | PASS | 10 dingo rows all labelled non_dog |
| 12 | label_species_consistency | PASS | labels match the binary definition |
| 13 | jpeg_format_only | PASS | formats found: ['JPEG'] |

## Files

- `metadata/manifest.csv` — per-image source and licence metadata
- `metadata/species_summary.json` — counts and label definition
- `metadata/failed_or_replaced.csv` — rejected/replaced candidates and reasons
