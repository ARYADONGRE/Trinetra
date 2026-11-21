# Reproducibility notes and implementation details

This document describes the behavior of the supplied `src/trinetra.py` code.
It distinguishes implemented behavior from recommendations for a future
publication-grade release.

## Preprocessing and split

- Timestamps are parsed with `utc=True`; rows with invalid timestamps are removed.
- Venue filtering (`POI_MIN=10`) is applied before user filtering
  (`MIN_CHECKINS_PER_USER=10`).
- User and venue IDs are label encoded; venue IDs are shifted by one so zero is
  the padding index.
- Category indices are derived from `venueCategory` strings and shifted by one.
- Each user's check-ins are sorted chronologically.
- For a user with n check-ins, target index n-1 is test, n-2 is validation,
  and target indices 1 through n-3 are training examples.
- Every example uses up to `MAX_SEQ_LEN` preceding check-ins.
- This is a per-user leave-last-two-out target strategy, not an 80/10/10
  chronological interaction split.

## Graph construction

- Transition edges are counted from non-padding venue IDs in the generated
  training sample histories and inserted in both directions.
- The spatial graph is a bidirectional K-nearest-neighbour graph over retained
  venue coordinates, with `SPATIAL_K=10`.
- KMeans uses `N_CLUSTERS=50` and the configured seed.
- The graph encoder uses separate GATv2 layers for transition and spatial edges,
  followed by a learned sigmoid gate.

## Model and objectives

- Venue, category, cluster, time, and inter-check-in interval representations
  are combined for each sequence position.
- Multi-head cross-attention fuses graph representations into sequence features.
- A two-layer GPT-2 transformer processes the fused input.
- The prediction heads perform venue classification and coordinate regression.
- The joint training loss is:
  `classification_loss + LAMBDA_CL * contrastive_loss + LAMBDA_LOC * MSE_coordinate_loss`.
- Graph pre-training uses positive graph edges and randomly sampled negative
  edges with binary cross entropy.

## Metrics

The code computes Acc@K and MRR@K for K in (1, 5, 10, 20). `mAcc` is the
arithmetic mean of Acc@1, Acc@5, Acc@10, and Acc@20. The code also contains
an NDCG helper, but the training/evaluation loop does not report NDCG.

## Important caveats to disclose

The following points describe the supplied code as written and should be
considered before treating a run as an exact final paper reproduction:

1. **Test-set monitoring:** the script evaluates the test set after every
   training epoch. For strict model selection, use validation only during
   training, reload the selected validation checkpoint, and evaluate test once.
2. **Checkpoint naming:** `best_gpt2_model.pth` is reused across seeds and
   dataset runs, so a later run can overwrite an earlier checkpoint. Rename
   checkpoints by dataset and seed before archiving.
3. **Recorded seed metrics:** the summary records the test metrics from the
   last evaluated epoch, not metrics obtained by reloading the best validation
   checkpoint.
4. **Validation loss:** validation loss reports the classification component,
   while training loss combines classification, contrastive, and coordinate
   regression losses.
5. **Graph pre-training:** the supplied code runs 400 graph pre-training epochs
   for each seed. This can be computationally expensive.
6. **Runtime environment:** PyTorch Geometric must match the installed PyTorch
   and CUDA versions. Exact dependency versions should be recorded with results.

These caveats are documented to make the implementation auditable. If the
training/evaluation protocol is modified, report the modified code and settings
alongside the results rather than silently attributing the changes to the
supplied implementation.
