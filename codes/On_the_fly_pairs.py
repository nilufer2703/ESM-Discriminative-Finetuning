def build_proteinmpnn_preference_pairs_from_batch(
    batch_rows,
    threshold,
    max_pairs=None,
):
    if len(batch_rows) < 2:
        return []

    targets = np.array(
        [float(row["target"]) for row in batch_rows],
        dtype=np.float32,
    )

    diff_matrix = np.abs(targets[:, None] - targets[None, :])
    upper_tri = np.triu(
        np.ones((len(batch_rows), len(batch_rows)), dtype=bool),
        k=1,
    )

    valid_pair_mask = (diff_matrix > threshold) & upper_tri
    valid_pair_indices = np.argwhere(valid_pair_mask)

    if len(valid_pair_indices) == 0:
        return []

    valid_pair_indices = valid_pair_indices.tolist()
    random.shuffle(valid_pair_indices)

    if max_pairs is None:
        max_pairs = len(batch_rows)

    selected_pairs = valid_pair_indices[:max_pairs]

    paired_items = []

    for i, j in selected_pairs:
        row_i = batch_rows[i]
        row_j = batch_rows[j]

        if float(row_i["target"]) > float(row_j["target"]):
            pos_row = row_i
            neg_row = row_j
        else:
            pos_row = row_j
            neg_row = row_i

        paired_items.append(
            {
                "pair_id": f"{pos_row['sequence_id']}__vs__{neg_row['sequence_id']}",
                "pdb_id": pos_row["pdb_id"],
                "pos_sequence": pos_row["sequence"],
                "neg_sequence": neg_row["sequence"],
                "pos_target": float(pos_row["target"]),
                "neg_target": float(neg_row["target"]),
                "target_gap": abs(float(pos_row["target"]) - float(neg_row["target"])),
                "pos_sequence_id": pos_row["sequence_id"],
                "neg_sequence_id": neg_row["sequence_id"],
                "neg_source": "on_the_fly",
            }
        )

    return paired_items
