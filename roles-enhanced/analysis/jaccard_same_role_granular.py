r"""
Granular Jaccard overlap heatmaps for same-role variants.

Uses only non-noise HDBSCAN cluster ids, not human-merged labels.

Run:
.\.venv\Scripts\python.exe rag_pipeline\roles-enhanced\analysis\jaccard_same_role_granular.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "context_barrier_mention_hdbscan_gpt5named_human.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "jaccard_overlap_heatmap_3panel_same_role_granular.png"

ROLE_ORDER = ["generalist", "normative", "subject_matter"]
ROLE_TITLES = {
    "generalist": "Generalist Variants",
    "normative": "Normative Variants",
    "subject_matter": "SME Variants",
}
FAMILY_TITLES = {
    "phi4": "Phi-4",
    "gemma3": "Gemma3",
    "llama": "Llama",
    "mistral": "Mistral",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Row-level clustered CSV input.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Heatmap PNG output path.")
    return parser.parse_args()


def parse_variant_meta(variant_id: str) -> tuple[str, str, int]:
    variant_id = str(variant_id).strip()
    if not variant_id:
        raise ValueError("Empty variant_id encountered.")

    if variant_id.startswith("phi4-"):
        family = "phi4"
        rest = variant_id[len("phi4-") :]
    elif variant_id.startswith("gemma3-"):
        family = "gemma3"
        rest = variant_id[len("gemma3-") :]
    elif variant_id.startswith("llama-"):
        family = "llama"
        rest = variant_id[len("llama-") :]
    elif variant_id.startswith("mistral-"):
        family = "mistral"
        rest = variant_id[len("mistral-") :]
    else:
        raise ValueError(f"Unrecognized variant_id format: {variant_id}")

    version = 2 if rest.endswith("2") else 1
    role = rest[:-1] if version == 2 else rest
    return family, role, version


def variant_pretty(variant_id: str) -> str:
    family, _role, version = parse_variant_meta(variant_id)
    return f"{FAMILY_TITLES[family]} V{version}"


def build_variant_sets(df: pd.DataFrame) -> dict[str, set[int]]:
    return {
        variant_id: set(pd.to_numeric(sub["cluster"], errors="coerce").dropna().astype(int).tolist())
        for variant_id, sub in df.groupby("variant_id")
    }


def jaccard_score(variant_sets: dict[str, set[int]], v1: str, v2: str) -> float:
    s1 = variant_sets.get(v1, set())
    s2 = variant_sets.get(v2, set())
    union = s1 | s2
    inter = s1 & s2
    return len(inter) / len(union) if union else 0.0


def keep_cell(v1: str, v2: str) -> bool:
    if v1 == v2:
        return False

    family1, role1, version1 = parse_variant_meta(v1)
    family2, role2, version2 = parse_variant_meta(v2)

    if role1 != role2:
        return False

    # Keep within-family profile comparisons: 1 vs 2
    if family1 == family2 and version1 != version2:
        return True

    # Keep cross-family direct matches only: 1 vs 1 and 2 vs 2
    if family1 != family2 and version1 == version2:
        return True

    return False


def role_group_variants(df: pd.DataFrame) -> dict[str, list[str]]:
    variant_ids = sorted(set(df["variant_id"].astype(str).str.strip()))
    groups: dict[str, list[str]] = {role: [] for role in ROLE_ORDER}
    family_order = {"phi4": 0, "gemma3": 1, "llama": 2, "mistral": 3}

    for role in ROLE_ORDER:
        role_variants = [v for v in variant_ids if parse_variant_meta(v)[1] == role]
        role_variants.sort(key=lambda v: (family_order[parse_variant_meta(v)[0]], parse_variant_meta(v)[2], v))
        groups[role] = role_variants
    return groups


def build_panel(variants: list[str], variant_sets: dict[str, set[int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pretty_labels = [variant_pretty(v) for v in variants]

    panel = pd.DataFrame(index=pretty_labels, columns=pretty_labels, dtype=float)
    mask = pd.DataFrame(True, index=pretty_labels, columns=pretty_labels)

    for v1 in variants:
        for v2 in variants:
            label1 = variant_pretty(v1)
            label2 = variant_pretty(v2)
            panel.loc[label1, label2] = jaccard_score(variant_sets, v1, v2)
            if keep_cell(v1, v2):
                mask.loc[label1, label2] = False

    return panel, mask


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
    df["variant_id"] = df["variant_id"].fillna("").astype(str).str.strip()
    df = df[(df["cluster"] >= 0) & (df["variant_id"] != "")].copy()

    variant_sets = build_variant_sets(df)
    role_groups = role_group_variants(df)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    cmap = plt.get_cmap("OrRd").copy()
    cmap.set_bad("#f3f3f3")

    for ax, role in zip(axes, ROLE_ORDER):
        variants = role_groups[role]
        panel, mask = build_panel(variants, variant_sets)
        pretty_labels = list(panel.index)
        data = panel.to_numpy(dtype=float)
        masked = data.copy()
        masked[mask.to_numpy(dtype=bool)] = float("nan")

        if sns is not None:
            sns.heatmap(
                panel,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                vmin=0,
                vmax=1,
                square=True,
                linewidths=1,
                linecolor="#9e9e9e",
                cbar=(role == ROLE_ORDER[-1]),
                cbar_kws={"label": "Jaccard Similarity Index"} if role == ROLE_ORDER[-1] else None,
                ax=ax,
            )
        else:
            image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)
            ax.set_aspect("equal")

            if role == ROLE_ORDER[-1]:
                cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Jaccard Similarity Index")

            for i in range(masked.shape[0]):
                for j in range(masked.shape[1]):
                    if pd.notna(masked[i, j]):
                        ax.text(j, i, f"{masked[i, j]:.2f}", ha="center", va="center", fontsize=9)

            ax.set_xticks(range(len(pretty_labels)))
            ax.set_yticks(range(len(pretty_labels)))
            ax.set_xticklabels(pretty_labels)
            if role != ROLE_ORDER[0]:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            else:
                ax.set_yticklabels(pretty_labels)

            ax.set_xticks([x - 0.5 for x in range(1, len(pretty_labels))], minor=True)
            ax.set_yticks([y - 0.5 for y in range(1, len(pretty_labels))], minor=True)
            ax.grid(which="minor", color="#9e9e9e", linestyle="-", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_title(ROLE_TITLES[role], fontsize=12)
        ax.set_xlabel("Variants")
        ax.set_ylabel("Variants" if role == ROLE_ORDER[0] else "")

        ax.set_xticklabels(pretty_labels, rotation=30, ha="right", rotation_mode="anchor")

        if role == ROLE_ORDER[0]:
            ax.set_yticklabels(pretty_labels, rotation=0)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#9e9e9e")

    fig.suptitle("Jaccard similarity index heatmaps for same-role variants", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded rows: {len(df)}")
    print(f"Saved heatmap: {output_path}")
    for role in ROLE_ORDER:
        print(f"{role}: {', '.join(role_groups[role])}")


if __name__ == "__main__":
    main()
