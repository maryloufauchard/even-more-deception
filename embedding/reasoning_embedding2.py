import argparse
import json
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    silhouette_samples,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import umap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DATA_DIR = (".../werewolf_arena/metric/aaai/embedding")

CV_FOLDS       = 5
RANDOM_STATE   = 42

TASK_INSTRUCTION = (
    "Classify the strategic reasoning mode expressed in this werewolf game "
    "internal monologue. The mode is one of: benevolent "
    ", individualistic, or malevolent."
)
TASK_INSTRUCTION = (
    " Classify the strategic reasoning objective expressed in this werewolf game of this player. The objective can be benevolent (want their original team to win), individualistic (only want to survive), or malevolent (want the opposite team to win)."
)

PALETTE = {
    "Benevolent":      {"color": "#1f77b4", "marker": "o"},
    "Individualistic": {"color": "#ff7f0e", "marker": "s"},
    "Malevolent":      {"color": "#d62728", "marker": "^"},
}

CLUSTER_COLORS = [
    "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#e41a1c", "#377eb8",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reasoning-trace distributional analysis"
    )
    p.add_argument(
        "--model", required=True,
        help="Model tag, e.g. Qwen or Llama (inserted into filename)"
    )
    p.add_argument(
        "--player", required=True,
        help="Player tag, e.g. Villager or Werewolf (inserted into filename)"
    )
    return p.parse_args()

def angular_distance_matrix(embeddings: np.ndarray) -> np.ndarray:

    cos = cosine_similarity(embeddings)
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos) / np.pi

def encode_with_instruction(
    model: SentenceTransformer,
    texts: list[str],
    instruction: str,
    batch_size: int = 16,
) -> np.ndarray:

    prompted = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    texts = [f"Query: {t}" for t in texts]
    #return model.encode(
    #    texts, 
    #    batch_size=batch_size,
    #    normalize_embeddings=True,
    #    convert_to_numpy=True,
    #    show_progress_bar=True,
    #)
    return model.encode(
        prompted,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

def run_spectral_clustering(
    embeddings: np.ndarray,
    labels: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> dict:

    #sc = SpectralClustering(
    #    n_clusters=n_clusters,
    #    affinity="rbf",
    #    assign_labels="kmeans",
    #    random_state=random_state,
    #    n_init=20,
    #)
    #cluster_labels = sc.fit_predict(embeddings)
    
    cosine_affinity = (embeddings @ embeddings.T).clip(0, 1)

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
        n_init=20,
    )
    cluster_labels = sc.fit_predict(cosine_affinity)
 
    labels_arr = np.array(labels)
    ari = adjusted_rand_score(labels_arr, cluster_labels)
    nmi = normalized_mutual_info_score(labels_arr, cluster_labels)
 
    le       = LabelEncoder()
    true_ids = le.fit_transform(labels_arr)
    conf     = confusion_matrix(true_ids, cluster_labels)
    confusion_df = pd.DataFrame(
        conf,
        index=[f"True: {m}" for m in le.classes_],
        columns=[f"Cluster {i}" for i in range(n_clusters)],
    )

    sil_samples = silhouette_samples(embeddings, cluster_labels, metric="cosine")
    sil_mean    = float(sil_samples.mean())
    sil_per_cluster = {
        int(cid): float(sil_samples[cluster_labels == cid].mean())
        for cid in range(n_clusters)
    }

    return {
        "cluster_labels":   cluster_labels,
        "ari":              float(ari),
        "nmi":              float(nmi),
        "confusion_df":     confusion_df,
        "n_clusters":       n_clusters,
        "sil_mean":         sil_mean,
        "sil_samples":      sil_samples,
        "sil_per_cluster":  sil_per_cluster,
    }

def run_linear_probe(
    embeddings: np.ndarray,
    labels: list[str],
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:

    clf    = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
    cv     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    labels = np.array(labels)
    y_pred = cross_val_predict(clf, embeddings, labels, cv=cv)
 
    classes = sorted(np.unique(labels))
    report  = classification_report(labels, y_pred, target_names=classes, digits=3)
    cm      = confusion_matrix(labels, y_pred, labels=classes, normalize="true")
    return {
        "y_true":  labels,
        "y_pred":  y_pred,
        "report":  report,
        "cm":      cm,
        "classes": classes,
    }


def compute_tsne(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    perplexity = min(30, max(5, len(embeddings) // 5))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=random_state,
    ).fit_transform(embeddings)
 
 
def compute_umap(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    return umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=15,
        min_dist=0.1,
        random_state=random_state,
    ).fit_transform(embeddings)


def fig_projection_by_mode(
    coords_2d: np.ndarray,
    labels: list[str],
    method_name: str,
    tag: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    handles = []
    for obj, style in PALETTE.items():
        mask = np.array(labels) == obj
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=style["color"], marker=style["marker"],
            s=60, alpha=0.80, linewidths=0.4, edgecolors="white",
        )
        handles.append(mpatches.Patch(color=style["color"], label=obj))
 
    ax.set_xlabel(f"{method_name} dimension 1", fontsize=11)
    ax.set_ylabel(f"{method_name} dimension 2", fontsize=11)
    ax.set_title(
        f"{method_name} projection for model and role: {tag}",
        fontsize=11,
    )
    ax.legend(handles=handles, title="Objective mode", fontsize=10,
              title_fontsize=10, framealpha=0.7)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")

def fig_projection_by_cluster(
    coords_2d: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: list[str],
    method_name: str,
    tag: str,
    out_path: Path,
) -> None:
    n_clusters         = int(cluster_labels.max()) + 1
    cluster_colors_used = CLUSTER_COLORS[:n_clusters]

    fig, ax = plt.subplots(figsize=(8, 6))
    for cid in range(n_clusters):
        for mode, style in PALETTE.items():
            mask = (cluster_labels == cid) & (np.array(true_labels) == mode)
            if not mask.any():
                continue
            ax.scatter(
                coords_2d[mask, 0], coords_2d[mask, 1],
                c=cluster_colors_used[cid],
                marker=style["marker"],
                edgecolors=PALETTE[mode]["color"],
                linewidths=1.0,
                s=65, alpha=0.85,
            )

    cluster_handles = [
        mpatches.Patch(facecolor=cluster_colors_used[i], label=f"Cluster {i}")
        for i in range(n_clusters)
    ]
    mode_handles = [
        plt.Line2D(
            [0], [0], marker=PALETTE[m]["marker"], color="w",
            markerfacecolor="grey", markeredgecolor=PALETTE[m]["color"],
            markeredgewidth=1.5, markersize=9, label=m,
        )
        for m in PALETTE
    ]
    leg1 = ax.legend(handles=cluster_handles, title="Spectral cluster",
                     loc="upper left", fontsize=9, title_fontsize=9, framealpha=0.7)
    ax.add_artist(leg1)
    ax.legend(handles=mode_handles, title="True mode (edge/shape)",
              loc="lower right", fontsize=9, title_fontsize=9, framealpha=0.7)
    ax.set_xlabel(f"{method_name} dimension 1", fontsize=11)
    ax.set_ylabel(f"{method_name} dimension 2", fontsize=11)
    ax.set_title(
        f"{method_name} — fill = spectral cluster, edge = true mode — {tag}\n"
        "(mismatches: fill colour does not dominate one mode)",
        fontsize=11,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")

def fig_spectral_contingency(sc_result: dict, tag: str, out_path: Path) -> None:
    """Row-normalised heatmap of spectral cluster × true mode."""
    df      = sc_result["confusion_df"]
    df_norm = df.div(df.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_norm, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title(
        f"Spectral clustering contingency with {tag}\n"
        f"ARI = {sc_result['ari']:.3f}  |  NMI = {sc_result['nmi']:.3f}",
        fontsize=11,
    )
    ax.set_xlabel("Spectral clustering", fontsize=11)
    ax.set_ylabel("True objective mode", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")

def fig_probe_confusion(probe_result: dict, tag: str, out_path: Path) -> None:
    """Normalised confusion matrix from the cross-validated linear probe."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=probe_result["cm"],
        display_labels=probe_result["classes"],
    ).plot(ax=ax, colorbar=True, cmap="Blues", values_format=".2f")
    ax.set_title(
        f"Linear probe — normalised confusion matrix — {tag}\n"
        f"Stratified {CV_FOLDS}-fold cross-validation",
        fontsize=11,
    )
    ax.set_xlabel("Predicted mode", fontsize=11)
    ax.set_ylabel("True mode", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def get_centroid_nearest_traces(
    embeddings: np.ndarray,
    labels: list[str],
    texts: list[str],
    out_path: Path,
    tag: str,
    top_k: int = 3,
) -> pd.DataFrame:
    labels_arr = np.array(labels)
    modes      = sorted(np.unique(labels_arr))
    rows       = []

    for mode in modes:
        idx      = np.where(labels_arr == mode)[0]
        mode_emb = embeddings[idx]

        centroid = mode_emb.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        # Cosine similarity between each trace and the centroid
        sims = (mode_emb @ centroid)

        # Top-k closest (highest cosine sim = smallest angular distance)
        top_idx_local = np.argsort(sims)[::-1][:top_k]

        for rank, local_i in enumerate(top_idx_local, start=1):
            global_i = idx[local_i]
            rows.append({
                "mode":       mode,
                "rank":       rank,
                "cosine_sim": float(sims[local_i]),
                "trace":      texts[global_i],
            })

    df = pd.DataFrame(rows, columns=["mode", "rank", "cosine_sim", "trace"])
    df.to_csv(out_path, index=False)
    print(f"  Saved centroid-nearest traces → {out_path}")
    print("\n" + "=" * 60)
    print(f"CENTROID-NEAREST TRACES  [{tag}]  (top {top_k} per mode)")
    print("=" * 60)
    for _, row in df.iterrows():
        print(f"\n[{row['mode']}]  rank {row['rank']}  "
              f"(cos_sim = {row['cosine_sim']:.4f})")
        print("-" * 40)
        print(row["trace"])
    return df


def get_cluster_objective_representatives(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    objective_labels: list[str],
    texts: list[str],
    out_path: Path,
    tag: str,
    top_k: int = 3,
) -> pd.DataFrame:
    cluster_labels = np.asarray(cluster_labels)
    objective_labels = np.asarray(objective_labels)

    rows = []

    for cluster_id in sorted(np.unique(cluster_labels)):
        cluster_idx = np.where(cluster_labels == cluster_id)[0]
        cluster_emb = embeddings[cluster_idx]
        centroid = cluster_emb.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        for objective in sorted(np.unique(objective_labels)):
            mask = (
                (cluster_labels == cluster_id)
                & (objective_labels == objective)
            )
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            subset_emb = embeddings[idx]
            sims = subset_emb @ centroid
            n_return = min(top_k, len(idx))
            top_idx_local = np.argsort(sims)[::-1][:n_return]
            for rank, local_i in enumerate(top_idx_local, start=1):
                global_i = idx[local_i]
                rows.append({
                    "cluster": int(cluster_id),
                    "objective": objective,
                    "rank": rank,
                    "cosine_sim": float(sims[local_i]),
                    "trace": texts[global_i],
                })
    df = pd.DataFrame(
        rows,
        columns=["cluster", "objective", "rank", "cosine_sim", "trace"]
    )
    df.to_csv(out_path, index=False)
    print(f"  Saved cluster-objective representatives → {out_path}")
    print("\n" + "=" * 70)
    print(f"CLUSTER-OBJECTIVE REPRESENTATIVES [{tag}]")
    print("=" * 70)
    for cluster_id in sorted(df["cluster"].unique()):
        print(f"\n{'#'*20} Cluster {cluster_id} {'#'*20}")
        cluster_df = df[df["cluster"] == cluster_id]
        for objective in sorted(cluster_df["objective"].unique()):
            print(f"\n[{objective}]")
            subset = cluster_df[cluster_df["objective"] == objective]
            for _, row in subset.iterrows():
                print(
                    f"\nrank {row['rank']} "
                    f"(cos_sim = {row['cosine_sim']:.4f})"
                )
                print("-" * 40)
                print(row["trace"])

    return df


def main():
    args  = parse_args()
    model_tag  = args.model
    player_tag = args.player
    tag        = f"{model_tag}_{player_tag}"

    json_path  = Path(BASE_DATA_DIR) / f"reasoning_{tag}.json"
    output_dir = Path(f"results_LlamaEmbed_{tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def out(filename: str) -> Path:
        """Prefix every output filename with the model+player tag."""
        return output_dir / f"{tag}_{filename}"



    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for objective, traces in data.items():
        for idx, trace in enumerate(traces):
            rows.append({
                "objective": objective,
                "trace_id":  f"{objective}_{idx}",
                "reasoning": trace,
            })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} reasoning traces")
    print(df.groupby("objective").size().to_string())

    labels = df["objective"].tolist()
    texts  = df["reasoning"].tolist()

    print("\n" + "=" * 60)
    print("Encoding with Qwen3-Embedding-8B + task instruction")
    print("=" * 60)

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    embedder = SentenceTransformer(
        #"Qwen/Qwen3-Embedding-8B",
        #"sentence-transformers/all-MiniLM-L6-v2",
        "nvidia/llama-embed-nemotron-8b",
        trust_remote_code=True,
        device=device,
        model_kwargs={"attn_implementation": "eager"},
    )
    embeddings = encode_with_instruction(embedder, texts, TASK_INSTRUCTION)
    print(f"Embedding shape: {embeddings.shape}")
    np.save(out("embeddings.npy"), embeddings)




    print("\n" + "=" * 60)
    print("Analysis 2: Spectral Clustering")
    print("=" * 60)

    n_modes = len(set(labels))
    sc      = run_spectral_clustering(embeddings, labels,
                                      n_clusters=n_modes,
                                      random_state=RANDOM_STATE)
    ari_interp = (
        "strong"               if sc["ari"] > 0.6 else
        "moderate"             if sc["ari"] > 0.3 else
        "weak"                 if sc["ari"] > 0.0 else
        "no better than random"
    )
    print(f"  ARI : {sc['ari']:.4f}  ({ari_interp})")
    print(f"  NMI : {sc['nmi']:.4f}")
    sil_interp = (
        "strong"   if sc["sil_mean"] > 0.5 else
        "moderate" if sc["sil_mean"] > 0.25 else
        "weak"
    )
    print(f"  Silhouette (Euclidean) : {sc['sil_mean']:.4f}  ({sil_interp})")
    for cid, v in sc["sil_per_cluster"].items():
        print(f"    Cluster {cid}: {v:.4f}")
    print("\n  Contingency table:")
    print(sc["confusion_df"].to_string())

    fig_spectral_contingency(sc, tag, out("spectral_contingency.png"))
    print("\n" + "=" * 60)
    print("Centroid-nearest representative traces")
    print("=" * 60)

    #centroid_df = get_centroid_nearest_traces(
    #    embeddings=embeddings,
    #    labels=labels,
    #    texts=texts,
    #    out_path=out("centroid_nearest_traces.csv"),
    #    tag=tag,
    #    top_k=3,      # raise to 5 if you want more candidates to choose from
    #)
    representative_df = get_cluster_objective_representatives(
        embeddings=embeddings,
        cluster_labels=sc["cluster_labels"],
        objective_labels=labels,
        texts=texts,
        out_path=out("cluster_objective_representatives.csv"),
        tag=tag,
        top_k=3,
    )

    print("\n" + "=" * 60)
    print(f"Analysis 3: Linear probe ({CV_FOLDS}-fold CV)")
    print("=" * 60)

    probe   = run_linear_probe(embeddings, labels,
                               cv_folds=CV_FOLDS,
                               random_state=RANDOM_STATE)
    n_cls   = len(probe["classes"])
    chance  = 1.0 / n_cls
    print(f"  Chance baseline: {chance:.3f}  ({n_cls} classes)")
    print()
    print(probe["report"])

    fig_probe_confusion(probe, tag, out("probe_confusion.png"))

    print("\n" + "=" * 60)
    print("Computing 2-D projections")
    print("=" * 60)
 
    print("  t-SNE …")
    tsne_2d = compute_tsne(embeddings, random_state=RANDOM_STATE)
    print("  UMAP …")
    umap_2d = compute_umap(embeddings, random_state=RANDOM_STATE)
 
    for method_name, coords in [("t-SNE", tsne_2d), ("UMAP", umap_2d)]:
        slug = method_name.lower().replace("-", "tsne")  # "tsne" or "umap"
        fig_projection_by_mode(
            coords, labels, method_name, tag,
            out_path=out(f"{slug}_by_mode.png"),
        )
        fig_projection_by_cluster(
            coords, sc["cluster_labels"], labels, method_name, tag,
            out_path=out(f"{slug}_by_cluster.png"),
        )

    print(f"\nAll outputs in: {output_dir.resolve()}")
    print("\nFigure inventory:")
    for p in sorted(output_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()