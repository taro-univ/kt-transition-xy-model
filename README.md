# KT転移の教師なし表現学習

2次元 XY モデルの Kosterlitz–Thouless (KT) 転移を対象に、  
**物理制約付きコントラスト学習**により、教師なしで転移構造を潜在空間に抽出する。

---

## 研究の動機

従来の機械学習による相転移解析は「二値分類」として定式化されることが多い。  
しかし KT 転移には **自発的対称性の破れがなく**、秩序変数による二値分類は原理的に成立しない。

本研究は分類ではなく次の問いを立てる。

> **教師ラベルなしに、潜在空間はKT転移の物理構造を反映できるか？**

転移の検出を「目的関数の設計による表現の質」の問題として再定式化し、  
ヘリシティモジュラス Υ を弱い物理制約として組み込んだ提案手法を評価する。

---

## 物理的背景

### 2次元 XY モデル

格子サイト $i$ にスピン角度 $\phi_i \in (-\pi, \pi]$ を置く O(2) 対称モデル。

$$H = -J \sum_{\langle i,j \rangle} \cos(\phi_i - \phi_j)$$

### KT 転移

通常の二次転移（自発的対称性の破れ）とは異なり、  
**ボルテックス対（渦と反渦）の束縛↔解束縛**というトポロジカルな機構で転移が起きる。

| 温度域 | 物理描像 |
|--------|----------|
| $T < T_c$ | ボルテックス対が束縛。スピン波支配。$G(r) \sim r^{-\eta(T)}$ のべき乗則減衰 |
| $T \approx T_c$ | ヘリシティモジュラスが普遍的ジャンプ $\Upsilon(T_c^-) = 2T_c/\pi$ を示す |
| $T > T_c$ | ボルテックスが自由に解束縛。$G(r) \sim e^{-r/\xi}$ の指数関数的減衰 |

### 測定量

**ヘリシティモジュラス** $\Upsilon$（超流動剛性に相当）

$$\Upsilon = \langle \cos \Delta\phi \rangle - \beta \frac{\langle (\sum \sin \Delta\phi)^2 \rangle}{L^2}$$

$\Upsilon > 0$ が「位相コヒーレンスの秩序度」を示し、KT 転移の最も鋭敏な指標となる。

---

## 提案手法：ヘリシティ制約付きコントラスト学習

### 4モデルの比較

| モデル | 目的関数 |
|--------|----------|
| AE | $\mathcal{L} = \|x - \hat{x}\|^2$ |
| VAE | $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \, D_{\text{KL}}$ |
| Contrastive (SimCLR) | $\mathcal{L} = \mathcal{L}_{\text{NT-Xent}}$ |
| **Helicity-Contrastive（提案）** | $\mathcal{L} = \mathcal{L}_{\text{NT-Xent}} + \lambda_\Upsilon \, \mathcal{L}_\Upsilon$ |

### 提案手法の設計

エンコーダ出力 $h$（projector より前段）に対して、  
ヘリシティ回帰ヘッドを接続し弱い物理バイアスを注入する。

```
入力 φ → Encoder → h ──→ Projector → z → NT-Xent 損失
                    └──→ Helicity Head → Υ̂ → MSE 損失
```

物理情報はあくまで「弱い制約」として機能し、  
latent 空間の自由度を潰さずに物理構造を誘導することが狙い。

---

## 主要な結果

### UMAP 潜在空間の可視化（4モデル比較）

提案手法（右下）のみ、温度に沿った連続的な構造が現れる。

![UMAP 4モデル比較](Unsupervised_Machine_Learning/results/notebook_data/umap_compare_4models.png)

---

### 潜在変数の温度依存性

提案手法では複数の潜在次元が温度に対して単調・協調的に変化する。  
AE・VAE・Contrastive では温度への感応性がほとんど見られない。

![潜在変数 vs T](Unsupervised_Machine_Learning/results/notebook_data/latent_vs_T_4models.png)

---

### クラスター確率の温度依存性（3クラスタリング）

提案手法（右下）では 3 つのクラスターが温度に沿って**非重複かつ単調に遷移**し、  
低温相・転移域・高温相の 3 レジーム構造を自律的に発見している。  
他の 3 モデルではクラスターが温度に対してほぼランダムに混在する。

![クラスター確率 vs T](Unsupervised_Machine_Learning/results/notebook_data/4model_Cluster_prob_vs_T.png)

---

### 転移感度：潜在勾配のピーク温度

提案手法で相関上位の潜在次元は、勾配ピークが $T \approx 0.88$–$1.06$ に集中し、  
理論値 $T_c \approx 0.893$（$L=32$有限サイズ推定）と整合する。

![選択潜在次元の勾配ピーク](Unsupervised_Machine_Learning/results/notebook_data/Y_helicity_contrastive_latent_small_selected_mean_vs_T_with_peaks.png)

---

### 相関関数の条件付き可視化

クラスターラベルを条件に $G(r)$ を可視化。  
$T=0.80$（低温）ではべき乗則、$T=1.20$（高温）では指数則という  
KT 理論と整合した崩壊挙動を各クラスターが反映している。

![条件付き G(r)](Unsupervised_Machine_Learning/results/notebook_data/cluster_cond_G_r_T080_T100_T120.png)

---

### 物理量との相関（Spearman 係数）

提案手法の潜在次元と物理量との Spearman 相関（上位次元）：

| 物理量 | 上位次元 | Spearman 相関係数 |
|--------|----------|-------------------|
| ヘリシティモジュラス Υ | z[581] | **0.659** |
| ボルテックス密度 $n_v$ | z[1116] | **0.791** |

教師なし学習にもかかわらず、KT 転移を特徴づける 2 つの物理量と有意な相関を示す次元が複数存在する。

---

### 4モデル定量比較

| モデル | 潜在空間の温度構造 | 3レジーム分離 | Spearman (Υ) | Spearman ($n_v$) |
|--------|:-----------------:|:------------:|:------------:|:----------------:|
| AE | 不明瞭 | × | — | — |
| VAE | 不明瞭 | × | — | — |
| Contrastive | 不明瞭 | × | — | — |
| **Helicity-Contrastive（提案）** | **明確** | **○** | **0.659** | **0.791** |

AE・VAE・Contrastive の潜在次元は温度に対してほぼ無感応。  
提案手法のみが教師なしで 3 レジームを分離し、物理量との高い相関を示す。

---

## ファイル構成

```
kt-transition-xy-model/
├── MCsim/                          # モンテカルロシミュレーション
│   ├── config.py                   # 実験設定（JSON シリアライズ対応）
│   ├── xy_model.py                 # データ構造（XYState, RunBuffers, CorrBins）
│   ├── angles.py                   # 角度演算・PBC ユーティリティ
│   ├── update.py                   # Metropolis / Over-relaxation / Wolff 更新
│   ├── measure.py                  # 物理量測定（Υ, G(r), ボルテックス, 比熱）
│   ├── analysis.py                 # Tc 推定・べき乗則/指数フィット
│   ├── initial.py                  # 格子初期化（ランダム・一様・ボルテックス対）
│   ├── loop.py                     # 温度スイープの実行ループ
│   └── check.py                    # KT 理論との整合性確認
│
└── Unsupervised_Machine_Learning/
    ├── configs/                    # YAML 実験設定
    ├── models/
    │   ├── nn_utils.py             # 共通ブロック（ConvBlock, DeconvBlock, get_activation）
    │   ├── auto_encoder.py         # Convolutional AE
    │   ├── vae.py                  # Convolutional VAE
    │   ├── contrastive_encoder.py  # SimCLR エンコーダ
    │   └── helicity_head.py        # ヘリシティ回帰ヘッド（提案手法用）
    ├── dataset/
    │   └── unsupervised_xy_dataset.py  # XY スピンデータセット・データ拡張
    ├── train/
    │   ├── train_contrastive.py          # SimCLR 学習スクリプト
    │   └── train_helicity_contrastive.py # 提案手法の学習スクリプト
    ├── analysis/
    │   ├── latent_extraction.py          # 潜在変数の抽出（全モデル対応）
    │   ├── latent_vs_T.py                # 潜在変数の温度依存性
    │   ├── cluster_vs_T.py               # クラスター確率の温度依存性
    │   ├── kmeans_tsne.py                # K-means + t-SNE 可視化
    │   ├── corr_ranking_latent_vs_observables.py  # 物理量との相関ランキング
    │   ├── Cluster_cond_G_r.py           # クラスター条件付き G(r)
    │   └── plot_selected_latent_vs_T.py  # 選択次元の勾配ピーク可視化
    └── utils/
        ├── lambda_schedule.py    # λ スケジューラ
        └── physics_utils.py      # 物理ユーティリティ（ML モジュール独立版）
```

---

## 環境構築

```bash
git clone https://github.com/taro-univ/kt-transition-xy-model.git
cd kt-transition-xy-model
pip install -r requirements.txt
```

**動作確認済み環境**

| ライブラリ | バージョン |
|------------|------------|
| Python | 3.11 |
| PyTorch | >= 2.2 |
| NumPy | >= 1.26 |
| scikit-learn | >= 1.4 |

> **再現性について**: `MCsim/config.py` の `seed` に整数を指定すると乱数を固定できる（例: `seed=42`）。
> PyTorch の再現性は `torch.manual_seed()` でトレーニングスクリプト側で制御する。

---

## 実験パイプライン

### Step 1 — モンテカルロシミュレーション（データ生成）

```bash
python MCsim/loop.py
```

`MCsim/config.py` でシステムサイズ・温度グリッド・スイープ数・乱数シードを管理。  
出力は `results/run-{timestamp}/` 以下に JSON 形式で保存され、設定も `config.json` として記録される。

### Step 2 — 学習データセットの準備

シミュレーション結果から `.npz` 形式のデータセットを作成する。  
フォーマット：スピン配置 `phi (N, L, L)`、温度ラベル `T (N,)`、ヘリシティ `Y (N,)`。

### Step 3 — モデルの学習

```bash
# 提案手法（ヘリシティ制約付きコントラスト学習）
python Unsupervised_Machine_Learning/train/train_helicity_contrastive.py \
    --config Unsupervised_Machine_Learning/configs/helicity_contrastive.yaml

# SimCLR ベースライン
python Unsupervised_Machine_Learning/train/train_contrastive.py \
    --config Unsupervised_Machine_Learning/configs/contrastive.yaml
```

### Step 4 — 潜在変数の抽出

```bash
python Unsupervised_Machine_Learning/analysis/latent_extraction.py \
    --model_type helicity_contrastive
```

`autoencoder` / `vae` / `contrastive` / `helicity_contrastive` の 4 種類に対応。

### Step 5 — 解析・可視化

```bash
# 潜在変数の温度依存性
python Unsupervised_Machine_Learning/analysis/latent_vs_T.py \
    --latent Unsupervised_Machine_Learning/results/latent/helicity_contrastive/helicity_contrastive_latent.npz

# 物理量との相関ランキング
python Unsupervised_Machine_Learning/analysis/corr_ranking_latent_vs_observables.py \
    --latent <latent.npz> --mc_npz <dataset.npz> --targets Y,nv
```

---

## ノートブック

| ノートブック | 内容 |
|---|---|
| `MCsim/Efficient_Monte_Carlo_Simulation.ipynb` | ハイブリッド MCMC の検証・ボルテックス可視化・G(r) の KT 理論整合性 |
| `Unsupervised_Machine_Learning/01_representation_learning.ipynb` | 4モデルの UMAP 比較・クラスタリング・物理量との相関・勾配ピーク解析 |

### 解析スクリプトと生成物の対応

ノートブック内の各図は以下のスクリプトで生成されている。  
ヘビーな計算はスクリプトに分離し、ノートブックは可視化・比較に専念する設計。

| 生成物 | スクリプト |
|--------|-----------|
| `*_latent.npz`（潜在変数） | `analysis/latent_extraction.py` |
| `*_mean_vs_T.png`（潜在変数 vs 温度） | `analysis/latent_vs_T.py` |
| `*_cluster_vs_T.png`（クラスター確率） | `analysis/cluster_vs_T.py` |
| `*_tsne_*.png`（K-means + t-SNE） | `analysis/kmeans_tsne.py` |
| `corr_*.csv / corr_*.png`（物理量相関） | `analysis/corr_ranking_latent_vs_observables.py` |
| `cluster_cond_G_r_*.png`（条件付き G(r)） | `analysis/Cluster_cond_G_r.py` |
| `*_with_peaks.png`（勾配ピーク） | `analysis/plot_selected_latent_vs_T.py` |

---

## 技術スタック

- **シミュレーション**: NumPy, 自作ハイブリッド MCMC（Metropolis + Over-relaxation + Wolff）
- **機械学習**: PyTorch, SimCLR（NT-Xent 損失）, 畳み込み AE / VAE
- **解析**: scikit-learn（K-means, t-SNE), UMAP
- **設定管理**: YAML + dataclass による再現性設計

---

*Physics × Machine Learning — structured representation learning for topological phase transitions.*
