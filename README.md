# Medical-Image-Classification

MRI画像の腫瘍有無（2値分類: 0=腫瘍なし, 1=腫瘍あり）を対象に、
ImageNet事前学習済みモデル（`vgg16` / `resnet18` / `efficientnet_b0`）を
**複数seedで学習・評価・解析**するためのコードです。

## 構成

- 学習・評価本体: `main.py`
- 設定/引数: `configs.py`
- データ読み込み/前処理: `data.py`
- モデル定義: `models.py`
- 学習ループ/評価/混同行列・ケース保存: `engine.py`
- 画像特徴量解析（TP/TN/FP/FN）: `analyze_case_trends.py`
- 統計検定（Mann-Whitney + FDR）: `stat_tests.py`
- ケースギャラリー作成: `make_case_gallery.py`
- train/test重複画像チェック: `check_duplicates.py`
- 共通ユーティリティ: `utils.py`

## 必要環境

- Python 3.10+ 推奨
- 主要ライブラリ:
  - `torch`
  - `torchvision`
  - `numpy`
  - `Pillow`

インストール例:

```bash
pip install torch torchvision numpy pillow
```

## データ配置

`Dataset/` 配下に以下を配置します。

- `Dataset/train.tar.gz`
- `Dataset/val.tar.gz`
- `Dataset/test.tar.gz`

`main.py` 実行時に、未展開であれば自動展開されます。

## クイックスタート

### 1) 学習 + test評価（5 seed）

```bash
python main.py --model_name resnet18 --dataset_dir Dataset --num_seeds 5 --seed 42
```

### 2) 主要オプション例

```bash
python main.py \
  --model_name efficientnet_b0 \
  --dataset_dir Dataset \
  --img_size 224 \
  --batch_size 64 \
  --epochs 20 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --use_augmentation \
  --save_test_preds \
  --save_case_images
```

### 3) LR/WDグリッド探索

```bash
python main.py --model_name resnet18 --tune_lr_wd
```

## 出力（`runs/<experiment>/`）

代表的な出力:

- `seed_XX/`
  - `best.pt`（best val accuracy時の重み）
  - `config.json`
  - `confusion_matrix.json`
  - `timing.json`
  - `misclassified/{TP,TN,FP,FN}/...png`
  - `analyze/`
    - `per_image_features.csv`
    - `summary_quality_intensity.csv`
    - `summary_texture.csv`
    - `summary_confidence.csv`
    - `summary_all.json`
- 実験ルート
  - `multi_seed_summary.json`
  - `confusion_matrix_mean.png/.pdf/.json`
  - `run_timing.json`
  - `stats_tests/`
    - `per_seed_mannwhitney.csv`
    - `reproducibility_summary.csv`
    - `stats_summary.json`

## 補助スクリプト

### analyze_case_trends.py

seed単位で、画像品質/テクスチャ/信頼度の特徴量を算出して集計します。

```bash
python analyze_case_trends.py --run_dir runs/<exp> --seed 42
```

### stat_tests.py

`analyze/per_image_features.csv` を使い、群間比較のMann-Whitney検定とFDR補正を実行します。

```bash
python stat_tests.py --run_dir runs/<exp>
```

比較指定（例: TP vs FNのみ）:

```bash
python stat_tests.py --run_dir runs/<exp> --comparisons TP:FN --out_dir runs/<exp>/stats_tests_tp_vs_fn
```

### make_case_gallery.py

TP/TN/FP/FNの代表画像ギャラリーを作成します。

```bash
python make_case_gallery.py --run_dir runs/<exp>
```

主なオプション:

- `--seed`: 単一seed指定（省略時はall seeds集約）
- `--n_per_category`: 各カテゴリ抽出枚数
- `--cols`: 列数
- `--tile_size`: タイルサイズ
- `--sample_seed`: 抽出再現用seed（省略時は毎回ランダム）
- `--min_id_gap`: 連番に偏らないようにするID間隔制約

### check_duplicates.py

train/test間の重複画像（MD5）を確認します。

```bash
python check_duplicates.py
```

## モデルの最終層置換

- `vgg16`: `classifier[-1]` を2クラスへ置換
- `resnet18`: `fc` を2クラスへ置換
- `efficientnet_b0`: `classifier[-1]` を2クラスへ置換

## 備考

- 入力は `Grayscale(num_output_channels=3)` で3チャネル化し、ImageNet正規化を適用します。
- 既定では全層ファインチューニング（freezeなし）です。
- 早期終了（Early Stopping）と `ReduceLROnPlateau` を利用します。
