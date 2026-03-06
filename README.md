MRI画像の腫瘍有無（2値分類: 0=腫瘍なし, 1=腫瘍あり）を対象に，ImageNet事前学習済みモデル（`vgg16` / `resnet18` / `efficientnet_b0`）を複数seedで学習・評価・解析するためのコードです．

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

### 1) 学習 + test評価（5 seed）

実行例：python main.py --model_name resnet18 --dataset_dir Dataset --num_seeds 5 --seed 42

### 2) 主要オプション例

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

## 出力（`runs/<experiment>/`）

- `/seed_XX/`（seedごとの個別実験結果）
  - `best.pt` : そのseedで`val accuracy`が最高だったモデル重み
  - `config.json` : そのseed実行時の設定値スナップショット
  - `confusion_matrix.json` : そのseedの混同行列とTP/FP/FN/TN
  - `timing.json` : そのseedの実行時間
  - `misclassified/{TP,TN,FP,FN}/...png` : 分類結果別に保存した症例画像
  - `analyze/`
    - `per_image_features.csv` : 画像ごとの特徴量・信頼度・グループ情報
    - `summary_quality_intensity.csv` : 明るさ/コントラスト等の群別要約
    - `summary_texture.csv` : テクスチャ特徴量の群別要約
    - `summary_confidence.csv` : 信頼度指標の群別要約
    - `summary_all.json` : 上記要約のJSON統合版
- `/`（実験全体=全seed集約結果）
  - `multi_seed_summary.json` : test指標・混同行列・時間の総合サマリ
  - `confusion_matrix_mean.png/.pdf/.json` : seed平均の混同行列可視化/数値
  - `run_timing.json` : 全seed合計およびseed別の時間統計
  - `stats_tests/`
    - `per_seed_mannwhitney.csv` : seed単位のMann-Whitney検定結果
    - `reproducibility_summary.csv` : seedを跨いだ有意性再現性サマリ
    - `stats_summary.json` : 検定結果のJSON統合版

### analyze_case_trends.py

seed単位で画像品質/テクスチャ/信頼度の特徴量を算出して集計します．

実行例：python analyze_case_trends.py --run_dir runs/<exp> --seed 42

### stat_tests.py

`analyze/per_image_features.csv` を使い，群間比較のMann-Whitney検定とFDR補正を実行します．

実行例：python stat_tests.py --run_dir runs/<exp>

### make_case_gallery.py

TP/TN/FP/FNの代表画像ギャラリーを作成します．

実行例：python make_case_gallery.py --run_dir runs/<exp>

主なオプション:

- `--seed`: 単一seed指定（省略時はall seeds集約）
- `--n_per_category`: 各カテゴリ抽出枚数
- `--cols`: 列数
- `--tile_size`: タイルサイズ
- `--sample_seed`: 抽出再現用seed（省略時は毎回ランダム）
- `--min_id_gap`: 連番に偏らないようにするID間隔制約

### check_duplicates.py

train/test間の重複画像（MD5）を確認します．

実行例：python check_duplicates.py

## 備考

- 入力は `Grayscale(num_output_channels=3)` で3チャネル化し，ImageNet正規化を適用します．
- 既定では全層ファインチューニング（freezeなし）です．
- 早期終了（Early Stopping）と `ReduceLROnPlateau` を利用します．
