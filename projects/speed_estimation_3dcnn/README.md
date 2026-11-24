# 走行速度推定：3D CNN・2D CNN と時系列モデルの比較

[← README に戻る](../../README.md)

白線画像生成ソフト（[lane-synth](../lane_following_dl/synth/README.md)）で作った 10 Hz の白線マスク時系列画像から、最後のフレームの時刻の走行速度を推定します。

## データ

- 1 走行 = ランダムに生成した 1 つのコースの片方の車線を 12〜20 s 走行（1200 走行 = 1200 コース）
- 速度：目標速度（0〜1.0 m/s）を 0.8〜3.0 s 保持し、一定の加速度（走行ごとに 0.3〜1.5 m/s²）で線形に加減速
- 中央破線のある 2 車線コース 80%、床の継ぎ目あり 50%、白線検出 CNN 相当のノイズを付与
- 入力：鳥瞰マスク 140×180（前方 0.30〜1.00 m、5 mm/px）または カメラ視点マスク 192×144
- 走行単位で学習 / 検証 / テストに分割（70 / 15 / 15 %）

## モデル

| 名前 | 時間の扱い |
|---|---|
| `single2d` | 最新の 1 枚だけ（2D CNN） |
| `early2d` | T 枚を入力チャンネルとして重ねる（2D CNN） |
| `early2d_wide` | early2d の幅を広げ、計算量を c3d にそろえたもの |
| `c3d` | 3×3×3 カーネルで時間方向にも畳み込む（3D CNN） |
| `r2p1d` | 3D 畳み込みを空間 1×3×3 と時間 3×1×1 に分解（R(2+1)D） |
| `tsm` | 2D CNN の各層でチャンネルの 1/8 ずつを前後のフレームへずらす（Temporal Shift Module） |
| `cnn_lstm` | フレームごとの 2D CNN の特徴ベクトルを LSTM でつなぐ |
| `convlstm` | 1/8 解像度の特徴マップを位置を保ったまま ConvLSTM でつなぐ |
| `vivit` | フレームごとの空間 Transformer → 時間 Transformer（ViViT，factorized encoder） |

`early2d_wide` 以外はパラメータ数を約 87〜90 万にそろえています。

## 使い方

```bash
pip install torch numpy scipy opencv-python-headless pyyaml pandas matplotlib

# データセット（鳥瞰のみ / カメラ視点も保存）
python scripts/make_dataset.py data/v1 --workers 18
python scripts/make_dataset.py data/v1cam --config config_cam.yaml --workers 18

# 学習（1 モデル）
python scripts/train.py data/v1 --model c3d --T 8 --epochs 20 --out results/c3d_T8_s0
python scripts/train.py data/v1cam --model c3d --T 8 --input cam --out results/c3d_T8_cam_s0

# 実験一式と評価
bash scripts/run_all.sh data/v1 results          # 2D vs 3D（T=8、3 シード）と T=4/16
bash scripts/run_cam.sh data/v1cam results       # 鳥瞰変換前の入力
bash scripts/run_tsweep.sh data/v1 results       # T = 2〜24
bash scripts/run_others.sh data/v1 results       # 時系列モデル（CNN-LSTM, ConvLSTM, R(2+1)D, TSM, ViViT）
bash scripts/run_vivit_lr.sh data/v1 results 3e-4   # ViViT を学習率 0.0003 で
python scripts/evaluate.py results/*_T8_s*/ --out results/report_T8
python scripts/bench_cost.py --out results/report_cost   # 逐次推論・学習の計算量と時間
python scripts/analyze_tsweep.py results --out results/report_tsweep_speed
```

データ（約 2.5 GB）と学習結果は `data/`・`results/` に出力され、リポジトリには含めていません。
