# MentorPi Research: 白線マスク時系列画像による走行速度推定と小型移動ロボット

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24466081.svg)](https://doi.org/10.5281/zenodo.24466081)

小型移動ロボット（Hiwonder MentorPi A1 相当、アッカーマン操舵）を対象にした個人研究のコードです。

著者：栗原 旺愛（Akichika Kurihara, Independent Researcher）

- **白線画像生成ソフト（lane-synth）**：コースを走ったときの車載カメラの白線マスク画像と鳥瞰画像を合成
- **走行速度推定**：10 Hz の白線マスク時系列画像から現在の走行速度を推定し、3D CNN・2D CNN と時系列モデル（CNN-LSTM、ConvLSTM、R(2+1)D、TSM、ViViT）を比較
- **ROS 2 パッケージ**：実機の起動・手動操縦・cmd_vel からモータ／ステア指令への変換

## 研究の概要（説明スライド）

![表紙](docs/slides/slide-01.png)

<table>
<tr><td width="50%"><a href="docs/slides/slide-05.png"><img src="docs/slides/slide-05.png" alt="本研究の目的"></a><br><sub>本研究の目的</sub></td><td width="50%"><a href="docs/slides/slide-10.png"><img src="docs/slides/slide-10.png" alt="走行したコース"></a><br><sub>走行したコース</sub></td></tr>
<tr><td width="50%"><a href="docs/slides/slide-14.png"><img src="docs/slides/slide-14.png" alt="入力画像の例"></a><br><sub>入力画像の例</sub></td><td width="50%"><a href="docs/slides/slide-19.png"><img src="docs/slides/slide-19.png" alt="3D CNNの構造"></a><br><sub>3D CNNの構造</sub></td></tr>
<tr><td width="50%"><a href="docs/slides/slide-22.png"><img src="docs/slides/slide-22.png" alt="テストデータに対する推定誤差"></a><br><sub>テストデータに対する推定誤差</sub></td><td width="50%"><a href="docs/slides/slide-26.png"><img src="docs/slides/slide-26.png" alt="鳥瞰変換の前と後の推定誤差"></a><br><sub>鳥瞰変換の前と後の推定誤差</sub></td></tr>
<tr><td width="50%"><a href="docs/slides/slide-32.png"><img src="docs/slides/slide-32.png" alt="ConvLSTMの構造"></a><br><sub>ConvLSTMの構造</sub></td><td width="50%"><a href="docs/slides/slide-36.png"><img src="docs/slides/slide-36.png" alt="逐次推論の計算量"></a><br><sub>逐次推論の計算量</sub></td></tr>
<tr><td width="50%"><a href="docs/slides/slide-39.png"><img src="docs/slides/slide-39.png" alt="計算コストと誤差の関係"></a><br><sub>計算コストと誤差の関係</sub></td><td width="50%"><a href="docs/slides/slide-43.png"><img src="docs/slides/slide-43.png" alt="まとめ"></a><br><sub>まとめ</sub></td></tr>
</table>

<details>
<summary>すべてのスライドを見る（35 枚）</summary>

**2. 目次**

![目次](docs/slides/slide-02.png)

**4. 研究背景**

![研究背景](docs/slides/slide-04.png)

**5. 本研究の目的**

![本研究の目的](docs/slides/slide-05.png)

**7. 速度推定の流れ**

![速度推定の流れ](docs/slides/slide-07.png)

**9. 学習データの作成方法**

![学習データの作成方法](docs/slides/slide-09.png)

**10. 走行したコース**

![走行したコース](docs/slides/slide-10.png)

**11. コースの種類と内訳**

![コースの種類と内訳](docs/slides/slide-11.png)

**12. 走行の内容（速度と加減速）**

![走行の内容（速度と加減速）](docs/slides/slide-12.png)

**13. データ量と分割**

![データ量と分割](docs/slides/slide-13.png)

**14. 入力画像の例**

![入力画像の例](docs/slides/slide-14.png)

**16. 比較したNNの構造**

![比較したNNの構造](docs/slides/slide-16.png)

**17. CNNに共通の構造**

![CNNに共通の構造](docs/slides/slide-17.png)

**18. 2D CNNの構造**

![2D CNNの構造](docs/slides/slide-18.png)

**19. 3D CNNの構造**

![3D CNNの構造](docs/slides/slide-19.png)

**20. 学習の設定**

![学習の設定](docs/slides/slide-20.png)

**22. テストデータに対する推定誤差**

![テストデータに対する推定誤差](docs/slides/slide-22.png)

**23. 推定値と真の速度の関係**

![推定値と真の速度の関係](docs/slides/slide-23.png)

**24. 走行条件ごとの推定誤差**

![走行条件ごとの推定誤差](docs/slides/slide-24.png)

**25. 入力画像枚数Tと破線の折り返し**

![入力画像枚数Tと破線の折り返し](docs/slides/slide-25.png)

**26. 鳥瞰変換の前と後の推定誤差**

![鳥瞰変換の前と後の推定誤差](docs/slides/slide-26.png)

**28. 比較した時系列モデル**

![比較した時系列モデル](docs/slides/slide-28.png)

**29. R(2+1)Dの構造**

![R(2+1)Dの構造](docs/slides/slide-29.png)

**30. TSMの構造**

![TSMの構造](docs/slides/slide-30.png)

**31. CNN-LSTMの構造**

![CNN-LSTMの構造](docs/slides/slide-31.png)

**32. ConvLSTMの構造**

![ConvLSTMの構造](docs/slides/slide-32.png)

**33. ViViTの構造**

![ViViTの構造](docs/slides/slide-33.png)

**34. 時系列モデルのテスト誤差**

![時系列モデルのテスト誤差](docs/slides/slide-34.png)

**35. 時系列モデルの条件ごとの誤差**

![時系列モデルの条件ごとの誤差](docs/slides/slide-35.png)

**36. 逐次推論の計算量**

![逐次推論の計算量](docs/slides/slide-36.png)

**37. 学習の計算量**

![学習の計算量](docs/slides/slide-37.png)

**38. 学習過程**

![学習過程](docs/slides/slide-38.png)

**39. 計算コストと誤差の関係**

![計算コストと誤差の関係](docs/slides/slide-39.png)

**41. 考察：3D CNNと鳥瞰変換**

![考察：3D CNNと鳥瞰変換](docs/slides/slide-41.png)

**42. 考察：時系列モデルの比較**

![考察：時系列モデルの比較](docs/slides/slide-42.png)

**43. まとめ**

![まとめ](docs/slides/slide-43.png)

</details>
PDF：[docs/research_v1.1.0.pdf](docs/research_v1.1.0.pdf)（v1.0.0 版：[docs/research_v1.0.0.pdf](docs/research_v1.0.0.pdf)）

## 構成

| ディレクトリ | 内容 |
|---|---|
| [projects/lane_following_dl/synth/](projects/lane_following_dl/synth/README.md) | 白線画像生成ソフト（コース生成、カメラ投影、検出ノイズ、鳥瞰変換、GUI） |
| [projects/speed_estimation_3dcnn/](projects/speed_estimation_3dcnn/README.md) | 速度推定の実験コード（データセット生成、学習、条件別評価） |
| [ros2_ws/src/](ros2_ws/src/) | ROS 2 Jazzy パッケージ（a1_bringup / a1_control / a1_vehicle） |

## 主な結果（シミュレーション、T = 8 フレーム、テスト MAE）

### 3D CNN と 2D CNN（鳥瞰変換の前と後）

| モデル | 鳥瞰画像入力 | カメラ視点入力 |
|---|---|---|
| 2D CNN（1枚入力） | 0.237 m/s | — |
| 2D CNN（時間をチャンネル方向として扱う） | 0.100 m/s | 0.084 m/s |
| 3D CNN | **0.088 m/s** | **0.079 m/s** |

1 枚の画像からは速度を推定できず、時系列画像が必要です。パラメータ数・計算量をそろえても、時間方向にも畳み込む 3D CNN の誤差が 2D CNN より小さくなりました。

### 時系列モデルの比較（v1.1.0 で追加、鳥瞰画像入力）

パラメータ数は約 87〜90 万にそろえています。逐次推論は、10 Hz で新しい画像が来るたびに最新 8 枚から推定する場合の 1 枚あたりのコストです（過去フレームの特徴を再利用できるモデルは再利用）。

| モデル | テスト MAE [m/s] | 計算量（8 枚の窓）[MMAC] | 逐次推論 [MMAC/枚] | 逐次推論 CPU [ms] | 学習時間 [分] |
|---|---|---|---|---|---|
| 2D CNN（時間→チャンネル） | 0.100 | 257 | 257 | 1.5 | 2 |
| 3D CNN | 0.088 | 1,193 | 1,193 | 24.9 | 18 |
| R(2+1)D | 0.084 | 1,262 | 1,262 | 11.6 | 42 |
| TSM | 0.087 | 1,956 | 1,956 | 8.4 | 27 |
| CNN-LSTM | 0.084 | 1,225 | **154** | **1.5** | 16 |
| ConvLSTM | **0.075** | 2,279 | 950 | 4.9 | 19 |
| ViViT（学習率 0.0003） | 0.176 | 1,600 | 202 | 1.8 | 29 |

CPU は Core i5-13600KF 4 スレッド（Raspberry Pi 5 を想定した参考値）、学習は RTX 4080 で 20 エポック（3 シードの平均）。ConvLSTM の誤差が最も小さく、CNN-LSTM は逐次推論の計算量が 3D CNN の約 1/8 で誤差も小さくなりました。詳細はスライドを参照してください。

## ROS 2 パッケージについて

`ros2_ws/src/` のパッケージは、ロボットの制御ボード用ドライバ `ros_robot_controller`（メッセージ定義 `ros_robot_controller_msgs`）と深度カメラのドライバ `ascamera` に依存します。これらはメーカー提供のソフトウェアのため、このリポジトリには含めていません。メーカーから入手して同じワークスペースに置いてください。

## 動作環境

- 学習：Ubuntu 24.04、Python 3.12、PyTorch 2.x（CUDA）、NVIDIA GPU（RTX 4080 で確認）
- ロボット：Raspberry Pi 5、Ubuntu 24.04、ROS 2 Jazzy

## 採番

Kurihara, A. (2026). *MentorPi Research: speed estimation from time-series lane-mask images (3D-CNN, 2D-CNN and sequence models) and lane-synth* (v1.1.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.24466081

[CITATION.cff](CITATION.cff) も参照してください（GitHub の「Cite this repository」から BibTeX を取得できます）。

## ライセンス

[Apache License 2.0](LICENSE)。説明スライドも同じライセンスで公開します。

---

# English summary

Code of a personal research project on a small Ackermann-steering mobile robot (MentorPi A1 class).

- **lane-synth**: synthesizes on-board lane-mask and bird's-eye (BEV) images of a robot driving on generated courses
- **Speed estimation**: estimates the current speed from 10 Hz lane-mask sequences and compares a 3D-CNN with 2D-CNNs
  (single frame / frames stacked as channels). With matched parameters and compute, the 3D-CNN gives the lowest error
  (test MAE 0.088 m/s on BEV input, 0.079 m/s on camera-view input); a single frame cannot tell the speed (0.237 m/s).
  v1.1.0 adds other sequence models with the same parameter budget: ConvLSTM gives the lowest error (0.075 m/s), and
  CNN-LSTM matches R(2+1)D (0.084 m/s) at 1/8 of the 3D-CNN compute per new frame in sequential inference.
- **ROS 2 packages** for bring-up, teleoperation and cmd_vel → motor/steering conversion. They depend on the vendor
  drivers `ros_robot_controller` and `ascamera`, which are not included.

Licensed under the Apache License 2.0.
