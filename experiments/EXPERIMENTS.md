# 小物件偵測實驗（YOLO26 fork）

**目標**：小顆粒（Particle，單類別）偵測，純追指標。物件尺寸跨度大——sqrt(w·h) 中位數 ~13px、65% 落在 8–16px，並含真實大顆粒尾巴至 ~400px。

## 資料集約束（已實測）

- **影像**：~1440–1480 px，略非正方，不下採樣
- **物件尺寸** sqrt(w·h)：median **12.8px**｜65% 在 8–16px｜p90=26｜min 5｜**max ~400（真實大顆粒，非標註錯誤）**
- **正/負樣本**：背景圖 ~40%，有物件的每張約 1 box（正樣本稀疏）
- **資料量**：train 1440 / val 360 / test 360；共 1354 boxes；單類別 `Particle`
- **VRAM**：24 GB（RTX 3090 Ti）
- Tiling 僅用於 TTA

## Baseline 超參數

依先驗訂定，不另做 ablation：

```yaml
model: yolo26n.yaml      # n 是 yolo26.yaml 的 scale，非獨立檔案
pretrained: yolo26n.pt
imgsz: 1408              # 影像 ~1450，幾乎不縮（要完全不縮可用 1472）
batch: 2                 # 24GB VRAM 上限
epochs: 200              # 正樣本稀疏，拉長訓練
scale: 0.1               # 預設 0.5 會把小物件砍半（配 mosaic 更小）；改 ±10% 保尺寸
close_mosaic: 20         # 最後 20 epoch 關 mosaic
# 其餘用預設
```

## 實驗分組

按相互依賴關係分三組，避免 confounding：

- **A 組｜獨立 loss/channel 改動**（`E_NWD` `E_DFL` `E_FOCAL`）：各自在 baseline 上單獨 ablate，與 head 配置、backbone 正交。勝出者永久併入下游所有實驗。
- **B 組｜head 架構**（`E0+` `E1` `E2`，互斥擇一）：A 組鎖定後才執行。
- **C 組｜替代 backbone**（`E_DINO`，平行對照）：整顆 backbone 換掉，與主軸平行跑，作保底與比較。

**為何 A 先於 B：**
1. 避免 confounding——用 CIoU baseline 先跑 E2，CIoU 在小物件上梯度消失，可能誤判「P1 架構不行」，但真正贏家其實是 E2+NWD。
2. 省算力——A 組只在最便宜的 E0 上跑一次；先鎖定 loss，免得在每個架構上重複 ablate。
3. A 組三項相容（NWD on box、DFL on dist、Focal on cls）可同時鎖定；B 組互斥只能擇一。

## 執行順序

```
量測 dataset stats ✅（C≈13、背景 40%、大顆粒至 ~400px）
        │
       E0 baseline ──┬── A 組並行 ablate：E_NWD / E_DFL / E_FOCAL
                     │        └─ 收勝出組合 → E0+（新 baseline）
                     │              └─ B 組擇一：E0+ vs E1 vs E2 → final config
                     └── C 組平行：E_DINO（獨立結論）
```

## 評估指標（benchmark）

**碰撞級 AP（IoU≥0.1）= recall + precision 綜合**，不是 mAP50/mAP50-95。物件極小（median ~13px、尾至 5px），mAP50 要求 IoU≥0.5 等於內建「框要貼緊」的要求——對小物件無意義，還把「找到但框鬆」的偵測誤判成漏抓（E0 test 從 IoU 0.5→0.1 回升 ~6 點：0.746→0.805，就是被冤枉掉的）。IoU 本身沒問題，問題在門檻；用 IoU≥0.1 當「有沒有相撞 = 有沒有找到」的判定，一對一指派。工具 `experiments/eval_iou.py`（只覆寫 `iouv`，AP@0.5 已驗證重現官方 mAP50 → AP@0.1 可信）。部署再從 P-R 曲線挑操作點（漏抓較糟就偏 recall）。

## 進度表

| ID | 組 | 名稱 | 依賴 | 採納條件 | 狀態 | AP@0.1 | Recall | Precision |
|----|----|------|------|----------|------|--------|--------|-----------|
| E0 | — | Baseline yolo26n | — | (anchor) | ✅ | 0.950 / 0.805 | 0.919 / 0.758 | 0.926 / 0.929 |
| E_NWD | A | CIoU → NWD | E0 | ΔAP@0.1 ≳ +.01 | ❌ | 0.929 / 0.797 | 0.876 / 0.750 | 0.915 / 0.905 |
| E_DFL | A | reg_max 1 → 4 | E0 | ΔAP@0.1 ≳ +.01 | ❌ | 0.936 / 0.806 | 0.898 / 0.748 | 0.909 / 0.914 |
| E_FOCAL | A | cls BCE → focal | E0 | ΔAP@0.1 ≳ +.01 | ❌ | 0.921 / 0.814 | 0.867 / 0.748 | 0.857 / 0.900 |
| E0+ | — | = E0（A 組無勝出） | — | (= E0) | 🔒 | 0.950 / 0.805 | 0.919 / 0.758 | 0.926 / 0.929 |
| E1 | B | +P2 head (P2/P3/P4/P5) | E0+ | 比 E0+ ΔAP@0.1 ≳ +.01 | ⬜ | | | |
| E2 | B | P1/P2/P3（無 P4/P5） | E0+ | 比 E1 ΔAP@0.1 ≳ +.01 | ⬜ | | | |
| E_DINO | C | DINOv3 ConvNeXt-B backbone | E0 | 對照組，獨立比較 | ⬜ | | | |

狀態圖例：⬜ pending ｜🟡 running ｜✅ done ｜❌ failed ｜🔒 locked-in
指標基準：每格 = **val / test** 的碰撞級（IoU≥0.1、一對一）`eval_iou.py` AP@0.1 / best-F1 Recall / Precision（定義見「評估指標」節）。test 僅 240 box，±.01 內為噪聲。**A 組在 val 上三個都明顯低於 E0（0.92–0.94 vs 0.950，超噪聲），test 上打平——兩個 split 都沒贏過 E0。** E_FOCAL test 名目最高(0.814)卻 val 最差(0.921) → 確認噪聲。

**A 組結論（2026-06-02，碰撞級 AP@0.1，val / test）**：E0 **0.950 / 0.805**｜E_NWD 0.929 / 0.797｜E_DFL 0.936 / 0.806｜E_FOCAL 0.921 / 0.814。**val 上三個都明顯低於 E0（−.014 ~ −.029，超出噪聲）**，test 上打平（±.01 內）；**recall 兩個 split 都 ≤ E0**——沒有一個救回那 ~24% 真漏抓。E_FOCAL test 名目最高卻 val 最差 → 噪聲。**三個全 reject，E0+ = E0。** loss 改動動不了瓶頸 → 瓶頸在偵測能力/解析度 → 轉 **Track B（P2/P1 head）**。工具 `experiments/eval_iou.py`。

## A 組採納規則

| 結果（test AP@0.1 vs E0，需穩、非噪聲） | 行動 |
|------|------|
| ΔAP@0.1 明顯超過噪聲（≳ +.01）且 recall 不降 | 🔒 永久採納，寫入 E0+ default |
| ΔAP@0.1 在噪聲帶內（±.01） | ⏸ 不採納（無有效增益） |
| ΔAP@0.1 < 0 或 recall 明顯下降 | ❌ reject |

永久採納 = B/C 組全部繼承，不再 ablate。

## 實驗細節

### E0 — Baseline yolo26n
所有 A 組 ablation 的 reference，**第一個跑**。

### E_NWD — box loss CIoU → NWD（A 組）
- 小物件（8–16px 主體）GT 與初期 prediction 的 IoU 幾乎為 0，CIoU 梯度消失；NWD 把 box 視為 2D Gaussian、算 Wasserstein-2 距離，零重疊時仍平滑。
- 改 `loss.py:132`（`BboxLoss.forward` 內 `iou = bbox_iou(..., CIoU=True)` 那行）。
- 正規化常數 `C` 取 dataset **中位** sqrt(w·h)：實測 median=12.8 → **C ≈ 13**（恰等於 NWD 原始 AI-TOD 基準的 12.8）。mean=18 被大顆粒尾巴拉高，不採用。
- ⚠️ NWD 增益集中在 8–16px 主體；大顆粒（>64px，IoU 不退化）受益有限。

### E_DFL — reg_max 1 → 4（A 組）
- YOLO26 預設 `reg_max=1`：DFL 被停用（`head.py:120` 走 `nn.Identity`），box head 的 4 個 channel 直接當 ltrb 距離回歸，只靠 CIoU 監督，無分佈式 bins。
- 改 yaml 一行 `reg_max: 1 → 4`，DFL loss 自動啟用（`loss.py:353` `use_dfl = m.reg_max > 1`）。
- reg_max=4 → bins 0–3，最大可表示偏移 `(reg_max-1)×stride`：P2(stride 4)=12px、P3(stride 8)=24px，涵蓋 13px 主體（p90=26px）；大顆粒尾巴超出 DFL 表示範圍，靠 CIoU + 粗 head 處理。
- ⚠️ 改 reg_max 會讓 box head（cv2）channel 對不上 pretrained，那幾層走 partial load 隨機初始化（`intersect_dicts`）；backbone 不受影響、照常載入。

### E_FOCAL — cls BCE → focal（A 組）
- 改 `loss.py:345`（`self.bce` 實例化處），或 loss 計算點 `loss.py:434-437`。
- 初始 α=0.25, γ=2.0（focal 原論文）。
- ⚠️ E0 上 imbalance 未必明顯（P3 cell 數還 hold 得住），收益可能有限；但 E2（P1 head）imbalance 必然嚴重。
- 若 E0 上未達採納門檻，仍須在 E2 階段重驗一次。

### E1 — yolo26-p2（B 組）
- 加 P2 head、保留 P3/P4/P5：`cfg/models/26/yolo26-p2.yaml`（`Detect(P2,P3,P4,P5)`）。
- P2(stride 4) 對 13px 主體給 ~3 cells；保留的 P4/P5 服務大顆粒尾巴（64–400px）。
- **小主體與大尾巴一網打盡**，是尺寸跨度大的場景較穩的選擇。
- 必帶 E0+ 的所有 A 組鎖定改動。對比基準：E0+。

### E2 — yolo26-p1（B 組）
- P1/P2/P3 head、無 P4/P5：`cfg/models/26/yolo26-p1.yaml`（`Detect(P1,P2,P3)`）。
- P1(stride 2) 主要服務最小那 7%（<8px）物件。
- ⚠️ **大顆粒風險**：實測有真實大顆粒至 ~400px（佔 ~2%）。砍掉 P4/P5 等於拿掉它們的天然 scale——P3(stride 8) 是 E2 最粗的 head，400px 物件只能硬塞。E2 形同用「大物件能力」換「最小 7% 的邊際收益」，跨度大的場景未必划算。
- 觀察重點：**P1 anchor 數暴增 → cls imbalance**。症狀：confidence 全偏低 / recall 偏低 / cls loss 不下降。若出現且 E_FOCAL 在 A 組被保留，則在 E2 上重啟 focal 做二輪 ablation。
- 對比基準：E1（多帶的 P1 是否抵得過丟掉 P4/P5）。

### E_DINO — DINOv3 ConvNeXt-Base backbone（C 組）
- backbone 換 DINOv3 ConvNeXt-Base（SSL on ~1.7B images）。
- ConvNeXt stem 為 stride 4，出 stride 4/8/16/32 → P2/P3/P4/P5，無 P1。
- 需寫 wrapper + channel adapter。
- 與最終 final config 直接比較；若顯著勝出，再考慮以 DINO backbone 重跑 A 組。

## 執行前待量測

E0 之前先解決：

- [x] **Dataset sqrt(w·h)**：median 12.8 / mean 18 → **NWD C ≈ 13**
- [x] **正/負樣本比**：背景 40%，每正樣本圖 ~1 box（imbalance 中等，非極端）
- [x] **大顆粒確認**：~400px 尾巴為真實顆粒（非標註錯誤）→ E2 砍 P4/P5 需謹慎
- [~] **標註抖動 audit**：跳過（人力成本高）。改用 E_DFL 結果直接判定，reg_max 高低交給 2/4/6 sweep。僅在 E_DFL 無增益、或 AP@0.75 跨實驗卡天花板時，才反應式抽樣重標診斷
- [ ] Dataset domain（Bosch 小顆粒）→ 影響 E_DINO 收益預期

勝出後再 sweep：

- [ ] DFL reg_max：2 / 4 / 6（若 E_DFL 勝出）
- [ ] focal α / γ 範圍（若 E_FOCAL 勝出）
- [ ] **NWD C：10 / 13 / 18**（±50% 圍繞 median）

## 補充

<!-- 新實驗放這 -->
