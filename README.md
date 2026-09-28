# Smart Posture Monitor - Há»‡ Thá»‘ng GiÃ¡m SÃ¡t TÆ° Tháº¿ Ngá»“i ThÃ´ng Minh Thá»i Gian Thá»±c

> **Dá»± Ã¡n Nháº­n diá»‡n & Cáº£nh bÃ¡o TÆ° tháº¿ Ngá»“i lÃ m viá»‡c/há»c táº­p qua Webcam mÃ¡y tÃ­nh báº±ng Thá»‹ giÃ¡c mÃ¡y tÃ­nh (YOLO Pose) vÃ  MÃ¡y há»c (Machine Learning)**  
> *PhiÃªn báº£n hiá»‡n táº¡i: V02 Benchmark & Äang triá»ƒn khai V03 (Depth Proxy & Personal Calibration)*

---

## Má»¥c lá»¥c

1. [Bá»‘i Cáº£nh & BÃ i ToÃ¡n Thá»±c Táº¿](#1-bá»‘i-cáº£nh--bÃ i-toÃ¡n-thá»±c-táº¿)
2. [CÃ¡c TÆ° Tháº¿ Má»¥c TiÃªu & Ã NghÄ©a Y Khoa](#2-cÃ¡c-tÆ°-tháº¿-má»¥c-tiÃªu--Ã½-nghÄ©a-y-khoa)
3. [Thiáº¿t Láº­p Thá»±c Táº¿ & ThÃ¡ch Thá»©c Ká»¹ Thuáº­t](#3-thiáº¿t-láº­p-thá»±c-táº¿--thÃ¡ch-thá»©c-ká»¹-thuáº­t)
4. [Kiáº¿n TrÃºc Há»‡ Thá»‘ng (End-to-End Pipeline)](#4-kiáº¿n-trÃºc-há»‡-thá»‘ng-end-to-end-pipeline)
5. [Cáº¥u TrÃºc ThÆ° Má»¥c Dá»± Ãn](#5-cáº¥u-trÃºc-thÆ°-má»¥c-dá»±-Ã¡n)
6. [TrÃ­ch Xuáº¥t Äáº·c TrÆ°ng HÃ¬nh Há»c (Feature Engineering)](#6-trÃ­ch-xuáº¥t-Ä‘áº·c-trÆ°ng-hÃ¬nh-há»c-feature-engineering)
7. [Táº­p Dá»¯ Liá»‡u & PhÃ¢n TÃ­ch KhÃ¡m PhÃ¡ (Dataset & EDA)](#7-táº­p-dá»¯-liá»‡u--phÃ¢n-tÃ­ch-khÃ¡m-phÃ¡-dataset--eda)
8. [Quy TrÃ¬nh Tiá»n Xá»­ LÃ½ Chá»‘ng RÃ² Rá»‰ Dá»¯ Liá»‡u](#8-quy-trÃ¬nh-tiá»n-xá»­-lÃ½-chá»‘ng-rÃ²-rá»‰-dá»¯-liá»‡u)
9. [Huáº¥n Luyá»‡n & Tuyá»ƒn Chá»n MÃ´ HÃ¬nh (Model Selection)](#9-huáº¥n-luyá»‡n--tuyá»ƒn-chá»n-mÃ´-hÃ¬nh-model-selection)
10. [Káº¿t Quáº£ ÄÃ¡nh GiÃ¡ TrÃªn Táº­p Kiá»ƒm Thá»­ Äá»™c Láº­p (Held-out Evaluation)](#10-káº¿t-quáº£-Ä‘Ã¡nh-giÃ¡-trÃªn-táº­p-kiá»ƒm-thá»­-Ä‘á»™c-láº­p-held-out-evaluation)
11. [PhÃ¢n TÃ­ch Lá»—i Thá»±c Táº¿ & Äá»™ng Lá»±c NÃ¢ng Cáº¥p V03](#11-phÃ¢n-tÃ­ch-lá»—i-thá»±c-táº¿--Ä‘á»™ng-lá»±c-nÃ¢ng-cáº¥p-v03)
12. [Äá»™t PhÃ¡ Ká»¹ Thuáº­t á»ž V03: Depth Proxy & Personal Baseline Calibration](#12-Ä‘á»™t-phÃ¡-ká»¹-thuáº­t-á»Ÿ-v03-depth-proxy--personal-baseline-calibration)
13. [Há»‡ Thá»‘ng Kiá»ƒm Thá»­ Tá»± Äá»™ng (Testing & QA)](#13-há»‡-thá»‘ng-kiá»ƒm-thá»­-tá»±-Ä‘á»™ng-testing--qa)
14. [HÆ°á»›ng Dáº«n CÃ i Äáº·t & Cháº¡y Há»‡ Thá»‘ng](#14-hÆ°á»›ng-dáº«n-cÃ i-Ä‘áº·t--cháº¡y-há»‡-thá»‘ng)
15. [Lá»™ TrÃ¬nh PhÃ¡t Triá»ƒn Sáº£n Pháº©m (Roadmap)](#15-lá»™-trÃ¬nh-phÃ¡t-triá»ƒn-sáº£n-pháº©m-roadmap)
16. [TÃ i Liá»‡u Ká»¹ Thuáº­t Äi KÃ¨m](#16-tÃ i-liá»‡u-ká»¹-thuáº­t-Ä‘i-kÃ¨m)

---

## 1. Bá»‘i Cáº£nh & BÃ i ToÃ¡n Thá»±c Táº¿

### 1.1. Äáº·t váº¥n Ä‘á»
ThÃ³i quen ngá»“i sai tÆ° tháº¿ trong thá»i gian dÃ i lÃ  nguyÃªn nhÃ¢n hÃ ng Ä‘áº§u dáº«n Ä‘áº¿n cÃ¡c bá»‡nh lÃ½ há»c Ä‘Æ°á»ng vÃ  vÄƒn phÃ²ng phá»• biáº¿n:
- ThoÃ¡i hÃ³a Ä‘á»‘t sá»‘ng cá»•, thoÃ¡t vá»‹ Ä‘Ä©a Ä‘á»‡m lÆ°ng, Ä‘au má»i vai gÃ¡y kinh niÃªn.
- Táº­t gÃ¹ lÆ°ng vÃ  há»™i chá»©ng "cá»• rÃ¹a" (*Forward Head Posture / Text Neck*).
- Váº¹o cá»™t sá»‘ng má»™t bÃªn do thÃ³i quen tá»³ tay hoáº·c nghiÃªng ngÆ°á»i khi dÃ¹ng chuá»™t/bÃ n phÃ­m.
- Giáº£m dung tÃ­ch phá»•i vÃ  giáº£m lÆ°u thÃ´ng mÃ¡u lÃªn nÃ£o, gÃ¢y má»‡t má»i vÃ  suy giáº£m nÄƒng suáº¥t lÃ m viá»‡c.

### 1.2. Má»¥c tiÃªu dá»± Ã¡n
**Smart Posture Monitor** lÃ  giáº£i phÃ¡p thá»‹ giÃ¡c mÃ¡y tÃ­nh káº¿t há»£p mÃ¡y há»c gá»n nháº¹, váº­n hÃ nh cá»¥c bá»™ (on-device) trÃªn mÃ¡y tÃ­nh cÃ¡ nhÃ¢n thÃ´ng qua webcam thÃ´ng thÆ°á»ng:
- **GiÃ¡m sÃ¡t liÃªn tá»¥c vÃ  tá»± Ä‘á»™ng** tÆ° tháº¿ ngÆ°á»i dÃ¹ng trong suá»‘t phiÃªn há»c táº­p/lÃ m viá»‡c.
- **PhÃ¢n loáº¡i chÃ­nh xÃ¡c 4 tráº¡ng thÃ¡i tÆ° tháº¿** vá»›i Ä‘á»™ trá»… tháº¥p vÃ  tÃ i nguyÃªn pháº§n cá»©ng tá»‘i thiá»ƒu (cháº¡y mÆ°á»£t mÃ  trÃªn CPU mÃ  khÃ´ng báº¯t buá»™c cÃ³ GPU rá»i).
- **Cáº£nh bÃ¡o thÃ´ng minh**: Nháº¯c nhá»Ÿ ngÆ°á»i dÃ¹ng khi phÃ¡t hiá»‡n tÆ° tháº¿ sai duy trÃ¬ liÃªn tá»¥c quÃ¡ ngÆ°á»¡ng thá»i gian quy Ä‘á»‹nh (trÃ¡nh bÃ¡o Ä‘á»™ng giáº£ khi ngÆ°á»i dÃ¹ng chá»‰ cá»­ Ä‘á»™ng táº¡m thá»i).
- **Thá»‘ng kÃª phiÃªn lÃ m viá»‡c**: Theo dÃµi tá»· lá»‡ ngá»“i chuáº©n, thá»i lÆ°á»£ng gÃ¹ lÆ°ng, cháº¥m Ä‘iá»ƒm tÆ° tháº¿ (*Posture Score*) theo ngÃ y/tuáº§n.

---

## 2. CÃ¡c TÆ° Tháº¿ Má»¥c TiÃªu & Ã NghÄ©a Y Khoa

Há»‡ thá»‘ng táº­p trung phÃ¢n loáº¡i 4 tráº¡ng thÃ¡i tÆ° tháº¿ then chá»‘t:

| NhÃ£n (Label) | Class ID | Tráº¡ng ThÃ¡i CÆ¡ Thá»ƒ | Ã NghÄ©a Y Khoa & Äá»i Sá»‘ng |
|---|:---:|---|---|
| `correct` | **0** | **Ngá»“i chuáº©n / Tháº³ng lÆ°ng**:<br/>Trá»¥c cá»™t sá»‘ng á»Ÿ tráº¡ng thÃ¡i tá»± nhiÃªn (*neutral spine*), Ä‘áº§u vÃ  hai vai cÃ¢n báº±ng, máº¯t nhÃ¬n ngang táº§m mÃ n hÃ¬nh. | PhÃ¢n bá»• táº£i trá»ng Ä‘á»“ng Ä‘á»u lÃªn cÃ¡c Ä‘Ä©a Ä‘á»‡m, giáº£m tá»‘i Ä‘a Ã¡p lá»±c cÆ¡ báº¯p vÃ¹ng cá»• vÃ  tháº¯t lÆ°ng. |
| `forward_slouch` | **1** | **CÃºi gÃ¹ ngÆ°á»i vá» phÃ­a trÆ°á»›c**:<br/>LÆ°ng trÃªn uá»‘n cong, Ä‘áº§u cÃºi tháº¥p vÃ  vÆ°Æ¡n vá» phÃ­a trÆ°á»›c mÃ n hÃ¬nh (táº­t "cá»• rÃ¹a"). | Trá»ng lá»±c tÃ¡c Ä‘á»™ng lÃªn Ä‘á»‘t sá»‘ng cá»• tÄƒng gáº¥p 3â€“5 láº§n (tá»« 5kg lÃªn Ä‘áº¿n 20â€“27kg), gÃ¢y co tháº¯t cÆ¡ cá»• vÃ  Ä‘au Ä‘áº§u. |
| `lean_left` | **2** | **NghiÃªng ngÆ°á»i / váº¹o Ä‘áº§u sang trÃ¡i**:<br/>Trá»¥c vai vÃ  Ä‘áº§u nghiÃªng lá»‡ch sang bÃªn trÃ¡i, tá»³ nÃ©n má»™t bÃªn hÃ´ng/tay. | GÃ¢y cÄƒng cÆ¡ báº¥t Ä‘á»‘i xá»©ng, dáº«n Ä‘áº¿n lá»‡ch cÆ¡ hoÃ nh vÃ  nguy cÆ¡ cong váº¹o cá»™t sá»‘ng ngá»±c (*scoliosis*). |
| `lean_right` | **3** | **NghiÃªng ngÆ°á»i / váº¹o Ä‘áº§u sang pháº£i**:<br/>Trá»¥c vai vÃ  Ä‘áº§u nghiÃªng lá»‡ch sang bÃªn pháº£i (nghiÃªng ra xa gÃ³c nhÃ¬n camera). | TÃ¡c háº¡i tÆ°Æ¡ng tá»± nghiÃªng trÃ¡i; thÆ°á»ng xuáº¥t hiá»‡n khi chá»‘ng cáº±m hoáº·c nghiÃªng ngÆ°á»i dÃ¹ng chuá»™t mÃ¡y tÃ­nh. |

---

## 3. Thiáº¿t Láº­p Thá»±c Táº¿ & ThÃ¡ch Thá»©c Ká»¹ Thuáº­t

KhÃ¡c vá»›i cÃ¡c nghiÃªn cá»©u lÃ½ thuyáº¿t chá»¥p áº£nh chÃ­nh diá»‡n trong phÃ²ng thÃ­ nghiá»‡m, bÃ i toÃ¡n thá»±c táº¿ Ä‘áº·t ra cÃ¡c rÃ ng buá»™c váº­t lÃ½ vÃ  hÃ¬nh há»c ráº¥t Ä‘áº·c thÃ¹:

### 3.1. Thiáº¿t láº­p váº­t lÃ½ (Physical Setup)
1. **GÃ³c Ä‘áº·t webcam: Cháº¿ch khoáº£ng 45Â° bÃªn trÃ¡i ngÆ°á»i dÃ¹ng**  
   - Trong khÃ´ng gian lÃ m viá»‡c thá»±c táº¿ vá»›i mÃ¡y tÃ­nh/mÃ n hÃ¬nh rá»i, webcam khÃ³ cÃ³ thá»ƒ Ä‘áº·t trá»±c diá»‡n 0Â° mÃ  thÆ°á»ng Ä‘Æ°á»£c gáº¯n á»Ÿ cáº¡nh mÃ n hÃ¬nh hoáº·c gÃ³c bÃ n lÃ m viá»‡c bÃªn trÃ¡i.  
   - GÃ³c 45Â° cho phÃ©p quan sÃ¡t Ä‘á»“ng thá»i cáº£ Ä‘á»™ nghiÃªng thÃ¢n ngÆ°á»i (chiá»u ngang) vÃ  Ä‘á»™ gÃ¹ lÆ°ng (chiá»u sÃ¢u).
2. **Che khuáº¥t ná»­a thÃ¢n dÆ°á»›i (Desk Occlusion)**  
   - BÃ n lÃ m viá»‡c luÃ´n che khuáº¥t pháº§n hÃ´ng vÃ  hai chÃ¢n.  
   - CÃ¡c mÃ´ hÃ¬nh Æ°á»›c lÆ°á»£ng toÃ n thÃ¢n (*Full-body Pose*) khi gáº·p bÃ n sáº½ bá»‹ áº£o giÃ¡c (*hallucination*) hoáº·c keypoints cÃ³ Ä‘á»™ tin cáº­y ráº¥t tháº¥p.  
   - Do Ä‘Ã³, há»‡ thá»‘ng chá»‰ sá»­ dá»¥ng **pháº§n thÃ¢n trÃªn (Upper-body)** vá»›i **6 keypoints then chá»‘t**:
     - `0 - nose` (mÅ©i)
     - `1 - left_eye` (máº¯t trÃ¡i)
     - `2 - right_eye` (máº¯t pháº£i)
     - `3 - left_ear` (tai trÃ¡i)
     - `5 - left_shoulder` (vai trÃ¡i)
     - `6 - right_shoulder` (vai pháº£i)
3. **Loáº¡i bá» hoÃ n toÃ n `right_ear` (tai pháº£i - index 4)**  
   - Do camera nhÃ¬n tá»« gÃ³c cháº¿ch 45Â° bÃªn trÃ¡i, tai pháº£i háº§u nhÆ° luÃ´n bá»‹ pháº§n Ä‘áº§u che khuáº¥t hoáº·c dao Ä‘á»™ng Ä‘á»™ tin cáº­y ráº¥t máº¡nh. Viá»‡c loáº¡i bá» keypoint nÃ y giÃºp trÃ¡nh Ä‘Æ°a nhiá»…u vÃ o mÃ´ hÃ¬nh.

```text
                  MÃ n hÃ¬nh chÃ­nh
               â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
               â”‚                  â”‚
Webcam (45Â°)   â”‚                  â”‚
    ðŸ“· â”€â”€â”€â”€â”€â”€â”€â”â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
     \  45Â°   â”‚
      \       â”‚
       \      â”‚
        â–¼     â”‚
      [NgÆ°á»i ngá»“i lÃ m viá»‡c]
      (BÃ n lÃ m viá»‡c che khuáº¥t tá»« hÃ´ng trá»Ÿ xuá»‘ng)
```

### 3.2. Hai thÃ¡ch thá»©c ká»¹ thuáº­t cá»‘t lÃµi

#### ThÃ¡ch thá»©c 1: PhÃ©p chiáº¿u 2D lÃ m suy biáº¿n thÃ´ng tin chiá»u sÃ¢u (2D Projection Ambiguity)
Khi camera Ä‘áº·t á»Ÿ gÃ³c 45Â° bÃªn trÃ¡i:
- **`forward_slouch` (gÃ¹ lÆ°ng)**: Äáº§u dá»‹ch chuyá»ƒn **xuá»‘ng dÆ°á»›i vÃ  tiáº¿n ra trÆ°á»›c** (theo trá»¥c Z khÃ´ng gian, hÆ°á»›ng láº¡i gáº§n camera).
- **`lean_right` (nghiÃªng pháº£i)**: Äáº§u dá»‹ch chuyá»ƒn **sang pháº£i vÃ  lÃ¹i ra xa** camera.
- **Hiá»‡n tÆ°á»£ng suy biáº¿n hÃ¬nh há»c**: TrÃªn máº·t pháº³ng chiáº¿u 2D cá»§a áº£nh, cáº£ hai hÃ nh Ä‘á»™ng trÃªn Ä‘á»u lÃ m **Ä‘áº§u háº¡ tháº¥p so vá»›i hai vai** vÃ  **khoáº£ng cÃ¡ch Ä‘áº§u-vai bá»‹ co ngáº¯n láº¡i**.
- **Háº­u quáº£ á»Ÿ V02**: Trong 707 máº«u kiá»ƒm thá»­ held-out, cÃ³ tá»›i **90 máº«u bá»‹ nháº§m láº«n giá»¯a `forward_slouch` vÃ  `lean_right`** (60 máº«u nghiÃªng pháº£i bá»‹ Ä‘oÃ¡n thÃ nh gÃ¹ lÆ°ng, 30 máº«u gÃ¹ lÆ°ng thÃ nh nghiÃªng pháº£i).

#### ThÃ¡ch thá»©c 2: Biáº¿n thiÃªn nhÃ¢n tráº¯c há»c giá»¯a cÃ¡c cÃ¡ nhÃ¢n (Subject-to-Subject Variation)
Má»—i cÃ¡ nhÃ¢n cÃ³ hÃ¬nh thá»ƒ khÃ¡c nhau (cá»• dÃ i/ngáº¯n, vai rá»™ng/háº¹p, tá»· lá»‡ Ä‘áº§u/vai, thÃ³i quen ngá»“i tá»± nhiÃªn khÃ¡c nhau). DÃ¹ Ä‘Ã£ chuáº©n hÃ³a theo chiá»u rá»™ng vai (`shoulder_width`), cÃ¡c háº±ng sá»‘ nhÃ¢n tráº¯c há»c nÃ y váº«n bÃ¡m theo feature vector tuyá»‡t Ä‘á»‘i, khiáº¿n mÃ´ hÃ¬nh khÃ³ khÃ¡i quÃ¡t hÃ³a trÃªn ngÆ°á»i hoÃ n toÃ n má»›i (F1-score dao Ä‘á»™ng tá»« 66.7% Ä‘áº¿n 85.1% tÃ¹y ngÆ°á»i).

---

## 4. Kiáº¿n TrÃºc Há»‡ Thá»‘ng (End-to-End Pipeline)

Há»‡ thá»‘ng Ä‘Æ°á»£c thiáº¿t káº¿ vá»›i hai luá»“ng váº­n hÃ nh Ä‘á»™c láº­p: **Luá»“ng huáº¥n luyá»‡n ngoáº¡i tuyáº¿n** (*Offline Training & Validation*) vÃ  **Luá»“ng suy diá»…n thá»i gian thá»±c** (*Online Realtime Inference*).

```mermaid
flowchart TD
    subgraph Offline_Pipeline["1. Luá»“ng Huáº¥n Luyá»‡n & ÄÃ¡nh GiÃ¡ Ngoáº¡i Tuyáº¿n (Offline Pipeline)"]
        Raw["Raw Images (data/raw/)<br/>4,341 áº£nh / 14 Ä‘á»‘i tÆ°á»£ng / gÃ³c 45Â°"] --> Detector["PoseDetector (YOLO Pose)<br/>Lá»c person confidence â‰¥ 0.5<br/>Chá»n chá»§ thá»ƒ theo tÃ¢m áº£nh"]
        Detector --> Kpts["6 Keypoints ThÃ¢n TrÃªn<br/>nose, eyes, left_ear, shoulders"]
        Kpts --> Extractor["FeatureExtractor<br/>Chuáº©n hÃ³a Body Frame & shoulder_width<br/>32 Engineered Features (29 V02 + 3 Depth Proxies)"]
        Extractor --> Builder["DatasetBuilder<br/>Xuáº¥t features.csv (4,014 máº«u há»£p lá»‡)<br/>Xuáº¥t rejected_images.csv (327 máº«u loáº¡i)"]
        Builder --> Prep["src/preprocessing.py<br/>XÃ¡c thá»±c schema & toÃ n váº¹n dá»¯ liá»‡u<br/>PhÃ¢n chia Group Holdout theo person_id"]
        Prep --> TrainExp["Huáº¥n luyá»‡n & ÄÃ¡nh giÃ¡ chÃ©o<br/>StratifiedGroupKFold (5-folds) trÃªn 11 ngÆ°á»i<br/>Tá»‘i Æ°u siÃªu tham sá»‘ GridSearchCV"]
        TrainExp --> BestModel["Tuyá»ƒn chá»n mÃ´ hÃ¬nh tá»‘t nháº¥t<br/>StandardScaler + SVM Kernel RBF"]
        BestModel --> Artifacts[("models/best_model.joblib<br/>training_metadata.json<br/>split_manifest.json")]
        Artifacts --> Eval["src/evaluate.py<br/>ÄÃ¡nh giÃ¡ Ä‘á»™c láº­p trÃªn locked test set<br/>(person01, person10, person12 - 707 máº«u)"]
        Eval --> Results["results/evaluation/<br/>BÃ¡o cÃ¡o Metrics, Confusion Matrix, Plots"]
    end

    subgraph Online_Pipeline["2. Luá»“ng Suy Diá»…n Thá»i Gian Thá»±c (Online Realtime Pipeline)"]
        Frame["Webcam Stream (45Â° bÃªn trÃ¡i)"] --> R_Det["PoseDetector (yolo26n-pose.pt)<br/>Auto-fallback CPU náº¿u CUDA lá»—i"]
        R_Det --> R_Kpts["6 Keypoints thÃ¢n trÃªn"]
        R_Kpts --> R_Ext["FeatureExtractor (32 features)"]
        R_Ext --> Calib{"Personal Calibration?<br/>(Láº¥y baseline 2-3s Ä‘áº§u)"}
        Calib -- "CÃ³ baseline" --> Hybrid["Vector Hybrid:<br/>Features Tuyá»‡t Äá»‘i + Î” Features"]
        Calib -- "ChÆ°a cÃ³" --> Direct["Features Tuyá»‡t Äá»‘i"]
        Hybrid --> Predictor["best_model.joblib<br/>SVM RBF Classification"]
        Direct --> Predictor
        Predictor --> Temporal["TemporalMonitor (Lá»c chuá»—i thá»i gian)<br/>Rolling Window / Majority Voting<br/>Khá»­ rung giáº­t nhÃ£n (flickering)"]
        Temporal --> Stats["SessionStatistics<br/>TÃ­nh thá»i lÆ°á»£ng ngá»“i Ä‘Ãºng/sai<br/>Cáº£nh bÃ¡o tÆ° tháº¿ xáº¥u liÃªn tá»¥c"]
        Stats --> Display["Giao Diá»‡n á»¨ng Dá»¥ng (App / Web / Overlay)<br/>Hiá»ƒn thá»‹ nhÃ£n, FPS, cáº£nh bÃ¡o & Posture Score"]
    end
```

---

## 5. Cáº¥u TrÃºc ThÆ° Má»¥c Dá»± Ãn

Cáº¥u trÃºc mÃ£ nguá»“n Ä‘Æ°á»£c phÃ¢n Ä‘á»‹nh rÃµ rÃ ng giá»¯a táº§ng dá»¯ liá»‡u, mÃ´ hÃ¬nh, mÃ£ nguá»“n chá»©c nÄƒng vÃ  tÃ i liá»‡u nghiÃªn cá»©u:

```text
smart-posture-monitor/
|-- data/                                   # Dá»¯ liá»‡u cá»¥c bá»™ (Ä‘Æ°á»£c ignore bá»Ÿi Git)
|   |-- raw/                                # data/raw/<label>/<person_id>_<session_id>/<img.jpg>
|   |-- processed/                          # features.csv (táº­p 32 Ä‘áº·c trÆ°ng Ä‘Ã£ trÃ­ch xuáº¥t tá»« 12 ngÆ°á»i)
|   `-- rejected/                           # rejected_images.csv (log cÃ¡c áº£nh bá»‹ loáº¡i khi build)
|-- docs/                                   # ToÃ n bá»™ tÃ i liá»‡u thiáº¿t káº¿ vÃ  bÃ¡o cÃ¡o nghiÃªn cá»©u
|   |-- Bao_cao_tien_do_V03_2026-09-27.md   # BÃ¡o cÃ¡o tiáº¿n Ä‘á»™ V03
|   |-- Danh_gia_lai_12_nguoi_LOPO_2026-09-28.md # ÄÃ¡nh giÃ¡ LOPO 12 ngÆ°á»i sau khi lá»c outlier
|   |-- Changelog_Webcam_TemporalMonitor_2026-09-28.md # Chi tiáº¿t bá»™ lá»c thá»i gian & gating
|   |-- Nghien_cuu_cai_tien_mo_hinh_va_baseline_calibration_...md # NghiÃªn cá»©u tá»‘i Æ°u mÃ´ hÃ¬nh & calibration
|   |-- V03_feature_design_proposal.md      # Thiáº¿t káº¿ chi tiáº¿t V03: Depth Proxy & Calibration
|   |-- v03_implementation_plan.md          # Káº¿ hoáº¡ch triá»ƒn khai ká»¹ thuáº­t V03 tá»«ng bÆ°á»›c
|   |-- ytuong.md                           # PhÃ¢n tÃ­ch toÃ¡n há»c & cÃ¡c báº«y cá»§a Baseline Calibration
|   `-- archive_v02/                        # LÆ°u trá»¯ lá»‹ch sá»­ tÃ i liá»‡u phÃ¡t triá»ƒn giai Ä‘oáº¡n V01 - V02
|-- models/
|   |-- best_model.joblib                   # MÃ´ hÃ¬nh SVM RBF Tuned chÃ­nh thá»©c Ä‘Ã£ huáº¥n luyá»‡n
|   |-- split_manifest.json                 # KhÃ³a phÃ¢n Ä‘á»‹nh train/test persons (Locked Holdout)
|   |-- training_metadata.json              # Metadata danh sÃ¡ch features vÃ  thÃ´ng sá»‘ huáº¥n luyá»‡n
|   `-- yolo26n-pose.pt                     # Trá»ng sá»‘ YOLO Pose (Ä‘Æ°á»£c ignore bá»Ÿi Git)
|-- notebooks/
|   |-- 01_eda_dataset.ipynb                # PhÃ¢n tÃ­ch dá»¯ liá»‡u khÃ¡m phÃ¡ (EDA)
|   `-- 02_training_experiments.ipynb       # HUáº¤N LUYá»†N CHÃNH THá»¨C: So sÃ¡nh mÃ´ hÃ¬nh & tune SVM
|-- results/
|   |-- baseline_cv_results.csv             # Káº¿t quáº£ CV cá»§a cÃ¡c baseline models
|   |-- model_selection_results.csv         # Báº£ng xáº¿p háº¡ng tuyá»ƒn chá»n mÃ´ hÃ¬nh
|   |-- svm_search_results.csv              # Lá»‹ch sá»­ tÃ¬m kiáº¿m siÃªu tham sá»‘ GridSearchCV cá»§a SVM
|   |-- xgboost_search_results.csv          # Lá»‹ch sá»­ tÃ¬m kiáº¿m RandomizedSearchCV cá»§a XGBoost
|   |-- evaluation/                         # BÃ¡o cÃ¡o Ä‘Ã¡nh giÃ¡ Ä‘á»™c láº­p trÃªn táº­p test unseen (3 ngÆ°á»i)
|   `-- lopo/                               # BÃ¡o cÃ¡o vÃ  báº£ng dá»¯ liá»‡u Leave-One-Person-Out CV
|-- scripts/
|   |-- test_webcam_model.py                # á»¨ng dá»¥ng kiá»ƒm thá»­ mÃ´ hÃ¬nh realtime qua webcam
|   `-- copy_rejected_images.py             # CÃ´ng cá»¥ váº½ lá»—i vÃ  trÃ­ch xuáº¥t áº£nh bá»‹ loáº¡i Ä‘á»ƒ audit
|-- src/
|   |-- __init__.py
|   |-- pose_detector.py                    # PhÃ¡t hiá»‡n keypoints báº±ng YOLO (kÃ¨m CPU fallback)
|   |-- feature_extractor.py                # TrÃ­ch xuáº¥t 32 Ä‘áº·c trÆ°ng (29 V02 + 3 Depth Proxies)
|   |-- dataset_builder.py                  # QuÃ©t áº£nh thÃ´ vÃ  táº¡o táº­p dá»¯ liá»‡u features.csv
|   |-- preprocessing.py                    # Schema validation, Group-aware split, chá»‘ng leakage
|   |-- temporal_monitor.py                 # Bá»™ lá»c lÃ m mÆ°á»£t thá»i gian (EMA, Hysteresis, Yaw Gating)
|   |-- evaluate.py                         # ÄÃ¡nh giÃ¡ Ä‘á»™c láº­p trÃªn locked test set (3 ngÆ°á»i)
|   `-- lopo_evaluate.py                    # ÄÃ¡nh giÃ¡ Ä‘a Ä‘á»‘i tÆ°á»£ng Leave-One-Person-Out (12 folds)
|-- tests/
|   |-- test_preprocessing.py               # Unit tests kiá»ƒm thá»­ toÃ n váº¹n pipeline tiá»n xá»­ lÃ½
|   |-- test_temporal_monitor.py            # Unit tests kiá»ƒm thá»­ bá»™ lá»c lÃ m mÆ°á»£t thá»i gian
|   |-- test_pose_detector.py               # Test chá»©c nÄƒng PoseDetector
|   `-- test_feature_extractor.py           # Test chá»©c nÄƒng trÃ­ch xuáº¥t Ä‘áº·c trÆ°ng
|-- requirements.txt                        # Danh sÃ¡ch thÆ° viá»‡n phá»¥ thuá»™c
`-- README.md                               # TÃ i liá»‡u tá»•ng quan dá»± Ã¡n
```

---

## 6. TrÃ­ch Xuáº¥t Äáº·c TrÆ°ng HÃ¬nh Há»c (Feature Engineering)

ToÃ n bá»™ Ä‘áº·c trÆ°ng Ä‘Æ°á»£c tÃ­nh toÃ¡n trong [`src/feature_extractor.py`](file:///d:/BTL/repo/smart-posture-monitor/src/feature_extractor.py).

### 6.1. Há»‡ tá»a Ä‘á»™ chuáº©n hÃ³a cÆ¡ thá»ƒ (Body Frame Normalization)
Äá»ƒ loáº¡i bá» sá»± áº£nh hÆ°á»Ÿng cá»§a viá»‡c ngÆ°á»i dÃ¹ng ngá»“i gáº§n hay xa webcam:
1. **Gá»‘c tá»a Ä‘á»™**: TÃ¢m hai vai $\text{shoulder-center} = \frac{\text{left-shoulder} + \text{right-shoulder}}{2}$.
2. **ÄÆ¡n vá»‹ Ä‘o chuáº©n hÃ³a**: Má»i khoáº£ng cÃ¡ch Ä‘Æ°á»£c chia cho chiá»u rá»™ng hai vai $\text{shoulder-width} = \|\text{right-shoulder} - \text{left-shoulder}\|$. Chiá»u rá»™ng vai lÃ  háº±ng sá»‘ giáº£i pháº«u khÃ´ng biáº¿n dáº¡ng khi ngá»“i gÃ¹ hay nghiÃªng, Ä‘á»“ng thá»i tá»· lá»‡ nghá»‹ch vá»›i khoáº£ng cÃ¡ch tá»›i camera.
3. **Há»‡ trá»¥c cÆ¡ thá»ƒ**:
   - Trá»¥c hoÃ nh $\vec{u}_{\text{body}}$: Vector Ä‘Æ¡n vá»‹ ná»‘i tá»« vai trÃ¡i sang vai pháº£i.
   - Trá»¥c tung $\vec{v}_{\text{body}}$: Vector Ä‘Æ¡n vá»‹ vuÃ´ng gÃ³c vá»›i trá»¥c vai, hÆ°á»›ng lÃªn trÃªn Ä‘áº§u.

### 6.2. Danh má»¥c 32 Ä‘áº·c trÆ°ng hÃ¬nh há»c hiá»‡n táº¡i

Codebase hiá»‡n táº¡i há»— trá»£ **32 Ä‘áº·c trÆ°ng** (gá»“m 29 Ä‘áº·c trÆ°ng V02 vÃ  3 Ä‘áº·c trÆ°ng Depth Proxy V03):

| STT | TÃªn Äáº·c TrÆ°ng | NhÃ³m | CÃ´ng Thá»©c / Ã NghÄ©a Váº­t LÃ½ |
|:---:|---|---|---|
| 1 | `shoulder_angle` | GÃ³c cÆ¡ báº£n | GÃ³c nghiÃªng cá»§a Ä‘Æ°á»ng ná»‘i hai vai so vá»›i phÆ°Æ¡ng ngang áº£nh. Nháº¡y nháº¥t vá»›i `lean_left` / `lean_right`. |
| 2 | `eye_shoulder_angle` | GÃ³c cÆ¡ báº£n | GÃ³c lá»‡ch tÆ°Æ¡ng Ä‘á»‘i giá»¯a trá»¥c hai máº¯t vÃ  trá»¥c hai vai. |
| 3 | `eye_vertical_difference` | GÃ³c cÆ¡ báº£n | Äá»™ chÃªnh lá»‡ch cao Ä‘á»™ hai máº¯t tÃ­nh theo há»‡ trá»¥c cÆ¡ thá»ƒ (chia cho `shoulder_width`). |
| 4-5 | `nose_x_body`, `nose_y_body` | Tá»a Ä‘á»™ Body Frame | Tá»a Ä‘á»™ x (trá»¥c vai) vÃ  y (trá»¥c dá»c cÆ¡ thá»ƒ) cá»§a mÅ©i. |
| 6-7 | `eye_center_x_body`, `eye_center_y_body` | Tá»a Ä‘á»™ Body Frame | Tá»a Ä‘á»™ x vÃ  y cá»§a tÃ¢m hai máº¯t trong Body Frame. |
| 8-9 | `left_ear_x_body`, `left_ear_y_body` | Tá»a Ä‘á»™ Body Frame | Tá»a Ä‘á»™ x vÃ  y cá»§a tai trÃ¡i trong Body Frame. |
| 10-11 | `nose_eye_dx`, `nose_eye_dy` | Tá»a Ä‘á»™ Body Frame | Vector khoáº£ng cÃ¡ch tÆ°Æ¡ng Ä‘á»‘i giá»¯a mÅ©i vÃ  tÃ¢m hai máº¯t chiáº¿u lÃªn trá»¥c cÆ¡ thá»ƒ. |
| 12 | `eye_width_ratio` | Tá»· lá»‡ khoáº£ng cÃ¡ch | Khoáº£ng cÃ¡ch giá»¯a hai máº¯t chia cho Ä‘á»™ rá»™ng vai ($\text{eye-width} / \text{shoulder-width}$). |
| 13 | `ear_eye_ratio` | Tá»· lá»‡ khoáº£ng cÃ¡ch | Khoáº£ng cÃ¡ch tá»« tai trÃ¡i Ä‘áº¿n máº¯t trÃ¡i chia cho Ä‘á»™ rá»™ng vai. |
| 14 | `nose_shoulder_center_distance` | Tá»· lá»‡ khoáº£ng cÃ¡ch | Khoáº£ng cÃ¡ch tá»« mÅ©i Ä‘áº¿n tÃ¢m hai vai (chuáº©n hÃ³a theo Ä‘á»™ rá»™ng vai). |
| 15 | `eye_shoulder_center_distance` | Tá»· lá»‡ khoáº£ng cÃ¡ch | Khoáº£ng cÃ¡ch tá»« tÃ¢m hai máº¯t Ä‘áº¿n tÃ¢m hai vai (chuáº©n hÃ³a theo Ä‘á»™ rá»™ng vai). |
| 16 | `nose_shoulder_asymmetry` | Äá»™ báº¥t Ä‘á»‘i xá»©ng | ChÃªnh lá»‡ch khoáº£ng cÃ¡ch tá»« mÅ©i Ä‘áº¿n vai trÃ¡i so vá»›i vai pháº£i: $(\text{dist}(N, L) - \text{dist}(N, R)) / W$. |
| 17 | `eye_shoulder_asymmetry` | Äá»™ báº¥t Ä‘á»‘i xá»©ng | ChÃªnh lá»‡ch khoáº£ng cÃ¡ch tá»« tÃ¢m máº¯t Ä‘áº¿n hai bÃªn vai. |
| 18 | `nose_ear_ratio` | Äá»™ báº¥t Ä‘á»‘i xá»©ng | Khoáº£ng cÃ¡ch tá»« mÅ©i Ä‘áº¿n tai trÃ¡i chia cho Ä‘á»™ rá»™ng vai. |
| 19 | `head_body_angle` | GÃ³c khÃ´ng gian | GÃ³c nghiÃªng cá»§a trá»¥c Ä‘áº§u (tÃ¢m máº¯t) so vá»›i trá»¥c tháº³ng Ä‘á»©ng cÆ¡ thá»ƒ: $\text{atan2}(x, y)$. |
| 20 | `head_gravity_angle` | GÃ³c trá»ng lá»±c | GÃ³c cÃ³ dáº¥u giá»¯a trá»¥c Ä‘áº§u (tÃ¢m vai $\to$ tÃ¢m máº¯t) vÃ  phÆ°Æ¡ng tháº³ng Ä‘á»©ng cá»§a áº£nh ($0^\circ$ lÃ  hÆ°á»›ng lÃªn). |
| 21 | `nose_gravity_angle` | GÃ³c trá»ng lá»±c | GÃ³c cÃ³ dáº¥u giá»¯a trá»¥c mÅ©i (tÃ¢m vai $\to$ mÅ©i) vÃ  phÆ°Æ¡ng tháº³ng Ä‘á»©ng trá»ng lá»±c. |
| 22 | `face_pitch_angle` | GÃ³c khuÃ´n máº·t | GÃ³c ngáº©ng/cÃºi cá»§a máº·t theo vector tÆ°Æ¡ng Ä‘á»‘i mÅ©i - máº¯t. Pháº£n Ã¡nh ráº¥t nháº¡y tÆ° tháº¿ `forward_slouch`. |
| 23 | `eye_vertical_axis_offset` | Lá»‡ch trá»¥c Ä‘á»©ng | Äá»™ lá»‡ch phÆ°Æ¡ng ngang cá»§a tÃ¢m máº¯t so vá»›i Ä‘Æ°á»ng tháº³ng Ä‘á»©ng Ä‘i qua tÃ¢m vai. |
| 24 | `nose_vertical_axis_offset` | Lá»‡ch trá»¥c Ä‘á»©ng | Äá»™ lá»‡ch phÆ°Æ¡ng ngang cá»§a mÅ©i so vá»›i Ä‘Æ°á»ng tháº³ng Ä‘á»©ng Ä‘i qua tÃ¢m vai. |
| 25 | `head_mean_height` | Cao Ä‘á»™ Ä‘áº§u | Cao Ä‘á»™ trung bÃ¬nh cá»§a 3 má»‘c Ä‘áº§u: $(\text{nose-y} + \text{eye-y} + \text{ear-y}) / 3$. |
| 26 | `head_height_spread` | PhÃ¢n tÃ¡n Ä‘áº§u | Äá»™ lá»‡ch chuáº©n cao Ä‘á»™ cá»§a 3 má»‘c Ä‘áº§u trong Body Frame. |
| 27 | `nose_body_angle` | GÃ³c má»‘c Ä‘áº§u | GÃ³c cá»§a vector mÅ©i so vá»›i trá»¥c Ä‘á»©ng cÆ¡ thá»ƒ. |
| 28 | `ear_body_angle` | GÃ³c má»‘c Ä‘áº§u | GÃ³c cá»§a vector tai trÃ¡i so vá»›i trá»¥c Ä‘á»©ng cÆ¡ thá»ƒ. |
| 29 | `head_axis_angle_spread` | PhÃ¢n tÃ¡n gÃ³c | Äá»™ lá»‡ch chuáº©n phÃ¢n tÃ¡n giá»¯a cÃ¡c gÃ³c trá»¥c Ä‘áº§u (máº¯t, mÅ©i, tai). Thá»ƒ hiá»‡n má»©c Ä‘á»™ xoay/váº·n Ä‘áº§u. |
| 30 | `face_shoulder_scale_ratio` *(V03 D1)* | **Depth Proxy** | **Diá»‡n tÃ­ch tam giÃ¡c máº·t (`left_eye`, `right_eye`, `nose`) chia cho $\text{shoulder-width}^2$**.<br/>â€¢ GÃ¹ lÆ°ng (gáº§n camera): diá»‡n tÃ­ch lá»›n $\to$ TÄ‚NG.<br/>â€¢ NghiÃªng pháº£i (xa camera): diá»‡n tÃ­ch nhá» $\to$ GIáº¢M. |
| 31 | `ear_nose_depth_proxy` *(V03 D3)* | **Depth Proxy** | **Tá»· lá»‡ $\text{dist}(\text{left-ear}, \text{nose}) / \text{dist}(\text{left-ear}, \text{eye-center})$**.<br/>â€¢ GÃ¹ lÆ°ng: mÅ©i vÆ°Æ¡n ra trÆ°á»›c $\to$ ear-nose tÄƒng $\to$ TÄ‚NG.<br/>â€¢ NghiÃªng pháº£i: Ä‘áº§u dá»‹ch Ä‘á»“ng bá»™ $\to$ GIá»® NGUYÃŠN. |
| 32 | `face_rotation_proxy` *(V03 D4)* | **Depth Proxy** | **Tá»· lá»‡ $\text{dist}(\text{left-eye}, \text{nose}) / \text{dist}(\text{right-eye}, \text{nose})$**.<br/>Proxy Ä‘o gÃ³c xoay máº·t (Yaw).<br/>â€¢ GÃ¹ lÆ°ng: máº·t nhÃ¬n tháº³ng $\to \approx 1.0$.<br/>â€¢ NghiÃªng pháº£i: máº·t xoay gÃ³c $\to \ne 1.0$. |

---

## 7. Táº­p Dá»¯ Liá»‡u & PhÃ¢n TÃ­ch KhÃ¡m PhÃ¡ (Dataset & EDA)

### 7.1. Thá»‘ng kÃª táº­p dá»¯ liá»‡u
Táº­p dá»¯ liá»‡u V02 Ä‘Æ°á»£c thu tháº­p thá»±c táº¿ tá»« nhiá»u Ä‘á»‘i tÆ°á»£ng ngÆ°á»i ngá»“i táº¡i bÃ n lÃ m viá»‡c vá»›i camera Ä‘áº·t cháº¿ch 45Â° bÃªn trÃ¡i:

| Chá»‰ sá»‘ | GiÃ¡ trá»‹ | Ghi chÃº |
|---|---:|---|
| **Tá»•ng sá»‘ áº£nh thÃ´** | **4,341** | QuÃ©t Ä‘á»‡ quy tá»« `data/raw/` |
| **Sá»‘ máº«u há»£p lá»‡** | **4,014** | Tá»· lá»‡ trÃ­ch xuáº¥t thÃ nh cÃ´ng: **92.47%** |
| **Sá»‘ máº«u bá»‹ loáº¡i** | **327** | Bá»‹ che khuáº¥t hoáº·c confidence keypoint $< 0.35$ |
| **Sá»‘ Ä‘á»‘i tÆ°á»£ng ngÆ°á»i** | **14** | `person01` Ä‘áº¿n `person14` |
| **Sá»‘ phiÃªn ghi hÃ¬nh** | **16** | `person07` cÃ³ 3 sessions, cÃ¡c Ä‘á»‘i tÆ°á»£ng cÃ²n láº¡i 1 session |
| **Sá»‘ lá»›p tÆ° tháº¿** | **4** | CÃ¢n báº±ng tá»± nhiÃªn giá»¯a cÃ¡c lá»›p |

PhÃ¢n bá»‘ máº«u giá»¯a cÃ¡c lá»›p tÆ° tháº¿:
- `correct`: 1,051 máº«u (26.18%)
- `forward_slouch`: 889 máº«u (22.15%)
- `lean_left`: 1,054 máº«u (26.26%)
- `lean_right`: 1,020 máº«u (25.41%)  
*(Tá»· lá»‡ chÃªnh lá»‡ch giá»¯a lá»›p nhiá»u nháº¥t vÃ  Ã­t nháº¥t chá»‰ lÃ  1.19x $\to$ Dá»¯ liá»‡u cÃ¢n báº±ng, khÃ´ng cáº§n SMOTE hay oversampling)*.

### 7.2. Káº¿t quáº£ chÃ­nh tá»« EDA (`notebooks/01_eda_dataset.ipynb`)
1. **ToÃ n váº¹n tuyá»‡t Ä‘á»‘i**: 4,014 máº«u khÃ´ng cÃ³ báº¥t ká»³ giÃ¡ trá»‹ thiáº¿u (`NaN`), khÃ´ng cÃ³ giÃ¡ trá»‹ vÃ´ cÃ¹ng (`Inf`), khÃ´ng cÃ³ báº£n ghi trÃ¹ng láº·p.
2. **Äáº·c trÆ°ng phÃ¢n biá»‡t Ä‘Æ¡n biáº¿n**:
   - `shoulder_angle` phÃ¢n biá»‡t dá»©t khoÃ¡t `lean_left` (gÃ³c Ã¢m/dÆ°Æ¡ng lá»›n) vÃ  `lean_right`.
   - `face_pitch_angle`, `nose_eye_dy` vÃ  `head_mean_height` pháº£n á»©ng ráº¥t máº¡nh khi ngÆ°á»i dÃ¹ng gÃ¹ lÆ°ng cÃºi Ä‘áº§u (`forward_slouch`).
3. **Hiá»‡n tÆ°á»£ng tÆ°Æ¡ng quan cao**: TÃ¬m tháº¥y 22 cáº·p Ä‘áº·c trÆ°ng cÃ³ há»‡ sá»‘ tÆ°Æ¡ng quan Pearson $|r| \ge 0.90$. PhÃ¢n tÃ­ch PCA cho tháº¥y cáº§n 4 thÃ nh pháº§n Ä‘á»ƒ Ä‘áº¡t 90% phÆ°Æ¡ng sai vÃ  6 thÃ nh pháº§n Ä‘á»ƒ Ä‘áº¡t 95% phÆ°Æ¡ng sai. Sá»± tÆ°Æ¡ng quan liÃªn tá»¥c nÃ y giáº£i thÃ­ch vÃ¬ sao mÃ´ hÃ¬nh háº¡t nhÃ¢n phi tuyáº¿n SVM RBF vÆ°á»£t trá»™i hoÃ n toÃ n so vá»›i mÃ´ hÃ¬nh dáº¡ng cÃ¢y.

---

## 8. Quy TrÃ¬nh Tiá»n Xá»­ LÃ½ Chá»‘ng RÃ² Rá»‰ Dá»¯ Liá»‡u

MÃ´-Ä‘un [`src/preprocessing.py`](file:///d:/BTL/repo/smart-posture-monitor/src/preprocessing.py) Ä‘Æ°á»£c thiáº¿t káº¿ tuÃ¢n thá»§ nghiÃªm ngáº·t nguyÃªn táº¯c **Group Leakage Prevention**:

1. **Tuyá»‡t Ä‘á»‘i khÃ´ng chia ngáº«u nhiÃªn theo tá»«ng frame (`random frame split`)**:  
   Náº¿u chia ngáº«u nhiÃªn theo frame, cÃ¡c frame liÃªn tiáº¿p cá»§a cÃ¹ng má»™t ngÆ°á»i trong cÃ¹ng má»™t buá»•i ngá»“i sáº½ lá»t vÃ o cáº£ táº­p train vÃ  táº­p test, khiáº¿n mÃ´ hÃ¬nh Ä‘áº¡t Ä‘á»™ chÃ­nh xÃ¡c áº£o 98-99% do "há»c váº¹t" khuÃ´n máº·t vÃ  trang phá»¥c cá»§a ngÆ°á»i Ä‘Ã³.
2. **KhÃ³a cá»‘ Ä‘á»‹nh táº­p kiá»ƒm thá»­ Ä‘á»™c láº­p (Group Holdout)**:  
   KhÃ³a riÃªng 3 ngÆ°á»i (`person01`, `person10`, `person12` - 707 máº«u) lÃ m táº­p kiá»ƒm thá»­ cuá»‘i cÃ¹ng. MÃ´ hÃ¬nh hoÃ n toÃ n khÃ´ng Ä‘Æ°á»£c tiáº¿p cáº­n báº¥t ká»³ thÃ´ng tin nÃ o cá»§a 3 ngÆ°á»i nÃ y trong lÃºc huáº¥n luyá»‡n hay tune tham sá»‘.
3. **ÄÆ°a toÃ n bá»™ Scaler vÃ o `Pipeline`**:  
   KhÃ´ng thá»±c hiá»‡n `StandardScaler` toÃ n cá»¥c trÃªn toÃ n bá»™ file CSV. Viá»‡c fit scaler chá»‰ Ä‘Æ°á»£c diá»…n ra bÃªn trong train folds cá»§a Cross-Validation.
4. **KhÃ´ng loáº¡i bá» outlier báº±ng IQR cÆ¡ há»c**:  
   Giá»¯ nguyÃªn cÃ¡c biáº¿n thiÃªn tá»± nhiÃªn cá»§a cÆ¡ thá»ƒ ngÆ°á»i Ä‘á»ƒ mÃ´ hÃ¬nh há»c Ä‘Æ°á»£c ranh giá»›i thá»±c táº¿.

---

## 9. Huáº¥n Luyá»‡n & Tuyá»ƒn Chá»n MÃ´ HÃ¬nh (Model Selection)

Thá»±c hiá»‡n trong notebook [`notebooks/02_training_experiments.ipynb`](file:///d:/BTL/repo/smart-posture-monitor/notebooks/02_training_experiments.ipynb) vá»›i **StratifiedGroupKFold** ($K = 5$ folds) chá»‰ trÃªn 11 ngÆ°á»i cá»§a táº­p huáº¥n luyá»‡n:

| Thá»© háº¡ng | MÃ´ hÃ¬nh | Ká»¹ thuáº­t tÃ¬m kiáº¿m | CV Macro F1 Mean | CV Macro F1 Std | Bá»™ siÃªu tham sá»‘ tá»‘t nháº¥t |
|:---:|---|---|:---:|:---:|---|
| ðŸ¥‡ **1** | **SVM Kernel RBF (Tuned)** | `GridSearchCV` | **0.6071** | **0.1701** | `C=10, gamma=0.001, class_weight=None` |
| 2 | SVM Kernel RBF (Default) | Baseline CV | 0.5601 | 0.1266 | `C=1.0, gamma='scale'` |
| 3 | Logistic Regression | Baseline CV | 0.5205 | 0.1661 | Máº·c Ä‘á»‹nh |
| 4 | XGBoost (Tuned) | `RandomizedSearchCV` | 0.5190 | 0.1256 | `n_estimators=600, max_depth=3, lr=0.05` |
| 5 | Extra Trees | Baseline CV | 0.5133 | 0.1458 | Máº·c Ä‘á»‹nh |
| 6 | XGBoost (Default) | Baseline CV | 0.5083 | 0.1329 | Máº·c Ä‘á»‹nh |
| 7 | Random Forest | Baseline CV | 0.4980 | 0.1508 | Máº·c Ä‘á»‹nh |
| 8 | Dummy Classifier | Baseline CV | 0.1050 | 0.0054 | Dá»± Ä‘oÃ¡n Ä‘a sá»‘ |

**LÃ½ do SVM RBF chiáº¿n tháº¯ng**:  
KhÃ´ng gian Ä‘áº·c trÆ°ng lÃ  cÃ¡c tá»· lá»‡ hÃ¬nh há»c vÃ  gÃ³c lÆ°á»£ng giÃ¡c liÃªn tá»¥c. Kernel RBF cÃ³ kháº£ nÄƒng Ã¡nh xáº¡ khÃ´ng gian nÃ y thÃ nh cÃ¡c ranh giá»›i phi tuyáº¿n mÆ°á»£t mÃ , khÃ´ng bá»‹ chia cáº¯t cá»¥c bá»™ dáº¡ng báº­c thang nhÆ° Decision Trees hay Random Forest khi gáº·p dá»¯ liá»‡u cá»§a ngÆ°á»i má»›i.

---

## 10. Káº¿t Quáº£ ÄÃ¡nh GiÃ¡ TrÃªn Táº­p Kiá»ƒm Thá»­ Äá»™c Láº­p (Held-out Evaluation)

ÄÆ°á»£c thá»±c hiá»‡n bá»Ÿi script Ä‘á»™c láº­p [`src/evaluate.py`](file:///d:/BTL/repo/smart-posture-monitor/src/evaluate.py) trÃªn 707 máº«u cá»§a 3 Ä‘á»‘i tÆ°á»£ng chÆ°a tá»«ng tháº¥y (`person01`, `person10`, `person12`):

### 10.1. CÃ¡c chá»‰ sá»‘ tá»•ng quÃ¡t

| Chá»‰ sá»‘ Ä‘Ã¡nh giÃ¡ | Káº¿t quáº£ Ä‘áº¡t Ä‘Æ°á»£c | Ã nghÄ©a thá»±c táº¿ |
|---|:---:|---|
| **Accuracy (Äá»™ chÃ­nh xÃ¡c tá»•ng thá»ƒ)** | **79.35%** | Dá»± Ä‘oÃ¡n chÃ­nh xÃ¡c 561 / 707 máº«u trÃªn ngÆ°á»i hoÃ n toÃ n má»›i. |
| **Balanced Accuracy** | **78.81%** | Äá»™ chÃ­nh xÃ¡c cÃ¢n báº±ng giá»¯a cáº£ 4 lá»›p. |
| **Macro Precision** | **78.42%** | Äá»™ chuáº©n xÃ¡c trung bÃ¬nh giá»¯a cÃ¡c lá»›p. |
| **Macro Recall** | **78.81%** | Tá»· lá»‡ thu há»“i trung bÃ¬nh giá»¯a cÃ¡c lá»›p. |
| **Macro F1-Score** | **77.91%** | Chá»‰ sá»‘ cÃ¢n báº±ng F1 tá»•ng thá»ƒ Ä‘áº¡t xáº¥p xá»‰ 78%. |
| **Weighted F1-Score** | **79.66%** | F1 cÃ³ tÃ­nh trá»ng sá»‘ kÃ­ch thÆ°á»›c máº«u. |

### 10.2. Hiá»‡u nÄƒng chi tiáº¿t tá»«ng lá»›p (Per-Class Performance)

| Lá»›p tÆ° tháº¿ | Precision | Recall | F1-Score | Sá»‘ máº«u (Support) | ÄÃ¡nh giÃ¡ |
|---|:---:|:---:|:---:|:---:|---|
| `correct` | **0.875** | **0.936** | **0.904** | 172 | **Xuáº¥t sáº¯c**: Báº¯t trá»n tÆ° tháº¿ Ä‘Ãºng, tá»· lá»‡ bá» sÃ³t chá»‰ 6.4%. |
| `lean_left` | **0.995** | **0.867** | **0.927** | 226 | **Äá»™ tin cáº­y gáº§n nhÆ° tuyá»‡t Ä‘á»‘i**: Khi bÃ¡o nghiÃªng trÃ¡i, 99.5% lÃ  chÃ­nh xÃ¡c. |
| `forward_slouch` | 0.564 | **0.773** | 0.652 | 132 | Recall tá»‘t nhÆ°ng Precision bá»‹ áº£nh hÆ°á»Ÿng do nháº§m láº«n vá»›i nghiÃªng pháº£i. |
| `lean_right` | 0.703 | 0.576 | 0.634 | 177 | Nháº­n diá»‡n má»©c khÃ¡, lÃ  Ä‘á»‘i tÆ°á»£ng chÃ­nh bá»‹ nháº§m sang gÃ¹ lÆ°ng. |

### 10.3. ÄÃ¡nh giÃ¡ tÃ­nh tá»•ng quÃ¡t hÃ³a theo tá»«ng ngÆ°á»i (Per-Person Evaluation)

| Äá»‘i tÆ°á»£ng test | Sá»‘ máº«u | Accuracy | Macro F1 | Weighted F1 | Nháº­n xÃ©t |
|---|:---:|:---:|:---:|:---:|---|
| `person01` | 148 | 79.73% | 0.6676 | 0.7690 | Gáº·p khÃ³ khÄƒn á»Ÿ lá»›p nghiÃªng pháº£i do gÃ³c ngá»“i Ä‘áº·c thÃ¹. |
| `person10` | 307 | 74.59% | 0.7449 | 0.7516 | Nháº­n diá»‡n tÆ° tháº¿ Ä‘Ãºng ráº¥t chuáº©n, á»•n Ä‘á»‹nh. |
| `person12` | 252 | **84.92%** | **0.8515** | **0.8517** | **Ráº¥t xuáº¥t sáº¯c**: F1 vÆ°á»£t trÃªn 85% trÃªn Ä‘á»‘i tÆ°á»£ng unseen. |

---

## 11. PhÃ¢n TÃ­ch Lá»—i Thá»±c Táº¿ & Äá»™ng Lá»±c NÃ¢ng Cáº¥p V03

Ma tráº­n nháº§m láº«n tá»« [`results/evaluation/confusion_matrix.csv`](file:///d:/BTL/repo/smart-posture-monitor/results/evaluation/confusion_matrix.csv):

```text
Thá»±c táº¿ \ Dá»± Ä‘oÃ¡n     correct   forward_slouch   lean_left   lean_right    Tá»•ng
-----------------------------------------------------------------------------
correct                 161             2             0            9      172
forward_slouch            0           102             0           30      132  â† 30 máº«u nháº§m sang lean_right
lean_left                 9            17           196            4      226
lean_right               14            60             1          102      177  â† 60 máº«u nháº§m sang forward_slouch
```

### Äiá»ƒm ngháº½n lá»›n nháº¥t: Cáº·p `forward_slouch` $\leftrightarrow$ `lean_right`
- CÃ³ tá»›i **90 máº«u bá»‹ nháº§m láº«n qua láº¡i** giá»¯a gÃ¹ lÆ°ng vÃ  nghiÃªng pháº£i.
- **Báº£n cháº¥t váº­t lÃ½**: Camera Ä‘áº·t á»Ÿ gÃ³c 45Â° bÃªn trÃ¡i:
  - Khi Ä‘á»‘i tÆ°á»£ng nghiÃªng ngÆ°á»i sang pháº£i (`lean_right`), Ä‘áº§u bá»‹ Ä‘áº©y ra xa camera vÃ  hÆ¡i lÃ¹i vá» sau gÃ³c nhÃ¬n $\to$ trÃªn áº£nh 2D, Ä‘á»™ cao Ä‘áº§u giáº£m xuá»‘ng vÃ  khoáº£ng cÃ¡ch Ä‘áº§u-vai co láº¡i.
  - Khi Ä‘á»‘i tÆ°á»£ng cÃºi gÃ¹ ngÆ°á»i (`forward_slouch`), Ä‘áº§u cÅ©ng háº¡ tháº¥p vÃ  khoáº£ng cÃ¡ch Ä‘áº§u-vai cÅ©ng co láº¡i.
  - Vá»›i 29 Ä‘áº·c trÆ°ng 2D ban Ä‘áº§u, mÃ´ hÃ¬nh bá»‹ máº¥t dáº¥u váº¿t chuyá»ƒn Ä‘á»™ng theo trá»¥c Z trong khÃ´ng gian 3D.
- **Káº¿t luáº­n**: ÄÃ¢y chÃ­nh lÃ  Ä‘á»™ng lá»±c cá»‘t lÃµi Ä‘á»ƒ xÃ¢y dá»±ng **V03 Depth Proxy Features** vÃ  **Personal Baseline Calibration**.

---

## 12. Äá»™t PhÃ¡ Ká»¹ Thuáº­t á»ž V03: Depth Proxy & Personal Baseline Calibration

V03 giáº£i quyáº¿t hai váº¥n Ä‘á» Ä‘á»™c láº­p báº±ng hai ká»¹ thuáº­t bá»• trá»£:

### 12.1. NhÃ³m Ä‘áº·c trÆ°ng chiá»u sÃ¢u khÃ´ng gian (Depth Proxy Features - 32 Features)
Dá»±a trÃªn nguyÃªn lÃ½ quang há»c phá»‘i cáº£nh cá»§a camera: $\text{KÃ­ch thÆ°á»›c biá»ƒu kiáº¿n (pixel)} \propto \frac{1}{Z}$. Khi má»™t váº­t tiáº¿n gáº§n camera, kÃ­ch thÆ°á»›c pixel cá»§a nÃ³ tÄƒng lÃªn; khi lÃ¹i xa, kÃ­ch thÆ°á»›c pixel giáº£m xuá»‘ng.

ÄÃ£ Ä‘Æ°á»£c tÃ­ch há»£p trá»±c tiáº¿p vÃ o [`src/feature_extractor.py`](file:///d:/BTL/repo/smart-posture-monitor/src/feature_extractor.py):
1. **`face_shoulder_scale_ratio` (D1 - Feature 30)**:  
   Diá»‡n tÃ­ch tam giÃ¡c máº·t (`left_eye`, `right_eye`, `nose`) chia cho $\text{shoulder-width}^2$.  
   - GÃ¹ lÆ°ng: Äáº§u tiáº¿n gáº§n camera $\to$ diá»‡n tÃ­ch tam giÃ¡c máº·t tÄƒng máº¡nh.  
   - NghiÃªng pháº£i: Äáº§u lÃ¹i xa camera $\to$ diá»‡n tÃ­ch tam giÃ¡c máº·t giáº£m.
2. **`ear_nose_depth_proxy` (D3 - Feature 31)**:  
   Tá»· lá»‡ $\text{dist}(\text{left-ear}, \text{nose}) / \text{dist}(\text{left-ear}, \text{eye-center})$.  
   - GÃ¹ lÆ°ng: MÅ©i vÆ°Æ¡n ra trÆ°á»›c trong khi tai á»Ÿ láº¡i phÃ­a sau $\to$ khoáº£ng cÃ¡ch tai-mÅ©i tÄƒng máº¡nh.  
   - NghiÃªng pháº£i: ToÃ n bá»™ Ä‘áº§u nghiÃªng Ä‘á»“ng bá»™ $\to$ tá»· lá»‡ á»•n Ä‘á»‹nh.
3. **`face_rotation_proxy` (D4 - Feature 32)**:  
   Tá»· lá»‡ $\text{dist}(\text{left-eye}, \text{nose}) / \text{dist}(\text{right-eye}, \text{nose})$.  
   - Äo gÃ³c xoay ngang (Yaw) cá»§a khuÃ´n máº·t. GÃ¹ lÆ°ng khÃ´ng xoay Ä‘áº§u ($\approx 1.0$), nghiÃªng pháº£i lÃ m máº·t xoay cháº¿ch so vá»›i camera ($\ne 1.0$).

### 12.2. Ã tÆ°á»Ÿng Hiá»‡u chuáº©n cÃ¡ nhÃ¢n (Personal Baseline Calibration)
Chi tiáº¿t toÃ¡n há»c Ä‘Æ°á»£c phÃ¢n tÃ­ch trong tÃ i liá»‡u [docs/ytuong.md](docs/ytuong.md):
- **NguyÃªn lÃ½**: YÃªu cáº§u ngÆ°á»i dÃ¹ng ngá»“i tháº³ng lÆ°ng trong 2â€“3 giÃ¢y Ä‘áº§u tiÃªn cá»§a phiÃªn lÃ m viá»‡c Ä‘á»ƒ láº¥y vector trung bÃ¬nh chuáº©n $\vec{f}_{\text{base}}$.
- Má»i frame tiáº¿p theo Ä‘Æ°á»£c tÃ­nh dÆ°á»›i dáº¡ng Ä‘á»™ lá»‡ch: $\Delta \vec{f}_t = \vec{f}_t - \vec{f}_{\text{base}}$.  
- **Khá»­ háº±ng sá»‘ nhÃ¢n tráº¯c há»c**: Khi ngá»“i Ä‘Ãºng, $\Delta \vec{f} \approx \vec{0}$ cho má»i ngÆ°á»i dÃ¹ng (dÃ¹ cá»• dÃ i hay ngáº¯n, vai rá»™ng hay háº¹p). Khi gÃ¹ lÆ°ng, $\Delta \vec{f}$ cÃ¹ng trá» vá» má»™t hÆ°á»›ng chuyá»ƒn Ä‘á»™ng.
- **CÆ¡ cháº¿ Hybrid an toÃ n**: Giá»¯ song song cáº£ Ä‘áº·c trÆ°ng tuyá»‡t Ä‘á»‘i vÃ  Ä‘áº·c trÆ°ng vi phÃ¢n $\Delta \vec{f}$ Ä‘á»ƒ trÃ¡nh trÆ°á»ng há»£p ngÆ°á»i dÃ¹ng hiá»‡u chuáº©n sai (Ä‘ang gÃ¹ mÃ  báº¥m hiá»‡u chuáº©n).

---

## 13. Há»‡ Thá»‘ng Kiá»ƒm Thá»­ Tá»± Äá»™ng (Testing & QA)

Dá»± Ã¡n thiáº¿t láº­p há»‡ thá»‘ng kiá»ƒm thá»­ toÃ n diá»‡n báº£o vá»‡ cháº¥t lÆ°á»£ng mÃ£ nguá»“n:

### 13.1. Unit tests tiá»n xá»­ lÃ½: `tests/test_preprocessing.py`
Bao gá»“m **28 unit tests** kiá»ƒm tra tá»± Ä‘á»™ng báº±ng pytest:
- Kiá»ƒm tra toÃ n váº¹n dá»¯ liá»‡u, báº¯t lá»—i khi file CSV rá»—ng, sai schema hoáº·c thiáº¿u cá»™t.
- Báº¯t lá»—i khi xuáº¥t hiá»‡n giÃ¡ trá»‹ khÃ´ng pháº£i sá»‘, giÃ¡ trá»‹ `NaN`, `+Inf`, `-Inf`.
- PhÃ¡t hiá»‡n dÃ²ng dá»¯ liá»‡u bá»‹ trÃ¹ng láº·p hoáº·c xung Ä‘á»™t nhÃ£n trÃªn cÃ¹ng 1 Ä‘Æ°á»ng dáº«n áº£nh.
- Kiá»ƒm tra cÆ¡ cháº¿ táº¡o `recording_id` duy nháº¥t vÃ  lá»c protocol invalid.
- Äáº£m báº£o ma tráº­n `X` tuyá»‡t Ä‘á»‘i khÃ´ng bá»‹ rÃ² rá»‰ metadata.

Cháº¡y test báº±ng lá»‡nh:
```bash
.venv\Scripts\python.exe -m pytest tests/test_preprocessing.py -v
```

### 13.2. Visual tests kiá»ƒm tra webcam
- [`tests/test_pose_detector.py`](file:///d:/BTL/repo/smart-posture-monitor/tests/test_pose_detector.py): Má»Ÿ webcam trá»±c tiáº¿p, kiá»ƒm tra tá»‘c Ä‘á»™ phÃ¡t hiá»‡n ngÆ°á»i vÃ  váº½ 6 keypoints.
- [`tests/test_feature_extractor.py`](file:///d:/BTL/repo/smart-posture-monitor/tests/test_feature_extractor.py): Má»Ÿ webcam, trÃ­ch xuáº¥t Ä‘áº§y Ä‘á»§ 32 Ä‘áº·c trÆ°ng vÃ  in ra terminal má»—i giÃ¢y.

---

## 14. HÆ°á»›ng Dáº«n CÃ i Äáº·t & Cháº¡y Há»‡ Thá»‘ng

### 14.1. YÃªu cáº§u mÃ´i trÆ°á»ng
- Há»‡ Ä‘iá»u hÃ nh: Windows 10/11, Linux, hoáº·c macOS.
- Python: PhiÃªn báº£n **3.10** hoáº·c **3.11** (khuyáº¿n nghá»‹ 3.10+).
- Webcam káº¿t ná»‘i trá»±c tiáº¿p vá»›i mÃ¡y tÃ­nh (náº¿u cháº¡y realtime).

### 14.2. CÃ i Ä‘áº·t tá»«ng bÆ°á»›c

Má»Ÿ terminal (PowerShell trÃªn Windows) táº¡i thÆ° má»¥c gá»‘c:

```powershell
# 1. Khá»Ÿi táº¡o mÃ´i trÆ°á»ng áº£o
python -m venv .venv

# 2. KÃ­ch hoáº¡t mÃ´i trÆ°á»ng áº£o
.venv\Scripts\activate

# 3. CÃ i Ä‘áº·t cÃ¡c thÆ° viá»‡n cáº§n thiáº¿t
pip install --upgrade pip
pip install -r requirements.txt
```

*LÆ°u Ã½ file trá»ng sá»‘*: Äáº£m báº£o file trá»ng sá»‘ YOLO Pose `yolo26n-pose.pt` Ä‘Ã£ Ä‘Æ°á»£c Ä‘áº·t táº¡i thÆ° má»¥c [`models/yolo26n-pose.pt`](file:///d:/BTL/repo/smart-posture-monitor/models/yolo26n-pose.pt).

### 14.3. Báº£ng tra cá»©u cÃ¡c lá»‡nh thá»±c thi chÃ­nh

| CÃ´ng viá»‡c | Lá»‡nh thá»±c thi (Command) |
|---|---|
| **Cháº¡y toÃ n bá»™ unit tests tá»± Ä‘á»™ng** | `pytest tests/test_preprocessing.py -v` |
| **TrÃ­ch xuáº¥t Ä‘áº·c trÆ°ng & build láº¡i dataset CSV** | `python -m src.dataset_builder` |
| **Kiá»ƒm tra quy trÃ¬nh tiá»n xá»­ lÃ½ dá»¯ liá»‡u** | `python -m src.preprocessing` |
| **Cháº¡y Ä‘Ã¡nh giÃ¡ chÃ­nh thá»©c trÃªn táº­p kiá»ƒm thá»­ Ä‘á»™c láº­p** | `python -m src.evaluate` |
| **Cháº¡y thá»­ nghiá»‡m webcam thá»i gian thá»±c** | `python scripts/test_webcam_model.py` |
| **Cháº¡y webcam vá»›i camera phá»¥ / táº¯t váº½ khung xÆ°Æ¡ng** | `python scripts/test_webcam_model.py --camera 1 --no-pose` |
| **TrÃ­ch xuáº¥t áº£nh lá»—i kÃ¨m lÃ½ do reject Ä‘á»ƒ audit dá»¯ liá»‡u** | `python scripts/copy_rejected_images.py` |
| **Test thá»§ cÃ´ng Pose Detector qua webcam** | `python -m tests.test_pose_detector` |
| **Test thá»§ cÃ´ng Feature Extractor qua webcam** | `python -m tests.test_feature_extractor` |

---

## 15. Lá»™ TrÃ¬nh PhÃ¡t Triá»ƒn Sáº£n Pháº©m (Roadmap)

```text
[V02: HoÃ n thÃ nh & ÄÃ³ng bÄƒng Benchmark]
  â”œâ”€â”€ Pipeline YOLO Pose + 29 Features + SVM RBF Tuned
  â”œâ”€â”€ Accuracy 79.35%, Macro F1 77.91% trÃªn 3 unseen test persons
  â””â”€â”€ PhÃ¡t hiá»‡n Ä‘iá»ƒm ngháº½n 90 máº«u nháº§m giá»¯a forward_slouch vÃ  lean_right
       â”‚
       â–¼
[V03 Phase 1: Depth Proxy Features - Äang triá»ƒn khai]
  â”œâ”€â”€ NÃ¢ng cáº¥p FeatureExtractor lÃªn 32 features (thÃªm D1, D3, D4)
  â”œâ”€â”€ Tá»± Ä‘á»™ng hÃ³a fallback CPU trong PoseDetector
  â”œâ”€â”€ Re-build táº­p dá»¯ liá»‡u features.csv vá»›i 32 features
  â””â”€â”€ Huáº¥n luyá»‡n láº¡i mÃ´ hÃ¬nh & ÄÃ¡nh giÃ¡ trÃªn cÃ¹ng locked test set
       â”‚
       â–¼
[V03 Phase 2: Personal Baseline Calibration - ÄÃ£ thiáº¿t káº¿]
  â”œâ”€â”€ Thá»­ nghiá»‡m offline backtest cÆ¡ cháº¿ trá»« baseline N frames Ä‘áº§u
  â”œâ”€â”€ XÃ¢y dá»±ng bá»™ Ä‘áº·c trÆ°ng Hybrid (Absolute + Delta features)
  â””â”€â”€ ÄÃ¡nh giÃ¡ má»©c Ä‘á»™ thu háº¹p variance giá»¯a cÃ¡c Ä‘á»‘i tÆ°á»£ng ngÆ°á»i dÃ¹ng
       â”‚
       â–¼
[V03 Phase 3: HoÃ n thiá»‡n Sáº£n Pháº©m & Giao Diá»‡n NgÆ°á»i DÃ¹ng]
  â”œâ”€â”€ posture_predictor.py: ÄÃ³ng gÃ³i API suy diá»…n cáº¥p cao
  â”œâ”€â”€ temporal_monitor.py: Bá»™ lá»c lÃ m mÆ°á»£t thá»i gian (chá»‘ng giáº­t nhÃ£n)
  â”œâ”€â”€ session_statistics.py: Äáº¿m thá»i gian ngá»“i sai & cáº£nh bÃ¡o sá»©c khá»e
  â””â”€â”€ app/app.py: Giao diá»‡n Desktop / Web Dashboard hiá»‡n Ä‘áº¡i
```

---

## 16. TÃ i Liá»‡u Ká»¹ Thuáº­t Äi KÃ¨m

Äá»ƒ náº¯m báº¯t sÃ¢u hÆ¡n cÃ¡c khÃ­a cáº¡nh toÃ¡n há»c, mÃ£ nguá»“n vÃ  dá»¯ liá»‡u thá»±c nghiá»‡m, tham kháº£o cÃ¡c tÃ i liá»‡u chuyÃªn Ä‘á» trong thÆ° má»¥c `docs/`:

- [docs/README_27_9.MD](docs/README_27_9.MD): PhÃ¢n tÃ­ch chuyÃªn sÃ¢u toÃ n bá»™ mÃ£ nguá»“n vÃ  pipeline V02.
- [docs/V03_feature_design_proposal.md](docs/V03_feature_design_proposal.md): Thiáº¿t káº¿ chi tiáº¿t cÃ¡c Ä‘áº·c trÆ°ng Depth Proxy vÃ  kiáº¿n trÃºc V03.
- [docs/ytuong.md](docs/ytuong.md): CÆ¡ sá»Ÿ toÃ¡n há»c, phÃ¢n tÃ¡ch tÃ­n hiá»‡u vÃ  8 báº«y cháº¿t ngÆ°á»i cáº§n trÃ¡nh cá»§a Personal Baseline Calibration.
- [docs/v03_implementation_plan.md](docs/v03_implementation_plan.md): Káº¿ hoáº¡ch rÃ  soÃ¡t vÃ  sá»­a Ä‘á»•i codebase tá»«ng bÆ°á»›c cho V03.
- [docs/Training_Realtime_Inference_Evaluation_V02_2026-09-27.md](docs/Training_Realtime_Inference_Evaluation_V02_2026-09-27.md): BÃ¡o cÃ¡o thá»±c nghiá»‡m tuyá»ƒn chá»n mÃ´ hÃ¬nh vÃ  benchmark V02.
- [results/evaluation/evaluation_summary.md](results/evaluation/evaluation_summary.md): BÃ¡o cÃ¡o tÃ³m táº¯t chá»‰ sá»‘ kiá»ƒm thá»­ chÃ­nh thá»©c trÃªn 707 máº«u held-out.

---
*Dá»± Ã¡n Smart Posture Monitor - Há»‡ thá»‘ng thá»‹ giÃ¡c mÃ¡y tÃ­nh vÃ  mÃ¡y há»c há»— trá»£ báº£o vá»‡ sá»©c khá»e tÆ° tháº¿ ngá»“i.*
