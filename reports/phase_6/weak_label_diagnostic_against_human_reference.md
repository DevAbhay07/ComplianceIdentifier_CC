# Weak-Label Diagnostic Evaluation Against Human Reference

> [!NOTE]
> **DIAGNOSTIC PURPOSE ONLY**: This evaluation measures the baseline concordance of deterministic
> legal heuristics against the 75 locked human ground-truth records. The heuristics were NOT
> tuned against this reference.

- **Total Reference Records:** 75
- **Overall Label Agreement:** 61/75 (81.3%)

## 1. Declaration-Level Concordance

| Declaration | Name | Agreement Count | Concordance Rate |
| :--- | :--- | :--- | :--- |
| **`D01`** | Manufacturer Name & Address | 33/75 | **44.0%** |
| **`D02`** | Commodity Name | 75/75 | **100.0%** |
| **`D03`** | Net Quantity | 25/75 | **33.3%** |
| **`D04`** | Month/Year of Manufacture | 43/75 | **57.3%** |
| **`D05`** | Maximum Retail Price (MRP) | 75/75 | **100.0%** |
| **`D06`** | Consumer Care Details | 62/75 | **82.7%** |
| **`D07`** | Country of Origin | 43/75 | **57.3%** |

## 2. Disagreement Analysis

### `D01` Disagreements (42 items)

| Product ID | Title | Human Label | Automated Label | Auto Rules |
| :--- | :--- | :--- | :--- | :--- |
| `4e858b8d` | Allin Exporters Vetiver Oil | `PRESENT` | `PARTIAL` | `D01_BRAND_FIELD_FOUND` |
| `660e91db` | Proganix Agave Nectar Plus Silica Curling Cream Pl | `PRESENT` | `PARTIAL` | `D01_BRAND_FIELD_FOUND` |
| `aa98d4c6` | Aroma Care Intense Color Nail Polish Combo 175007  | `PRESENT` | `PARTIAL` | `D01_BRAND_FIELD_FOUND` |
| `86464d19` | My DT Lifestyle STONE STUDD PARTYWEAR METAL CLUTCH | `PRESENT` | `PARTIAL` | `D01_BRAND_FIELD_FOUND` |
| `a61a2d38` | Kartique Hand Made Box in smooth Finish with Glass | `PRESENT` | `PARTIAL` | `D01_BRAND_FIELD_FOUND` |

### `D02` Disagreements (0 items)

No disagreements found.

### `D03` Disagreements (50 items)

| Product ID | Title | Human Label | Automated Label | Auto Rules |
| :--- | :--- | :--- | :--- | :--- |
| `a11773bc` | Sivanna Baking Powder Rouge  Highlighter | `PARTIAL` | `ABSENT` | `D03_NO_QUANTITY_EVIDENCE` |
| `1835f47e` | Khadi Herbal Trifala Hair Oil | `PRESENT` | `ABSENT` | `D03_NO_QUANTITY_EVIDENCE` |
| `4942e0a9` | Giorgio Armani SMOOTH SILK LIP PENCIL 12 | `PARTIAL` | `ABSENT` | `D03_NO_QUANTITY_EVIDENCE` |
| `0abe931b` | Jovan Live It Combo's In Men's Deo The Musk Body S | `PRESENT` | `ABSENT` | `D03_NO_QUANTITY_EVIDENCE` |
| `e5d41d5a` | Adidas Lavish Life Combo's In Ice Dive EDT & Team  | `PRESENT` | `ABSENT` | `D03_NO_QUANTITY_EVIDENCE` |

### `D04` Disagreements (32 items)

| Product ID | Title | Human Label | Automated Label | Auto Rules |
| :--- | :--- | :--- | :--- | :--- |
| `4e858b8d` | Allin Exporters Vetiver Oil | `PRESENT` | `CANNOT_DETERMINE` | `D04_NO_DATE_IN_LISTING` |
| `660e91db` | Proganix Agave Nectar Plus Silica Curling Cream Pl | `PRESENT` | `CANNOT_DETERMINE` | `D04_NO_DATE_IN_LISTING` |
| `aa98d4c6` | Aroma Care Intense Color Nail Polish Combo 175007  | `PRESENT` | `CANNOT_DETERMINE` | `D04_NO_DATE_IN_LISTING` |
| `86464d19` | My DT Lifestyle STONE STUDD PARTYWEAR METAL CLUTCH | `PRESENT` | `CANNOT_DETERMINE` | `D04_NO_DATE_IN_LISTING` |
| `a61a2d38` | Kartique Hand Made Box in smooth Finish with Glass | `PRESENT` | `CANNOT_DETERMINE` | `D04_NO_DATE_IN_LISTING` |

### `D05` Disagreements (0 items)

No disagreements found.

### `D06` Disagreements (13 items)

| Product ID | Title | Human Label | Automated Label | Auto Rules |
| :--- | :--- | :--- | :--- | :--- |
| `4e858b8d` | Allin Exporters Vetiver Oil | `PRESENT` | `ABSENT` | `D06_NO_CARE_FOUND` |
| `ac112618` | StyleStuffs AMEYMUG200 Ceramic Mug | `CANNOT_DETERMINE` | `ABSENT` | `D06_NO_CARE_FOUND` |
| `8ac3d716` | SUNSHINE Sweety 4 Liter Plastic Dustbin | `CANNOT_DETERMINE` | `ABSENT` | `D06_NO_CARE_FOUND` |
| `8946a0e5` | Instapower 5 W LED Bulb | `PARTIAL` | `ABSENT` | `D06_NO_CARE_FOUND` |
| `9e822ea8` | Wallmantra Extra Large Vinyl Stickers Sticker | `PARTIAL` | `ABSENT` | `D06_NO_CARE_FOUND` |

### `D07` Disagreements (32 items)

| Product ID | Title | Human Label | Automated Label | Auto Rules |
| :--- | :--- | :--- | :--- | :--- |
| `4e858b8d` | Allin Exporters Vetiver Oil | `PRESENT` | `CANNOT_DETERMINE` | `D07_NO_ORIGIN_IN_LISTING` |
| `660e91db` | Proganix Agave Nectar Plus Silica Curling Cream Pl | `PRESENT` | `CANNOT_DETERMINE` | `D07_NO_ORIGIN_IN_LISTING` |
| `aa98d4c6` | Aroma Care Intense Color Nail Polish Combo 175007  | `PRESENT` | `CANNOT_DETERMINE` | `D07_NO_ORIGIN_IN_LISTING` |
| `86464d19` | My DT Lifestyle STONE STUDD PARTYWEAR METAL CLUTCH | `PRESENT` | `CANNOT_DETERMINE` | `D07_NO_ORIGIN_IN_LISTING` |
| `a61a2d38` | Kartique Hand Made Box in smooth Finish with Glass | `PRESENT` | `CANNOT_DETERMINE` | `D07_NO_ORIGIN_IN_LISTING` |

