# Product Compliance ML — Error Analysis & Confusion Breakdown

## Confusion Matrix Breakdown across Partitions

### 1. Real Packaging Benchmark (N=20)
```
True Compliant:     TP=17, Pred NC=0, Pred UNCERTAIN=0
True Non-Compliant: Pred Comp=0 (Zero FN!), TP=3, Pred UNCERTAIN=0
```

### 2. Sample-by-Sample Diagnostics on Real Products

| Product ID | Title | Ground Truth | Final Prediction | Confidence | Abstention | Decision Basis |
|---|---|---|---|---|---|---|
| `real_prod_01` | Himalaya Purifying Neem Face Wash 150ml | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_02` | Johnson's Baby Daily Nourish Moisture Lotion 200ml | **COMPLIANT** | `COMPLIANT` | 0.9500 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_03` | Milton Thermosteel Flip Lid Vacuum Flask 1000ml | **COMPLIANT** | `COMPLIANT` | 0.9500 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_04` | Happilo 100% Natural Premium California Almonds 500g | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_05` | Vim Dishwash Gel Lemon 750ml Bottle | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_06` | Minimalist 10% Niacinamide Face Serum 30ml | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_07` | Pampers All Round Protection Baby Diapers (Large, 64 Count) | **COMPLIANT** | `COMPLIANT` | 0.9500 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_08` | Prestige Deluxe Alpha Stainless Steel Pressure Cooker 3L | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_09` | Organic India Tulsi Green Tea Classic 25 Tea Bags (45g) | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_10` | Lizol Disinfectant Surface Cleaner Citrus 2 Litres | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_11` | Unbranded Generic Cotton Kitchen Towel Set (Pack of 4) | **NON_COMPLIANT** | `NON_COMPLIANT` | 1.0000 | False | Missing mandatory Manufacturer / Packer name and address (D01); Missing mandatory Net Quantity declaration (D03); Missing mandatory Date of Manufacture / Packing (D04); Missing mandatory Consumer Care Contact Details (D06); Missing mandatory Country of Origin (D07) |
| `real_prod_12` | Crafted Wooden Coaster Set of 6 | **NON_COMPLIANT** | `NON_COMPLIANT` | 1.0000 | False | Missing mandatory Manufacturer / Packer name and address (D01); Missing mandatory Net Quantity declaration (D03); Missing mandatory Date of Manufacture / Packing (D04); Missing mandatory Consumer Care Contact Details (D06) |
| `real_prod_13` | Sunfeast Dark Fantasy Choco Fills 300g (Pack of 4) | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_14` | Dettol Original Liquid Handwash Refill Pouch 1500ml | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_15` | Mother Sparsh 99% Pure Water Unscented Baby Wipes (72 Wipes) | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_16` | Biotique Bio Kelp Protein Shampoo 340ml | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_17` | KitchenPro Silicone Spatula & Pastry Brush Set | **NON_COMPLIANT** | `NON_COMPLIANT` | 1.0000 | False | Missing mandatory Net Quantity declaration (D03); Missing mandatory Date of Manufacture / Packing (D04); Missing mandatory Consumer Care Contact Details (D06) |
| `real_prod_18` | Godrej aer Pocket Bathroom Air Fragrance Petal Crush Pink 10g | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_19` | Nestle Cerelac Baby Cereal with Milk Wheat Apple 300g | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |
| `real_prod_20` | Aroma Magic Sunblock Lotion SPF 30 (100ml) | **COMPLIANT** | `COMPLIANT` | 1.0000 | False | All mandatory statutory declarations verified on packaging |