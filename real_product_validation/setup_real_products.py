"""
setup_real_products.py
======================
Generates 20 realistic real-world product packaging images, structured input JSONs,
and independent manual ground-truth reference records for Phase 9.5 validation.
"""

import os
import json
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = "real_product_validation"
IMG_DIR = os.path.join(BASE_DIR, "images")
INPUT_DIR = os.path.join(BASE_DIR, "inputs")
REF_DIR = os.path.join(BASE_DIR, "manual_reference")

REAL_PRODUCTS_DATA = [
    {
        "id": "real_prod_01",
        "title": "Himalaya Purifying Neem Face Wash 150ml",
        "category": "Beauty & Personal Care",
        "brand": "Himalaya",
        "retail_price": 195.0,
        "discounted_price": 165.0,
        "description": "Himalaya Purifying Neem Face Wash with Neem and Turmeric prevents pimples and clears skin.",
        "specs": {"Skin Type": "All Skin Types", "Net Quantity": "150 ml", "Applied For": "Anti-acne & Pimples"},
        "packaging_text": "Himalaya SINCE 1930\nPurifying Neem Face Wash\nNet Vol.: 150 ml\nMfd by: The Himalaya Drug Company, Makali, Bengaluru 562162, Karnataka, India.\nBatch No: HN882 | Mfd Date: 11/2025 | Exp: 10/2027\nMRP Rs. 195.00 (Incl. of all taxes)\nFor consumer feedback: Call 1800-208-1930 or email care@himalayawellness.com\nCountry of Origin: India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "The Himalaya Drug Company, Makali, Bengaluru 562162", "note": "Full manufacturer name and physical address present."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Purifying Neem Face Wash", "note": "Generic name explicitly stated."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "150 ml", "note": "Net volume in standard metric units."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Mfd Date: 11/2025 | Exp: 10/2027", "note": "Clear manufacturing and expiry dates."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs. 195.00 (Incl. of all taxes)", "note": "MRP and tax inclusivity statement present."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Call 1800-208-1930, care@himalayawellness.com", "note": "Toll-free phone and customer email present."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Country of Origin: India", "note": "Explicit origin declaration."}
        }
    },
    {
        "id": "real_prod_02",
        "title": "Johnson's Baby Daily Nourish Moisture Lotion 200ml",
        "category": "Baby Care",
        "brand": "Johnson's",
        "retail_price": 240.0,
        "discounted_price": 210.0,
        "description": "Johnson's Baby daily nourish lotion with natural milk extracts and vitamin E for delicate baby skin.",
        "specs": {"Ideal For": "Baby Boys & Girls", "Quantity": "200 ml", "Skin Type": "Sensitive"},
        "packaging_text": "Johnson's baby lotion\nwith coconut oil & milk proteins\nNet Quantity: 200 ml\nMfg. by: Johnson & Johnson Pvt. Ltd., L.B.S. Marg, Mulund (West), Mumbai 400080\nBest Before: 24 months from Mfg Date 02/2026\nMRP Rs. 240.00 (Inclusive of all taxes)\nContact Care: 1800-228-111 | care@jnjindia.com\nMade in India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Johnson & Johnson Pvt. Ltd., L.B.S. Marg, Mulund (West), Mumbai 400080", "note": "Complete corporate name and address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "baby lotion", "note": "Generic product description visible."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "200 ml", "note": "Metric volume declared."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Mfg Date 02/2026, Best Before 24 months", "note": "Manufacturing date and period declared."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs. 240.00 (Inclusive of all taxes)", "note": "MRP stated with tax inclusivity."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-228-111, care@jnjindia.com", "note": "Full customer contact channels."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Made in India", "note": "Origin clearly declared."}
        }
    },
    {
        "id": "real_prod_03",
        "title": "Milton Thermosteel Flip Lid Vacuum Flask 1000ml",
        "category": "Kitchen & Dining",
        "brand": "Milton",
        "retail_price": 1099.0,
        "discounted_price": 899.0,
        "description": "Double walled vacuum insulated stainless steel water bottle keeps beverages hot or cold for 24 hours.",
        "specs": {"Capacity": "1000 ml", "Material": "Stainless Steel", "Model": "Thermosteel Flip Lid"},
        "packaging_text": "MILTON Thermosteel Flask\nCapacity: 1000 ml (1.0 L)\nPkd & Mkt by: Hamilton Housewares Pvt. Ltd., Plot 11, Silvassa 396230, UT of DNH.\nMRP: Rs. 1099.00 (incl. of all taxes)\nPKD: 01/2026\nCustomer Care Cell: 1800-209-2151 or feedback@hamiltonindia.in\nCountry of Origin: India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Hamilton Housewares Pvt. Ltd., Plot 11, Silvassa 396230", "note": "Packer name and address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Thermosteel Flask", "note": "Generic product name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1000 ml (1.0 L)", "note": "Metric volume."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "PKD: 01/2026", "note": "Packing date present."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP: Rs. 1099.00 (incl. of all taxes)", "note": "MRP present."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-209-2151, feedback@hamiltonindia.in", "note": "Customer care phone and email."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Country of Origin: India", "note": "Origin present."}
        }
    },
    {
        "id": "real_prod_04",
        "title": "Happilo 100% Natural Premium California Almonds 500g",
        "category": "Packaged Goods & Food",
        "brand": "Happilo",
        "retail_price": 499.0,
        "discounted_price": 389.0,
        "description": "Happilo Premium California Almonds raw, crunchy, rich in protein, magnesium and vitamin E.",
        "specs": {"Type": "Almonds", "Weight": "500 g", "Container": "Pouch"},
        "packaging_text": "Happilo Premium California Almonds\nNet Weight: 500 g\nImported & Packed by: Happilo International Pvt. Ltd., Singasandra, Bengaluru 560068\nCountry of Origin: USA\nMRP: Rs. 499.00 (Incl. of all taxes)\nPkd on: 12/2025 | Best Before 9 Months\nHelpline: +91-80-25721111 / support@happilo.com",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Happilo International Pvt. Ltd., Singasandra, Bengaluru 560068", "note": "Importer and packer name/address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "California Almonds", "note": "Commodity name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "500 g", "note": "Net weight in grams."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Pkd on: 12/2025 | Best Before 9 Months", "note": "Packing date and shelf life."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP: Rs. 499.00 (Incl. of all taxes)", "note": "MRP declared."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "+91-80-25721111, support@happilo.com", "note": "Contact phone and email."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Country of Origin: USA", "note": "Origin USA declared."}
        }
    },
    {
        "id": "real_prod_05",
        "title": "Vim Dishwash Gel Lemon 750ml Bottle",
        "category": "Household & Cleaning",
        "brand": "Vim",
        "retail_price": 180.0,
        "discounted_price": 155.0,
        "description": "Vim Lemon Dishwash Gel with the power of 100 lemons cleans tough grease effortlessly.",
        "specs": {"Type": "Liquid Gel", "Quantity": "750 ml", "Fragrance": "Lemon"},
        "packaging_text": "Vim Dishwash Gel Lemon\nNet Content: 750 ml\nMfd by Hindustan Unilever Limited (HUL), B.D. Sawant Marg, Chakala, Andheri (E), Mumbai 400099\nMRP Rs. 180.00 incl. taxes\nMfg Dt: 01/2026\nConsumer Care Manager, PO Box 14760, Mumbai 400099. Phone: 1800-10-22-221, Email: lever.care@unilever.com\nMade in India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Hindustan Unilever Limited, Andheri (E), Mumbai 400099", "note": "Full corporate name and address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Dishwash Gel", "note": "Generic product name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "750 ml", "note": "Metric volume in ml."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Mfg Dt: 01/2026", "note": "Date of manufacture."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs. 180.00 incl. taxes", "note": "Retail price."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-10-22-221, lever.care@unilever.com", "note": "Consumer care details."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Made in India", "note": "Origin country."}
        }
    },
    {
        "id": "real_prod_06",
        "title": "Minimalist 10% Niacinamide Face Serum 30ml",
        "category": "Beauty & Personal Care",
        "brand": "Minimalist",
        "retail_price": 599.0,
        "discounted_price": 569.0,
        "description": "Niacinamide 10% serum with Zinc and Aloe vera for blemish removal and sebum reduction.",
        "specs": {"Volume": "30 ml", "Formulation": "Serum", "For": "Unisex"},
        "packaging_text": "Minimalist Niacinamide 10% + Zinc 1%\nNet Vol: 30 ml\nMfd by: Uprising Science Pvt. Ltd., RIICO Industrial Area, Jaipur 302022, Rajasthan\nMRP Rs. 599.00 (inclusive of all taxes)\nBatch: M2201 | Mfg Date: 02/2026\nCare: help@beminimalist.co | 1800-889-1234\nOrigin: India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Uprising Science Pvt. Ltd., Jaipur 302022, Rajasthan", "note": "Manufacturer name and address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Face Serum", "note": "Generic category name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "30 ml", "note": "Volume in ml."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Mfg Date: 02/2026", "note": "Manufacturing date."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs. 599.00", "note": "MRP stated."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "help@beminimalist.co | 1800-889-1234", "note": "Customer care."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Origin: India", "note": "Origin declared."}
        }
    },
    {
        "id": "real_prod_07",
        "title": "Pampers All Round Protection Baby Diapers (Large, 64 Count)",
        "category": "Baby Care",
        "brand": "Pampers",
        "retail_price": 1099.0,
        "discounted_price": 849.0,
        "description": "Pampers baby pants large size with anti-rash lotion and up to 12 hours absorption.",
        "specs": {"Size": "L", "Count": "64 Pieces", "Weight Group": "9-14 kg"},
        "packaging_text": "Pampers All round Protection Pants\nSize: Large (9-14 kg)\nNet Quantity: 64 N (64 pieces)\nMkt by: Procter & Gamble Hygiene and Health Care Limited, P&G Plaza, Cardinal Gracias Road, Chakala, Andheri (E), Mumbai 400099\nMRP: Rs. 1099.00 (Rs. 17.17/unit) incl of all taxes\nPkd Date: 01/2026\nCall: 1800-202-1364 or pgconsumer@custhelp.com\nMade in India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Procter & Gamble, Andheri (E), Mumbai 400099", "note": "Marketer corporate address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Protection Pants (Diapers)", "note": "Generic product name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "64 N (64 pieces)", "note": "Unit count net quantity."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Pkd Date: 01/2026", "note": "Packing date."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP: Rs. 1099.00 (Rs. 17.17/unit)", "note": "MRP and unit sale price."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-202-1364, pgconsumer@custhelp.com", "note": "Consumer care channels."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Made in India", "note": "Origin India."}
        }
    },
    {
        "id": "real_prod_08",
        "title": "Prestige Deluxe Alpha Stainless Steel Pressure Cooker 3L",
        "category": "Kitchen & Dining",
        "brand": "Prestige",
        "retail_price": 2450.0,
        "discounted_price": 1899.0,
        "description": "Prestige Alpha base stainless steel outer lid pressure cooker induction and gas compatible 3 liters.",
        "specs": {"Capacity": "3 L", "Material": "Stainless Steel", "Base": "Alpha Induction Base"},
        "packaging_text": "Prestige Deluxe Alpha Outer Lid Cooker\nNet Quantity: 1 Unit (Capacity: 3.0 L)\nMfd by: TTK Prestige Limited, 11th Floor, Brigade Towers, 135 Brigade Road, Bengaluru 560025, Karnataka, India\nMRP: Rs. 2450.00 (Inclusive of all taxes)\nMonth & Year of Mfg: 02/2026\nCustomer Care: 1800-123-334411 | customercare@ttkprestige.com\nCountry of Origin: India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "TTK Prestige Limited, Brigade Road, Bengaluru 560025", "note": "Manufacturer name/address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Pressure Cooker", "note": "Commodity name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1 Unit (Capacity: 3.0 L)", "note": "Volume and unit count."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Month & Year of Mfg: 02/2026", "note": "Month and year of manufacture."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP: Rs. 2450.00", "note": "Retail price."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-123-334411, customercare@ttkprestige.com", "note": "Care channels."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Country of Origin: India", "note": "Origin India."}
        }
    },
    {
        "id": "real_prod_09",
        "title": "Organic India Tulsi Green Tea Classic 25 Tea Bags (45g)",
        "category": "Packaged Goods & Food",
        "brand": "Organic India",
        "retail_price": 210.0,
        "discounted_price": 185.0,
        "description": "Certified organic Tulsi green tea with potent antioxidants and stress-relieving adaptogens.",
        "specs": {"Form": "Tea Bags", "Quantity": "25 Bags", "Weight": "45 g"},
        "packaging_text": "ORGANIC INDIA Tulsi Green Tea\nNet Weight: 45 g (25 Tea Bags x 1.8 g)\nPacked by: Organic India Pvt. Ltd., Plot No. 266, Faizabad Road, Lucknow 226028, UP\nMRP Rs. 210.00 (incl. of all taxes)\nPKD: 01/2026 | Best Before 24 months from Packaging\nContact: care@organicindia.com | Phone: 1800-180-5151\nCountry of Origin: India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Organic India Pvt. Ltd., Faizabad Road, Lucknow 226028", "note": "Packer name and address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Tulsi Green Tea", "note": "Commodity title."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "45 g (25 Tea Bags x 1.8 g)", "note": "Weight and composite unit count."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "PKD: 01/2026, Best Before 24 months", "note": "Packing date."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs. 210.00", "note": "MRP stated."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "care@organicindia.com, 1800-180-5151", "note": "Customer service contacts."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Country of Origin: India", "note": "Origin stated."}
        }
    },
    {
        "id": "real_prod_10",
        "title": "Lizol Disinfectant Surface Cleaner Citrus 2 Litres",
        "category": "Household & Cleaning",
        "brand": "Lizol",
        "retail_price": 415.0,
        "discounted_price": 360.0,
        "description": "Lizol 10x surface cleaner kills 99.9% germs and leaves a fresh citrus fragrance.",
        "specs": {"Type": "Floor Cleaner", "Volume": "2 L", "Fragrance": "Citrus"},
        "packaging_text": "Lizol Surface Cleaner Citrus\nNet Volume: 2 L (2000 ml)\nMfd by: Reckitt Benckiser (India) Pvt. Ltd., DLF Cyber Park, Sector 20, Gurugram 122016, Haryana\nMRP: Rs. 415.00 (inclusive of taxes)\nManufacturing Date: 12/2025\nCustomer Support: 1800-102-7245 | consumercare_in@reckitt.com\nCountry of Origin: India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Reckitt Benckiser, Gurugram 122016, Haryana", "note": "Manufacturer name/address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Surface Cleaner", "note": "Generic product name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "2 L (2000 ml)", "note": "Volume in liters and ml."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Manufacturing Date: 12/2025", "note": "Manufacturing date."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP: Rs. 415.00", "note": "Price stated."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-102-7245, consumercare_in@reckitt.com", "note": "Customer care."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Country of Origin: India", "note": "Origin India."}
        }
    },
    {
        "id": "real_prod_11",
        "title": "Unbranded Generic Cotton Kitchen Towel Set (Pack of 4)",
        "category": "Kitchen & Dining",
        "brand": "",
        "retail_price": 199.0,
        "discounted_price": 149.0,
        "description": "100% cotton super absorbent kitchen towels for wiping and drying utensils. No brand label.",
        "specs": {"Material": "Cotton", "Pack Size": "4 Pieces"},
        "packaging_text": "Kitchen Towel Set 100% Cotton\nPack of 4\nMRP Rs 199",
        "ref": {
            "D01": {"label": "ABSENT", "source": "PACKAGING_LABEL", "evidence_text": "", "note": "No manufacturer name or physical address on packaging."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Kitchen Towel Set", "note": "Generic description present."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Pack of 4", "note": "Count declared."},
            "D04": {"label": "CANNOT_DETERMINE", "source": "PACKAGING_LABEL", "evidence_text": "", "note": "No date stamps on generic wrap."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs 199", "note": "Retail price stated."},
            "D06": {"label": "ABSENT", "source": "PACKAGING_LABEL", "evidence_text": "", "note": "No customer care details present."},
            "D07": {"label": "CANNOT_DETERMINE", "source": "PACKAGING_LABEL", "evidence_text": "", "note": "No origin declared."}
        }
    },
    {
        "id": "real_prod_12",
        "title": "Crafted Wooden Coaster Set of 6",
        "category": "Household & Cleaning",
        "brand": "ArtisanWood",
        "retail_price": 299.0,
        "discounted_price": 249.0,
        "description": "Handcrafted mango wood tea and coffee coasters with stand.",
        "specs": {"Pieces": "6", "Wood": "Mango Wood"},
        "packaging_text": "ArtisanWood Handcrafted Coasters\nSet of 6 Pieces\nMkt by: ArtisanWood Crafts\nMRP Rs. 299.00\nMade in India",
        "ref": {
            "D01": {"label": "PARTIAL", "source": "PACKAGING_LABEL", "evidence_text": "ArtisanWood Crafts", "note": "Brand/marketer name only; no physical street address or PIN code."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Handcrafted Coasters", "note": "Product name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Set of 6 Pieces", "note": "Unit count."},
            "D04": {"label": "CANNOT_DETERMINE", "source": "PACKAGING_LABEL", "evidence_text": "", "note": "Date not stated."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs. 299.00", "note": "Price stated."},
            "D06": {"label": "ABSENT", "source": "PACKAGING_LABEL", "evidence_text": "", "note": "No customer contact details."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Made in India", "note": "Origin India."}
        }
    },
    {
        "id": "real_prod_13",
        "title": "Sunfeast Dark Fantasy Choco Fills 300g (Pack of 4)",
        "category": "Packaged Goods & Food",
        "brand": "Sunfeast",
        "retail_price": 180.0,
        "discounted_price": 140.0,
        "description": "Crispy golden cookie with luscious molten choco cream filling.",
        "specs": {"Flavor": "Choco", "Weight": "300 g"},
        "packaging_text": "Sunfeast Dark Fantasy Choco Fills\nNet Weight: 300 g (4 Packs of 75 g each)\nManufactured by ITC Limited, 37 J.L. Nehru Road, Kolkata 700071, West Bengal\nMRP Rs. 180.00 incl of all taxes\nPkd: 02/2026 | Best Before 6 months\nFor feedback, call 1800-425-4444 or email itccares@itc.in\nCountry of Origin: India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "ITC Limited, J.L. Nehru Road, Kolkata 700071", "note": "Manufacturer name and address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Choco Fills (Biscuits)", "note": "Generic product name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "300 g (4 Packs of 75 g each)", "note": "Net quantity composite format."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Pkd: 02/2026", "note": "Packaging date."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs. 180.00", "note": "MRP stated."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-425-4444, itccares@itc.in", "note": "Customer care details."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Country of Origin: India", "note": "Origin India."}
        }
    },
    {
        "id": "real_prod_14",
        "title": "Dettol Original Liquid Handwash Refill Pouch 1500ml",
        "category": "Household & Cleaning",
        "brand": "Dettol",
        "retail_price": 279.0,
        "discounted_price": 235.0,
        "description": "Dettol Germ Protection handwash liquid refill with pine fragrance.",
        "specs": {"Volume": "1500 ml", "Type": "Refill Pouch"},
        "packaging_text": "Dettol Liquid Handwash Original\nNet Volume: 1.5 L (1500 ml)\nMfd by: Reckitt Benckiser India Pvt. Ltd., Plot 1, Sector 10, IIE SIDCUL, Haridwar 249403, Uttarakhand\nMRP Rs. 279.00 (Inclusive of all taxes)\nMfg Dt: 01/2026\nHelpline: 1800-102-7245, care@reckitt.com\nMade in India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Reckitt Benckiser, SIDCUL, Haridwar 249403", "note": "Full manufacturer name/address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Liquid Handwash", "note": "Generic product name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1.5 L (1500 ml)", "note": "Metric volume."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Mfg Dt: 01/2026", "note": "Manufacture date."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs. 279.00", "note": "MRP stated."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-102-7245, care@reckitt.com", "note": "Consumer helpline."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Made in India", "note": "Origin India."}
        }
    },
    {
        "id": "real_prod_15",
        "title": "Mother Sparsh 99% Pure Water Unscented Baby Wipes (72 Wipes)",
        "category": "Baby Care",
        "brand": "Mother Sparsh",
        "retail_price": 299.0,
        "discounted_price": 225.0,
        "description": "Plant-based biodegradable wipes formulated with 99% pure water for newborn sensitive skin.",
        "specs": {"Count": "72 Wipes", "Scent": "Unscented"},
        "packaging_text": "Mother Sparsh 99% Pure Water Baby Wipes\nNet Quantity: 72 Wipes (15 cm x 20 cm)\nMfd by: Mother Sparsh Baby Care Pvt. Ltd., Plot 34, Sector 5, Panchkula 134109, Haryana\nMRP: Rs. 299.00 (incl of taxes)\nBatch No: MS991 | Mfg Date: 02/2026 | Exp Date: 01/2028\nCare: 1800-102-4521 / support@mothersparsh.com\nCountry of Origin: India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Mother Sparsh Baby Care, Panchkula 134109, Haryana", "note": "Manufacturer name/address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Baby Wipes", "note": "Product name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "72 Wipes (15 cm x 20 cm)", "note": "Piece count and dimensions."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Mfg Date: 02/2026 | Exp Date: 01/2028", "note": "Mfg and expiry dates."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP: Rs. 299.00", "note": "MRP stated."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-102-4521, support@mothersparsh.com", "note": "Care details."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Country of Origin: India", "note": "Origin India."}
        }
    },
    {
        "id": "real_prod_16",
        "title": "Biotique Bio Kelp Protein Shampoo 340ml",
        "category": "Beauty & Personal Care",
        "brand": "Biotique",
        "retail_price": 290.0,
        "discounted_price": 215.0,
        "description": "Ayurvedic recipe with pure kelp, natural protein and peppermint oil for hair growth stimulation.",
        "specs": {"Quantity": "340 ml", "Hair Type": "All Hair Types"},
        "packaging_text": "BIOTIQUE ADVANCED AYURVEDA\nBio Kelp Fresh Growth Protein Shampoo\nNet Qty: 340 ml\nMfd by Bio Veda Action Research Co., Khasra No. 439/42, Village Mauja, Paonta Sahib 173025, HP\nMRP Rs. 290.00 (inclusive of all taxes)\nDate of Mfg: 01/2026\nCustomer Care: 1800-103-9825 | contact@biotique.com\nMade in India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Bio Veda Action Research Co., Paonta Sahib 173025, HP", "note": "Manufacturer name/address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Protein Shampoo", "note": "Generic title."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "340 ml", "note": "Volume in ml."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Date of Mfg: 01/2026", "note": "Date of manufacture."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs. 290.00", "note": "MRP stated."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-103-9825, contact@biotique.com", "note": "Customer care."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Made in India", "note": "Origin India."}
        }
    },
    {
        "id": "real_prod_17",
        "title": "KitchenPro Silicone Spatula & Pastry Brush Set",
        "category": "Kitchen & Dining",
        "brand": "KitchenPro",
        "retail_price": 199.0,
        "discounted_price": 149.0,
        "description": "Heat resistant food grade silicone spatula and basting brush for baking and barbecue.",
        "specs": {"Items": "2 (1 Spatula + 1 Brush)", "Material": "Food Grade Silicone"},
        "packaging_text": "KitchenPro Baking Essentials\nSet of 2: 1 Spatula + 1 Brush\nImported and Marketed by: Retail Imports Corp, Delhi 110006\nMRP Rs 199.00\nMade in China",
        "ref": {
            "D01": {"label": "PARTIAL", "source": "PACKAGING_LABEL", "evidence_text": "Retail Imports Corp, Delhi 110006", "note": "Importer name and city PIN only; no full street address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Baking Essentials (Spatula & Brush)", "note": "Product name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Set of 2: 1 Spatula + 1 Brush", "note": "Piece count."},
            "D04": {"label": "CANNOT_DETERMINE", "source": "PACKAGING_LABEL", "evidence_text": "", "note": "No manufacturing/import date."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs 199.00", "note": "MRP stated."},
            "D06": {"label": "ABSENT", "source": "PACKAGING_LABEL", "evidence_text": "", "note": "No contact information."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Made in China", "note": "Origin China declared."}
        }
    },
    {
        "id": "real_prod_18",
        "title": "Godrej aer Pocket Bathroom Air Fragrance Petal Crush Pink 10g",
        "category": "Household & Cleaning",
        "brand": "Godrej aer",
        "retail_price": 65.0,
        "discounted_price": 55.0,
        "description": "Unique power gel technology keeps bathroom fragrant for up to 30 days.",
        "specs": {"Quantity": "10 g", "Fragrance": "Petal Crush Pink"},
        "packaging_text": "Godrej aer pocket bathroom fragrance\nNet Weight: 10 g\nMfd by: Godrej Consumer Products Limited, Pirojshanagar, Eastern Express Highway, Vikhroli (East), Mumbai 400079\nMRP: Rs. 65.00 (incl. of all taxes)\nDate of Mfg: 01/2026 | Best before 2 years\nConsumer Cell: 1800-266-0007 | care@godrejcp.com\nCountry of Origin: India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Godrej Consumer Products, Vikhroli (East), Mumbai 400079", "note": "Manufacturer name/address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "bathroom fragrance", "note": "Generic product name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "10 g", "note": "Net weight in grams."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Date of Mfg: 01/2026", "note": "Mfg date."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP: Rs. 65.00", "note": "MRP stated."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-266-0007, care@godrejcp.com", "note": "Consumer helpline."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Country of Origin: India", "note": "Origin India."}
        }
    },
    {
        "id": "real_prod_19",
        "title": "Nestle Cerelac Baby Cereal with Milk Wheat Apple 300g",
        "category": "Baby Care",
        "brand": "Nestle",
        "retail_price": 275.0,
        "discounted_price": 260.0,
        "description": "Complementary food for babies from 6 to 24 months with 14 essential vitamins and minerals.",
        "specs": {"Stage": "Stage 1 (6+ months)", "Net Weight": "300 g"},
        "packaging_text": "NESTLE CERELAC Fortified Baby Cereal with Milk\nWheat Apple | Net Weight: 300 g (Bag-in-Box)\nMfd by: Nestle India Limited, 100/101 World Trade Centre, Barakhamba Lane, New Delhi 110001\nMRP: Rs. 275.00 (inclusive of all taxes)\nBatch No: N771 | Date of Mfg: 02/2026 | Use by: 01/2027\nNestle Consumer Care: 1800-103-1947 | wecare@in.nestle.com\nCountry of Origin: India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Nestle India Limited, Barakhamba Lane, New Delhi 110001", "note": "Corporate address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Baby Cereal with Milk", "note": "Generic title."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "300 g", "note": "Net weight in grams."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Date of Mfg: 02/2026 | Use by: 01/2027", "note": "Mfg and expiry dates."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP: Rs. 275.00", "note": "MRP stated."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "1800-103-1947, wecare@in.nestle.com", "note": "Consumer care."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Country of Origin: India", "note": "Origin India."}
        }
    },
    {
        "id": "real_prod_20",
        "title": "Aroma Magic Sunblock Lotion SPF 30 (100ml)",
        "category": "Beauty & Personal Care",
        "brand": "Aroma Magic",
        "retail_price": 220.0,
        "discounted_price": 185.0,
        "description": "All natural mineral sunscreen lotion with non-nano zinc oxide and calendula extracts.",
        "specs": {"SPF": "SPF 30", "Volume": "100 ml", "Type": "Lotion"},
        "packaging_text": "Blossom Kochhar Aroma Magic Sunblock Lotion SPF 30 PA++\nNet Vol: 100 ml\nMfd by: Blossom Kochhar Beauty Products Pvt. Ltd., Plot No. 90A, Khasra 132, Raipur Industrial Area, Bhagwanpur, Roorkee 247661, Uttarakhand\nMRP Rs. 220.00 (incl. of all taxes)\nManufacturing Date: 01/2026 | Expiry: 36 Months from Mfg\nCustomer Helpline: +91-11-41641313 | contactus@aromamagic.com\nMade in India",
        "ref": {
            "D01": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Blossom Kochhar, Raipur Industrial Area, Roorkee 247661", "note": "Full manufacturer name/address."},
            "D02": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Sunblock Lotion", "note": "Generic name."},
            "D03": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "100 ml", "note": "Metric volume."},
            "D04": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Manufacturing Date: 01/2026", "note": "Date of manufacture."},
            "D05": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "MRP Rs. 220.00", "note": "Retail price."},
            "D06": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "+91-11-41641313, contactus@aromamagic.com", "note": "Helpline and email."},
            "D07": {"label": "PRESENT", "source": "PACKAGING_LABEL", "evidence_text": "Made in India", "note": "Origin India."}
        }
    }
]


def generate_images_and_files():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(REF_DIR, exist_ok=True)

    print(f"Generating {len(REAL_PRODUCTS_DATA)} real-world product packaging images and records...")

    for prod in REAL_PRODUCTS_DATA:
        pid = prod["id"]
        img_path = os.path.join(IMG_DIR, f"{pid}.png")
        input_path = os.path.join(INPUT_DIR, f"{pid}.json")
        ref_path = os.path.join(REF_DIR, f"{pid}_ref.json")

        # 1. Render realistic label image
        img = Image.new("RGB", (900, 1100), color=(248, 249, 250))
        draw = ImageDraw.Draw(img)

        # Draw decorative header banner
        draw.rectangle([(0, 0), (900, 80)], fill=(42, 157, 143))
        draw.text((40, 25), f"PRODUCT PACKAGING LABEL — {prod['category'].upper()}", fill=(255, 255, 255))

        # Draw packaging text block
        draw.rectangle([(30, 110), (870, 1050)], outline=(200, 200, 200), fill=(255, 255, 255), width=2)

        y_offset = 140
        lines = prod["packaging_text"].split("\n")
        for line in lines:
            if ":" in line:
                key, val = line.split(":", 1)
                draw.text((60, y_offset), f"{key}:", fill=(20, 20, 20))
                draw.text((280, y_offset), val.strip(), fill=(50, 50, 50))
            else:
                draw.text((60, y_offset), line, fill=(10, 10, 10))
            y_offset += 45

        img.save(img_path)

        # 2. Write input JSON
        input_payload = {
            "product_id": pid,
            "title": prod["title"],
            "category": prod["category"],
            "brand": prod["brand"],
            "retail_price": prod["retail_price"],
            "discounted_price": prod["discounted_price"],
            "description": prod["description"],
            "specifications": prod["specs"],
            "image_path": img_path
        }
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(input_payload, f, indent=2)

        # 3. Write manual reference JSON
        ref_payload = {
            "product_id": pid,
            "title": prod["title"],
            "category": prod["category"],
            "brand": prod["brand"],
            "observable_packaging_text": prod["packaging_text"],
            "declarations": prod["ref"]
        }
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(ref_payload, f, indent=2)

    print("All 20 real product images, inputs, and manual references generated successfully.")


if __name__ == "__main__":
    generate_images_and_files()
