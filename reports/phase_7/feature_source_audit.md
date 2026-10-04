# Phase 7: Feature Source Audit & Classification Report

**Total Fields Audited:** 654 across 4 datasets  

## 1. Feature Role Classification Summary

| Feature Role | Field Count | Leakage Risk Description |
| :--- | :--- | :--- |
| **`METADATA`** | 590 | Auxiliary descriptors. |
| **`INDEPENDENT_INPUT`** | 34 | Clean observable attributes (text length, price, category, image counts). SAFE for ML. |
| **`IDENTIFIER`** | 13 | Unique IDs and group clustering keys. Used for splitting only. |
| **`PROVENANCE`** | 13 | Auditing and pipeline version metadata. EXCLUDED from X. |
| **`TARGET`** | 2 | Ground truth or weak label target columns (D01-D07, overall). EXCLUDED from X. |
| **`RULE_DERIVED`** | 2 | Rule engine scores and confidence. EXCLUDED from feature matrix X. |

## 2. Comprehensive Field Inventory

| Source | Field Name | Data Type | Missing % | Feature Role | Leakage Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FLIPKART_RAW_CSV | `uniq_id` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| FLIPKART_RAW_CSV | `crawl_timestamp` | str | 0.0% | `METADATA` | `CAUTION` |
| FLIPKART_RAW_CSV | `product_url` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| FLIPKART_RAW_CSV | `product_name` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| FLIPKART_RAW_CSV | `product_category_tree` | str | 0.0% | `METADATA` | `CAUTION` |
| FLIPKART_RAW_CSV | `pid` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| FLIPKART_RAW_CSV | `retail_price` | float64 | 0.6% | `INDEPENDENT_INPUT` | `NONE` |
| FLIPKART_RAW_CSV | `discounted_price` | float64 | 0.6% | `INDEPENDENT_INPUT` | `NONE` |
| FLIPKART_RAW_CSV | `image` | str | 0.0% | `METADATA` | `CAUTION` |
| FLIPKART_RAW_CSV | `is_FK_Advantage_product` | bool | 0.0% | `METADATA` | `CAUTION` |
| FLIPKART_RAW_CSV | `description` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| FLIPKART_RAW_CSV | `product_rating` | str | 0.0% | `METADATA` | `CAUTION` |
| FLIPKART_RAW_CSV | `overall_rating` | str | 0.0% | `METADATA` | `CAUTION` |
| FLIPKART_RAW_CSV | `brand` | str | 59.2% | `INDEPENDENT_INPUT` | `NONE` |
| FLIPKART_RAW_CSV | `product_specifications` | str | 0.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `product_id` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| PRODUCT_EVIDENCE_STREAM | `product_url` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| PRODUCT_EVIDENCE_STREAM | `pid` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| PRODUCT_EVIDENCE_STREAM | `title` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `raw_title` | str | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `description` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `raw_description` | str | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `brand` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `retail_price` | float64 | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `discounted_price` | float64 | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `category_hierarchy` | object | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `root_category` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `leaf_category` | str | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `extra_spec_values` | object | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `spec_parse_status` | str | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `image_urls` | object | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `accessible_images` | object | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `image_evidence_status` | str | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `evidence_sources` | object | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `extraction_timestamp` | str | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `extraction_status` | str | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `extraction_version` | str | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| PRODUCT_EVIDENCE_STREAM | `specifications.Pet Type` | str | 98.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Brand` | str | 71.66% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Quantity` | str | 96.81% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Model Number` | str | 78.44% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Type` | str | 30.14% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Fragrance` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Form Factor` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Sales Package` | str | 52.3% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `image_metadata.total_urls` | int64 | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `image_metadata.valid_url_count` | int64 | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `image_metadata.primary_domain` | str | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `image_metadata.download_policy` | str | 0.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Model Name` | str | 81.44% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Weight` | str | 64.87% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Paper Weight Material` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Paper Weight Finish` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Suitable For` | str | 97.41% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Activity Type` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Material` | str | 61.48% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Chewable` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Color` | str | 64.07% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Pack of` | str | 82.63% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Bonsai` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Pot Included` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Total Height` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Sleeve` | str | 90.42% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Contents in Sales Package` | str | 71.66% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Fabric` | str | 78.24% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Neck` | str | 90.62% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Pattern` | str | 79.44% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Occasion` | str | 74.05% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Ideal For` | str | 53.49% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Style Code` | str | 89.82% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Detection Angle` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Brand Color` | str | 92.81% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Wired/Wireless` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Display Type` | str | 99.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Installation Type` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Audible Alarm` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Maximum Alarm Distance` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Minimum Alarm Distance` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Detection Range` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Sensors` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Other Dimensions` | str | 98.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Service Type` | str | 98.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Other Features` | str | 96.81% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Technology Used` | str | 98.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.LED Indicators` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Stand By` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Certification` | str | 98.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Faders` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Application Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Channels` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Channel Equalizer Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Bus` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.On-board Effects` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Height` | str | 76.25% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Width` | str | 76.85% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Covered in Warranty` | str | 96.61% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Warranty Summary` | str | 96.01% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Not Covered in Warranty` | str | 96.81% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Power Consumption` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Power Requirements` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Face Material` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Inner Material` | str | 98.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Suitable for` | str | 98.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Designed For` | str | 99.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Model ID` | str | 96.01% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Length` | str | 96.61% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Machine Washable` | str | 99.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Design` | str | 92.61% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.GSM` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Size` | str | 96.41% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Dietary Preference` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Flavor` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Usage Timings` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Scoops per Container` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Container Type` | str | 98.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Ayurvedic` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Protein Type` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Form` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Serving Size` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Food Preference` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Composition` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Shelf Life` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Sole Material` | str | 95.21% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Closure` | str | 95.81% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Heel Height` | str | 93.01% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Outer Material` | str | 92.22% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Brand Fit` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Chronograph` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Altimeter` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Date Display` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Barometer` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Alarm Clock` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Compass` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Calendar` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Luminous` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Mechanism` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Series` | str | 94.01% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Novelty Feature` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Dial Shape` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Strap Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Strap Color` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Scratch Resistant` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Water Resistant` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Water Resistance Depth` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Clasp Type` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Other Body Features` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Dial Color` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Strap Material` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Lining` | str | 98.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Boot Shaft Height` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Tip Shape` | str | 97.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Style` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Other Details` | str | 98.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Care Instructions` | str | 98.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Diameter` | str | 85.23% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Sport Type` | str | 98.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Ideal for` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Designed for` | str | 98.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Depth` | str | 91.42% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Touch Screen Digitizer` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Compatible Model` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Color Code` | str | 98.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Right Hand Gloves` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Playing Level` | str | 97.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Left Hand Gloves` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Fingerless` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Full Fingered` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Age Group` | str | 97.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Three Fingered` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Fingers` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Belt Included` | str | 97.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Pockets` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Collar` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Bulb Base` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Light Color` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Bulb Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Lumen` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Racks & Trays Available` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Other Convenience Features` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Child Lock` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Other Performance Features` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Preheat` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Power Levels` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Defrost` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Timer` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Shade` | str | 96.41% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Frequency` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Control Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Capacity` | str | 92.81% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Turntable` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Turntable Material` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Cavity Material` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Demo Details` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Auto Indian Cook Menu` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Other Cooking Features` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Cooking Completion Indicator` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Auto Cook Menu` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Turntable Diameter` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Power Consumption - Convection` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Power Consumption - Microwave` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Power Requirement` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Power Consumption - Grill` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Power Output` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Other Power Features` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Cable Length` | str | 99.8% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Model` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.LED Indicator` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Connector` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Warranty Service Type` | str | 98.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Hair Condition` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Hair Type` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Applied For` | str | 98.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Mug Capacity` | str | 79.44% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Freezer Safe` | str | 83.83% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Microwave Safe` | str | 78.84% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Replaceable Refill` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Marker Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Tip Design` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Organic Type` | str | 98.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Application Area` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Skin Type` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Treatment Form` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Compartment Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Thumb` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Ventilation` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Palm` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Other Glove Details` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Casserole Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Lid Included` | str | 97.01% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Non-stick Coating` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Mirror Width` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Mirror Height` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Purpose` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Body Material` | str | 98.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Handle/Strap Features` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Contents in Combo Set` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Combo Set Contents` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Organic` | str | 98.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Professional Care` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Configuration` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Power Source` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Impedance` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Total Power Output RMS (W)` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Frequency Response` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Memory Card Slot` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Fit` | str | 98.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Insulated` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Washable` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Trolley Support` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Laptop Sleeve` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Compatible Laptop Size` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Compartments` | str | 98.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Seating Capacity` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Model Id` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Power Required (Volts)` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Power Consumption (Watts)` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Lunch Box Material` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Containers` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Lunch Box Capacity` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Net Weight` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.W x H x D` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Placement Position` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Dishwasher Safe` | str | 95.61% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Break Resistant` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Cup Diameter` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Cup Height` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Saucer Width` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Saucer Diameter` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Belt` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Rise` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Blade Material` | str | 96.41% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Handle Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Foldable` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Assembly Required` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Weight Supported` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Wheeled Bottom` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Brakes` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Height Adjustment` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Knit Type` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Primary Product Size` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Usage Type` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Cups` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Card Slots` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Pleated` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Waistband` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Headset Design` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Compatible Devices` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Type of Headset` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.In Sales Package` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Call Controls` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Bluetooth` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Noise Cancellation Headphones` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Connector Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Cord Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Noise Cancellation Mircrophone` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Bag Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Waterproof` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Bag Capacity` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Bag Design` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Application Frequency` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Contents in Kit` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Design Code` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Buttonhole Styles` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Sewing Speed` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Stitches` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Cordless` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Temperature Control` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Self Cleaning` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Spray` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Indicator Light` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Automatic shut-off` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Steam Burst` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Soleplate type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Lockable` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Professional Use` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Functions` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Finish` | str | 99.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Battery Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Usage Instructions` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Wall Mount` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Hand Washable` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Collection` | str | 98.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Precious/Artificial Jewellery` | str | 97.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Diamond Weight` | str | 99.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Diamond Color Grade` | str | 99.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Diamond Clarity` | str | 99.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Diamond Color` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Gold Purity` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Gold Weight` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Metal Weight` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Base Material` | str | 97.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Gemstone` | str | 98.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Ring Size` | str | 99.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Setting` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Character` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Design & Style` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Closure Type` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Width x Height x Depth` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Additional Features` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Digital Display` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Corded/Cordless` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Rechargeable` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Battery Size` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Recharging Dock` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Cleaning and Care` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Kurta Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Salwar Fabric` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Kurta Fabric` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.color` | str | 94.41% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Wire Support` | str | 94.41% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Straps` | str | 94.41% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Cup Type` | str | 94.61% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Seam Type` | str | 97.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Detachable Straps` | str | 96.81% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Inner Lining` | str | 97.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Other Bra Details` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Fastening` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Back` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Frame Material` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Other Goggle Features` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Frame Color` | str | 99.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Sign Type` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Text on Sign` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Glow In The Dark` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Reflective Sign` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Flexible` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Double Sided` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Shape` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Lens Color` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.UV Protection` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Product Weight` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Product Depth` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Product Width` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Country of Manufacture` | str | 99.6% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Contents` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Product Height` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.External Material` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Belt Loops` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Fly` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Attachment Mechanism` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.UV Ray Protection` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Reflects Heat` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Driver Visibility` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Vehicle Model Name` | str | 97.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Plating` | str | 99.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Vehicle Brand` | str | 97.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Vehicle Model Year` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Corrosion Resistant` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Set Contents` | str | 98.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Sheets` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Sticky Note Colors` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Bag Style` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Reusable` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Maximum Altitude` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Light Source` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number Of Lights` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Usage` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Frame material` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Lens Material` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Frame Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.character` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Pearl Type` | str | 99.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Silver Weight` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Diamond Shape` | str | 99.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Silver Purity` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Diamond Height` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Natural/Synthetic Diamond` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Diamond Width` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Gold Color` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Gemstones` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Silver Color` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Metal Color` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Contents in Set` | str | 94.81% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Display` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Used for` | str | 98.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Handle Material` | str | 96.61% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Thumb Guard` | str | 96.61% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Grip Type` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Child Safe` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Model Year` | str | 97.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Necklace Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Chain Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Pan Type` | str | 95.61% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Non-stick` | str | 94.01% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Induction Bottom` | str | 94.21% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Audience Size` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Microphone Included` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Portable` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Maximum Power Output` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Sound Mixer Channel` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Memory Card Slot Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Microphone Input` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Wireless` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.USB Port` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Setting Controls` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Maximum Frequency Response` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Minimum Frequency Response` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Speaker Form Factor` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Limit Indicator` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Maximum Sound Output` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Speaker Mount` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Handle Features` | str | 98.2% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Base Thickness` | str | 98.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Dupatta Fabric` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Top Fabric` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Bottom Fabric` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Top Length` | str | 99.8% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Top Width` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Apparel Material` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Embellished` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Filter Thread Size` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Mount Type` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Covered In Warranty` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Not Covered In Warranty` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Brush Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Pieces per Set` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Set Content` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Water Dispenser` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Water & Ice Dispenser` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Door Lock` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Installation & Demo` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Net Total Capacity` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Star Rating` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Refrigerator Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Defrosting Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Doors` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Shelf Material` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Door Finish` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Refrigerator Door Bins` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Number of Refrigerator Shelves` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Egg Tray` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Net Height` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Net Depth` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Net Width` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Freezer Tray Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Ice Cube Tray Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Stabilizer Required` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Usage Time` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Charge Time` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Charging Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Nicotine Strength` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Tip Type` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Activation Type` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.LED Light Color` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Puff Count` | str | 99.6% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Certifications` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Locking` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Oven Safe` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Broiler Safe` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Cleaning` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Body Weight` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Body Mass Index (BMI)` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Body Fat Percentage` | str | 99.8% | `INDEPENDENT_INPUT` | `NONE` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Basal Metabolic Rate (BMR)` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Memory` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Maximum Weight Capacity` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Powered By` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Shoe Size` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Wheel Size` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Frame/Chassis` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Cuff` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Bearings` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Combo Set Content` | str | 95.41% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Fragrance Segment` | str | 98.0% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Gift Set Content` | str | 99.4% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Pencil Box Art` | str | 97.21% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Pencil Box Material` | str | 97.21% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Integrated Sharpener` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Pencil Box Closure Type` | str | 99.6% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Universal Fit` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Compatible With` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Breathable` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Disposable` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Wireless Speed` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.In The Box` | str | 99.8% | `METADATA` | `CAUTION` |
| PRODUCT_EVIDENCE_STREAM | `specifications.Antennae` | str | 99.8% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `product_id` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| WEAK_LABEL_DATASET_V1 | `product_url` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| WEAK_LABEL_DATASET_V1 | `pid` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| WEAK_LABEL_DATASET_V1 | `product_name` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| WEAK_LABEL_DATASET_V1 | `category` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| WEAK_LABEL_DATASET_V1 | `category_hierarchy` | object | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| WEAK_LABEL_DATASET_V1 | `brand` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| WEAK_LABEL_DATASET_V1 | `retail_price` | float64 | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| WEAK_LABEL_DATASET_V1 | `discounted_price` | float64 | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| WEAK_LABEL_DATASET_V1 | `image_urls` | object | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| WEAK_LABEL_DATASET_V1 | `image_evidence_status` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `overall_label` | str | 0.0% | `TARGET` | **`HIGH_TARGET_LEAKAGE`** |
| WEAK_LABEL_DATASET_V1 | `overall_confidence` | float64 | 0.0% | `RULE_DERIVED` | **`PROVENANCE_LEAKAGE`** |
| WEAK_LABEL_DATASET_V1 | `overall_reason` | str | 0.0% | `RULE_DERIVED` | **`PROVENANCE_LEAKAGE`** |
| WEAK_LABEL_DATASET_V1 | `quality_tier` | str | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| WEAK_LABEL_DATASET_V1 | `label_type` | str | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| WEAK_LABEL_DATASET_V1 | `provenance` | str | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| WEAK_LABEL_DATASET_V1 | `rule_engine_version` | str | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| WEAK_LABEL_DATASET_V1 | `evidence_version` | str | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| WEAK_LABEL_DATASET_V1 | `extraction_version` | str | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| WEAK_LABEL_DATASET_V1 | `generation_timestamp` | str | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| WEAK_LABEL_DATASET_V1 | `duplicate_group_id` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| WEAK_LABEL_DATASET_V1 | `product_family_id` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| WEAK_LABEL_DATASET_V1 | `normalized_family_title` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D01.label` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D01.confidence` | float64 | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D01.evidence_nature` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D01.evidence_sources` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D01.evidence_text` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D01.rule_ids` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D01.label_type` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D01.extraction_method` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D02.label` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D02.confidence` | float64 | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D02.evidence_nature` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D02.evidence_sources` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D02.evidence_text` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D02.rule_ids` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D02.label_type` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D02.extraction_method` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D03.label` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D03.confidence` | float64 | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D03.evidence_nature` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D03.evidence_sources` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D03.evidence_text` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D03.rule_ids` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D03.label_type` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D03.extraction_method` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D04.label` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D04.confidence` | float64 | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D04.evidence_nature` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D04.evidence_sources` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D04.evidence_text` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D04.rule_ids` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D04.label_type` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D04.extraction_method` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D05.label` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D05.confidence` | float64 | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D05.evidence_nature` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D05.evidence_sources` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D05.evidence_text` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D05.rule_ids` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D05.label_type` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D05.extraction_method` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D06.label` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D06.confidence` | float64 | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D06.evidence_nature` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D06.evidence_sources` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D06.evidence_text` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D06.rule_ids` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D06.label_type` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D06.extraction_method` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D07.label` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D07.confidence` | float64 | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D07.evidence_nature` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D07.evidence_sources` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D07.evidence_text` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D07.rule_ids` | object | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D07.label_type` | str | 0.0% | `METADATA` | `CAUTION` |
| WEAK_LABEL_DATASET_V1 | `declarations.D07.extraction_method` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `product_id` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| HUMAN_GROUND_TRUTH_PILOT | `product_name` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| HUMAN_GROUND_TRUTH_PILOT | `product_url` | str | 0.0% | `IDENTIFIER` | `SPLIT_GROUPING_KEY` |
| HUMAN_GROUND_TRUTH_PILOT | `category` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| HUMAN_GROUND_TRUTH_PILOT | `category_tree` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `brand` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| HUMAN_GROUND_TRUTH_PILOT | `retail_price` | float64 | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| HUMAN_GROUND_TRUTH_PILOT | `discounted_price` | float64 | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| HUMAN_GROUND_TRUTH_PILOT | `image_urls` | str | 0.0% | `INDEPENDENT_INPUT` | `NONE` |
| HUMAN_GROUND_TRUTH_PILOT | `overall_label` | str | 0.0% | `TARGET` | **`HIGH_TARGET_LEAKAGE`** |
| HUMAN_GROUND_TRUTH_PILOT | `provenance` | str | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| HUMAN_GROUND_TRUTH_PILOT | `human_confirmed` | bool | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| HUMAN_GROUND_TRUTH_PILOT | `reviewed_by_human` | bool | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| HUMAN_GROUND_TRUTH_PILOT | `reviewer_id` | str | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| HUMAN_GROUND_TRUTH_PILOT | `finalized_timestamp` | str | 0.0% | `PROVENANCE` | **`PROVENANCE_LEAKAGE`** |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D01.label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D01.evidence_source` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D01.evidence_notes` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D01.review_action` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D01.original_label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D02.label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D02.evidence_source` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D02.evidence_notes` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D02.review_action` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D02.original_label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D03.label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D03.evidence_source` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D03.evidence_notes` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D03.review_action` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D03.original_label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D04.label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D04.evidence_source` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D04.evidence_notes` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D04.review_action` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D04.original_label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D05.label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D05.evidence_source` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D05.evidence_notes` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D05.review_action` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D05.original_label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D06.label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D06.evidence_source` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D06.evidence_notes` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D06.review_action` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D06.original_label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D07.label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D07.evidence_source` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D07.evidence_notes` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D07.review_action` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `declarations.D07.original_label` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `audit.reviewer_decision` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `audit.review_reason` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `audit.reviewer_notes` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `audit.review_confidence` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `audit.overall_review_action` | str | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `audit.ai_draft_metadata.draft_generated` | bool | 0.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `audit.ai_draft_metadata.model_name` | str | 52.0% | `METADATA` | `CAUTION` |
| HUMAN_GROUND_TRUTH_PILOT | `audit.ai_draft_metadata.draft_timestamp` | str | 52.0% | `METADATA` | `CAUTION` |

