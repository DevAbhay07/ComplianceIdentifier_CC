"""
evidence_fusion.py
==================
Multi-image evidence fusion engine for physical product compliance screening.
Fuses declaration evidence from multiple photographs/views (front, back, sides) of a single product
into ONE unified, conflict-aware product-level evidence representation.

CRITICAL RULES:
1. The product is the unit of inference (not individual images).
2. Evidence union across views (e.g. front has MRP, back has Manufacturer).
3. Corroboration vs Conflict:
   - Multiple observations of the SAME normalized value (e.g. '250 g', '250g', '250 gm', '0.25 kg')
     are corroborating evidence and MUST produce PRESENT, NOT a conflict.
   - Genuine conflicts (e.g. 250 g vs 200 g, ₹999 vs ₹1099) produce CONFLICTING_EVIDENCE.
4. Non-readable views (e.g. back view lacks MRP) must NOT create a conflict with readable views.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from src.extraction.declaration_schema import (
    DeclarationEvidence,
    DeclarationStatus,
    EvidenceSource,
    ProductEvidenceRecord,
    DECLARATION_DEFINITIONS
)
from src.extraction.normalization import (
    normalize_quantity_value, are_quantities_conflicting,
    normalize_price_value, are_prices_conflicting,
    normalize_date_value, are_dates_conflicting,
    normalize_country_value, are_countries_conflicting,
    normalize_manufacturer_value, are_manufacturers_conflicting,
    normalize_consumer_care_value, are_consumer_care_conflicting,
    normalize_generic_name_value, are_generic_names_conflicting
)


class EvidenceFusionEngine:
    """
    Fuses multiple image-level declaration evidence sets into a single unified product-level evidence record.
    """

    def fuse_image_evidences(
        self,
        image_evidences: List[Dict[str, Any]],
        product_id: str = "product_01",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProductEvidenceRecord:
        """
        Takes a list of image-level evidence dictionaries:
        [
            {
                "image_name": "front.png",
                "declarations": Dict[str, DeclarationEvidence],
                "ocr_text": str,
                "ocr_confidence": float,
                "visual_regions": List[Dict[str, Any]]
            },
            ...
        ]
        and returns a unified ProductEvidenceRecord.
        """
        if not image_evidences:
            # Empty fallback
            empty_decls = {}
            for d_id, meta in DECLARATION_DEFINITIONS.items():
                empty_decls[d_id] = DeclarationEvidence(
                    declaration_id=d_id,
                    declaration_name=meta["name"],
                    status=DeclarationStatus.CANNOT_DETERMINE.value,
                    evidence_text="No images provided for analysis."
                )
            return ProductEvidenceRecord(
                product_id=product_id,
                image_count=0,
                source_images=[],
                declarations=empty_decls,
                overall_ocr_confidence=0.0,
                image_quality=0.0,
                metadata=metadata or {}
            )

        source_images = [img_item.get("image_name", f"image_{i+1}") for i, img_item in enumerate(image_evidences)]
        raw_ocr_texts = {img_item.get("image_name", f"image_{i+1}"): img_item.get("ocr_text", "") for i, img_item in enumerate(image_evidences)}
        visual_detections = {img_item.get("image_name", f"image_{i+1}"): img_item.get("visual_regions", []) for i, img_item in enumerate(image_evidences)}
        
        # Calculate mean OCR confidence
        ocr_confs = [img_item.get("ocr_confidence", 0.90) for img_item in image_evidences]
        mean_ocr_conf = float(sum(ocr_confs) / len(ocr_confs)) if ocr_confs else 0.0

        fused_declarations: Dict[str, DeclarationEvidence] = {}

        for d_id in ["D01", "D02", "D03", "D04", "D05", "D06", "D07"]:
            d_name = DECLARATION_DEFINITIONS[d_id]["name"]
            
            # Collect all evidence items for this declaration across all images
            candidates: List[DeclarationEvidence] = []
            for img_item in image_evidences:
                decl_dict = img_item.get("declarations", {})
                if d_id in decl_dict:
                    ev = decl_dict[d_id]
                    if isinstance(ev, dict):
                        ev = DeclarationEvidence.from_dict(ev)
                    candidates.append(ev)

            # Separate positive vs absent vs cannot_determine candidates
            present_candidates = [c for c in candidates if c.status in [DeclarationStatus.PRESENT.value, "PRESENT"]]
            absent_candidates = [c for c in candidates if c.status in [DeclarationStatus.ABSENT.value, "ABSENT"]]
            uncertain_candidates = [c for c in candidates if c.status in [DeclarationStatus.CANNOT_DETERMINE.value, "CANNOT_DETERMINE"]]

            if present_candidates:
                # Check for genuine value conflicts across images
                conflict_detected, unique_norm_values, observations, conflict_items = self._check_declaration_conflicts(d_id, present_candidates)

                if conflict_detected:
                    # Genuine conflict between differing values across views
                    fused_declarations[d_id] = DeclarationEvidence(
                        declaration_id=d_id,
                        declaration_name=d_name,
                        status=DeclarationStatus.CONFLICTING_EVIDENCE.value,
                        raw_value=" | ".join([f"[{c.source_image}]: {c.raw_value}" for c in present_candidates]),
                        normalized_value={
                            "conflicting_values": conflict_items,
                            "unique_normalized_values": unique_norm_values,
                            "observations": observations
                        },
                        source=EvidenceSource.MULTI_IMAGE_FUSION.value,
                        source_image=",".join([c.source_image for c in present_candidates]),
                        ocr_confidence=mean_ocr_conf,
                        extraction_confidence=0.60,
                        evidence_text=f"Conflicting values detected across images: {' vs '.join([str(item.get('value', item.get('raw', ''))) for item in conflict_items])}",
                        conflicting_values=conflict_items
                    )
                else:
                    # Unify / merge positive corroborating evidence
                    best_cand = max(present_candidates, key=lambda c: c.extraction_confidence)
                    merged_norm = self._merge_normalized_values(d_id, present_candidates)
                    merged_norm["observations"] = observations
                    merged_norm["unique_normalized_values"] = unique_norm_values
                    images_with_ev = list(dict.fromkeys([c.source_image for c in present_candidates]))

                    corrob_text = f" (Corroborated across {len(images_with_ev)} views)" if len(images_with_ev) > 1 else ""
                    fused_declarations[d_id] = DeclarationEvidence(
                        declaration_id=d_id,
                        declaration_name=d_name,
                        status=DeclarationStatus.PRESENT.value,
                        raw_value=best_cand.raw_value,
                        normalized_value=merged_norm,
                        source=EvidenceSource.MULTI_IMAGE_FUSION.value if len(present_candidates) > 1 else best_cand.source,
                        source_image=",".join(images_with_ev),
                        source_region=best_cand.source_region,
                        bounding_box=best_cand.bounding_box,
                        ocr_confidence=max(c.ocr_confidence for c in present_candidates),
                        extraction_confidence=min(1.0, sum(c.extraction_confidence for c in present_candidates) / len(present_candidates) + 0.02 * (len(present_candidates) - 1)),
                        evidence_text=f"{best_cand.evidence_text}{corrob_text}"
                    )

            elif absent_candidates and len(absent_candidates) == len(candidates):
                # All images unanimously confirm absence with good OCR
                best_absent = max(absent_candidates, key=lambda c: c.extraction_confidence)
                fused_declarations[d_id] = DeclarationEvidence(
                    declaration_id=d_id,
                    declaration_name=d_name,
                    status=DeclarationStatus.ABSENT.value,
                    raw_value="",
                    normalized_value={},
                    source=EvidenceSource.MULTI_IMAGE_FUSION.value,
                    source_image=",".join(source_images),
                    ocr_confidence=mean_ocr_conf,
                    extraction_confidence=best_absent.extraction_confidence,
                    evidence_text=f"Declaration '{d_name}' ({d_id}) not found across any of the {len(source_images)} packaging views."
                )
            else:
                # Indeterminate or insufficient optical quality across uploaded views
                fused_declarations[d_id] = DeclarationEvidence(
                    declaration_id=d_id,
                    declaration_name=d_name,
                    status=DeclarationStatus.CANNOT_DETERMINE.value,
                    raw_value="",
                    normalized_value={},
                    source=EvidenceSource.MULTI_IMAGE_FUSION.value,
                    source_image=",".join(source_images),
                    ocr_confidence=mean_ocr_conf,
                    extraction_confidence=0.50,
                    evidence_text=f"Insufficient optical clarity across uploaded views to confirm '{d_name}' ({d_id})."
                )

        return ProductEvidenceRecord(
            product_id=product_id,
            image_count=len(image_evidences),
            source_images=source_images,
            declarations=fused_declarations,
            overall_ocr_confidence=mean_ocr_conf,
            image_quality=1.0,
            raw_ocr_texts=raw_ocr_texts,
            visual_detections=visual_detections,
            metadata=metadata or {}
        )

    def _check_declaration_conflicts(
        self,
        declaration_id: str,
        candidates: List[DeclarationEvidence]
    ) -> Tuple[bool, List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Checks if candidate values for the same declaration across images have irreconcilable conflicts.
        Distinguishes corroborating observations (same normalized value) from true conflicts.
        
        Returns:
            (conflict_detected, unique_normalized_values, observations, conflict_items)
        """
        observations = []
        for c in candidates:
            obs_display = (
                c.normalized_value.get("canonical_display")
                or c.normalized_value.get("formatted")
                or c.normalized_value.get("canonical_key")
                or c.raw_value
            )
            observations.append({
                "value": obs_display,
                "source": c.source_image,
                "raw": c.raw_value,
                "confidence": c.extraction_confidence,
                "normalized": c.normalized_value
            })

        if len(candidates) <= 1:
            unique_vals = list(dict.fromkeys([obs["value"] for obs in observations if obs["value"]]))
            return False, unique_vals, observations, []

        conflict_items = []
        unique_norm_values = []

        if declaration_id == "D03":  # Net Quantity
            norm_qtys = []
            for c in candidates:
                q_dict = c.normalized_value or {}
                if "base_value" not in q_dict:
                    q_num = q_dict.get("quantity")
                    u_str = q_dict.get("unit")
                    if q_num is not None:
                        q_dict = normalize_quantity_value(q_num, u_str)
                    elif c.raw_value:
                        q_dict = normalize_quantity_value(c.raw_value)
                if q_dict.get("base_value", 0) > 0:
                    norm_qtys.append((c, q_dict))
            
            unique_keys = set(q_dict.get("canonical_key", f"{q_dict.get('quantity')} {q_dict.get('unit')}") for _, q_dict in norm_qtys)
            unique_norm_values = sorted(list(unique_keys))

            conflicts_found = False
            for i in range(len(norm_qtys)):
                for j in range(i + 1, len(norm_qtys)):
                    c1, q1 = norm_qtys[i]
                    c2, q2 = norm_qtys[j]
                    if are_quantities_conflicting(q1, q2):
                        conflicts_found = True
                        c_dict1 = {"image": c1.source_image, "value": q1.get("canonical_display", q1.get("formatted")), "raw": c1.raw_value}
                        c_dict2 = {"image": c2.source_image, "value": q2.get("canonical_display", q2.get("formatted")), "raw": c2.raw_value}
                        if c_dict1 not in conflict_items:
                            conflict_items.append(c_dict1)
                        if c_dict2 not in conflict_items:
                            conflict_items.append(c_dict2)

            return conflicts_found, unique_norm_values, observations, conflict_items

        elif declaration_id == "D05":  # MRP
            norm_prices = []
            for c in candidates:
                p_dict = c.normalized_value or {}
                if "mrp" not in p_dict and c.raw_value:
                    p_dict = normalize_price_value(c.raw_value)
                if p_dict.get("mrp", 0) > 0:
                    norm_prices.append((c, p_dict))
            
            unique_keys = set(p_dict.get("canonical_key", f"₹{p_dict.get('mrp', 0):.2f}") for _, p_dict in norm_prices)
            unique_norm_values = sorted(list(unique_keys))
            
            conflicts_found = False
            for i in range(len(norm_prices)):
                for j in range(i + 1, len(norm_prices)):
                    c1, p1 = norm_prices[i]
                    c2, p2 = norm_prices[j]
                    if are_prices_conflicting(p1, p2):
                        conflicts_found = True
                        c_dict1 = {"image": c1.source_image, "value": p1.get("canonical_key"), "raw": c1.raw_value}
                        c_dict2 = {"image": c2.source_image, "value": p2.get("canonical_key"), "raw": c2.raw_value}
                        if c_dict1 not in conflict_items:
                            conflict_items.append(c_dict1)
                        if c_dict2 not in conflict_items:
                            conflict_items.append(c_dict2)

            return conflicts_found, unique_norm_values, observations, conflict_items

        elif declaration_id == "D04":  # Date
            norm_dates = []
            for c in candidates:
                d_dict = c.normalized_value or {}
                if ("month" not in d_dict or "year" not in d_dict) and c.raw_value:
                    # fallback
                    d_dict = normalize_date_value(1, 2026, c.raw_value)
                if "month" in d_dict and "year" in d_dict:
                    norm_dates.append((c, d_dict))
            
            unique_keys = set(d_dict.get("canonical_key", f"{d_dict.get('month'):02d}/{d_dict.get('year')}") for _, d_dict in norm_dates)
            unique_norm_values = sorted(list(unique_keys))
            
            conflicts_found = False
            for i in range(len(norm_dates)):
                for j in range(i + 1, len(norm_dates)):
                    c1, d1 = norm_dates[i]
                    c2, d2 = norm_dates[j]
                    if are_dates_conflicting(d1, d2):
                        conflicts_found = True
                        c_dict1 = {"image": c1.source_image, "value": d1.get("canonical_key"), "raw": c1.raw_value}
                        c_dict2 = {"image": c2.source_image, "value": d2.get("canonical_key"), "raw": c2.raw_value}
                        if c_dict1 not in conflict_items:
                            conflict_items.append(c_dict1)
                        if c_dict2 not in conflict_items:
                            conflict_items.append(c_dict2)

            return conflicts_found, unique_norm_values, observations, conflict_items

        elif declaration_id == "D07":  # Country of Origin
            norm_countries = []
            for c in candidates:
                c_dict = c.normalized_value or {}
                if "canonical_country" not in c_dict and c.raw_value:
                    c_dict = normalize_country_value(c.raw_value)
                if c_dict.get("canonical_country"):
                    norm_countries.append((c, c_dict))
            
            unique_keys = set(c_dict.get("canonical_country", c_dict.get("country", "")).lower() for _, c_dict in norm_countries)
            unique_norm_values = sorted(list(unique_keys))
            
            conflicts_found = False
            for i in range(len(norm_countries)):
                for j in range(i + 1, len(norm_countries)):
                    c1, cn1 = norm_countries[i]
                    c2, cn2 = norm_countries[j]
                    if are_countries_conflicting(cn1, cn2):
                        conflicts_found = True
                        c_dict1 = {"image": c1.source_image, "value": cn1.get("country"), "raw": c1.raw_value}
                        c_dict2 = {"image": c2.source_image, "value": cn2.get("country"), "raw": c2.raw_value}
                        if c_dict1 not in conflict_items:
                            conflict_items.append(c_dict1)
                        if c_dict2 not in conflict_items:
                            conflict_items.append(c_dict2)

            return conflicts_found, unique_norm_values, observations, conflict_items

        elif declaration_id == "D01":  # Manufacturer
            norm_mfgs = [(c, c.normalized_value) for c in candidates if c.normalized_value]
            unique_keys = set(m.get("canonical_key", m.get("entity_name", "")).lower() for _, m in norm_mfgs if m.get("entity_name"))
            unique_norm_values = sorted(list(unique_keys))
            conflicts_found = False
            for i in range(len(norm_mfgs)):
                for j in range(i + 1, len(norm_mfgs)):
                    c1, m1 = norm_mfgs[i]
                    c2, m2 = norm_mfgs[j]
                    if are_manufacturers_conflicting(m1, m2):
                        conflicts_found = True
                        c_dict1 = {"image": c1.source_image, "value": m1.get("entity_name"), "raw": c1.raw_value}
                        c_dict2 = {"image": c2.source_image, "value": m2.get("entity_name"), "raw": c2.raw_value}
                        if c_dict1 not in conflict_items:
                            conflict_items.append(c_dict1)
                        if c_dict2 not in conflict_items:
                            conflict_items.append(c_dict2)

            return conflicts_found, unique_norm_values, observations, conflict_items

        elif declaration_id == "D06":  # Consumer Care
            norm_cares = [(c, c.normalized_value) for c in candidates if c.normalized_value]
            unique_keys = set(cr.get("canonical_key", "") for _, cr in norm_cares if cr.get("canonical_key"))
            unique_norm_values = sorted(list(unique_keys))
            conflicts_found = False
            for i in range(len(norm_cares)):
                for j in range(i + 1, len(norm_cares)):
                    c1, cr1 = norm_cares[i]
                    c2, cr2 = norm_cares[j]
                    if are_consumer_care_conflicting(cr1, cr2):
                        conflicts_found = True
                        c_dict1 = {"image": c1.source_image, "value": cr1.get("formatted"), "raw": c1.raw_value}
                        c_dict2 = {"image": c2.source_image, "value": cr2.get("formatted"), "raw": c2.raw_value}
                        if c_dict1 not in conflict_items:
                            conflict_items.append(c_dict1)
                        if c_dict2 not in conflict_items:
                            conflict_items.append(c_dict2)

            return conflicts_found, unique_norm_values, observations, conflict_items

        elif declaration_id == "D02":  # Generic Name
            norm_names = [(c, c.normalized_value) for c in candidates if c.normalized_value]
            unique_keys = set(n.get("canonical_key", "") for _, n in norm_names if n.get("canonical_key"))
            unique_norm_values = sorted(list(unique_keys))
            return False, unique_norm_values, observations, []

        unique_vals = list(dict.fromkeys([obs["value"] for obs in observations if obs["value"]]))
        return False, unique_vals, observations, []

    def _merge_normalized_values(
        self,
        declaration_id: str,
        candidates: List[DeclarationEvidence]
    ) -> Dict[str, Any]:
        """
        Merges dictionary representations across multiple matching images,
        preserving role semantics, numerical values, and non-empty metadata.
        """
        if not candidates:
            return {}

        if declaration_id == "D01":
            # Specialized D01 Role Merging
            merged_roles = {}
            merged_entities = []
            seen_entity_keys = set()
            merged_addrs = []
            is_mkt = False
            is_pck = False
            is_imp = False
            primary_name = ""
            pin = ""

            for c in candidates:
                norm = c.normalized_value or {}
                if norm.get("is_marketer"):
                    is_mkt = True
                if norm.get("is_packer"):
                    is_pck = True
                if norm.get("is_importer"):
                    is_imp = True
                if norm.get("pincode") and not pin:
                    pin = str(norm["pincode"])
                if norm.get("entity_name") and not primary_name:
                    primary_name = str(norm["entity_name"])

                for addr in norm.get("address_markers", []):
                    if addr and addr not in merged_addrs:
                        merged_addrs.append(addr)

                c_roles = norm.get("roles", {})
                for r_key, r_val in c_roles.items():
                    if r_key not in merged_roles or not merged_roles[r_key].get("name"):
                        merged_roles[r_key] = r_val
                    else:
                        # Combine address markers for this role
                        existing_addrs = merged_roles[r_key].get("address_markers", [])
                        new_addrs = r_val.get("address_markers", [])
                        comb_addrs = list(dict.fromkeys(existing_addrs + new_addrs))
                        merged_roles[r_key]["address_markers"] = comb_addrs
                        if not merged_roles[r_key].get("pincode") and r_val.get("pincode"):
                            merged_roles[r_key]["pincode"] = r_val["pincode"]

                for ent in norm.get("entities", []):
                    ent_key = (ent.get("role", "").upper(), ent.get("name", "").strip().lower())
                    if ent_key not in seen_entity_keys:
                        seen_entity_keys.add(ent_key)
                        merged_entities.append(ent)

            # Build canonical formatted string
            role_order = ["manufacturer", "marketer", "packer", "importer"]
            formatted_parts = []
            for r in role_order:
                if r in merged_roles and merged_roles[r].get("name"):
                    data = merged_roles[r]
                    label = "Mfg" if r == "manufacturer" else ("Mkt" if r == "marketer" else r.title())
                    p_str = f" (PIN: {data['pincode']})" if data.get("pincode") else ""
                    formatted_parts.append(f"{label}: {data['name']}{p_str}")

            formatted_display = " | ".join(formatted_parts) if formatted_parts else (primary_name or "Detected")

            return {
                "entity_name": primary_name or (merged_entities[0]["name"] if merged_entities else ""),
                "address_markers": merged_addrs,
                "pincode": pin,
                "is_packer": is_pck,
                "is_importer": is_imp,
                "is_marketer": is_mkt,
                "roles": merged_roles,
                "entities": merged_entities,
                "canonical_key": (primary_name or "").lower(),
                "formatted": formatted_display
            }

        elif declaration_id == "D05":
            # Prioritize positive MRP and preserve tax inclusivity
            best_mrp = 0.0
            tax_inc = False
            best_norm = {}
            for c in candidates:
                norm = c.normalized_value or {}
                m = float(norm.get("mrp", 0.0) or 0.0)
                if m > 0.0 and (m > best_mrp or not best_norm):
                    best_mrp = m
                    best_norm = dict(norm)
                if norm.get("tax_inclusive"):
                    tax_inc = True

            if not best_norm:
                best_norm = dict(candidates[0].normalized_value or {})

            best_norm["tax_inclusive"] = tax_inc
            if best_mrp > 0.0:
                best_norm["mrp"] = best_mrp
                best_norm["canonical_key"] = f"₹{best_mrp:.2f}"
                best_norm["formatted"] = f"₹{best_mrp:.2f}"

            return best_norm

        elif declaration_id == "D03":
            # Prioritize positive Net Quantity
            best_base = 0.0
            best_norm = {}
            for c in candidates:
                norm = c.normalized_value or {}
                b = float(norm.get("base_value", 0.0) or 0.0)
                if b > 0.0 and (b > best_base or not best_norm):
                    best_base = b
                    best_norm = dict(norm)

            if not best_norm:
                best_norm = dict(candidates[0].normalized_value or {})

            return best_norm

        # Generic merge for other declarations (D02, D04, D06, D07)
        merged = {}
        for c in candidates:
            norm = c.normalized_value or {}
            for k, v in norm.items():
                if k not in merged or not merged[k]:
                    merged[k] = v
                elif isinstance(v, list) and isinstance(merged[k], list):
                    if v and isinstance(v[0], dict):
                        # List of dicts
                        seen = set()
                        comb = []
                        for item in merged[k] + v:
                            item_repr = str(sorted(item.items())) if isinstance(item, dict) else str(item)
                            if item_repr not in seen:
                                seen.add(item_repr)
                                comb.append(item)
                        merged[k] = comb
                    else:
                        merged[k] = list(dict.fromkeys(merged[k] + v))
                elif isinstance(v, dict) and isinstance(merged[k], dict):
                    merged[k] = {**merged[k], **v}
        return merged

    def fuse_product_evidence(
        self,
        records: List[Union[ProductEvidenceRecord, Dict[str, Any]]],
        product_id: str = "PROD_FUSED",
        product_title: Optional[str] = None,
        category: Optional[str] = None
    ) -> ProductEvidenceRecord:
        """
        Fuses a list of ProductEvidenceRecord objects or image evidence dicts.
        """
        image_evidences = []
        for i, item in enumerate(records):
            if isinstance(item, ProductEvidenceRecord):
                img_name = item.source_images[0] if item.source_images else f"image_{i+1}"
                raw_t = item.raw_ocr_texts.get(img_name, "")
                if not raw_t and item.raw_ocr_texts:
                    raw_t = next(iter(item.raw_ocr_texts.values()))
                vis = item.visual_detections.get(img_name, [])
                if not vis and item.visual_detections:
                    vis = next(iter(item.visual_detections.values()))
                image_evidences.append({
                    "image_name": img_name,
                    "declarations": item.declarations,
                    "ocr_text": raw_t,
                    "ocr_confidence": item.overall_ocr_confidence,
                    "visual_regions": vis
                })
            elif isinstance(item, dict):
                image_evidences.append(item)

        meta = {}
        if product_title:
            meta["product_title"] = product_title
        if category:
            meta["category"] = category

        return self.fuse_image_evidences(image_evidences, product_id=product_id, metadata=meta)
