"""
Load DTI metadata from IDA XML files

Extracts TR, TE, flip angle, voxel size, and other acquisition parameters from XML metadata.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple
import re


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_dti_xml(xml_file: Path) -> Dict:
    """Parse single DTI XML metadata file"""

    def _find_with_fallback(elem, tag: str):
        return (
            elem.find(tag)
            or elem.find(f"ida:{tag}", ns)
            or elem.find(f".//{{*}}{tag}")
            or elem.find(f".//{tag}")
        )

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        ns = {"ida": "http://ida.loni.usc.edu"}

        subject_elem = root.find(".//{*}subject") or root.find(".//subject", ns) or root.find(".//subject")
        if subject_elem is None:
            return {}

        subject_id_elem = _find_with_fallback(subject_elem, "subjectIdentifier")
        subject_id = subject_id_elem.text if subject_id_elem is not None else None

        study_elem = _find_with_fallback(subject_elem, "study")
        if study_elem is None:
            return {"subject_id": subject_id}

        series_elem = _find_with_fallback(study_elem, "seriesIdentifier")
        series_id = series_elem.text if series_elem is not None else None

        date_elem = _find_with_fallback(study_elem, "dateAcquired")
        date_acquired = date_elem.text if date_elem is not None else None

        desc_elem = _find_with_fallback(study_elem, "description")
        description = desc_elem.text if desc_elem is not None else None

        protocol_terms = {}
        protocol_elem = study_elem.find(".//imagingProtocol/protocolTerm", ns) or study_elem.find(".//protocolTerm")
        if protocol_elem is not None:
            for protocol in [protocol_elem] + list(protocol_elem.iter("protocol")):
                term_name = protocol.get("term")
                if term_name:
                    try:
                        protocol_terms[term_name] = float(protocol.text)
                    except (ValueError, TypeError):
                        protocol_terms[term_name] = protocol.text

            for sibling in protocol_elem.iter():
                if sibling.tag.endswith("protocol") or sibling.tag == "protocol":
                    term_name = sibling.get("term")
                    if term_name:
                        try:
                            protocol_terms[term_name] = float(sibling.text)
                        except (ValueError, TypeError):
                            protocol_terms[term_name] = sibling.text

        return {
            "subject_id": subject_id,
            "series_id": series_id,
            "date_acquired": date_acquired,
            "description": description,
            "protocols": protocol_terms,
            "xml_file": str(xml_file),
        }

    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return {}


def build_dti_metadata_index(metadata_dir: Path = None) -> Dict:
    """Build index of all DTI metadata

    Returns:
        Dict with keys:
          - by_subject: {subject_id: [metadata, ...]}
          - by_key: {subject_id_description: metadata}
    """

    if metadata_dir is None:
        metadata_dir = Path("/home/emanuele/Desktop/Studi/preprocessing/DTI_IDA_Metadata/PPMI")

    index = {"by_subject": {}, "by_key": {}}

    xml_files = list(metadata_dir.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML metadata files")

    for xml_file in xml_files:
        metadata = parse_dti_xml(xml_file)
        subject_id = metadata.get("subject_id")
        description = metadata.get("description")

        if not subject_id or not description:
            continue

        index["by_subject"].setdefault(subject_id, []).append(metadata)
        key = f"{subject_id}_{description}"
        index["by_key"][key] = metadata

    print(f"Indexed {len(index['by_key'])} metadata entries")
    return index


def _score_match(scan_type: str, description: str) -> int:
    scan_tokens = set(_normalize_text(scan_type).split())
    desc_tokens = set(_normalize_text(description).split())
    return len(scan_tokens & desc_tokens)


def get_dti_parameters(
    subject_id: int,
    scan_type: str,
    scan_date: Optional[str],
    metadata_index: Dict,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Get DTI parameters for a specific scan

    Returns:
        (params_dict, raw_metadata_dict) or (None, None) if not found
    """

    if metadata_index is None:
        return None, None

    subject_entries = metadata_index.get("by_subject", {}).get(str(subject_id), [])
    if not subject_entries:
        return None, None

    candidates = subject_entries
    if scan_date:
        candidates = [m for m in candidates if m.get("date_acquired") == scan_date] or candidates

    best = None
    best_score = -1
    for meta in candidates:
        score = _score_match(scan_type, meta.get("description"))
        if score > best_score:
            best_score = score
            best = meta

    if best is None:
        return None, None

    protocols = best.get("protocols", {})

    tr = protocols.get("TR")
    if tr is not None and tr > 1:
        tr = tr / 1000.0

    te = protocols.get("TE")
    if te is not None and te > 1:
        te = te / 1000.0

    params = {
        "TR": tr,
        "TE": te,
        "FlipAngle": protocols.get("Flip Angle"),
        "Manufacturer": protocols.get("Manufacturer"),
        "MriModel": protocols.get("Mfg Model"),
        "FieldStrength": protocols.get("Field Strength"),
        "PulseSequence": protocols.get("Pulse Sequence"),
        "MatrixX": protocols.get("Matrix X"),
        "MatrixY": protocols.get("Matrix Y"),
        "MatrixZ": protocols.get("Matrix Z"),
        "PixelSizeX": protocols.get("Pixel Size X"),
        "PixelSizeY": protocols.get("Pixel Size Y"),
        "SliceThickness": protocols.get("Slice Thickness"),
        "GradientDirections": protocols.get("Gradient Directions"),
        "Description": best.get("description"),
        "DateAcquired": best.get("date_acquired"),
        "SeriesID": best.get("series_id"),
    }

    return params, best
