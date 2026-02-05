"""
Load fMRI metadata from IDA XML files

Extracts TR, flip angle, TE, and other acquisition parameters from XML metadata.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from typing import Dict, Optional, Tuple

def parse_fmri_xml(xml_file: Path) -> Dict:
    """Parse single fMRI XML metadata file"""
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Define namespace
        ns = {'ida': 'http://ida.loni.usc.edu'}
        
        # Extract subject and series info
        subject_elem = root.find('.//subject', ns)
        if subject_elem is None:
            subject_elem = root.find('.//subject')
        
        if subject_elem is None:
            return {}
        
        # Get subject ID
        subject_id_elem = subject_elem.find('subjectIdentifier')
        if subject_id_elem is None:
            subject_id_elem = subject_elem.find('ida:subjectIdentifier', ns)
        subject_id = subject_id_elem.text if subject_id_elem is not None else None
        
        # Get series/study info
        study_elem = subject_elem.find('.//study', ns)
        if study_elem is None:
            study_elem = subject_elem.find('.//study')
        
        if study_elem is None:
            return {'subject_id': subject_id}
        
        # Get series identifier
        series_elem = study_elem.find('seriesIdentifier')
        if series_elem is None:
            series_elem = study_elem.find('ida:seriesIdentifier', ns)
        series_id = series_elem.text if series_elem is not None else None
        
        # Get description
        desc_elem = study_elem.find('.//imagingProtocol/description', ns)
        if desc_elem is None:
            desc_elem = study_elem.find('.//description')
        description = desc_elem.text if desc_elem is not None else None
        
        # Extract protocol terms
        protocol_terms = {}
        protocol_elem = study_elem.find('.//imagingProtocol/protocolTerm', ns)
        if protocol_elem is None:
            protocol_elem = study_elem.find('.//protocolTerm')
        
        if protocol_elem is not None:
            for protocol in [protocol_elem] + list(protocol_elem.iter('protocol')):
                term_name = protocol.get('term')
                if term_name:
                    try:
                        # Try to convert to float
                        protocol_terms[term_name] = float(protocol.text)
                    except (ValueError, TypeError):
                        protocol_terms[term_name] = protocol.text
            
            # Also check parent for additional protocols
            parent = protocol_elem
            for sibling in parent.iter():
                if sibling.tag.endswith('protocol') or sibling.tag == 'protocol':
                    term_name = sibling.get('term')
                    if term_name:
                        try:
                            protocol_terms[term_name] = float(sibling.text)
                        except (ValueError, TypeError):
                            protocol_terms[term_name] = sibling.text
        
        result = {
            'subject_id': subject_id,
            'series_id': series_id,
            'description': description,
            'protocols': protocol_terms
        }
        
        return result
        
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return {}

def build_metadata_index(metadata_dir: Path = None) -> Dict:
    """Build index of all fMRI metadata
    
    Returns:
        Dict with keys like 'subject_id_description' -> metadata
    """
    
    if metadata_dir is None:
        metadata_dir = Path('/home/emanuele/Desktop/Studi/preprocessing/fMRI_IDA_Metadata/PPMI')
    
    index = {}
    
    # Find all XML files
    xml_files = list(metadata_dir.rglob('*.xml'))
    print(f"Found {len(xml_files)} XML metadata files")
    
    for xml_file in xml_files:
        metadata = parse_fmri_xml(xml_file)
        
        if 'subject_id' in metadata and 'description' in metadata:
            subject_id = metadata['subject_id']
            description = metadata['description']
            
            # Create key: subject_id + description
            key = f"{subject_id}_{description}"
            index[key] = metadata
    
    print(f"Indexed {len(index)} metadata entries")
    return index

def get_fmri_parameters(subject_id: int, description: str, metadata_index: Dict) -> Tuple[Optional[float], Dict]:
    """Get TR and other parameters for specific scan
    
    Args:
        subject_id: Subject ID
        description: Protocol description (e.g., 'ep2d_bold_rest', 'rsfMRI_RL')
        metadata_index: Metadata index from build_metadata_index()
    
    Returns:
        (tr_seconds, params_dict) or (None, {}) if not found
    """
    
    key = f"{subject_id}_{description}"
    
    if key not in metadata_index:
        return None, {}
    
    metadata = metadata_index[key]
    protocols = metadata.get('protocols', {})
    
    # Extract key parameters
    tr = protocols.get('TR')  # in milliseconds, convert to seconds
    if tr is not None and tr > 1:  # If in milliseconds
        tr = tr / 1000.0
    
    te = protocols.get('TE')  # in milliseconds
    if te is not None and te > 1:
        te = te / 1000.0
    
    params = {
        'TR': tr,
        'TE': te,
        'FlipAngle': protocols.get('Flip Angle'),
        'Manufacturer': protocols.get('Manufacturer'),
        'MriModel': protocols.get('Mfg Model'),
        'FieldStrength': protocols.get('Field Strength'),
        'PulseSequence': protocols.get('Pulse Sequence'),
        'MatrixX': protocols.get('Matrix X'),
        'MatrixY': protocols.get('Matrix Y'),
        'Slices': protocols.get('Slices'),
        'PixelSpacingX': protocols.get('Pixel Spacing X'),
        'PixelSpacingY': protocols.get('Pixel Spacing Y'),
        'SliceThickness': protocols.get('Slice Thickness'),
    }
    
    return tr, params

def create_metadata_csv(metadata_dir: Path = None, output_file: Path = None):
    """Create CSV with all metadata parameters for quick lookup"""
    
    if metadata_dir is None:
        metadata_dir = Path('/home/emanuele/Desktop/Studi/preprocessing/fMRI_IDA_Metadata/PPMI')
    
    if output_file is None:
        output_file = Path('/home/emanuele/Desktop/Studi/preprocessing/fMRI_Acquisition_Parameters.csv')
    
    index = build_metadata_index(metadata_dir)
    
    records = []
    for key, metadata in index.items():
        subject_id = metadata.get('subject_id')
        description = metadata.get('description')
        protocols = metadata.get('protocols', {})
        
        # Convert TR from ms to seconds if needed
        tr = protocols.get('TR')
        if tr is not None and tr > 1:
            tr = tr / 1000.0
        
        record = {
            'SubjectID': subject_id,
            'Description': description,
            'TR_seconds': tr,
            'TE_ms': protocols.get('TE'),
            'FlipAngle_deg': protocols.get('Flip Angle'),
            'FieldStrength_T': protocols.get('Field Strength'),
            'Manufacturer': protocols.get('Manufacturer'),
            'MriModel': protocols.get('Mfg Model'),
            'PulseSequence': protocols.get('Pulse Sequence'),
            'MatrixX': protocols.get('Matrix X'),
            'MatrixY': protocols.get('Matrix Y'),
            'Slices': protocols.get('Slices'),
            'PixelSpacingX_mm': protocols.get('Pixel Spacing X'),
            'PixelSpacingY_mm': protocols.get('Pixel Spacing Y'),
            'SliceThickness_mm': protocols.get('Slice Thickness'),
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df = df.sort_values(['SubjectID', 'Description'])
    df.to_csv(output_file, index=False)
    
    print(f"Saved metadata CSV to {output_file}")
    print(f"Total scans: {len(df)}")
    print(f"\nTR statistics:")
    print(df['TR_seconds'].describe())
    
    return df

if __name__ == '__main__':
    # Create metadata CSV on first run
    df = create_metadata_csv()
    print("\nFirst few rows:")
    print(df.head(10))
