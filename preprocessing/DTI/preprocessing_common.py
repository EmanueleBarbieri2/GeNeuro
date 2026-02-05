"""
Shared utilities for PPMI Connectivity Preprocessing
=====================================================

Common configuration, logging, and data loading functions used by both
fMRI and DTI preprocessing pipelines.
"""

import numpy as np
import pandas as pd
import ants
import logging
from pathlib import Path
from datetime import datetime
import nibabel as nib
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Pipeline configuration"""
    
    # Paths
    DATA_ROOT = Path(os.getenv('PPMI_PREPROCESSING_ROOT', Path(__file__).resolve().parent))
    OUTPUT_ROOT = DATA_ROOT / 'derivatives'
    FREESURFER_TEMPLATE = Path('/usr/local/freesurfer/subjects/fsaverage/mri')
    
    # Create output directories
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # APARC+ASEG atlas
    # If not available locally, will use AAL-116 as fallback
    APARC_ASEG_ATLAS = FREESURFER_TEMPLATE / 'aparc+aseg.mgz'
    APARC_ASEG_PATH = None  # Optional explicit path provided by user
    
    # Preprocessing parameters
    TR = 2.0  # Repetition time (sec) - check from JSON
    SMOOTHING_SIGMA = 3  # Gaussian smoothing
    HIGHPASS_CUTOFF = 0.01  # Hz
    LOWPASS_CUTOFF = 0.1  # Hz
    MOTION_THRESHOLD = 0.5  # mm (for QC)
    
    # Templates (will download if needed)
    MNI_TEMPLATE = None  # Will use ANTsPy default
    
    # Logging
    LOG_DIR = OUTPUT_ROOT / 'logs'
    LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(subject_id, modality):
    """Setup logging for subject"""
    
    log_file = Config.LOG_DIR / f'{subject_id}_{modality}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logger = logging.getLogger(f'{subject_id}_{modality}')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ============================================================================
# FREESURFER DATA LOADING
# ============================================================================

def load_freesurfer_data(data_root):
    """Load and merge FreeSurfer structural metrics"""
    
    cth = pd.read_csv(data_root / 'FS7_APARC_CTH_28Jan2026.csv')
    sa = pd.read_csv(data_root / 'FS7_APARC_SA_28Jan2026.csv')
    vol = pd.read_csv(data_root / 'FS7_ASEG_VOL_28Jan2026.csv')
    
    fs_data = cth.merge(sa, on=['PATNO', 'EVENT_ID']).merge(vol, on=['PATNO', 'EVENT_ID'])
    
    return fs_data

def get_fs_confounds(fs_data, subject_id, event_id='BL'):
    """Extract FreeSurfer confounds for subject"""
    
    try:
        row = fs_data[(fs_data['PATNO'] == int(subject_id)) & 
                      (fs_data['EVENT_ID'] == event_id)]
        
        if row.empty:
            return None
        
        row = row.iloc[0]
        
        return {
            'mean_thickness': float(row.get('lh_MeanThickness', np.nan)),
            'brain_volume': float(row.get('BrainSegVol', np.nan)),
            'cortex_volume': float(row.get('CortexVol', np.nan)),
            'ticv': float(row.get('EstimatedTotalIntraCranialVol', np.nan))
        }
    except Exception as e:
        print(f"Warning: Could not extract FS confounds: {e}")
        return None

# ============================================================================
# ATLAS LOADING
# ============================================================================

def load_aparc_aseg_atlas():
    """Load APARC+ASEG atlas - REQUIRED, no fallbacks"""
    
    # Try explicit path via Config or environment variable
    import os
    explicit_path = Config.APARC_ASEG_PATH or os.environ.get('APARC_ASEG_PATH')
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            try:
                img = nib.load(str(p))
                data = img.get_fdata()
                spacing = img.header.get_zooms()[:3]
                # Get proper affine matrix from nibabel
                affine = img.affine
                # Create ANTs image with proper orientation
                atlas = ants.from_numpy(
                    data.astype(np.int32),
                    origin=tuple(affine[:3, 3]),
                    spacing=spacing,
                    direction=affine[:3, :3] / spacing
                )
                # Store valid labels for filtering after resampling
                atlas.valid_labels = np.unique(data.astype(int))[1:]  # Exclude 0
                print(f"✓ Loaded APARC+ASEG atlas from {p}")
                print(f"  Atlas contains {len(atlas.valid_labels)} regions")
                print(f"  Origin: {atlas.origin}, Spacing: {atlas.spacing}")
                return atlas
            except Exception as e:
                raise RuntimeError(f"Failed to load APARC+ASEG atlas from {p}: {e}")
        else:
            raise FileNotFoundError(f"APARC+ASEG atlas not found at: {p}")
    
    # Try FreeSurfer installation default
    if Config.APARC_ASEG_ATLAS.exists():
        try:
            img = nib.load(str(Config.APARC_ASEG_ATLAS))
            data = img.get_fdata()
            spacing = img.header.get_zooms()[:3]
            affine = img.affine
            atlas = ants.from_numpy(
                data.astype(np.int32),
                origin=tuple(affine[:3, 3]),
                spacing=spacing,
                direction=affine[:3, :3] / spacing
            )
            atlas.valid_labels = np.unique(data.astype(int))[1:]
            print(f"✓ Loaded APARC+ASEG atlas from FreeSurfer")
            print(f"  Atlas contains {len(atlas.valid_labels)} regions")
            print(f"  Origin: {atlas.origin}, Spacing: {atlas.spacing}")
            return atlas
        except Exception as e:
            raise RuntimeError(f"Failed to load APARC+ASEG atlas from FreeSurfer: {e}")
    
    # No fallbacks - raise error
    raise FileNotFoundError(
        "APARC+ASEG atlas not found!\n"
        "Please provide the atlas using one of these methods:\n"
        "  1. Set APARC_ASEG_PATH environment variable:\n"
        "     export APARC_ASEG_PATH=/path/to/aparc+aseg.mgz\n"
        "  2. Pass --atlas-path argument:\n"
        "     python ppmi_connectivity_pipeline.py --atlas-path /path/to/aparc+aseg.mgz\n"
        "  3. Install FreeSurfer and ensure aparc+aseg.mgz is in the expected location"
    )
