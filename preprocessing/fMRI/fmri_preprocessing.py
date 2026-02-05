"""
fMRI Preprocessing - Version 2
Processes (subject, visit) pairs with multi-scan merging and validation

Output structure: /fmri_final/{subject}/{visit}/fc_matrix.npy
"""

import numpy as np
import pandas as pd
import ants
import json
import logging
import gc
from pathlib import Path
from datetime import datetime
import argparse
import os
from sklearn.linear_model import LinearRegression
from scipy.signal import butter, filtfilt
from scipy.ndimage import binary_opening, binary_closing, label, binary_fill_holes
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

from load_fmri_metadata import build_metadata_index, get_fmri_parameters
from preprocessing_common import (
    Config, setup_logging, load_freesurfer_data, get_fs_confounds, 
    load_aparc_aseg_atlas
)
import xml.etree.ElementTree as ET

# ============================================================================
# UTILITIES
# ============================================================================

def build_subject_visit_to_timestamp_map(nifti_root):
    """Build mapping from (subject, visit) to actual timestamps in NIfTI directory
    
    Maps (subject_id, visit_code) from CSV → {session_type: timestamp}
    by examining directory structure and matching patterns
    """
    mapping = {}
    
    for subject_dir in nifti_root.iterdir():
        if not subject_dir.is_dir():
            continue
        
        try:
            subject_id = int(subject_dir.name)
        except ValueError:
            continue
        
        # Scan each session type directory
        for session_dir in subject_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            session_type = session_dir.name
            
            # Each timestamp directory represents a unique session/visit
            for timestamp_dir in session_dir.iterdir():
                if not timestamp_dir.is_dir():
                    continue
                
                timestamp = timestamp_dir.name
                
                # Check if any .nii.gz files exist
                nii_files = list(timestamp_dir.rglob('*.nii.gz'))
                if nii_files:
                    if subject_id not in mapping:
                        mapping[subject_id] = {}
                    if session_type not in mapping[subject_id]:
                        mapping[subject_id][session_type] = []
                    
                    mapping[subject_id][session_type].append(timestamp)
    
    return mapping

def extract_visit_code_from_xml(xml_dir):
    """Extract (subject, visit) mapping from XML metadata files
    
    Returns: dict of {(subject_id, visit_code): [timestamps]}
    """
    visit_mapping = {}
    
    xml_files = list(xml_dir.rglob('*.xml'))
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract subject ID
            subject_elem = root.find('.//{http://ida.loni.usc.edu}subjectIdentifier')
            if subject_elem is None:
                subject_elem = root.find('.//subjectIdentifier')
            
            if subject_elem is None or not subject_elem.text:
                continue
            
            subject_id = int(subject_elem.text)
            
            # Extract visit identifier
            visit_elem = root.find('.//{http://ida.loni.usc.edu}visitIdentifier')
            if visit_elem is None:
                visit_elem = root.find('.//visitIdentifier')
            
            if visit_elem is None or not visit_elem.text:
                continue
            
            visit_identifier = visit_elem.text.strip()
            
            # Map visit identifier to standard code
            # Baseline -> BL, Month 12 -> V04, Month 24 -> V08, etc.
            visit_code = map_visit_identifier_to_code(visit_identifier)
            
            if visit_code:
                key = (subject_id, visit_code)
                if key not in visit_mapping:
                    visit_mapping[key] = set()
        
        except Exception as e:
            pass
    
    return visit_mapping

def map_visit_identifier_to_code(visit_identifier):
    """Map XML visitIdentifier to CSV visit code
    
    Examples:
        'Baseline' -> 'BL'
        'Month 12' -> 'V04'
        'Month 24' -> 'V08'
        'Month 36' -> 'V12'
        'Month 48' -> 'V14'
    """
    visit_id_lower = visit_identifier.lower().strip()
    
    if 'baseline' in visit_id_lower:
        return 'BL'
    elif 'month' in visit_id_lower:
        # Extract month number
        parts = visit_id_lower.split()
        for i, part in enumerate(parts):
            if part == 'month' and i + 1 < len(parts):
                try:
                    month = int(parts[i + 1])
                    # Common PPMI mapping: V04=12mo, V08=24mo, V12=36mo, V14=48mo
                    if month == 12:
                        return 'V04'
                    elif month == 24:
                        return 'V08'
                    elif month == 36:
                        return 'V12'
                    elif month == 48:
                        return 'V14'
                    else:
                        return None
                except ValueError:
                    pass
    
    return None

def get_visit_code_for_subject_timestamp(subject_id, timestamp, nifti_root, csv_visits):
    """Given subject and timestamp, find corresponding visit code from CSV
    
    Args:
        subject_id: Subject ID
        timestamp: Timestamp directory name (e.g., '2020-12-09_13_53_07.0')
        nifti_root: Root of NIfTI directory
        csv_visits: List of (subject, visit) tuples from CSV
    
    Returns: visit_code (e.g., 'BL', 'V04') or None
    """
    # Try to match by checking if subject+visit combo is in CSV
    subject_visits = [v for s, v in csv_visits if s == subject_id]
    
    if len(subject_visits) == 1:
        # Only one visit for this subject
        return subject_visits[0]
    elif len(subject_visits) > 1:
        # Multiple visits - use simple heuristic: BL usually first chronologically
        # Sort timestamps to find order
        all_timestamps = []
        subject_dir = nifti_root / str(subject_id)
        
        if subject_dir.exists():
            for session_dir in subject_dir.iterdir():
                if session_dir.is_dir():
                    for ts_dir in session_dir.iterdir():
                        if ts_dir.is_dir():
                            try:
                                dt = datetime.fromisoformat(ts_dir.name.replace('_', ':').rsplit('.', 1)[0])
                                all_timestamps.append((dt, ts_dir.name))
                            except:
                                pass
        
        if all_timestamps:
            all_timestamps.sort(key=lambda x: x[0])
            sorted_ts = [ts for _, ts in all_timestamps]
            
            if timestamp in sorted_ts:
                idx = sorted_ts.index(timestamp)
                # Assign based on chronological order
                if idx == 0:
                    return 'BL'
                else:
                    visit_options = sorted(subject_visits)
                    if idx < len(visit_options):
                        return visit_options[idx]
    
    return None

def get_subject_visit_from_csv(csv_path):
    """Load (subject, visit) pairs from CSV where data is complete
    
    Returns: (pairs_list, acq_date_mapping, expected_scan_counts)
        pairs_list: [(subject, visit), ...]
        acq_date_mapping: {(subject, visit): [acq_dates]}
        expected_scan_counts: {(subject, visit): num_scans_expected}
    """
    df = pd.read_csv(csv_path)
    
    # Remove duplicate header row if it exists
    df = df[df['Subject'] != 'Subject']
    
    # Convert Subject to int to match filesystem directory names (int-based)
    df['Subject'] = df['Subject'].astype(int)
    
    pairs = df[['Subject', 'Visit']].drop_duplicates().values.tolist()
    
    # Build acquisition date mapping and count expected scans
    acq_mapping = {}
    scan_counts = {}
    
    for _, row in df.iterrows():
        subject = row['Subject']
        visit = row['Visit']
        acq_date = row['Acq Date']
        
        key = (subject, visit)
        if key not in acq_mapping:
            acq_mapping[key] = set()
            scan_counts[key] = 0
        acq_mapping[key].add(acq_date)
        scan_counts[key] += 1
    
    # Convert sets to lists
    acq_mapping = {k: list(v) for k, v in acq_mapping.items()}
    
    return pairs, acq_mapping, scan_counts

def timestamp_matches_acq_date(timestamp, acq_dates):
    """Check if a timestamp directory matches any acquisition date from CSV
    
    Args:
        timestamp: Directory name like '2023-02-09_10_45_02.0'
        acq_dates: List of dates from CSV like ['2/09/2023', '2/9/2023']
    
    Returns: True if timestamp matches any acquisition date
    """
    # Extract date from timestamp (format: YYYY-MM-DD_HH_MM_SS.0)
    try:
        timestamp_date = timestamp.split('_')[0]  # '2023-02-09'
        timestamp_dt = datetime.strptime(timestamp_date, '%Y-%m-%d')
    except:
        return False
    
    # Check against all acquisition dates
    for acq_date in acq_dates:
        try:
            # Parse CSV date (formats like '2/09/2023', '10/26/2021')
            acq_dt = datetime.strptime(acq_date, '%m/%d/%Y')
            
            # Match on date only (ignore time)
            if timestamp_dt.date() == acq_dt.date():
                return True
        except:
            continue
    
    return False

def find_all_scans_for_visit(subject_id, visit_id, nifti_root, acq_date_mapping=None, expected_count=None):
    """Find ALL fMRI scans for a (subject, visit) pair and verify completeness
    
    Args:
        subject_id: Subject ID (int)
        visit_id: Visit code from CSV (e.g., 'BL', 'V04')
        nifti_root: Root of NIfTI directory
        acq_date_mapping: Dict of {(subject, visit): [acq_dates]} for timestamp validation
        expected_count: Number of scans expected for this (subject, visit) pair
    
    Returns: dict of {session_type: [list of .nii.gz files]} or None if incomplete
    """
    base_path = nifti_root / str(subject_id)
    
    if not base_path.exists():
        return None
    
    # Get acquisition dates for this (subject, visit) pair
    expected_dates = []
    if acq_date_mapping:
        expected_dates = acq_date_mapping.get((subject_id, visit_id), [])
    
    # Group scans by session type (rsfMRI_RL, rsfMRI_LR, etc.)
    scans_by_session = {}
    total_scans = 0
    
    for session_dir in base_path.glob('*'):
        if not session_dir.is_dir():
            continue
        
        session_type = session_dir.name
        
        # Each subdirectory under session is a timestamp
        for timestamp_dir in session_dir.glob('*'):
            if not timestamp_dir.is_dir():
                continue
            
            timestamp = timestamp_dir.name
            
            # If we have acquisition dates, only include scans from those dates
            if expected_dates:
                if not timestamp_matches_acq_date(timestamp, expected_dates):
                    continue
            
            nii_files = sorted(timestamp_dir.glob('**/*.nii.gz'))
            if nii_files:
                if session_type not in scans_by_session:
                    scans_by_session[session_type] = []
                scans_by_session[session_type].extend(nii_files)
                total_scans += len(nii_files)
    
    # Validate we have all expected scans
    if expected_count is not None and total_scans != expected_count:
        return None
    
    return scans_by_session if scans_by_session else None

def merge_multiple_fc_matrices_fisher(fc_list, logger=None):
    """Merge multiple FC matrices using Fisher z-transform (for multiple scans)"""
    if len(fc_list) == 1:
        return fc_list[0]
    
    if logger:
        logger.info(f"Merging {len(fc_list)} FC matrices using Fisher z-transform")
    
    # Convert to z-scores
    z_list = [np.arctanh(np.clip(fc, -0.9999, 0.9999)) for fc in fc_list]
    
    # Average z-scores
    z_mean = np.mean(z_list, axis=0)
    
    # Convert back
    fc_avg = np.tanh(z_mean)
    
    return fc_avg

# ============================================================================
# PREPROCESSING FUNCTIONS (same as before)
# ============================================================================

def motion_correct_fmri(img_4d, logger):
    """Motion correction using rigid body registration"""
    logger.info("Starting motion correction...")
    try:
        data = img_4d.numpy()
        n_frames = data.shape[3]
        spacing = tuple(img_4d.spacing[:3])
        origin = tuple(img_4d.origin[:3])
        direction_full = img_4d.direction
        if direction_full is not None:
            direction = tuple(tuple(row[:3]) for row in direction_full[:3])
        else:
            direction = None
        
        ref_idx = n_frames // 2
        reference = ants.from_numpy(data[..., ref_idx], spacing=spacing, origin=origin, direction=direction)
        
        motion_corrected_frames = []
        for i in range(n_frames):
            if i % 50 == 0:
                logger.info(f"  Motion correcting frame {i}/{n_frames}")
            frame_img = ants.from_numpy(data[..., i], spacing=spacing, origin=origin, direction=direction)
            reg = ants.registration(fixed=reference, moving=frame_img,
                                   type_of_transform='Rigid', verbose=False)
            motion_corrected_frames.append(reg['warpedmovout'])
        
        motion_corrected_4d = ants.merge_channels(motion_corrected_frames)
        logger.info("✓ Motion correction complete")
        return motion_corrected_4d
        
    except Exception as e:
        logger.error(f"Motion correction failed: {e}")
        raise

def denoise_fmri(img_4d, logger):
    """Denoise fMRI"""
    logger.info("Denoising fMRI...")
    try:
        frames = ants.split_channels(img_4d)
        denoised_frames = []
        for i, frame in enumerate(frames):
            if i % 50 == 0:
                logger.info(f"  Denoising volume {i}/{len(frames)}")
            denoised = ants.denoise_image(frame, noise_model='Rician')
            denoised_frames.append(denoised)
        denoised_4d = ants.merge_channels(denoised_frames)
        logger.info("✓ Denoising complete")
        return denoised_4d
    except Exception as e:
        logger.warning(f"Denoising failed, skipping: {e}")
        return img_4d

def extract_brain_mask_fmri(img_3d, logger):
    """Brain extraction"""
    logger.info("Starting brain extraction...")
    try:
        data = img_3d.numpy()
        data = data - np.min(data)
        if np.max(data) > 0:
            data = data / np.max(data)
        mask_np = data > 0.2
        mask_np = binary_opening(mask_np, iterations=2)
        mask_np = binary_closing(mask_np, iterations=2)
        labeled, nlab = label(mask_np)
        if nlab > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            largest = np.argmax(sizes)
            mask_np = labeled == largest
        brain_mask = ants.from_numpy(mask_np.astype(np.float32),
                                     origin=img_3d.origin,
                                     spacing=img_3d.spacing,
                                     direction=img_3d.direction)
        logger.info("✓ Brain extraction complete")
        return brain_mask
    except Exception as e:
        logger.error(f"Brain extraction failed: {e}")
        raise

def register_to_template(img, template=None, logger=None):
    """Register to MNI template"""
    if logger:
        logger.info("Starting registration to template...")
    try:
        if template is None:
            template = ants.get_ants_data('mni')
            template_img = ants.image_read(template)
        else:
            template_img = ants.image_read(str(template))
        reg = ants.registration(fixed=template_img, moving=img,
                               type_of_transform='SyNRA', verbose=False)
        if logger:
            logger.info("✓ Registration complete")
        return reg['warpedmovout'], reg['fwdtransforms']
    except Exception as e:
        if logger:
            logger.error(f"Registration failed: {e}")
        raise

def smooth_fmri(img_4d, sigma=3, logger=None):
    """Spatial smoothing"""
    if logger:
        logger.info(f"Starting smoothing (sigma={sigma})...")
    try:
        frames = ants.split_channels(img_4d)
        n_volumes = len(frames)
        smoothed_frames = []
        for i, frame in enumerate(frames):
            if i % 50 == 0 and logger:
                logger.info(f"  Smoothing volume {i}/{n_volumes}")
            smoothed = ants.smooth_image(frame, sigma=sigma)
            smoothed_frames.append(smoothed)
        smoothed_4d = ants.merge_channels(smoothed_frames)
        if logger:
            logger.info("✓ Smoothing complete")
        return smoothed_4d
    except Exception as e:
        if logger:
            logger.error(f"Smoothing failed: {e}")
        raise

def extract_timeseries_aparc_aseg(img_4d, atlas, logger):
    """Extract regional timeseries"""
    try:
        atlas_data = atlas.numpy().astype(int)
        regions = np.unique(atlas_data)[1:]
        
        logger.info(f"  Atlas shape: {atlas_data.shape}, unique regions: {len(regions)}")
        
        frames = ants.split_channels(img_4d)
        n_regions = len(regions)
        n_timepoints = len(frames)
        regional_ts = np.zeros((n_regions, n_timepoints))
        
        for idx, region_id in enumerate(regions):
            if idx % 20 == 0:
                logger.info(f"  Extracting region {idx}/{n_regions}")
            mask = (atlas_data == region_id)
            for t, frame in enumerate(frames):
                volume = frame.numpy()
                regional_ts[idx, t] = np.mean(volume[mask])
        
        logger.info(f"✓ Extracted {n_regions} regional timeseries")
        return regional_ts, regions
    except Exception as e:
        logger.error(f"Timeseries extraction failed: {e}")
        raise

def temporal_filter(timeseries, tr=2.0, high_pass=0.01, low_pass=0.1, logger=None):
    """Temporal bandpass filtering"""
    if logger:
        logger.info(f"Temporal filtering: {high_pass}-{low_pass} Hz...")
    
    try:
        nyquist = 1 / (2 * tr)
        low = high_pass / nyquist
        high = low_pass / nyquist
        
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, low + 0.001, 0.999)
        
        b, a = butter(4, [low, high], btype='band')
        
        n_timepoints = timeseries.shape[1]
        padlen = 3 * (max(len(a), len(b)) - 1)
        if n_timepoints <= padlen:
            if logger:
                logger.warning(f"Temporal filtering skipped: n_timepoints={n_timepoints} <= padlen={padlen}")
            return timeseries
        
        filtered = np.zeros_like(timeseries)
        for i in range(timeseries.shape[0]):
            filtered[i, :] = filtfilt(b, a, timeseries[i, :])
        
        if logger:
            logger.info("✓ Temporal filtering complete")
        return filtered
    except Exception as e:
        if logger:
            logger.error(f"Temporal filtering failed: {e}")
        raise

def compute_fc_matrix(timeseries, logger=None):
    """Compute functional connectivity from regional timeseries"""
    if logger:
        logger.info("Computing FC matrix...")
    
    # Pearson correlation
    fc = np.corrcoef(timeseries)
    
    if logger:
        logger.info(f"✓ FC matrix computed: {fc.shape}")
    
    return fc

def preprocess_fmri_complete(nii_file, subject_id, visit_id, fs_data, atlas, output_dir, 
                            logger, metadata_index=None, session_type=None):
    """Complete fMRI preprocessing for a single scan file"""
    
    logger.info(f"fMRI PREPROCESSING: {nii_file.name}")
    logger.info("=" * 80)
    
    try:
        logger.info(f"Loading fMRI from {nii_file}")
        img_4d = ants.image_read(str(nii_file))
        
        data = img_4d.numpy()
        logger.info(f"  Shape: {data.shape}")
        
        # Get TR and other params
        json_file = nii_file.with_suffix('.json')
        tr = 2.0
        high_pass, low_pass = 0.01, 0.1
        
        if json_file.exists():
            with open(json_file) as f:
                json_data = json.load(f)
                tr = json_data.get('RepetitionTime', tr)
                logger.info(f"  TR: {tr} sec (json)")
        
        if metadata_index and nii_file.name in metadata_index:
            params = get_fmri_parameters(metadata_index, nii_file.name)
            if params:
                tr = params.get('TR', tr)
                high_pass = params.get('HighPass', high_pass)
                low_pass = params.get('LowPass', low_pass)
                logger.info(f"  TR: {tr} sec (ida_xml)")
        
        logger.info(f"  Bandpass: {high_pass}-{low_pass} Hz")
        
        # Preprocessing pipeline
        img_mc = motion_correct_fmri(img_4d, logger)
        img_dn = denoise_fmri(img_mc, logger)
        
        # Brain extraction on mean - compute mean from 4D data
        data_dn = img_dn.numpy()
        mean_data = np.mean(data_dn, axis=3)
        mean_img = ants.from_numpy(mean_data, spacing=img_dn.spacing[:3], 
                                   origin=img_dn.origin[:3], direction=img_dn.direction)
        
        brain_mask = extract_brain_mask_fmri(mean_img, logger)
        
        # Mask and register
        frames = ants.split_channels(img_dn)
        masked_frames = [ants.mask_image(f, brain_mask) for f in frames]
        
        img_mean_masked = ants.mask_image(mean_img, brain_mask)
        template_img, transforms = register_to_template(img_mean_masked, logger=logger)
        
        logger.info("Registering all volumes to template...")
        registered_frames = [
            ants.apply_transforms(fixed=template_img, moving=f,
                                 transformlist=transforms, whichtoinvert=[0, 0])
            for f in masked_frames
        ]
        
        # Smooth - with aggressive garbage collection to prevent OOM
        logger.info("Smoothing registered volumes...")
        smoothing_sigma = tr * 2.5 / 3.5  # Approximate voxel size
        
        # For large scans (>400 volumes), downsample to reduce memory
        n_volumes = len(registered_frames)
        if n_volumes > 400:
            logger.warning(f"  Large scan ({n_volumes} volumes) - downsampling to reduce memory")
            # Downsample temporal dimension by 50% (keep every 2nd frame)
            registered_frames = registered_frames[::2]
            logger.info(f"  Downsampled to {len(registered_frames)} frames")
        
        # Process frames one by one to avoid memory buildup
        smoothed_frames = []
        for i, f in enumerate(registered_frames):
            if i % 50 == 0:  # Force cleanup every 50 frames
                gc.collect()
            smoothed = ants.smooth_image(f, sigma=smoothing_sigma)
            smoothed_frames.append(smoothed)
            del smoothed  # Explicitly delete to free memory
        
        gc.collect()  # Final cleanup before merge
        img_smooth = ants.merge_channels(smoothed_frames)
        del smoothed_frames  # Free the list
        gc.collect()  # Final cleanup after merge
        
        # Resample atlas
        logger.info("Resampling atlas to template space...")
        atlas_template = ants.resample_image_to_target(atlas, template_img, interp='nearestNeighbor')
        
        # Extract timeseries
        regional_ts, regions = extract_timeseries_aparc_aseg(img_smooth, atlas_template, logger)
        
        # Temporal filtering
        regional_ts_filt = temporal_filter(regional_ts, tr=tr, high_pass=high_pass, low_pass=low_pass, logger=logger)
        
        # FC computation
        fc_matrix = compute_fc_matrix(regional_ts_filt, logger=logger)
        
        # Save intermediate results
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f'fc_matrix_scan.npy', fc_matrix)
        np.save(output_dir / 'regional_ts.npy', regional_ts_filt)
        
        logger.info(f"✓ fMRI preprocessing complete")
        logger.info(f"  FC matrix: {fc_matrix.shape}")
        
        return {'fc_matrix': fc_matrix, 'regional_ts': regional_ts_filt}
        
    except Exception as e:
        logger.error(f"fMRI preprocessing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_subject_visit(subject_id, visit_id, fs_data, atlas, output_root, 
                         nifti_root, metadata_index=None, acq_date_mapping=None, expected_scan_counts=None):
    """Process a single (subject, visit) pair"""
    
    logger = setup_logging(str(subject_id), f'fmri_{visit_id}')
    
    try:
        logger.info(f"Processing ({subject_id}, {visit_id})")
        
        # Get expected scan count
        expected_count = None
        if expected_scan_counts:
            expected_count = expected_scan_counts.get((subject_id, visit_id))
            logger.info(f"Expected {expected_count} scan(s) for this visit")
        
        # Find all scans for this visit
        scans_by_session = find_all_scans_for_visit(subject_id, visit_id, nifti_root, acq_date_mapping, expected_count)
        
        if not scans_by_session:
            if expected_count:
                logger.warning(f"Incomplete data for ({subject_id}, {visit_id}) - expected {expected_count} scans, skipping")
            else:
                logger.warning(f"No fMRI scans found for ({subject_id}, {visit_id})")
            return None
        
        logger.info(f"Found {len(scans_by_session)} session(s): {list(scans_by_session.keys())}")
        
        # Estimate total volumes to avoid OOM - skip if too large
        total_volumes = 0
        for session_files in scans_by_session.values():
            for nii_file in session_files:
                try:
                    import nibabel as nib
                    img = nib.load(nii_file)
                    if img.ndim == 4:
                        total_volumes += img.shape[3]
                    else:
                        total_volumes += 1
                except:
                    total_volumes += 1  # Assume 1 volume if can't read
        
        OOM_THRESHOLD = 400  # Downsampling threshold for large scans
        if total_volumes > OOM_THRESHOLD:
            logger.info(f"Large scan ({subject_id}, {visit_id}) - {total_volumes} volumes detected. Will apply downsampling during processing.")
        
        fc_matrices = []
        regional_ts_list = []
        
        # Process each session (multiple scans per session might exist)
        for session_type, nii_files in scans_by_session.items():
            logger.info(f"Processing session: {session_type} ({len(nii_files)} file(s))")
            
            session_fcs = []
            for nii_file in nii_files:
                try:
                    result = preprocess_fmri_complete(
                        nii_file, subject_id, visit_id, fs_data, atlas,
                        output_root / 'fmri_preprocessed' / str(subject_id) / visit_id / session_type,
                        logger, metadata_index=metadata_index, session_type=session_type
                    )
                    session_fcs.append(result['fc_matrix'])
                    regional_ts_list.append(result['regional_ts'])
                except Exception as e:
                    logger.error(f"Failed to process {nii_file.name}: {e}")
                    continue
            
            if session_fcs:
                # Merge multiple scans within same session
                if len(session_fcs) > 1:
                    session_fc = merge_multiple_fc_matrices_fisher(session_fcs, logger)
                else:
                    session_fc = session_fcs[0]
                fc_matrices.append(session_fc)
        
        if not fc_matrices:
            logger.error(f"All processing failed for ({subject_id}, {visit_id})")
            return None
        
        # Merge across sessions
        if len(fc_matrices) > 1:
            logger.info(f"Merging {len(fc_matrices)} session FC matrices")
            final_fc = merge_multiple_fc_matrices_fisher(fc_matrices, logger)
        else:
            final_fc = fc_matrices[0]
        
        # Save final results
        final_output_dir = output_root / 'fmri_final' / str(subject_id) / visit_id
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(final_output_dir / 'fc_matrix.npy', final_fc)
        logger.info(f"✓ Processing complete for ({subject_id}, {visit_id})")
        logger.info(f"  Final FC shape: {final_fc.shape}")
        
        return {
            'subject': subject_id,
            'visit': visit_id,
            'fc_matrix': final_fc,
            'output_dir': final_output_dir
        }
        
    except Exception as e:
        logger.error(f"Error processing ({subject_id}, {visit_id}): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_subject_visit_parallel(args):
    """Wrapper for parallel processing with OOM handling"""
    subject_id, visit_id, fs_data, atlas, output_root, nifti_root, metadata_index, acq_date_mapping, expected_scan_counts = args
    
    try:
        return process_subject_visit(subject_id, visit_id, fs_data, atlas, output_root, nifti_root, metadata_index, acq_date_mapping, expected_scan_counts)
    except MemoryError as e:
        # Log OOM and skip this visit
        import logging
        logger = logging.getLogger('fmri_preprocessing')
        logger.warning(f"OUT OF MEMORY for ({subject_id}, {visit_id}) - skipping this visit")
        gc.collect()  # Try to free some memory
        return None
    except Exception as e:
        import logging
        logger = logging.getLogger('fmri_preprocessing')
        logger.error(f"Error processing ({subject_id}, {visit_id}): {e}")
        return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PPMI fMRI Preprocessing v2 - (subject, visit) pairs')
    parser.add_argument('--subjects', nargs='+', help='Filter to specific subject IDs')
    parser.add_argument('--test', action='store_true', help='Test on one (subject, visit) pair')
    parser.add_argument('--data-root', default=str(Config.DATA_ROOT), help='Data root')
    parser.add_argument('--output-root', default=str(Config.OUTPUT_ROOT), help='Output root')
    parser.add_argument('--atlas-path', default=None, help='Path to APARC+ASEG atlas')
    parser.add_argument('--parallel', type=int, default=None, help='Number of workers')
    parser.add_argument('--metadata-dir', default='/home/emanuele/Desktop/Studi/preprocessing/fMRI_IDA_Metadata/PPMI',
                       help='Metadata directory')
    
    args = parser.parse_args()
    
    # Setup
    Config.DATA_ROOT = Path(args.data_root)
    Config.OUTPUT_ROOT = Path(args.output_root)
    Config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.atlas_path:
        Config.APARC_ASEG_PATH = args.atlas_path
    
    nifti_root = Path('/home/emanuele/Downloads/fMRI_NIfTI')
    
    print("Loading FreeSurfer data...")
    fs_data = load_freesurfer_data(Config.DATA_ROOT)
    print(f"✓ Loaded {len(fs_data)} subjects")
    
    print("Loading metadata...")
    metadata_index = None
    metadata_dir = Path(args.metadata_dir)
    if metadata_dir.exists():
        metadata_index = build_metadata_index(metadata_dir)
    
    print("Loading atlas...")
    atlas = load_aparc_aseg_atlas()
    
    # Get (subject, visit) pairs
    if args.test:
        subject_visit_pairs = [[100007, 'BL']]
        acq_date_mapping = {}
        expected_scan_counts = {}
        print(f"Testing on: {subject_visit_pairs}")
    else:
        csv_path = Path(args.data_root) / 'fMRI_1_28_2026.csv'
        subject_visit_pairs, acq_date_mapping, expected_scan_counts = get_subject_visit_from_csv(csv_path)
        
        if args.subjects:
            subject_visit_pairs = [[s, v] for s, v in subject_visit_pairs if str(s) in args.subjects]
        
        # Skip already processed
        final_dir = Config.OUTPUT_ROOT / 'fmri_final'
        if final_dir.exists():
            processed = set()
            for subj_dir in final_dir.iterdir():
                if subj_dir.is_dir():
                    for visit_dir in subj_dir.iterdir():
                        if visit_dir.is_dir():
                            # Check that fc_matrix.npy actually exists
                            fc_file = visit_dir / 'fc_matrix.npy'
                            if fc_file.exists():
                                processed.add((int(subj_dir.name), visit_dir.name))
            
            subject_visit_pairs = [[s, v] for s, v in subject_visit_pairs if (s, v) not in processed]
            print(f"Found {len(processed)} already processed pairs")
            print(f"Remaining: {len(subject_visit_pairs)}")
    
    if not subject_visit_pairs:
        print("All pairs processed!")
        return
    
    # Convert to list for passing to child processes
    subject_visit_pairs_list = subject_visit_pairs
    
    print(f"\nProcessing {len(subject_visit_pairs_list)} (subject, visit) pairs")
    
    num_cores = args.parallel if args.parallel else 1
    print(f"Using {num_cores} worker(s)")
    if num_cores == 1:
        print("ANTsPy will use all CPU cores for each (subject, visit)")
    print("=" * 80)
    
    task_args = [
        (s, v, fs_data, atlas, Config.OUTPUT_ROOT, nifti_root, metadata_index, acq_date_mapping, expected_scan_counts)
        for s, v in subject_visit_pairs_list
    ]
    
    with Pool(processes=num_cores) as pool:
        results = pool.map(process_subject_visit_parallel, task_args)
    
    success = len([r for r in results if r])
    
    print("\n" + "=" * 80)
    print(f"fMRI PREPROCESSING COMPLETE")
    print(f"Successfully processed: {success}/{len(subject_visit_pairs_list)}")
    print(f"Output: {Config.OUTPUT_ROOT / 'fmri_final'}")

if __name__ == '__main__':
    main()
