"""
DTI Preprocessing and Structural Connectivity (SC) Construction
================================================================

Complete DTI processing pipeline including:
- Motion and eddy current correction
- Tensor model fitting
- DTI metrics extraction (FA, MD, AD, RD)
- Structural connectivity matrix construction
- Multi-scan averaging

Usage:
    python dti_preprocessing.py --subjects 100007 100018
    python dti_preprocessing.py --test  # Test on one subject
"""

import numpy as np
import ants
import logging
from pathlib import Path
import argparse
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from multiprocessing import Pool, cpu_count
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation, label
import os

from preprocessing_common import (
    Config, setup_logging, load_aparc_aseg_atlas
)
from load_dti_metadata import build_dti_metadata_index, get_dti_parameters
from collections import defaultdict
from datetime import datetime as dt
import pandas as pd

# ============================================================================
# MULTI-SHELL DTI MERGING
# ============================================================================

def merge_dti_scans(scan_list, output_dir, logger):
    """Merge multiple DTI scans (multi-shell) into single 4D volume
    
    Args:
        scan_list: List of dicts with keys 'nii', 'bval', 'bvec'
        output_dir: Directory to save merged files
        logger: Logger instance
    
    Returns:
        Dict with merged 'nii', 'bval', 'bvec' paths
    """
    
    logger.info(f"Merging {len(scan_list)} DTI scans (multi-shell acquisition)...")
    
    try:
        import nibabel as nib
        from scipy.ndimage import zoom
        
        # Load all volumes
        all_volumes = []
        all_bvals = []
        all_bvecs = []
        ref_shape = None
        
        for i, scan in enumerate(scan_list):
            logger.info(f"  Loading scan {i+1}/{len(scan_list)}: {scan['nii'].name}")
            
            # Load NIfTI
            img = nib.load(str(scan['nii']))
            data = img.get_fdata()
            logger.info(f"    Original shape: {data.shape}")
            
            # Ensure data is 4D (some scans may be 3D single volumes)
            if data.ndim == 3:
                data = data[..., np.newaxis]
                logger.info(f"    Expanded to 4D: {data.shape}")
            
            # Store reference shape from first scan
            if ref_shape is None:
                ref_shape = data.shape[:3]
                logger.info(f"    Reference spatial shape: {ref_shape}")
            else:
                # Check if shape matches
                if data.shape[:3] != ref_shape:
                    logger.warning(f"    Shape mismatch: {data.shape[:3]} vs reference {ref_shape}")
                    logger.info(f"    Resampling to match reference shape...")
                    import time
                    resample_start = time.time()
                    # Calculate zoom factors for spatial dimensions only
                    zoom_factors = [ref_shape[i] / data.shape[i] for i in range(3)] + [1.0]
                    logger.info(f"    Zoom factors: {[f'{z:.3f}' for z in zoom_factors[:3]]}")
                    # Resample using linear interpolation (faster than cubic)
                    data_resampled = zoom(data, zoom_factors, order=1)
                    resample_time = time.time() - resample_start
                    logger.info(f"    ✓ Resampled in {resample_time:.2f}s to: {data_resampled.shape}")
                    data = data_resampled
                else:
                    logger.info(f"    Shape matches reference")
            
            all_volumes.append(data)
            
            # Load bvals
            bvals = np.loadtxt(scan['bval'])
            if bvals.ndim == 0:
                bvals = np.array([bvals])
            all_bvals.append(bvals)
            
            # Load bvecs
            bvecs = np.loadtxt(scan['bvec'])
            if bvecs.ndim == 1:
                bvecs = bvecs.reshape(3, 1)
            all_bvecs.append(bvecs)
        
        # Concatenate along 4th dimension (time/gradient)
        logger.info("  Concatenating volumes...")
        merged_data = np.concatenate(all_volumes, axis=-1)
        merged_bvals = np.concatenate(all_bvals)
        merged_bvecs = np.concatenate(all_bvecs, axis=1)
        
        logger.info(f"  Merged shape: {merged_data.shape}")
        logger.info(f"  Total gradients: {merged_data.shape[-1]}")
        logger.info(f"  B-values range: {merged_bvals.min()}-{merged_bvals.max()}")
        
        # Save merged files
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save NIfTI (use first scan as reference for affine/header)
        ref_img = nib.load(str(scan_list[0]['nii']))
        merged_img = nib.Nifti1Image(merged_data, ref_img.affine, ref_img.header)
        merged_nii_path = output_dir / 'merged_dti.nii'
        nib.save(merged_img, str(merged_nii_path))
        
        # Save bvals
        merged_bval_path = output_dir / 'merged_dti.bval'
        np.savetxt(merged_bval_path, merged_bvals.reshape(1, -1), fmt='%d')
        
        # Save bvecs
        merged_bvec_path = output_dir / 'merged_dti.bvec'
        np.savetxt(merged_bvec_path, merged_bvecs, fmt='%.6f')
        
        logger.info(f"✓ Merged DTI data saved to {output_dir}")
        
        return {
            'nii': merged_nii_path,
            'bval': merged_bval_path,
            'bvec': merged_bvec_path
        }
        
    except Exception as e:
        logger.error(f"DTI merging failed: {e}")
        raise

# ============================================================================
# DTI PREPROCESSING
# ============================================================================

def extract_b0_and_mask(dti_img, bvals, logger, mask_params=None):
    """Extract B0 image and create brain mask"""
    
    logger.info("Extracting B0 and creating brain mask...")
    logger.info(f"  DTI image dimension: {dti_img.dimension}, shape: {dti_img.shape}")
    
    try:
        b0_idx = np.where(np.array(bvals) == 0)[0]
        
        if len(b0_idx) == 0:
            logger.warning("No B0 image found, using first volume")
            b0_idx = 0
        else:
            b0_idx = b0_idx[0]
        
        # For 4D images, extract the B0 volume
        if dti_img.dimension == 4:
            b0 = ants.slice_image(dti_img, axis=3, idx=int(b0_idx))
        else:
            # If already 3D (shouldn't happen), use as is
            logger.warning("DTI image is already 3D, using entire image as B0")
            b0 = dti_img
        
        # Brain extraction with fallback
        try:
            # Simple robust threshold-based masking
            b0_data = b0.numpy()
            threshold_percentile = 20
            erosion_iter = 1
            dilation_iter = 2
            if mask_params:
                threshold_percentile = mask_params.get('threshold_percentile', threshold_percentile)
                erosion_iter = mask_params.get('erosion_iter', erosion_iter)
                dilation_iter = mask_params.get('dilation_iter', dilation_iter)

            threshold = np.percentile(b0_data[b0_data > 0], threshold_percentile)
            mask = b0_data > threshold
            
            # Morphological operations
            mask = binary_fill_holes(mask)
            mask = binary_erosion(mask, iterations=erosion_iter)
            mask = binary_dilation(mask, iterations=dilation_iter)
            
            # Remove small objects
            labeled, num_labels = label(mask)
            if num_labels > 0:
                largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
                mask = (labeled == largest)
            
            brain_mask = b0.new_image_like(mask.astype(float))
            logger.info("✓ Brain mask created")
        
        except Exception as e:
            logger.error(f"Brain mask creation failed: {e}")
            # Emergency fallback: simple binary threshold
            b0_data = b0.numpy()
            threshold = np.percentile(b0_data[b0_data > 0], 30)
            mask = b0_data > threshold
            brain_mask = b0.new_image_like(mask.astype(float))
            logger.info("✓ Emergency fallback brain mask generated")
        
        logger.info("✓ B0 extracted and mask created")
        return b0, brain_mask
        
    except Exception as e:
        logger.error(f"B0 extraction failed: {e}")
        raise

def fit_tensor_model(dti_img, bvals, bvecs, brain_mask, logger):
    """Fit DTI tensor model using DIPY"""
    
    logger.info("Fitting tensor model...")
    
    try:
        gtab = gradient_table(bvals, bvecs)
        tenmodel = TensorModel(gtab)
        
        # Convert to numpy
        data = dti_img.numpy()
        mask_data = brain_mask.numpy() > 0
        
        logger.info(f"  Data shape: {data.shape}")
        logger.info(f"  Mask shape: {mask_data.shape}")
        
        logger.info("  Fitting (this may take a while)...")
        tenfit = tenmodel.fit(data, mask=mask_data)
        
        logger.info("✓ Tensor model fitted")
        return tenfit
        
    except Exception as e:
        logger.error(f"Tensor fitting failed: {e}")
        raise

def extract_dti_metrics(tenfit, logger):
    """Extract DTI metrics: FA, MD, AD, RD"""
    
    logger.info("Extracting DTI metrics...")
    
    try:
        metrics = {
            'fa': tenfit.fa,
            'md': tenfit.md,
            'ad': tenfit.ad,
            'rd': tenfit.rd
        }
        
        logger.info(f"✓ Metrics extracted")
        for name, data in metrics.items():
            logger.info(f"  {name.upper()}: min={np.nanmin(data):.3f}, max={np.nanmax(data):.3f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metric extraction failed: {e}")
        raise

def build_sc_from_fa(fa_data, atlas, atlas_labels, logger):
    """Build SC matrix from FA correlations
    
    Args:
        fa_data: FA values in template space
        atlas: Resampled atlas in template space
        atlas_labels: List of valid APARC+ASEG labels (113 regions)
        logger: Logger instance
    """
    
    logger.info("Building SC matrix from FA...")
    
    try:
        atlas_data = atlas.numpy().astype(int)
        
        # Use provided valid labels (113 regions)
        if atlas_labels is not None and len(atlas_labels) > 0:
            regions = atlas_labels
            logger.info(f"  Using {len(regions)} valid APARC+ASEG labels from atlas")
        else:
            regions = np.unique(atlas_data)
            regions = regions[regions > 0]
            logger.warning(f"  No atlas_labels provided, using all {len(regions)} unique labels from resampled atlas")
        
        n_regions = len(regions)
        sc_matrix = np.zeros((n_regions, n_regions))
        
        fa_data = np.nan_to_num(fa_data, nan=0)
        
        for i, region_i in enumerate(regions):
            if i % 20 == 0:
                logger.info(f"  Processing region {i}/{n_regions}")
            
            mask_i = (atlas_data == region_i)
            
            # Skip if region not found in resampled atlas
            if not np.any(mask_i):
                continue
            
            fa_i = fa_data[mask_i]
            
            for j, region_j in enumerate(regions):
                mask_j = (atlas_data == region_j)
                
                # Skip if region not found in resampled atlas
                if not np.any(mask_j):
                    continue
                
                fa_j = fa_data[mask_j]
                
                if len(fa_i) > 0 and len(fa_j) > 0:
                    connectivity = np.mean(fa_i) * np.mean(fa_j)
                    sc_matrix[i, j] = connectivity
        
        # Symmetrize
        sc_matrix = (sc_matrix + sc_matrix.T) / 2
        
        logger.info(f"✓ SC matrix built: {sc_matrix.shape}")
        return sc_matrix
        
    except Exception as e:
        logger.error(f"SC matrix building failed: {e}")
        raise

def motion_correct_dti(dti_img, bvals, logger):
    """Motion and eddy current correction for DTI"""
    
    logger.info("Starting DTI motion/eddy correction...")
    
    try:
        # Find B0 volumes as reference
        b0_idx = np.where(bvals == 0)[0]
        if len(b0_idx) == 0:
            b0_idx = [0]
            logger.warning("No B0 found, using first volume as reference")
        
        # Extract reference B0
        ref_b0 = ants.slice_image(dti_img, axis=3, idx=int(b0_idx[0]))
        
        # Register all volumes to B0
        n_volumes = dti_img.shape[-1]
        corrected_data = []
        
        for i in range(n_volumes):
            if i % 10 == 0:
                logger.info(f"  Correcting volume {i}/{n_volumes}")
            
            vol = ants.slice_image(dti_img, axis=3, idx=i)
            
            # Skip registration for B0 volumes
            if i in b0_idx:
                corrected_data.append(vol.numpy())
            else:
                # Rigid registration to correct motion and eddy currents
                reg = ants.registration(
                    fixed=ref_b0,
                    moving=vol,
                    type_of_transform='Affine',
                    verbose=False
                )
                corrected_data.append(reg['warpedmovout'].numpy())
        
        # Stack back to 4D
        corrected_4d = np.stack(corrected_data, axis=-1)
        
        # Convert to ANTs image
        dti_corrected = ants.from_numpy(
            corrected_4d,
            origin=tuple(dti_img.origin),
            spacing=tuple(dti_img.spacing),
            direction=dti_img.direction
        )
        
        logger.info("✓ Motion/eddy correction complete")
        return dti_corrected
        
    except Exception as e:
        logger.error(f"DTI motion correction failed: {e}")
        raise

def denoise_dti(dti_img, logger):
    """Denoise DTI using non-local means"""
    
    logger.info("Denoising DTI...")
    
    try:
        # Extract each volume manually (split_channels doesn't work for 4D)
        n_volumes = dti_img.shape[-1]
        denoised_data = []
        
        for i in range(n_volumes):
            if i % 10 == 0:
                logger.info(f"  Denoising volume {i}/{n_volumes}")
            # Extract volume
            vol = ants.slice_image(dti_img, axis=3, idx=i)
            # Denoise
            denoised = ants.denoise_image(vol, noise_model='Rician')
            denoised_data.append(denoised.numpy())
        
        # Stack back into 4D
        denoised_4d = np.stack(denoised_data, axis=-1)
        
        # Convert to ANTs image
        dti_denoised = ants.from_numpy(
            denoised_4d,
            origin=tuple(dti_img.origin),
            spacing=tuple(dti_img.spacing),
            direction=dti_img.direction
        )
        
        logger.info("✓ Denoising complete")
        return dti_denoised
        
    except Exception as e:
        logger.warning(f"Denoising failed, skipping: {e}")
        return dti_img

def register_to_template(img, template=None, logger=None):
    """Register image to template (MNI or FreeSurfer)"""
    
    if logger:
        logger.info("Starting registration to template...")
    
    try:
        # Use ANTsPy default MNI template
        if template is None:
            template = ants.get_ants_data('mni')
            template_img = ants.image_read(template)
        else:
            template_img = ants.image_read(str(template))
        
        # SyN registration
        reg = ants.registration(fixed=template_img, moving=img,
                               type_of_transform='SyNRA', verbose=False)
        
        if logger:
            logger.info("✓ Registration complete")
        
        return reg['warpedmovout'], reg['fwdtransforms']
        
    except Exception as e:
        if logger:
            logger.error(f"Registration failed: {e}")
        raise

def preprocess_dti_complete(dti_file, bval_file, bvec_file, subject_id, 
                           atlas, output_dir, logger, atlas_labels=None,
                           metadata_index=None, scan_type=None, scan_date=None):
    """Complete DTI preprocessing pipeline
    
    Args:
        atlas_labels: List of valid atlas labels (113 regions) to use for SC matrix
    """
    
    logger.info(f"=" * 80)
    logger.info(f"DTI PREPROCESSING: {dti_file.name}")
    logger.info(f"=" * 80)
    
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info(f"Loading DTI from {dti_file}")
        dti_img = ants.image_read(str(dti_file))
        bvals = np.loadtxt(bval_file)
        bvecs = np.loadtxt(bvec_file)
        
        logger.info(f"  Shape: {dti_img.shape}")
        logger.info(f"  B-values: {bvals.shape}")
        logger.info(f"  B-vectors: {bvecs.shape}")

        # Load metadata-derived parameters when available
        params = None
        if metadata_index is not None and scan_type:
            params, _ = get_dti_parameters(int(subject_id), scan_type, scan_date, metadata_index)

        tr = params.get('TR') if params else None
        te = params.get('TE') if params else None
        flip_angle = params.get('FlipAngle') if params else None
        grad_dirs = params.get('GradientDirections') if params else None

        voxel_sizes = None
        if params and params.get('PixelSizeX') and params.get('PixelSizeY') and params.get('SliceThickness'):
            voxel_sizes = (float(params['PixelSizeX']), float(params['PixelSizeY']), float(params['SliceThickness']))
        else:
            voxel_sizes = tuple(dti_img.spacing[:3])

        mean_spacing = float(np.mean(voxel_sizes))
        erosion_iter = max(1, int(round(2.0 / mean_spacing)))
        dilation_iter = max(1, int(round(3.0 / mean_spacing)))

        logger.info(f"  Voxel size: {voxel_sizes}")
        if tr is not None:
            logger.info(f"  TR: {tr} sec")
        if te is not None:
            logger.info(f"  TE: {te} sec")
        if flip_angle is not None:
            logger.info(f"  FlipAngle: {flip_angle} deg")
        if grad_dirs is not None:
            logger.info(f"  Gradient Directions (metadata): {grad_dirs}")
            logger.info(f"  Gradients (data): {len(bvals)}")
        
        # Motion and eddy current correction
        dti_corrected = motion_correct_dti(dti_img, bvals, logger)
        
        # Denoise
        dti_denoised = denoise_dti(dti_corrected, logger)
        
        # Extract B0 and mask from corrected data
        b0, brain_mask = extract_b0_and_mask(
            dti_denoised, bvals, logger,
            mask_params={
                'threshold_percentile': 20,
                'erosion_iter': erosion_iter,
                'dilation_iter': dilation_iter
            }
        )
        
        # Fit tensor on preprocessed data
        tenfit = fit_tensor_model(dti_denoised, bvals, bvecs, brain_mask, logger)
        
        # Extract metrics
        dti_metrics = extract_dti_metrics(tenfit, logger)
        
        # Register to template
        logger.info("Registering FA to template...")
        fa_img = ants.from_numpy(
            dti_metrics['fa'],
            origin=tuple(b0.origin[:3]),
            spacing=tuple(b0.spacing[:3]),
            direction=np.array(b0.direction)[:3, :3]
        )
        fa_reg, fa_transforms = register_to_template(fa_img, logger=logger)
        
        # Register atlas to template, then resample into FA-registered space
        atlas_reg, _ = register_to_template(atlas, logger=logger)
        atlas_in_template = ants.resample_image_to_target(atlas_reg, fa_reg, interp='nearestNeighbor')
        # Build SC in template space using FA-registered data
        sc_matrix = build_sc_from_fa(fa_reg.numpy(), atlas_in_template, atlas_labels, logger)
        
        # Save outputs
        np.save(output_dir / 'fa.npy', dti_metrics['fa'])
        np.save(output_dir / 'md.npy', dti_metrics['md'])
        np.save(output_dir / 'ad.npy', dti_metrics['ad'])
        np.save(output_dir / 'rd.npy', dti_metrics['rd'])
        np.save(output_dir / 'sc_matrix.npy', sc_matrix)
        
        ants.image_write(fa_img, str(output_dir / 'fa.nii.gz'))
        ants.image_write(fa_reg, str(output_dir / 'fa_registered.nii.gz'))
        
        logger.info(f"✓ DTI preprocessing complete")
        logger.info(f"  SC matrix: {sc_matrix.shape}")
        logger.info(f"  Outputs saved to {output_dir}")
        
        return {
            'metrics': dti_metrics,
            'sc_matrix': sc_matrix,
            'regions': np.unique(atlas.numpy())[1:],  # Remove 0 (background)
            'output_dir': output_dir
        }
        
    except Exception as e:
        logger.error(f"DTI preprocessing failed: {e}")
        raise

# ============================================================================
# SUBJECT PROCESSING
# ============================================================================

def load_dti_metadata(data_root):
    """Load DTI metadata CSV to get visit information"""
    # Try data_root first
    csv_files = list(data_root.glob('DTI_*.csv'))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        return df
    
    # Try Config.DATA_ROOT (where preprocessing files are)
    csv_files = list(Config.DATA_ROOT.glob('DTI_*.csv'))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        return df
    
    return None

def get_visit_for_scan(metadata_df, subject_id, scan_date):
    """Get visit ID (BL, V04, etc.) for a scan based on acquisition date
    
    Args:
        metadata_df: DataFrame with DTI metadata
        subject_id: Subject ID
        scan_date: Date string like "2023-03-29"
    
    Returns:
        Visit ID (e.g., 'BL', 'V06') or scan_date if not found
    """
    if metadata_df is None:
        return scan_date  # Fallback to date
    
    try:
        # Filter by subject
        subj_data = metadata_df[metadata_df['Subject'] == int(subject_id)]
        
        if subj_data.empty:
            return scan_date
        
        # Convert scan_date to match CSV format (m/d/yyyy)
        from datetime import datetime
        dt_obj = datetime.strptime(scan_date, '%Y-%m-%d')
        csv_date = dt_obj.strftime('%-m/%-d/%Y')  # e.g., "3/29/2023"
        
        # Find matching visit
        matching = subj_data[subj_data['Acq Date'] == csv_date]
        
        if not matching.empty:
            return matching.iloc[0]['Visit']
        
        return scan_date  # Fallback
        
    except Exception:
        return scan_date  # Fallback

def find_dti_scans(subject_id, data_root):
    """Find all DTI scans for subject, grouped by visit (using CSV metadata)"""
    
    # Load metadata CSV for visit information
    metadata_df = load_dti_metadata(data_root)
    
    # Handle two possible directory structures:
    # 1. data_root/DTI/subject_id/  (Config.DATA_ROOT structure)
    # 2. data_root/subject_id/      (DTI_all_nii structure)
    base_path = data_root / 'DTI' / str(subject_id)
    if not base_path.exists():
        base_path = data_root / str(subject_id)
    
    visits = defaultdict(list)  # Group scans by visit ID (BL, V04, etc.)
    
    if not base_path.exists():
        return visits
    
    # Iterate through all scan types (DTI_B700, DTI_B1000, etc.)
    for scan_type_dir in base_path.iterdir():
        if not scan_type_dir.is_dir():
            continue
        
        # Look for timestamped folders (e.g., 2023-03-29_15_40_22.0)
        for date_dir in scan_type_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            # Extract scan date (use first part as date)
            scan_date = date_dir.name.split('_')[0]  # e.g., "2023-03-29"
            
            # Get proper visit ID from CSV metadata
            visit_id = get_visit_for_scan(metadata_df, subject_id, scan_date)
            
            # Find actual scan files
            for scan_dir in date_dir.iterdir():
                if not scan_dir.is_dir():
                    continue
                
                nii_files = list(scan_dir.glob('*.nii'))
                bval_files = list(scan_dir.glob('*.bval'))
                bvec_files = list(scan_dir.glob('*.bvec'))
                
                if nii_files and bval_files and bvec_files:
                    # Check if this is raw DTI (4D) vs derived map (3D)
                    # Derived maps like FA, MD, TRACEW are 3D single volumes
                    # Raw DTI is 4D (multiple gradients)
                    try:
                        import nibabel as nib
                        nii_data = nib.load(str(nii_files[0])).get_fdata()
                        
                        if nii_data.ndim < 4:
                            # Skip 3D files (derived maps)
                            continue
                        
                        scan_info = {
                            'nii': nii_files[0],
                            'bval': bval_files[0],
                            'bvec': bvec_files[0],
                            'scan_type': scan_type_dir.name,
                            'date': scan_date,
                            'visit': visit_id
                        }
                        visits[visit_id].append(scan_info)
                    except Exception as e:
                        # If can't load, skip this file
                        continue
    
    return visits

def process_subject_dti(subject_id, atlas, output_root, data_root, metadata_index=None, visit_id=None):
    """Process all DTI scans for subject (merge multi-shell per visit)"""
    
    logger = setup_logging(subject_id, 'dti')
    
    logger.info(f"Processing DTI for subject {subject_id}")
    
    dti_visits = find_dti_scans(subject_id, data_root)
    
    if not dti_visits:
        logger.warning(f"No DTI scans found for subject {subject_id}")
        return None
    
    logger.info(f"Found {len(dti_visits)} visit(s): {list(dti_visits.keys())}")
    
    sc_matrices = []
    
    for visit_id, scan_list in dti_visits.items():
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing visit: {visit_id} ({len(scan_list)} scan(s))")
            logger.info(f"  Scan types: {[s['scan_type'] for s in scan_list]}")
            
            output_dir = output_root / str(subject_id) / visit_id
            
            # Merge multi-shell scans if more than one
            if len(scan_list) > 1:
                merge_dir = output_dir / 'merged'
                merged_files = merge_dti_scans(scan_list, merge_dir, logger)
                nii_file = merged_files['nii']
                bval_file = merged_files['bval']
                bvec_file = merged_files['bvec']
            else:
                # Single scan - use directly
                logger.info("  Single scan - no merging needed")
                nii_file = scan_list[0]['nii']
                bval_file = scan_list[0]['bval']
                bvec_file = scan_list[0]['bvec']
            
            # Process the (merged or single) DTI scan
            # Extract valid labels from atlas if available
            atlas_labels = atlas.valid_labels if hasattr(atlas, 'valid_labels') else None
            result = preprocess_dti_complete(
                nii_file, bval_file, bvec_file,
                subject_id, atlas, output_dir, logger,
                atlas_labels=atlas_labels,
                metadata_index=metadata_index,
                scan_type=scan_list[0].get('scan_type'),
                scan_date=scan_list[0].get('date')
            )
            
            sc_matrices.append(result['sc_matrix'])
            
        except Exception as e:
            logger.error(f"Failed to process visit {visit_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    if not sc_matrices:
        logger.error(f"All DTI processing failed for subject {subject_id}")
        return None
    
    logger.info(f"✓ DTI processing complete for subject {subject_id}")
    logger.info(f"  Processed {len(sc_matrices)} visit(s), each with separate SC matrix")
    
    # Return results for all visits (no averaging)
    return {
        'subject_id': subject_id,
        'n_visits': len(sc_matrices),
        'sc_matrices': sc_matrices  # List of SC matrices, one per visit
    }

# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def process_subject_parallel(args_tuple):
    """Wrapper for parallel processing of single subject"""
    subject_id, atlas, output_root, data_root, atlas_labels, metadata_index, visit_id = args_tuple
    
    # Restore valid_labels to atlas (may have been lost during pickling)
    if atlas_labels is not None and not hasattr(atlas, 'valid_labels'):
        atlas.valid_labels = atlas_labels
    
    try:
        result = process_subject_dti(subject_id, atlas, output_root, data_root, metadata_index, visit_id=visit_id)
        return {'subject': subject_id, 'result': result}
    except Exception as e:
        print(f"✗ Error processing subject {subject_id}: {e}")
        return {'subject': subject_id, 'result': None}

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main DTI preprocessing pipeline"""
    
    parser = argparse.ArgumentParser(description='PPMI DTI Preprocessing')
    parser.add_argument('--subjects', nargs='+', help='Subject IDs to process')
    parser.add_argument('--test', action='store_true', help='Test on one subject')
    parser.add_argument('--data-root', default=str(Config.DATA_ROOT),
                       help='Path to data root')
    parser.add_argument('--output-root', default=str(Config.OUTPUT_ROOT),
                       help='Path to output root')
    parser.add_argument('--atlas-path', default=None,
                        help='Explicit path to APARC+ASEG atlas (e.g., aparc+aseg.mgz)')
    parser.add_argument('--visit', default=None,
                        help='Visit ID to process (e.g., BL, V04). If set, only that visit is processed.')
    parser.add_argument('--parallel', type=int, default=None,
                        help='Number of parallel cores (default: all available)')
    parser.add_argument('--metadata-dir', default='/home/preprocessing/DTI_IDA_Metadata',
                        help='Path to DTI IDA XML metadata directory')
    
    args = parser.parse_args()
    
    # Setup
    Config.DATA_ROOT = Path(args.data_root)
    Config.OUTPUT_ROOT = Path(args.output_root)
    Config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.atlas_path:
        Config.APARC_ASEG_PATH = args.atlas_path
    
    # Load atlas
    print("Loading APARC+ASEG atlas...")
    atlas = load_aparc_aseg_atlas()
    if atlas is None:
        print("ERROR: Could not load atlas")
        return

    # Load DTI IDA metadata (for acquisition parameters)
    metadata_index = None
    metadata_dir = Path(args.metadata_dir)
    if metadata_dir.exists():
        print("Loading DTI IDA metadata...")
        metadata_index = build_dti_metadata_index(metadata_dir)
    else:
        print(f"WARNING: metadata directory not found: {metadata_dir}")
    
    # Get subject list
    if args.test:
        subject_ids = ['100007']
        print(f"Testing on subject: {subject_ids}")
    elif args.subjects:
        subject_ids = args.subjects
    else:
        # Use all subjects with DTI - check both /DTI and direct structure
        dti_path = Config.DATA_ROOT / 'DTI'
        if dti_path.exists():
            subject_ids = [d.name for d in dti_path.iterdir() if d.is_dir()]
        else:
            # Data root contains subjects directly (no DTI subfolder)
            subject_ids = [d.name for d in Config.DATA_ROOT.iterdir() if d.is_dir() and d.name.isdigit()]
    
    print(f"\nProcessing {len(subject_ids)} subjects")
    
    # Determine number of parallel cores
    num_cores = args.parallel if args.parallel else cpu_count()
    print(f"Using {num_cores} cores for parallel processing")
    print("=" * 80)
    
    # Extract valid labels before parallel processing (ANTsImage attributes may not survive pickling)
    atlas_labels = atlas.valid_labels if hasattr(atlas, 'valid_labels') else None
    
    # Prepare arguments for parallel processing
    task_args = [
        (subj_id, atlas, Config.OUTPUT_ROOT, Config.DATA_ROOT, atlas_labels, metadata_index, args.visit)
        for subj_id in subject_ids
    ]
    
    # Process in parallel
    results = {'dti': []}
    with Pool(processes=num_cores) as pool:
        task_results = pool.map(process_subject_parallel, task_args)
    
    # Collect results
    for result in task_results:
        if result['result']:
            results['dti'].append(result['result'])
    
    # Summary
    print("\n" + "=" * 80)
    print("DTI PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"DTI processed: {len(results['dti'])} subjects")
    print(f"Output directory: {Config.OUTPUT_ROOT}")
    
    return results

if __name__ == '__main__':
    main()
