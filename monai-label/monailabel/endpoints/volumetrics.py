# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Volumetric metrics endpoint for calculating volume, surface area, and other 
metrics from segmentation masks.
"""

import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

try:
    from scipy import ndimage
    from skimage import measure
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/volumetrics",
    tags=["Volumetrics"],
    responses={
        404: {"description": "Not found"},
        200: {"description": "Volumetric metrics calculated successfully"},
    },
)


class VolumetricMetrics(BaseModel):
    """Model for volumetric metrics response"""
    segment_index: int
    segment_label: str
    volume_mm3: float
    volume_ml: float
    volume_cc: float
    voxel_count: int
    surface_area_mm2: Optional[float] = None
    bounding_box: Optional[Dict[str, int]] = None
    center_of_mass: Optional[List[float]] = None
    sphericity: Optional[float] = None


class VolumetricReport(BaseModel):
    """Model for complete volumetric report"""
    series_instance_uid: str
    spacing: List[float]
    dimensions: List[int]
    segments: List[VolumetricMetrics]
    total_volume_ml: float
    calculation_method: str


def calculate_surface_area_from_mask(mask: np.ndarray, spacing: tuple) -> float:
    """
    Calculate surface area using marching cubes algorithm.
    
    :param mask: Binary mask array (Z, Y, X)
    :param spacing: Voxel spacing (z, y, x) in mm
    :return: Surface area in mm²
    """
    if not SCIPY_AVAILABLE:
        return None
    
    try:
        # Use marching cubes to get surface mesh
        verts, faces, normals, values = measure.marching_cubes(
            mask.astype(float), 
            level=0.5, 
            spacing=spacing
        )
        
        # Calculate surface area from mesh
        # Each face is a triangle, calculate area using cross product
        surface_area = measure.mesh_surface_area(verts, faces)
        return float(surface_area)
    except Exception as e:
        logger.warning(f"Could not calculate surface area: {e}")
        return None


def calculate_sphericity(volume: float, surface_area: float) -> float:
    """
    Calculate sphericity - how spherical the shape is (1.0 = perfect sphere).
    
    Sphericity = (π^(1/3) * (6V)^(2/3)) / A
    Where V is volume and A is surface area
    
    :param volume: Volume in mm³
    :param surface_area: Surface area in mm²
    :return: Sphericity value between 0 and 1
    """
    if surface_area is None or surface_area <= 0 or volume <= 0:
        return None
    
    try:
        sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
        return float(min(sphericity, 1.0))  # Cap at 1.0 due to discretization errors
    except Exception:
        return None


def calculate_bounding_box(mask: np.ndarray) -> Dict[str, int]:
    """
    Calculate bounding box of the mask.
    
    :param mask: Binary mask array (Z, Y, X)
    :return: Dictionary with bounding box coordinates
    """
    try:
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None
        
        return {
            "z_min": int(coords[0].min()),
            "z_max": int(coords[0].max()),
            "y_min": int(coords[1].min()),
            "y_max": int(coords[1].max()),
            "x_min": int(coords[2].min()),
            "x_max": int(coords[2].max()),
            "depth": int(coords[0].max() - coords[0].min() + 1),
            "height": int(coords[1].max() - coords[1].min() + 1),
            "width": int(coords[2].max() - coords[2].min() + 1),
        }
    except Exception as e:
        logger.warning(f"Could not calculate bounding box: {e}")
        return None


def calculate_center_of_mass(mask: np.ndarray, spacing: tuple) -> List[float]:
    """
    Calculate center of mass in physical coordinates (mm).
    
    :param mask: Binary mask array (Z, Y, X)
    :param spacing: Voxel spacing (z, y, x) in mm
    :return: Center of mass coordinates [z, y, x] in mm
    """
    try:
        if SCIPY_AVAILABLE:
            com = ndimage.center_of_mass(mask)
        else:
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                return None
            com = [coords[i].mean() for i in range(3)]
        
        # Convert to physical coordinates
        com_physical = [float(com[i] * spacing[i]) for i in range(3)]
        return com_physical
    except Exception as e:
        logger.warning(f"Could not calculate center of mass: {e}")
        return None


def calculate_metrics_for_segment(
    mask: np.ndarray, 
    segment_index: int,
    segment_label: str,
    spacing: tuple,
    calculate_surface: bool = True
) -> VolumetricMetrics:
    """
    Calculate all volumetric metrics for a single segment.
    
    :param mask: Binary mask for this segment (Z, Y, X)
    :param segment_index: Index of the segment
    :param segment_label: Label name of the segment
    :param spacing: Voxel spacing (z, y, x) in mm
    :param calculate_surface: Whether to calculate surface area (slower)
    :return: VolumetricMetrics object
    """
    # Count voxels
    voxel_count = int(np.sum(mask > 0))
    
    # Calculate voxel volume in mm³
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    
    # Calculate total volume
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0  # 1 ml = 1000 mm³
    volume_cc = volume_ml  # cc and ml are equivalent
    
    # Calculate optional metrics
    surface_area = None
    sphericity = None
    if calculate_surface and voxel_count > 0:
        surface_area = calculate_surface_area_from_mask(mask, spacing)
        if surface_area is not None:
            sphericity = calculate_sphericity(volume_mm3, surface_area)
    
    bounding_box = calculate_bounding_box(mask) if voxel_count > 0 else None
    center_of_mass = calculate_center_of_mass(mask, spacing) if voxel_count > 0 else None
    
    return VolumetricMetrics(
        segment_index=segment_index,
        segment_label=segment_label,
        volume_mm3=round(volume_mm3, 2),
        volume_ml=round(volume_ml, 4),
        volume_cc=round(volume_cc, 4),
        voxel_count=voxel_count,
        surface_area_mm2=round(surface_area, 2) if surface_area else None,
        bounding_box=bounding_box,
        center_of_mass=[round(c, 2) for c in center_of_mass] if center_of_mass else None,
        sphericity=round(sphericity, 4) if sphericity else None,
    )


# Store the last segmentation result for quick access
_cached_segmentation = {
    "series_uid": None,
    "mask": None,
    "spacing": None,
    "labels": {}
}


def cache_segmentation(series_uid: str, mask: np.ndarray, spacing: tuple, labels: dict = None):
    """Cache the segmentation for volumetric calculations."""
    global _cached_segmentation
    _cached_segmentation["series_uid"] = series_uid
    _cached_segmentation["mask"] = mask
    _cached_segmentation["spacing"] = spacing
    _cached_segmentation["labels"] = labels or {}
    logger.info(f"Cached segmentation for series {series_uid}, shape: {mask.shape}, spacing: {spacing}")


def get_cached_segmentation():
    """Get the cached segmentation."""
    return _cached_segmentation


@router.post("/calculate")
async def calculate_volumetrics(
    series_instance_uid: str = Form(None),
    calculate_surface_area: bool = Form(True),
    labels: str = Form("{}"),
) -> JSONResponse:
    """
    Calculate volumetric metrics for the cached segmentation.
    
    This endpoint calculates volume, surface area, and other metrics
    for each segment in the current segmentation mask.
    
    - **series_instance_uid**: Optional series UID to verify correct segmentation
    - **calculate_surface_area**: Whether to calculate surface area (slower but more metrics)
    - **labels**: JSON string mapping segment indices to label names
    
    Returns a VolumetricReport with metrics for each segment.
    """
    try:
        cached = get_cached_segmentation()
        
        if cached["mask"] is None:
            raise HTTPException(
                status_code=404, 
                detail="No segmentation found. Run segmentation first."
            )
        
        if series_instance_uid and cached["series_uid"] != series_instance_uid:
            logger.warning(f"Series mismatch: requested {series_instance_uid}, cached {cached['series_uid']}")
        
        mask = cached["mask"]
        spacing = cached["spacing"]
        
        # Parse labels
        try:
            label_dict = json.loads(labels)
        except json.JSONDecodeError:
            label_dict = cached.get("labels", {})
        
        # Find unique segment indices
        unique_segments = np.unique(mask)
        unique_segments = unique_segments[unique_segments > 0]  # Exclude background (0)
        
        if len(unique_segments) == 0:
            raise HTTPException(
                status_code=404, 
                detail="No segments found in the segmentation mask."
            )
        
        # Calculate metrics for each segment
        segment_metrics = []
        total_volume = 0.0
        
        for seg_idx in unique_segments:
            seg_idx = int(seg_idx)
            segment_mask = (mask == seg_idx).astype(np.uint8)
            label = label_dict.get(str(seg_idx), f"Segment {seg_idx}")
            
            metrics = calculate_metrics_for_segment(
                mask=segment_mask,
                segment_index=seg_idx,
                segment_label=label,
                spacing=spacing,
                calculate_surface=calculate_surface_area
            )
            segment_metrics.append(metrics)
            total_volume += metrics.volume_ml
        
        # Create report
        report = VolumetricReport(
            series_instance_uid=cached["series_uid"] or "unknown",
            spacing=list(spacing),
            dimensions=list(mask.shape),
            segments=segment_metrics,
            total_volume_ml=round(total_volume, 4),
            calculation_method="voxel_counting" if not calculate_surface_area else "voxel_counting_with_marching_cubes"
        )
        
        logger.info(f"Volumetric report: {len(segment_metrics)} segments, total volume: {total_volume:.2f} ml")
        
        return JSONResponse(content=report.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating volumetrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def volumetrics_status() -> JSONResponse:
    """
    Get the status of cached segmentation for volumetric calculations.
    
    Returns information about whether a segmentation is cached and ready
    for volumetric analysis.
    """
    cached = get_cached_segmentation()
    
    if cached["mask"] is None:
        return JSONResponse(content={
            "has_segmentation": False,
            "message": "No segmentation cached. Run segmentation first."
        })
    
    unique_segments = np.unique(cached["mask"])
    unique_segments = unique_segments[unique_segments > 0]
    
    return JSONResponse(content={
        "has_segmentation": True,
        "series_instance_uid": cached["series_uid"],
        "mask_shape": list(cached["mask"].shape),
        "spacing": list(cached["spacing"]) if cached["spacing"] else None,
        "num_segments": len(unique_segments),
        "segment_indices": [int(s) for s in unique_segments],
        "labels": cached.get("labels", {}),
        "scipy_available": SCIPY_AVAILABLE,
        "sitk_available": SITK_AVAILABLE,
    })
