import checkIfPerpendicular from './checkIfPerpendicular.js';
import { utilities } from '@cornerstonejs/core';

// FIX: Normalize orientation values to handle scientific notation corruption bug
// When values like 3.86e-10 are corrupted to 3.86 during DICOM SEG save/load,
// we can fix this by normalizing near-zero values in the reference and comparing
function normalizeOrientation(iop, reference) {
  if (!iop || !reference || iop.length !== 6 || reference.length !== 6) {
    return iop;
  }
  const normalized = [...iop];
  for (let i = 0; i < 6; i++) {
    const refVal = reference[i];
    const segVal = iop[i];
    // If reference is essentially 0 (< 1e-6) but SEG has a value > 1,
    // this is likely the scientific notation corruption bug - fix it
    if (Math.abs(refVal) < 1e-6 && Math.abs(segVal) > 1) {
      console.warn(`[checkOrientation] Correcting corrupted orientation at index ${i}: ${segVal} -> 0`);
      normalized[i] = 0;
    }
  }
  return normalized;
}

function checkOrientation(multiframe, validOrientations, sourceDataDimensions, tolerance) {
  const {
    SharedFunctionalGroupsSequence,
    PerFrameFunctionalGroupsSequence
  } = multiframe;
  const sharedImageOrientationPatient = SharedFunctionalGroupsSequence.PlaneOrientationSequence ? SharedFunctionalGroupsSequence.PlaneOrientationSequence.ImageOrientationPatient : undefined;
  const PerFrameFunctionalGroups = PerFrameFunctionalGroupsSequence[0];
  let iop = sharedImageOrientationPatient || PerFrameFunctionalGroups.PlaneOrientationSequence.ImageOrientationPatient;
  
  // FIX: Normalize IOP using reference orientation to fix scientific notation corruption
  if (validOrientations && validOrientations[0]) {
    iop = normalizeOrientation(iop, validOrientations[0]);
  }
  
  const inPlane = validOrientations.some(operation => utilities.isEqual(iop, operation, tolerance));
  if (inPlane) {
    return "Planar";
  }
  if (checkIfPerpendicular(iop, validOrientations[0], tolerance) && sourceDataDimensions.includes(multiframe.Rows) && sourceDataDimensions.includes(multiframe.Columns)) {
    return "Perpendicular";
  }
  return "Oblique";
}

export { checkOrientation as default };
