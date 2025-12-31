import dcmjs from 'dcmjs';
import { utils } from '@ohif/core';
import { metaData, triggerEvent, eventTarget } from '@cornerstonejs/core';
import { CONSTANTS, segmentation as cstSegmentation } from '@cornerstonejs/tools';
import { adaptersSEG, Enums } from '@cornerstonejs/adapters';

import { SOPClassHandlerId } from './id';
import { dicomlabToRGB } from './utils/dicomlabToRGB';

const sopClassUids = ['1.2.840.10008.5.1.4.1.1.66.4'];

const loadPromises = {};

function _getDisplaySetsFromSeries(
  instances,
  servicesManager: AppTypes.ServicesManager,
  extensionManager
) {
  const instance = instances[0];

  // DEBUG: Log SEG display set creation
  console.log('[SEG getSopClassHandler] Creating display set from SEG series');
  console.log('[SEG getSopClassHandler] Instance:', {
    StudyInstanceUID: instance.StudyInstanceUID,
    SeriesInstanceUID: instance.SeriesInstanceUID,
    SOPInstanceUID: instance.SOPInstanceUID,
    SeriesDescription: instance.SeriesDescription,
  });

  const {
    StudyInstanceUID,
    SeriesInstanceUID,
    SOPInstanceUID,
    SeriesDescription,
    SeriesNumber,
    SeriesDate,
    SOPClassUID,
    wadoRoot,
    wadoUri,
    wadoUriRoot,
  } = instance;

  const displaySet = {
    Modality: 'SEG',
    loading: false,
    isReconstructable: true, // by default for now since it is a volumetric SEG currently
    displaySetInstanceUID: utils.guid(),
    SeriesDescription,
    SeriesNumber,
    SeriesDate,
    SOPInstanceUID,
    SeriesInstanceUID,
    StudyInstanceUID,
    SOPClassHandlerId,
    SOPClassUID,
    referencedImages: null,
    referencedSeriesInstanceUID: null,
    referencedDisplaySetInstanceUID: null,
    isDerivedDisplaySet: true,
    isLoaded: false,
    isHydrated: false,
    segments: {},
    sopClassUids,
    instance,
    instances: [instance],
    wadoRoot,
    wadoUriRoot,
    wadoUri,
    isOverlayDisplaySet: true,
  };

  const referencedSeriesSequence = instance.ReferencedSeriesSequence;

  if (!referencedSeriesSequence) {
    console.error('[SEG getSopClassHandler] ReferencedSeriesSequence is missing for the SEG');
    console.error('[SEG getSopClassHandler] Instance data:', instance);
    // Return empty array instead of undefined to avoid breaking the display set service
    return [];
  }

  const referencedSeries = referencedSeriesSequence[0] || referencedSeriesSequence;

  displaySet.referencedImages = instance.ReferencedSeriesSequence.ReferencedInstanceSequence;
  displaySet.referencedSeriesInstanceUID = referencedSeries.SeriesInstanceUID;
  const { displaySetService } = servicesManager.services;
  const referencedDisplaySets = displaySetService.getDisplaySetsForSeries(
    displaySet.referencedSeriesInstanceUID
  );

  const referencedDisplaySet = referencedDisplaySets[0];

  if (!referencedDisplaySet) {
    // subscribe to display sets added which means at some point it will be available
    const { unsubscribe } = displaySetService.subscribe(
      displaySetService.EVENTS.DISPLAY_SETS_ADDED,
      ({ displaySetsAdded }) => {
        // here we can also do a little bit of search, since sometimes DICOM SEG
        // does not contain the referenced display set uid , and we can just
        // see which of the display sets added is more similar and assign it
        // to the referencedDisplaySet
        const addedDisplaySet = displaySetsAdded[0];
        if (addedDisplaySet.SeriesInstanceUID === displaySet.referencedSeriesInstanceUID) {
          displaySet.referencedDisplaySetInstanceUID = addedDisplaySet.displaySetInstanceUID;
          unsubscribe();
        }
      }
    );
  } else {
    displaySet.referencedDisplaySetInstanceUID = referencedDisplaySet.displaySetInstanceUID;
  }

  displaySet.load = async ({ headers }) =>
    await _load(displaySet, servicesManager, extensionManager, headers);

  // DEBUG: Log successful display set creation
  console.log('[SEG getSopClassHandler] Successfully created SEG display set:', {
    displaySetInstanceUID: displaySet.displaySetInstanceUID,
    SeriesDescription: displaySet.SeriesDescription,
    SeriesInstanceUID: displaySet.SeriesInstanceUID,
    referencedSeriesInstanceUID: displaySet.referencedSeriesInstanceUID,
    referencedDisplaySetInstanceUID: displaySet.referencedDisplaySetInstanceUID,
    excludeFromThumbnailBrowser: displaySet.excludeFromThumbnailBrowser,
  });

  return [displaySet];
}

function _load(
  segDisplaySet,
  servicesManager: AppTypes.ServicesManager,
  extensionManager,
  headers
) {
  const { SOPInstanceUID } = segDisplaySet;
  const { segmentationService } = servicesManager.services;

  if (
    (segDisplaySet.loading || segDisplaySet.isLoaded) &&
    loadPromises[SOPInstanceUID] &&
    _segmentationExists(segDisplaySet)
  ) {
    console.log('[SEG _load] Already loading or loaded:', SOPInstanceUID);
    return loadPromises[SOPInstanceUID];
  }

  console.log('[SEG _load] Starting load for:', SOPInstanceUID);
  segDisplaySet.loading = true;

  // We don't want to fire multiple loads, so we'll wait for the first to finish
  // and also return the same promise to any other callers.
  loadPromises[SOPInstanceUID] = new Promise(async (resolve, reject) => {
    if (!segDisplaySet.segments || Object.keys(segDisplaySet.segments).length === 0) {
      try {
        console.log('[SEG _load] Loading segments...');
        await _loadSegments({
          extensionManager,
          servicesManager,
          segDisplaySet,
          headers,
        });
        console.log('[SEG _load] Segments loaded successfully');
        console.log('[SEG _load] segDisplaySet after load:', {
          hasLabelMapImages: !!segDisplaySet.labelMapImages,
          labelMapImagesLength: segDisplaySet.labelMapImages?.length,
          hasSegMetadata: !!segDisplaySet.segMetadata,
        });
      } catch (e) {
        console.error('[SEG _load] Failed to load segments:', e);
        segDisplaySet.loading = false;
        return reject(e);
      }
    }

    console.log('[SEG _load] Creating segmentation from SEG display set...');
    segmentationService
      .createSegmentationForSEGDisplaySet(segDisplaySet)
      .then(() => {
        console.log('[SEG _load] Segmentation created successfully');
        segDisplaySet.loading = false;
        resolve();
      })
      .catch(error => {
        console.error('[SEG _load] Failed to create segmentation:', error);
        segDisplaySet.loading = false;
        reject(error);
      });
  });

  return loadPromises[SOPInstanceUID];
}

async function _loadSegments({
  extensionManager,
  servicesManager,
  segDisplaySet,
  headers,
}: withAppTypes) {
  const utilityModule = extensionManager.getModuleEntry(
    '@ohif/extension-cornerstone.utilityModule.common'
  );

  const { segmentationService, uiNotificationService } = servicesManager.services;

  const { dicomLoaderService } = utilityModule.exports;
  let arrayBuffer = await dicomLoaderService.findDicomDataPromise(segDisplaySet, null, headers);

  const referencedDisplaySet = servicesManager.services.displaySetService.getDisplaySetByUID(
    segDisplaySet.referencedDisplaySetInstanceUID
  );

  if (!referencedDisplaySet) {
    throw new Error('referencedDisplaySet is missing for SEG');
  }

  let { imageIds } = referencedDisplaySet;

  if (!imageIds) {
    // try images
    const { images } = referencedDisplaySet;
    imageIds = images.map(image => image.imageId);
  }

  // DEBUG: Log loading information
  console.log('[SEG _loadSegments] Loading SEG:', {
    segSeriesDescription: segDisplaySet.SeriesDescription,
    segSeriesInstanceUID: segDisplaySet.SeriesInstanceUID,
    referencedSeriesInstanceUID: segDisplaySet.referencedSeriesInstanceUID,
    numImageIds: imageIds.length,
    firstImageId: imageIds[0],
  });

  // Progressive tolerance values to try - start strict, then become more lenient
  // The default 0.001 is too strict for many real-world DICOM SEG files
  // due to floating-point precision differences during save/load cycles
  // Added 5.0 to handle scientific notation corruption bug where 3.86e-10 becomes 3.86
  const toleranceValues = [0.1, 0.5, 1.0, 2.0, 5.0];
  
  eventTarget.addEventListener(Enums.Events.SEGMENTATION_LOAD_PROGRESS, evt => {
    const { percentComplete } = evt.detail;
    segmentationService._broadcastEvent(segmentationService.EVENTS.SEGMENT_LOADING_COMPLETE, {
      percentComplete,
    });
  });

  // Extract orientation info for debugging before attempting loads
  let segIOP = null;
  let sourceIOP = null;
  let dicomData = null;
  let dataset = null;
  
  try {
    dicomData = dcmjs.data.DicomMessage.readFile(arrayBuffer);
    dataset = dcmjs.data.DicomMetaDictionary.naturalizeDataset(dicomData.dict);
    const sharedFG = dataset.SharedFunctionalGroupsSequence;
    const perFrameFG = dataset.PerFrameFunctionalGroupsSequence;
    
    if (sharedFG?.PlaneOrientationSequence?.ImageOrientationPatient) {
      segIOP = sharedFG.PlaneOrientationSequence.ImageOrientationPatient;
    } else if (perFrameFG?.[0]?.PlaneOrientationSequence?.ImageOrientationPatient) {
      segIOP = perFrameFG[0].PlaneOrientationSequence.ImageOrientationPatient;
    }
    
    // Get source image orientation
    sourceIOP = metaData.get('imagePlaneModule', imageIds[0])?.imageOrientationPatient;
    
    console.log('[SEG _loadSegments] Orientation values:', {
      segImageOrientationPatient: segIOP,
      sourceImageOrientationPatient: sourceIOP,
    });
    
    // Calculate actual differences for debugging
    if (segIOP && sourceIOP) {
      const diffs = segIOP.map((v, i) => Math.abs(v - sourceIOP[i]));
      const maxDiff = Math.max(...diffs);
      console.log('[SEG _loadSegments] Orientation differences:', {
        differences: diffs,
        maxDifference: maxDiff,
      });
      
      // FIX: Detect and correct scientific notation corruption bug
      // When a value like 3.86e-10 was corrupted to 3.86 during SEG save,
      // we can detect this by checking if the source value is near zero but SEG is not
      let needsCorrection = false;
      const correctedSegIOP = [...segIOP];
      for (let i = 0; i < 6; i++) {
        const srcVal = sourceIOP[i];
        const segVal = segIOP[i];
        // If source is essentially 0 (< 1e-6) but SEG has a value > 1,
        // this is likely the scientific notation corruption bug
        if (Math.abs(srcVal) < 1e-6 && Math.abs(segVal) > 1) {
          console.warn(`[SEG _loadSegments] Detected corrupted orientation value at index ${i}: SEG=${segVal}, Source=${srcVal}. Correcting to 0.`);
          correctedSegIOP[i] = 0;
          needsCorrection = true;
        }
      }
      
      if (needsCorrection) {
        console.log('[SEG _loadSegments] Applying orientation correction to SEG data');
        // Update the dataset with corrected orientation
        const sharedFG = dataset.SharedFunctionalGroupsSequence;
        const perFrameFG = dataset.PerFrameFunctionalGroupsSequence;
        
        if (sharedFG?.PlaneOrientationSequence) {
          sharedFG.PlaneOrientationSequence.ImageOrientationPatient = correctedSegIOP;
        }
        if (perFrameFG) {
          for (const frame of perFrameFG) {
            if (frame?.PlaneOrientationSequence?.ImageOrientationPatient) {
              frame.PlaneOrientationSequence.ImageOrientationPatient = correctedSegIOP;
            }
          }
        }
        
        // Re-encode the corrected dataset to a new ArrayBuffer
        const denaturalizedDataset = dcmjs.data.DicomMetaDictionary.denaturalizeDataset(dataset);
        const newDicomData = new dcmjs.data.DicomMessage({
          _meta: dicomData.meta,
          _dict: denaturalizedDataset,
        });
        arrayBuffer = newDicomData.write();
        console.log('[SEG _loadSegments] Re-encoded SEG with corrected orientation values');
      }
    }
  } catch (debugError) {
    console.warn('[SEG _loadSegments] Could not extract orientation info:', debugError);
  }

  let results = null;
  let lastError = null;
  
  // Try with progressively larger tolerance values
  for (const tolerance of toleranceValues) {
    try {
      console.log('[SEG _loadSegments] Attempting createFromDICOMSegBuffer with tolerance:', tolerance);
      results = await adaptersSEG.Cornerstone3D.Segmentation.createFromDICOMSegBuffer(
        imageIds,
        arrayBuffer,
        { metadataProvider: metaData, tolerance }
      );
      console.log('[SEG _loadSegments] Successfully parsed SEG buffer with tolerance:', tolerance);
      break; // Success, exit the loop
    } catch (parseError) {
      console.warn(`[SEG _loadSegments] Failed with tolerance ${tolerance}:`, parseError.message);
      lastError = parseError;
      // Continue to try with larger tolerance
    }
  }
  
  // If all tolerance values failed, throw the last error
  if (!results) {
    const errorMsg = lastError?.message || 'Unknown error';
    
    // Detailed diagnosis of the orientation mismatch
    console.error('[SEG _loadSegments] ORIENTATION MISMATCH DIAGNOSIS:');
    console.error('  SEG ImageOrientationPatient:', segIOP);
    console.error('  Source ImageOrientationPatient:', sourceIOP);
    console.error('  Tried tolerances:', toleranceValues);
    console.error('  Error message:', errorMsg);
    
    if (segIOP && sourceIOP) {
      // Calculate row and column direction vectors
      const segRow = [segIOP[0], segIOP[1], segIOP[2]];
      const segCol = [segIOP[3], segIOP[4], segIOP[5]];
      const srcRow = [sourceIOP[0], sourceIOP[1], sourceIOP[2]];
      const srcCol = [sourceIOP[3], sourceIOP[4], sourceIOP[5]];
      
      // Calculate dot products to understand orientation relationship
      const dotRowRow = segRow[0]*srcRow[0] + segRow[1]*srcRow[1] + segRow[2]*srcRow[2];
      const dotColCol = segCol[0]*srcCol[0] + segCol[1]*srcCol[1] + segCol[2]*srcCol[2];
      
      console.error('  Row-Row dot product:', dotRowRow.toFixed(4), '(should be ~1 for same orientation)');
      console.error('  Col-Col dot product:', dotColCol.toFixed(4), '(should be ~1 for same orientation)');
      
      if (Math.abs(dotRowRow) < 0.1 || Math.abs(dotColCol) < 0.1) {
        console.error('  DIAGNOSIS: Planes are PERPENDICULAR - SEG was likely created while viewing in a different plane');
      } else if (Math.abs(Math.abs(dotRowRow) - 1) > 0.1 || Math.abs(Math.abs(dotColCol) - 1) > 0.1) {
        console.error('  DIAGNOSIS: Planes are OBLIQUE - orientation vectors are not aligned');
      }
    }
    
    throw lastError;
  }

  let usedRecommendedDisplayCIELabValue = true;
  results.segMetadata.data.forEach((data, i) => {
    if (i > 0) {
      data.rgba = data.RecommendedDisplayCIELabValue;

      if (data.rgba) {
        data.rgba = dicomlabToRGB(data.rgba);
      } else {
        usedRecommendedDisplayCIELabValue = false;
        data.rgba = CONSTANTS.COLOR_LUT[i % CONSTANTS.COLOR_LUT.length];
      }
    }
  });

  if (!usedRecommendedDisplayCIELabValue) {
    // Display a notification about the non-utilization of RecommendedDisplayCIELabValue
    uiNotificationService.show({
      title: 'DICOM SEG import',
      message:
        'RecommendedDisplayCIELabValue not found for one or more segments. The default color was used instead.',
      type: 'warning',
      duration: 5000,
    });
  }

  Object.assign(segDisplaySet, results);
}

function _segmentationExists(segDisplaySet) {
  return cstSegmentation.state.getSegmentation(segDisplaySet.displaySetInstanceUID);
}

function getSopClassHandlerModule({ servicesManager, extensionManager }) {
  const getDisplaySetsFromSeries = instances => {
    return _getDisplaySetsFromSeries(instances, servicesManager, extensionManager);
  };

  return [
    {
      name: 'dicom-seg',
      sopClassUids,
      getDisplaySetsFromSeries,
    },
  ];
}

export default getSopClassHandlerModule;
