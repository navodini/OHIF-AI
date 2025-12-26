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

  console.log('[SEG SOP Handler] Creating SEG display set:', {
    SeriesDescription,
    SeriesInstanceUID,
    StudyInstanceUID,
    displaySetInstanceUID: displaySet.displaySetInstanceUID,
  });

  const referencedSeriesSequence = instance.ReferencedSeriesSequence;

  if (!referencedSeriesSequence) {
    console.error('[SEG SOP Handler] ReferencedSeriesSequence is missing for the SEG');
    return;
  }

  const referencedSeries = referencedSeriesSequence[0] || referencedSeriesSequence;

  displaySet.referencedImages = instance.ReferencedSeriesSequence.ReferencedInstanceSequence;
  displaySet.referencedSeriesInstanceUID = referencedSeries.SeriesInstanceUID;
  console.log('[SEG SOP Handler] Referenced series:', displaySet.referencedSeriesInstanceUID);
  
  const { displaySetService } = servicesManager.services;
  const referencedDisplaySets = displaySetService.getDisplaySetsForSeries(
    displaySet.referencedSeriesInstanceUID
  );

  const referencedDisplaySet = referencedDisplaySets[0];

  if (!referencedDisplaySet) {
    console.log('[SEG SOP Handler] Referenced display set not found yet, subscribing to DISPLAY_SETS_ADDED');
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
          console.log('[SEG SOP Handler] Found referenced display set:', addedDisplaySet.displaySetInstanceUID);
          displaySet.referencedDisplaySetInstanceUID = addedDisplaySet.displaySetInstanceUID;
          unsubscribe();
        }
      }
    );
  } else {
    console.log('[SEG SOP Handler] Referenced display set found:', referencedDisplaySet.displaySetInstanceUID);
    displaySet.referencedDisplaySetInstanceUID = referencedDisplaySet.displaySetInstanceUID;
  }

  displaySet.load = async ({ headers }) =>
    await _load(displaySet, servicesManager, extensionManager, headers);

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
    return loadPromises[SOPInstanceUID];
  }

  segDisplaySet.loading = true;

  // We don't want to fire multiple loads, so we'll wait for the first to finish
  // and also return the same promise to any other callers.
  loadPromises[SOPInstanceUID] = new Promise(async (resolve, reject) => {
    if (!segDisplaySet.segments || Object.keys(segDisplaySet.segments).length === 0) {
      try {
        await _loadSegments({
          extensionManager,
          servicesManager,
          segDisplaySet,
          headers,
        });
      } catch (e) {
        segDisplaySet.loading = false;
        return reject(e);
      }
    }

    segmentationService
      .createSegmentationForSEGDisplaySet(segDisplaySet)
      .then(() => {
        segDisplaySet.loading = false;
        resolve();
      })
      .catch(error => {
        segDisplaySet.loading = false;
        
        // Fire SEGMENTATION_LOADING_COMPLETE event even on error
        // to prevent UI from being stuck on "Loading SEG..."
        segmentationService._broadcastEvent(
          segmentationService.EVENTS.SEGMENTATION_LOADING_COMPLETE,
          {
            segmentationId: segDisplaySet.displaySetInstanceUID,
            segDisplaySet,
            error: true,
            errorMessage: error?.message || String(error),
          }
        );
        
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
  const arrayBuffer = await dicomLoaderService.findDicomDataPromise(segDisplaySet, null, headers);

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

  // Increased tolerance for orientation matching to handle minor floating point differences
  // and slight variations in patient positioning between MRI sessions
  const tolerance = 0.01;
  eventTarget.addEventListener(Enums.Events.SEGMENTATION_LOAD_PROGRESS, evt => {
    const { percentComplete } = evt.detail;
    segmentationService._broadcastEvent(segmentationService.EVENTS.SEGMENT_LOADING_COMPLETE, {
      percentComplete,
    });
  });

  let results;
  try {
    // Log debug information about the SEG being loaded
    console.log('[SEG Load] Loading SEG:', {
      SeriesDescription: segDisplaySet.SeriesDescription,
      referencedSeriesUID: segDisplaySet.referencedSeriesInstanceUID,
      imageIdsCount: imageIds.length,
      tolerance,
    });
    
    results = await adaptersSEG.Cornerstone3D.Segmentation.createFromDICOMSegBuffer(
      imageIds,
      arrayBuffer,
      { metadataProvider: metaData, tolerance }
    );
  } catch (error) {
    // Handle orientation mismatch error gracefully
    const errorMessage = error?.message || String(error);
    if (errorMessage.includes('orthogonal to the acquisition plane')) {
      console.warn('[SEG Load] Segmentation orientation mismatch:', {
        SeriesDescription: segDisplaySet.SeriesDescription,
        referencedSeriesUID: segDisplaySet.referencedSeriesInstanceUID,
        error: errorMessage,
      });
      uiNotificationService.show({
        title: 'DICOM SEG Load Warning',
        message: `Cannot display "${segDisplaySet.SeriesDescription || 'Segmentation'}": The segmentation was saved in a different orientation than the source images. This is a known limitation.`,
        type: 'warning',
        duration: 8000,
      });
      // Mark as failed to load but don't crash
      segDisplaySet.loadError = 'orientation_mismatch';
      segDisplaySet.loadErrorMessage = 'Segmentation orientation does not match source images';
      throw new Error('Segmentation orientation mismatch - cannot display overlay');
    }
    // Re-throw other errors
    throw error;
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
