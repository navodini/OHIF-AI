import React, { useEffect } from 'react';
import PropTypes from 'prop-types';
import { useSystem } from '@ohif/core';
import PanelStudyBrowser from '@ohif/extension-default/src/Panels/StudyBrowser/PanelStudyBrowser';
import { UntrackSeriesModal } from './untrackSeriesModal';
import { useTrackedMeasurements } from '../../getContextModule';

const thumbnailNoImageModalities = [
  'SR',
  'SEG',
  'SM',
  'RTSTRUCT',
  'RTPLAN',
  'RTDOSE',
  'DOC',
  'OT',
  'PMAP',
];

/**
 * Panel component for the Study Browser with tracking capabilities
 */
export default function PanelStudyBrowserTracking({
  getImageSrc,
  getStudiesForPatientByMRN,
  requestDisplaySetCreationForStudy,
  dataSource,
}) {
  const { servicesManager } = useSystem();
  const { displaySetService, uiModalService, measurementService, viewportGridService } =
    servicesManager.services;
  const [trackedMeasurements, sendTrackedMeasurementsEvent] = useTrackedMeasurements();
  const { trackedSeries } = trackedMeasurements.context;

  const checkDirtyMeasurements = displaySetInstanceUID => {
    const displaySet = displaySetService.getDisplaySetByUID(displaySetInstanceUID);
    if (displaySet.Modality === 'SR') {
      const activeViewportId = viewportGridService.getActiveViewportId();
      sendTrackedMeasurementsEvent('CHECK_DIRTY', {
        viewportId: activeViewportId,
        displaySetInstanceUID: displaySetInstanceUID,
      });
    }
  };

  useEffect(() => {
    const subscriptionOndropFired = viewportGridService.subscribe(
      viewportGridService.EVENTS.VIEWPORT_ONDROP_HANDLED,
      ({ eventData }) => {
        checkDirtyMeasurements(eventData.displaySetInstanceUID);
      }
    );

    return () => {
      subscriptionOndropFired.unsubscribe();
    };
  }, []);
  const onClickUntrack = displaySetInstanceUID => {
    const onConfirm = () => {
      const displaySet = displaySetService.getDisplaySetByUID(displaySetInstanceUID);
      sendTrackedMeasurementsEvent('UNTRACK_SERIES', {
        SeriesInstanceUID: displaySet.SeriesInstanceUID,
      });
      const measurements = measurementService.getMeasurements();
      measurements.forEach(m => {
        if (m.referenceSeriesUID === displaySet.SeriesInstanceUID) {
          measurementService.remove(m.uid);
        }
      });
    };

    uiModalService.show({
      title: 'Untrack Series',
      content: UntrackSeriesModal,
      contentProps: {
        onConfirm,
        message: 'Are you sure you want to untrack this series?',
      },
    });
  };

  // Custom mapping function to add tracking data to display sets
  const mapDisplaySetsWithTracking = (
    displaySets,
    displaySetLoadingState,
    thumbnailImageSrcMap,
    viewports
  ) => {
    // DEBUG: Log input display sets
    console.log('[PanelStudyBrowserTracking] Input display sets:', displaySets.length);
    displaySets.forEach((ds, i) => {
      console.log(`[PanelStudyBrowserTracking] Input DS ${i}: Modality=${ds.Modality}, excludeFromThumbnail=${ds.excludeFromThumbnailBrowser}`);
    });

    const thumbnailDisplaySets = [];
    const thumbnailNoImageDisplaySets = [];
    displaySets
      .filter(ds => !ds.excludeFromThumbnailBrowser)
      .forEach(ds => {
        const { thumbnailSrc, displaySetInstanceUID } = ds;
        const componentType = getComponentType(ds);

        // DEBUG: Log each display set being processed
        console.log(`[PanelStudyBrowserTracking] Processing: Modality=${ds.Modality}, componentType=${componentType}`);

        const array =
          componentType === 'thumbnailTracked' ? thumbnailDisplaySets : thumbnailNoImageDisplaySets;

        const loadingProgress = displaySetLoadingState?.[displaySetInstanceUID];

        array.push({
          displaySetInstanceUID,
          description: ds.SeriesDescription || '',
          seriesNumber: ds.SeriesNumber,
          modality: ds.Modality,
          seriesDate: ds.SeriesDate ? new Date(ds.SeriesDate).toLocaleDateString() : '',
          numInstances: ds.numImageFrames,
          loadingProgress,
          countIcon: ds.countIcon,
          messages: ds.messages,
          StudyInstanceUID: ds.StudyInstanceUID,
          componentType,
          imageSrc: thumbnailSrc || thumbnailImageSrcMap[displaySetInstanceUID],
          dragData: {
            type: 'displayset',
            displaySetInstanceUID,
          },
          isTracked: trackedSeries.includes(ds.SeriesInstanceUID),
          isHydratedForDerivedDisplaySet: ds.isHydrated,
        });
      });

    const result = [...thumbnailDisplaySets, ...thumbnailNoImageDisplaySets];
    // DEBUG: Log result
    console.log(`[PanelStudyBrowserTracking] Returning ${result.length} display sets (${thumbnailDisplaySets.length} tracked, ${thumbnailNoImageDisplaySets.length} noImage)`);
    result.forEach((ds, i) => {
      console.log(`[PanelStudyBrowserTracking] Result DS ${i}: modality=${ds.modality}, componentType=${ds.componentType}`);
    });
    return result;
  };

  // Override component type to use tracking specific components
  const getComponentType = ds => {
    if (thumbnailNoImageModalities.includes(ds.Modality) || ds?.unsupported) {
      return 'thumbnailNoImage';
    }
    return 'thumbnailTracked';
  };

  return (
    <PanelStudyBrowser
      getImageSrc={getImageSrc}
      getStudiesForPatientByMRN={getStudiesForPatientByMRN}
      requestDisplaySetCreationForStudy={requestDisplaySetCreationForStudy}
      dataSource={dataSource}
      customMapDisplaySets={mapDisplaySetsWithTracking}
      onClickUntrack={onClickUntrack}
      onDoubleClickThumbnailHandlerCallBack={checkDirtyMeasurements}
    />
  );
}

PanelStudyBrowserTracking.propTypes = {
  dataSource: PropTypes.shape({
    getImageIdsForDisplaySet: PropTypes.func.isRequired,
  }).isRequired,
  getImageSrc: PropTypes.func.isRequired,
  getStudiesForPatientByMRN: PropTypes.func.isRequired,
  requestDisplaySetCreationForStudy: PropTypes.func.isRequired,
};
