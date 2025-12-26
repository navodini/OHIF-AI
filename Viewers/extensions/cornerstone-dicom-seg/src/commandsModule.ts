import dcmjs from 'dcmjs';
import { classes, Types } from '@ohif/core';
import { cache, metaData } from '@cornerstonejs/core';
import { segmentation as cornerstoneToolsSegmentation } from '@cornerstonejs/tools';
import { adaptersRT, helpers, adaptersSEG } from '@cornerstonejs/adapters';
import { createReportDialogPrompt } from '@ohif/extension-default';
import { DicomMetadataStore } from '@ohif/core';

import PROMPT_RESPONSES from '../../default/src/utils/_shared/PROMPT_RESPONSES';

const { datasetToBlob } = dcmjs.data;

const getTargetViewport = ({ viewportId, viewportGridService }) => {
  const { viewports, activeViewportId } = viewportGridService.getState();
  const targetViewportId = viewportId || activeViewportId;

  const viewport = viewports.get(targetViewportId);

  return viewport;
};

const {
  Cornerstone3D: {
    Segmentation: { generateSegmentation },
  },
} = adaptersSEG;

const {
  Cornerstone3D: {
    RTSS: { generateRTSSFromSegmentations },
  },
} = adaptersRT;

const { downloadDICOMData } = helpers;

const commandsModule = ({
  servicesManager,
  extensionManager,
}: Types.Extensions.ExtensionParams): Types.Extensions.CommandsModule => {
  const { segmentationService, displaySetService, viewportGridService, toolGroupService } =
    servicesManager.services as AppTypes.Services;

  const actions = {
    /**
     * Loads segmentations for a specified viewport.
     * The function prepares the viewport for rendering, then loads the segmentation details.
     * Additionally, if the segmentation has scalar data, it is set for the corresponding label map volume.
     *
     * @param {Object} params - Parameters for the function.
     * @param params.segmentations - Array of segmentations to be loaded.
     * @param params.viewportId - the target viewport ID.
     *
     */
    loadSegmentationsForViewport: async ({ segmentations, viewportId }) => {
      // Todo: handle adding more than one segmentation
      const viewport = getTargetViewport({ viewportId, viewportGridService });
      const displaySetInstanceUID = viewport.displaySetInstanceUIDs[0];

      const segmentation = segmentations[0];
      const segmentationId = segmentation.segmentationId;
      const label = segmentation.config.label;
      const segments = segmentation.config.segments;

      const displaySet = displaySetService.getDisplaySetByUID(displaySetInstanceUID);

      await segmentationService.createLabelmapForDisplaySet(displaySet, {
        segmentationId,
        segments,
        label,
      });

      segmentationService.addOrUpdateSegmentation(segmentation);

      await segmentationService.addSegmentationRepresentation(viewport.viewportId, {
        segmentationId,
      });

      return segmentationId;
    },
    /**
     * Generates a segmentation from a given segmentation ID.
     * This function retrieves the associated segmentation and
     * its referenced volume, extracts label maps from the
     * segmentation volume, and produces segmentation data
     * alongside associated metadata.
     *
     * @param {Object} params - Parameters for the function.
     * @param params.segmentationId - ID of the segmentation to be generated.
     * @param params.options - Optional configuration for the generation process.
     *
     * @returns Returns the generated segmentation data.
     */
    generateSegmentation: ({ segmentationId, options = {} }) => {
      const segmentation = cornerstoneToolsSegmentation.state.getSegmentation(segmentationId);

      const { imageIds, referencedImageIds: storedReferencedImageIds } = segmentation.representationData.Labelmap;

      const segImages = imageIds.map(imageId => cache.getImage(imageId));
      
      // Use the stored referencedImageIds from the representation data if available,
      // otherwise fall back to the referencedImageId property on each derived image.
      // This is important for cases where the slice order was flipped - the stored
      // referencedImageIds will be in the correct order.
      let referencedImages;
      if (storedReferencedImageIds && storedReferencedImageIds.length === segImages.length) {
        referencedImages = storedReferencedImageIds.map(refId => cache.getImage(refId));
        console.log('[SEG Generate] Using stored referencedImageIds from representation data');
      } else {
        referencedImages = segImages.map(image => cache.getImage(image.referencedImageId));
        console.log('[SEG Generate] Using referencedImageId property from derived images');
      }

      console.log('[SEG Generate Debug] segmentationId:', segmentationId);
      console.log('[SEG Generate Debug] imageIds count:', imageIds?.length);
      console.log('[SEG Generate Debug] storedReferencedImageIds count:', storedReferencedImageIds?.length);
      console.log('[SEG Generate Debug] First 3 imageIds:', imageIds?.slice(0, 3));
      console.log('[SEG Generate Debug] First 3 storedReferencedImageIds:', storedReferencedImageIds?.slice(0, 3));
      console.log('[SEG Generate Debug] First 3 segImage.referencedImageIds:', segImages?.slice(0, 3).map(img => img?.referencedImageId));
      console.log('[SEG Generate Debug] First 3 referencedImages imageIds:', referencedImages?.slice(0, 3).map(img => img?.imageId));

      const labelmaps2D = [];

      let z = 0;

      for (const segImage of segImages) {
        const segmentsOnLabelmap = new Set();
        const pixelData = segImage.getPixelData();
        const { rows, columns } = segImage;

        // Use a single pass through the pixel data
        for (let i = 0; i < pixelData.length; i++) {
          const segment = pixelData[i];
          if (segment !== 0) {
            segmentsOnLabelmap.add(segment);
          }
        }

        labelmaps2D[z++] = {
          segmentsOnLabelmap: Array.from(segmentsOnLabelmap),
          pixelData,
          rows,
          columns,
        };
      }

      const allSegmentsOnLabelmap = labelmaps2D.map(labelmap => labelmap.segmentsOnLabelmap);

      const labelmap3D = {
        segmentsOnLabelmap: Array.from(new Set(allSegmentsOnLabelmap.flat())),
        metadata: [],
        labelmaps2D,
      };

      const segmentationInOHIF = segmentationService.getSegmentation(segmentationId);
      const representations = segmentationService.getRepresentationsForSegmentation(segmentationId);

      Object.entries(segmentationInOHIF.segments).forEach(([segmentIndex, segment]) => {
        // segmentation service already has a color for each segment
        if (!segment) {
          return;
        }

        const { label } = segment;

        const firstRepresentation = representations[0];
        const color = segmentationService.getSegmentColor(
          firstRepresentation.viewportId,
          segmentationId,
          segment.segmentIndex
        );

        const RecommendedDisplayCIELabValue = dcmjs.data.Colors.rgb2DICOMLAB(
          color.slice(0, 3).map(value => value / 255)
        ).map(value => Math.round(value));

        let segmentMetadata = {};
        if (segmentation.cachedStats.data !== undefined && segmentation.cachedStats.data.length > 1) {
          segmentMetadata = segmentation.cachedStats.data
          .filter(e => e !== undefined && e !== null)
          .find(e => e.SegmentNumber == segmentIndex);
          if (segmentMetadata !== undefined && Object.keys(segmentMetadata).length !== 0){ 
            segmentMetadata.SegmentNumber = segmentIndex.toString();
            segmentMetadata.SegmentLabel = label;
            segmentMetadata.RecommendedDisplayCIELabValue = RecommendedDisplayCIELabValue;
            segmentMetadata.SegmentAlgorithmType = segmentation.cachedStats.seriesInstanceUid;
          }
        }

        if (segmentMetadata === undefined || Object.keys(segmentMetadata).length === 0) {
          segmentMetadata = {
            SegmentNumber: segmentIndex.toString(),
            SegmentLabel: label,
            SegmentAlgorithmType: segment?.algorithmType || 'MANUAL',
            SegmentAlgorithmName: segment?.algorithmName || 'OHIF Brush',
            RecommendedDisplayCIELabValue,
            SegmentedPropertyCategoryCodeSequence: {
              CodeValue: 'T-D0050',
              CodingSchemeDesignator: 'SRT',
              CodeMeaning: 'Tissue',
            },
            SegmentedPropertyTypeCodeSequence: {
              CodeValue: 'T-D0050',
              CodingSchemeDesignator: 'SRT',
              CodeMeaning: 'Tissue',
            },
          };
        }
        if (segment.cachedStats.description !== undefined){
          segmentMetadata.SegmentDescription = segment.cachedStats.description;
        }
        if (segment.cachedStats.algorithmName !== undefined){
          segmentMetadata.SegmentAlgorithmName = segment.cachedStats.algorithmName;
        }
        if (segment.cachedStats.algorithmType !== undefined){
          segmentMetadata.SegmentAlgorithmType = segment.cachedStats.algorithmType;
        }
        if (segmentation.cachedStats.seriesInstanceUid !== undefined){
          segmentMetadata.SegmentAlgorithmType = segmentation.cachedStats.seriesInstanceUid;
        }
        
        labelmap3D.metadata[segmentIndex] = segmentMetadata;
      });

      const generatedSegmentation = generateSegmentation(
        referencedImages,
        labelmap3D,
        metaData,
        options
      );

      return generatedSegmentation;
    },
    /**
     * Downloads a segmentation based on the provided segmentation ID.
     * This function retrieves the associated segmentation and
     * uses it to generate the corresponding DICOM dataset, which
     * is then downloaded with an appropriate filename.
     *
     * @param {Object} params - Parameters for the function.
     * @param params.segmentationId - ID of the segmentation to be downloaded.
     *
     */
    downloadSegmentation: ({ segmentationId }) => {
      const segmentationInOHIF = segmentationService.getSegmentation(segmentationId);
      const generatedSegmentation = actions.generateSegmentation({
        segmentationId,
      });

      downloadDICOMData(generatedSegmentation.dataset, `${segmentationInOHIF.label}`);
    },
    /**
     * Stores a segmentation based on the provided segmentationId into a specified data source.
     * The SeriesDescription is derived from user input or defaults to the segmentation label,
     * and in its absence, defaults to 'Research Derived Series'.
     *
     * @param {Object} params - Parameters for the function.
     * @param params.segmentationId - ID of the segmentation to be stored.
     * @param params.dataSource - Data source where the generated segmentation will be stored.
     *
     * @returns {Object|void} Returns the naturalized report if successfully stored,
     * otherwise throws an error.
     */
    storeSegmentation: async ({ segmentationId, dataSource, defaultSeriesDescription, autoSave = false, deleteOldSegs = false }) => {
      const segmentation = segmentationService.getSegmentation(segmentationId);

      if (!segmentation) {
        throw new Error('No segmentation found');
      }

      const { label } = segmentation;
      const defaultDataSource = dataSource ?? extensionManager.getActiveDataSource();

      let reportName = defaultSeriesDescription || label || 'Research Derived Series';
      let action = PROMPT_RESPONSES.CREATE_REPORT;

      // If not auto-save, show dialog
      if (!autoSave) {
        const result = await createReportDialogPrompt({
          servicesManager,
          extensionManager,
          title: 'Store Segmentation',
          defaultValue: reportName,
        });
        reportName = result.value;
        action = result.action;
      }

      if (action === PROMPT_RESPONSES.CREATE_REPORT) {
        try {
          const selectedDataSourceConfig = defaultDataSource;

          const generatedData = actions.generateSegmentation({
            segmentationId,
            options: {
              SeriesDescription: reportName || label || 'Research Derived Series',
            },
          });

          if (!generatedData || !generatedData.dataset) {
            throw new Error('Error during segmentation generation');
          }

          const { dataset: naturalizedReport } = generatedData;
          let selectedDataSourceConfig_new = undefined;
          if (selectedDataSourceConfig.store == undefined) {
            selectedDataSourceConfig_new = selectedDataSourceConfig[0];
          } else {
            selectedDataSourceConfig_new = selectedDataSourceConfig;
          }

          // Get the study UID from the naturalized report for deletion logic
          const reportStudyUID = naturalizedReport.StudyInstanceUID;

          // First, save the new SEG
          await selectedDataSourceConfig_new.store.dicom(naturalizedReport);
          
          // Then delete old SEGs ONLY from the SAME study/session
          // This replaces previous segmentations of the same session while preserving other sessions
          if (deleteOldSegs && selectedDataSourceConfig_new?.store?.deleteSeries && reportStudyUID) {
            try {
              const displaySets = displaySetService.getActiveDisplaySets();
              // Only delete SEGs from the SAME study (current session)
              const segDisplaySets = displaySets.filter(
                ds => ds.Modality === 'SEG' && ds.StudyInstanceUID === reportStudyUID
              );
              
              for (const segDS of segDisplaySets) {
                // Don't delete the series we just saved
                if (segDS.SeriesInstanceUID !== naturalizedReport.SeriesInstanceUID) {
                  try {
                    await selectedDataSourceConfig_new.store.deleteSeries(reportStudyUID, segDS.SeriesInstanceUID);
                    console.log('Deleted previous SEG from same session:', segDS.SeriesInstanceUID);
                  } catch (err) {
                    console.warn('Failed to delete old SEG series', segDS.SeriesInstanceUID, err);
                  }
                }
              }
            } catch (err) {
              console.warn('Error during cleanup of old SEGs:', err);
              // Don't throw - the save was successful, this is just cleanup
            }
          }
          
          // add the information for where we stored it to the instance as well
          naturalizedReport.wadoRoot = selectedDataSourceConfig_new.getConfig().wadoRoot;

          DicomMetadataStore.addInstances([naturalizedReport], true);

          return naturalizedReport;
        } catch (error) {
          console.debug('Error storing segmentation:', error);
          throw error;
        }
      }
    },
    /**
     * Converts segmentations into RTSS for download.
     * This sample function retrieves all segentations and passes to
     * cornerstone tool adapter to convert to DICOM RTSS format. It then
     * converts dataset to downloadable blob.
     *
     */
    downloadRTSS: async ({ segmentationId }) => {
      const segmentations = segmentationService.getSegmentation(segmentationId);

      // inject colors to the segmentIndex
      const firstRepresentation =
        segmentationService.getRepresentationsForSegmentation(segmentationId)[0];
      Object.entries(segmentations.segments).forEach(([segmentIndex, segment]) => {
        segment.color = segmentationService.getSegmentColor(
          firstRepresentation.viewportId,
          segmentationId,
          segmentIndex
        );
      });

      const RTSS = await generateRTSSFromSegmentations(
        segmentations,
        classes.MetadataProvider,
        DicomMetadataStore
      );

      try {
        const reportBlob = datasetToBlob(RTSS);

        //Create a URL for the binary.
        const objectUrl = URL.createObjectURL(reportBlob);
        window.location.assign(objectUrl);
      } catch (e) {
        console.warn(e);
      }
    },
  };

  const definitions = {
    loadSegmentationsForViewport: {
      commandFn: actions.loadSegmentationsForViewport,
    },

    generateSegmentation: {
      commandFn: actions.generateSegmentation,
    },
    downloadSegmentation: {
      commandFn: actions.downloadSegmentation,
    },
    storeSegmentation: {
      commandFn: actions.storeSegmentation,
    },
    downloadRTSS: {
      commandFn: actions.downloadRTSS,
    },
  };

  return {
    actions,
    definitions,
    defaultContext: 'SEGMENTATION',
  };
};

export default commandsModule;
