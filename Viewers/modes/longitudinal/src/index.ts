import i18n from 'i18next';
import { id } from './id';
import initToolGroups from './initToolGroups';
import toolbarButtons from './toolbarButtons';

// Allow this mode by excluding non-imaging modalities such as SR, SEG
// Also, SM is not a simple imaging modalities, so exclude it.
const NON_IMAGE_MODALITIES = ['ECG', 'SEG', 'RTSTRUCT', 'RTPLAN', 'PR'];

const ohif = {
  layout: '@ohif/extension-default.layoutTemplateModule.viewerLayout',
  sopClassHandler: '@ohif/extension-default.sopClassHandlerModule.stack',
  thumbnailList: '@ohif/extension-default.panelModule.seriesList',
  longitudinalVolumetrics: '@ohif/extension-default.panelModule.longitudinalVolumetrics',
  wsiSopClassHandler:
    '@ohif/extension-cornerstone.sopClassHandlerModule.DicomMicroscopySopClassHandler',
};

const cornerstone = {
  measurements: '@ohif/extension-cornerstone.panelModule.panelMeasurement',
  segmentation: '@ohif/extension-cornerstone.panelModule.panelSegmentationWithTools',
};

const tracked = {
 // measurements: '@ohif/extension-measurement-tracking.panelModule.trackedMeasurements',
  thumbnailList: '@ohif/extension-measurement-tracking.panelModule.seriesList',
  viewport: '@ohif/extension-measurement-tracking.viewportModule.cornerstone-tracked',
};

const dicomsr = {
  sopClassHandler: '@ohif/extension-cornerstone-dicom-sr.sopClassHandlerModule.dicom-sr',
  sopClassHandler3D: '@ohif/extension-cornerstone-dicom-sr.sopClassHandlerModule.dicom-sr-3d',
  viewport: '@ohif/extension-cornerstone-dicom-sr.viewportModule.dicom-sr',
};

const dicomvideo = {
  sopClassHandler: '@ohif/extension-dicom-video.sopClassHandlerModule.dicom-video',
  viewport: '@ohif/extension-dicom-video.viewportModule.dicom-video',
};

const dicompdf = {
  sopClassHandler: '@ohif/extension-dicom-pdf.sopClassHandlerModule.dicom-pdf',
  viewport: '@ohif/extension-dicom-pdf.viewportModule.dicom-pdf',
};

const dicomSeg = {
  sopClassHandler: '@ohif/extension-cornerstone-dicom-seg.sopClassHandlerModule.dicom-seg',
  viewport: '@ohif/extension-cornerstone-dicom-seg.viewportModule.dicom-seg',
};

const dicomPmap = {
  sopClassHandler: '@ohif/extension-cornerstone-dicom-pmap.sopClassHandlerModule.dicom-pmap',
  viewport: '@ohif/extension-cornerstone-dicom-pmap.viewportModule.dicom-pmap',
};

const dicomRT = {
  viewport: '@ohif/extension-cornerstone-dicom-rt.viewportModule.dicom-rt',
  sopClassHandler: '@ohif/extension-cornerstone-dicom-rt.sopClassHandlerModule.dicom-rt',
};

const extensionDependencies = {
  // Can derive the versions at least process.env.from npm_package_version
  '@ohif/extension-default': '^3.0.0',
  '@ohif/extension-cornerstone': '^3.0.0',
  '@ohif/extension-measurement-tracking': '^3.0.0',
  '@ohif/extension-cornerstone-dicom-sr': '^3.0.0',
  '@ohif/extension-cornerstone-dicom-seg': '^3.0.0',
  '@ohif/extension-cornerstone-dicom-pmap': '^3.0.0',
  '@ohif/extension-cornerstone-dicom-rt': '^3.0.0',
  '@ohif/extension-dicom-pdf': '^3.0.1',
  '@ohif/extension-dicom-video': '^3.0.1',
};

function modeFactory({ modeConfiguration }) {
  let _activatePanelTriggersSubscriptions = [];
  let _segDisplaySetSubscription = null;
  return {
    // TODO: We're using this as a route segment
    // We should not be.
    id,
    routeName: 'viewer',
    displayName: i18n.t('Modes:Basic Viewer'),
    /**
     * Lifecycle hooks
     */
    onModeEnter: function ({ servicesManager, extensionManager, commandsManager }: withAppTypes) {
      const { measurementService, toolbarService, toolGroupService, customizationService, displaySetService, segmentationService, viewportGridService } =
        servicesManager.services;

      measurementService.clearMeasurements();

      // Init Default and SR ToolGroups
      initToolGroups(extensionManager, toolGroupService, commandsManager);

      toolbarService.addButtons(toolbarButtons);
      toolbarService.createButtonSection('primary', [
        'MeasurementTools',
        'Zoom',
        'Pan',
        'TrackballRotate',
        'WindowLevel',
        'Capture',
        'Layout',
        'Crosshairs',
        'MoreTools',
      ]);

      toolbarService.createButtonSection('measurementSection', [
        'Length',
        'Bidirectional',
        'ArrowAnnotate',
        'EllipticalROI',
        'RectangleROI',
        'CircleROI',
        'PlanarFreehandROI',
        'SplineROI',
        'LivewireContour',
      ]);

      toolbarService.createButtonSection('moreToolsSection', [
        'Reset',
        'rotate-right',
        'flipHorizontal',
        'ImageSliceSync',
        'ReferenceLines',
        'ImageOverlayViewer',
        'StackScroll',
        'invert',
        'Probe',
        'Cine',
        'Angle',
        'CobbAngle',
        'Magnify',
        'CalibrationLine',
        'TagBrowser',
        'AdvancedMagnify',
        'UltrasoundDirectionalTool',
        'WindowLevelRegion',
      ]);

      customizationService.setCustomizations({
        'panelSegmentation.disableEditing': {
          $set: false,
        },
      });

      toolbarService.createButtonSection('segmentationToolbox', [
        'SegmentationUtilities',
        'SegmentationTools',
      ]);

      toolbarService.createButtonSection('aiToolBox', ['aiToolBoxContainer']);

      toolbarService.createButtonSection('aiToolBoxSection', [
        'Probe2',
        'PlanarFreehandROI2',
        'PlanarFreehandROI3',
        'RectangleROI2',
        //'sam2',
        'nninter',
        'nnunetAuto',
        'nnunetInit',
        'calculateVolume',
        'longitudinalVolumetrics',
        'saveSegmentation',
        //'resetNninter',
        //'jumpToSegment',
        //'toggleCurrentSegment',
      ]);
      toolbarService.createButtonSection('segmentationToolboxUtilitySection', [
        //'LabelmapSlicePropagation',
        'InterpolateLabelmap',
        'SegmentBidirectional',
      ]);
      toolbarService.createButtonSection('segmentationToolboxToolsSection', [
        'BrushTools',
        //'MarkerLabelmap',
        //'RegionSegmentPlus',
        'Shapes',
      ]);
      toolbarService.createButtonSection('brushToolsSection', ['Brush', 'Eraser', 'Threshold']);

      // Helper function to load and overlay a SEG display set
      const loadAndOverlaySEG = async (displaySet, delayMs = 1000) => {
        console.log('[SEG Auto-Load] Loading DICOM SEG:', displaySet.SeriesDescription);
        
        // Skip if already marked as having a load error (e.g., orientation mismatch)
        if (displaySet.loadError) {
          console.log('[SEG Auto-Load] Skipping SEG with previous load error:', displaySet.loadError);
          return;
        }
        
        try {
          // Load the SEG display set
          await displaySet.load({ headers: {} });
          
          // Wait a bit for the viewport to be ready
          setTimeout(async () => {
            const { activeViewportId } = viewportGridService.getState();
            if (activeViewportId && displaySet.referencedDisplaySetInstanceUID) {
              // Check if the active viewport is showing the referenced display set
              const viewport = viewportGridService.getState().viewports.get(activeViewportId);
              if (viewport && viewport.displaySetInstanceUIDs.includes(displaySet.referencedDisplaySetInstanceUID)) {
                // Add the segmentation representation to the viewport
                const segmentationId = displaySet.displaySetInstanceUID;
                try {
                  await segmentationService.addSegmentationRepresentation(activeViewportId, {
                    segmentationId,
                  });
                  console.log('[SEG Auto-Load] Segmentation overlaid successfully:', displaySet.SeriesDescription);
                } catch (e) {
                  console.log('[SEG Auto-Load] Segmentation may already be added or viewport not ready:', e.message);
                }
              } else {
                console.log('[SEG Auto-Load] Viewport not showing referenced display set, skipping overlay');
              }
            }
          }, delayMs);
        } catch (error) {
          const errorMsg = error?.message || String(error);
          // Don't log orientation mismatch as an error since it's handled gracefully
          if (errorMsg.includes('orientation mismatch')) {
            console.log('[SEG Auto-Load] SEG has orientation mismatch, cannot display as overlay:', displaySet.SeriesDescription);
          } else {
            console.error('[SEG Auto-Load] Failed to auto-load SEG:', error);
          }
        }
      };

      // Auto-load DICOM SEG display sets when they are added
      // This ensures that saved segmentations are automatically overlaid when reopening the study
      _segDisplaySetSubscription = displaySetService.subscribe(
        displaySetService.EVENTS.DISPLAY_SETS_ADDED,
        async ({ displaySetsAdded }) => {
          console.log('[SEG Auto-Load] DISPLAY_SETS_ADDED event fired with', displaySetsAdded.length, 'display sets');
          for (const displaySet of displaySetsAdded) {
            console.log('[SEG Auto-Load] Display set added:', {
              Modality: displaySet.Modality,
              SeriesDescription: displaySet.SeriesDescription,
              hasLoad: !!displaySet.load,
              isLoaded: displaySet.isLoaded,
              displaySetInstanceUID: displaySet.displaySetInstanceUID?.substring(0, 16) + '...',
            });
            // Check if this is a SEG display set
            if (displaySet.Modality === 'SEG' && displaySet.load) {
              console.log('[SEG Auto-Load] Detected SEG display set, attempting to load...');
              await loadAndOverlaySEG(displaySet);
            }
          }
        }
      );

      // Also check for existing SEG display sets that were added before mode entered
      // This handles the case when reopening a study with existing SEG files
      setTimeout(async () => {
        const allDisplaySets = displaySetService.getActiveDisplaySets();
        console.log('[SEG Auto-Load] Checking for existing SEG display sets...', allDisplaySets.length, 'total display sets');
        
        // Log all display sets for debugging
        allDisplaySets.forEach((ds, idx) => {
          console.log(`[SEG Auto-Load] DisplaySet ${idx}:`, {
            Modality: ds.Modality,
            SeriesDescription: ds.SeriesDescription,
            hasLoad: !!ds.load,
            isLoaded: ds.isLoaded,
            StudyInstanceUID: ds.StudyInstanceUID?.substring(0, 16) + '...',
          });
        });
        
        for (const displaySet of allDisplaySets) {
          if (displaySet.Modality === 'SEG' && displaySet.load) {
            // Check if this SEG is already loaded/represented
            const existingSegmentations = segmentationService.getSegmentations();
            console.log('[SEG Auto-Load] Existing segmentations:', existingSegmentations.length);
            const alreadyLoaded = existingSegmentations.some(
              seg => seg.segmentationId === displaySet.displaySetInstanceUID
            );
            if (!alreadyLoaded) {
              console.log('[SEG Auto-Load] Found existing SEG that needs loading:', displaySet.SeriesDescription);
              await loadAndOverlaySEG(displaySet, 500);
            } else {
              console.log('[SEG Auto-Load] SEG already loaded:', displaySet.SeriesDescription);
            }
          }
        }
      }, 2000); // Wait for viewports to be fully initialized

      // // ActivatePanel event trigger for when a segmentation or measurement is added.
      // // Do not force activation so as to respect the state the user may have left the UI in.
      // _activatePanelTriggersSubscriptions = [
      //   ...panelService.addActivatePanelTriggers(
      //     cornerstone.segmentation,
      //     [
      //       {
      //         sourcePubSubService: segmentationService,
      //         sourceEvents: [segmentationService.EVENTS.SEGMENTATION_ADDED],
      //       },
      //     ],
      //     true
      //   ),
      //   ...panelService.addActivatePanelTriggers(
      //     tracked.measurements,
      //     [
      //       {
      //         sourcePubSubService: measurementService,
      //         sourceEvents: [
      //           measurementService.EVENTS.MEASUREMENT_ADDED,
      //           measurementService.EVENTS.RAW_MEASUREMENT_ADDED,
      //         ],
      //       },
      //     ],
      //     true
      //   ),
      //   true,
      // ];
    },
    onModeExit: ({ servicesManager }: withAppTypes) => {
      const {
        toolGroupService,
        syncGroupService,
        segmentationService,
        cornerstoneViewportService,
        uiDialogService,
        uiModalService,
      } = servicesManager.services;

      _activatePanelTriggersSubscriptions.forEach(sub => sub.unsubscribe());
      _activatePanelTriggersSubscriptions = [];

      // Clean up the SEG display set subscription
      if (_segDisplaySetSubscription) {
        _segDisplaySetSubscription.unsubscribe();
        _segDisplaySetSubscription = null;
      }

      uiDialogService.hideAll();
      uiModalService.hide();
      toolGroupService.destroy();
      syncGroupService.destroy();
      segmentationService.destroy();
      cornerstoneViewportService.destroy();
    },
    validationTags: {
      study: [],
      series: [],
    },

    isValidMode: function ({ modalities }) {
      const modalities_list = modalities.split('\\');

      // Exclude non-image modalities
      return {
        valid: !!modalities_list.filter(modality => NON_IMAGE_MODALITIES.indexOf(modality) === -1)
          .length,
        description:
          'The mode does not support studies that ONLY include the following modalities: SM, ECG, SEG, RTSTRUCT',
      };
    },
    routes: [
      {
        path: 'longitudinal',
        /*init: ({ servicesManager, extensionManager }) => {
          //defaultViewerRouteInit
        },*/
        layoutTemplate: () => {
          return {
            id: ohif.layout,
            props: {
              leftPanels: [tracked.thumbnailList],
              leftPanelResizable: true,
              rightPanels: [cornerstone.segmentation, ohif.longitudinalVolumetrics],
              rightPanelClosed: false,
              rightPanelResizable: true,
              viewports: [
                {
                  namespace: tracked.viewport,
                  displaySetsToDisplay: [
                    ohif.sopClassHandler,
                    dicomvideo.sopClassHandler,
                    dicomsr.sopClassHandler3D,
                    ohif.wsiSopClassHandler,
                  ],
                },
                {
                  namespace: dicomsr.viewport,
                  displaySetsToDisplay: [dicomsr.sopClassHandler],
                },
                {
                  namespace: dicompdf.viewport,
                  displaySetsToDisplay: [dicompdf.sopClassHandler],
                },
                {
                  namespace: dicomSeg.viewport,
                  displaySetsToDisplay: [dicomSeg.sopClassHandler],
                },
                {
                  namespace: dicomPmap.viewport,
                  displaySetsToDisplay: [dicomPmap.sopClassHandler],
                },
                {
                  namespace: dicomRT.viewport,
                  displaySetsToDisplay: [dicomRT.sopClassHandler],
                },
              ],
            },
          };
        },
      },
    ],
    extensions: extensionDependencies,
    // Default protocol gets self-registered by default in the init
    hangingProtocol: 'default',
    // Order is important in sop class handlers when two handlers both use
    // the same sop class under different situations.  In that case, the more
    // general handler needs to come last.  For this case, the dicomvideo must
    // come first to remove video transfer syntax before ohif uses images
    sopClassHandlers: [
      dicomvideo.sopClassHandler,
      dicomSeg.sopClassHandler,
      dicomPmap.sopClassHandler,
      ohif.sopClassHandler,
      ohif.wsiSopClassHandler,
      dicompdf.sopClassHandler,
      dicomsr.sopClassHandler3D,
      dicomsr.sopClassHandler,
      dicomRT.sopClassHandler,
    ],
    ...modeConfiguration,
  };
}

const mode = {
  id,
  modeFactory,
  extensionDependencies,
};

export default mode;
export { initToolGroups, toolbarButtons };
