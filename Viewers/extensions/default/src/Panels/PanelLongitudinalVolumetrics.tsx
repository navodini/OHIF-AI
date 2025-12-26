import React, { useState, useCallback } from 'react';
import { useSystem } from '@ohif/core';
import { Button, Icons } from '@ohif/ui-next';
import { cache } from '@cornerstonejs/core';
import html2canvas from 'html2canvas';
import { jsPDF } from 'jspdf';

interface VolumetricData {
  studyDate: string;
  studyDescription: string;
  studyInstanceUID: string;
  seriesDescription: string;
  seriesInstanceUID: string;
  displaySetInstanceUID: string;
  segments: {
    segmentIndex: number;
    label: string;
    volumeMl: number;
    voxelCount: number;
  }[];
  totalVolumeMl: number;
  spacing: number[];
  dimensions: number[];
}

interface LongitudinalReport {
  patientId: string;
  patientName: string;
  timepoints: VolumetricData[];
  volumeChanges: {
    label: string;
    changes: {
      fromDate: string;
      toDate: string;
      fromVolume: number;
      toVolume: number;
      absoluteChange: number;
      percentChange: number;
    }[];
  }[];
}

/**
 * Helper to parse DICOM date format (YYYYMMDD) to readable format
 */
function formatDicomDate(dateStr: string): string {
  if (!dateStr || dateStr.length !== 8) {
    return dateStr || 'Unknown Date';
  }
  const year = dateStr.substring(0, 4);
  const month = dateStr.substring(4, 6);
  const day = dateStr.substring(6, 8);
  return `${year}-${month}-${day}`;
}

/**
 * Extract the best available date from a SEG display set
 * Tries multiple DICOM date fields in order of preference
 */
function extractDateFromDisplaySet(displaySet: any): string {
  console.log('Extracting date from displaySet:', {
    SeriesDescription: displaySet.SeriesDescription,
    StudyDate: displaySet.StudyDate,
    SeriesDate: displaySet.SeriesDate,
    ContentDate: displaySet.ContentDate,
    hasInstance: !!displaySet.instance,
    hasInstances: !!(displaySet.instances && displaySet.instances.length),
  });

  // First try direct properties on displaySet
  if (displaySet.StudyDate && displaySet.StudyDate.length === 8) {
    console.log('Found StudyDate on displaySet:', displaySet.StudyDate);
    return displaySet.StudyDate;
  }
  if (displaySet.SeriesDate && displaySet.SeriesDate.length === 8) {
    console.log('Found SeriesDate on displaySet:', displaySet.SeriesDate);
    return displaySet.SeriesDate;
  }
  if (displaySet.ContentDate && displaySet.ContentDate.length === 8) {
    console.log('Found ContentDate on displaySet:', displaySet.ContentDate);
    return displaySet.ContentDate;
  }
  
  // Try to get from the instance metadata
  const instance = displaySet.instance || (displaySet.instances && displaySet.instances[0]);
  if (instance) {
    console.log('Instance metadata:', {
      StudyDate: instance.StudyDate,
      SeriesDate: instance.SeriesDate,
      ContentDate: instance.ContentDate,
      AcquisitionDate: instance.AcquisitionDate,
      InstanceCreationDate: instance.InstanceCreationDate,
    });

    // Priority: StudyDate > SeriesDate > ContentDate > AcquisitionDate > InstanceCreationDate
    const dateFields = [
      'StudyDate',
      'SeriesDate', 
      'ContentDate',
      'AcquisitionDate',
      'InstanceCreationDate'
    ];
    
    for (const field of dateFields) {
      const dateValue = instance[field];
      if (dateValue && typeof dateValue === 'string' && dateValue.length === 8) {
        console.log(`Found ${field} on instance:`, dateValue);
        return dateValue;
      }
    }
  }
  
  console.log('No valid date found on displaySet');
  return '';
}

/**
 * Extract date from SEG display set, falling back to referenced display set if needed
 * For longitudinal tracking, we PREFER the referenced study's date (when the images were acquired)
 * over the SEG creation date (when the segmentation was saved)
 */
function extractDateWithFallback(displaySet: any, displaySetService: any): string {
  console.log('extractDateWithFallback called for:', displaySet.SeriesDescription);
  console.log('  referencedDisplaySetInstanceUID:', displaySet.referencedDisplaySetInstanceUID);
  
  // For longitudinal analysis, prioritize the REFERENCED study date
  // (the original imaging date) over the SEG creation date
  if (displaySet.referencedDisplaySetInstanceUID) {
    const referencedDisplaySet = displaySetService.getDisplaySetByUID(
      displaySet.referencedDisplaySetInstanceUID
    );
    console.log('  Referenced displaySet found:', !!referencedDisplaySet);
    if (referencedDisplaySet) {
      const refDate = extractDateFromDisplaySet(referencedDisplaySet);
      if (refDate) {
        console.log('  Using referenced study date:', refDate);
        return refDate;
      }
    }
  }
  
  // Fallback: try to get date from the SEG display set itself
  let date = extractDateFromDisplaySet(displaySet);
  if (date) {
    console.log('  Using SEG displaySet date:', date);
    return date;
  }
  
  // Last resort: try to get StudyDate from the study level
  if (displaySet.StudyInstanceUID && displaySetService) {
    const allDisplaySets = displaySetService.getActiveDisplaySets();
    // Find any display set from the same study that has a date
    const sameStudyDs = allDisplaySets.find(
      (ds: any) => ds.StudyInstanceUID === displaySet.StudyInstanceUID && 
                   (ds.StudyDate || (ds.instance && ds.instance.StudyDate))
    );
    if (sameStudyDs) {
      const studyDate = sameStudyDs.StudyDate || 
                       (sameStudyDs.instance && sameStudyDs.instance.StudyDate) ||
                       (sameStudyDs.instances && sameStudyDs.instances[0]?.StudyDate);
      if (studyDate && studyDate.length === 8) {
        console.log('  Found StudyDate from same study:', studyDate);
        return studyDate;
      }
    }
  }
  
  console.log('  No date found');
  return '';
}

/**
 * Calculate volumetrics from a labelmap (SEG)
 */
async function calculateVolumetricsForDisplaySet(
  displaySet: any,
  segmentationService: any,
  displaySetService: any
): Promise<VolumetricData | null> {
  try {
    // Get the segmentation for this display set
    const segmentations = segmentationService.getSegmentations();
    const segmentation = segmentations.find(
      (seg: any) => seg.segmentationId === displaySet.displaySetInstanceUID
    );

    if (!segmentation) {
      // Need to load the SEG first
      console.log('SEG not loaded yet:', displaySet.SeriesDescription);
      return null;
    }

    const representationData = segmentation.representationData;
    const labelmapData = representationData?.Labelmap;

    if (!labelmapData) {
      return null;
    }

    let segmentCounts: { [key: number]: number } = {};
    let spacing: number[] = [1, 1, 1];
    let dimensions: number[] = [0, 0, 0];

    // Try to get spacing from the referenced display set (the actual CT/MR images)
    const referencedDisplaySetUID = (displaySet as any).referencedDisplaySetInstanceUID;
    if (referencedDisplaySetUID) {
      const referencedDisplaySet = displaySetService.getDisplaySetByUID(referencedDisplaySetUID) as any;
      if (referencedDisplaySet?.images?.[0]) {
        const refImage = referencedDisplaySet.images[0];
        // Get spacing from referenced image metadata
        const rowSpacing = refImage.rowPixelSpacing || refImage.PixelSpacing?.[0] || 1;
        const colSpacing = refImage.columnPixelSpacing || refImage.PixelSpacing?.[1] || 1;
        // Use SliceThickness for Z spacing (actual thickness of each slice)
        const zSpacing = refImage.sliceThickness || refImage.SliceThickness || 1;
        spacing = [colSpacing, rowSpacing, zSpacing];
        console.log('[Volumetrics] Using spacing from referenced display set:', spacing);
      }
    }

    if ('volumeId' in labelmapData && labelmapData.volumeId) {
      const labelmapVolume = cache.getVolume(labelmapData.volumeId);
      if (!labelmapVolume) {
        return null;
      }

      const voxelManager = labelmapVolume.voxelManager;
      dimensions = labelmapVolume.dimensions as number[];
      
      // Use labelmap volume spacing if we didn't get it from referenced display set
      if (spacing[0] === 1 && spacing[1] === 1 && spacing[2] === 1) {
        spacing = labelmapVolume.spacing as number[];
      }
      console.log('[Volumetrics] Volume spacing:', labelmapVolume.spacing, 'Using:', spacing);

      let scalarData;
      if (voxelManager && typeof voxelManager.getScalarData === 'function') {
        scalarData = voxelManager.getScalarData();
      } else if (labelmapVolume.scalarData) {
        scalarData = labelmapVolume.scalarData;
      } else {
        return null;
      }

      for (let i = 0; i < scalarData.length; i++) {
        const value = scalarData[i];
        if (value > 0) {
          segmentCounts[value] = (segmentCounts[value] || 0) + 1;
        }
      }
    } else if ('imageIds' in labelmapData && labelmapData.imageIds) {
      const imageIds = labelmapData.imageIds;
      if (!imageIds || imageIds.length === 0) {
        return null;
      }

      const firstImage = cache.getImage(imageIds[0]) as any;
      if (firstImage) {
        // Only use image spacing if we didn't get it from referenced display set
        if (spacing[0] === 1 && spacing[1] === 1 && spacing[2] === 1) {
          const rowSpacing = firstImage.rowPixelSpacing || 1;
          const colSpacing = firstImage.columnPixelSpacing || 1;
          // Use sliceThickness for Z spacing
          const zSpacing = firstImage.sliceThickness || 1;
          spacing = [colSpacing, rowSpacing, zSpacing];
        }
        dimensions = [firstImage.columns || 0, firstImage.rows || 0, imageIds.length];
        console.log('[Volumetrics] Stack image spacing:', spacing);
      }

      for (const imageId of imageIds) {
        const image = cache.getImage(imageId) as any;
        if (!image) continue;

        const voxelManager = image.voxelManager;
        let scalarData;

        if (voxelManager && typeof voxelManager.getScalarData === 'function') {
          scalarData = voxelManager.getScalarData();
        } else if (image.getPixelData) {
          scalarData = image.getPixelData();
        } else {
          continue;
        }

        for (let i = 0; i < scalarData.length; i++) {
          const value = scalarData[i];
          if (value > 0) {
            segmentCounts[value] = (segmentCounts[value] || 0) + 1;
          }
        }
      }
    } else {
      return null;
    }

    const voxelVolumeMm3 = spacing[0] * spacing[1] * spacing[2];
    const segments = segmentation.segments || {};
    let totalVolumeMl = 0;
    const segmentReports: any[] = [];

    console.log('[Volumetrics] Final spacing used:', spacing, '-> Voxel volume:', voxelVolumeMm3, 'mm³');

    for (const [segmentIndex, voxelCount] of Object.entries(segmentCounts)) {
      const segIdx = parseInt(segmentIndex);
      const volumeMm3 = (voxelCount as number) * voxelVolumeMm3;
      const volumeCm3 = volumeMm3 / 1000; // 1 cm³ = 1000 mm³

      console.log(`[Volumetrics] Segment ${segIdx}: ${voxelCount} voxels × ${voxelVolumeMm3} mm³/voxel = ${volumeMm3} mm³ = ${volumeCm3.toFixed(4)} cm³`);

      const segment = segments[segIdx];
      const label = segment?.label || `Segment ${segIdx}`;

      segmentReports.push({
        segmentIndex: segIdx,
        label: label,
        volumeMl: volumeCm3, // cm³ = ml
        voxelCount: voxelCount,
      });

      totalVolumeMl += volumeCm3;
    }

    segmentReports.sort((a, b) => a.segmentIndex - b.segmentIndex);

    // Extract the best available date from the display set (with fallback to referenced display set)
    const studyDate = extractDateWithFallback(displaySet, displaySetService);

    return {
      studyDate: studyDate,
      studyDescription: displaySet.StudyDescription || '',
      studyInstanceUID: displaySet.StudyInstanceUID || '',
      seriesDescription: displaySet.SeriesDescription || '',
      seriesInstanceUID: displaySet.SeriesInstanceUID || '',
      displaySetInstanceUID: displaySet.displaySetInstanceUID,
      segments: segmentReports,
      totalVolumeMl: totalVolumeMl,
      spacing: spacing,
      dimensions: dimensions,
    };
  } catch (error) {
    console.error('Error calculating volumetrics:', error);
    return null;
  }
}

/**
 * Panel component for longitudinal volumetrics tracking
 */
function PanelLongitudinalVolumetrics() {
  const { servicesManager, extensionManager } = useSystem();
  const { 
    displaySetService, 
    segmentationService, 
    uiNotificationService,
    hangingProtocolService,
    cornerstoneViewportService,
    viewportGridService,
  } = servicesManager.services;

  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [report, setReport] = useState<LongitudinalReport | null>(null);
  const [segDisplaySets, setSegDisplaySets] = useState<any[]>([]);
  const [selectedSegments, setSelectedSegments] = useState<string[]>([]);
  const [allPatientStudies, setAllPatientStudies] = useState<any[]>([]);

  // Get the data source
  const getDataSource = useCallback(() => {
    const dataSourceDefinition = extensionManager.getActiveDataSource();
    return dataSourceDefinition?.[0];
  }, [extensionManager]);

  // Get current patient ID from loaded display sets
  const getCurrentPatientId = useCallback(() => {
    const allDisplaySets = displaySetService.getActiveDisplaySets();
    for (const ds of allDisplaySets) {
      const instance = ds.instance || (ds.instances && ds.instances[0]);
      if (instance?.PatientID) {
        return instance.PatientID;
      }
    }
    return null;
  }, [displaySetService]);

  // Fetch all studies for the current patient from the data source
  const fetchAllPatientStudies = useCallback(async () => {
    const patientId = getCurrentPatientId();
    if (!patientId) {
      console.log('No patient ID found');
      return [];
    }

    console.log('Fetching all studies for patient:', patientId);
    const dataSource = getDataSource();
    
    if (!dataSource?.query?.studies?.search) {
      console.log('Data source query not available');
      return [];
    }

    try {
      const studies = await dataSource.query.studies.search({
        patientId: patientId,
      });
      console.log('Found studies for patient:', studies);
      setAllPatientStudies(studies);
      return studies;
    } catch (error) {
      console.error('Error fetching patient studies:', error);
      return [];
    }
  }, [getCurrentPatientId, getDataSource]);

  // Find all SEG display sets from currently loaded studies
  const findSegDisplaySets = useCallback(() => {
    const allDisplaySets = displaySetService.getActiveDisplaySets();
    console.log('All display sets:', allDisplaySets.map((ds: any) => ({
      Modality: ds.Modality,
      SeriesDescription: ds.SeriesDescription,
      StudyDate: ds.StudyDate,
      SeriesDate: ds.SeriesDate,
      StudyInstanceUID: ds.StudyInstanceUID,
      displaySetInstanceUID: ds.displaySetInstanceUID,
      referencedDisplaySetInstanceUID: ds.referencedDisplaySetInstanceUID,
      hasInstance: !!ds.instance,
      instanceStudyDate: ds.instance?.StudyDate,
    })));
    
    const segs = allDisplaySets.filter((ds: any) => ds.Modality === 'SEG');
    console.log('Found SEG display sets:', segs.length);
    segs.forEach((seg: any, idx: number) => {
      console.log(`SEG ${idx}:`, {
        SeriesDescription: seg.SeriesDescription,
        StudyDate: seg.StudyDate,
        SeriesDate: seg.SeriesDate,
        StudyInstanceUID: seg.StudyInstanceUID,
        referencedDisplaySetInstanceUID: seg.referencedDisplaySetInstanceUID,
        instanceStudyDate: seg.instance?.StudyDate,
      });
    });
    
    setSegDisplaySets(segs);
    return segs;
  }, [displaySetService]);

  // Load a study's display sets into OHIF
  const loadStudyDisplaySets = useCallback(async (studyInstanceUID: string, forceRefresh = false) => {
    console.log('[Longitudinal] Loading study:', studyInstanceUID, 'forceRefresh:', forceRefresh);
    const dataSource = getDataSource();
    
    if (!dataSource?.retrieve?.series?.metadata) {
      console.log('[Longitudinal] Cannot retrieve series metadata');
      return;
    }

    try {
      // Clear the metadata cache if forcing refresh (to pick up newly saved SEG files)
      if (forceRefresh && dataSource?.deleteStudyMetadataPromise) {
        console.log('[Longitudinal] Clearing metadata cache for study:', studyInstanceUID);
        dataSource.deleteStudyMetadataPromise(studyInstanceUID);
      }
      
      // First, query the series in this study to see what's available
      if (dataSource?.query?.series?.search) {
        try {
          const seriesList = await dataSource.query.series.search(studyInstanceUID);
          console.log('[Longitudinal] Series query for study:', studyInstanceUID, 
            'found', seriesList?.length || 0, 'series');
          seriesList?.forEach((s: any, idx: number) => {
            console.log(`[Longitudinal] Series ${idx}:`, {
              SeriesInstanceUID: s.SeriesInstanceUID,
              Modality: s.Modality,
              SeriesDescription: s.SeriesDescription,
            });
          });
        } catch (e) {
          console.log('[Longitudinal] Series query not available:', e);
        }
      }
      
      // Get series for this study - this triggers instance loading
      const seriesMetadata = await dataSource.retrieve.series.metadata({
        StudyInstanceUID: studyInstanceUID,
      });
      
      // Log the series metadata returned
      const seriesArray = Array.isArray(seriesMetadata) ? seriesMetadata : Object.values(seriesMetadata || {});
      console.log('[Longitudinal] Retrieved series metadata:', seriesArray.length, 'series');
      seriesArray.forEach((s: any, idx: number) => {
        console.log(`[Longitudinal] Loaded series ${idx}:`, {
          SeriesInstanceUID: s.SeriesInstanceUID,
          Modality: s.Modality,
          SeriesDescription: s.SeriesDescription,
        });
      });
      
      // Wait longer for display sets to be created (SEG needs referenced series first)
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Log what display sets now exist
      const allDisplaySets = displaySetService.getActiveDisplaySets();
      const studyDisplaySets = allDisplaySets.filter((ds: any) => ds.StudyInstanceUID === studyInstanceUID);
      console.log('[Longitudinal] Display sets after loading study:', studyInstanceUID,
        'total:', studyDisplaySets.length);
      studyDisplaySets.forEach((ds: any, idx: number) => {
        console.log(`[Longitudinal] DisplaySet ${idx}:`, {
          Modality: ds.Modality,
          SeriesDescription: ds.SeriesDescription,
          displaySetInstanceUID: ds.displaySetInstanceUID?.substring(0, 8) + '...',
        });
      });
      
    } catch (error) {
      console.error('[Longitudinal] Error loading study display sets:', error);
    }
  }, [getDataSource, displaySetService]);

  // Calculate volumetrics for all SEG display sets across ALL patient studies
  const calculateLongitudinalVolumetrics = useCallback(async () => {
    setLoading(true);
    setLoadingMessage('Fetching patient studies...');
    
    try {
      // Step 1: Get all studies for this patient
      const patientStudies = await fetchAllPatientStudies();
      console.log('[Longitudinal] Patient has', patientStudies.length, 'studies');
      patientStudies.forEach((s: any, idx: number) => {
        console.log(`[Longitudinal] Study ${idx}:`, {
          studyInstanceUid: s.studyInstanceUid || s.StudyInstanceUID,
          studyDescription: s.studyDescription || s.StudyDescription,
          studyDate: s.studyDate || s.StudyDate,
          modalities: s.modalities || s.ModalitiesInStudy,
        });
      });
      
      if (patientStudies.length === 0) {
        uiNotificationService.show({
          title: 'Longitudinal Volumetrics',
          message: 'Could not fetch patient studies.',
          type: 'warning',
          duration: 5000,
        });
        setLoading(false);
        return;
      }

      // Step 2: Load display sets for all patient studies
      // Note: We need to load ALL series for each study, not just check if the study has any display sets
      // because a study might have MR loaded but not SEG
      setLoadingMessage(`Loading ${patientStudies.length} studies...`);
      
      // Get currently loaded study UIDs and their loaded modalities
      const currentDisplaySets = displaySetService.getActiveDisplaySets();
      const loadedStudyModalities = new Map<string, Set<string>>();
      currentDisplaySets.forEach((ds: any) => {
        const studyUID = ds.StudyInstanceUID;
        if (!loadedStudyModalities.has(studyUID)) {
          loadedStudyModalities.set(studyUID, new Set());
        }
        loadedStudyModalities.get(studyUID)!.add(ds.Modality);
      });
      
      console.log('[Longitudinal] Currently loaded studies and modalities:', 
        Array.from(loadedStudyModalities.entries()).map(([uid, mods]) => ({
          studyUID: uid.substring(0, 20) + '...',
          modalities: Array.from(mods)
        }))
      );
      
      for (const study of patientStudies) {
        const studyUID = study.studyInstanceUid || study.StudyInstanceUID;
        if (!studyUID) continue;
        
        // Check if this study has SEG loaded, not just any modality
        const loadedMods = loadedStudyModalities.get(studyUID);
        const hasSegLoaded = loadedMods?.has('SEG');
        
        // Always try to load studies that don't have SEG loaded
        // Use forceRefresh=true to clear cache and pick up newly saved SEG files
        if (!hasSegLoaded) {
          console.log('[Longitudinal] Loading study (SEG not loaded yet):', studyUID);
          setLoadingMessage(`Loading study ${study.studyDescription || studyUID}...`);
          await loadStudyDisplaySets(studyUID, true); // forceRefresh=true to clear metadata cache
        } else {
          console.log('[Longitudinal] Study already has SEG loaded:', studyUID);
        }
      }

      // Wait for display sets to be created
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Step 3: Find all SEG display sets
      setLoadingMessage('Finding segmentations...');
      let segs = findSegDisplaySets();
      
      if (segs.length === 0) {
        uiNotificationService.show({
          title: 'Longitudinal Volumetrics',
          message: 'No segmentation (DICOM SEG) files found across all patient studies.',
          type: 'warning',
          duration: 5000,
        });
        setLoading(false);
        return;
      }

      // Group segmentations by study only to get one segmentation per study (most recent)
      // This ensures longitudinal tracking uses only the latest segmentation from each timepoint
      const segsByStudy = new Map<string, any[]>();
      for (const seg of segs) {
        const studyUID = seg.StudyInstanceUID;
        
        if (!segsByStudy.has(studyUID)) {
          segsByStudy.set(studyUID, []);
        }
        segsByStudy.get(studyUID)!.push(seg);
      }

      console.log('[Longitudinal Filter] Studies with multiple SEGs:', 
        Array.from(segsByStudy.entries())
          .filter(([_, segs]) => segs.length > 1)
          .map(([studyUID, segs]) => ({
            studyUID: studyUID.substring(0, 20) + '...',
            count: segs.length,
            descriptions: segs.map(s => s.SeriesDescription)
          }))
      );

      // For each study, pick the most recent SEG based on series description timestamp
      const filteredSegs: any[] = [];
      for (const [studyUID, studySegs] of segsByStudy.entries()) {
        if (studySegs.length === 1) {
          filteredSegs.push(studySegs[0]);
        } else {
          // Sort by series description timestamp (most recent first) then by other criteria
          const sortedSegs = studySegs.sort((a, b) => {
            // Extract timestamp from series description (format: "nnUNet Auto Segmentation - DD/MM/YYYY, HH:MM:SS")
            const extractTimestamp = (seriesDesc: string) => {
              const match = seriesDesc.match(/(\d{2}\/\d{2}\/\d{4}), (\d{2}:\d{2}:\d{2})/);
              if (match) {
                const [, dateStr, timeStr] = match;
                const [day, month, year] = dateStr.split('/').map(Number);
                const [hour, minute, second] = timeStr.split(':').map(Number);
                return new Date(year, month - 1, day, hour, minute, second).getTime();
              }
              return 0;
            };
            
            const aTimestamp = extractTimestamp(a.SeriesDescription || '');
            const bTimestamp = extractTimestamp(b.SeriesDescription || '');
            
            if (aTimestamp && bTimestamp && aTimestamp !== bTimestamp) {
              return bTimestamp - aTimestamp; // Most recent first
            }
            
            // Fallback: Try SeriesDate + SeriesTime
            const aSeriesDateTime = (a.SeriesDate || '') + (a.SeriesTime || '');
            const bSeriesDateTime = (b.SeriesDate || '') + (b.SeriesTime || '');
            if (aSeriesDateTime && bSeriesDateTime) {
              return bSeriesDateTime.localeCompare(aSeriesDateTime); // Descending (most recent first)
            }
            
            // Try instance-level dates
            const aInstance = a.instances?.[0] || {};
            const bInstance = b.instances?.[0] || {};
            const aInstanceDate = (aInstance.InstanceCreationDate || aInstance.ContentDate || '') + 
                                  (aInstance.InstanceCreationTime || aInstance.ContentTime || '');
            const bInstanceDate = (bInstance.InstanceCreationDate || bInstance.ContentDate || '') + 
                                  (bInstance.InstanceCreationTime || bInstance.ContentTime || '');
            if (aInstanceDate && bInstanceDate) {
              return bInstanceDate.localeCompare(aInstanceDate); // Descending
            }
            
            // Fallback: use SeriesInstanceUID (higher UID typically means more recent)
            return (b.SeriesInstanceUID || '').localeCompare(a.SeriesInstanceUID || '');
          });
          
          console.log(`Study ${studyUID.substring(0, 20)}... has ${studySegs.length} SEGs, using most recent:`, sortedSegs[0].SeriesDescription || 'Unknown');
          console.log(`  Rejected:`, sortedSegs.slice(1).map(s => s.SeriesDescription));
          filteredSegs.push(sortedSegs[0]);
        }
      }
      segs = filteredSegs;
      console.log(`Filtered to ${segs.length} SEGs (one most recent segmentation per study)`);
      console.log('Final SEGs used:', segs.map(s => ({ 
        studyUID: s.StudyInstanceUID?.substring(0, 20) + '...',
        description: s.SeriesDescription 
      })));

      uiNotificationService.show({
        title: 'Longitudinal Volumetrics',
        message: `Found ${segs.length} segmentation(s) across ${patientStudies.length} studies (one per study)`,
        type: 'info',
        duration: 3000,
      });

      // Step 4: Load all SEGs that aren't loaded yet
      setLoadingMessage('Loading segmentation files...');
      for (let i = 0; i < segs.length; i++) {
        const seg = segs[i];
        setLoadingMessage(`Loading SEG ${i + 1}/${segs.length}: ${seg.SeriesDescription || 'Segmentation'}...`);
        
        // Skip if this SEG has a previous load error (e.g., orientation mismatch)
        if (seg.loadError) {
          console.log(`[Longitudinal] Skipping SEG with load error: ${seg.SeriesDescription} (${seg.loadError})`);
          continue;
        }
        
        if (seg.load && !seg.isLoaded) {
          try {
            await seg.load({ headers: {} });
            // Wait a bit for segmentation service to register it
            await new Promise(resolve => setTimeout(resolve, 500));
          } catch (e) {
            const errorMsg = e?.message || String(e);
            if (errorMsg.includes('orientation mismatch')) {
              console.log(`[Longitudinal] SEG orientation mismatch (cannot calculate volume): ${seg.SeriesDescription}`);
              // Mark for skipping in volumetrics
              seg.loadError = 'orientation_mismatch';
            } else {
              console.log('[Longitudinal] SEG already loaded or error loading:', e);
            }
          }
        }
      }

      // Filter out SEGs that failed to load
      const loadableSegs = segs.filter(seg => !seg.loadError);
      
      // Show warning if some SEGs couldn't be loaded
      const failedSegs = segs.filter(seg => seg.loadError);
      if (failedSegs.length > 0) {
        uiNotificationService.show({
          title: 'Longitudinal Volumetrics',
          message: `${failedSegs.length} segmentation(s) could not be loaded due to orientation mismatch with source images.`,
          type: 'warning',
          duration: 6000,
        });
      }

      // Wait for segmentations to be registered
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Step 5: Calculate volumetrics for each loadable SEG
      setLoadingMessage('Calculating volumes...');
      const timepoints: VolumetricData[] = [];
      
      for (const seg of loadableSegs) {
        const volumeData = await calculateVolumetricsForDisplaySet(seg, segmentationService, displaySetService);
        if (volumeData) {
          timepoints.push(volumeData);
        }
      }

      if (timepoints.length === 0) {
        const errorMsg = failedSegs.length > 0 
          ? 'All segmentations have orientation mismatches. The SEG files were saved in a different orientation than the source images.'
          : 'Could not calculate volumetrics. Please ensure SEG files are loaded.';
        uiNotificationService.show({
          title: 'Longitudinal Volumetrics',
          message: errorMsg,
          type: 'warning',
          duration: 5000,
        });
        setLoading(false);
        return;
      }

      // Sort by study date
      timepoints.sort((a, b) => {
        const dateA = a.studyDate || '';
        const dateB = b.studyDate || '';
        return dateA.localeCompare(dateB);
      });

      // Get patient info from first SEG
      const firstSeg = segs[0];
      const instances = firstSeg.instances || [];
      const firstInstance = instances[0] || {};

      // Normalize label function for consistent grouping across different segmentation sources
      // This handles various naming conventions from nnUNet, nnInteractive, and manual segmentations
      const normalizeLabel = (label: string) => {
        let normalized = label.trim().toLowerCase();
        
        // Remove "background;" prefix
        normalized = normalized.replace(/^background;?\s*/i, '');
        
        // Handle nninter_pred_* patterns -> "segmentation"
        if (normalized.match(/^nninter_pred_\d+$/)) {
          return 'segmentation';
        }
        
        // Handle nnunet_init_* patterns -> "segmentation"
        if (normalized.match(/^nnunet_init_\d+$/)) {
          return 'segmentation';
        }
        
        // Handle nnunet_auto_pred_* patterns -> "segmentation"
        if (normalized.match(/^nnunet_auto_pred_\d+$/)) {
          return 'segmentation';
        }
        
        // Handle "nnunet + interactive" and similar -> "segmentation"
        if (normalized.includes('nnunet') || normalized.includes('interactive')) {
          return 'segmentation';
        }
        
        // Handle "segment 1", "segment 2", etc. -> keep as is but normalize
        const segmentMatch = normalized.match(/segment\s*(\d+)/);
        if (segmentMatch) {
          return `segment ${segmentMatch[1]}`;
        }
        
        // Normalize whitespace for anything else
        return normalized.replace(/\s+/g, ' ').trim();
      };

      // Calculate volume changes between consecutive timepoints
      const volumeChanges: LongitudinalReport['volumeChanges'] = [];
      
      // Group segments by NORMALIZED label across timepoints
      const normalizedToOriginal = new Map<string, string>();
      timepoints.forEach(tp => {
        tp.segments.forEach(seg => {
          const normalized = normalizeLabel(seg.label);
          console.log(`[Label Normalization] Original: "${seg.label}" -> Normalized: "${normalized}"`);
          if (!normalizedToOriginal.has(normalized)) {
            normalizedToOriginal.set(normalized, seg.label);
          }
        });
      });
      
      console.log('[Label Normalization] Unique normalized labels:', Array.from(normalizedToOriginal.keys()));
      console.log('[Label Normalization] Final segment labels:', Array.from(normalizedToOriginal.values()));
      
      const segmentLabels = new Set<string>(normalizedToOriginal.values());

      for (const label of segmentLabels) {
        const normalizedLabel = normalizeLabel(label);
        const changes: LongitudinalReport['volumeChanges'][0]['changes'] = [];
        
        for (let i = 1; i < timepoints.length; i++) {
          const prevTp = timepoints[i - 1];
          const currTp = timepoints[i];
          
          // Find segments by normalized label
          const prevSegment = prevTp.segments.find(s => normalizeLabel(s.label) === normalizedLabel);
          const currSegment = currTp.segments.find(s => normalizeLabel(s.label) === normalizedLabel);
          
          if (prevSegment && currSegment) {
            const absoluteChange = currSegment.volumeMl - prevSegment.volumeMl;
            const percentChange = prevSegment.volumeMl > 0 
              ? ((currSegment.volumeMl - prevSegment.volumeMl) / prevSegment.volumeMl) * 100 
              : 0;
            
            changes.push({
              fromDate: prevTp.studyDate,
              toDate: currTp.studyDate,
              fromVolume: prevSegment.volumeMl,
              toVolume: currSegment.volumeMl,
              absoluteChange: absoluteChange,
              percentChange: percentChange,
            });
          }
        }
        
        if (changes.length > 0) {
          volumeChanges.push({ label, changes });
        }
      }

      // Create the report
      const longitudinalReport: LongitudinalReport = {
        patientId: firstInstance.PatientID || 'Unknown',
        patientName: firstInstance.PatientName?.Alphabetic || firstInstance.PatientName || 'Unknown',
        timepoints: timepoints,
        volumeChanges: volumeChanges,
      };

      setReport(longitudinalReport);
      setSelectedSegments(Array.from(segmentLabels));

      uiNotificationService.show({
        title: 'Longitudinal Volumetrics',
        message: `Analyzed ${timepoints.length} timepoints with ${segmentLabels.size} segments`,
        type: 'success',
        duration: 3000,
      });

    } catch (error: any) {
      console.error('Longitudinal volumetrics error:', error);
      uiNotificationService.show({
        title: 'Error',
        message: error.message || 'Failed to calculate volumetrics',
        type: 'error',
        duration: 5000,
      });
    }
    
    setLoading(false);
    setLoadingMessage('');
  }, [findSegDisplaySets, fetchAllPatientStudies, loadStudyDisplaySets, segmentationService, displaySetService, uiNotificationService]);

  // Export report as CSV
  const exportToCSV = useCallback(() => {
    if (!report) return;

    let csv = 'Patient ID,Patient Name,Study Date,Series Description,Segment,Volume (cm³),Voxel Count\n';
    
    for (const tp of report.timepoints) {
      for (const seg of tp.segments) {
        csv += `"${report.patientId}","${report.patientName}","${formatDicomDate(tp.studyDate)}","${tp.seriesDescription}","${seg.label}",${seg.volumeMl.toFixed(4)},${seg.voxelCount}\n`;
      }
    }

    // Add volume changes section
    csv += '\n\nVolume Changes Over Time\n';
    csv += 'Segment,From Date,To Date,From Volume (cm³),To Volume (cm³),Absolute Change (cm³),Percent Change (%)\n';
    
    for (const vc of report.volumeChanges) {
      for (const change of vc.changes) {
        csv += `"${vc.label}","${formatDicomDate(change.fromDate)}","${formatDicomDate(change.toDate)}",${change.fromVolume.toFixed(4)},${change.toVolume.toFixed(4)},${change.absoluteChange.toFixed(4)},${change.percentChange.toFixed(2)}\n`;
      }
    }

    // Download the CSV
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `volumetrics_${report.patientId}_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [report]);

  // Capture viewport screenshot for a specific display set
  const captureViewportScreenshot = useCallback(async (displaySetInstanceUID: string): Promise<string | null> => {
    try {
      // Find the viewport element that's displaying this display set
      const viewportGridState = viewportGridService.getState();
      const viewports = viewportGridState?.viewports || new Map();
      
      let targetViewportId: string | null = null;
      
      // Find viewport showing this display set (or its referenced display set)
      for (const [viewportId, viewport] of viewports) {
        const displaySetUIDs = viewport?.displaySetInstanceUIDs || [];
        if (displaySetUIDs.includes(displaySetInstanceUID)) {
          targetViewportId = viewportId;
          break;
        }
      }

      // If not found, try finding viewport with the referenced display set
      if (!targetViewportId) {
        const displaySet = displaySetService.getDisplaySetByUID(displaySetInstanceUID);
        const referencedUID = displaySet?.referencedDisplaySetInstanceUID;
        if (referencedUID) {
          for (const [viewportId, viewport] of viewports) {
            const displaySetUIDs = viewport?.displaySetInstanceUIDs || [];
            if (displaySetUIDs.includes(referencedUID)) {
              targetViewportId = viewportId;
              break;
            }
          }
        }
      }

      if (!targetViewportId) {
        console.log('No viewport found for display set:', displaySetInstanceUID);
        return null;
      }

      // Get the DOM element for this viewport
      const viewportElement = document.querySelector(`[data-viewport-uid="${targetViewportId}"]`) as HTMLElement;
      if (!viewportElement) {
        console.log('Viewport element not found for:', targetViewportId);
        return null;
      }

      // Use html2canvas to capture the viewport
      const canvas = await html2canvas(viewportElement, {
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#000000',
      });

      return canvas.toDataURL('image/png');
    } catch (error) {
      console.error('Error capturing viewport screenshot:', error);
      return null;
    }
  }, [viewportGridService, displaySetService]);

  // Capture a bounding box cropped view of the tumor region on the largest area axial slice
  const captureTumorBoundingBox = useCallback(async (displaySetInstanceUID: string): Promise<string | null> => {
    try {
      const displaySet = displaySetService.getDisplaySetByUID(displaySetInstanceUID) as any;
      if (!displaySet) {
        console.log('[TumorCapture] Display set not found:', displaySetInstanceUID);
        return null;
      }

      console.log('[TumorCapture] Processing:', displaySet.SeriesDescription);

      // Get the segmentation
      const segmentationId = displaySet.displaySetInstanceUID;
      const segmentation = segmentationService.getSegmentation(segmentationId) as any;
      
      if (!segmentation) {
        console.log('[TumorCapture] Segmentation not found, trying to load...');
        // Try to load the segmentation first
        if (displaySet.load && !displaySet.isLoaded) {
          try {
            await displaySet.load({ headers: {} });
            await new Promise(resolve => setTimeout(resolve, 500));
          } catch (e) {
            console.log('[TumorCapture] Failed to load segmentation');
          }
        }
      }

      // Get labelmap representation
      const seg = segmentationService.getSegmentation(segmentationId) as any;
      const labelmapRep = seg?.representationData?.Labelmap as any;
      
      // Try to get volume data for bounding box calculation
      let volumeData: any = null;
      let dimensions: number[] = [0, 0, 0];
      let scalarData: any = null;
      
      if (labelmapRep?.volumeId) {
        volumeData = cache.getVolume(labelmapRep.volumeId) as any;
        if (volumeData) {
          dimensions = volumeData.dimensions;
          const voxelManager = volumeData.voxelManager;
          if (voxelManager && typeof voxelManager.getScalarData === 'function') {
            scalarData = voxelManager.getScalarData();
          } else if (volumeData.scalarData) {
            scalarData = volumeData.scalarData;
          }
        }
      }

      // Find the referenced display set and viewport
      const referencedUID = displaySet.referencedDisplaySetInstanceUID;
      const viewportGridState = viewportGridService.getState();
      const viewports = viewportGridState?.viewports || new Map();
      
      let targetViewportId: string | null = null;
      
      for (const [viewportId, viewport] of viewports) {
        const displaySetUIDs = (viewport as any)?.displaySetInstanceUIDs || [];
        if (displaySetUIDs.includes(referencedUID) || displaySetUIDs.includes(displaySetInstanceUID)) {
          targetViewportId = viewportId;
          break;
        }
      }

      // If no viewport found, try using the first available viewport
      if (!targetViewportId) {
        const viewportIds = Array.from(viewports.keys());
        if (viewportIds.length > 0) {
          targetViewportId = viewportIds[0] as string;
          console.log('[TumorCapture] Using first viewport:', targetViewportId);
        }
      }

      if (!targetViewportId) {
        console.log('[TumorCapture] No viewport available');
        return null;
      }

      // If we have volume data, find the largest slice and navigate to it
      if (volumeData && scalarData && dimensions[0] > 0) {
        // Find bounding box of segmentation and slice with largest area
        let minX = dimensions[0], maxX = 0;
        let minY = dimensions[1], maxY = 0;
        
        const sliceAreas: number[] = new Array(dimensions[2]).fill(0);
        const pixelsPerSlice = dimensions[0] * dimensions[1];

        for (let z = 0; z < dimensions[2]; z++) {
          for (let y = 0; y < dimensions[1]; y++) {
            for (let x = 0; x < dimensions[0]; x++) {
              const idx = z * pixelsPerSlice + y * dimensions[0] + x;
              if (scalarData[idx] > 0) {
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
                sliceAreas[z]++;
              }
            }
          }
        }

        // Find slice with maximum area
        let maxSliceIndex = 0;
        let maxArea = 0;
        for (let z = 0; z < sliceAreas.length; z++) {
          if (sliceAreas[z] > maxArea) {
            maxArea = sliceAreas[z];
            maxSliceIndex = z;
          }
        }

        if (maxArea > 0) {
          console.log(`[TumorCapture] Largest slice: ${maxSliceIndex} with ${maxArea} voxels`);
          
          // Navigate to the largest area slice
          const cornerstoneViewport = cornerstoneViewportService.getCornerstoneViewport(targetViewportId) as any;
          if (cornerstoneViewport && typeof cornerstoneViewport.setImageIdIndex === 'function') {
            await cornerstoneViewport.setImageIdIndex(maxSliceIndex);
            await new Promise(resolve => setTimeout(resolve, 300));
          }
        }
      }

      // Capture the viewport
      const viewportElement = document.querySelector(`[data-viewport-uid="${targetViewportId}"]`) as HTMLElement;
      if (!viewportElement) {
        console.log('[TumorCapture] Viewport element not found');
        return null;
      }

      console.log('[TumorCapture] Capturing viewport...');
      const fullCanvas = await html2canvas(viewportElement, {
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#000000',
      });

      // If we have bounding box info, crop the image
      if (volumeData && scalarData && dimensions[0] > 0) {
        // Recalculate bounding box for current slice
        let minX = dimensions[0], maxX = 0;
        let minY = dimensions[1], maxY = 0;
        const pixelsPerSlice = dimensions[0] * dimensions[1];
        
        for (let z = 0; z < dimensions[2]; z++) {
          for (let y = 0; y < dimensions[1]; y++) {
            for (let x = 0; x < dimensions[0]; x++) {
              const idx = z * pixelsPerSlice + y * dimensions[0] + x;
              if (scalarData[idx] > 0) {
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
              }
            }
          }
        }

        if (maxX > minX && maxY > minY) {
          // Add padding
          const bboxWidth = maxX - minX;
          const bboxHeight = maxY - minY;
          const paddingX = Math.max(20, Math.round(bboxWidth * 0.3));
          const paddingY = Math.max(20, Math.round(bboxHeight * 0.3));
          
          const cropMinX = Math.max(0, minX - paddingX);
          const cropMaxX = Math.min(dimensions[0] - 1, maxX + paddingX);
          const cropMinY = Math.max(0, minY - paddingY);
          const cropMaxY = Math.min(dimensions[1] - 1, maxY + paddingY);

          // Calculate crop coordinates in canvas space
          const scaleX = fullCanvas.width / dimensions[0];
          const scaleY = fullCanvas.height / dimensions[1];

          const cropX = Math.round(cropMinX * scaleX);
          const cropY = Math.round(cropMinY * scaleY);
          const cropWidth = Math.round((cropMaxX - cropMinX) * scaleX);
          const cropHeight = Math.round((cropMaxY - cropMinY) * scaleY);

          // Create cropped canvas
          const croppedCanvas = document.createElement('canvas');
          croppedCanvas.width = cropWidth;
          croppedCanvas.height = cropHeight;
          const ctx = croppedCanvas.getContext('2d');
          
          if (ctx && cropWidth > 0 && cropHeight > 0) {
            ctx.drawImage(fullCanvas, cropX, cropY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);
            console.log('[TumorCapture] Cropped image created');
            return croppedCanvas.toDataURL('image/png');
          }
        }
      }

      // Return full viewport if cropping failed
      console.log('[TumorCapture] Returning full viewport capture');
      return fullCanvas.toDataURL('image/png');
    } catch (error) {
      console.error('[TumorCapture] Error:', error);
      return null;
    }
  }, [displaySetService, segmentationService, viewportGridService, cornerstoneViewportService]);

  // Generate markdown report
  const generateMarkdownReport = useCallback((screenshotDataUrls: Map<string, string>): string => {
    if (!report) return '';

    let md = '# Longitudinal Volumetrics Report\n\n';
    md += `**Patient ID:** ${report.patientId}\n\n`;
    md += `**Patient Name:** ${report.patientName}\n\n`;
    md += `**Report Date:** ${new Date().toLocaleDateString()}\n\n`;
    md += `**Number of Timepoints:** ${report.timepoints.length}\n\n`;
    md += '---\n\n';

    // Volume Changes Summary
    if (report.volumeChanges.length > 0) {
      md += '## Volume Changes Summary\n\n';
      
      for (const vc of report.volumeChanges) {
        if (!selectedSegments.includes(vc.label)) continue;
        
        md += `### ${vc.label}\n\n`;
        md += '| From Date | To Date | From Volume (cm³) | To Volume (cm³) | Change (cm³) | Change (%) |\n';
        md += '|-----------|---------|-------------------|-----------------|--------------|------------|\n';
        
        for (const change of vc.changes) {
          const sign = change.percentChange >= 0 ? '+' : '';
          md += `| ${formatDicomDate(change.fromDate)} | ${formatDicomDate(change.toDate)} | ${change.fromVolume.toFixed(2)} | ${change.toVolume.toFixed(2)} | ${change.absoluteChange.toFixed(2)} | ${sign}${change.percentChange.toFixed(1)}% |\n`;
        }
        md += '\n';
      }
    }

    // Timepoint Details
    md += '## Timepoint Details\n\n';
    
    for (let i = 0; i < report.timepoints.length; i++) {
      const tp = report.timepoints[i];
      
      md += `### Timepoint ${i + 1}: ${formatDicomDate(tp.studyDate)}\n\n`;
      md += `**Series:** ${tp.seriesDescription}\n\n`;
      md += `**Total Volume:** ${tp.totalVolumeMl.toFixed(2)} cm³\n\n`;
      
      md += '| Segment | Volume (cm³) | Voxel Count |\n';
      md += '|---------|--------------|-------------|\n';
      
      for (const seg of tp.segments) {
        if (!selectedSegments.includes(seg.label)) continue;
        md += `| ${seg.label} | ${seg.volumeMl.toFixed(4)} | ${seg.voxelCount} |\n`;
      }
      md += '\n';

      // Add screenshot reference if available
      if (screenshotDataUrls.has(tp.displaySetInstanceUID)) {
        md += `![Timepoint ${i + 1} - Largest Area Axial Slice](timepoint_${i + 1}_axial.png)\n\n`;
      }
      
      md += '---\n\n';
    }

    return md;
  }, [report, selectedSegments]);

  // Export report as Markdown
  const exportToMarkdown = useCallback((screenshotDataUrls: Map<string, string>) => {
    if (!report) return;

    const markdown = generateMarkdownReport(screenshotDataUrls);
    
    const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `volumetrics_${report.patientId}_${new Date().toISOString().split('T')[0]}.md`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [report, generateMarkdownReport]);

  // Export report as PDF with viewport captures
  const exportToPDF = useCallback(async () => {
    if (!report) return;

    setLoading(true);
    setLoadingMessage('Generating PDF report...');

    try {
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
      });

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 15;
      const contentWidth = pageWidth - 2 * margin;
      let yPos = margin;

      // Helper function to add new page if needed
      const checkPageBreak = (requiredSpace: number) => {
        if (yPos + requiredSpace > pageHeight - margin) {
          pdf.addPage();
          yPos = margin;
        }
      };

      // Title
      pdf.setFontSize(18);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Longitudinal Volumetrics Report', pageWidth / 2, yPos, { align: 'center' });
      yPos += 12;

      // Patient Info
      pdf.setFontSize(12);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Patient ID: ${report.patientId}`, margin, yPos);
      yPos += 6;
      pdf.text(`Patient Name: ${report.patientName}`, margin, yPos);
      yPos += 6;
      pdf.text(`Report Date: ${new Date().toLocaleDateString()}`, margin, yPos);
      yPos += 6;
      pdf.text(`Number of Timepoints: ${report.timepoints.length}`, margin, yPos);
      yPos += 12;

      // Divider line
      pdf.setDrawColor(100, 100, 100);
      pdf.line(margin, yPos, pageWidth - margin, yPos);
      yPos += 8;

      // Volume Changes Summary
      if (report.volumeChanges.length > 0) {
        checkPageBreak(40);
        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Volume Changes Summary', margin, yPos);
        yPos += 8;

        pdf.setFontSize(10);
        pdf.setFont('helvetica', 'normal');

        for (const vc of report.volumeChanges) {
          if (!selectedSegments.includes(vc.label)) continue;
          
          checkPageBreak(25);
          pdf.setFont('helvetica', 'bold');
          pdf.text(vc.label, margin, yPos);
          yPos += 5;
          pdf.setFont('helvetica', 'normal');

          for (const change of vc.changes) {
            checkPageBreak(6);
            const changeText = `${formatDicomDate(change.fromDate)} → ${formatDicomDate(change.toDate)}: `;
            const volumeText = `${change.fromVolume.toFixed(2)} cm³ → ${change.toVolume.toFixed(2)} cm³ `;
            const percentText = `(${change.percentChange >= 0 ? '+' : ''}${change.percentChange.toFixed(1)}%)`;
            pdf.text(`  ${changeText}${volumeText}${percentText}`, margin, yPos);
            yPos += 5;
          }
          yPos += 3;
        }
        yPos += 5;
      }

      // Timepoints with screenshots
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      checkPageBreak(20);
      pdf.text('Timepoint Details', margin, yPos);
      yPos += 10;

      // First, capture all tumor bounding box screenshots
      // We need to load each segmentation into the viewport before capturing
      setLoadingMessage('Capturing tumor images for all timepoints...');
      const allScreenshots: Map<string, string> = new Map();
      
      // Get the first viewport ID to use for loading display sets
      const viewportGridState = viewportGridService.getState();
      const viewports = viewportGridState?.viewports || new Map();
      const viewportIds = Array.from(viewports.keys());
      const targetViewportId = viewportIds[0] as string;
      
      for (let i = 0; i < report.timepoints.length; i++) {
        const tp = report.timepoints[i];
        setLoadingMessage(`Loading and capturing timepoint ${i + 1}/${report.timepoints.length}...`);
        
        try {
          // Get the SEG display set and its referenced image display set
          const segDisplaySet = displaySetService.getDisplaySetByUID(tp.displaySetInstanceUID) as any;
          if (!segDisplaySet) {
            console.log(`[PDF] SEG display set not found for timepoint ${i + 1}`);
            continue;
          }
          
          const referencedUID = segDisplaySet.referencedDisplaySetInstanceUID;
          
          // Load the referenced image display set into the viewport
          if (referencedUID && targetViewportId) {
            try {
              // Use viewportGridService to set the display sets
              await viewportGridService.setDisplaySetsForViewport({
                viewportId: targetViewportId,
                displaySetInstanceUIDs: [referencedUID],
              });
              
              // Wait for the images to load
              await new Promise(resolve => setTimeout(resolve, 500));
              
              // Now add the segmentation overlay
              // Try to load the segmentation if not already loaded
              if (segDisplaySet.load && !segDisplaySet.isLoaded) {
                await segDisplaySet.load({ headers: {} });
              }
              
              // Add segmentation to viewport
              const segmentationId = segDisplaySet.displaySetInstanceUID;
              try {
                await (segmentationService as any).addSegmentationRepresentation(
                  targetViewportId,
                  {
                    segmentationId: segmentationId,
                    type: 'Labelmap',
                  }
                );
              } catch (segError) {
                console.log('[PDF] Segmentation may already be added:', segError);
              }
              
              // Wait for segmentation to render
              await new Promise(resolve => setTimeout(resolve, 500));
              
            } catch (loadError) {
              console.log(`[PDF] Error loading display sets for timepoint ${i + 1}:`, loadError);
            }
          }
          
          // Now capture the screenshot
          const screenshot = await captureTumorBoundingBox(tp.displaySetInstanceUID);
          if (screenshot) {
            allScreenshots.set(tp.displaySetInstanceUID, screenshot);
            console.log(`[PDF] Captured screenshot for timepoint ${i + 1}`);
          } else {
            console.log(`[PDF] Failed to capture screenshot for timepoint ${i + 1}`);
          }
          
        } catch (error) {
          console.error(`[PDF] Error processing timepoint ${i + 1}:`, error);
        }
        
        // Small delay between captures
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Now add each timepoint with its screenshot
      for (let i = 0; i < report.timepoints.length; i++) {
        const tp = report.timepoints[i];
        
        checkPageBreak(100);
        
        // Timepoint header
        pdf.setFontSize(12);
        pdf.setFont('helvetica', 'bold');
        pdf.text(`Timepoint ${i + 1}: ${formatDicomDate(tp.studyDate)}`, margin, yPos);
        yPos += 6;
        
        pdf.setFontSize(10);
        pdf.setFont('helvetica', 'normal');
        pdf.text(`Series: ${tp.seriesDescription}`, margin, yPos);
        yPos += 5;
        pdf.text(`Total Volume: ${tp.totalVolumeMl.toFixed(2)} cm³`, margin, yPos);
        yPos += 6;

        // Segment volumes table
        pdf.setFont('helvetica', 'bold');
        pdf.text('Segment', margin, yPos);
        pdf.text('Volume (cm³)', margin + 60, yPos);
        pdf.text('Voxel Count', margin + 100, yPos);
        yPos += 5;
        
        pdf.setFont('helvetica', 'normal');
        for (const seg of tp.segments) {
          if (!selectedSegments.includes(seg.label)) continue;
          checkPageBreak(6);
          pdf.text(seg.label.substring(0, 25), margin, yPos);
          pdf.text(seg.volumeMl.toFixed(4), margin + 60, yPos);
          pdf.text(seg.voxelCount.toString(), margin + 100, yPos);
          yPos += 5;
        }
        
        // Add the tumor bounding box screenshot
        const screenshot = allScreenshots.get(tp.displaySetInstanceUID);
        
        if (screenshot) {
          checkPageBreak(70);
          yPos += 3;
          
          // Add screenshot to PDF - tumor bounding box cropped view
          const imgWidth = contentWidth * 0.6;
          const imgHeight = imgWidth; // Square-ish for bounding box crop
          const imgX = margin + (contentWidth - imgWidth) / 2;
          
          try {
            pdf.addImage(screenshot, 'PNG', imgX, yPos, imgWidth, imgHeight);
            yPos += imgHeight + 3;
            
            // Add caption
            pdf.setFontSize(8);
            pdf.setFont('helvetica', 'italic');
            pdf.text('Tumor region - largest axial slice', pageWidth / 2, yPos, { align: 'center' });
            yPos += 5;
          } catch (imgError) {
            console.error('Error adding image to PDF:', imgError);
            pdf.text('(Screenshot not available)', margin, yPos);
            yPos += 5;
          }
        }
        
        yPos += 8;
        
        // Divider between timepoints
        if (i < report.timepoints.length - 1) {
          checkPageBreak(5);
          pdf.setDrawColor(180, 180, 180);
          pdf.line(margin, yPos, pageWidth - margin, yPos);
          yPos += 8;
        }
      }

      // Bar chart representation (simplified text version)
      checkPageBreak(40);
      pdf.addPage();
      yPos = margin;
      
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Volume Trend Over Time', margin, yPos);
      yPos += 10;

      // Create a simple text-based volume chart
      const uniqueLabels = Array.from(new Set(report.timepoints.flatMap(tp => tp.segments.map(s => s.label))))
        .filter(label => selectedSegments.includes(label));

      for (const label of uniqueLabels) {
        checkPageBreak(25);
        pdf.setFontSize(11);
        pdf.setFont('helvetica', 'bold');
        pdf.text(label, margin, yPos);
        yPos += 6;

        pdf.setFontSize(9);
        pdf.setFont('helvetica', 'normal');
        
        for (const tp of report.timepoints) {
          const seg = tp.segments.find(s => s.label === label);
          const volume = seg?.volumeMl || 0;
          const barLength = Math.min((volume / 100) * contentWidth, contentWidth - 40);
          
          checkPageBreak(6);
          pdf.text(formatDicomDate(tp.studyDate).substring(5), margin, yPos);
          
          // Draw bar
          pdf.setFillColor(66, 135, 245);
          pdf.rect(margin + 25, yPos - 3, barLength > 0 ? barLength : 1, 4, 'F');
          
          // Volume label
          pdf.text(`${volume.toFixed(2)} cm³`, margin + 30 + barLength, yPos);
          yPos += 6;
        }
        yPos += 5;
      }

      // Footer
      const totalPages = pdf.getNumberOfPages();
      for (let i = 1; i <= totalPages; i++) {
        pdf.setPage(i);
        pdf.setFontSize(8);
        pdf.setFont('helvetica', 'normal');
        pdf.text(
          `Page ${i} of ${totalPages} | Generated by OHIF Viewer`,
          pageWidth / 2,
          pageHeight - 8,
          { align: 'center' }
        );
      }

      // Download PDF
      pdf.save(`volumetrics_report_${report.patientId}_${new Date().toISOString().split('T')[0]}.pdf`);

      uiNotificationService.show({
        title: 'PDF Export',
        message: 'Report exported successfully',
        type: 'success',
        duration: 3000,
      });
    } catch (error) {
      console.error('Error generating PDF:', error);
      uiNotificationService.show({
        title: 'PDF Export Error',
        message: 'Failed to generate PDF report',
        type: 'error',
        duration: 5000,
      });
    } finally {
      setLoading(false);
      setLoadingMessage('');
    }
  }, [report, selectedSegments, captureTumorBoundingBox, uiNotificationService]);

  // Get color for percentage change
  const getChangeColor = (percentChange: number): string => {
    if (percentChange > 10) return 'text-red-500';
    if (percentChange > 0) return 'text-orange-400';
    if (percentChange < -10) return 'text-green-500';
    if (percentChange < 0) return 'text-green-400';
    return 'text-gray-400';
  };

  // Get arrow icon for change direction
  const getChangeArrow = (percentChange: number): string => {
    if (percentChange > 0) return '↑';
    if (percentChange < 0) return '↓';
    return '→';
  };

  return (
    <div className="flex flex-col h-full bg-black text-white p-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Longitudinal Volumetrics</h2>
        <div className="flex gap-2">
          <Button
            variant="default"
            size="sm"
            onClick={calculateLongitudinalVolumetrics}
            disabled={loading}
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </Button>
          {report && (
            <>
              <Button
                variant="ghost"
                size="sm"
                onClick={exportToCSV}
                disabled={loading}
              >
                CSV
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => exportToMarkdown(new Map())}
                disabled={loading}
              >
                MD
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={exportToPDF}
                disabled={loading}
              >
                PDF
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Found SEGs indicator */}
      <div className="text-sm text-gray-400 mb-4">
        {allPatientStudies.length > 0 && (
          <div className="mb-1">Patient has {allPatientStudies.length} study/studies</div>
        )}
        {segDisplaySets.length > 0 
          ? `Found ${segDisplaySets.length} segmentation file(s)`
          : 'Click "Analyze" to find all patient studies and segmentations'
        }
      </div>

      {/* Loading state */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mb-3"></div>
          <span className="text-sm">{loadingMessage || 'Analyzing...'}</span>
        </div>
      )}

      {/* No report yet */}
      {!loading && !report && (
        <div className="flex flex-col items-center justify-center py-8 text-gray-400">
          <Icons.TabLinear className="w-12 h-12 mb-4 opacity-50" />
          <p className="text-center text-sm">
            Click "Analyze" to:<br />
            1. Fetch all studies for this patient<br />
            2. Load all segmentations (DICOM SEG)<br />
            3. Calculate and compare volumes over time
          </p>
        </div>
      )}

      {/* Report display */}
      {!loading && report && (
        <div className="space-y-6">
          {/* Patient Info */}
          <div className="bg-gray-900 rounded p-3">
            <h3 className="text-sm font-medium text-gray-300 mb-2">Patient Information</h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <span className="text-gray-400">Patient ID:</span>
              <span>{report.patientId}</span>
              <span className="text-gray-400">Patient Name:</span>
              <span>{report.patientName}</span>
              <span className="text-gray-400">Timepoints:</span>
              <span>{report.timepoints.length}</span>
            </div>
          </div>

          {/* Warning when only 1 timepoint */}
          {report.timepoints.length === 1 && (
            <div className="bg-yellow-900/30 border border-yellow-600 rounded p-3">
              <h3 className="text-sm font-medium text-yellow-400 mb-2">⚠️ Single Timepoint</h3>
              <p className="text-xs text-yellow-300/80">
                Only 1 study with segmentations is currently loaded. 
                To see volume changes over time, load multiple studies for this patient 
                (e.g., baseline and follow-up scans with saved DICOM SEG files).
              </p>
            </div>
          )}

          {/* Volume Changes Summary - Show first when multiple timepoints */}
          {report.volumeChanges.length > 0 && (
            <div className="bg-blue-900/30 border border-blue-500 rounded p-3">
              <h3 className="text-sm font-medium text-blue-300 mb-3">📊 Volume Changes Summary</h3>
              
              <div className="space-y-3">
                {report.volumeChanges
                  .filter(vc => selectedSegments.includes(vc.label))
                  .map(vc => {
                    // Calculate overall change from first to last
                    const firstVolume = vc.changes[0]?.fromVolume || 0;
                    const lastVolume = vc.changes[vc.changes.length - 1]?.toVolume || 0;
                    const overallAbsChange = lastVolume - firstVolume;
                    const overallPctChange = firstVolume > 0 
                      ? ((lastVolume - firstVolume) / firstVolume) * 100 
                      : 0;
                    
                    return (
                      <div key={vc.label} className="bg-gray-800 rounded p-2">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium">{vc.label}</span>
                          <span className={`text-lg font-bold ${getChangeColor(overallPctChange)}`}>
                            {getChangeArrow(overallPctChange)} {Math.abs(overallPctChange).toFixed(1)}%
                          </span>
                        </div>
                        <div className="text-xs text-gray-400">
                          {formatDicomDate(vc.changes[0]?.fromDate)}: <span className="text-white">{firstVolume.toFixed(2)} cm³</span>
                          {' → '}
                          {formatDicomDate(vc.changes[vc.changes.length - 1]?.toDate)}: <span className="text-white">{lastVolume.toFixed(2)} cm³</span>
                          {' '}
                          <span className={getChangeColor(overallPctChange)}>
                            ({overallAbsChange >= 0 ? '+' : ''}{overallAbsChange.toFixed(2)} cm³)
                          </span>
                        </div>
                        
                        {/* Show individual interval changes if more than 2 timepoints */}
                        {vc.changes.length > 1 && (
                          <div className="mt-2 pt-2 border-t border-gray-700">
                            <div className="text-xs text-gray-500 mb-1">Interval Changes:</div>
                            {vc.changes.map((change, idx) => (
                              <div key={idx} className="flex justify-between text-xs text-gray-400">
                                <span>{formatDicomDate(change.fromDate)} → {formatDicomDate(change.toDate)}</span>
                                <span className={getChangeColor(change.percentChange)}>
                                  {getChangeArrow(change.percentChange)} {Math.abs(change.percentChange).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })
                }
              </div>
            </div>
          )}

          {/* Timeline View */}
          <div className="bg-gray-900 rounded p-3">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Volume Timeline</h3>
            
            <div className="space-y-4">
              {report.timepoints.map((tp, index) => (
                <div key={tp.displaySetInstanceUID} className="border-l-2 border-blue-500 pl-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium">{formatDicomDate(tp.studyDate)}</span>
                    <span className="text-sm text-gray-400">{tp.seriesDescription}</span>
                  </div>
                  <div className="text-sm text-gray-300 mb-2">
                    Total: <span className="font-mono">{tp.totalVolumeMl.toFixed(2)} cm³</span>
                  </div>
                  <div className="grid grid-cols-1 gap-1">
                    {tp.segments
                      .filter(seg => selectedSegments.includes(seg.label))
                      .map(seg => (
                        <div 
                          key={seg.segmentIndex} 
                          className="flex justify-between text-xs bg-gray-800 rounded px-2 py-1"
                        >
                          <span>{seg.label}</span>
                          <span className="font-mono">{seg.volumeMl.toFixed(2)} cm³</span>
                        </div>
                      ))
                    }
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Volume Changes */}

          {/* Line Chart: Volume (Y) over Time (X) */}
          {report.timepoints.length > 0 && (
            <div className="bg-gray-900 rounded p-3">
              <h3 className="text-sm font-medium text-gray-300 mb-3">Volume Over Time</h3>
              
              {/* Get all unique segment labels across all timepoints */}
              {(() => {
                // Normalize label function to ensure consistency across different segmentation sources
                // This handles various naming conventions from nnUNet, nnInteractive, and manual segmentations
                const normalizeLabel = (label: string) => {
                  let normalized = label.trim().toLowerCase();
                  
                  // Remove "background;" prefix
                  normalized = normalized.replace(/^background;?\s*/i, '');
                  
                  // Handle nninter_pred_* patterns -> "segmentation"
                  if (normalized.match(/^nninter_pred_\d+$/)) {
                    return 'segmentation';
                  }
                  
                  // Handle nnunet_init_* patterns -> "segmentation"
                  if (normalized.match(/^nnunet_init_\d+$/)) {
                    return 'segmentation';
                  }
                  
                  // Handle nnunet_auto_pred_* patterns -> "segmentation"
                  if (normalized.match(/^nnunet_auto_pred_\d+$/)) {
                    return 'segmentation';
                  }
                  
                  // Handle "nnunet + interactive" and similar -> "segmentation"
                  if (normalized.includes('nnunet') || normalized.includes('interactive')) {
                    return 'segmentation';
                  }
                  
                  // Handle "segment 1", "segment 2", etc. -> keep as is but normalize
                  const segmentMatch = normalized.match(/segment\s*(\d+)/);
                  if (segmentMatch) {
                    return `segment ${segmentMatch[1]}`;
                  }
                  
                  // Normalize whitespace for anything else
                  return normalized.replace(/\s+/g, ' ').trim();
                };
                
                // Get all unique NORMALIZED segment labels
                const allRawLabels = report.timepoints.flatMap(tp => tp.segments.map(s => s.label));
                const normalizedToOriginal = new Map<string, string>();
                
                // Build mapping of normalized labels to their first occurrence (original form)
                for (const rawLabel of allRawLabels) {
                  const normalized = normalizeLabel(rawLabel);
                  if (!normalizedToOriginal.has(normalized)) {
                    normalizedToOriginal.set(normalized, rawLabel);
                  }
                }
                
                const allLabels = Array.from(normalizedToOriginal.values())
                  .filter(label => selectedSegments.includes(label));
                
                // Calculate global max for consistent Y-axis across all segments
                const allVolumes = allLabels.flatMap(label => {
                  const normalizedLabel = normalizeLabel(label);
                  return report.timepoints.map(tp => {
                    // Find segment by normalized label match
                    const seg = tp.segments.find(s => normalizeLabel(s.label) === normalizedLabel);
                    return seg?.volumeMl || 0;
                  });
                });
                const globalMaxVolume = Math.max(...allVolumes, 0.1);
                
                // Generate Y-axis tick values (5 ticks)
                const yTicks = [0, globalMaxVolume * 0.25, globalMaxVolume * 0.5, globalMaxVolume * 0.75, globalMaxVolume];
                
                // Colors for different segments
                const colors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4'];
                
                const chartHeight = 140;
                const chartWidth = 280;
                const padding = { left: 40, right: 20, top: 10, bottom: 30 };
                const plotWidth = chartWidth - padding.left - padding.right;
                const plotHeight = chartHeight - padding.top - padding.bottom;
                
                return (
                  <div className="mb-4">
                    {/* Legend */}
                    <div className="flex flex-wrap gap-3 mb-3">
                      {allLabels.map((label, idx) => (
                        <div key={label} className="flex items-center gap-1">
                          <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: colors[idx % colors.length] }}
                          />
                          <span className="text-xs text-gray-300">{label}</span>
                        </div>
                      ))}
                    </div>
                    
                    {/* SVG Line Chart */}
                    <svg 
                      viewBox={`0 0 ${chartWidth} ${chartHeight}`} 
                      className="w-full h-auto"
                      style={{ maxHeight: '180px' }}
                    >
                      {/* Y-axis grid lines and labels */}
                      {yTicks.map((tick, idx) => {
                        const y = padding.top + plotHeight - (tick / globalMaxVolume) * plotHeight;
                        return (
                          <g key={idx}>
                            <line 
                              x1={padding.left} 
                              y1={y} 
                              x2={chartWidth - padding.right} 
                              y2={y} 
                              stroke="#374151" 
                              strokeWidth="1"
                              strokeDasharray={idx === 0 ? "0" : "2,2"}
                            />
                            <text 
                              x={padding.left - 5} 
                              y={y + 3} 
                              textAnchor="end" 
                              fill="#6b7280" 
                              fontSize="8"
                            >
                              {tick < 10 ? tick.toFixed(1) : tick.toFixed(0)}
                            </text>
                          </g>
                        );
                      })}
                      
                      {/* Y-axis label */}
                      <text 
                        x={10} 
                        y={chartHeight / 2} 
                        textAnchor="middle" 
                        fill="#6b7280" 
                        fontSize="8"
                        transform={`rotate(-90, 10, ${chartHeight / 2})`}
                      >
                        Volume (cm³)
                      </text>
                      
                      {/* X-axis labels (dates) */}
                      {report.timepoints.map((tp, idx) => {
                        const x = padding.left + (idx / Math.max(report.timepoints.length - 1, 1)) * plotWidth;
                        return (
                          <text 
                            key={tp.displaySetInstanceUID}
                            x={x} 
                            y={chartHeight - 5} 
                            textAnchor="middle" 
                            fill="#6b7280" 
                            fontSize="7"
                          >
                            {formatDicomDate(tp.studyDate).substring(5)}
                          </text>
                        );
                      })}
                      
                      {/* Lines and points for each segment */}
                      {allLabels.map((segmentLabel, labelIdx) => {
                        const color = colors[labelIdx % colors.length];
                        const normalizedLabel = normalizeLabel(segmentLabel);
                        const points = report.timepoints.map((tp, idx) => {
                          // Find segment by normalized label match
                          const seg = tp.segments.find(s => normalizeLabel(s.label) === normalizedLabel);
                          const volume = seg?.volumeMl || 0;
                          const x = padding.left + (idx / Math.max(report.timepoints.length - 1, 1)) * plotWidth;
                          const y = padding.top + plotHeight - (volume / globalMaxVolume) * plotHeight;
                          return { x, y, volume, date: formatDicomDate(tp.studyDate) };
                        });
                        
                        // Create path for the line
                        const pathD = points.map((p, idx) => 
                          `${idx === 0 ? 'M' : 'L'} ${p.x} ${p.y}`
                        ).join(' ');
                        
                        return (
                          <g key={segmentLabel}>
                            {/* Line */}
                            <path 
                              d={pathD} 
                              fill="none" 
                              stroke={color} 
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                            
                            {/* Data points */}
                            {points.map((p, idx) => (
                              <g key={idx}>
                                <circle 
                                  cx={p.x} 
                                  cy={p.y} 
                                  r="4" 
                                  fill={color}
                                  stroke="#1f2937"
                                  strokeWidth="1"
                                  className="cursor-pointer"
                                />
                                {/* Tooltip - shows on hover via CSS */}
                                <title>{`${p.date}: ${p.volume.toFixed(2)} cm³`}</title>
                              </g>
                            ))}
                          </g>
                        );
                      })}
                    </svg>
                    
                    {/* X-axis label */}
                    <div className="text-xs text-gray-500 text-center mt-1">
                      Time
                    </div>
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default PanelLongitudinalVolumetrics;
