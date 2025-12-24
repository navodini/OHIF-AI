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

    if ('volumeId' in labelmapData && labelmapData.volumeId) {
      const labelmapVolume = cache.getVolume(labelmapData.volumeId);
      if (!labelmapVolume) {
        return null;
      }

      const voxelManager = labelmapVolume.voxelManager;
      dimensions = labelmapVolume.dimensions as number[];
      spacing = labelmapVolume.spacing as number[];

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

      const firstImage = cache.getImage(imageIds[0]);
      if (firstImage) {
        const rowSpacing = firstImage.rowPixelSpacing || 1;
        const colSpacing = firstImage.columnPixelSpacing || 1;
        const sliceThickness = firstImage.sliceThickness || 1;
        spacing = [colSpacing, rowSpacing, sliceThickness];
        dimensions = [firstImage.columns || 0, firstImage.rows || 0, imageIds.length];
      }

      for (const imageId of imageIds) {
        const image = cache.getImage(imageId);
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

    for (const [segmentIndex, voxelCount] of Object.entries(segmentCounts)) {
      const segIdx = parseInt(segmentIndex);
      const volumeMm3 = (voxelCount as number) * voxelVolumeMm3;
      const volumeMl = volumeMm3 / 1000;

      const segment = segments[segIdx];
      const label = segment?.label || `Segment ${segIdx}`;

      segmentReports.push({
        segmentIndex: segIdx,
        label: label,
        volumeMl: volumeMl,
        voxelCount: voxelCount,
      });

      totalVolumeMl += volumeMl;
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
  const loadStudyDisplaySets = useCallback(async (studyInstanceUID: string) => {
    console.log('Loading study:', studyInstanceUID);
    const dataSource = getDataSource();
    
    if (!dataSource?.retrieve?.series?.metadata) {
      console.log('Cannot retrieve series metadata');
      return;
    }

    try {
      // Get series for this study
      const seriesMetadata = await dataSource.retrieve.series.metadata({
        StudyInstanceUID: studyInstanceUID,
      });
      
      console.log('Retrieved series metadata:', seriesMetadata?.length || 0, 'series');
      
      // The display sets should now be created by displaySetService
      // Wait a moment for processing
      await new Promise(resolve => setTimeout(resolve, 500));
      
    } catch (error) {
      console.error('Error loading study display sets:', error);
    }
  }, [getDataSource]);

  // Calculate volumetrics for all SEG display sets across ALL patient studies
  const calculateLongitudinalVolumetrics = useCallback(async () => {
    setLoading(true);
    setLoadingMessage('Fetching patient studies...');
    
    try {
      // Step 1: Get all studies for this patient
      const patientStudies = await fetchAllPatientStudies();
      console.log('Patient has', patientStudies.length, 'studies');
      
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
      setLoadingMessage(`Loading ${patientStudies.length} studies...`);
      const loadedStudyUIDs = new Set(
        displaySetService.getActiveDisplaySets().map((ds: any) => ds.StudyInstanceUID)
      );
      
      for (const study of patientStudies) {
        const studyUID = study.studyInstanceUid || study.StudyInstanceUID;
        if (studyUID && !loadedStudyUIDs.has(studyUID)) {
          console.log('Loading additional study:', studyUID);
          setLoadingMessage(`Loading study ${study.studyDescription || studyUID}...`);
          await loadStudyDisplaySets(studyUID);
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

      // Filter to use only the most recent SEG per study
      // This handles cases where multiple SEG versions exist for the same study
      const segsByStudy = new Map<string, any[]>();
      for (const seg of segs) {
        const studyUID = seg.StudyInstanceUID;
        if (!segsByStudy.has(studyUID)) {
          segsByStudy.set(studyUID, []);
        }
        segsByStudy.get(studyUID)!.push(seg);
      }

      // For each study, pick the most recent SEG (by SeriesDate/SeriesTime or InstanceCreationDate)
      const filteredSegs: any[] = [];
      for (const [studyUID, studySegs] of segsByStudy.entries()) {
        if (studySegs.length === 1) {
          filteredSegs.push(studySegs[0]);
        } else {
          // Sort by series date/time or instance creation date, most recent first
          const sortedSegs = studySegs.sort((a, b) => {
            // Try SeriesDate + SeriesTime first
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
          
          console.log(`Study ${studyUID} has ${studySegs.length} SEGs, using most recent:`, sortedSegs[0].SeriesDescription);
          filteredSegs.push(sortedSegs[0]);
        }
      }

      segs = filteredSegs;
      console.log(`Filtered to ${segs.length} SEGs (one per study)`);

      uiNotificationService.show({
        title: 'Longitudinal Volumetrics',
        message: `Found ${segs.length} segmentation(s) across ${patientStudies.length} studies`,
        type: 'info',
        duration: 3000,
      });

      // Step 4: Load all SEGs that aren't loaded yet
      setLoadingMessage('Loading segmentation files...');
      for (let i = 0; i < segs.length; i++) {
        const seg = segs[i];
        setLoadingMessage(`Loading SEG ${i + 1}/${segs.length}: ${seg.SeriesDescription || 'Segmentation'}...`);
        if (seg.load && !seg.isLoaded) {
          try {
            await seg.load({ headers: {} });
            // Wait a bit for segmentation service to register it
            await new Promise(resolve => setTimeout(resolve, 500));
          } catch (e) {
            console.log('SEG already loaded or error loading:', e);
          }
        }
      }

      // Wait for segmentations to be registered
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Step 5: Calculate volumetrics for each SEG
      setLoadingMessage('Calculating volumes...');
      const timepoints: VolumetricData[] = [];
      
      for (const seg of segs) {
        const volumeData = await calculateVolumetricsForDisplaySet(seg, segmentationService, displaySetService);
        if (volumeData) {
          timepoints.push(volumeData);
        }
      }

      if (timepoints.length === 0) {
        uiNotificationService.show({
          title: 'Longitudinal Volumetrics',
          message: 'Could not calculate volumetrics. Please ensure SEG files are loaded.',
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

      // Calculate volume changes between consecutive timepoints
      const volumeChanges: LongitudinalReport['volumeChanges'] = [];
      
      // Group segments by label across timepoints
      const segmentLabels = new Set<string>();
      timepoints.forEach(tp => {
        tp.segments.forEach(seg => segmentLabels.add(seg.label));
      });

      for (const label of segmentLabels) {
        const changes: LongitudinalReport['volumeChanges'][0]['changes'] = [];
        
        for (let i = 1; i < timepoints.length; i++) {
          const prevTp = timepoints[i - 1];
          const currTp = timepoints[i];
          
          const prevSegment = prevTp.segments.find(s => s.label === label);
          const currSegment = currTp.segments.find(s => s.label === label);
          
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

    let csv = 'Patient ID,Patient Name,Study Date,Series Description,Segment,Volume (ml),Voxel Count\n';
    
    for (const tp of report.timepoints) {
      for (const seg of tp.segments) {
        csv += `"${report.patientId}","${report.patientName}","${formatDicomDate(tp.studyDate)}","${tp.seriesDescription}","${seg.label}",${seg.volumeMl.toFixed(4)},${seg.voxelCount}\n`;
      }
    }

    // Add volume changes section
    csv += '\n\nVolume Changes Over Time\n';
    csv += 'Segment,From Date,To Date,From Volume (ml),To Volume (ml),Absolute Change (ml),Percent Change (%)\n';
    
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
            const volumeText = `${change.fromVolume.toFixed(2)} ml → ${change.toVolume.toFixed(2)} ml `;
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

      for (let i = 0; i < report.timepoints.length; i++) {
        const tp = report.timepoints[i];
        
        checkPageBreak(80);
        
        // Timepoint header
        pdf.setFontSize(12);
        pdf.setFont('helvetica', 'bold');
        pdf.text(`Timepoint ${i + 1}: ${formatDicomDate(tp.studyDate)}`, margin, yPos);
        yPos += 6;
        
        pdf.setFontSize(10);
        pdf.setFont('helvetica', 'normal');
        pdf.text(`Series: ${tp.seriesDescription}`, margin, yPos);
        yPos += 5;
        pdf.text(`Total Volume: ${tp.totalVolumeMl.toFixed(2)} ml`, margin, yPos);
        yPos += 6;

        // Segment volumes table
        pdf.setFont('helvetica', 'bold');
        pdf.text('Segment', margin, yPos);
        pdf.text('Volume (ml)', margin + 60, yPos);
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
        
        // Try to capture viewport screenshot
        setLoadingMessage(`Capturing screenshot for timepoint ${i + 1}...`);
        const screenshot = await captureViewportScreenshot(tp.displaySetInstanceUID);
        
        if (screenshot) {
          checkPageBreak(70);
          yPos += 3;
          
          // Add screenshot to PDF
          const imgWidth = contentWidth * 0.7;
          const imgHeight = imgWidth * 0.75; // Approximate aspect ratio
          const imgX = margin + (contentWidth - imgWidth) / 2;
          
          try {
            pdf.addImage(screenshot, 'PNG', imgX, yPos, imgWidth, imgHeight);
            yPos += imgHeight + 5;
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
          pdf.text(`${volume.toFixed(2)} ml`, margin + 30 + barLength, yPos);
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
  }, [report, selectedSegments, captureViewportScreenshot, uiNotificationService]);

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
                          {formatDicomDate(vc.changes[0]?.fromDate)}: <span className="text-white">{firstVolume.toFixed(2)} ml</span>
                          {' → '}
                          {formatDicomDate(vc.changes[vc.changes.length - 1]?.toDate)}: <span className="text-white">{lastVolume.toFixed(2)} ml</span>
                          {' '}
                          <span className={getChangeColor(overallPctChange)}>
                            ({overallAbsChange >= 0 ? '+' : ''}{overallAbsChange.toFixed(2)} ml)
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
                    Total: <span className="font-mono">{tp.totalVolumeMl.toFixed(2)} ml</span>
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
                          <span className="font-mono">{seg.volumeMl.toFixed(2)} ml</span>
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
                const allLabels = Array.from(new Set(report.timepoints.flatMap(tp => tp.segments.map(s => s.label))))
                  .filter(label => selectedSegments.includes(label));
                
                // Calculate global max for consistent Y-axis across all segments
                const allVolumes = allLabels.flatMap(label => 
                  report.timepoints.map(tp => {
                    const seg = tp.segments.find(s => s.label === label);
                    return seg?.volumeMl || 0;
                  })
                );
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
                        Volume (ml)
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
                        const points = report.timepoints.map((tp, idx) => {
                          const seg = tp.segments.find(s => s.label === segmentLabel);
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
                                <title>{`${p.date}: ${p.volume.toFixed(2)} ml`}</title>
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
