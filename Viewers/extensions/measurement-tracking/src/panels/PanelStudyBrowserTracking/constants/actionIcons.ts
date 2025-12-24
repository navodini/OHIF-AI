import type { actionIcon } from '../PanelStudyBrowserTracking/types/actionsIcon';

const defaultActionIcons = [
  {
    id: 'settings',
    iconName: 'Settings',
    value: true,  // Show tabs (Primary/Recent/All) by default for longitudinal comparison
  },
] as actionIcon[];

export { defaultActionIcons };
