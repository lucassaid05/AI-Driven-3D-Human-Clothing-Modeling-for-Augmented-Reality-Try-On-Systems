import { ARTryOn } from './ar-tryon-app';

declare global {
  interface Window {
    app?: ARTryOn;
  }
}

function bootTryOn() {
  if (window.app) {
    console.warn('[try-on] Already initialized — skipping duplicate boot (HMR or double script load)');
    return;
  }
  window.app = new ARTryOn();
}

if (document.readyState === 'complete') {
  bootTryOn();
} else {
  window.addEventListener('load', bootTryOn, { once: true });
}
