import { ARTryOn } from './ar-tryon-app';

declare global {
  interface Window {
    app?: ARTryOn;
  }
}

window.addEventListener('load', () => {
  const app = new ARTryOn();
  window.app = app;
});
