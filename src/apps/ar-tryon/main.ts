import './ar-tryon.css';
import { ARTryOn } from './ar-tryon-app';

declare global {
  interface Window {
    app?: ARTryOn;
  }
}

window.addEventListener('load', () => {
  window.app = new ARTryOn();
});
