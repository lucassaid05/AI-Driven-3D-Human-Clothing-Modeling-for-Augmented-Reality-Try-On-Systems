/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './apps/**/*.html',
    './pages/**/*.html',
    './legacy/**/*.html',
    './src/**/*.{html,ts,css}',
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"Space Mono"', 'ui-monospace', 'monospace'],
        display: ['Syne', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      colors: {
        tryon: {
          bg: '#0a0a0f',
          panel: '#12121a',
          cyan: '#00f5d4',
          magenta: '#f72585',
          yellow: '#fee440',
          muted: '#a0a0b0',
        },
      },
      boxShadow: {
        glow: '0 0 12px rgba(0, 245, 212, 0.45)',
      },
      keyframes: {
        'status-pulse': {
          '0%, 100%': { transform: 'scale(1)', opacity: '1' },
          '50%': { transform: 'scale(1.2)', opacity: '0.7' },
        },
      },
      animation: {
        'status-pulse': 'status-pulse 2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
