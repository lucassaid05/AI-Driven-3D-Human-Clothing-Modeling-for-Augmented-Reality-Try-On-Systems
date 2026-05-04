# HTTPS and tunnels (mobile camera)

Many mobile browsers only expose the camera on **https://** or **http://localhost**. When you open the site via your PC’s LAN URL (e.g. `http://192.168.x.x:PORT`) from your phone, the camera may be blocked as an insecure context.

## ngrok: why the site can look “stale”

**ngrok does not cache your code.** It only forwards to a **local port** you choose. Stale UI usually means the **server on that port** is not what you expect:

| Local server | Port | What you get | To see latest try-on code |
|--------------|------|--------------|----------------------------|
| `python server-local.py` | **8000** | Serves **pre-built** files in **`dist/`** (e.g. `/dist/apps/ar-tryon/`) | Run **`npm run build`**, then reload. |
| `npm run dev` (Vite) | **5173** | **Live** `src/` + `apps/` (dev server) | Save files; refresh if HMR does not update. |

So **`ngrok http 8000`** points at the **static** Python app. Until you rebuild, the tunnel will show the **old** `dist` bundle (not an ngrok bug).

**Live code through ngrok (no rebuild each time):**

1. Start API: `python server-local.py` (keeps **:8000** for `/api`).
2. Start Vite: `npm run dev` (**:5173**, HTTPS in this project).
3. Tunnel to Vite (HTTPS backend):

   `ngrok http https://127.0.0.1:5173`

4. Open the ngrok **https** URL and go to **`/apps/ar-tryon/`**. Vite proxies `/api` and `/generated` to `127.0.0.1:8000`.

If it still looks old, hard-refresh the phone or use a private window (browser cache).

## Other options

1. **HTTPS on LAN** — Set `HTTPS_CERT` and `HTTPS_KEY` in `.env` to PEM files (e.g. from [mkcert](https://github.com/FiloSottile/mkcert)), then restart `server-local.py`.

2. **ngrok to Python only** — `ngrok http 8000` — use the printed **https** URL; run **`npm run build`** when you change the try-on app.

3. **Cloudflare Tunnel** — [Cloudflare Tunnel docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) — exposes your local server behind HTTPS.

4. **no-ip / dynamic DNS** — [noip.com](https://www.noip.com/) — if you need a stable hostname to your network (often combined with reverse proxy + TLS).

After tunneling, confirm URLs with **`GET /api/server-info`** (while `server-local.py` is running).
