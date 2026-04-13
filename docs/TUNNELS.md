# HTTPS and tunnels (mobile camera)

Many mobile browsers only expose the camera on **https://** or **http://localhost**. When you open the site via your PC’s LAN URL (e.g. `http://192.168.x.x:PORT`) from your phone, the camera may be blocked as an insecure context.

## Options

1. **HTTPS on LAN** — Set `HTTPS_CERT` and `HTTPS_KEY` in `.env` to PEM files (e.g. from [mkcert](https://github.com/FiloSottile/mkcert)), then restart `server-local.py`.

2. **ngrok** — [ngrok.com](https://ngrok.com/) — `ngrok http 8000` (use the printed https URL on your phone).

3. **Cloudflare Tunnel** — [Cloudflare Tunnel docs](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/) — exposes your local server behind HTTPS.

4. **no-ip / dynamic DNS** — [noip.com](https://www.noip.com/) — if you need a stable hostname to your network (often combined with reverse proxy + TLS).

After tunneling, open the **https** try-on URL from `GET /api/server-info` or the pipeline dashboard.
