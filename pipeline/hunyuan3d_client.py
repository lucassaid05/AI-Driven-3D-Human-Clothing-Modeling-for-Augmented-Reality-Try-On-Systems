"""
Tencent Hunyuan 3D (international endpoint) — Submit + Query for Pro and Rapid jobs.

Docs: https://www.tencentcloud.com/document/product/1284/75287
API host: hunyuan.intl.tencentcloudapi.com — regions ap-singapore, na-siliconvalley.

Requires: pip install tencentcloud-sdk-python-intl-en
"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

Edition = Literal["pro", "rapid"]


def _strip_data_uri(b64_or_uri: str) -> str:
    s = (b64_or_uri or "").strip()
    if "," in s and s.lower().startswith("data:"):
        return s.split(",", 1)[1].strip()
    return s


def _resolve_credentials(
    secret_id: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> tuple[str, str]:
    sid = (secret_id or os.environ.get("TENCENTCLOUD_SECRET_ID") or os.environ.get("HUNYUAN_SECRET_ID") or "").strip()
    sk = (secret_key or os.environ.get("TENCENTCLOUD_SECRET_KEY") or os.environ.get("HUNYUAN_SECRET_KEY") or "").strip()
    if not sid or not sk:
        raise RuntimeError(
            "Missing Tencent credentials. Set TENCENTCLOUD_SECRET_ID and TENCENTCLOUD_SECRET_KEY "
            "(or HUNYUAN_SECRET_ID / HUNYUAN_SECRET_KEY) in .env — see docs/HUNYUAN_EVAL.md."
        )
    return sid, sk


def credentials_configured(
    secret_id: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> bool:
    try:
        _resolve_credentials(secret_id, secret_key)
        return True
    except RuntimeError:
        return False


def _client(region: str, secret_id: Optional[str], secret_key: Optional[str]):
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.hunyuan.v20230901 import hunyuan_client

    sid, sk = _resolve_credentials(secret_id, secret_key)
    cred = credential.Credential(sid, sk)
    hp = HttpProfile()
    hp.endpoint = "hunyuan.intl.tencentcloudapi.com"
    cp = ClientProfile()
    cp.httpProfile = hp
    return hunyuan_client.HunyuanClient(cred, region, cp)


def _serialize_files(resp) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    files = getattr(resp, "ResultFile3Ds", None) or []
    for f in files:
        out.append(
            {
                "Type": getattr(f, "Type", None),
                "Url": getattr(f, "Url", None),
                "PreviewImageUrl": getattr(f, "PreviewImageUrl", None),
            }
        )
    return out


def submit_job(
    *,
    region: str,
    edition: Edition = "pro",
    secret_id: Optional[str] = None,
    secret_key: Optional[str] = None,
    prompt: Optional[str] = None,
    image_base64: Optional[str] = None,
    image_url: Optional[str] = None,
    model: Optional[str] = None,
    enable_pbr: Optional[bool] = None,
    face_count: Optional[int] = None,
    generate_type: Optional[str] = None,
    polygon_type: Optional[str] = None,
    enable_geometry: Optional[bool] = None,
    result_format: Optional[str] = None,
) -> str:
    """Submit async job; returns JobId. Exactly one of prompt, image_base64, or image_url (Rapid)."""
    from tencentcloud.hunyuan.v20230901 import models

    has_prompt = bool((prompt or "").strip())
    has_b64 = bool((image_base64 or "").strip())
    has_url = bool((image_url or "").strip())
    n = sum(1 for x in (has_prompt, has_b64, has_url) if x)
    if n != 1:
        raise ValueError("Provide exactly one of: prompt, image_base64, or image_url")

    client = _client(region, secret_id, secret_key)

    if edition == "rapid":
        req = models.SubmitHunyuanTo3DRapidJobRequest()
        if has_prompt:
            req.Prompt = prompt.strip()
        if has_b64:
            req.ImageBase64 = _strip_data_uri(image_base64 or "")
        if has_url:
            req.ImageUrl = (image_url or "").strip()
        if enable_pbr is not None:
            req.EnablePBR = enable_pbr
        if enable_geometry is not None:
            req.EnableGeometry = enable_geometry
        if result_format:
            req.ResultFormat = result_format
        resp = client.SubmitHunyuanTo3DRapidJob(req)
    else:
        req = models.SubmitHunyuanTo3DProJobRequest()
        if has_prompt:
            req.Prompt = prompt.strip()
        if has_b64:
            req.ImageBase64 = _strip_data_uri(image_base64 or "")
        if has_url:
            req.ImageUrl = (image_url or "").strip()
        if model:
            req.Model = model
        if enable_pbr is not None:
            req.EnablePBR = enable_pbr
        if face_count is not None:
            req.FaceCount = face_count
        if generate_type:
            req.GenerateType = generate_type
        if polygon_type:
            req.PolygonType = polygon_type
        resp = client.SubmitHunyuanTo3DProJob(req)

    jid = getattr(resp, "JobId", None) or ""
    if not jid:
        raise RuntimeError(f"No JobId in response: {resp.to_json_string()}")
    return jid


def query_job(
    job_id: str,
    *,
    region: str,
    edition: Edition = "pro",
    secret_id: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> dict[str, Any]:
    from tencentcloud.hunyuan.v20230901 import models

    client = _client(region, secret_id, secret_key)
    if edition == "rapid":
        req = models.QueryHunyuanTo3DRapidJobRequest()
        req.JobId = job_id
        resp = client.QueryHunyuanTo3DRapidJob(req)
    else:
        req = models.QueryHunyuanTo3DProJobRequest()
        req.JobId = job_id
        resp = client.QueryHunyuanTo3DProJob(req)

    return {
        "Status": getattr(resp, "Status", None),
        "ErrorCode": getattr(resp, "ErrorCode", None) or "",
        "ErrorMessage": getattr(resp, "ErrorMessage", None) or "",
        "RequestId": getattr(resp, "RequestId", None) or "",
        "ResultFile3Ds": _serialize_files(resp),
    }
