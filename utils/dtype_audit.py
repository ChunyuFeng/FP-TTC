import json
import threading
from collections import Counter
from typing import Any, Dict, Optional


_LOCK = threading.Lock()
_ENABLED = False
_EVENTS = []


def _stringify_dtype(dtype: Any) -> Optional[str]:
    if dtype is None:
        return None
    return str(dtype).replace("torch.", "")


def _stringify_shape(shape: Any):
    if shape is None:
        return None
    if hasattr(shape, "tolist"):
        try:
            return tuple(shape.tolist())
        except TypeError:
            pass
    return tuple(shape)


def set_dtype_audit_enabled(enabled: bool):
    global _ENABLED
    with _LOCK:
        _ENABLED = bool(enabled)


def reset_dtype_audit():
    global _EVENTS
    with _LOCK:
        _EVENTS = []


def dtype_audit_enabled() -> bool:
    with _LOCK:
        return _ENABLED


def log_dtype_event(
    name: str,
    *,
    src_dtype: Any = None,
    dst_dtype: Any = None,
    shape: Any = None,
    device: Any = None,
    copied: Optional[bool] = None,
    note: Optional[str] = None,
):
    if not dtype_audit_enabled():
        return
    event = {
        "name": name,
        "src_dtype": _stringify_dtype(src_dtype),
        "dst_dtype": _stringify_dtype(dst_dtype),
        "shape": _stringify_shape(shape),
        "device": str(device) if device is not None else None,
        "copied": copied,
        "note": note,
    }
    with _LOCK:
        _EVENTS.append(event)


def build_dtype_audit_payload() -> Dict[str, Any]:
    with _LOCK:
        events = list(_EVENTS)

    summary = Counter()
    for event in events:
        key = (
            event["name"],
            event["src_dtype"],
            event["dst_dtype"],
            event["device"],
            event["copied"],
        )
        summary[key] += 1

    summary_rows = [
        {
            "name": name,
            "src_dtype": src_dtype,
            "dst_dtype": dst_dtype,
            "device": device,
            "copied": copied,
            "count": count,
        }
        for (name, src_dtype, dst_dtype, device, copied), count in sorted(summary.items())
    ]
    return {
        "event_count": len(events),
        "summary": summary_rows,
        "events": events,
    }


def dump_dtype_audit(path: str):
    payload = build_dtype_audit_payload()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
