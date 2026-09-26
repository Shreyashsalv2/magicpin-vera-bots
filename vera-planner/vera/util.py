"""Small, dependency-free helpers shared by every module."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utcnow().strftime("%Y-%m-%dT%H:%M:%S.") + f"{utcnow().microsecond // 1000:03d}Z"


def parse_dt(value: Any) -> Optional[datetime]:
    """Parse ISO dates/datetimes (with or without zone). Returns aware UTC datetimes."""
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    try:
        if len(s) == 10:
            d = date.fromisoformat(s)
            return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def days_between(a: Optional[datetime], b: Optional[datetime]) -> Optional[int]:
    if not a or not b:
        return None
    return int((b - a).total_seconds() // 86400)


def fmt_day(dt: Optional[datetime]) -> str:
    """'28 Apr' style date."""
    if not dt:
        return ""
    return f"{dt.day} {MONTHS[dt.month - 1]}"


def fmt_day_full(dt: Optional[datetime]) -> str:
    """'Sat, 2 May' style date."""
    if not dt:
        return ""
    return f"{WEEKDAYS[dt.weekday()]} {dt.day} {MONTHS[dt.month - 1]}"


def fmt_int(n: Any) -> str:
    """Indian-friendly integer formatting with thousands separators (2,410)."""
    try:
        v = int(round(float(n)))
    except (TypeError, ValueError):
        return str(n)
    return f"{v:,}"


def fmt_money(n: Any) -> str:
    return "₹" + fmt_int(n)


def pct(x: Any, digits: int = 0) -> str:
    """0.382 -> '38%'; -0.5 -> '50%' (sign handled by caller)."""
    try:
        v = abs(float(x)) * 100
    except (TypeError, ValueError):
        return str(x)
    if digits == 0:
        return f"{int(round(v))}%"
    return f"{v:.{digits}f}%"


def ctr_pct(x: Any) -> str:
    """0.021 -> '2.1%'."""
    try:
        v = float(x) * 100
    except (TypeError, ValueError):
        return str(x)
    s = f"{v:.1f}"
    return (s[:-2] if s.endswith(".0") else s) + "%"


def humanize(token: Any) -> str:
    """'6_month_cleaning' -> '6-month cleaning'; 'delivery_late' -> 'delivery late'."""
    if token is None:
        return ""
    s = str(token)
    s = re.sub(r"(\d)_(month|week|day|year|mo|min)", r"\1-\2", s)
    s = s.replace("_", " ").strip()
    return re.sub(r"\s+", " ", s)


def first_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"\(.*?\)", "", name).strip()
    parts = name.split()
    if not parts:
        return ""
    if parts[0].rstrip(".").lower() in {"dr", "mr", "mrs", "ms"} and len(parts) > 1:
        return parts[0] + " " + parts[1]
    return parts[0]


def stable_hash(obj: Any, n: int = 16) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:n]


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d+))?")


def extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in NUM_RE.finditer(text or ""):
        whole = m.group(1).replace(",", "")
        frac = m.group(2)
        try:
            out.append(float(f"{whole}.{frac}" if frac else whole))
        except ValueError:
            continue
    return out


def walk_numbers(obj: Any, sink: set[float]) -> None:
    """Collect every number that appears anywhere inside a JSON-like structure,
    including numbers embedded in strings (dates, prices, batch ids)."""
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        sink.add(float(obj))
        return
    if isinstance(obj, str):
        for n in extract_numbers(obj):
            sink.add(n)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_numbers(k, sink)
            walk_numbers(v, sink)
        return
    if isinstance(obj, Iterable):
        for v in obj:
            walk_numbers(v, sink)


def join_nonempty(parts: Iterable[Optional[str]], sep: str = " ") -> str:
    return sep.join(p.strip() for p in parts if p and p.strip())


def clean_body(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r" +([,.?!])", r"\1", text)
    text = re.sub(r"\.\.+", ".", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
