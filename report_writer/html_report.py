from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


@dataclass
class _Entry:
    kind: str
    title: str
    payload: str


class HTMLReport:
    """Simple, notebook-friendly HTML report builder for EDA logging.

    This class is designed to be a drop-in replacement for many `print(...)`
    statements during exploration work.
    """

    def __init__(self, title: str = "Exploration Report") -> None:
        self.title = title
        self._entries: list[_Entry] = []

    def log(self, message: Any, *, title: str = "Log") -> None:
        """Record a text value (or any object cast to text)."""
        self._entries.append(
            _Entry(kind="log", title=title, payload=f"<pre>{escape(str(message))}</pre>")
        )

    def table(
        self,
        data: Any,
        *,
        title: str = "Table",
        max_rows: int = 20,
    ) -> None:
        """Record tabular output from DataFrame/Series/list/dict/iterables."""
        df = self._to_dataframe(data)
        if len(df) > max_rows:
            display_df = df.head(max_rows)
            note = (
                f"<p class='meta'>Showing first {max_rows} rows of {len(df)} total rows.</p>"
            )
        else:
            display_df = df
            note = ""

        html_table = display_df.to_html(index=True, border=0, classes="report-table")
        self._entries.append(
            _Entry(kind="table", title=title, payload=note + html_table)
        )

    def chart(
        self,
        image_path: str,
        *,
        title: str = "Chart",
        caption: Optional[str] = None,
    ) -> None:
        """Record a chart image by file path (e.g., saved matplotlib figure)."""
        path = Path(image_path)
        caption_html = f"<p class='meta'>{escape(caption)}</p>" if caption else ""
        self._entries.append(
            _Entry(
                kind="chart",
                title=title,
                payload=(
                    f"<figure><img src='{escape(path.as_posix())}' alt='{escape(title)}'/>"
                    f"{caption_html}</figure>"
                ),
            )
        )

    def write(self, output_path: str = "report.html") -> str:
        """Render all captured entries into a single formatted HTML file."""
        html = self._render_html()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return str(out)

    def __call__(
        self,
        value: Any,
        *,
        kind: str = "log",
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Drop-in style call helper.

        Examples:
            report("phi = 0.93")
            report(df, kind="table", title="Input data")
            report("charts/scatter.png", kind="chart", title="bd vs br")
        """
        chosen_title = title or kind.capitalize()
        if kind == "log":
            self.log(value, title=chosen_title)
        elif kind == "table":
            self.table(value, title=chosen_title, **kwargs)
        elif kind == "chart":
            self.chart(str(value), title=chosen_title, **kwargs)
        else:
            raise ValueError("kind must be one of: 'log', 'table', 'chart'")

    def _render_html(self) -> str:
        created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        sections = "\n".join(
            (
                "<section class='entry entry-{}'>"
                "<h2>{}</h2>"
                "{}"
                "</section>"
            ).format(e.kind, escape(e.title), e.payload)
            for e in self._entries
        )
        return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{escape(self.title)}</title>
  <style>
    body {{
      margin: 0 auto;
      max-width: 1000px;
      padding: 24px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.45;
      color: #1f2937;
      background: #f7fafc;
    }}
    header {{ margin-bottom: 24px; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #6b7280; font-size: 0.95rem; }}
    .entry {{
      margin-bottom: 18px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.9rem;
    }}
    table.report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    table.report-table th,
    table.report-table td {{
      border: 1px solid #e5e7eb;
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    table.report-table th {{ background: #f3f4f6; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d1d5db; border-radius: 8px; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(self.title)}</h1>
    <p class=\"meta\">Generated: {created}</p>
  </header>
  {sections}
</body>
</html>
"""

    @staticmethod
    def _to_dataframe(data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, pd.Series):
            return data.to_frame()
        if isinstance(data, dict):
            return pd.DataFrame([data])
        if isinstance(data, (list, tuple)):
            if len(data) == 0:
                return pd.DataFrame()
            if isinstance(data[0], dict):
                return pd.DataFrame(data)
            return pd.DataFrame(data)

        if hasattr(data, "shape") and hasattr(data, "__array__"):
            return pd.DataFrame(data)

        if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            return pd.DataFrame(list(data))

        return pd.DataFrame({"value": [data]})


def report(
    value: Any,
    writer: HTMLReport,
    *,
    kind: str = "log",
    title: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Drop-in helper function for replacing print calls.

    Example:
        report(phi, my_report, title="phi")
        report(mydata.dtypes.index, my_report, kind="table", title="dtypes")
    """
    writer(value, kind=kind, title=title, **kwargs)
