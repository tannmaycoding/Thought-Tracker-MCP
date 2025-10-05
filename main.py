from mcp.server.fastmcp import FastMCP
import pandas as pd
import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

mcp = FastMCP("totd-mcp")
load_dotenv()
# Update these paths if your CSVs live elsewhere
REPORTS_CSV = os.environ.get(
    "REPORTS_CSV")  # should contain columns like: month, year, total_thoughts, most_frequent, happy_count, sad_count, angry_count, ai_summary ...
THOUGHTS_CSV = os.environ.get("THOUGHTS_CSV")  # should contain columns: date, emotion, thought (tab or comma separated)


# Helper: load CSV safely (with caching in-memory)
def _load_reports() -> pd.DataFrame:
    if not os.path.exists(REPORTS_CSV):
        return pd.DataFrame()
    df = pd.read_csv(REPORTS_CSV)
    # normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    return df


def _load_thoughts() -> pd.DataFrame:
    if not os.path.exists(THOUGHTS_CSV):
        return pd.DataFrame(columns=["date", "emotion", "thought"])
    # try to read common separators; fallback to tab
    try:
        df = pd.read_csv(THOUGHTS_CSV)
    except Exception:
        df = pd.read_csv(THOUGHTS_CSV, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    # normalize date column to datetime if possible
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
        except Exception:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _save_thoughts(df: pd.DataFrame):
    df.to_csv(THOUGHTS_CSV, index=False)


# -------------------- Tools --------------------

@mcp.tool()
def list_available_reports() -> List[Dict]:
    """
    Return list of months/years available in reports.csv
    """
    df = _load_reports()
    if df.empty:
        return []
    rows = df[["month", "year", "total_thoughts", "most_frequent"]].to_dict(orient="records")
    return rows


@mcp.tool()
def get_month_report(month: int, year: int) -> Dict:
    """
    Get the report row for a given month and year.
    Args:
        month: int (1-12)
        year: int e.g., 2025
    Returns a dict with all columns, or an error message if not found.
    """
    df = _load_reports()
    if df.empty:
        return {"error": "reports.csv not found or empty"}
    # ensure numeric month/year
    try:
        month = int(month)
        year = int(year)
    except:
        return {"error": "month and year must be integers"}
    match = df[(df["month"] == month) & (df["year"] == year)]
    if match.empty:
        return {"error": f"No report for month={month}, year={year}"}
    # return the first matching row as dict
    return match.iloc[0].to_dict()


@mcp.tool()
def get_thoughts_by_month(month: int, year: int) -> List[Dict]:
    """
    Return list of thoughts that happened in a given month/year.
    """
    df = _load_thoughts()
    if df.empty:
        return []
    month = int(month)
    year = int(year)
    # ensure date column exists
    if "date" not in df.columns:
        return []
    result = df[df["date"].dt.month == month][df["date"].dt.year == year] if not df[
        "date"].isna().all() else pd.DataFrame()
    # fallback: if dates weren't parsed, try to parse strings with month-year matches
    if result.empty:
        # try parsing naive string date contains month/day/year pattern
        df2 = df.copy()
        try:
            df2["date_parsed"] = pd.to_datetime(df2["date"], dayfirst=False, errors="coerce")
            result = df2[(df2["date_parsed"].dt.month == month) & (df2["date_parsed"].dt.year == year)]
        except Exception:
            result = pd.DataFrame()
    if result.empty:
        return []
    # return list of dicts with original date strings if available
    out = []
    for _, r in result.iterrows():
        out.append({
            "date": str(r.get("date")),
            "emotion": str(r.get("emotion")),
            "thought": str(r.get("thought"))
        })
    return out


@mcp.tool()
def emotion_counts(month: Optional[int] = None, year: Optional[int] = None) -> Dict:
    """
    Return counts of emotions. If month/year provided, filter thoughts to that month-year.
    """
    df = _load_thoughts()
    if df.empty:
        return {}
    # if month/year provided, filter
    if month is not None and year is not None and "date" in df.columns:
        try:
            month = int(month)
            year = int(year)
            df = df[df["date"].dt.month == month][df["date"].dt.year == year]
        except Exception:
            pass
    counts = df["emotion"].value_counts(dropna=True).to_dict()
    # make JSON friendly
    return {str(k): int(v) for k, v in counts.items()}


@mcp.tool()
def top_emotions(n: int = 3) -> List[Dict]:
    """
    Return top-n emotions across all thoughts.
    """
    df = _load_thoughts()
    if df.empty:
        return []
    vc = df["emotion"].value_counts().head(n)
    return [{"emotion": e, "count": int(c)} for e, c in vc.items()]


@mcp.tool()
def monthly_trend(metric: str = "happy_count", year: Optional[int] = None) -> list[Any] | dict[str, str] | Any:
    """
    From reports.csv, return metric values per month sorted descending by month.
    metric examples: happy_count, sad_count, angry_count, total_thoug
    If year provided, filter to that year.
    """
    df = _load_reports()
    if df.empty:
        return []
    if year is not None:
        try:
            year = int(year)
            df = df[df["year"] == year]
        except:
            pass
    if metric not in df.columns:
        return {"error": f"Metric '{metric}' not found in reports.csv columns: {list(df.columns)}"}
    out = df[["month", "year", metric]].sort_values(["year", "month"], ascending=[True, True]).to_dict(orient="records")
    return out


@mcp.tool()
def get_ai_summary(month: int, year: int) -> str:
    """
    Return the ai_summary field from reports.csv for the given month/year.
    """
    df = _load_reports()
    if df.empty:
        return "reports.csv not found"
    try:
        month = int(month)
        year = int(year)
    except:
        return "month and year must be integers"
    row = df[(df["month"] == month) & (df["year"] == year)]
    if row.empty:
        return f"No report for {month}/{year}"
    # return ai_summary column if present
    if "ai_summary" in row.columns:
        return str(row.iloc[0]["ai_summary"])
    # try other possible column spellings
    for col in row.columns:
        if col.lower().startswith("ai_sum"):
            return str(row.iloc[0][col])
    return "No ai_summary column found in reports.csv"


@mcp.tool()
def search_thoughts(query: str, max_results: int = 10) -> List[Dict]:
    """
    Simple substring search over thoughts column (case-insensitive).
    """
    df = _load_thoughts()
    if df.empty or "thought" not in df.columns:
        return []
    q = str(query).lower()
    matches = df[df["thought"].astype(str).str.lower().str.contains(q, na=False)].head(int(max_results))
    return [{"date": str(r.get("date")), "emotion": str(r.get("emotion")), "thought": str(r.get("thought"))} for _, r in
            matches.iterrows()]


@mcp.tool()
def add_thought(date_str: str, emotion: str, thought: str) -> Dict:
    """
    Append a new thought to thought.csv.
    date_str: string like '2025-06-01' or '1/6/2025'
    emotion: Happy/Sad/Angry/...
    thought: text
    Returns status dict.
    """
    df = _load_thoughts()
    # create new row
    new = {"date": date_str, "emotion": emotion, "thought": thought}
    df = df.append(new, ignore_index=True)
    try:
        _save_thoughts(df)
        return {"status": "ok", "message": "Thought added"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def overall_stats(year: Optional[int] = None) -> Dict:
    """
    From reports.csv produce some aggregate stats: total_thoughts, avg_happy_per_month, month_with_max_happy
    """
    df = _load_reports()
    if df.empty:
        return {}
    if year:
        try:
            year = int(year)
            df = df[df["year"] == year]
        except:
            pass
    total = int(df["total_thoughts"].sum()) if "total_thoughts" in df.columns else int(df.shape[0])
    stats = {"total_thoughts": total}
    for col in ["happy_count", "sad_count", "angry_count"]:
        if col in df.columns:
            stats[f"total_{col}"] = int(df[col].sum())
            stats[f"avg_{col}_per_month"] = float(df[col].mean())
    # month with max happy (if available)
    if "happy_count" in df.columns:
        i = df["happy_count"].idxmax()
        stats["month_with_max_happy"] = df.loc[i, ["month", "year"]].to_dict()
    return stats


# -------------------- Run server --------------------
if __name__ == "__main__":
    mcp.run()
