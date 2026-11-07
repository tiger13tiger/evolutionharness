"""SQL guardrails applied BEFORE any agent-generated SQL touches the database."""
import re
from dataclasses import dataclass

ALLOWED_TABLES = {"dim_venue", "dim_game", "fact_play", "fact_refund", "venue_notes", "raw_play_feed"}
BLOCKED_TABLES = {"customer_pii"}                      # PII: never readable by the agent
FORBIDDEN = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|GRANT|REVOKE|ATTACH|PRAGMA|EXEC)\b"
PII_PATTERNS = [r"[\w.+-]+@[\w-]+\.[\w.]+", r"\b04\d{2}\s?\d{3}\s?\d{3}\b"]   # email, AU mobile


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)


def _strip_literals(sql: str) -> str:
    """Blank out quoted strings so a ';' or 'DELETE' inside a value (e.g. a game name) isn't read as SQL."""
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def check_sql(sql: str) -> GuardResult:
    if not sql or not sql.strip():
        return GuardResult(False, "empty_sql")
    s = _strip_literals(_strip_comments(sql)).strip().rstrip(";")   # text inside quotes is data, not SQL
    if ";" in s:
        return GuardResult(False, "multiple_statements")
    if not re.match(r"^\s*(SELECT|WITH)\b", s, re.I):
        return GuardResult(False, "not_read_only")
    if re.search(FORBIDDEN, s, re.I):
        return GuardResult(False, "forbidden_keyword")
    # CTE names are fine to reference; everything else must be allow-listed
    ctes = {m.lower() for m in re.findall(r"\b(\w+)\s+AS\s*\(", s, re.I)}
    tables = {t.lower() for t in re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w.]*)", s, re.I)}
    if tables & BLOCKED_TABLES:
        return GuardResult(False, "blocked_table_pii")
    unknown = tables - ALLOWED_TABLES - ctes
    if unknown:
        return GuardResult(False, f"table_not_allowed:{','.join(sorted(unknown))}")
    return GuardResult(True)


def contains_pii(text: str) -> bool:
    return any(re.search(p, text or "") for p in PII_PATTERNS)
