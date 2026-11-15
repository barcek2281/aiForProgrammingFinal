"""Analytical SQL queries. Every query is a dict: key, title, techniques, sql, params.

The results are loaded into pandas by db.run_query and saved as CSV, so they
can be used in the EDA, in the modelling stage and in the report.
"""
from src.db import run_query
from src.utils import Timer

# Acts adopted before this date go to the training set, the rest to the test set.
SPLIT_DATE = "2023-01-01"

QUERIES = [
    {
        "key": "q01_sections",
        "title": "Acts and share of repealed acts by thematic section",
        "techniques": "JOIN, GROUP BY, aggregate functions",
        "sql": """
            SELECT s.name                               AS section,
                   COUNT(*)                             AS acts,
                   ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM acts), 2) AS share_pct,
                   ROUND(100.0 * AVG(a.is_repealed), 1) AS repealed_pct,
                   MIN(a.date_adopted)                  AS first_act,
                   MAX(a.date_adopted)                  AS last_act
            FROM acts a
            JOIN database_sections s ON s.section_id = a.section_id
            GROUP BY s.name
            ORDER BY acts DESC;
        """,
    },
    {
        "key": "q02_active_bodies",
        "title": "Adopting bodies with at least 1000 acts",
        "techniques": "JOIN through a many-to-many table, GROUP BY, HAVING",
        "sql": """
            SELECT b.name                               AS body,
                   COUNT(*)                             AS acts,
                   COUNT(DISTINCT a.section_id)         AS sections_covered,
                   ROUND(100.0 * AVG(a.is_repealed), 1) AS repealed_pct
            FROM bodies b
            JOIN act_bodies ab ON ab.body_id = b.body_id
            JOIN acts a        ON a.act_id = ab.act_id
            GROUP BY b.body_id
            HAVING COUNT(*) >= :min_acts
            ORDER BY acts DESC;
        """,
        "params": {"min_acts": 1000},
        "sample_params": {"min_acts": 10},
    },
    {
        "key": "q03_yearly_dynamics",
        "title": "Acts per year: change to the previous year and 3-year moving average",
        "techniques": "CTE, window functions LAG and AVG ... ROWS BETWEEN",
        "sql": """
            WITH yearly AS (
                SELECT CAST(strftime('%Y', date_adopted) AS INTEGER) AS year,
                       COUNT(*) AS acts
                FROM acts
                GROUP BY year
            )
            SELECT year,
                   acts,
                   acts - LAG(acts) OVER (ORDER BY year) AS change,
                   ROUND(100.0 * (acts - LAG(acts) OVER (ORDER BY year))
                         / LAG(acts) OVER (ORDER BY year), 1) AS change_pct,
                   ROUND(AVG(acts) OVER (ORDER BY year
                         ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)) AS moving_avg_3y
            FROM yearly
            WHERE year >= 1991
            ORDER BY year;
        """,
    },
    {
        "key": "q04_top_sections_by_period",
        "title": "Top-3 sections in every five-year period",
        "techniques": "two CTEs, RANK() OVER (PARTITION BY), SUM() OVER (PARTITION BY)",
        "sql": """
            WITH counts AS (
                SELECT (CAST(strftime('%Y', a.date_adopted) AS INTEGER) / 5) * 5 AS period_start,
                       s.name   AS section,
                       COUNT(*) AS acts
                FROM acts a
                JOIN database_sections s ON s.section_id = a.section_id
                GROUP BY period_start, s.name
            ),
            ranked AS (
                SELECT period_start, section, acts,
                       RANK() OVER (PARTITION BY period_start ORDER BY acts DESC) AS place,
                       ROUND(100.0 * acts / SUM(acts) OVER (PARTITION BY period_start), 1) AS share_pct
                FROM counts
            )
            SELECT period_start, place, section, acts, share_pct
            FROM ranked
            WHERE place <= 3 AND period_start >= 1995
            ORDER BY period_start, place;
        """,
    },
    {
        "key": "q05_pareto_sections",
        "title": "Class imbalance: cumulative share of acts by section",
        "techniques": "CTE, window functions SUM() OVER (ORDER BY) and ROW_NUMBER()",
        "sql": """
            WITH counts AS (
                SELECT s.name AS section, COUNT(*) AS acts
                FROM acts a
                JOIN database_sections s ON s.section_id = a.section_id
                GROUP BY s.name
            )
            SELECT ROW_NUMBER() OVER (ORDER BY acts DESC) AS place,
                   section,
                   acts,
                   ROUND(100.0 * SUM(acts) OVER (ORDER BY acts DESC, section)
                         / SUM(acts) OVER (), 1) AS cumulative_pct
            FROM counts
            ORDER BY place;
        """,
    },
    {
        "key": "q06_areas",
        "title": "Most frequent areas of legal regulation",
        "techniques": "many-to-many JOIN, GROUP BY, correlated subquery",
        "sql": """
            SELECT ar.name   AS area,
                   COUNT(*)  AS acts,
                   ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM acts), 2) AS pct_of_all_acts,
                   (SELECT COUNT(*)
                    FROM act_areas only_one
                    WHERE only_one.area_id = ar.area_id
                      AND (SELECT COUNT(*) FROM act_areas x
                           WHERE x.act_id = only_one.act_id) = 1)       AS acts_with_this_area_only
            FROM areas ar
            JOIN act_areas aa ON aa.area_id = ar.area_id
            GROUP BY ar.area_id
            ORDER BY acts DESC
            LIMIT 20;
        """,
    },
    {
        "key": "q07_area_pairs",
        "title": "Areas that most often appear together in one act",
        "techniques": "self-JOIN of the many-to-many table",
        "sql": """
            SELECT a1.name  AS area_1,
                   a2.name  AS area_2,
                   COUNT(*) AS acts_together
            FROM act_areas x
            JOIN act_areas y ON y.act_id = x.act_id AND y.area_id > x.area_id
            JOIN areas a1    ON a1.area_id = x.area_id
            JOIN areas a2    ON a2.area_id = y.area_id
            GROUP BY x.area_id, y.area_id
            ORDER BY acts_together DESC
            LIMIT 15;
        """,
    },
    {
        "key": "q08_conflicting_titles",
        "title": "Identical titles that have different sections (label noise)",
        "techniques": "GROUP BY, HAVING COUNT(DISTINCT), GROUP_CONCAT",
        "sql": """
            SELECT a.title,
                   COUNT(*)                      AS acts,
                   COUNT(DISTINCT a.section_id)  AS different_sections,
                   GROUP_CONCAT(DISTINCT s.name) AS sections
            FROM acts a
            JOIN database_sections s ON s.section_id = a.section_id
            GROUP BY a.title
            HAVING COUNT(DISTINCT a.section_id) > 1
            ORDER BY acts DESC
            LIMIT 15;
        """,
    },
    {
        "key": "q09_main_form_by_section",
        "title": "Most typical act form in every section",
        "techniques": "CTE, ROW_NUMBER() OVER (PARTITION BY), many-to-many JOIN",
        "sql": """
            WITH form_counts AS (
                SELECT s.name   AS section,
                       f.name   AS form,
                       COUNT(*) AS acts
                FROM acts a
                JOIN database_sections s ON s.section_id = a.section_id
                JOIN act_forms af        ON af.act_id = a.act_id
                JOIN forms f             ON f.form_id = af.form_id
                GROUP BY s.name, f.name
            ),
            ranked AS (
                SELECT section, form, acts,
                       ROW_NUMBER() OVER (PARTITION BY section ORDER BY acts DESC) AS place,
                       ROUND(100.0 * acts / SUM(acts) OVER (PARTITION BY section), 1) AS share_pct
                FROM form_counts
            )
            SELECT section, form AS main_form, acts, share_pct
            FROM ranked
            WHERE place = 1
            ORDER BY share_pct DESC;
        """,
    },
    {
        "key": "q10_time_to_modification",
        "title": "Years between adoption and last modification, by legal force",
        "techniques": "JOIN, date arithmetic with julianday, GROUP BY, HAVING",
        "sql": """
            SELECT lf.name  AS legal_force,
                   COUNT(*) AS modified_acts,
                   ROUND(AVG(julianday(a.date_modified) - julianday(a.date_adopted)) / 365.25, 1)
                       AS avg_years_to_modification,
                   ROUND(MAX(julianday(a.date_modified) - julianday(a.date_adopted)) / 365.25, 1)
                       AS max_years
            FROM acts a
            JOIN legal_forces lf ON lf.legal_force_id = a.legal_force_id
            WHERE a.date_modified IS NOT NULL
              AND a.date_modified >= a.date_adopted
            GROUP BY lf.name
            HAVING COUNT(*) >= :min_acts
            ORDER BY avg_years_to_modification DESC;
        """,
        "params": {"min_acts": 100},
        "sample_params": {"min_acts": 5},
    },
    {
        "key": "q11_split_by_section",
        "title": f"Time-based split (test = adopted from {SPLIT_DATE}): acts per section",
        "techniques": "CTE, CASE, conditional aggregation",
        "sql": """
            WITH labelled AS (
                SELECT section_id,
                       CASE WHEN date_adopted < :split_date THEN 'train' ELSE 'test' END AS part
                FROM acts
                WHERE section_id IS NOT NULL
            )
            SELECT s.name                        AS section,
                   SUM(l.part = 'train')         AS train_acts,
                   SUM(l.part = 'test')          AS test_acts,
                   ROUND(100.0 * SUM(l.part = 'test') / COUNT(*), 1) AS test_share_pct
            FROM labelled l
            JOIN database_sections s ON s.section_id = l.section_id
            GROUP BY s.name
            ORDER BY train_acts DESC;
        """,
        "params": {"split_date": SPLIT_DATE},
    },
    {
        "key": "q12_split_leakage",
        "title": "Leakage check: test titles that already exist in the training period",
        "techniques": "subquery with IN, CTE",
        "sql": """
            WITH test_acts AS (
                SELECT title,
                       title IN (SELECT title FROM acts WHERE date_adopted < :split_date) AS seen_in_train
                FROM acts
                WHERE date_adopted >= :split_date
            )
            SELECT COUNT(*)                              AS test_acts,
                   SUM(seen_in_train)                    AS titles_seen_in_train,
                   ROUND(100.0 * AVG(seen_in_train), 1)  AS seen_pct,
                   COUNT(*) - SUM(seen_in_train)         AS test_acts_after_removal
            FROM test_acts;
        """,
        "params": {"split_date": SPLIT_DATE},
    },
]


def run_one(db_path, query: dict, use_sample: bool = False) -> tuple:
    """Run one query; returns (DataFrame, seconds)."""
    params = query.get("params", {})
    if use_sample:      # the 500-row sample needs smaller thresholds
        params = query.get("sample_params", params)
    with Timer() as timer:
        result = run_query(db_path, query["sql"], params)
    return result, timer.seconds


def find_query(key_or_number: str) -> dict:
    """Find a query by its number ('3') or key ('q03_yearly_dynamics')."""
    for number, query in enumerate(QUERIES, start=1):
        if key_or_number == str(number) or key_or_number == query["key"]:
            return query
    raise KeyError(f"no SQL query '{key_or_number}'")
