# ER diagram

SQLite schema created by `src/db.py` (menu item 7). GitHub renders the diagram below.

```mermaid
erDiagram
    database_sections ||--o{ acts : "classifies"
    legal_forces      ||--o{ acts : "has force"
    regions           ||--o{ acts : "adopted in"
    acts   ||--o{ act_areas  : ""
    areas  ||--o{ act_areas  : ""
    acts   ||--o{ act_forms  : ""
    forms  ||--o{ act_forms  : ""
    acts   ||--o{ act_bodies : ""
    bodies ||--o{ act_bodies : ""

    acts {
        INTEGER act_id PK
        TEXT    law_id UK
        TEXT    title
        TEXT    date_adopted
        TEXT    date_modified
        TEXT    date_published
        INTEGER is_repealed
        INTEGER section_id FK
        INTEGER legal_force_id FK
        INTEGER region_id FK
    }
    database_sections {
        INTEGER section_id PK
        TEXT    name UK
    }
    legal_forces {
        INTEGER legal_force_id PK
        TEXT    name UK
    }
    regions {
        INTEGER region_id PK
        TEXT    name UK
    }
    areas {
        INTEGER area_id PK
        TEXT    name UK
    }
    forms {
        INTEGER form_id PK
        TEXT    name UK
    }
    bodies {
        INTEGER body_id PK
        TEXT    name UK
        TEXT    history_note
    }
    act_areas {
        INTEGER act_id PK, FK
        INTEGER area_id PK, FK
    }
    act_forms {
        INTEGER act_id PK, FK
        INTEGER form_id PK, FK
    }
    act_bodies {
        INTEGER act_id PK, FK
        INTEGER body_id PK, FK
    }
```

## Normalisation decisions

| Flat column                                       | Problem                                                                                                                   | Solution                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `legal_relation_area`                           | several areas in one cell, separated by commas that also occur inside the names                                           | `areas` + link table `act_areas` (many-to-many)                       |
| `act_form`                                      | several forms in one cell, written in a different order (`Шешім, Өзгерту` / `Өзгерту, Шешім`) | `forms` + `act_forms`; the order no longer matters                    |
| `adopting_body`                                 | joint acts of several bodies; history of the body in parentheses                                                          | `bodies` + `act_bodies`; the note is moved to `bodies.history_note` |
| `database_section`, `legal_force`, `region` | the same long string repeated in thousands of rows                                                                        | lookup tables, `acts` keeps integer foreign keys                        |
| `database_section` = `3700`, `3800`         | numeric codes instead of names (19 rows)                                                                                  | stored as NULL (unknown section)                                          |
| `adilet_url`                                    | fully derived from `law_id`                                                                                             | not stored: `https://adilet.zan.kz/rus/docs/` + `law_id`              |
| `is_archive`                                    | constant                                                                                                                  | not stored                                                                |

Row counts for the full dataset are in `reports/full/database_report.json`.
