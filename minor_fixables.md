# Minor fixables (deferred)

Low-priority cleanups intentionally deferred, not bugs. Tracked here so they
aren't lost.

- **Duplicated reason-serialisation line** — `distance_filter_reason=json.dumps(reason) if reason else ""`
  is repeated in `vacancy_generator/io.py:215` (`build_seed_record`) and
  `vacancy_generator/io.py:283` (`build_migration_seed_record`). Extract to a
  small `_reason_text(reason: Optional[dict]) -> str` helper.

- **Stale `List[dict]` type hints on record consumers** — these annotate their
  `records` params as `List[dict]`/`List[object]` but now receive
  `List[MetadataRecord]` (dicts are still accepted for SOAP rows, so a
  `Union`/`Sequence` hint would be most accurate):
  - `vacancy_generator/reporting.py:17` — `describe_output_summary(records: List[dict])`
  - `vacancy_generator/reporting.py:35` — `build_dataset_report(records: List[dict], ...)`
  - `vacancy_generator/reporting.py:78` — return `Tuple[List[dict], dict]`
  - `vacancy_generator/io.py:125` — `save_metadata_csv(records: List[object], ...)`

  (The two hints directly touched by the `ComboContext` change —
  `_write_structures` and `_write_migration_paths` in `main.py`, previously
  `-> List[dict]` — were fixed inline to `-> List[MetadataRecord]`.)
