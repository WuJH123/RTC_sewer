"""Stream-merge the frozen control-core manifest with development candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _write(path: Path, writer: pq.ParquetWriter, schema: pa.Schema) -> int:
    rows = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
        table = pa.Table.from_batches([batch])
        columns = []
        for field in schema:
            if field.name in table.column_names:
                column = table[field.name]
                if column.type != field.type:
                    column = column.cast(field.type)
            else:
                column = pa.chunked_array([pa.nulls(table.num_rows, type=field.type)])
            columns.append(column)
        writer.write_table(pa.Table.from_arrays(columns, schema=schema))
        rows += table.num_rows
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schema = pa.unify_schemas(
        [pq.ParquetFile(args.base).schema_arrow, pq.ParquetFile(args.expanded).schema_arrow]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.output, schema)
    try:
        base_rows = _write(args.base, writer, schema)
        expanded_rows = _write(args.expanded, writer, schema)
    finally:
        writer.close()
    print({"base_rows": base_rows, "expanded_rows": expanded_rows, "total_rows": base_rows + expanded_rows, "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
