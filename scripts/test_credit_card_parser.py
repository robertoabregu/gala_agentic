from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.credit_card_pdf_parser import parse_credit_card_statement_pdf


def main() -> int:
    if len(sys.argv) != 2:
        print("Uso: python scripts/test_credit_card_parser.py <ruta_al_pdf>")
        return 1

    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    if not pdf_path.exists():
        print(f"No existe el archivo: {pdf_path}")
        return 1

    try:
        parsed = parse_credit_card_statement_pdf(str(pdf_path))
    except Exception as exc:
        print(f"Error parseando el PDF: {exc}")
        return 1

    preview = {
        "summary": parsed.get("summary", {}),
        "metadata": parsed.get("metadata", {}),
        "transactions_preview": (parsed.get("transactions") or [])[:5],
        "taxes_and_fees_preview": (parsed.get("taxes_and_fees") or [])[:5],
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
