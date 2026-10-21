"""The single instruction used for zero-shot evaluation, fine-tuning and serving.

Keeping one prompt for all three makes the zero-shot vs fine-tuned comparison fair.
"""

SYSTEM_PROMPT = "You are a precise receipt-reading assistant. You only report what is printed on the page."

USER_PROMPT = """Extract the receipt on this page as JSON with exactly these keys:
{"store_name": string|null, "date": "YYYY-MM-DD"|null,
 "line_items": [{"name": string, "qty": number|null, "unit_price": number|null, "amount": number|null}],
 "subtotal": number|null, "tax": number|null, "total": number|null}
Rules:
- Copy item names exactly as printed.
- Numbers are plain JSON numbers without currency or thousand separators (e.g. "Rp 45.000" -> 45000).
- "amount" is the line total printed for the item; "unit_price" only if a per-unit price is printed.
- If a value is not printed or not readable, use null. Never guess or compute a missing value.
Return only the JSON object."""
