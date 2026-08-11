# tools/model_converter

For now, the model converter *is* [`reference/export.py`](../../reference/export.py) —
per spec §45 ("the exact model format may be changed... External libraries... the
interesting work is the runtime architecture"), there's no value yet in a separate
C++ conversion tool when the Python export path already produces a spec-compliant
`model.bin` (`docs/model_format.md`). Revisit only if a C++-side converter becomes
necessary (e.g. converting directly from safetensors without a Python dependency).
