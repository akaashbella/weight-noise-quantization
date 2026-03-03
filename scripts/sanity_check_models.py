"""Self-test: build each model, print param counts, run forward check. Exit nonzero on failure."""

from __future__ import annotations

import sys

# Allow importing from project root
sys.path.insert(0, ".")

from src.models import (
    assert_forward_works,
    build_model,
    count_params,
    count_trainable_params,
    list_models,
)


def main() -> int:
    failed = []
    for name in list_models():
        print(f"  {name} ... ", end="", flush=True)
        try:
            model = build_model(name, num_classes=100)
            total = count_params(model)
            trainable = count_trainable_params(model)
            print(f"params={total}, trainable={trainable} ... ", end="", flush=True)
            assert_forward_works(model, device="cpu", batch_size=2, img_size=224)
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}")
            failed.append(name)
    if failed:
        print(f"\nFailed models: {failed}")
        return 1
    print("\nAll models OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
