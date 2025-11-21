# Verifier: Validate `hash_report.txt` and Coordinator Global Hashes
 
It proves two things:

1. **Authenticity** – `hash_report.txt` was signed by the coordinator’s private key.
2. **Integrity** – the *contents* of the report (global hashes for dataset, model, initial weights, and config) match what is recomputed independently from the same code and data.

If both checks pass, you can be confident that:

- The report came from the coordinator.
- The coordinator really trained on the dataset, architecture, initial weights, and training configuration you expect.

---

## What the verifier checks

Given:

- `data/hash_report.txt`
- `data/hash_report.txt.sig`
- `keys/coordinator_public.pem`
- The same codebase that the coordinator used (`data_preprocess`, `models`, etc.),

`verifier.py` does the following:

1. **Signature verification**

   - Reads `hash_report.txt` and its signature file `hash_report.txt.sig`.
   - Extracts `signature_hex: ...` from the `.sig` file.
   - Verifies the signature with `coordinator_public.pem` using RSA PKCS#1 v1.5 + SHA256.
   - If valid, you know the report was produced by whoever holds the coordinator private key.

2. **Global hash recomputation**

   Using the same logic as in `coordinator.py`, it recomputes:

   - `H_dataset` – hash of the full training dataset  
     `{"X_train": X_train.tolist(), "y_train": y_train.tolist()}`
   - `H_arch` – hash of the JSON model string (`MODEL_JSON` or default JSON)
   - `H_weights_init` – hash of initial model weights (from `INITIAL_WEIGHTS` JSON if used, otherwise the model’s default `state_dict()`)
   - `H_config` – hash of training configuration:
     ```python
     {
       "lr": float(LR),
       "epochs": int(EPOCHS),
       "seed": int(SEED),
       "num_workers": int(NUM_WORKERS)
     }
     ```

   Hashing uses the same `_stable_json_hash` helper as the coordinator:

   - `json.dumps(..., sort_keys=True, separators=(",", ":"))`
   - Then `sha256` of the resulting UTF-8 bytes, hex-encoded.

3. **Comparison with the report**

   From the report body, it parses:

   ```text
   Global hashes:
     H_dataset: <hex>
     H_arch: <hex>
     H_weights_init: <hex>
     ...
     H_config: <hex>
