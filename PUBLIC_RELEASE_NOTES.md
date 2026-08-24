# Public Release Notes

Before uploading this directory to a public GitHub repository, check the following:

1. Add a real `LICENSE` file.
   This directory does not include one yet because license choice should be explicit.

2. Confirm redistribution rights for `esco_cmo_binding.ttl`.
   If the ontology bundle is derived from ESCO, ROME, or another controlled resource, verify that the merged file can be redistributed publicly under your intended repository license.

3. Decide whether to keep `sample_results/`.
   It is useful for readers and reviewers, but you may prefer to reduce repository size or publish only summary CSVs.

4. The repeated-run and repeated-analysis scripts are included, but the paid
   model-response caches and embedding cache are not. Reproducing the full
   study therefore requires fresh provider access and may incur charges.

5. Review model-specific references and endpoint availability.
   The wrappers record the model/provider configurations used in the study,
   but providers may later retire or reroute those endpoints.

6. Run a secret scan before publishing. Never commit `OPENROUTER_API_KEY`,
   `.env` files, terminal logs containing authorization headers, or provider
   account identifiers.

7. If cached generations are published later, preserve their prompt/configuration
   hashes and verify that they contain no secrets, unpublished annotations, or
   material that cannot be redistributed.
