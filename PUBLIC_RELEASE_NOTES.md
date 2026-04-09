# Public Release Notes

Before uploading this directory to a public GitHub repository, check the following:

1. Add a real `LICENSE` file.
   This directory does not include one yet because license choice should be explicit.

2. Confirm redistribution rights for `esco_cmo_binding.ttl`.
   If the ontology bundle is derived from ESCO, ROME, or another controlled resource, verify that the merged file can be redistributed publicly under your intended repository license.

3. Decide whether to keep `sample_results/`.
   It is useful for readers and reviewers, but you may prefer to reduce repository size or publish only summary CSVs.

4. Decide whether to add notebooks separately.
   They were intentionally excluded here because the public package is centered on the reusable LOSD pipeline rather than exploratory local work.

5. Review model-specific references.
   The public package excludes private caches and local batch outputs. If you later add cached generations, verify that they do not include secrets, provider-specific tokens, or unpublished annotations.
