# Security

Do not commit Hugging Face or GitHub access tokens.

Fleet downloads third-party adapter repositories. Production deployments should:

- pin trusted adapter revisions,
- review source licenses,
- maintain an allowlist of repositories,
- verify hashes or revisions before promotion,
- avoid automatically promoting newly discovered adapters to production,
- run the compiler in a restricted build environment.

The current expert config is a research discovery list, not a supply-chain trust policy.
