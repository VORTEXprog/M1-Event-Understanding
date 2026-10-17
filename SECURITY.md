# Security policy

The initial alpha is experimental and has no production security certification.
Use supported Python/PyTorch releases and process only inputs you are authorized
to use. No hosted service, credentials or pretrained downloads are required by
the default examples and tests.

## Reporting a vulnerability

Use the repository's **Security → Report a vulnerability** option when available.
Include the affected version, minimal invented reproduction and impact. Remove
credentials, personal information, private text and model assets from reports.

If private reporting is unavailable, open an issue titled **Private security
contact requested** containing only the affected component and a request for a
private channel. Do not publish exploit instructions or sensitive attachments.
The maintainer can then arrange a private conversation.

Response times are best effort. No response-time guarantee or bug bounty is
offered. Before a fix is released, the maintainer will assess affected versions
and document a mitigation when possible.

## Model and input boundary

Use safetensors state bundles with independently obtained SHA-256 commitments.
Never treat a checksum downloaded alongside an untrusted file as authentication.
Do not load untrusted pickle or executable model code. Feature providers are
external code under the caller's control; review their dependencies, licenses,
network behavior and data handling separately.

