"""Real Experimental 1 checkpoint + pinned local donor, without Gold inputs."""
import argparse
import json
import os

# Set before CUDA is initialized; no model or data download occurs in this CLI.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest-sha256', required=True, help='Trusted top-level CHECKPOINT.json SHA-256')
    parser.add_argument('--donor', required=True, help='Exact locally downloaded Qwen snapshot directory')
    parser.add_argument('--device', default='cpu', help='Donor device, e.g. cpu or cuda:0; M1 runs on CPU')
    parser.add_argument('--language', required=True, choices=('en', 'ru'))
    parser.add_argument('--text', required=True)
    parser.add_argument('--max-units', type=int, default=128, help='Explicit input budget; overflow fails without trimming')
    args = parser.parse_args()
    import torch
    from m1 import RawText, predict_raw
    from m1.experimental import load_experimental_checkpoint, canonicalize_components
    from m1.providers.qwen35 import Qwen35FeatureProvider
    torch.set_num_threads(2)
    loaded = load_experimental_checkpoint(args.checkpoint, expected_manifest_sha256=args.manifest_sha256)
    rows = (RawText('input-1', args.text, args.language),)
    with Qwen35FeatureProvider(args.donor, device=args.device, max_units=args.max_units) as provider:
        result = predict_raw(rows, models=loaded.models, provider=provider)
    print(json.dumps({'checkpoint': 'M1 Experimental 1', 'experimental': True,
        'qualification_claimed': False, 'features': result.features.provider_metadata,
        'factors': canonicalize_components(result, loaded.canonicalizer)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
