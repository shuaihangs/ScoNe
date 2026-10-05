"""Record input identity and settings for reproducible runs."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record_run(args, output_dir):
    from . import config
    settings = {name: getattr(args, name) for name in (
        'seed', 'epochs', 'cv_folds', 'k_neighbours', 'soft_loss_divisor',
        'contrastive_temperature', 'positiveness_temperature',
    )}
    settings['data_sha256'] = sha256(args.csv_path)
    settings['training'] = {name: getattr(config, name) for name in (
        'BATCH_SIZE', 'MAX_LENGTH', 'PROJ_DIM', 'DROPOUT', 'LR', 'WEIGHT_DECAY',
        'FEATURE_CACHE_BATCH_SIZE', 'NEIGHBOUR_LLM_BATCH_SIZE', 'VALIDATION_RATIO',
        'NORMALIZE_PROJECTED_STATES', 'USE_FEATURE_STANDARDIZATION',
    )}
    path = output_dir / 'run_manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['settings'] != settings:
            raise ValueError('Run settings or input data changed; use a new output directory.')
        return
    manifest = {'settings': settings, 'arguments': vars(args), 'python': platform.python_version()}
    manifest['packages'] = {name: importlib.metadata.version(name) for name in (
        'torch', 'transformers', 'numpy', 'pandas', 'scikit-learn',
    )}
    root = Path(__file__).resolve().parents[1]
    manifest['source_sha256'] = {str(p.relative_to(root)): sha256(p)
                               for p in [root / 'run_model.py', *sorted((root / 'src').glob('*.py'))]}
    path.write_text(json.dumps(manifest, indent=2) + '\n')
