import numpy as np
import json
import torch
import torch.nn.functional as F

from dataclasses import asdict
from pathlib import Path
from glob import glob
from typing import List, Tuple, Union
from .utils import mkpath, compact_json_number_array
from .typings import AllInOneOutput, AnalysisResult, PathLike
from .postprocessing import (
  postprocess_metrical_structure,
  postprocess_functional_structure,
  estimate_tempo_from_beats,
)


def run_inference(
  path: Path,
  spec_path: Path,
  model: torch.nn.Module,
  device: str,
  include_activations: bool,
  include_embeddings: bool,
) -> AnalysisResult:
  spec = np.load(spec_path)
  spec = torch.from_numpy(spec).unsqueeze(0).to(device)

  logits = model(spec)

  metrical_structure = postprocess_metrical_structure(logits, model.cfg)
  functional_structure = postprocess_functional_structure(logits, model.cfg)
  bpm = estimate_tempo_from_beats(metrical_structure['beats'])

  result = AnalysisResult(
    path=path,
    bpm=bpm,
    segments=functional_structure,
    **metrical_structure,
  )

  if include_activations:
    activations = compute_activations(logits)
    result.activations = activations

  if include_embeddings:
    result.embeddings = logits.embeddings[0].cpu().numpy()

  return result


def compute_activations(logits: AllInOneOutput):
  activations_beat = torch.sigmoid(logits.logits_beat[0]).cpu().numpy()
  activations_downbeat = torch.sigmoid(logits.logits_downbeat[0]).cpu().numpy()
  activations_segment = torch.sigmoid(logits.logits_section[0]).cpu().numpy()
  activations_label = torch.softmax(logits.logits_function[0], dim=0).cpu().numpy()
  return {
    'beat': activations_beat,
    'downbeat': activations_downbeat,
    'segment': activations_segment,
    'label': activations_label,
  }


def collate_spectrograms(
  spec_paths: List[Path],
  device: str,
) -> Tuple[torch.Tensor, List[int]]:
  """Load and pad spectrograms into a single batch tensor.

  Returns the batched tensor [B, 4, T_max, 12] and a list of original lengths.
  """
  specs = [np.load(p) for p in spec_paths]
  lengths = [s.shape[1] for s in specs]  # spec shape is [4, T, 12]
  max_len = max(lengths)

  padded = []
  for spec in specs:
    t = torch.from_numpy(spec)  # [4, T, 12]
    pad_amount = max_len - t.shape[1]
    if pad_amount > 0:
      t = F.pad(t, (0, 0, 0, pad_amount))  # pad T dimension with zeros
    padded.append(t)

  batch = torch.stack(padded, dim=0).to(device)  # [B, 4, T_max, 12]
  return batch, lengths


def run_batch_inference(
  paths: List[Path],
  spec_paths: List[Path],
  model: torch.nn.Module,
  device: str,
  include_activations: bool,
  include_embeddings: bool,
) -> List[AnalysisResult]:
  """Run inference on a batch of tracks and return per-track results."""
  batch, lengths = collate_spectrograms(spec_paths, device)

  logits = model(batch)

  results = []
  for i, (path, orig_len) in enumerate(zip(paths, lengths)):
    # Slice this track's logits to its original length, removing padding
    track_logits = AllInOneOutput(
      logits_beat=logits.logits_beat[i:i+1, :orig_len],
      logits_downbeat=logits.logits_downbeat[i:i+1, :orig_len],
      logits_section=logits.logits_section[i:i+1, :orig_len],
      logits_function=logits.logits_function[i:i+1, :, :orig_len],
      embeddings=logits.embeddings[i:i+1, :, :orig_len, :],
    )

    metrical_structure = postprocess_metrical_structure(track_logits, model.cfg)
    functional_structure = postprocess_functional_structure(track_logits, model.cfg)
    bpm = estimate_tempo_from_beats(metrical_structure['beats'])

    result = AnalysisResult(
      path=path,
      bpm=bpm,
      segments=functional_structure,
      **metrical_structure,
    )

    if include_activations:
      result.activations = compute_activations(track_logits)

    if include_embeddings:
      result.embeddings = track_logits.embeddings[0].cpu().numpy()

    results.append(result)

  return results


def expand_paths(paths: List[Path]):
  expanded_paths = set()
  for path in paths:
    if '*' in str(path) or '?' in str(path):
      matches = [Path(p) for p in glob(str(path))]
      if not matches:
        raise FileNotFoundError(f'Could not find any files matching {path}')
      expanded_paths.update(matches)
    else:
      expanded_paths.add(path)

  return sorted(expanded_paths)


def check_paths(paths: List[Path]):
  missing_files = []
  for path in paths:
    if not path.is_file():
      missing_files.append(str(path))
  if missing_files:
    raise FileNotFoundError(f'Could not find the following files: {missing_files}')


def rmdir_if_empty(path: Path):
  try:
    path.rmdir()
  except (FileNotFoundError, OSError):
    pass


def save_results(
  results: Union[AnalysisResult, List[AnalysisResult]],
  out_dir: PathLike,
):
  if not isinstance(results, list):
    results = [results]

  out_dir = mkpath(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  for result in results:
    out_path = out_dir / result.path.with_suffix('.json').name
    result = asdict(result)
    result['path'] = str(result['path'])

    activations = result.pop('activations')
    if activations is not None:
      np.savez(str(out_path.with_suffix('.activ.npz')), **activations)

    embeddings = result.pop('embeddings')
    if embeddings is not None:
      np.save(str(out_path.with_suffix('.embed.npy')), embeddings)

    json_str = json.dumps(result, indent=2)
    json_str = compact_json_number_array(json_str)
    out_path.with_suffix('.json').write_text(json_str)
