from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys


def export_onnx(source_dir: Path, checkpoint: Path, out: Path, sid: int, opset: int = 17) -> None:
    os.environ.setdefault("NUMBA_DEBUG", "0")
    logging.getLogger("numba").setLevel(logging.WARNING)
    sys.path.insert(0, str(source_dir.resolve()))

    import torch
    import utils
    from models import SynthesizerTrn

    hps = utils.get_hparams_from_file(str(source_dir / "config" / "config.json"))
    net = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    utils.load_checkpoint(str(checkpoint), net, None)
    net.eval()

    class InferWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, x_lengths, sid_tensor, noise_scale, noise_scale_w, length_scale):
            audio = self.model.infer(
                x,
                x_lengths,
                sid=sid_tensor,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0]
            return audio

    wrapper = InferWrapper(net).eval()
    x = torch.randint(low=0, high=len(hps.symbols), size=(1, 32), dtype=torch.long)
    x_lengths = torch.LongTensor([x.shape[1]])
    sid_tensor = torch.LongTensor([sid])
    noise_scale = torch.FloatTensor([0.6])
    noise_scale_w = torch.FloatTensor([0.668])
    length_scale = torch.FloatTensor([1.2])
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (x, x_lengths, sid_tensor, noise_scale, noise_scale_w, length_scale),
        str(out),
        input_names=["x", "x_lengths", "sid", "noise_scale", "noise_scale_w", "length_scale"],
        output_names=["audio"],
        dynamic_axes={
            "x": {1: "text_length"},
            "audio": {2: "audio_length"},
        },
        opset_version=opset,
        dynamo=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export zomehwh/vits-models-genshin-bh3 VITS pth to ONNX.")
    parser.add_argument("--source-dir", default="artifacts/sources/vits-models-genshin-bh3")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sid", type=int, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export_onnx(Path(args.source_dir), Path(args.checkpoint), Path(args.out), sid=args.sid, opset=args.opset)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
