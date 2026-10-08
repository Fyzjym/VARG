"""CPU smoke tests for the paper-aligned interfaces; no model downloads."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models import SAT, HCEM, HEU, CAM, VARG, VARGConditioner
from data_loader.loader import HMEDataset, ContentImage
from parse_config import cfg, cfg_from_file
from utils.checkpoint import canonicalize_state_dict, load_varg_checkpoint


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sat_and_hcem_contract(self):
        content = torch.randn(2, 8, 4, 4) * 0.01
        style = torch.randn_like(content)
        sat = SAT(in_chans=8, depth=1, embed_dim=16, num_heads=2,
                  output_dim=8, patch_nums=(1, 2, 4)).eval()
        hcem = HCEM(in_chans=8, embed_dim=4, depth=1, output_dim=8).eval()
        with torch.no_grad():
            sat_tokens = sat(content, style)
            fused = hcem(sat_tokens, content)
            direct = hcem.cam(sat_tokens, hcem.heu(content))
        self.assertEqual(tuple(sat_tokens.shape), (2, 16, 8))
        self.assertTrue(torch.equal(fused, direct))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertIsInstance(hcem.heu, HEU)
        self.assertIsInstance(hcem.cam, CAM)

    def test_checkpoint_roundtrip_and_prefix(self):
        model = CAM(8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            torch.save({'state_dict': {'module.' + key: value
                                      for key, value in model.state_dict().items()}}, path)
            restored = CAM(8)
            load_varg_checkpoint(restored, path)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))
        tensor = torch.randn(1)
        migrated = canonicalize_state_dict({'module.conditioner.sat.weight': tensor})
        self.assertIs(migrated['conditioner.sat.weight'], tensor)
        with self.assertRaises(ValueError):
            canonicalize_state_dict({'weight': tensor, 'module.weight': tensor})

    def test_dataset_and_content_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('images', 'styles', 'content'):
                folder = root / name / 'train' / '12'
                folder.mkdir(parents=True)
                for stem in ('expr001', 'expr002'):
                    image = Image.fromarray(np.full((64, 160), 220, dtype=np.uint8))
                    image.save(folder / (stem + '.png'))
            annotations = root / 'train.txt'
            annotations.write_text('expr001  unused  12  x = r \\cos \\theta\n', encoding='utf-8')
            dataset = HMEDataset(str(root / 'images'), str(root / 'styles'), '',
                                 str(root / 'content'), 'train', annotation_path=annotations)
            item = dataset[0]
            batch = dataset.collate_batch([item])
            self.assertEqual(tuple(batch['img'].shape), (1, 3, 256, 256))
            self.assertEqual(tuple(batch['style'].shape), (1, 2, 256, 256))
            self.assertEqual(batch['wid'].tolist(), [12])
            self.assertEqual(torch.count_nonzero(batch['laplace']).item(), 0)
            content = ContentImage(root / 'content/train/12/expr001.png').load()
            self.assertEqual(tuple(content.shape), (1, 1, 256, 256))
            self.assertTrue((-1 <= content).all() and (content <= 1).all())

    def test_example_configuration(self):
        cfg_from_file(str(Path(__file__).resolve().parents[1] / 'configs/crohme.yaml'))
        self.assertEqual(cfg.MODEL.EMB_DIM, 512)
        self.assertEqual(cfg.TRAIN.IMS_PER_BATCH, 32)
        self.assertEqual(cfg.SOLVER.EPOCHS, 700)


if __name__ == '__main__':
    unittest.main()
