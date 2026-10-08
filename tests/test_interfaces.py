"""CPU smoke tests for the paper-aligned interfaces; no model downloads."""

import tempfile
import unittest
import inspect
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from models import SAT, HCEM, HEU, CAM, VARG, VARGConditioner
from data_loader.loader import HMEDataset, ContentImage, RandomStyleHMEDataset
from models.diffusion import Diffusion
from parse_config import cfg, cfg_from_file
from utils.checkpoint import canonicalize_state_dict, load_varg_checkpoint
from trainer.trainer import Trainer


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
            dataset = HMEDataset(str(root / 'images'), str(root / 'styles'),
                                 str(root / 'content'), 'train', annotation_path=annotations)
            item = dataset[0]
            batch = dataset.collate_batch([item])
            self.assertEqual(tuple(batch['img'].shape), (1, 3, 256, 256))
            self.assertEqual(tuple(batch['style'].shape), (1, 2, 256, 256))
            self.assertEqual(batch['wid'].tolist(), [12])
            self.assertEqual(set(batch), {'img', 'style', 'content', 'wid', 'target',
                                          'target_lengths', 'image_name', 'latex_seq'})
            references = RandomStyleHMEDataset(str(root / 'styles/train'),
                                                str(root / 'content/train'), ref_num=1)
            reference_batch = references[0]
            self.assertEqual(set(reference_batch), {'style', 'wid'})
            self.assertEqual(tuple(reference_batch['style'].shape), (1, 1, 256, 256))
            content = ContentImage(root / 'content/train/12/expr001.png').load()
            self.assertEqual(tuple(content.shape), (1, 1, 256, 256))
            self.assertTrue((-1 <= content).all() and (content <= 1).all())

    def test_example_configuration(self):
        cfg_from_file(str(Path(__file__).resolve().parents[1] / 'configs/crohme.yaml'))
        self.assertEqual(cfg.MODEL.EMB_DIM, 512)
        self.assertEqual(cfg.TRAIN.IMS_PER_BATCH, 32)
        self.assertEqual(cfg.SOLVER.EPOCHS, 700)

    def test_conditioner_and_network_signatures(self):
        self.assertEqual(list(inspect.signature(VARGConditioner.forward).parameters),
                         ['self', 'style', 'content', 'latex'])
        self.assertEqual(list(inspect.signature(VARGConditioner.generate).parameters),
                         ['self', 'style', 'content', 'latex'])
        self.assertEqual(list(inspect.signature(VARG.forward).parameters),
                         ['self', 'x', 'timesteps', 'style', 'content', 'latex_embed', 'tag', 'kwargs'])

    def test_sampling_interfaces(self):
        style = torch.rand(2, 1, 8, 8)
        content = torch.rand(2, 1, 8, 8)
        noise = torch.randn(2, 4, 2, 2)
        diffusion = Diffusion(noise_steps=8, device='cpu')

        class Denoiser(torch.nn.Module):
            def forward(inner, x, timesteps, styles, rendered_content, tag='test'):
                self.assertIs(styles, style)
                self.assertIs(rendered_content, content)
                prediction = torch.zeros_like(x)
                if tag == 'train':
                    embeddings = torch.ones(x.shape[0], 2, 4)
                    return prediction, embeddings, embeddings
                return prediction

        class Decoder:
            def decode(inner, latent):
                return SimpleNamespace(sample=latent[:, :3])

        for sampler in (diffusion.ddim_sample, diffusion.ddpm_sample):
            images = sampler(Denoiser(), Decoder(), 2, noise.clone(), style, content)
            self.assertEqual(tuple(images.shape), (2, 3, 2, 2))
            self.assertTrue(torch.isfinite(images).all())
        trained = diffusion.train_ddim(Denoiser(), noise.clone(), style, content,
                                       total_t=torch.tensor([6, 6]), sampling_timesteps=2)
        self.assertEqual(tuple(trained[0].shape), tuple(noise.shape))

    def test_training_batch_interface(self):
        class TrainingModel(torch.nn.Module):
            def __init__(inner):
                super().__init__()
                inner.scale = torch.nn.Parameter(torch.tensor(0.2))

            def forward(inner, x, timesteps, style, content, tag):
                self.assertEqual(tag, 'train')
                self.assertEqual(tuple(style.shape), (2, 2, 4, 4))
                self.assertEqual(tuple(content.shape), (2, 3, 4, 4))
                return x * inner.scale, inner.scale.expand(2, 2, 4)

        class Encoder:
            def encode(inner, images):
                latent = torch.cat([images, images[:, :1]], dim=1)
                return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: latent))

        model = TrainingModel()
        trainer = SimpleNamespace(
            model=model, vae=Encoder(), device='cpu',
            diffusion=Diffusion(noise_steps=8, device='cpu'),
            recon_criterion=torch.nn.MSELoss(),
            nce_criterion=lambda embeddings, labels: embeddings.square().mean(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        )
        batch = {'img': torch.rand(2, 3, 4, 4), 'style': torch.rand(2, 2, 4, 4),
                 'content': torch.rand(2, 3, 4, 4), 'wid': torch.tensor([12, 13]),
                 'target': torch.zeros(2, 128), 'latex_seq': ['x', 'y']}
        initial = model.scale.detach().clone()
        with mock.patch('trainer.trainer.dist.get_rank', return_value=1):
            Trainer._train_iter(trainer, batch, step=0, pbar=None)
        self.assertTrue(torch.isfinite(model.scale))
        self.assertFalse(torch.equal(initial, model.scale))


if __name__ == '__main__':
    unittest.main()
