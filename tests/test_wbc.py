import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_wbc import metrics_from_confusion, prepare_data
from wbc_classifier import WBCClassifier


class WBCTrainingTests(unittest.TestCase):
    def test_metrics_use_equal_class_weight_for_macro_f1(self):
        result = metrics_from_confusion(np.array([[8, 2], [1, 1]]))
        self.assertAlmostEqual(result["accuracy"], 9 / 12)
        self.assertAlmostEqual(result["balanced_accuracy"], (0.8 + 0.5) / 2)
        self.assertAlmostEqual(result["macro_f1"], ((16 / 19) + (2 / 5)) / 2)
        self.assertEqual(result["support"], [10, 2])

    def test_duplicate_cleanup_protects_held_out_splits_and_quarantines_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            rng = np.random.default_rng(123)
            pixels = {}
            for split in ("train", "val", "test"):
                for name in ("a", "b"):
                    folder = root / split / name
                    folder.mkdir(parents=True)
                    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                    Image.fromarray(a).save(folder / "original.png")
                    pixels[split, name] = a
            Image.fromarray(pixels["val", "a"]).save(root / "train/a/overlap_val.png")
            Image.fromarray(pixels["test", "b"]).save(root / "train/b/overlap_test.png")
            Image.fromarray(pixels["train", "a"]).save(root / "test/b/conflicting.png")
            source_files = sorted(str(p.relative_to(root)) for p in root.rglob("*.png"))
            run = Path(directory) / "run"
            run.mkdir()
            data, audit = prepare_data(root, ["a", "b"], run)
            self.assertEqual(audit["conflicting_label_groups"], 1)
            self.assertEqual(len(audit["excluded"]), 4)
            self.assertEqual([len(data[s]) for s in ("train", "val", "test")], [1, 2, 2])
            manifest = json.loads((run / "manifest.json").read_text())
            hashes = [{r["sha256_pixels"] for r in manifest[s]} for s in ("train", "val", "test")]
            self.assertFalse(hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
            self.assertEqual(source_files, sorted(str(p.relative_to(root)) for p in root.rglob("*.png")))

    def test_zero_predictions_for_a_class_are_finite(self):
        result = metrics_from_confusion(np.array([[10, 0], [2, 0]]))
        self.assertEqual(result["precision"][1], 0)
        self.assertTrue(np.isfinite(result["macro_f1"]))

    def test_detection_crops_are_rgb_square_padded_and_only_wbcs_are_classified(self):
        class FakeClassifier:
            names = ["a", "b"]
            seen = []
            def predict(self, images):
                self.seen.extend(images)
                return [{"probabilities": {"a": 0.8, "b": 0.2}} for _ in images]
        fake = FakeClassifier()
        image = np.full((40, 40, 3), (230, 20, 10), dtype=np.uint8)
        boxes = np.array([[0, 0, 20, 10, 0.9, 0], [20, 20, 35, 35, 0.8, 1],
                          [100, 100, 110, 110, 0.8, 0]], dtype=float)
        probabilities = WBCClassifier.classify_detections(fake, image, boxes)
        self.assertEqual(probabilities, [[0.8, 0.2], None, None])
        self.assertEqual(len(fake.seen), 1)
        crop = np.asarray(fake.seen[0])
        self.assertEqual(crop.shape, (24, 24, 3))
        self.assertEqual(crop[0, 0].tolist(), [245, 245, 245])
        self.assertEqual(crop[12, 12].tolist(), [230, 20, 10])

    def test_wbc_classification_batches_keep_original_detection_order(self):
        class FakeClassifier:
            names = ["a", "b"]
            batches = []
            def predict(self, images):
                self.batches.append(len(images))
                return [{"probabilities": {"a": 0.8, "b": 0.2}} for _ in images]
        fake = FakeClassifier()
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 30, 30, 0.8, i % 2] for i in range(129)], dtype=float)
        output = WBCClassifier.classify_detections(fake, image, boxes)
        self.assertEqual(fake.batches, [32, 32, 1])
        self.assertEqual(len(output), 129)
        self.assertTrue(all((value is None) == (i % 2 == 1) for i, value in enumerate(output)))


if __name__ == "__main__":
    unittest.main()
