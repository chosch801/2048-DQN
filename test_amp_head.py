"""Regression for finite features overflowing the pre-normalization FP16 head."""
import unittest
import torch
from model import AfterstateValueNet

@unittest.skipUnless(torch.cuda.is_available(), "CUDA AMP regression")
class AmpHeadTests(unittest.TestCase):
    def test_large_prenorm_activation_keeps_amp_output_and_gradients_finite(self):
        net = AfterstateValueNet().cuda().eval()
        with torch.no_grad():
            net.head[1].weight.zero_()
            net.head[1].bias.zero_()
            net.head[1].bias[0] = 70000.0
        observation = torch.zeros((2, 16), device="cuda")
        dtypes = {}
        handles = [net.features.register_forward_hook(lambda m, i, o: dtypes.update(features=o.dtype)),
                   net.head[1].register_forward_hook(lambda m, i, o: dtypes.update(linear=o.dtype))]
        with torch.autocast("cuda", dtype=torch.float16):
            output = net(observation)
        for handle in handles:
            handle.remove()
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertEqual(dtypes["features"], torch.float16)
        self.assertEqual(dtypes["linear"], torch.float32)
        self.assertEqual(output.dtype, torch.float16)
        output.float().square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in net.parameters() if p.grad is not None))

if __name__ == "__main__":
    unittest.main()
