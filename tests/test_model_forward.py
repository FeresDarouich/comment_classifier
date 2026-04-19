import torch

from Model import build_model


def test_model_forward_shape_cpu():
    model = build_model(
        vocab_size=100,
        max_len=12,
        num_classes=3,
        pad_id=0,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        use_cls_pooling=True,
    )

    input_ids = torch.randint(0, 100, (2, 12), dtype=torch.long)
    attention_mask = torch.ones((2, 12), dtype=torch.long)

    logits = model(input_ids, attention_mask)
    assert logits.shape == (2, 3)
