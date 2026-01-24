if __main__ == "__main__":
    import time
    B, H, W, D = 1, 32, 1800, 800
    x = torch.randn(B, H, W, D).to('cuda')
    model = DenoiseModel(in_channels=1, num_classes=3, hidden_dim=32, use_axial_attn=True).to('cuda')
    with torch.no_grad():
        # Warmup
        _ = model(x)
        torch.cuda.synchronize()

        # Actual measurement
        st = time.time()
        y = model(x)
        torch.cuda.synchronize()  # Wait for GPU to finish
        ed = time.time()
        print('frame time (with sync): {}'.format(ed - st))
    print(y)