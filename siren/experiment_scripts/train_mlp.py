
def get_model(cfg):
    if cfg.model_type == "mlp_3d":
        model = MLP3D(**cfg.mlp_config)
    nparameters = sum(p.numel() for p in model.parameters())
    print(model)
    print("Total number of parameters: %d" % nparameters)

    return model


def main():
    pass

if __name__ == "__main__":
    main()
