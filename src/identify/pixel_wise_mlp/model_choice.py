from models.MLP_BatchNorm import MLP_BatchNorm

def model_choice(input_dim, output_dim, cfg_model):
    print("model:{}".format(cfg_model.name))
    if cfg_model.name == "MLP_BatchNorm":
        model = MLP_BatchNorm(input_dim, output_dim, cfg_model.dropout_prob)
    else:
        print("error")
    return model