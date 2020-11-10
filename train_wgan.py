from gan.wgan import GAN, celeba_data

gan = GAN(input_dim=(128, 128, 3), batch_size=32)
gan.weight_save_dir = "./data/weight/wgan/"

data = celeba_data("./celeba/")
gan.load_weight()
print("-------------------------Train-------------------------")
gan.train(20000, data)
