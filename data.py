from torch.utils.data import Dataset, DataLoader


class LatestDataset(Dataset):
    def __init__(self):
        super().__init__()
        # TODO: load the pre-saved data (.pt file for example)
        #  containing both latent and text embeddings.
        pass

    def __getitem__(self, index):
        pass


class LatentCaching:
    # TODO: For video, caching will require massive disk space.
    #  Implement chunked loading or stream from webdataset/S3.
    def store_latents(self, dataloader, vae, text_encoder):
        # TODO: Run the one-time encoding pass.
        pass
