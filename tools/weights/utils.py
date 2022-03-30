from tqdm import tqdm


def quiet_tqdm(x, quiet):
    return x if quiet else tqdm(x)
