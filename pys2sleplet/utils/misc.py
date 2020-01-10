import toml


def load_config(filename):
    return toml.load(filename)
