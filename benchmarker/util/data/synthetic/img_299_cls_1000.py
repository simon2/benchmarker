"""Generate synthetic data for 299x299 images"""


from .helpers import gen_classification_data


def get_data(params):
    """Generate synthetic 299x299 images. Set `params["size"]`
    appropriately. Import this function in the `data.py` of the
    problem, so it can be called by `INeuralNet`.

    """
    return gen_classification_data(params, 1000, 299)
