# gpmp/hyperparam/utils.py
# --------------------------------------------------------------
import gpmp.num as gnp

def check_xi_zi_or_loader(xi, zi, dataloader):
    """Validate arguments and determine data source."""
    arrays_provided = xi is not None and zi is not None
    loader_provided = dataloader is not None
    if arrays_provided and loader_provided:
        raise ValueError("Provide either (xi, zi) or dataloader, not both.")
    if not arrays_provided and not loader_provided:
        raise ValueError("Provide either (xi, zi) or dataloader.")
    return "arrays" if arrays_provided else "dataloader"

def prepare_data(xi=None, zi=None, loader=None):
    """Prepare access to data either from (xi, zi) arrays or from a DataLoader."""
    arrays_provided = xi is not None and zi is not None
    loader_provided = loader is not None
    if arrays_provided and loader_provided:
        raise ValueError("Provide either (xi, zi) or loader, not both.")
    if not arrays_provided and not loader_provided:
        raise ValueError("Provide either (xi, zi) or loader.")
    if arrays_provided:
        xi_ = gnp.asarray(xi)
        zi_ = gnp.asarray(zi).reshape(-1, 1)
        n, d = xi_.shape
        return xi_, zi_, n, d, "arrays"
    # loader path (assume dataset has length and per-item shapes)
    n = len(loader.dataset)
    d = loader.dataset.x_list[0].shape[1]
    return None, None, n, d, "loader"
