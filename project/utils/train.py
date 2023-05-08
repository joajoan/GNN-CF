import pickle
import torch
import tqdm
import os
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def dispatch_epoch(
    module: type[Module], 
    loader: type[DataLoader],
    loss_fn: callable,
    *,
    batch_handler: callable,
    optimizer: type[Optimizer] = None, 
    device: torch.device = None,
    verbose: bool = False
) -> float:

    # Sends the model to the specified device.
    module = module.to(device)
    # Empties the GPU cache, if that device is set.
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Wraps the loader in a progress measurer, if verbose is set.
    if verbose is True:
        loader = tqdm.tqdm(loader, mininterval=1., position=0, leave=True)
    # Initializes the cumulative loss tracker.
    cum_loss = 0
    # Iterates over the data-loader's batches.
    for batch in loader:

        # Resets the gradient if an optimizer exists.
        if optimizer:
            optimizer.zero_grad()

        # Constructs the design and target data structures.
        loss = batch_handler(module, batch, loss_fn, 
            device=device
        )

        # Updates the module, if an optimizer has been given
        if optimizer:
            loss.backward()
            optimizer.step()
        # Updates the cumulative loss.
        cum_loss += loss.item()
                
    # Returns the traced loss.
    return cum_loss / len(loader)


def dispatch_session(
    module: type[Module],
    *,
    update_fn: callable,
    num_epochs: int,
    path: str,
    device: torch.device = None,
    validate_fn: callable = None,
    score_fn: list[callable] = None,
    verbose: bool = False,
) -> None:
    
    # Ensures a device is specified.
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
    # Builds the needed paths.
    if not os.path.exists(path):
        os.mkdir(path)
    stt_base = os.path.join(path, 'mdl')
    if not os.path.exists(stt_base):
        os.mkdir(stt_base)
    stt_path = os.path.join(stt_base, '{:02d}.pt')
    trc_path = os.path.join(path, 'trc.pkl')

    # Clears the GPU cache.
    torch.cuda.empty_cache()

    # Initializes the trace buffer.
    trace = {'update': []}
    if validate_fn is not None:
        trace |= {'validate': []}
    if score_fn is not None:
        trace |= {'score': []}

    # Iterates
    for epoch_index in range(1, num_epochs+1):

        # Outputs the current epoch index.
        if verbose and epoch_index != 1:
            print()
        if verbose:
            print(f'Epoch({epoch_index})')

        # Updates the model.
        loss = update_fn(module, 
            verbose=verbose, 
            device=device
        )
        trace['update'].append(loss)
        if verbose:
            print(f'Update({loss:.4f})')

        # Validates the model, if a function is given.
        if validate_fn is not None:
            loss = validate_fn(module, 
                verbose=verbose, 
                device=device
            )
            trace['validate'].append(loss)
            if verbose:
                print(f'Validate({loss:.4f})')

        # Scores the model, if a function is given.
        if score_fn is not None:
            score = score_fn(module, 
                verbose=verbose, 
                device=device
            )
            trace['score'].append(score)
            print(f'Score([{", ".join([f"{s:.2%}" for s in score])}])')
        
        # Saves the trace.
        with open(trc_path, 'wb') as file:
            pickle.dump(trace, file)

        # Saves the model.
        torch.save({
            key: tensor.cpu() 
                for key, tensor 
                in module.state_dict().items()
        }, stt_path.format(epoch_index))