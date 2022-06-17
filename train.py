import torch
from collections import defaultdict
import os
from tqdm import tqdm


def do_train(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    weight_directory,
):
    result = defaultdict(list)
    current_loss = 1e10

    pbar = tqdm(range(num_epochs), total=num_epochs)
    for epoch in pbar:
        for mode in ["train", "valid"]:
            if mode == "train":
                model.train()
            else:
                model.eval()
            current_loss = 0
            data_size = len(dataloaders[mode].dataset.inputs)

            for inputs, label, names in dataloaders[mode]:
                optimizer.zero_grad()

                with torch.set_grad_enabled(mode == "train"):
                    outputs = []
                    for inp in inputs:
                        inp = inp.to(device)
                        out = model(inp)
                        outputs.append(out)
                    label = label.to(device)
                    loss_value = criterion(*outputs, label)
                    current_loss += loss_value.item()
                    if mode == "train":
                        loss_value.backward()
                        optimizer.step()
                        scheduler.step()

            epoch_loss = torch.true_divide(current_loss, data_size)

            cur_lr = scheduler.get_last_lr()[0]

            if mode == "valid":
                cur_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"epoch": epoch, "loss": epoch_loss, "lr": cur_lr})
                if epoch_loss < current_loss:
                    model_state_dict = model.cpu().state_dict()
                    torch.save(
                        model_state_dict,
                        os.path.join(weight_directory, "best_model.pt"),
                    )
                    model = model.to(device)
                    current_loss = epoch_loss
            result[f"{mode}_loss"].append(float(epoch_loss.numpy()))
    return result
