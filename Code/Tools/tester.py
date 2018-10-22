import torch
import torch.nn.functional as F
import torch.optim as optm
import indexes

def test(model, device, testLoader, lossFunction = nn.MSELos()):
        print("Commence Testing!")        
        loss = 0 
        true = 0
        acc  = 0
        # Inform Pytorch that keeping track of gradients is not required in
        # testing phase.
        with torch.no_grad():
            for data, label in testLoader:
                data, label = data.to(device), label.to(device)
                pred = model.forward(data).view_as(label)
                # Sum all loss terms and tern then into a numpy number for late use.
                loss  = lossFunction(pred, label, reduction = 'elementwise_mean').item()
                MAE  += torch.FloatTensor.abs(pred.sub(label)).sum().item()
                MAPE += torch.FloatTensor.abs(pred.sub(label)).div(label).mul(100).sum().item()

                # Find the max along a row but maitain the original dimenions.
                # in this case  a 10 -dimensional array.
                
                if idx % 20 == 0 or idx % pred.shape[0] == 0 : 
                    print("Epoch: {}-> Batch: {} / {}, Size: {}. Loss = {}".format(args, idx, len(indata),
                                                                           pred.shape[0], loss.item() ))
                    factor = (idx+1)*pred.shape[0]
                    print("Average MAE: {}, Average MAPE: {:.4f}%".format(MAE / factor, MAPE /factor))

        # Log the current train loss
        MAE  = MAE/len(testLoader.dataset)
        MAPE = MAPE/len(testLoader.dataset)
        model.history[ridx.testLoss].append(loss.item())   #get only the loss value
        model.history[ridx.testMAE].append(MAE)
        model.history[ridx.testMAPE].append(MAPE)

        # Print accuracy report!
        print("Accuracy: {} ({} / {})".format(acc, true,
                                              len(testLoader.dataset)))


