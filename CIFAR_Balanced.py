import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

from models.model import Net_2_layer_CNN, Net_3_layer_CNN
from datasets.dataset import gen_balance_client_data, val


def parseArg():
    parser = argparse.ArgumentParser(description='FedMD_pytorch, a federated learning framework. \
    Participants are training collaboratively. ')
    parser.add_argument(
        "--config_file", default="./configs/CIFAR_balance_conf.json", help="path to config file", type=str
    )

    args = parser.parse_args()

    return args


args =  parseArg()
conf_file = args.config_file
with open(conf_file, "r") as f:
    conf_dict = eval(f.read())
    model_config = conf_dict["models"]


CANDIDATE_MODELS = {"2_layer_CNN": Net_2_layer_CNN, 
                    "3_layer_CNN": Net_3_layer_CNN} 

print()
print("*************************")

print(model_config)


print("*************************")
print()




server_dataset, client_dataset, cs_dataset, test_dataset = gen_balance_client_data(prefix="./data/cifar-10-batches-py/")

server_loader = torch.utils.data.DataLoader(server_dataset, batch_size=4, shuffle=True, num_workers=2)

client_loader = []
cs_loader = []
for i in range(len(client_dataset)):
    client_loader.append(torch.utils.data.DataLoader(client_dataset[i], batch_size=4, shuffle=True, num_workers=2))
    cs_loader.append(torch.utils.data.DataLoader(cs_dataset[i], batch_size=4, shuffle=True, num_workers=2))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)



print()
print("*************************")

server_img, server_label = iter(server_loader).next()
print("len(server_dataset) : ", len(server_dataset))
print("server_img.shape : ", server_img.shape)
print("type(server_label) : ", type(server_label))
print("server_label.shape : ", server_label.shape)
print("server_label : ", server_label)
print(server_label.dtype)

print()

client_img, client_label = iter(client_loader[0]).next()
print("len(client_dataset0) : ", len(client_dataset[0]))
print("client_img.shape : ", client_img.shape)
print("type(client_label) : ", type(client_label))
print("client_label.shape : ", client_label.shape)
print("client_label : ", client_label)
print(client_label.dtype)

print()

client_img, client_label = iter(client_loader[1]).next()
print("len(client_dataset1) : ", len(client_dataset[1]))
print("client_img.shape : ", client_img.shape)
print("type(client_label) : ", type(client_label))
print("client_label.shape : ", client_label.shape)
print("client_label : ", client_label)
print(client_label.dtype)

print()

test_img, test_label = iter(test_loader).next()
print("len(test_dataset) : ", len(test_dataset))
print("test_img.shape : ", test_img.shape)
print("type(test_label) : ", type(test_label))
print("test_label.shape : ", test_label.shape)
print("test_label : ", test_label)
print(test_label.dtype)

print("*************************")
print()


clinet_nets = nn.ModuleList()
for i, item in enumerate(model_config):
    model_name = item["model_type"]
    model_params = item["params"]
    tmp = CANDIDATE_MODELS[model_name](n_classes=10, **model_params)
    clinet_nets.append(tmp)
print(clinet_nets)

out0 = clinet_nets[0](server_img)
print(out0.shape)

out1 = clinet_nets[1](server_img)
print(out1.shape)



print()
print("training...")
print()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for net in clinet_nets:
    net.to(device)

for n, net in enumerate(clinet_nets):

    total_server_epoch = 30
    log_interval = 500

    criterion_pretrain = nn.CrossEntropyLoss()
    optimizer_pretrain = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler_pretrain = optim.lr_scheduler.MultiStepLR(optimizer_pretrain, milestones=[20], gamma=0.1)
    
    for epoch in range(total_server_epoch):

        running_loss = 0.0
        for i, data in enumerate(cs_loader[n]):

            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer_pretrain.zero_grad()
            outputs = net(inputs)
            loss = criterion_pretrain(outputs, labels)
            loss.backward()
            optimizer_pretrain.step()

            running_loss += loss.item()
            if i % log_interval == log_interval - 1:
                lr_now = optimizer_pretrain.param_groups[0]['lr']
                print('[Pretraining : model %d, epoch %d, iter %d, lr_now %f] loss: %f' % (n, epoch + 1, i + 1, lr_now, running_loss / log_interval))
                running_loss = 0.0

        scheduler_pretrain.step()
        
        print("Pretraining : model %d : epoch %d/%d finished." % (n, epoch+1, total_server_epoch))
        val(test_loader, net, device)
        save_path = "./save/Pretrain_client_%d_epoch_%d.pkl" % (n, epoch+1)
        torch.save(net.state_dict(), save_path)
        print()

    print('Pretraining : model %d Finished Training.' % (n))
    print()


# collaborative_training
distillation_batch_size = 4
data_sampler = torch.utils.data.RandomSampler(server_dataset)
distillation_dataloader = torch.utils.data.DataLoader(server_dataset, batch_size=distillation_batch_size, sampler=data_sampler)

Distillation_Total_Epoch = 30
optimizer_distillation = [optim.SGD(net.parameters(), lr=0.0001, momentum=0.9) for _ in clinet_nets]
scheduler_distillation = [optim.lr_scheduler.MultiStepLR(opt, milestones=[20], gamma=0.1) for opt in optimizer_distillation]

criterion_refine = nn.CrossEntropyLoss()
optimizer_refine = [optim.SGD(net.parameters(), lr=0.0001, momentum=0.9) for _ in clinet_nets]
scheduler_refine = [optim.lr_scheduler.MultiStepLR(opt, milestones=[20], gamma=0.1) for opt in optimizer_refine]


def loss_func(output, label):
    LogSoftmaxOutput = F.log_softmax(output, dim=1)
    loss = label * LogSoftmaxOutput
    loss = -1 * loss.sum() / torch.tensor(float(distillation_batch_size))
    return loss

log_interval = 500
for ep in range(Distillation_Total_Epoch):
    for i, data in enumerate(distillation_dataloader):

        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = []
        distillation_losses = [0.0 for _ in clinet_nets]
        with torch.no_grad():
            for n, net in enumerate(clinet_nets): 
                outputs.append(net(inputs))
            logit_avg = torch.mean(torch.stack(outputs), 0)

        for n, net in enumerate(clinet_nets):
            optimizer_distillation[n].zero_grad()
            out = net(inputs)
            loss = loss_func(out, logit_avg) # distillation
            loss.backward()
            optimizer_distillation[n].step()

            distillation_losses[n] += loss.item()
            if i % log_interval == log_interval - 1:
                lr_now = optimizer_distillation[n].param_groups[0]['lr']
                print('[Distillation ：model %d, epoch %d, iter %d, lr_now %f] loss: %f' % (n, ep + 1, i + 1, lr_now, distillation_losses[n] / log_interval))
                distillation_losses[n] = 0.0
    
    for n in range(len(scheduler_distillation)):
        scheduler_distillation[n].step()
    
    print("Distillation : epoch %d/%d finished." % (ep+1, Distillation_Total_Epoch))
    for n, net in enumerate(clinet_nets):
        print("val for model %d : " % (n)) 
        val(test_loader, net, device)
        save_path = "./save/Distillation_client_%d_epoch_%d.pkl" % (n, ep+1)
        torch.save(net.state_dict(), save_path)
        print()
    

    
    # refine in client data
    refine_losses = [0.0 for _ in clinet_nets]
    for n, net in enumerate(clinet_nets):
        for i, data in enumerate(client_loader[n]):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer_refine[n].zero_grad()
            out = net(inputs)
            loss = criterion_refine(out, labels) # refine
            loss.backward()
            optimizer_refine[n].step()

            refine_losses[n] += loss.item()
            if i % log_interval == log_interval - 1:
                lr_now = optimizer_refine[n].param_groups[0]['lr']
                print('[Refine ：model %d, epoch %d, iter %d, lr_now %f] loss: %f' % (n, ep + 1, i + 1, lr_now, refine_losses[n] / log_interval))
                refine_losses[n] = 0.0
        
        for n in range(len(scheduler_refine)):
            scheduler_refine[n].step()
        
    print("Refine : epoch %d/%d finished." % (ep+1, Distillation_Total_Epoch))
    for n, net in enumerate(clinet_nets):
        print("val for model %d : " % (n)) 
        val(test_loader, net, device)
        save_path = "./save/Refine_client_%d_epoch_%d.pkl" % (n, ep+1)
        torch.save(net.state_dict(), save_path)
        print()
    

