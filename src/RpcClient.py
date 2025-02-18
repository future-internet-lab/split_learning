import time
import pickle
import pika
import random
import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn
from collections import defaultdict

import src.Log
import src.Model


class RpcClient:
    def __init__(self, client_id, layer_id, address, username, password, train_func, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.address = address
        self.username = username
        self.password = password
        self.train_func = train_func
        self.device = device

        self.channel = None
        self.connection = None
        self.response = None
        self.model = None
        self.cluster = None
        self.connect()

        self.train_set = None
        self.label_to_indices = None
        if self.layer_id == 1:
            # Read and load dataset
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            self.train_set = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)

            self.label_to_indices = defaultdict(list)
            for idx, (_, label) in enumerate(self.train_set):
                self.label_to_indices[label].append(idx)

    def wait_response(self):
        status = True
        reply_queue_name = f'reply_{self.client_id}'
        self.channel.queue_declare(reply_queue_name, durable=False)
        while status:
            method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
            if body:
                status = self.response_message(body)
            time.sleep(0.5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        src.Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
        action = self.response["action"]
        state_dict = self.response["parameters"]

        if action == "START":
            model_name = self.response["model_name"]
            cut_layers = self.response['layers']
            label_count = self.response['label_count']
            num_layers = self.response['num_layers']
            if self.response['cluster'] is not None:
                self.cluster = self.response['cluster']
            if label_count is not None:
                src.Log.print_with_color(f"Label distribution of client: {label_count}", "yellow")
            if self.model is None:
                klass = getattr(src.Model, model_name)
                full_model = klass()

                if cut_layers:
                    from_layer = cut_layers[0]
                    to_layer = cut_layers[1]
                    if to_layer == -1:
                        self.model = nn.Sequential(*nn.ModuleList(full_model.children())[from_layer:])
                    else:
                        self.model = nn.Sequential(*nn.ModuleList(full_model.children())[from_layer:to_layer])

                self.model.to(self.device)

            # Read parameters and load to model
            if state_dict:
                self.model.load_state_dict(state_dict)

            batch_size = self.response["batch_size"]
            lr = self.response["lr"]
            momentum = self.response["momentum"]
            control_count = self.response["control_count"]

            # Start training
            if self.layer_id == 1:
                selected_indices = []
                for label, count in enumerate(label_count):
                    selected_indices.extend(random.sample(self.label_to_indices[label], count))

                subset = torch.utils.data.Subset(self.train_set, selected_indices)
                train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

                result, size = self.train_func(self.model, lr, momentum, num_layers, control_count, train_loader, self.cluster)
            else:
                result, size = self.train_func(self.model, lr, momentum, num_layers, control_count, None, self.cluster)

            # Stop training, then send parameters to server
            model_state_dict = self.model.state_dict()
            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')
            data = {"action": "UPDATE", "client_id": self.client_id, "layer_id": self.layer_id,
                    "result": result, "size": size, "cluster": self.cluster,
                    "message": "Sent parameters to Server", "parameters": model_state_dict}
            src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            return True
        elif action == "STOP":
            return False

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
        self.channel = self.connection.channel()

    def send_to_server(self, message):
        self.connect()
        self.response = None

        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

        return self.response
