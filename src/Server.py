import os
import random
import pika
import pickle
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import requests
import copy

from requests.auth import HTTPBasicAuth

import src.Model
import src.Log
import src.Utils
from src.Cluster import clustering_algorithm
import src.Validation

num_labels = 10


def delete_old_queues(address, username, password):
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "gradient_queue") or queue_name.startswith("rpc_queue"):
                try:
                    http_channel.queue_delete(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' deleted.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to delete queue '{queue_name}': {e}", "yellow")
            else:
                try:
                    http_channel.queue_purge(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' purged.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to purge queue '{queue_name}': {e}", "yellow")

        connection.close()
        return True
    else:
        src.Log.print_with_color(
            f"Failed to fetch queues from RabbitMQ Management API. Status code: {response.status_code}", "yellow")
        return False


class Server:
    def __init__(self, config_dir):
        with open(config_dir, 'r') as file:
            config = yaml.safe_load(file)

        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        delete_old_queues(address, username, password)

        self.model_name = config["server"]["model"]
        self.total_clients = config["server"]["clients"]
        self.cut_layers = config["server"]["cut_layers"]
        self.local_round = config["server"]["local-round"]
        self.global_round = config["server"]["global-round"]
        self.round = self.global_round
        self.save_parameters = config["server"]["parameters"]["save"]
        self.load_parameters = config["server"]["parameters"]["load"]
        self.validation = config["server"]["validation"]

        # Clients
        self.batch_size = config["learning"]["batch-size"]
        self.lr = config["learning"]["learning-rate"]
        self.momentum = config["learning"]["momentum"]
        self.control_count = config["learning"]["control-count"]
        self.data_distribution = config["server"]["data-distribution"]

        # Cluster
        self.client_cluster_config = config["server"]["client-cluster"]
        self.mode_cluster = self.client_cluster_config["enable"]
        self.special = self.client_cluster_config["special"]
        if not self.mode_cluster:
            self.local_round = 1

        # Data non-iid
        self.data_mode = config["server"]["data-mode"]
        self.data_range = self.data_distribution["num-data-range"]
        self.non_iid_rate = self.data_distribution["non-iid-rate"]
        self.refresh_each_round = self.data_distribution["refresh-each-round"]
        self.random_seed = config["server"]["random-seed"]

        if self.random_seed:
            random.seed(self.random_seed)

        log_path = config["log_path"]

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue='rpc_queue')

        self.current_clients = [0 for _ in range(len(self.total_clients))]
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.first_layer_clients_in_each_cluster = []
        self.responses = {}  # Save response
        self.list_clients = []
        self.global_avg_state_dict = [[] for _ in range(len(self.total_clients))]
        self.round_result = True

        self.global_model_parameters = [[] for _ in range(len(self.total_clients))]
        self.global_client_sizes = [[] for _ in range(len(self.total_clients))]
        self.local_model_parameters = None
        self.local_client_sizes = None
        self.local_avg_state_dict = None
        self.total_cluster_size = None
        self.list_cut_layers = [self.cut_layers]

        self.label_counts = None
        self.non_iid_label = None
        if not self.refresh_each_round:
            self.non_iid_label = [src.Utils.non_iid_rate(num_labels,
                                                         self.non_iid_rate) for _ in range(self.total_clients[0])]

        self.num_cluster = None
        self.current_local_training_round = None
        self.infor_cluster = None
        self.current_infor_cluster = None
        self.local_update_count = 0

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)
        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.logger.log_info("Application start")

        src.Log.print_with_color(f"Server is waiting for {self.total_clients} clients.", "green")

    def distribution(self):
        if self.data_mode == "even":
            # self.label_counts = np.array(
            #     [[50 // self.total_clients[0] for _ in range(num_labels)] for _ in range(self.total_clients[0])])
            self.label_counts = np.array(
                [[250 for _ in range(num_labels)] for _ in range(self.total_clients[0])])
        else:
            if self.refresh_each_round:
                self.non_iid_label = [src.Utils.non_iid_rate(num_labels, self.non_iid_rate) for _ in
                                      range(self.total_clients[0])]
            # self.label_counts = [np.array([random.randint(int(self.data_range[0] // self.non_iid_rate),
            #                                               int(self.data_range[1] // self.non_iid_rate))
            #                      for _ in range(num_labels)]) *
            #                      self.non_iid_label[i] for i in range(self.total_clients[0])]
            #

            self.label_counts = [[50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
                                 [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
                                 [50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
                                 [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]]

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]
        self.responses[routing_key] = message

        if action == "REGISTER":
            performance = message['performance']
            if (str(client_id), layer_id, performance, 0) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id, performance, -1))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # Save messages from clients
            self.register_clients[layer_id - 1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                self.distribution()
                self.cluster_client()
                print(self.list_cut_layers)
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                src.Log.print_with_color(f"Start training round {self.global_round - self.round + 1}", "yellow")
                self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                self.notify_clients()
        elif action == "NOTIFY":
            cluster = message["cluster"]
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            message = {"action": "PAUSE",
                       "message": "Pause training and please send your parameters",
                       "parameters": None}
            if layer_id == 1:
                self.first_layer_clients_in_each_cluster[cluster] += 1

            if self.first_layer_clients_in_each_cluster[cluster] == self.infor_cluster[cluster][0]:
                self.first_layer_clients_in_each_cluster[cluster] = 0
                src.Log.print_with_color(f"Received finish training notification cluster {cluster}", "yellow")

                for (client_id, layer_id, _, clustering) in self.list_clients:
                    if clustering == cluster:
                        if self.special is False:
                            self.send_to_response(client_id, pickle.dumps(message))
                        else:
                            if layer_id == 1:
                                self.send_to_response(client_id, pickle.dumps(message))
                self.local_update_count += 1

            if self.special and self.local_update_count == self.num_cluster * self.local_round:
                self.local_update_count = 0
                for (client_id, layer_id, _) in self.list_clients:
                    if layer_id != 1:
                        self.send_to_response(client_id, pickle.dumps(message))

        elif action == "UPDATE":
            # self.distribution()
            data_message = message["message"]
            result = message["result"]
            src.Log.print_with_color(f"[<<<] Received message from client: {data_message}", "blue")
            cluster = message["cluster"]
            # Global update
            if self.current_local_training_round[cluster] == self.local_round - 1:
                self.current_clients[layer_id - 1] += 1
                if not result:
                    self.round_result = False

                # Save client's model parameters
                if self.save_parameters and self.round_result:
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    self.local_model_parameters[cluster][layer_id - 1].append(model_state_dict)
                    self.local_client_sizes[cluster][layer_id - 1].append(client_size)

                # If consumed all client's parameters
                if self.current_clients == self.total_clients:
                    src.Log.print_with_color("Collected all parameters.", "yellow")
                    if self.save_parameters and self.round_result:
                        for i in range(0, self.num_cluster):
                            self.total_cluster_size[i] = sum(self.local_client_sizes[i][0])
                            self.avg_all_parameters(i)
                            self.local_model_parameters[i] = [[] for _ in range(len(self.total_clients))]
                            self.local_client_sizes[i] = [[] for _ in range(len(self.total_clients))]
                    self.current_clients = [0 for _ in range(len(self.total_clients))]
                    self.current_local_training_round = [0 for _ in range(self.num_cluster)]
                    # Test
                    if self.save_parameters and self.validation and self.round_result:
                        state_dict_full = self.concatenate_state_dict()
                        if not src.Validation.test(self.model_name, state_dict_full, self.logger):
                            src.Log.print_with_color("Training failed!", "yellow")
                        else:
                            # Save to files
                            torch.save(state_dict_full, f'{self.model_name}.pth')
                            self.round -= 1
                    else:
                        self.round -= 1

                    # Start a new training round
                    self.round_result = True

                    if self.round > 0:
                        src.Log.print_with_color(f"Start training round {self.global_round - self.round + 1}", "yellow")
                        if self.save_parameters:
                            self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                            self.notify_clients(special=self.special)
                        else:
                            self.notify_clients(register=False, special=self.special)
                    else:
                        self.logger.log_info("Stop training !!!")
                        self.notify_clients(start=False)
                        sys.exit()

            # Local update
            else:
                if not result:
                    self.round_result = False
                if self.round_result:
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    self.local_model_parameters[cluster][layer_id - 1].append(model_state_dict)
                    self.local_client_sizes[cluster][layer_id - 1].append(client_size)
                self.current_infor_cluster[cluster][layer_id - 1] += 1

                if self.special is False:
                    if self.current_infor_cluster[cluster] == self.infor_cluster[cluster]:
                        self.avg_all_parameters(cluster=cluster)
                        self.notify_clients(cluster=cluster, special=False)
                        self.current_local_training_round[cluster] += 1

                        self.local_model_parameters[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.local_client_sizes[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.current_infor_cluster[cluster] = [0 for _ in range(len(self.total_clients))]
                else:
                    if self.current_infor_cluster[cluster][0] == self.infor_cluster[cluster][0]:
                        self.avg_all_parameters(cluster=cluster)
                        self.notify_clients(cluster=cluster, special=True)
                        self.current_local_training_round[cluster] += 1

                        self.local_model_parameters[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.local_client_sizes[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.current_infor_cluster[cluster] = [0 for _ in range(len(self.total_clients))]

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, start=True, register=True, cluster=None, special=False):
        label_counts = copy.copy(self.label_counts)
        label_counts = label_counts.tolist()
        if cluster is not None and special is False:
            for (client_id, layer_id, _, clustering) in self.list_clients:
                if clustering == cluster:
                    if layer_id == 1:
                        layers = [0, self.list_cut_layers[cluster][0]]
                    elif layer_id == len(self.total_clients):
                        layers = [self.list_cut_layers[cluster][-1], -1]
                    else:
                        layers = [self.list_cut_layers[cluster][layer_id - 2], self.list_cut_layers[cluster][layer_id - 1]]
                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    if layer_id == 1:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": self.local_avg_state_dict[cluster][layer_id - 1],
                                    "num_layers": len(self.total_clients),
                                    "layers": layers,
                                    "model_name": self.model_name,
                                    "control_count": self.control_count,
                                    "batch_size": self.batch_size,
                                    "lr": self.lr,
                                    "momentum": self.momentum,
                                    "label_count": None,
                                    "cluster": None,
                                    "special": False}
                    else:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": self.local_avg_state_dict[cluster][layer_id - 1],
                                    "num_layers": len(self.total_clients),
                                    "layers": layers,
                                    "model_name": self.model_name,
                                    "control_count": self.control_count,
                                    "batch_size": self.batch_size,
                                    "lr": self.lr,
                                    "momentum": self.momentum,
                                    "label_count": None,
                                    "cluster": None,
                                    "special": False}
                    self.send_to_response(client_id, pickle.dumps(response))
        if cluster is None:
            # Send message to clients when consumed all clients
            klass = getattr(src.Model, self.model_name)
            full_model = klass()
            full_model = nn.Sequential(*nn.ModuleList(full_model.children()))
            for (client_id, layer_id, _, clustering) in self.list_clients:
                # Read parameters file
                filepath = f'{self.model_name}.pth'
                state_dict = None

                if start:
                    if layer_id == 1:
                        layers = [0, self.list_cut_layers[clustering][0]]
                    elif layer_id == len(self.total_clients):
                        layers = [self.list_cut_layers[clustering][-1], -1]
                    else:
                        layers = [self.list_cut_layers[clustering][layer_id - 2], self.list_cut_layers[clustering][layer_id - 1]]

                    if self.load_parameters and register:
                        if os.path.exists(filepath):
                            full_state_dict = torch.load(filepath, weights_only=True)
                            full_model.load_state_dict(full_state_dict)

                            if layer_id == 1:
                                model_part = nn.Sequential(*nn.ModuleList(full_model.children())[:layers[1]])
                            elif layer_id == len(self.total_clients):
                                model_part = nn.Sequential(*nn.ModuleList(full_model.children())[layers[0]:])
                            else:
                                model_part = nn.Sequential(*nn.ModuleList(full_model.children())[layers[0]:layers[1]])

                            state_dict = model_part.state_dict()
                            src.Log.print_with_color("Model loaded successfully.", "green")
                        else:
                            src.Log.print_with_color(f"File {filepath} does not exist.", "yellow")

                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    if layer_id == 1:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": state_dict,
                                    "num_layers": len(self.total_clients),
                                    "layers": layers,
                                    "model_name": self.model_name,
                                    "control_count": self.control_count,
                                    "batch_size": self.batch_size,
                                    "lr": self.lr,
                                    "momentum": self.momentum,
                                    "label_count": label_counts.pop(),
                                    "cluster": clustering,
                                    "special": self.special}
                    else:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": state_dict,
                                    "num_layers": len(self.total_clients),
                                    "layers": layers,
                                    "model_name": self.model_name,
                                    "control_count": self.control_count,
                                    "batch_size": self.batch_size,
                                    "lr": self.lr,
                                    "momentum": self.momentum,
                                    "label_count": None,
                                    "cluster": clustering,
                                    "special": self.special}

                else:
                    src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                    response = {"action": "STOP",
                                "message": "Stop training!",
                                "parameters": None}
                self.send_to_response(client_id, pickle.dumps(response))
        if cluster is not None and special is True:
            for (client_id, layer_id, _, clustering) in self.list_clients:
                if clustering == cluster:
                    if layer_id == 1:
                        layers = [0, self.list_cut_layers[cluster][0]]
                    elif layer_id == len(self.total_clients):
                        layers = [self.list_cut_layers[cluster][-1], -1]
                    else:
                        layers = [self.list_cut_layers[cluster][layer_id - 2], self.list_cut_layers[cluster][layer_id - 1]]

                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    if layer_id == 1:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": self.local_avg_state_dict[cluster][layer_id - 1],
                                    "num_layers": len(self.total_clients),
                                    "layers": layers,
                                    "model_name": self.model_name,
                                    "control_count": self.control_count,
                                    "batch_size": self.batch_size,
                                    "lr": self.lr,
                                    "momentum": self.momentum,
                                    "label_count": None,
                                    "cluster": None,
                                    "special": True}
                        self.send_to_response(client_id, pickle.dumps(response))

    def cluster_client(self):
        list_performance = [-1 for _ in range(len(self.list_clients))]
        for idx, (client_id, layer_id, performance, cluster) in enumerate(self.list_clients):
            list_performance[idx] = performance
        # Phân cụm ở đây chỉ layer đầu
        if self.mode_cluster is True:
            list_cluster, infor_cluster, num_cluster, list_cut_layers = clustering_algorithm(list_performance, self.total_clients[1], self.client_cluster_config)
            self.infor_cluster = infor_cluster
            self.num_cluster = num_cluster
            self.list_cut_layers = list_cut_layers
        else:
            list_cluster = [0 for _ in range(len(list_performance))]
            self.num_cluster = 1
            self.infor_cluster = [self.total_clients]
        for idx, (client_id, layer_id, performance, cluster) in enumerate(self.list_clients):
            self.list_clients[idx] = (client_id, layer_id, performance, list_cluster[idx])

        self.local_model_parameters = [[[] for _ in range(len(self.total_clients))] for _ in range(self.num_cluster)]
        self.local_client_sizes = [[[] for _ in range(len(self.total_clients))] for _ in range(self.num_cluster)]
        self.local_avg_state_dict = [[[] for _ in range(len(self.total_clients))] for _ in range(self.num_cluster)]
        self.total_cluster_size = [0 for _ in range(self.num_cluster)]
        if self.mode_cluster:
            self.first_layer_clients_in_each_cluster = [0 for _ in range(self.num_cluster)]
        else:
            self.first_layer_clients_in_each_cluster = [0]
        self.current_infor_cluster = [[0] * len(row) for row in self.infor_cluster]
        self.current_local_training_round = [0 for _ in range(len(self.infor_cluster))]

    def start(self):
        self.channel.start_consuming()

    def send_to_response(self, client_id, message):
        reply_queue_name = f'reply_{client_id}'
        self.reply_channel.queue_declare(reply_queue_name, durable=False)

        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def avg_all_parameters(self, cluster=None):
        size = self.local_client_sizes[cluster]
        parameters = self.local_model_parameters[cluster]
        for layer, state_dicts in enumerate(parameters):
            local_layer_client_size = size[layer]
            num_models = len(state_dicts)
            if num_models == 0:
                return
            self.local_avg_state_dict[cluster][layer] = state_dicts[0]

            for key in state_dicts[0].keys():
                if state_dicts[0][key].dtype != torch.long:
                    self.local_avg_state_dict[cluster][layer][key] = sum(
                        state_dicts[i][key] * local_layer_client_size[i]
                        for i in range(num_models)) / sum(local_layer_client_size)
                else:
                    self.local_avg_state_dict[cluster][layer][key] = sum(
                        state_dicts[i][key] * local_layer_client_size[i]
                        for i in range(num_models)) // sum(local_layer_client_size)

    def concatenate_state_dict(self):
        state_dict_cluster = {}
        list_state_dict_cluster = [state_dict_cluster for _ in range(self.num_cluster)]
        for cluster in range(self.num_cluster):
            for i, state_dicts in enumerate(self.local_avg_state_dict[cluster]):
                if i > 0:
                    state_dicts = src.Utils.change_state_dict(state_dicts, self.list_cut_layers[cluster][i - 1])
                list_state_dict_cluster[cluster].update(state_dicts)

        state_dict_full = list_state_dict_cluster[0]
        for key in list_state_dict_cluster[0].keys():
            if list_state_dict_cluster[0][key].dtype != torch.long:
                state_dict_full[key] = sum(
                    list_state_dict_cluster[i][key] * self.total_cluster_size[i]
                    for i in range(self.num_cluster)) / sum(self.total_cluster_size)
            else:
                state_dict_full[key] = sum(
                    list_state_dict_cluster[i][key] * self.total_cluster_size[i]
                    for i in range(self.num_cluster)) // sum(self.total_cluster_size)

        return state_dict_full
