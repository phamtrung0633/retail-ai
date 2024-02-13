import os
import sys

import yaml
import dotenv
import dvc.api

import torch

import embeddings.schema as schema
from embeddings.vit import VIT

from numpy import zeros
from torchvision import models, transforms
from torchvision.models import feature_extraction
from pymilvus import Collection, Partition, MilvusClient, connections

NUM_CHANNELS = 3
PRECLASSIFICATION_IDX = -2

PARAMS_FILE = 'params.yaml'

USE_VIT = True

class EmbedderResNet:

    K_MAX = 5

    def __init__(self):
        self._collection = None
        self._partitions = {}

        self.preclassifier = None
        self.preprocess = None
        self.device = None
        self.fx = None

    def initialise(self):
        dotenv.load_dotenv()

        module = os.path.dirname(__file__)

        params = dvc.api.params_show(os.path.join(module, PARAMS_FILE))
        path = os.path.join(module, params['model']['path'], params['model']['name'])
        config_artifact = os.path.join(module, params['config'])

        print(path)

        if not (os.path.exists(path) and os.path.exists(config_artifact)):
            print("Artifacts from stage 'prepare' are missing, run `dvc repro` in the containing directory to restore them.")
            sys.exit()

        with open(config_artifact) as config:
            arch = yaml.load(config.read(), Loader = yaml.Loader)['arch']
            print(f"Model architecture '{arch}' detected")

        model = models.get_model(arch, weights = None)
        weights = models.get_model_weights(arch).DEFAULT
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model = model.to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            weights.transforms(antialias = True),
        ])

        train_nodes, eval_nodes = models.feature_extraction.get_graph_node_names(model)
        preclassifier = eval_nodes[PRECLASSIFICATION_IDX]

        fx = models.feature_extraction.create_feature_extractor(model, return_nodes = [preclassifier])

        print(f"Feature extractor traced to layer '{preclassifier}'")
        print("Tracing model execution...")

        with torch.no_grad():
            dummy = preprocess(zeros((1, 1, NUM_CHANNELS)))
            output = fx(dummy.unsqueeze(0).to(device))[preclassifier].squeeze(0)

            print(f"\tIn Dims: {list(dummy.shape)}")
            print(f"\tOut Dims: {list(output.shape)}\n")

        milvus = MilvusClient(
            uri = os.getenv('MILVUS_URI'),
            token = os.getenv('MILVUS_TOKEN')
        )

        context = next(iter(connections._connected_alias))
        products = os.getenv('PRODUCTS_COLLECTION')

        if products not in milvus.list_collections():
            products = Collection(products, schema = schema.create_schema(dim = output.shape[0], model = arch), using = context)
            products.create_index(
                schema.FIELDS['VECTOR'],
                schema.Params
            )
        else:
            products = Collection(products, using = context)

        products.load()
        print(f"Collection '{products.name}' loaded")

        self._collection = products

        self.preclassifier = preclassifier
        self.preprocess = preprocess
        self.device = device
        self.fx = fx

    def _vectorize(self, image):
        with torch.no_grad():
            batch = self.preprocess(image).unsqueeze(0).to(self.device)
            return self.fx(batch)[self.preclassifier].squeeze(0)

    def _vectorize_many(self, images):
        with torch.no_grad():
            batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
            return self.fx(batch)[self.preclassifier]
    
    def _get_partition(self, shelf, create_on_missing = False):
        partition = self._partitions.get(shelf)

        if not partition: # If partition not cached
            if not self._collection.has_partition(shelf) and create_on_missing:
                partition = Partition(self._collection, shelf) # Create partition
                self._partitions[shelf] = partition
            else:
                partition = self._collection.partition(shelf) # Retrieve partition
                self._partitions[shelf] = partition

        return partition

    def insert(self, shelf, image, sku = '', weight = 0):
        partition = self._get_partition(shelf, create_on_missing = True)
        vector = self._vectorize(image)

        partition.insert([[sku], [weight], [vector]])
        self._collection.flush()

    def insert_many(self, shelf, images, sku = '', weight = 0):
        partition = self._get_partition(shelf, create_on_missing = True)
        vector = self._vectorize_many(images)

        partition.insert([[sku], [weight], [vector]])
        self._collection.flush()

    def _query(self, partition, vector):
        return partition.search(
            [vector.cpu().numpy()],
            anns_field = schema.FIELDS['VECTOR'],
            param = schema.Params,
            limit = EmbedderResNet.K_MAX,
            output_fields = [schema.FIELDS['SKU'], schema.FIELDS['WEIGHT']]
        )[0]

    def search(self, shelf, image):
        partition = self._get_partition(shelf)

        if not partition:
            print(f"Partition '{shelf}' does not exist")
            return

        vector = self._vectorize(image)
        print(len(vector))
        return self._query(partition, vector)

    def search_many(self, shelf, images):
        partition = self._get_partition(shelf)

        if not partition:
            print(f"Partition '{shelf}' does not exist")
            return

        vector = self._vectorize_many(images).mean(0)
        return self._query(partition, vector)

    def get_products(self, shelf):
        partition = self._get_partition(shelf)

        if not partition:
            print(f"Partition '{shelf}' does not exist")
            return

        return partition.query('', output_fields = [schema.FIELDS['SKU'], schema.FIELDS['WEIGHT']], limit = partition.num_entities)

class EmbedderVIT:
    
    K_MAX = 5

    def __init__(self):
        self._collection = None
        self._partitions = {}

        self.device = None
        self._vit = None

    def initialise(self):
        dotenv.load_dotenv()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self._vit = VIT(self.device)

        milvus = MilvusClient(
            uri = os.getenv('MILVUS_URI'),
            token = os.getenv('MILVUS_TOKEN')
        )

        context = next(iter(connections._connected_alias))
        products = os.getenv('PRODUCTS_COLLECTION')

        if products not in milvus.list_collections():
            products = Collection(products, schema = schema.create_schema(dim = self._vit.dims[0], model = arch), using = context)
            products.create_index(
                schema.FIELDS['VECTOR'],
                schema.Params
            )
        else:
            products = Collection(products, using = context)

        products.load()
        print(f"Collection '{products.name}' loaded")

        self._collection = products

    def _vectorize(self, image):
        return self._vit.forward(image).squeeze().cpu()

    def _vectorize_many(self, images):
        return self._vit.forward_many(images).cpu()
    
    def _get_partition(self, shelf, create_on_missing = False):
        partition = self._partitions.get(shelf)

        if not partition: # If partition not cached
            if not self._collection.has_partition(shelf) and create_on_missing:
                partition = Partition(self._collection, shelf) # Create partition
                self._partitions[shelf] = partition
            else:
                partition = self._collection.partition(shelf) # Retrieve partition
                self._partitions[shelf] = partition

        return partition

    def insert(self, shelf, image, sku = '', weight = 0):
        partition = self._get_partition(shelf, create_on_missing = True)
        vector = self._vectorize(image)

        partition.insert([[sku], [weight], [vector]])
        self._collection.flush()

    def insert_many(self, shelf, images, sku = '', weight = 0):
        partition = self._get_partition(shelf, create_on_missing = True)
        vector = self._vectorize_many(images)

        partition.insert([[sku], [weight], [vector]])
        self._collection.flush()

    def _query(self, partition, vector):
        return partition.search(
            [vector.cpu().numpy()],
            anns_field = schema.FIELDS['VECTOR'],
            param = schema.Params,
            limit = EmbedderVIT.K_MAX,
            output_fields = [schema.FIELDS['SKU'], schema.FIELDS['WEIGHT']]
        )[0]

    def search(self, shelf, image):
        partition = self._get_partition(shelf)

        if not partition:
            print(f"Partition '{shelf}' does not exist")
            return

        vector = self._vectorize(image)
        print(len(vector))
        return self._query(partition, vector)

    def search_many(self, shelf, images):
        partition = self._get_partition(shelf)

        if not partition:
            print(f"Partition '{shelf}' does not exist")
            return

        vector = self._vectorize_many(images).mean(0)
        return self._query(partition, vector)

    def get_products(self, shelf):
        partition = self._get_partition(shelf)

        if not partition:
            print(f"Partition '{shelf}' does not exist")
            return

        return partition.query('', output_fields = [schema.FIELDS['SKU'], schema.FIELDS['WEIGHT']], limit = partition.num_entities)

if USE_VIT:
    Embedder = EmbedderVIT
else:
    Embedder = EmbedderResNet