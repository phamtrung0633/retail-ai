from pymilvus import CollectionSchema, FieldSchema, DataType

FIELDS = {
    'VECTOR': 'embedding',
    'WEIGHT': 'weight',
    'SKU': 'sku'
}

def create_schema(dim, model):
    return CollectionSchema(
        fields = [
            FieldSchema(
                name = 'id',
                dtype = DataType.INT64,
                is_primary = True
            ),
            FieldSchema(
                name = FIELDS['SKU'],
                dtype = DataType.VARCHAR,
                max_length = 64
            ),
            FieldSchema(
                name = FIELDS['WEIGHT'],
                dtype = DataType.FLOAT
            ),
            FieldSchema(
                name = FIELDS['VECTOR'],
                dtype = DataType.FLOAT_VECTOR,
                dim = dim
            )
        ],
        description = f'Product embeddings and associated SKUs ({model})',
        enable_dynamic_field = True,
        auto_id = True,
    )

Params = {
    'metric_type': 'COSINE',
    'index_type': 'FLAT'
}