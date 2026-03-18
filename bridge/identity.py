from pipeline.knowledge_base import load_model_master, resolve_listing_identity


class IdentityEngine:
    def __init__(self, model_master_path: str):
        self.model_master_path = model_master_path
        self.models = load_model_master(model_master_path)

    def resolve(self, raw_text: str) -> dict:
        return resolve_listing_identity(raw_text, self.models)