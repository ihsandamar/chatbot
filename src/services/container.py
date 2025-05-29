# src/services/container.py
class ServiceContainer:
    def __init__(self):
        self.services = {}

    def register(self, key, provider):
        self.services[key] = provider

    def resolve(self, key):
        if key not in self.services:
            raise ValueError(f"Service {key} not registered")
        return self.services[key]()


