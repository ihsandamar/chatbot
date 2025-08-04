
# src/core/states/state_registry.py
from typing import Dict, Type, Optional, Set
from src.core.states.base_state import StateMetadata, ChatbotState, ValidationError
from src.core.states.state_adapter import StateAdapter
from src.services.app_logger import log
import inspect

class StateRegistry:
    """Central registry for all module states and their adapters"""
    
    def __init__(self):
        self._states: Dict[str, StateMetadata] = {}
        self._adapters: Dict[str, StateAdapter] = {}
        self.logger = log.get(module="state_registry")
    
    def register_state(self, 
                      name: str, 
                      state_class: type, 
                      adapter_class: type,
                      description: str = "",
                      version: str = "1.0.0") -> None:
        """Register a new state and its adapter"""
        
        # Validation

        if not issubclass(adapter_class, StateAdapter):
            raise ValidationError(f"Adapter class {adapter_class.__name__} must inherit from StateAdapter")
        
        # Get module path
        module_path = inspect.getmodule(state_class).__name__
        
        metadata = StateMetadata(
            name=name,
            description=description,
            version=version,
            module_path=module_path,
            state_class=state_class,
            adapter_class=adapter_class
        )
        
        self._states[name] = metadata
        self._adapters[name] = adapter_class(name)
        
        self.logger.info("State registered", 
                        state_name=name, 
                        module_path=module_path,
                        version=version)
    
    def get_adapter(self, state_name: str) -> Optional[StateAdapter]:
        """Get adapter for a specific state"""
        return self._adapters.get(state_name)
    
    def get_state_metadata(self, state_name: str) -> Optional[StateMetadata]:
        """Get metadata for a specific state"""
        return self._states.get(state_name)
    
    def list_states(self) -> Set[str]:
        """List all registered state names"""
        return set(self._states.keys())
    
    def validate_compatibility(self, state_name: str) -> bool:
        """Validate if a state is compatible with current system"""
        metadata = self._states.get(state_name)
        if not metadata:
            return False
        
        try:
            # Try to instantiate the adapter
            adapter = self._adapters[state_name]
            return True
        except Exception as e:
            self.logger.error("State compatibility check failed", 
                            state_name=state_name, 
                            error=str(e))
            return False

# Global registry instance
state_registry = StateRegistry()

# Decorator for easy state registration
def register_state(name: str, description: str = "", version: str = "1.0.0"):
    """Decorator to register a state class and its adapter"""
    def decorator(adapter_class: Type[StateAdapter]):
        # Extract state class from adapter's generic type
        state_class = adapter_class.get_state_class()
        state_registry.register_state(
            name=name,
            state_class=state_class,
            adapter_class=adapter_class,
            description=description,
            version=version
        )
        return adapter_class
    return decorator
