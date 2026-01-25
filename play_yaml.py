from pathlib import Path
from typing import TypeVar, Type, Any, Optional, Union
from pydantic import BaseModel, Field
import yaml

T = TypeVar('T', bound=BaseModel)

class YamlFileReader:
    """Handles YAML file I/O operations"""
    
    @staticmethod
    def read(file_path: Path) -> dict[str, Any]:
        """Read and parse YAML file"""
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = yaml.safe_load(file)
            if not isinstance(content, dict):
                raise ValueError("YAML root must be a dictionary")
            return content


# Single Responsibility: Path navigation
class YamlPathNavigator:
    """Navigates YAML structure using path notation"""
    
    @staticmethod
    def get_at_path(data: dict[str, Any], *path_parts: str) -> Any:
        """Navigate through nested dictionary using path parts"""
        current = data
        traversed = []
        
        for part in path_parts:
            traversed.append(part)
            if not isinstance(current, dict):
                raise ValueError(
                    f"Cannot navigate into non-dict at path: {'.'.join(traversed[:-1])}"
                )
            if part not in current:
                raise KeyError(
                    f"Key '{part}' not found at path: {'.'.join(traversed[:-1])}. "
                    f"Available keys: {', '.join(current.keys())}"
                )
            current = current[part]
        
        return current


# Single Responsibility: Config object mapping
class ConfigDeserializer:
    """Maps YAML data to Pydantic config objects"""
    
    def deserialize(self, data: Any, config_class: Type[T]) -> T:
        """Deserialize YAML data into a Pydantic model"""
        if not isinstance(data, dict):
            raise ValueError(
                f"Data must be a dictionary to deserialize into {config_class.__name__}, "
                f"got {type(data)}"
            )
        
        try:
            return config_class.model_validate(data)
        except Exception as e:
            raise ValueError(f"Failed to validate {config_class.__name__}: {e}")
    
    def deserialize_list(self, data: Any, item_class: Type[T]) -> list[T]:
        """Deserialize YAML data into a list of Pydantic models"""
        if not isinstance(data, list):
            raise ValueError(
                f"Data must be a list to deserialize into list[{item_class.__name__}], "
                f"got {type(data)}"
            )
        
        return [self.deserialize(item, item_class) for item in data]


# Main SOLID YamlLoader
class YamlLoader:
    """
    Loads YAML configuration files and deserializes data to Pydantic models.
    """
    
    def __init__(
        self, 
        config_path: Union[str, Path],
        file_reader: Optional[YamlFileReader] = None,
        navigator: Optional[YamlPathNavigator] = None,
        deserializer: Optional[ConfigDeserializer] = None
    ):
        """Initialize YamlLoader with a config file path"""
        self._config_path = Path(config_path)
        self._file_reader = file_reader or YamlFileReader()
        self._navigator = navigator or YamlPathNavigator()
        self._deserializer = deserializer or ConfigDeserializer()
        self._raw_data: Optional[dict[str, Any]] = None
        self._cache: dict[str, Union[BaseModel, list[BaseModel]]] = {}
    
    @property
    def config_file_path(self) -> Path:
        """Get the full path of the config file"""
        return self._config_path
    
    @property
    def config_file_name(self) -> str:
        """Get the name of the config file"""
        return self._config_path.name
    
    def _load_raw_data(self) -> dict[str, Any]:
        """Load raw YAML data from file (cached)"""
        if self._raw_data is None:
            self._raw_data = self._file_reader.read(self._config_path)
        return self._raw_data
    
    def _make_cache_key(self, config_class: Type[T], *path: str, is_list: bool = False) -> str:
        """Create a unique cache key"""
        prefix = "list:" if is_list else "single:"
        if path:
            return f"{prefix}{'.'.join(path)}.{config_class.__name__}"
        return f"{prefix}{config_class.__name__}"
    
    def get(self, config_class: Type[T], *path: str) -> T:
        """
        Get a config object by Pydantic model type and optional path.
        
        Args:
            config_class: The Pydantic model class to deserialize
            *path: Optional path components to navigate before deserializing
            
        Returns:
            Instance of the Pydantic model
            
        Examples:
            model: ModelConfig = loader.get(ModelConfig)
            db: DatabaseConfig = loader.get(DatabaseConfig)
        """
        cache_key = self._make_cache_key(config_class, *path, is_list=False)
        
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore
        
        raw_data = self._load_raw_data()
        
        if path:
            target_data = self._navigator.get_at_path(raw_data, *path)
        else:
            class_name = config_class.__name__
            if class_name not in raw_data:
                available_keys = ', '.join(raw_data.keys())
                raise KeyError(
                    f"Config key '{class_name}' not found in {self.config_file_name}. "
                    f"Available keys: {available_keys}"
                )
            target_data = raw_data[class_name]
        
        config = self._deserializer.deserialize(target_data, config_class)
        self._cache[cache_key] = config
        
        return config
    
    def get_list(self, item_class: Type[T], *path: str) -> list[T]:
        """
        Get a list of config objects by Pydantic model type and path.
        
        Args:
            item_class: The Pydantic model class to deserialize each item into
            *path: Path components to navigate to the list
            
        Returns:
            List of instances of the Pydantic model
            
        Examples:
            models: list[ModelConfig] = loader.get_list(ModelConfig, "Models")
        """
        cache_key = self._make_cache_key(item_class, *path, is_list=True)
        
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore
        
        raw_data = self._load_raw_data()
        
        if path:
            target_data = self._navigator.get_at_path(raw_data, *path)
        else:
            default_key = f"{item_class.__name__}s"
            if default_key not in raw_data:
                available_keys = ', '.join(raw_data.keys())
                raise KeyError(
                    f"List key '{default_key}' not found in {self.config_file_name}. "
                    f"Available keys: {available_keys}"
                )
            target_data = raw_data[default_key]
        
        config_list = self._deserializer.deserialize_list(target_data, item_class)
        self._cache[cache_key] = config_list
        
        return config_list
    
    def reload(self) -> None:
        """Clear cache and reload data from file"""
        self._raw_data = None
        self._cache.clear()
    
    def get_raw(self, *path: str) -> Any:
        """Get raw data at a specific path"""
        raw_data = self._load_raw_data()
        if not path:
            return raw_data
        return self._navigator.get_at_path(raw_data, *path)
    
    def available_keys(self, *path: str) -> list[str]:
        """Get list of available keys at a specific path"""
        data = self.get_raw(*path)
        if not isinstance(data, dict):
            raise ValueError(f"Path {'.'.join(path)} does not point to a dictionary")
        return list(data.keys())


# Example usage
if __name__ == "__main__":
    from pydantic import field_validator
    
    # Define your Pydantic config models
    class ModelConfig(BaseModel):
        model_name: str
        version: str
        max_tokens: int = Field(gt=0)
    
    class DatabaseConfig(BaseModel):
        host: str
        port: int = Field(ge=1, le=65535)
        database: str
        username: str
    
    class ServerConfig(BaseModel):
        name: str
        ip: str
        roles: list[str]
    
    # Create loader
    loader = YamlLoader("config.yaml")
    
    # The IDE will infer types based on the type annotation you provide!
    model = loader.get(ModelConfig)
    print(model.model_name)  # ✓ IDE knows model_name is str

    models = loader.get_list(ModelConfig, "Models")
    for m in models:
        print(m.version)  # ✓ IDE knows m is ModelConfig
    
    servers = loader.get_list(ServerConfig, "Infrastructure", "Servers")
    for s in servers:
        print(s.model_config)  # ✓ IDE knows s is ServerConfig