import os
import yaml
from src.machinelearningproject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError


@ensure_annotstions
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Args:
        path_to_yaml (str): path like input
    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type

    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

    @ensure_annotations
    def create_directories(path_to_directories:list,verbose=True):
        """ create list of dictionaries
        
        Args:
            path_to_directories (list): list of path of directories
            ignore_log(bool,optional): ignore if multiple dirs is to be created.Defaults to """
        
        for path in path_to_directories:
            os.makedirs(path,exist_ok=True)
            if verbose:
                logger.info(f"created directory at {path} .")

    
    @ensure_annotations
    def save_jason(path: Path,data : dict):
        """ save jason data
        
        Args:
            path (Path): path to jason file
            data (path): data to be saved in jason file
        """
        with open(path,'w') as f:
            json.dump(data,f,indent=4)
        
        logger.info(f"json file save at:{path}")

    @ensure_annotations
    def load_json(path: path) -> ConfigBox:
        """Load jason file data
        Args: 
            path (path): path to jason file
        
        returns:
            configbox: data as class attributes instead of dict
        """

        with open(path) as f:
            content = json.load(f)

        logger.info(f"json file loaded succesfully from: {path}")
        return ConfigBox(content)    

    @ensure_annotations
    def save_bin(data: Any, path : Path):
        """Save binary file
        Args:
            data(Any): data to be saved as binary
            path(path): path to binary file
        """ 

        joblib.dump(value=data,filename=path)
        logger.info(f"binary file saved at {path}")

    @ensure_annotations
    def load_bin(path: Path) -> Any:
        """load binary data
        
        Args:
            path (Path): path to binary file
        
        Returns:
            Any: object stored in the file
        """

        data = joblib.load(path)
        logger.info(f"binary file loaded drom : {path}")
        return data