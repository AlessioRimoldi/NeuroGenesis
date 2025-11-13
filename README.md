# NeuroGenesis
An implementation of the contents and ideas from the book "A Brief history of Intelligence" by Max Bennett
Think of Neurogenesis as a brain simulator that learns inside video-game worlds.

We want to :
- try different algorithms ("the five breakthroughs")
- Try different games (Unity, Godot, Gym, etc.)

The architecture is built like LEGO blocks, every piece can be swapped.

## Folder Structure
```
neurogenesis
|____proto/ : .proto files for Env and service messages
|____envs/ : Video game engines wrappers (for example Godot or Gymnasium)
|____trainer/ 
|    |____algos/ : folder containing the algorithms implementations
|    |____modules/: word models folder
|    |____runners/: folder for specific pipelines rx. rollout.py
|    |____buffers/: Bufferes for experience replay Redis/RocksDB
|    |____config/: Hydra configs for Algo/env/task/compute
|    services/
|    |____dashboard/: Streamit/FastAPI
|    |____registry/: MLflow server bootstrap
|____scripts/: launch scripts, profiling, dataset export
|____docker/: Dockerfiles
|____tests/: self explanatory
|____res/: resources images etc.
|____.devcontainer: VS Code devcontainer
```

## Architecture 
![image](./res/architecture.png)