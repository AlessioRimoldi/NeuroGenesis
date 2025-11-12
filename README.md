# NeuroGenesis
An implementation of the contents and ideas from the book "A Brief history of Intelligence" by Max Bennett
Think of Neurogenesis as a brain simulator that learns inside video-game worlds.

We want to :
    - try different algorithms ("the five breakthroughs")
    - Try different games (Unity, Godot, Gym, etc.)

The architecture is built like LEGO blocks, every piece can be swapped.

## Folder Structure

proto/ : .proto files for Env and service messages
envs/ : Video game engines wrappers (for example Godot or Gymnasium)
trainer/ 
    algos/ : folder containing the algorithms implementations
    modules/: word models folder
    runners/: folder for specific pipelines rx. rollout.py
    buffers/: Bufferes for experience replay Redis/RocksDB
    config/: Hydra configs for Algo/env/task/compute
services/
    dashboard/: Streamit/FastAPI
    registry/: MLflow server bootstrap
scripts/: launch scripts, profiling, dataset export
docker/: Dockerfiles
tests/: self explanatory
res/: resources images etc.
.devcontainer: VS Code devcontainer


## Architecture 
![image](./res/architecture.png)