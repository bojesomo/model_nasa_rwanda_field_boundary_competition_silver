{{

A template repository for a ML model to be published on
[Radiant MLHub](https://mlhub.earth/models).


# BorderAttention: a segmentation model for fields
In the [NASA Harvest Field Boundary Detection Challenge](https://zindi.africa/competitions/nasa-harvest-field-boundary-detection-challenge/leaderboard)
this was the second place solution by the team `HungryLearner`.

## ML Model Documentation

Please review the model architecture, license, applicable spatial and temporal extents
and other details in the [model documentation](/docs/index.md).

## System Requirements

* Git client
* [Docker](https://www.docker.com/) with
    [Compose](https://docs.docker.com/compose/) v1.28 or newer.

## Hardware Requirements

|Inferencing|Training|
|-----------|--------|
|16GB RAM | 16GB RAM|
|           | NVIDIA GPU |

## Get Started With Inferencing

First clone this Git repository.

Please note: this repository uses
[Git Large File Support (LFS)](https://git-lfs.github.com/) to include the
model checkpoint file. Either install `git lfs` support for your git client,
use the official Mac or Windows GitHub client to clone this repository.

:zap: Shell commands have been tested with Linux and MacOS but will
differ on Windows, or depending on your environment.

```bash
git clone https://github.com/radiantearth/model_nasa_rwanda_field_boundary_competition_bronze.git
cd model_nasa_rwanda_field_boundary_competition_bronze/
```

After cloning the model repository, you can use the Docker Compose runtime
files as described below.

## Pull or Build the Docker Image

Pull pre-built image from Docker Hub (recommended):

```bash
docker pull docker.io/radiantearth/model_nasa_rwanda_field_boundary_competition_silver:1
```

Or build image from source:

```bash
cd docker-services/
docker build -t radiantearth/model_nasa_rwanda_field_boundary_competition_silver:1 .
## Run Model to Generate New Inferences

{{

:pushpin: Model developer: do not commit training data to the data folder on
this repo, this is only a placeholder to run the model locally for inferencing.

}}

1. Prepare your input and output data folders. The `data/` folder in this repository
    contains some placeholder files to guide you.

    * The `data/` folder must contain:
        * `input/chips` {{ Landsat, Maxar Open-Data 30cm, Sentinel-2, etc. }} imagery chips for inferencing:
            * File name: {{ `chip_id.tif` }} e.g. {{ `0fec2d30-882a-4d1d-a7af-89dac0198327.tif` }}
            * File Format: {{ GeoTIFF, 256x256 }}
            * Coordinate Reference System: {{ WGS84, EPSG:4326 }}
            * Bands: {{ 3 bands per file:
                * Band 1 Type=Byte, ColorInterp=Red
                * Band 2 Type=Byte, ColorInterp=Green
                * Band 3 Type=Byte, ColorInterp=Blue
                }}
        * `/input/checkpoint` the model checkpoint {{ file | folder }}, `{{ checkpoint file or folder name }}`.
            Please note: the model checkpoint is included in this repository.
    * The `output/` folder is where the model will write inferencing results.

2. Set `INPUT_DATA` and `OUTPUT_DATA` environment variables corresponding with
    your input and output folders. These commands will vary depending on operating
    system and command-line shell:

    ```bash
    # change paths to your actual input and output folders
    export INPUT_DATA="/home/my_user/{{repository_name}}/data/input/"
    export OUTPUT_DATA="/home/my_user/{{repository_name}}/data/output/"
    ```

3. Run the appropriate Docker Compose command for your system

    ```bash
    # cpu
    docker compose up {{model_id}}_cpu
    # optional, for NVIDIA gpu driver
    docker compose up {{model_id}}_gpu
    ```

4. Wait for the `docker compose` to finish running, then inspect the
`OUTPUT_DATA` folder for results.

## Understanding Output Data

Please review the model output format and other technical details in the [model
documentation](/docs/index.md).
