# BBrta

## Docker image

```bash
docker build -t bblocks:3.0.0 -f ./Dockerfile.matlab .

./bootstrap.sh bblocks:3.0.0 falco

docker run -it -d -v /home/falco/Documents/BBlocks/BBrta:/home/userbblocks/workspace/BBrta --name bblocks3 bblocks:3.0.0_falco /bin/bash
```

Finally exec the container

```bash
docker exec -it bblocks3 /bin/bash
```

In the container run

```bash
cd
./entrypoint.sh
```
