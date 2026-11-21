# Run MobilityGPT on the SJSU HPC

This guide starts with the project on a Windows computer, uploads the source,
NYC files, and container image, then submits the full NYC training and
evaluation workflow through Slurm. Commands are labeled for **PowerShell**,
**PuTTY on coe-hpc1**, or **HPC3**. WSL alternatives are included in short
notes. Replace example paths and the SJSU ID with your own values.

The commands use `gpuqm` and one H100 because those resources were visible in
the `sinfo` output from this account. They requested up to 7 days in that
output, while this job asks for 48 hours. Check the live cluster again before
submitting because resource availability and account rules can change.

## 1. Get HPC access and connect

SJSU says CoE students should ask their professor or research advisor to
request HPC access. Connect from Eduroam/on-campus network, or connect to the
[SJSU VPN](https://phones.sjsu.edu/it/services/network/vpn.php) first. Follow
the [SJSU CoE HPC access guide](https://www.sjsu.edu/cmpe/current-students/hpc.php)
for account access and first-time HPC3 SSH setup.

In PuTTY, connect to `coe-hpc1.sjsu.edu` using your SJSU ID. If the guide's
first-time steps apply, create/configure your SSH key on coe-hpc1, then connect
from coe-hpc1 to `coe-hpc3` (also called `g17` in this account). Use the HPC3
terminal for the remaining cluster commands. Do not run training on the login
host.

On HPC3, confirm the tools, partition, and storage:

```bash
hostname
command -v sbatch srun sinfo apptainer
sinfo -o '%P %a %l %G %c %m %N'
quota -s
df -h "$HOME"
```


## 2. Set the Windows project folder

On your Windows computer, open PowerShell and set `$Baseline` to the outer
MobilityGPT directory: the one containing `Dockerfile`, `docker-compose.yml`,
`repo`, and `hpc`. Quotes allow spaces in the path.

```powershell
$Baseline = 'C:\Users\YOUR_WINDOWS_NAME\source\mobilitygpt'
Get-ChildItem $Baseline
```

Confirm it shows `Dockerfile`, `docker-compose.yml`, `repo`, and `hpc`.
The paths have different meanings throughout this guide:

- `C:\...` is a Windows path used by PowerShell and PSCP.
- `/mnt/c/...` is the same drive as seen from WSL.
- `~/mobilitygpt-hpc/...` is a path on HPC3.
- `/workspace/mobilitygpt/repo/...` is a path inside the container.

## 3. Build and export the container image

In PowerShell, with Docker Desktop running:

```powershell
Set-Location $Baseline
docker compose build
docker image ls mobilitygpt
docker save mobilitygpt:latest -o "$env:TEMP\mobilitygpt.tar"
Get-Item "$env:TEMP\mobilitygpt.tar" | Select-Object FullName, Length
```

The image archive is large (about 4 GB in the tested setup). Wait for
`docker save` to finish and ensure the file has nonzero size.

**WSL alternative:** if Docker Desktop is available to Docker commands in
WSL, run `docker compose build` from the mounted project and export the image
to `/tmp/mobilitygpt.tar`. The source and data archives can also be created
from WSL as shown below.

```bash
cd '/mnt/c/Users/YOUR_WINDOWS_NAME/source/mobilitygpt'
docker compose build
docker save mobilitygpt:latest -o /tmp/mobilitygpt.tar
```

## 4. Create separate source and NYC data archives

The source archive contains the code, Docker files, and HPC scripts. The NYC
files are ignored by Git and are deliberately excluded from the source and
Docker image, so a Git checkout alone does not provide the data. Package them
separately. This archive command follows the NYC symlinks so the archive
contains regular input files and does not depend on symlink support in Windows
or the HPC transfer path.

In PowerShell:

```powershell
$SourceArchive = "$env:TEMP\mobilitygpt-source.tar.gz"
$DataArchive = "$env:TEMP\mobilitygpt-nyc.tar.gz"

tar.exe -czf $SourceArchive `
  --exclude=./.venv --exclude=./mobilitygpt.egg-info `
  --exclude=./repo/NYC --exclude=./repo/nyc_workflow/output `
  --exclude=./logs --exclude=./.git --exclude=./*.sif --exclude=./*.tar `
  -C $Baseline .

tar.exe -czhf $DataArchive -C "$Baseline\repo" NYC
Get-Item $SourceArchive, $DataArchive | Select-Object FullName, Length
```

Confirm both files have nonzero size. The `-h` option on the data archive
dereferences symlinks, packaging their target contents as regular files.

**WSL alternative:** use a WSL terminal and the project path as mounted under
`/mnt/c`. For example:

```bash
BASELINE='/mnt/c/Users/YOUR_WINDOWS_NAME/source/mobilitygpt'
tar -czf /tmp/mobilitygpt-source.tar.gz \
  --exclude='./.venv' --exclude='./mobilitygpt.egg-info' \
  --exclude='./repo/NYC' --exclude='./repo/nyc_workflow/output' \
  --exclude='./logs' --exclude='./.git' --exclude='./*.sif' --exclude='./*.tar' \
  -C "$BASELINE" .
tar -czhf /tmp/mobilitygpt-nyc.tar.gz -C "$BASELINE/repo" NYC
cp /tmp/mobilitygpt-source.tar.gz /tmp/mobilitygpt.tar \
  /tmp/mobilitygpt-nyc.tar.gz /mnt/c/Users/YOUR_WINDOWS_NAME/Downloads/
```

The copy puts the archives in Windows Downloads so the next PSCP commands can
use them. Replace `YOUR_WINDOWS_NAME` with your Windows profile folder name.

## 5. Upload the archives with PuTTY PSCP

First, in your PuTTY session on **coe-hpc1**, create the upload directory:

```bash
mkdir -p "$HOME/mobilitygpt-upload"
ls -ld "$HOME/mobilitygpt-upload"
```

In PowerShell, set the paths to the files you created. The example uses the
Windows temporary directory:

```powershell
$SourceArchive = "$env:TEMP\mobilitygpt-source.tar.gz"
$ImageArchive = "$env:TEMP\mobilitygpt.tar"
$DataArchive = "$env:TEMP\mobilitygpt-nyc.tar.gz"
$Pscp = 'C:\Program Files\PuTTY\pscp.exe'

& $Pscp $SourceArchive $ImageArchive $DataArchive `
  '018326048@coe-hpc1.sjsu.edu:./mobilitygpt-upload/'
```

If you used the WSL alternative, point the three variables at the files in
`$env:USERPROFILE\Downloads` instead. Enter your HPC password when prompted
and wait for PSCP to finish all transfers.

On **coe-hpc1**, verify the uploads:

```bash
ls -lh "$HOME/mobilitygpt-upload/"
```

Then check whether the same home directory is visible on HPC3. If it is, the
files are already available there. If not, from coe-hpc1 create a destination
on HPC3 and copy the files onward:

```bash
ssh coe-hpc3 'mkdir -p "$HOME/mobilitygpt-upload"'
scp "$HOME/mobilitygpt-upload/mobilitygpt-source.tar.gz" \
    "$HOME/mobilitygpt-upload/mobilitygpt.tar" \
    "$HOME/mobilitygpt-upload/mobilitygpt-nyc.tar.gz" \
    coe-hpc3:mobilitygpt-upload/
```

Reconnect to HPC3 if needed, then verify the files there:

```bash
ls -lh "$HOME/mobilitygpt-upload/"
```

## 6. Extract files and create the Apptainer image

On **HPC3**, unpack source and data into `~/mobilitygpt-hpc`:

```bash
mkdir -p "$HOME/mobilitygpt-hpc"
tar -xzf "$HOME/mobilitygpt-upload/mobilitygpt-source.tar.gz" \
  -C "$HOME/mobilitygpt-hpc"
tar -xzf "$HOME/mobilitygpt-upload/mobilitygpt-nyc.tar.gz" \
  -C "$HOME/mobilitygpt-hpc/repo"
cd "$HOME/mobilitygpt-hpc"
mkdir -p logs
ls -l Dockerfile docker-compose.yml hpc/train_nyc.sbatch repo/pyproject.toml
ls -lh repo/NYC
```

The project data folder is now `~/mobilitygpt-hpc/repo/NYC`. Check that it
contains geography, relations, and both train/test CSV files. These filenames
are accepted as aliases:

| Input | Accepted alias |
|---|---|
| `nyc.geo` | `roadmap.geo` |
| `nyc.rel` | `roadmap.rel` |
| `nyc_mm_train.csv` | `NYC_Taxi_trajectory_train.csv` |
| `nyc_mm_test.csv` | `NYC_Taxi_trajectory_test.csv` |

The archive command follows local symlinks; files extracted on HPC should be
regular files. If an input is already stored in another cluster-visible
directory, you can use that directory later via `INPUT_DIR` instead.

Apptainer conversion may be restricted to compute nodes or disabled by local
policy. The SJSU HPC page describes Slurm GPU allocations but does not specify
whether image builds are allowed on login or compute nodes. Ask the HPC
administrator if conversion policy is unclear.

If conversion on a GPU allocation is permitted, request the resources shown
by your live `sinfo` output. From HPC3:

```bash
srun --partition=gpuqm --gres=gpu:h100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=64G --time=01:00:00 --pty /bin/bash
```

On the allocated node, convert the uploaded Docker archive:

```bash
cd "$HOME/mobilitygpt-hpc"
apptainer build mobilitygpt.sif \
  "docker-archive://$HOME/mobilitygpt-upload/mobilitygpt.tar"
ls -lh mobilitygpt.sif
exit
```

If build is not allowed on the allocation, do not run it on the login node;
ask which approved host or process to use. Once the SIF exists, the job script
uses it from `~/mobilitygpt-hpc/mobilitygpt.sif`.

## 7. Set inputs and submit full training

Back on the **HPC3 login shell**, set the input path. This must be set again
after reconnecting because shell variables do not persist between sessions:

```bash
cd "$HOME/mobilitygpt-hpc"
export INPUT_DIR="$HOME/mobilitygpt-hpc/repo/NYC"
ls -lh "$INPUT_DIR"
mkdir -p logs
```

If data is in another HPC-visible location, substitute its full path, for
example `export INPUT_DIR=/shared/project/NYC`. Do not use a Windows `C:\...`
path here; the script runs on HPC and binds this host directory into the
container at `/workspace/mobilitygpt/repo/NYC`.

Submit the full training and evaluation workflow with the settings confirmed
in your `sinfo` output:

```bash
sbatch --partition=gpuqm --gres=gpu:h100:1 hpc/train_nyc.sbatch
```

The script also requests 8 CPUs, 64 GB RAM, and 48 hours. The logged
configuration you supplied showed `gpuqm` allowing seven days and one H100.
The four inputs are checked before execution, and the container checks CUDA
before launching the NYC workflow. The workflow creates a fresh run directory
under `repo/nyc_workflow/output/` by default.