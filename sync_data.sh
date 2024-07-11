#!/bin/bash
rm -rf /pmglocal/scGraphLLM/modeldata/
mkdir -p /pmglocal/scGraphLLM/modeldata/
cp /burg/pmg/collab/scGraphLLM/data/pilotdata_cache.tgz  /pmglocal/scGraphLLM/modeldata/
cd /pmglocal/scGraphLLM/modeldata/
tar -xzf pilotdata_cache.tgz
