#!/bin/bash
#source venvTF24/bin/activate  # or run this from the terminal first manually
for i in {1..10}
  do
    echo "STARTING RUN $i"
    python cell_graph_GNN.py
  done
echo "--- ALL COMPLETE ---"