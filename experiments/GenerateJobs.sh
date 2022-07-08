#!/bin/bash
num_nodes_exp="0 2 4 6" 
n_exp_per_node="20"
m_exp_per_node="25"
time_limit="30:00"
num_cores="64"
binary="../build/SparseA2ABenchmark"
submit_file="submit.sh"
output_file="results.csv"

# Find benchmark binary
if [[ ! -f "$binary" ]]; then 
    echo "Error: benchmark binary does not exist"
    exit 1
fi

create_script() {
    filename="$1"
    rm -rf "$filename"
    touch "$filename"
    chmod +x "$filename"
    echo "#!/bin/bash" >> "$filename"
}

create_submit_file_contents() {
    num_nodes="$1"
    output="$2"

    echo "#SBATCH --nodes=${num_nodes}"
    echo "#SBATCH --ntasks=$((num_cores*num_nodes))"
    echo "#SBATCH --cpus-per-task=1"
    echo "#SBATCH --ntasks-per-node=${num_cores}"
    echo "#SBATCH --time=${time_limit}"
    echo "#SBATCH --export=ALL"
    echo "#SBATCH --mem=230gb"
    echo "module load mpi/openmpi/4.0"

    n_exp=$((num_nodes*n_exp_per_node))
    m_exp=$((num_nodes*m_exp_per_node))

    echo "mpirun -n $((num_cores*num_nodes)) --bind-to core ./$binary $n_exp $m_exp >> $output_file"
}

create_script "$submit_file"
for num_nodes_exp in $num_nodes_exp; do 
    num_nodes=$((2**num_nodes_exp))
    this_submit_file="$submit_file.$num_nodes"
    this_output_file="$output_file.$num_nodes"

    create_script "$this_submit_file"
    create_submit_file_contents "$num_nodes" "$this_output_file" >> "$this_submit_file"
    echo "./$this_submit_file" >> "$submit_file"
done


