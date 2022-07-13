#!/bin/bash
num_nodes_exp="-1 1 3 5 7 9"
n_exp_per_node="15"
m_exp_per_node="20"
time_limit="2:00:00"
num_cores="32"
num_cores_one_node="16"
binary="../build/SparseA2ABenchmark"
working_dir="supermuc"
submit_file="submit.sh"
output_file="results.csv"

mkdir -p "$working_dir"

# Find benchmark binary
if [[ ! -f "$binary" ]]; then 
    echo "Error: benchmark binary does not exist"
    exit 1
fi

# Import secret variables
if [[ ! -f "secrets" ]]; then
    echo "Missing file 'secrets'"
    exit 1
fi
. secrets

compute_num_nodes() {
    num_nodes_exp="$1"
    if [[ "$num_nodes_exp" == "-1" ]]; then
        echo 1
    else
        echo $((2 ** num_nodes_exp))
    fi
}

create_script() {
    filename="$1"
    rm -rf "$filename"
    touch "$filename"
    chmod +x "$filename"
    echo "#!/bin/bash" >> "$filename"
}

create_submit_file_contents() {
    num_nodes_exp="$1"
    num_nodes=$(compute_num_nodes "$num_nodes_exp")
    this_output_file="$2"

    my_num_cores="$num_cores"
    if [[ "$num_nodes_exp" == "-1" ]]; then
	my_num_cores=16
    fi

    echo "#SBATCH -J SparseAlltoallBenchmark_${num_nodes_exp}"
    echo "#SBATCH -o ./%x.%j.out"
    echo "#SBATCH -e ./%x.%j.err"
    echo "#SBATCH -D ${PWD}"
    echo "#SBATCH --mail-type=END"
    echo "#SBATCH --mail-user=${supermuc_email}"
    echo "#SBATCH --time=${time_limit}"
    echo "#SBATCH --no-requeue"
    echo "#SBATCH --export=NONE"
    echo "#SBATCH --get-user-env"
    echo "#SBATCH --account=${supermuc_project}"
    if (( $num_nodes <= 16 )); then 
    	echo "#SBATCH --partition=micro"
    else 
        echo "#SBATCH --partition=general"
    fi
    echo "#SBATCH --nodes=${num_nodes}"
    echo "#SBATCH --ntasks=$((my_num_cores*num_nodes))"
    echo "#SBATCH --ntasks-per-node=${my_num_cores}"
    echo "#SBATCH --mem=80gb"
    echo "#SBATCH --ear=off"
   
    echo "module load openmpi/4.0.7-gcc11"
    echo "module load slurm_setup" 

    n_exp=$((num_nodes_exp+n_exp_per_node))
    m_exp=$((num_nodes_exp+m_exp_per_node))

    echo "mpiexec -n $((my_num_cores*num_nodes)) ./$binary $n_exp $m_exp 1>> $this_output_file"
}

create_script "$working_dir/$submit_file"
for num_nodes_exp in $num_nodes_exp; do 
    num_nodes=$(compute_num_nodes "$num_nodes_exp")
    this_submit_file="$working_dir/$submit_file.$num_nodes"
    this_output_file="$working_dir/$output_file.$num_nodes"

    create_script "$this_submit_file"
    create_submit_file_contents "$num_nodes_exp" "$this_output_file" >> "$this_submit_file"
    echo "sbatch $this_submit_file" >> "$working_dir/$submit_file"
done

