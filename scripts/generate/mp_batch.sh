
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--device)
      NUM_DEVICES="$2"
      shift # past argument
      shift # past value
      ;;
    -i|--input-file)
      INPUT_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output-folder)
      OUTPUT_FOLDER="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

echo "NUM DEVICES       = ${NUM_DEVICES}"
echo "INPUT FILE        = ${INPUT_FILE}"
echo "OUTPUT FOLDER     = ${OUTPUT_FOLDER}"

for i in $(seq 1 $NUM_DEVICES)
do
    echo "Generating MP data for device $i"
    python scripts/generate/generate_mp_data.py \
        -i $INPUT_FILE \
        -o "data" \
        -r $i -w $NUM_DEVICES
done


for i in $(seq 1 $NUM_DEVICES)
do
    echo "Generating Voice data for device $i"
    CUDA_VISIBLE_DEVICES=$(($i - 1)) python scripts/generate/generate_per_rank.py \
        -i "data/mp_data_$i.jsonl" \
        -o $OUTPUT_FOLDER \
        -of "processed_mp_data_$i.jsonl" &
done
