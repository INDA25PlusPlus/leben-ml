MODE=${1:-cpu_only}

case $MODE in
  cpu_only)
    TARGET=leben_ml_cpu_only
    ;;
  cuda)
    TARGET=leben_ml_cuda
    ;;
  *)
    echo "Invalid argument: '$MODE'"
    exit 1
    ;;
esac

cmake --build build/ --target $TARGET
./build/$TARGET
