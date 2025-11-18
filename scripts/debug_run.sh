MODE=${1:-cpu_only}

case $MODE in
  cpu_only)
    TARGET=leben_ml_cpu_only
    ;;
  cuda)
    TARGET=leben_ml_cuda
    ;;
  cpu_only_test)
    TARGET=leben_ml_cpu_only_test
    ;;
  cuda_test)
    TARGET=leben_ml_cuda_test
    ;;
  *)
    echo "Invalid argument: '$MODE'"
    exit 1
    ;;
esac

cmake --build build/ --target $TARGET
./build/$TARGET
