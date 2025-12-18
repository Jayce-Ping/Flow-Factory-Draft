accelerate launch \
    --config_file config/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 2 \
    -m flow_factory.cli \
    --config config/flux_t2is.yaml