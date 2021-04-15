

echo "JN, ARGS" > 2_batch_size.csv
parallel echo {0#}, --batch_size {1} --buffer_capacity {2} ::: {4..400..4} ::: {400..40000..400} >> 2_batch_size.csv

echo "JN, ARGS" > 2_model_shape.csv
parallel echo {0#}, --actor_layer_width {1} --actor_num_layers {2} ::: {2..1024..2} ::: {1..20..1} >> 2_model_shape.csv

echo "JN, ARGS" > 2_noise_params.csv
parallel echo {0#}, --std {1} --theta {2} ::: 0.0{1..9} 0.{10..99} 1.00 ::: 0.0{1..9} 0.{10..99} 1.00 >> 2_noise_params.csv

echo "JN, ARGS" > 2_update_params.csv
parallel echo {0#}, --update_freq {1} --tau {2} ::: {1..25} ::: 0.00{1..9} 0.0{10..99} 0.{100..400} >> 2_update_params.csv

