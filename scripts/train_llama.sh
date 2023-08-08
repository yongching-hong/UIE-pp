model_name_or_path=meta-llama/Llama-2-7b-chat-hf

python src/run_uie.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path ${model_name_or_path} \
   --data_dir ./data/ie_instruct \
   --task_config_dir ./configs/single_task_configs \
   --instruction_file ./configs/instruction_config.json \
   --prompt_file ./prompts/instructUIE.json \
   --instruction_strategy single \
   --min_negative_labels -1 \
   --min_positive_labels -1 \
   --output_dir ./output/ACE05/llama2-7B-chat_1 \
   --input_record_file flan-t5.record \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --gradient_accumulation_steps 8 \
   --learning_rate 5e-05 \
   --num_train_epochs 10 \
   --run_name ${model_name_or_path}-fewnerd \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --max_num_instances_per_task 10000 \
   --max_num_instances_per_eval_task 200 \
   --add_task_name False \
   --add_dataset_name False \
   --num_examples 0 \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy steps \
   --fp16 \
   --cache_dir ./huggingface \
   --ddp_find_unused_parameters False \
   --save_total_limit 1 \
   --save_steps 100 \
   --lora_target_modules q_proj,k_proj,v_proj,o_proj \
   --lora_r 16 \
   --use_auth_token True \