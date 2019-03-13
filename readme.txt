1，生成数据的tfrecord模式
python src/create_face_obj_tf_record.py --label_map_path=data/face_label_map.pbtxt --data_dir=../data/multi_obj --output_dir=data

2，编译protobuf文件（只需运行一次就行，下次训练就不用了）
cd  tensorflow的models/research
protoc object_detection/protos/*.proto --python_out=.

3，nohup python object_detection/model_main.py \
--pipeline_config_path=/home/tensortec/riki/workspace/pro/object_detection/ssd/face_tiny/data/ssd_mobilenet_v1_gray_0.25_face.config \
--model_dir=/home/tensortec/riki/workspace/pro/object_detection/ssd/face_tiny/model \
--num_train_steps=50000 \
--sample_1_of_n_eval_examples=1 \
--alsologtostderr &

4，export frozen graph
python object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=/home/tensortec/riki/workspace/pro/object_detection/ssd/face_tiny/data/ssd_mobilenet_v1_gray_0.25_face.config \
--trained_checkpoint_prefix=/home/tensortec/riki/workspace/pro/object_detection/ssd/face_tiny/model/model.ckpt-0 \
--output_directory=/home/tensortec/riki/workspace/pro/object_detection/ssd/face_tiny/exported_model \
--add_postprocessing_op=true

4.1 export predict graph
python object_detection/export_inference_graph.py  \
--input_type=image_tensor \
--pipeline_config_path=/home/tensortec/riki/workspace/pro/object_detection/ssd/face_tiny/data/ssd_mobilenet_v1_gray_0.25_face.config \
--trained_checkpoint_prefix=/home/tensortec/riki/workspace/pro/object_detection/ssd/face_tiny/model/model.ckpt-50000 \
--output_directory=/home/tensortec/riki/workspace/pro/object_detection/ssd/face_tiny/predict_model


5，转换成frozen
bazel run --config=opt tensorflow/lite/toco:toco -- \
--input_file=/home/tensortec/riki/workspace/pro/object_detection/ssd/face_tiny/exported_model/tflite_graph.pb \
--output_file=/home/tensortec/riki/workspace/pro/object_detection/ssd/face_tiny/tflite/detect.tflite \
--input_shapes=1,300,300,1 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops \
--default_ranges_min=0 \
--default_ranges_max=6
