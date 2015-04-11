#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void SelectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	block_size_ = this->layer_param_.select_param().block_size();
	CHECK(bottom[0]->count(1) % block_size_ == 0);
	CHECK_EQ(bottom[1]->count(), bottom[1]->shape(0));
	CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
	block_num_ = bottom[0]->count(1) / block_size_;
	select_type_ = this->layer_param_.select_param().select_type();
}

template<typename Dtype>
void SelectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK(bottom[0]->count(1) % block_size_ == 0);
	CHECK_EQ(bottom[1]->count(), bottom[1]->shape(0));
	CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
	block_num_ = bottom[0]->count(1) / block_size_;
	top[0]->ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void SelectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data_a = bottom[0]->cpu_data();
	const Dtype* bottom_data_b = bottom[1]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	int count = bottom[0]->count();
	int dim = bottom[0]->count(1);
	switch (select_type_) {
	case SelectParameter_SelectType_INCLUDE:
		caffe_set(count, Dtype(0), top_data);
		for (int i = 0; i < bottom[0]->shape(0); ++i) {
			int block_offset = static_cast<int>(bottom_data_b[i]) * block_size_;
			caffe_copy(block_size_, bottom_data_a + i * dim + block_offset,
					top_data + i * dim + block_offset);
		}
		break;
	case SelectParameter_SelectType_EXCLUDE:
		caffe_copy(count, bottom_data_a, top_data);
		for (int i = 0; i < bottom[0]->shape(0); ++i) {
			int block_offset = static_cast<int>(bottom_data_b[i]) * block_size_;
			caffe_set(block_size_, Dtype(0), top_data + i * dim + block_offset);
		}
		break;
	default:
		LOG(FATAL)<< "Unknown select type.";
	}

}

template<typename Dtype>
void SelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* bottom_data_b = bottom[1]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	int count = top[0]->count();
	int dim = top[0]->count(1);
	switch (select_type_) {
	case SelectParameter_SelectType_INCLUDE:
		caffe_set(count, Dtype(0), bottom_diff);
		for (int i = 0; i < top[0]->shape(0); ++i) {
			int block_offset = static_cast<int>(bottom_data_b[i]) * block_size_;
			caffe_copy(block_size_, top_diff + i * dim + block_offset,
					bottom_diff + i * dim + block_offset);
		}
		break;
	case SelectParameter_SelectType_EXCLUDE:
		caffe_copy(count, top_diff, bottom_diff);
		for (int i = 0; i < top[0]->shape(0); ++i) {
			int block_offset = static_cast<int>(bottom_data_b[i]) * block_size_;
			caffe_set(block_size_, Dtype(0), bottom_diff + i * dim + block_offset);
		}
		break;
	default:
		LOG(FATAL)<< "Unknown select type.";
	}
}

#ifdef CPU_ONLY
STUB_GPU(SelectLayer);
#endif

INSTANTIATE_CLASS(SelectLayer);
REGISTER_LAYER_CLASS(Select);

}  // namespace caffe
