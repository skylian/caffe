#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template<typename Dtype>
void ScalingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->count(3), 1);
	CHECK_EQ(bottom[1]->count(2), 1);
	CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
	CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(1));
}

template<typename Dtype>
void ScalingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->count(3), 1);
	CHECK_EQ(bottom[1]->count(2), 1);
	CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
	CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(1));

	vector<int> top_shape;
	top_shape.push_back(bottom[0]->shape(0));
	top_shape.push_back(bottom[0]->shape(1));
	top[0]->Reshape(top_shape);
}

template<typename Dtype>
void ScalingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* A = bottom[0]->cpu_data();
	const Dtype* W = bottom[1]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	int num = bottom[0]->shape(0);
	int K = bottom[0]->shape(1);
	int L = bottom[0]->shape(2);
	for (int i = 0; i < num; ++i) {
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K, 1, L, (Dtype)1.,
				A+i*K*L, W+i*L, (Dtype)0., top_data+i*K);
	}
}

template<typename Dtype>
void ScalingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* A = bottom[0]->cpu_data();
	const Dtype* W = bottom[1]->cpu_data();
	Dtype* A_diff = bottom[0]->mutable_cpu_diff();
	Dtype* W_diff = bottom[1]->mutable_cpu_diff();
    // Gradient with respect to weight
	int num = bottom[0]->shape(0);
	int K = bottom[0]->shape(1);
	int L = bottom[0]->shape(2);
	if (propagate_down[0]) {
		for (int i = 0; i < num; ++i) {
			caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, L, 1, K, (Dtype)1.,
					A+i*L*K, top_diff+i*K, (Dtype)0., W_diff+i*L);
		}
	}
	if (propagate_down[1]) {
		for (int n = 0; n < num; ++n, W+=L) {
			for (int k = 0; k < K; ++k, A_diff+=L, top_diff++) {
				caffe_copy<Dtype>(L, W, A_diff);
				caffe_scal(L, *top_diff, A_diff);
			}
		}
    }
}

#ifdef CPU_ONLY
STUB_GPU(SelectLayer);
#endif

INSTANTIATE_CLASS(ScalingLayer);
REGISTER_LAYER_CLASS(Scaling);

}  // namespace caffe
