#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void JitterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	NeuronLayer<Dtype>::LayerSetUp(bottom, top);
	jitter_radius_ = this->layer_param_.jitter_param().jitter_radius();
	jitter_ratio_ = this->layer_param_.jitter_param().jitter_ratio();
	CHECK_GE(jitter_radius_, 1);
	CHECK_LE(jitter_radius_, 10);
}

template<typename Dtype>
void JitterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	NeuronLayer<Dtype>::Reshape(bottom, top);
}

template<typename Dtype>
void JitterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();

	int dim = bottom[0]->count() / bottom[0]->num();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int channel = bottom[0]->channels();
	int spatial_dim = height * width;
	int r = 2 * jitter_radius_ + 1;
	int do_jitter;
	for (int i = 0; i < bottom[0]->num(); ++i) {
		caffe_rng_bernoulli(1, jitter_ratio_, &do_jitter);
		if (!do_jitter) {
			caffe_copy(dim, bottom_data + i * dim, top_data + i * dim);
		} else {
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					int id = y * width + x;

					int yy = y + ((int) caffe_rng_rand()) % r - jitter_radius_;
					int xx = x + ((int) caffe_rng_rand()) % r - jitter_radius_;
					while (!(xx >= 0 && xx < width && yy >= 0 && yy < height)) {
						yy = y + ((int) caffe_rng_rand()) % r - jitter_radius_;
						xx = x + ((int) caffe_rng_rand()) % r - jitter_radius_;
					}
					for (int c = 0; c < channel; ++c) {
						top_data[i * dim + spatial_dim * c + id] = bottom_data[i
								* dim + c * spatial_dim + yy * width + xx];
					}
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(JitterLayer);
#endif

INSTANTIATE_CLASS(JitterLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
