#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template<typename TypeParam>
class ScalingLayerTest: public MultiDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;
protected:
	ScalingLayerTest() :
			blob_bottom_a_(new Blob<Dtype>(10, 9, 5, 1)), blob_bottom_b_(
					new Blob<Dtype>(10, 5, 1, 1)), blob_top_(new Blob<Dtype>()) {

		// fill the values
		Caffe::set_random_seed(1701);
		FillerParameter filler_param;
		UniformFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom_a_);
		filler.Fill(this->blob_bottom_b_);
		blob_bottom_vec_.push_back(blob_bottom_a_);
		blob_bottom_vec_.push_back(blob_bottom_b_);
		blob_top_vec_.push_back(blob_top_);
	}
	virtual ~ScalingLayerTest() {
		delete blob_bottom_a_;
		delete blob_bottom_b_;
		delete blob_top_;
	}
	Blob<Dtype>* const blob_bottom_a_;
	Blob<Dtype>* const blob_bottom_b_;
	Blob<Dtype>* const blob_top_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ScalingLayerTest, TestDtypesAndDevices);

TYPED_TEST(ScalingLayerTest, TestSetUp){
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	shared_ptr<ScalingLayer<Dtype> > layer(
			new ScalingLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	EXPECT_EQ(this->blob_top_->num(), 10);
	EXPECT_EQ(this->blob_top_->channels(), 9);
	EXPECT_EQ(this->blob_top_->height(), 1);
	EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(ScalingLayerTest, TestForward){
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
	#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
	#endif
	if (Caffe::mode() == Caffe::CPU ||
			sizeof(Dtype) == 4 || IS_VALID_CUDA) {
		LayerParameter layer_param;
		shared_ptr<ScalingLayer<Dtype> > layer(new ScalingLayer<Dtype>(layer_param));
		layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype *A = this->blob_bottom_a_->cpu_data();
		const Dtype *W = this->blob_bottom_b_->cpu_data();
		int K = this->blob_bottom_a_->shape(1);
		int L = this->blob_bottom_a_->shape(2);
		const Dtype* data = this->blob_top_->cpu_data();
		for (int n = 0; n < this->blob_bottom_a_->shape(0); ++n) {
			for (int k = 0; k < this->blob_bottom_a_->shape(1); ++k) {
				Dtype sum = 0;
				for (int l = 0; l < this->blob_bottom_a_->shape(2); ++l)
					sum += A[(n*K+k)*L+l] * W[n*L+l];
				EXPECT_NEAR(data[n*K+k], sum, 1e-5);
			}
		}
	} else {
		LOG(ERROR) << "Skipping test due to old architecture.";
	}
}

TYPED_TEST(ScalingLayerTest, TestGradient){
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
	#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
	#endif
	if (Caffe::mode() == Caffe::CPU ||
			sizeof(Dtype) == 4 || IS_VALID_CUDA) {
		LayerParameter layer_param;
		ScalingLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-2, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
				this->blob_top_vec_);
	} else {
		LOG(ERROR) << "Skipping test due to old architecture.";
	}
}
}  // namespace caffe
