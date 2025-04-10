// src/components/SentimentCarousel.jsx
import { Swiper, SwiperSlide } from 'swiper/react';
import 'swiper/css';
import 'swiper/css/autoplay';
import { Autoplay } from 'swiper/modules';

const Carousel = () => {
  return (
    <div className="w-full max-w-4xl mx-auto mt-8">
      <Swiper
        spaceBetween={30}
        slidesPerView={1}
        loop={true}
        autoplay={{ delay: 3000 }}
        modules={[Autoplay]}
        className="rounded-2xl shadow-xl"
      >
        <SwiperSlide>
          <div className="bg-blue-600 text-white p-10 rounded-2xl text-center">
            <h2 className="text-2xl font-bold mb-2">Real-Time Sentiment Analysis</h2>
            <p>Dive into live Twitter data and spot trends as they happen.</p>
          </div>
        </SwiperSlide>

        <SwiperSlide>
          <div className="bg-green-500 text-white p-10 rounded-2xl text-center">
            <h2 className="text-2xl font-bold mb-2">Custom NLP Model</h2>
            <p>Analyze emotions using your trained machine learning model.</p>
          </div>
        </SwiperSlide>

        <SwiperSlide>
          <div className="bg-purple-500 text-white p-10 rounded-2xl text-center">
            <h2 className="text-2xl font-bold mb-2">Interactive Dashboard</h2>
            <p>Visualize sentiments with graphs, charts, and filters.</p>
          </div>
        </SwiperSlide>
      </Swiper>
    </div>
  );
};

export default Carousel;
