#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/matrix.h>
#include <dlib/dnn.h>
#include <cstring>
#include <string>
#include <vector>

#include "facerec.h"

// Face recognition network definition (ResNet)
template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
    alevel0<alevel1<alevel2<alevel3<alevel4<dlib::max_pool<3, 3, 2, 2,
    dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2, dlib::input_rgb_image_sized<150>>>>>>>>>>>>>;

// Internal face recognizer struct
struct FaceRecognizer {
    std::string error_msg;
    std::string model_dir;

    dlib::frontal_face_detector hog_detector;
    dlib::shape_predictor shape_predictor_68;
    dlib::shape_predictor shape_predictor_5;
    anet_type face_encoder;

    bool hog_loaded;
    bool sp68_loaded;
    bool sp5_loaded;
    bool encoder_loaded;

    FaceRecognizer() : hog_loaded(false), sp68_loaded(false), sp5_loaded(false),
                       encoder_loaded(false) {}
};

// Convert Go image to dlib matrix
dlib::matrix<dlib::rgb_pixel> image_to_matrix(const image& img) {
    dlib::matrix<dlib::rgb_pixel> mat(img.height, img.width);

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            int idx = y * img.stride + x * 3;
            mat(y, x) = dlib::rgb_pixel(
                img.data[idx],
                img.data[idx + 1],
                img.data[idx + 2]
            );
        }
    }

    return mat;
}

extern "C" {

facerec facerec_init(const char* model_dir) {
    FaceRecognizer* rec = new FaceRecognizer();
    rec->model_dir = std::string(model_dir);

    try {
        // Load HOG detector (built-in, no model file needed)
        rec->hog_detector = dlib::get_frontal_face_detector();
        rec->hog_loaded = true;

        // Load 68-point shape predictor
        std::string sp68_path = rec->model_dir + "shape_predictor_68_face_landmarks.dat";
        dlib::deserialize(sp68_path) >> rec->shape_predictor_68;
        rec->sp68_loaded = true;

        // Try to load 5-point shape predictor (optional)
        try {
            std::string sp5_path = rec->model_dir + "shape_predictor_5_face_landmarks.dat";
            dlib::deserialize(sp5_path) >> rec->shape_predictor_5;
            rec->sp5_loaded = true;
        } catch (...) {
            // 5-point model is optional
        }

        // Load face recognition model
        std::string encoder_path = rec->model_dir + "dlib_face_recognition_resnet_model_v1.dat";
        dlib::deserialize(encoder_path) >> rec->face_encoder;
        rec->encoder_loaded = true;

    } catch (const std::exception& e) {
        rec->error_msg = e.what();
    }

    return static_cast<facerec>(rec);
}

void facerec_free(facerec handle) {
    if (handle) {
        FaceRecognizer* rec = static_cast<FaceRecognizer*>(handle);
        delete rec;
    }
}

const char* facerec_get_error(facerec handle) {
    if (!handle) return "null handle";

    FaceRecognizer* rec = static_cast<FaceRecognizer*>(handle);
    if (rec->error_msg.empty()) {
        return nullptr;
    }

    char* err = static_cast<char*>(malloc(rec->error_msg.size() + 1));
    strcpy(err, rec->error_msg.c_str());
    return err;
}

void facerec_free_error(const char* err) {
    if (err) {
        free(const_cast<char*>(err));
    }
}

rect* facerec_detect(facerec handle, image img, int upsample_times, int use_cnn, int* num_faces) {
    *num_faces = 0;
    if (!handle) return nullptr;

    FaceRecognizer* rec = static_cast<FaceRecognizer*>(handle);

    try {
        auto mat = image_to_matrix(img);
        std::vector<dlib::rectangle> dets;

        // Use HOG detector (CNN not supported in this simplified version)
        if (rec->hog_loaded) {
            dets = rec->hog_detector(mat, upsample_times);
        } else {
            return nullptr;
        }

        if (dets.empty()) {
            return nullptr;
        }

        *num_faces = static_cast<int>(dets.size());
        rect* rects = static_cast<rect*>(malloc(sizeof(rect) * dets.size()));

        for (size_t i = 0; i < dets.size(); i++) {
            rects[i].left = dets[i].left();
            rects[i].top = dets[i].top();
            rects[i].right = dets[i].right();
            rects[i].bottom = dets[i].bottom();
        }

        return rects;

    } catch (const std::exception& e) {
        rec->error_msg = e.what();
        return nullptr;
    }
}

point* facerec_landmarks(facerec handle, image img, rect* faces, int num_faces, int use_small) {
    if (!handle || !faces || num_faces <= 0) return nullptr;

    FaceRecognizer* rec = static_cast<FaceRecognizer*>(handle);

    try {
        auto mat = image_to_matrix(img);

        dlib::shape_predictor* predictor;
        int points_per_face;

        if (use_small && rec->sp5_loaded) {
            predictor = &rec->shape_predictor_5;
            points_per_face = 5;
        } else if (rec->sp68_loaded) {
            predictor = &rec->shape_predictor_68;
            points_per_face = 68;
        } else {
            return nullptr;
        }

        point* landmarks = static_cast<point*>(malloc(sizeof(point) * num_faces * points_per_face));

        for (int i = 0; i < num_faces; i++) {
            dlib::rectangle face_rect(
                faces[i].left,
                faces[i].top,
                faces[i].right,
                faces[i].bottom
            );

            auto shape = (*predictor)(mat, face_rect);

            for (int j = 0; j < points_per_face; j++) {
                landmarks[i * points_per_face + j].x = shape.part(j).x();
                landmarks[i * points_per_face + j].y = shape.part(j).y();
            }
        }

        return landmarks;

    } catch (const std::exception& e) {
        rec->error_msg = e.what();
        return nullptr;
    }
}

double* facerec_encode(facerec handle, image img, point* landmarks, int num_faces, int points_per_face, int num_jitters) {
    if (!handle || !landmarks || num_faces <= 0) return nullptr;

    FaceRecognizer* rec = static_cast<FaceRecognizer*>(handle);

    if (!rec->encoder_loaded) return nullptr;

    try {
        auto mat = image_to_matrix(img);

        double* encodings = static_cast<double*>(malloc(sizeof(double) * num_faces * 128));

        for (int i = 0; i < num_faces; i++) {
            // Build full_object_detection from landmarks
            std::vector<dlib::point> parts;
            for (int j = 0; j < points_per_face; j++) {
                parts.push_back(dlib::point(
                    landmarks[i * points_per_face + j].x,
                    landmarks[i * points_per_face + j].y
                ));
            }

            // Find bounding rectangle
            long min_x = parts[0].x(), max_x = parts[0].x();
            long min_y = parts[0].y(), max_y = parts[0].y();
            for (const auto& p : parts) {
                if (p.x() < min_x) min_x = p.x();
                if (p.x() > max_x) max_x = p.x();
                if (p.y() < min_y) min_y = p.y();
                if (p.y() > max_y) max_y = p.y();
            }

            dlib::rectangle rect(min_x, min_y, max_x, max_y);
            dlib::full_object_detection shape(rect, parts);

            // Extract aligned face chip
            dlib::matrix<dlib::rgb_pixel> face_chip;
            dlib::extract_image_chip(mat, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);

            // Compute descriptor
            dlib::matrix<float, 0, 1> face_descriptor = rec->face_encoder(face_chip);

            // Copy to output
            for (int j = 0; j < 128; j++) {
                encodings[i * 128 + j] = static_cast<double>(face_descriptor(j));
            }
        }

        return encodings;

    } catch (const std::exception& e) {
        rec->error_msg = e.what();
        return nullptr;
    }
}

} // extern "C"
