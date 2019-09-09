// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

// #include "mvs/fusion.h"
#include "mvs/depthMapConverter.h"

#include "util/misc.h"

namespace colmap {
namespace mvs {

DataCollector::DataCollector(const StereoFusionOptions& options,
                           const std::string& workspace_path,
                           const std::string& workspace_format,
                           const std::string& pmvs_option_name,
                           const std::string& input_type)
    : StereoFusion(options, workspace_path, workspace_format, pmvs_option_name, input_type){}

void DataCollector::getData(){

  Workspace::Options workspace_options;

  auto workspace_format_lower_case = workspace_format_;
  StringToLower(&workspace_format_lower_case);
  if (workspace_format_lower_case == "pmvs") {
    workspace_options.stereo_folder =
        StringPrintf("stereo-%s", pmvs_option_name_.c_str());
  }

  workspace_options.max_image_size = options_.max_image_size;
  workspace_options.image_as_rgb = true;
  workspace_options.cache_size = options_.cache_size;
  workspace_options.workspace_path = workspace_path_;
  workspace_options.workspace_format = workspace_format_;
  workspace_options.input_type = input_type_;

  workspace_.reset(new Workspace(workspace_options));

  const auto image_names = ReadTextFileLines(JoinPaths(
      workspace_path_, workspace_options.stereo_folder, "fusion.cfg"));

  const auto& model = workspace_->GetModel();


  used_images_.resize(model.images.size(), false);
  fused_images_.resize(model.images.size(), false);
  fused_pixel_masks_.resize(model.images.size());
  depth_map_sizes_.resize(model.images.size());
  bitmap_scales_.resize(model.images.size());
  P_.resize(model.images.size());
  inv_P_.resize(model.images.size());
  inv_R_.resize(model.images.size());

  std::vector<std::pair<int, std::vector<Eigen::Vector3f>>> all_points;
  for (const auto& image_name : image_names) {
    const int image_idx = model.GetImageIdx(image_name);

    if (!workspace_->HasBitmap(image_idx) ||
        !workspace_->HasDepthMap(image_idx) ||
        !workspace_->HasNormalMap(image_idx)) {
      std::cout
          << StringPrintf(
                 "WARNING: Ignoring image %s, because input does not exist.",
                 image_name.c_str())
          << std::endl;
      continue;
    }

    const auto& image = model.images.at(image_idx);
    const auto& depth_map = workspace_->GetDepthMap(image_idx);

    used_images_.at(image_idx) = true;

    fused_pixel_masks_.at(image_idx) =
        Mat<bool>(depth_map.GetWidth(), depth_map.GetHeight(), 1);
    fused_pixel_masks_.at(image_idx).Fill(false);

    depth_map_sizes_.at(image_idx) =
        std::make_pair(depth_map.GetWidth(), depth_map.GetHeight());

    bitmap_scales_.at(image_idx) = std::make_pair(
        static_cast<float>(depth_map.GetWidth()) / image.GetWidth(),
        static_cast<float>(depth_map.GetHeight()) / image.GetHeight());

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K =
        Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
            image.GetK());
    K(0, 0) *= bitmap_scales_.at(image_idx).first;
    K(0, 2) *= bitmap_scales_.at(image_idx).first;
    K(1, 1) *= bitmap_scales_.at(image_idx).second;
    K(1, 2) *= bitmap_scales_.at(image_idx).second;


    ComposeProjectionMatrix(K.data(), image.GetR(), image.GetT(),
                            P_.at(image_idx).data());
    ComposeInverseProjectionMatrix(K.data(), image.GetR(), image.GetT(),
                                   inv_P_.at(image_idx).data());
    inv_R_.at(image_idx) =
        Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
            image.GetR())
            .transpose();


    const int width = depth_map_sizes_.at(image_idx).first;
    const int height = depth_map_sizes_.at(image_idx).second;
    const auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx);

    FusionData data;
    data.image_idx = image_idx;
    data.traversal_depth = 0;

    std::vector<Eigen::Vector3f> points;
    for (data.row = 0; data.row < height; ++data.row) {
      for (data.col = 0; data.col < width; ++data.col) {
        if (fused_pixel_mask.Get(data.row, data.col)) {
          continue;
        }


        Eigen::Vector4f fused_ref_point = Eigen::Vector4f::Zero();
        Eigen::Vector3f fused_ref_normal = Eigen::Vector3f::Zero();

        fused_point_x_.clear();
        fused_point_y_.clear();
        fused_point_z_.clear();
        fused_point_nx_.clear();
        fused_point_ny_.clear();
        fused_point_nz_.clear();
        fused_point_r_.clear();
        fused_point_g_.clear();
        fused_point_b_.clear();
        fused_point_visibility_.clear();


        const int row = data.row;
        const int col = data.col;
        const int traversal_depth = data.traversal_depth;

        fusion_queue_.pop_back();

        // Check if pixel already fused.
        auto& fused_pixel_mask = fused_pixel_masks_.at(image_idx);
        if (fused_pixel_mask.Get(row, col)) {
          continue;
        }

        const auto& depth_map = workspace_->GetDepthMap(image_idx);
        const float depth = depth_map.Get(row, col);

        // Pixels with negative depth are filtered.
        if (depth <= 0.0f) {
          continue;
        }

        // If the traversal depth is greater than zero, the initial reference
        // pixel has already been added and we need to check for consistency.
        if (traversal_depth > 0) {
          // Project reference point into current view.
          const Eigen::Vector3f proj = P_.at(image_idx) * fused_ref_point;

          // Depth error of reference depth with current depth.
          const float depth_error = std::abs((proj(2) - depth) / depth);
          if (depth_error > options_.max_depth_error) {
            continue;
          }

          // Reprojection error reference point in the current view.
          const float col_diff = proj(0) / proj(2) - col;
          const float row_diff = proj(1) / proj(2) - row;
          const float squared_reproj_error =
              col_diff * col_diff + row_diff * row_diff;
          if (squared_reproj_error > max_squared_reproj_error_) {
            continue;
          }
        }

        // Determine normal direction in global reference frame.
        const auto& normal_map = workspace_->GetNormalMap(image_idx);
        const Eigen::Vector3f normal =
            inv_R_.at(image_idx) * Eigen::Vector3f(normal_map.Get(row, col, 0),
                                                   normal_map.Get(row, col, 1),
                                                   normal_map.Get(row, col, 2));

        // Check for consistent normal direction with reference normal.
        if (traversal_depth > 0) {
          const float cos_normal_error = fused_ref_normal.dot(normal);
          if (cos_normal_error < min_cos_normal_error_) {
            continue;
          }
        }

        // Determine 3D location of current depth value.
        Eigen::Vector3f xyz =
            inv_P_.at(image_idx) *
            Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);

        points.push_back(xyz);

      }
    }
    all_points.push_back(std::make_pair(image_idx, points));
    std::cout << "image_idx: " << image_idx << " has " << points.size() << " number of points" << std::endl;
  } // end of loop over all images



}





}  // namespace mvs
}  // namespace colmap
