// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

namespace opensn
{

/// Structure containing edge properties
struct Edge
{
  /// Indices of the vertices
  int v_index[2]{};
  /// Indices of faces adjoining it
  int f_index[4]{};
  /// Vector vertices
  Vector3 vertices[2];

  Edge()
  {
    v_index[0] = -1;
    v_index[1] = -1;

    f_index[0] = -1;
    f_index[1] = -1;
    f_index[2] = -1;
    f_index[3] = -1;
  }

  Edge& operator=(const Edge& that)
  {
    if (&that != this)
    {
      this->v_index[0] = that.v_index[0];
      this->v_index[1] = that.v_index[1];

      this->f_index[0] = that.f_index[0];
      this->f_index[1] = that.f_index[1];
      this->f_index[2] = that.f_index[2];
      this->f_index[3] = that.f_index[3];

      this->vertices[0] = that.vertices[0];
      this->vertices[1] = that.vertices[1];
    }

    return *this;
  }
};

} // namespace opensn
