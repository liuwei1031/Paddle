/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/executor_lite.h"
#include <memory>
#include <vector>
#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/core/scope.h"
#include "pybind11/pybind11.h"

namespace paddle {
namespace pybind {

namespace lt = paddle::lite;
namespace py = pybind11;

void BindVariable(pybind11::module* m) {
  py::class_<lt::Variable>(*m, "Variable");
}

void BindScope(pybind11::module* m) {
  py::class_<lt::Scope>(*m, "Scope")
      .def(pybind11::init<>())
      .def("new_scope",
           [](lt::Scope& self) -> lt::Scope* { return &self.NewScope(); },
           py::return_value_policy::reference)
      .def("var", &lt::Scope::Var, pybind11::return_value_policy::reference)
      .def("find_var", &lt::Scope::FindVar,
           pybind11::return_value_policy::reference)
      .def("find_local_var", &lt::Scope::FindLocalVar,
           pybind11::return_value_policy::reference)
      .def("parent", &lt::Scope::parent,
           pybind11::return_value_policy::reference)
      .def("local_var_names", &lt::Scope::LocalVarNames,
           pybind11::return_value_policy::reference);
}

void BindExecutorLite(pybind11::module* m) {
  py::class_<lt::ExecutorLite>(*m, "ExecutorLite")
      .def(pybind11::init<>())
      .def("__init__",
           [](lt::ExecutorLite& self,
              const std::shared_ptr<lt::Scope>& root_scope) {
             new (&self) lt::ExecutorLite(root_scope);
           })
      .def("get_input", &lt::ExecutorLite::GetInput,
           pybind11::return_value_policy::reference)
      .def("get_output", &lt::ExecutorLite::GetOutput,
           pybind11::return_value_policy::reference)
      .def("run", &lt::ExecutorLite::Run);
}

void BindEnums(pybind11::module* m) {
  py::enum_<lt::TargetType>(*m, "TargetType", py::arithmetic(),
                            "TargetType enum")
      .value("kUnk", lt::TargetType::kUnk)
      .value("kHost", lt::TargetType::kHost)
      .value("kX86", lt::TargetType::kX86)
      .value("kCUDA", lt::TargetType::kCUDA)
      .value("kARM", lt::TargetType::kARM)
      .value("kAny", lt::TargetType::kAny)
      .value("NUM", lt::TargetType::NUM);

  py::enum_<lt::PrecisionType>(*m, "PrecisionType", py::arithmetic(),
                               "PrecisionType enum")
      .value("kUnk", lt::PrecisionType::kUnk)
      .value("kFloat", lt::PrecisionType::kFloat)
      .value("kInt8", lt::PrecisionType::kInt8)
      .value("kAny", lt::PrecisionType::kAny)
      .value("NUM", lt::PrecisionType::NUM);

  py::enum_<lt::DataLayoutType>(*m, "DataLayoutType", py::arithmetic(),
                                "DataLayoutType enum")
      .value("kUnk", lt::DataLayoutType::kUnk)
      .value("kNCHW", lt::DataLayoutType::kNCHW)
      .value("kAny", lt::DataLayoutType::kAny)
      .value("NUM", lt::DataLayoutType::NUM);
}

void BindPlace(pybind11::module* m) {
  pybind11::class_<lt::Place>(*m, "Place")
      .def(pybind11::init<>())
      .def("__init__",
           [](lt::Place& self, lt::TargetType target,
              lt::PrecisionType precision, lt::DataLayoutType layout,
              int16_t device) {
             new (&self) lt::Place(target, precision, layout, device);
           })
      .def("is_valid", &lt::Place::is_valid,
           pybind11::return_value_policy::reference);
}

void BindCXXTrainer(pybind11::module* m) {
  pybind11::class_<lt::CXXTrainer>(*m, "CXXTrainer")
      .def(
          "__init__",
          [](lt::CXXTrainer& self, const std::shared_ptr<lt::Scope>& root_scope,
             const lt::Place& preferred_place,
             const std::vector<lt::Place>& valid_places) {
            new (&self)
                lt::CXXTrainer(root_scope, preferred_place, valid_places);
          })
      .def("build_main_program_executor",
           &lt::CXXTrainer::BuildMainProgramExecutor,
           pybind11::return_value_policy::reference)
      .def("run_startup_program", &lt::CXXTrainer::RunStartupProgram);
}

void BindLite(pybind11::module* m) {
  BindVariable(m);
  BindScope(m);
  BindExecutorLite(m);
  BindEnums(m);
  BindPlace(m);
  BindCXXTrainer(m);
}

}  // namespace pybind
}  // namespace paddle
