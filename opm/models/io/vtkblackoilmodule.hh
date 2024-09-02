// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 * \copydoc Opm::VtkBlackOilModule
 */
#ifndef EWOMS_VTK_BLACK_OIL_MODULE_HH
#define EWOMS_VTK_BLACK_OIL_MODULE_HH

#include "vtkmultiwriter.hh"
#include "baseoutputmodule.hh"

#include <dune/common/fvector.hh>

#include <opm/material/densead/Math.hpp>

#include <opm/models/blackoil/blackoilproperties.hh>

#include <opm/models/discretization/common/fvbaseparameters.hh>

#include <opm/models/utils/parametersystem.hh>
#include <opm/models/utils/propertysystem.hh>

namespace Opm::Parameters {

// set default values for what quantities to output
struct VtkWriteGasDissolutionFactor { static constexpr bool value = false; };
struct VtkWriteOilVaporizationFactor { static constexpr bool value = false; };
struct VtkWriteOilFormationVolumeFactor { static constexpr bool value = false; };
struct VtkWriteGasFormationVolumeFactor { static constexpr bool value = false; };
struct VtkWriteWaterFormationVolumeFactor { static constexpr bool value = false; };
struct VtkWriteOilSaturationPressure { static constexpr bool value = false; };
struct VtkWriteGasSaturationPressure { static constexpr bool value = false; };
struct VtkWriteSaturationRatios { static constexpr bool value = false; };
struct VtkWriteSaturatedOilGasDissolutionFactor { static constexpr bool value = false; };
struct VtkWriteSaturatedGasOilVaporizationFactor { static constexpr bool value = false; };
struct VtkWritePrimaryVarsMeaning { static constexpr bool value = false; };

} // namespace Opm::Parameters

namespace Opm {
/*!
 * \ingroup Vtk
 *
 * \brief VTK output module for the black oil model's parameters.
 */
template <class TypeTag>
class VtkBlackOilModule : public BaseOutputModule<TypeTag>
{
    using ParentType = BaseOutputModule<TypeTag>;

    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;

    using GridView = GetPropType<TypeTag, Properties::GridView>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;

    static const int vtkFormat = getPropValue<TypeTag, Properties::VtkOutputFormat>();
    using VtkMultiWriter = ::Opm::VtkMultiWriter<GridView, vtkFormat>;

    enum { oilPhaseIdx = FluidSystem::oilPhaseIdx };
    enum { gasPhaseIdx = FluidSystem::gasPhaseIdx };
    enum { waterPhaseIdx = FluidSystem::waterPhaseIdx };

    enum { gasCompIdx = FluidSystem::gasCompIdx };
    enum { oilCompIdx = FluidSystem::oilCompIdx };
    enum { waterCompIdx = FluidSystem::waterCompIdx };

    using ScalarBuffer = typename ParentType::ScalarBuffer;

public:
    VtkBlackOilModule(const Simulator& simulator)
        : ParentType(simulator)
    { }

    /*!
     * \brief Register all run-time parameters for the multi-phase VTK output
     * module.
     */
    static void registerParameters()
    {
        Parameters::Register<Parameters::VtkWriteGasDissolutionFactor>
            ("Include the gas dissolution factor (R_s) of the observed oil "
             "in the VTK output files");
        Parameters::Register<Parameters::VtkWriteOilVaporizationFactor>
            ("Include the oil vaporization factor (R_v) of the observed gas "
             "in the VTK output files");
        Parameters::Register<Parameters::VtkWriteOilFormationVolumeFactor>
            ("Include the oil formation volume factor (B_o) in the VTK output files");
        Parameters::Register<Parameters::VtkWriteGasFormationVolumeFactor>
            ("Include the gas formation volume factor (B_g) in the "
             "VTK output files");
        Parameters::Register<Parameters::VtkWriteWaterFormationVolumeFactor>
            ("Include the water formation volume factor (B_w) in the "
             "VTK output files");
        Parameters::Register<Parameters::VtkWriteOilSaturationPressure>
            ("Include the saturation pressure of oil (p_o,sat) in the "
             "VTK output files");
        Parameters::Register<Parameters::VtkWriteGasSaturationPressure>
            ("Include the saturation pressure of gas (p_g,sat) in the "
             "VTK output files");
        Parameters::Register<Parameters::VtkWriteSaturatedOilGasDissolutionFactor>
            ("Include the gas dissolution factor (R_s,sat) of gas saturated "
             "oil in the VTK output files");
        Parameters::Register<Parameters::VtkWriteSaturatedGasOilVaporizationFactor>
            ("Include the oil vaporization factor (R_v,sat) of oil saturated "
             "gas in the VTK output files");
        Parameters::Register<Parameters::VtkWriteSaturationRatios>
            ("Write the ratio of the actually and maximum dissolved component of "
             "the mixtures");
        Parameters::Register<Parameters::VtkWritePrimaryVarsMeaning>
            ("Include how the primary variables should be interpreted to the "
             "VTK output files");
    }

    /*!
     * \brief Allocate memory for the scalar fields we would like to
     *        write to the VTK file.
     */
    void allocBuffers()
    {
        if (gasDissolutionFactorOutput_())
            this->resizeScalarBuffer_(gasDissolutionFactor_);
        if (oilVaporizationFactorOutput_())
            this->resizeScalarBuffer_(oilVaporizationFactor_);
        if (oilFormationVolumeFactorOutput_())
            this->resizeScalarBuffer_(oilFormationVolumeFactor_);
        if (gasFormationVolumeFactorOutput_())
            this->resizeScalarBuffer_(gasFormationVolumeFactor_);
        if (waterFormationVolumeFactorOutput_())
            this->resizeScalarBuffer_(waterFormationVolumeFactor_);
        if (oilSaturationPressureOutput_())
            this->resizeScalarBuffer_(oilSaturationPressure_);
        if (gasSaturationPressureOutput_())
            this->resizeScalarBuffer_(gasSaturationPressure_);
        if (saturatedOilGasDissolutionFactorOutput_())
            this->resizeScalarBuffer_(saturatedOilGasDissolutionFactor_);
        if (saturatedGasOilVaporizationFactorOutput_())
            this->resizeScalarBuffer_(saturatedGasOilVaporizationFactor_);
        if (saturationRatiosOutput_()) {
            this->resizeScalarBuffer_(oilSaturationRatio_);
            this->resizeScalarBuffer_(gasSaturationRatio_);
        }
        if (primaryVarsMeaningOutput_()) {
            this->resizeScalarBuffer_(primaryVarsMeaningPressure_);
            this->resizeScalarBuffer_(primaryVarsMeaningWater_);
            this->resizeScalarBuffer_(primaryVarsMeaningGas_);
        }
    }

    /*!
     * \brief Modify the internal buffers according to the intensive quantities relevant for
     *        an element
     */
    void processElement(const ElementContext& elemCtx)
    {
        if (!Parameters::Get<Parameters::EnableVtkOutput>()) {
            return;
        }

        for (unsigned dofIdx = 0; dofIdx < elemCtx.numPrimaryDof(/*timeIdx=*/0); ++dofIdx) {
            const auto& fs = elemCtx.intensiveQuantities(dofIdx, /*timeIdx=*/0).fluidState();
            using FluidState = typename std::remove_const<typename std::remove_reference<decltype(fs)>::type>::type;
            unsigned globalDofIdx = elemCtx.globalSpaceIndex(dofIdx, /*timeIdx=*/0);

            const auto& primaryVars = elemCtx.primaryVars(dofIdx, /*timeIdx=*/0);

            unsigned pvtRegionIdx = elemCtx.primaryVars(dofIdx, /*timeIdx=*/0).pvtRegionIndex();
            Scalar SoMax = 0.0;
            if (FluidSystem::phaseIsActive(oilPhaseIdx))
                SoMax = std::max(getValue(fs.saturation(oilPhaseIdx)),
                               elemCtx.problem().maxOilSaturation(globalDofIdx));

            if (FluidSystem::phaseIsActive(gasPhaseIdx) && FluidSystem::phaseIsActive(oilPhaseIdx)) {
                Scalar x_oG = getValue(fs.moleFraction(oilPhaseIdx, gasCompIdx));
                Scalar x_gO = getValue(fs.moleFraction(gasPhaseIdx, oilCompIdx));
                Scalar X_oG = getValue(fs.massFraction(oilPhaseIdx, gasCompIdx));
                Scalar X_gO = getValue(fs.massFraction(gasPhaseIdx, oilCompIdx));
                Scalar Rs = FluidSystem::convertXoGToRs(X_oG, pvtRegionIdx);
                Scalar Rv = FluidSystem::convertXgOToRv(X_gO, pvtRegionIdx);

                Scalar RsSat =
                    FluidSystem::template saturatedDissolutionFactor<FluidState, Scalar>(fs,
                                                                                         oilPhaseIdx,
                                                                                         pvtRegionIdx,
                                                                                         SoMax);
                Scalar X_oG_sat = FluidSystem::convertRsToXoG(RsSat, pvtRegionIdx);
                Scalar x_oG_sat = FluidSystem::convertXoGToxoG(X_oG_sat, pvtRegionIdx);

                Scalar RvSat =
                    FluidSystem::template saturatedDissolutionFactor<FluidState, Scalar>(fs,
                                                                                         gasPhaseIdx,
                                                                                         pvtRegionIdx,
                                                                                         SoMax);
                Scalar X_gO_sat = FluidSystem::convertRvToXgO(RvSat, pvtRegionIdx);
                Scalar x_gO_sat = FluidSystem::convertXgOToxgO(X_gO_sat, pvtRegionIdx);
                if (gasDissolutionFactorOutput_())
                    gasDissolutionFactor_[globalDofIdx] = Rs;
                if (oilVaporizationFactorOutput_())
                    oilVaporizationFactor_[globalDofIdx] = Rv;
                if (oilSaturationPressureOutput_())
                    oilSaturationPressure_[globalDofIdx] =
                        FluidSystem::template saturationPressure<FluidState, Scalar>(fs, oilPhaseIdx, pvtRegionIdx);
                if (gasSaturationPressureOutput_())
                    gasSaturationPressure_[globalDofIdx] =
                        FluidSystem::template saturationPressure<FluidState, Scalar>(fs, gasPhaseIdx, pvtRegionIdx);
                if (saturatedOilGasDissolutionFactorOutput_())
                    saturatedOilGasDissolutionFactor_[globalDofIdx] = RsSat;
                if (saturatedGasOilVaporizationFactorOutput_())
                    saturatedGasOilVaporizationFactor_[globalDofIdx] = RvSat;
                if (saturationRatiosOutput_()) {
                    if (x_oG_sat <= 0.0)
                        oilSaturationRatio_[globalDofIdx] = 1.0;
                    else
                        oilSaturationRatio_[globalDofIdx] = x_oG / x_oG_sat;

                    if (x_gO_sat <= 0.0)
                        gasSaturationRatio_[globalDofIdx] = 1.0;
                    else
                        gasSaturationRatio_[globalDofIdx] = x_gO / x_gO_sat;
                }
            }
            if (oilFormationVolumeFactorOutput_())
                oilFormationVolumeFactor_[globalDofIdx] =
                    1.0/FluidSystem::template inverseFormationVolumeFactor<FluidState, Scalar>(fs, oilPhaseIdx, pvtRegionIdx);
            if (gasFormationVolumeFactorOutput_())
                gasFormationVolumeFactor_[globalDofIdx] =
                    1.0/FluidSystem::template inverseFormationVolumeFactor<FluidState, Scalar>(fs, gasPhaseIdx, pvtRegionIdx);
            if (waterFormationVolumeFactorOutput_())
                waterFormationVolumeFactor_[globalDofIdx] =
                    1.0/FluidSystem::template inverseFormationVolumeFactor<FluidState, Scalar>(fs, waterPhaseIdx, pvtRegionIdx);

            if (primaryVarsMeaningOutput_()) {
                primaryVarsMeaningWater_[globalDofIdx] =
                    static_cast<int>(primaryVars.primaryVarsMeaningWater());
                primaryVarsMeaningGas_[globalDofIdx] =
                    static_cast<int>(primaryVars.primaryVarsMeaningGas());
                primaryVarsMeaningPressure_[globalDofIdx] =
                    static_cast<int>(primaryVars.primaryVarsMeaningPressure());
            }
        }
    }

    /*!
     * \brief Add all buffers to the VTK output writer.
     */
    void commitBuffers(BaseOutputWriter& baseWriter)
    {
        VtkMultiWriter *vtkWriter = dynamic_cast<VtkMultiWriter*>(&baseWriter);
        if (!vtkWriter)
            return;

        if (gasDissolutionFactorOutput_())
            this->commitScalarBuffer_(baseWriter, "R_s", gasDissolutionFactor_);
        if (oilVaporizationFactorOutput_())
            this->commitScalarBuffer_(baseWriter, "R_v", oilVaporizationFactor_);
        if (oilFormationVolumeFactorOutput_())
            this->commitScalarBuffer_(baseWriter, "B_o", oilFormationVolumeFactor_);
        if (gasFormationVolumeFactorOutput_())
            this->commitScalarBuffer_(baseWriter, "B_g", gasFormationVolumeFactor_);
        if (waterFormationVolumeFactorOutput_())
            this->commitScalarBuffer_(baseWriter, "B_w", waterFormationVolumeFactor_);
        if (oilSaturationPressureOutput_())
            this->commitScalarBuffer_(baseWriter, "p_o,sat", oilSaturationPressure_);
        if (gasSaturationPressureOutput_())
            this->commitScalarBuffer_(baseWriter, "p_g,sat", gasSaturationPressure_);
        if (saturatedOilGasDissolutionFactorOutput_())
            this->commitScalarBuffer_(baseWriter, "R_s,sat", saturatedOilGasDissolutionFactor_);
        if (saturatedGasOilVaporizationFactorOutput_())
            this->commitScalarBuffer_(baseWriter, "R_v,sat", saturatedGasOilVaporizationFactor_);
        if (saturationRatiosOutput_()) {
            this->commitScalarBuffer_(baseWriter, "saturation ratio_oil", oilSaturationRatio_);
            this->commitScalarBuffer_(baseWriter, "saturation ratio_gas", gasSaturationRatio_);
        }

        if (primaryVarsMeaningOutput_()) {
            this->commitScalarBuffer_(baseWriter, "primary vars meaning water", primaryVarsMeaningWater_);
            this->commitScalarBuffer_(baseWriter, "primary vars meaning gas", primaryVarsMeaningGas_);
            this->commitScalarBuffer_(baseWriter, "primary vars meaning pressure", primaryVarsMeaningPressure_);
        }
    }

private:
    static bool gasDissolutionFactorOutput_()
    {
        static bool val = Parameters::Get<Parameters::VtkWriteGasDissolutionFactor>();
        return val;
    }

    static bool oilVaporizationFactorOutput_()
    {
        static bool val = Parameters::Get<Parameters::VtkWriteOilVaporizationFactor>();
        return val;
    }

    static bool oilFormationVolumeFactorOutput_()
    {
        static bool val = Parameters::Get<Parameters::VtkWriteOilFormationVolumeFactor>();
        return val;
    }

    static bool gasFormationVolumeFactorOutput_()
    {
        static bool val = Parameters::Get<Parameters::VtkWriteGasFormationVolumeFactor>();
        return val;
    }

    static bool waterFormationVolumeFactorOutput_()
    {
        static bool val = Parameters::Get<Parameters::VtkWriteWaterFormationVolumeFactor>();
        return val;
    }

    static bool oilSaturationPressureOutput_()
    {
        static bool val = Parameters::Get<Parameters::VtkWriteOilSaturationPressure>();
        return val;
    }

    static bool gasSaturationPressureOutput_()
    {
        static bool val = Parameters::Get<Parameters::VtkWriteGasSaturationPressure>();
        return val;
    }

    static bool saturatedOilGasDissolutionFactorOutput_()
    {
        static bool val = Parameters::Get<Parameters::VtkWriteSaturatedOilGasDissolutionFactor>();
        return val;
    }

    static bool saturatedGasOilVaporizationFactorOutput_()
    {
        static bool val = Parameters::Get<Parameters::VtkWriteSaturatedGasOilVaporizationFactor>();
        return val;
    }

    static bool saturationRatiosOutput_()
    {
        static bool val = Parameters::Get<Parameters::VtkWriteSaturationRatios>();
        return val;
    }

    static bool primaryVarsMeaningOutput_()
    {
        static bool val = Parameters::Get<Parameters::VtkWritePrimaryVarsMeaning>();
        return val;
    }

    ScalarBuffer gasDissolutionFactor_;
    ScalarBuffer oilVaporizationFactor_;
    ScalarBuffer oilFormationVolumeFactor_;
    ScalarBuffer gasFormationVolumeFactor_;
    ScalarBuffer waterFormationVolumeFactor_;
    ScalarBuffer oilSaturationPressure_;
    ScalarBuffer gasSaturationPressure_;

    ScalarBuffer saturatedOilGasDissolutionFactor_;
    ScalarBuffer saturatedGasOilVaporizationFactor_;
    ScalarBuffer oilSaturationRatio_;
    ScalarBuffer gasSaturationRatio_;

    ScalarBuffer primaryVarsMeaningPressure_;
    ScalarBuffer primaryVarsMeaningWater_;
    ScalarBuffer primaryVarsMeaningGas_;
};

} // namespace Opm

#endif
