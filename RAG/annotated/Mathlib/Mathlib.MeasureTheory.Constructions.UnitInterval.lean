noncomputable instance : MeasureSpace I := Measure.Subtype.measureSpace


theorem volume_def : (volume : Measure I) = volume.comap Subtype.val := rfl


instance : IsProbabilityMeasure (volume : Measure I) where
  measure_univ := by
    rw [Measure.Subtype.volume_univ nullMeasurableSet_Icc, Real.volume_Icc, sub_zero,
      ENNReal.ofReal_one]


@[measurability]
theorem measurable_symm : Measurable symm := continuous_symm.measurable


