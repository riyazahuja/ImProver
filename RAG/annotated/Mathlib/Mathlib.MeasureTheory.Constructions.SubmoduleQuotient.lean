instance [MeasurableSpace M] : MeasurableSpace (M ⧸ p) := Quotient.instMeasurableSpace

instance [MeasurableSpace M] [DiscreteMeasurableSpace M] : DiscreteMeasurableSpace (M ⧸ p) :=
  Quotient.instDiscreteMeasurableSpace


