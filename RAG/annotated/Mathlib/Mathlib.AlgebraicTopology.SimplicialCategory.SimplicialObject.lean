noncomputable instance : EnrichedCategory SSet.{v} (SimplicialObject D)  :=
  inferInstanceAs (EnrichedCategory (_ ⥤ Type v) (_ ⥤ D))


noncomputable instance : SimplicialCategory (SimplicialObject D) where
  homEquiv := Functor.natTransEquiv.symm


noncomputable instance : SimplicialCategory SSet.{v} :=
  inferInstanceAs (SimplicialCategory (SimplicialObject (Type v)))


