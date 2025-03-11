instance charP [CharP F p] : CharP L p := RingHom.charP _ (algebraMap _ F).injective p


instance expChar [ExpChar F p] : ExpChar L p := RingHom.expChar _ (algebraMap _ F).injective p


instance charZero [CharZero F] : CharZero L :=
  charZero_of_injective_algebraMap (algebraMap F _).injective


instance charP [CharP F p] : CharP L p :=
  charP_of_injective_algebraMap (algebraMap F _).injective p


instance expChar [ExpChar F p] : ExpChar L p :=
  expChar_of_injective_algebraMap (algebraMap F _).injective p


instance charP' [CharP E p] : CharP L p := Subfield.charP L.toSubfield p


instance expChar' [ExpChar E p] : ExpChar L p := Subfield.expChar L.toSubfield p


