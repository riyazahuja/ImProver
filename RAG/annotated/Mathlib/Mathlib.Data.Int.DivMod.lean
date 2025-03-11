theorem emod_eq_sub_self_emod {a b : Int} : a % b = (a - b) % b :=
  (emod_sub_cancel a b).symm


theorem emod_eq_add_self_emod {a b : Int} : a % b = (a + b) % b :=
  add_emod_self.symm


