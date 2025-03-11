instance NonUnitalNonAssocSemiring.toDistribSMul [NonUnitalNonAssocSemiring α] :
    DistribSMul α α where smul_add := mul_add


/-- Note that `AddMonoid.nat_smulCommClass` requires stronger assumptions on `α`. -/
instance NonUnitalNonAssocSemiring.nat_smulCommClass [NonUnitalNonAssocSemiring α] :
    SMulCommClass ℕ α α where
  smul_comm n x y := by
    induction n with
    | zero => simp [zero_nsmul]
    | succ n ih => simp_rw [succ_nsmul, smul_eq_mul, mul_add, ← smul_eq_mul, ih]


/-- Note that `AddCommMonoid.nat_isScalarTower` requires stronger assumptions on `α`. -/
instance NonUnitalNonAssocSemiring.nat_isScalarTower [NonUnitalNonAssocSemiring α] :
    IsScalarTower ℕ α α where
  smul_assoc n x y := by
    induction n with
    | zero => simp [zero_nsmul]
    | succ n ih => simp_rw [succ_nsmul, ← ih, smul_eq_mul, add_mul]


/-- Note that `AddMonoid.int_smulCommClass` requires stronger assumptions on `α`. -/
instance NonUnitalNonAssocRing.int_smulCommClass [NonUnitalNonAssocRing α] :
    SMulCommClass ℤ α α where
  smul_comm n x y :=
    match n with
                    /-
                      α : Type u_1
                      inst✝ : NonUnitalNonAssocRing α
                      n✝ : Int
                      x y : α
                      n : Nat
                      ⊢ Eq (HSMul.hSMul (↑n) (HSMul.hSMul x y)) (HSMul.hSMul x (HSMul.hSMul (↑n) y))
                    -/
    | (n : ℕ) => by simp_rw [natCast_zsmul, smul_comm]
                    /-
                      🎉 no goals
                    -/
                   /-
                     α : Type u_1
                     inst✝ : NonUnitalNonAssocRing α
                     n✝ : Int
                     x y : α
                     n : Nat
                     ⊢ Eq (HSMul.hSMul (Int.negSucc n) (HSMul.hSMul x y)) (HSMul.hSMul x (HSMul.hSM …
                   -/
    | -[n+1] => by simp_rw [negSucc_zsmul, smul_eq_mul, mul_neg, mul_smul_comm]
                   /-
                     🎉 no goals
                   -/


/-- Note that `AddCommGroup.int_isScalarTower` requires stronger assumptions on `α`. -/
instance NonUnitalNonAssocRing.int_isScalarTower [NonUnitalNonAssocRing α] :
    IsScalarTower ℤ α α where
  smul_assoc n x y :=
    match n with
                    /-
                      α : Type u_1
                      inst✝ : NonUnitalNonAssocRing α
                      n✝ : Int
                      x y : α
                      n : Nat
                      ⊢ Eq (HSMul.hSMul (HSMul.hSMul (↑n) x) y) (HSMul.hSMul (↑n) (HSMul.hSMul x y))
                    -/
    | (n : ℕ) => by simp_rw [natCast_zsmul, smul_assoc]
                    /-
                      🎉 no goals
                    -/
                   /-
                     α : Type u_1
                     inst✝ : NonUnitalNonAssocRing α
                     n✝ : Int
                     x y : α
                     n : Nat
                     ⊢ Eq (HSMul.hSMul (HSMul.hSMul (Int.negSucc n) x) y) (HSMul.hSMul (Int.negSucc …
                   -/
    | -[n+1] => by simp_rw [negSucc_zsmul, smul_eq_mul, neg_mul, smul_mul_assoc]
                   /-
                     🎉 no goals
                   -/

