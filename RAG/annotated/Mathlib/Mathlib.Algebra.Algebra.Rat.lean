@[simp]
theorem map_rat_algebraMap [Semiring R] [Semiring S] [Algebra ℚ R] [Algebra ℚ S] (f : R →+* S)
    (r : ℚ) : f (algebraMap ℚ R r) = algebraMap ℚ S r :=
  RingHom.ext_iff.1 (Subsingleton.elim (f.comp (algebraMap ℚ R)) (algebraMap ℚ S)) r


variable (R) in
/-- `nnqsmul` is equal to any other module structure via a cast. -/
lemma cast_smul_eq_nnqsmul [Module R S] (q : ℚ≥0) (a : S) : (q : R) • a = q • a := by
  /-
    R : Type u_2
    S : Type u_3
    inst✝⁴ : DivisionSemiring R
    inst✝³ : CharZero R
    inst✝² : Semiring S
    inst✝¹ : Module NNRat S
    inst✝ : Module R S
    q : NNRat
    a : S
    ⊢ Eq (HSMul.hSMul (↑q) a) (HSMul.hSMul q a)
  -/
  refine MulAction.injective₀ (G₀ := ℚ≥0) (Nat.cast_ne_zero.2 q.den_pos.ne') ?_
  /-
    R : Type u_2
    S : Type u_3
    inst✝⁴ : DivisionSemiring R
    inst✝³ : CharZero R
    inst✝² : Semiring S
    inst✝¹ : Module NNRat S
    inst✝ : Module R S
    q : NNRat
    a : S
    ⊢ Eq ((fun x => HSMul.hSMul (↑q.den) x) (HSMul.hSMul (↑q) a)) ((fun x => HSMul …
  -/
  dsimp
  rw [← mul_smul, den_mul_eq_num, Nat.cast_smul_eq_nsmul, Nat.cast_smul_eq_nsmul, ← smul_assoc,
    nsmul_eq_mul q.den, ← cast_natCast, ← cast_mul, den_mul_eq_num, cast_natCast,
    Nat.cast_smul_eq_nsmul]


instance _root_.DivisionSemiring.toNNRatAlgebra : Algebra ℚ≥0 R where
  smul_def' := smul_def
  toRingHom := castHom _
  commutes' := cast_commute


instance _root_.RingHomClass.toLinearMapClassNNRat [FunLike F R S] [RingHomClass F R S] :
    LinearMapClass F ℚ≥0 R S where
                         /-
                           F : Type u_1
                           R : Type u_2
                           S : Type u_3
                           inst✝⁵ : DivisionSemiring R
                           inst✝⁴ : CharZero R
                           inst✝³ : DivisionSemiring S
                           inst✝² : CharZero S
                           inst✝¹ : FunLike F R S
                           inst✝ : RingHomClass F R S
                           f : F
                           q : NNRat
                           a : R
                           ⊢ Eq (f (HSMul.hSMul q a)) (HSMul.hSMul ((RingHom.id NNRat) q) (f a))
                         -/
  map_smulₛₗ f q a := by simp [smul_def, cast_id]
                         /-
                           🎉 no goals
                         -/


instance instSMulCommClass [SMulCommClass R S S] : SMulCommClass ℚ≥0 R S where
                        /-
                          F : Type u_1
                          R : Type u_2
                          S : Type u_3
                          inst✝⁵ : DivisionSemiring R
                          inst✝⁴ : CharZero R
                          inst✝³ : DivisionSemiring S
                          inst✝² : CharZero S
                          inst✝¹ : SMul R S
                          inst✝ : SMulCommClass R S S
                          q : NNRat
                          a : R
                          b : S
                          ⊢ Eq (HSMul.hSMul q (HSMul.hSMul a b)) (HSMul.hSMul a (HSMul.hSMul q b))
                        -/
  smul_comm q a b := by simp [smul_def, mul_smul_comm]
                        /-
                          🎉 no goals
                        -/


instance instSMulCommClass' [SMulCommClass S R S] : SMulCommClass R ℚ≥0 S :=
  have := SMulCommClass.symm S R S; SMulCommClass.symm _ _ _


variable (R) in
/-- `nnqsmul` is equal to any other module structure via a cast. -/
lemma cast_smul_eq_qsmul [Module R S] (q : ℚ) (a : S) : (q : R) • a = q • a := by
  /-
    R : Type u_2
    S : Type u_3
    inst✝⁴ : DivisionRing R
    inst✝³ : CharZero R
    inst✝² : Ring S
    inst✝¹ : Module Rat S
    inst✝ : Module R S
    q : Rat
    a : S
    ⊢ Eq (HSMul.hSMul (↑q) a) (HSMul.hSMul q a)
  -/
  refine MulAction.injective₀ (G₀ := ℚ) (Nat.cast_ne_zero.2 q.den_pos.ne') ?_
  /-
    R : Type u_2
    S : Type u_3
    inst✝⁴ : DivisionRing R
    inst✝³ : CharZero R
    inst✝² : Ring S
    inst✝¹ : Module Rat S
    inst✝ : Module R S
    q : Rat
    a : S
    ⊢ Eq ((fun x => HSMul.hSMul (↑q.den) x) (HSMul.hSMul (↑q) a)) ((fun x => HSMul …
  -/
  dsimp
  rw [← mul_smul, den_mul_eq_num, Nat.cast_smul_eq_nsmul, Int.cast_smul_eq_zsmul, ← smul_assoc,
    nsmul_eq_mul q.den, ← cast_natCast, ← cast_mul, den_mul_eq_num, cast_intCast,
    Int.cast_smul_eq_zsmul]


instance _root_.DivisionRing.toRatAlgebra : Algebra ℚ R where
  smul_def' := smul_def
  toRingHom := castHom _
  commutes' := cast_commute


instance _root_.RingHomClass.toLinearMapClassRat [FunLike F R S] [RingHomClass F R S] :
    LinearMapClass F ℚ R S where
                         /-
                           F : Type u_1
                           R : Type u_2
                           S : Type u_3
                           inst✝⁵ : DivisionRing R
                           inst✝⁴ : CharZero R
                           inst✝³ : DivisionRing S
                           inst✝² : CharZero S
                           inst✝¹ : FunLike F R S
                           inst✝ : RingHomClass F R S
                           f : F
                           q : Rat
                           a : R
                           ⊢ Eq (f (HSMul.hSMul q a)) (HSMul.hSMul ((RingHom.id Rat) q) (f a))
                         -/
  map_smulₛₗ f q a := by simp [smul_def, cast_id]
                         /-
                           🎉 no goals
                         -/


instance instSMulCommClass [SMulCommClass R S S] : SMulCommClass ℚ R S where
                        /-
                          F : Type u_1
                          R : Type u_2
                          S : Type u_3
                          inst✝⁵ : DivisionRing R
                          inst✝⁴ : CharZero R
                          inst✝³ : DivisionRing S
                          inst✝² : CharZero S
                          inst✝¹ : SMul R S
                          inst✝ : SMulCommClass R S S
                          q : Rat
                          a : R
                          b : S
                          ⊢ Eq (HSMul.hSMul q (HSMul.hSMul a b)) (HSMul.hSMul a (HSMul.hSMul q b))
                        -/
  smul_comm q a b := by simp [smul_def, mul_smul_comm]
                        /-
                          🎉 no goals
                        -/


instance instSMulCommClass' [SMulCommClass S R S] : SMulCommClass R ℚ S :=
  have := SMulCommClass.symm S R S; SMulCommClass.symm _ _ _


@[deprecated Algebra.id.map_eq_id (since := "2024-07-30")]
lemma _root_.algebraMap_rat_rat : algebraMap ℚ ℚ = RingHom.id ℚ := rfl


instance algebra_rat_subsingleton {R} [Semiring R] : Subsingleton (Algebra ℚ R) :=
  ⟨fun x y => Algebra.algebra_ext x y <| RingHom.congr_fun <| Subsingleton.elim _ _⟩


