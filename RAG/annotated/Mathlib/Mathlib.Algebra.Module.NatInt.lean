instance AddCommGroup.toNatModule : Module ℕ M where
  one_smul := one_nsmul
  mul_smul m n a := mul_nsmul' a m n
  smul_add n a b := nsmul_add a b n
  smul_zero := nsmul_zero
  zero_smul := zero_nsmul
  add_smul r s x := add_nsmul x r s


instance AddCommGroup.toIntModule : Module ℤ M where
  one_smul := one_zsmul
  mul_smul m n a := mul_zsmul a m n
  smul_add n a b := zsmul_add a b n
  smul_zero := zsmul_zero
  zero_smul := zero_zsmul
  add_smul r s x := add_zsmul x r s


/-- An `AddCommMonoid` that is a `Module` over a `Ring` carries a natural `AddCommGroup`
structure.
See note [reducible non-instances]. -/
abbrev Module.addCommMonoidToAddCommGroup
    [Ring R] [AddCommMonoid M] [Module R M] : AddCommGroup M :=
  { (inferInstance : AddCommMonoid M) with
    neg := fun a => (-1 : R) • a
    neg_add_cancel := fun a =>
      show (-1 : R) • a + a = 0 by
        /-
          R : Type u_1
          S : Type u_2
          M : Type u_3
          M₂ : Type u_4
          inst✝² : Ring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          a : M
          ⊢ Eq (HAdd.hAdd (HSMul.hSMul (-1) a) a) 0
        -/
        nth_rw 2 [← one_smul R a]
        /-
          R : Type u_1
          S : Type u_2
          M : Type u_3
          M₂ : Type u_4
          inst✝² : Ring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          a : M
          ⊢ Eq (HAdd.hAdd (HSMul.hSMul (-1) a) (HSMul.hSMul 1 a)) 0
        -/
                               /-
                                 R : Type u_1
                                 S : Type u_2
                                 M : Type u_3
                                 M₂ : Type u_4
                                 inst✝² : Ring R
                                 inst✝¹ : AddCommMonoid M
                                 inst✝ : Module R M
                                 a : M
                                 ⊢ Eq ((fun z a => HSMul.hSMul (↑z) a) 0 a) 0
                               -/
        rw [← add_smul, neg_add_cancel, zero_smul]
                               /-
                                 🎉 no goals
                               -/
                                 /-
                                   R : Type u_1
                                   S : Type u_2
                                   M : Type u_3
                                   M₂ : Type u_4
                                   inst✝² : Ring R
                                   inst✝¹ : AddCommMonoid M
                                   inst✝ : Module R M
                                   z : Nat
                                   a : M
                                   ⊢ Eq ((fun z a => HSMul.hSMul (↑z) a) (↑z.succ) a) (HAdd.hAdd ((fun z a => HSM …
                                 -/
        /-
          🎉 no goals
        -/
                                 /-
                                   🎉 no goals
                                 -/
                                /-
                                  R : Type u_1
                                  S : Type u_2
                                  M : Type u_3
                                  M₂ : Type u_4
                                  inst✝² : Ring R
                                  inst✝¹ : AddCommMonoid M
                                  inst✝ : Module R M
                                  z : Nat
                                  a : M
                                  ⊢ Eq ((fun z a => HSMul.hSMul (↑z) a) (Int.negSucc z) a) (Neg.neg ((fun z a => …
                                -/
    zsmul := fun z a => (z : R) • a
                                /-
                                  🎉 no goals
                                -/
    zsmul_zero' := fun a => by simpa only [Int.cast_zero] using zero_smul R a
    zsmul_succ' := fun z a => by simp [add_comm, add_smul]
    zsmul_neg' := fun z a => by simp [← smul_assoc, neg_one_smul] }


/-- `nsmul` is equal to any other module structure via a cast. -/
@[norm_cast]
lemma Nat.cast_smul_eq_nsmul (n : ℕ) (b : M) : (n : R) • b = n • b := by
  induction n with
  | zero => rw [Nat.cast_zero, zero_smul, zero_smul]
  | succ n ih => rw [Nat.cast_succ, add_smul, add_smul, one_smul, ih, one_smul]


/-- `nsmul` is equal to any other module structure via a cast. -/
lemma ofNat_smul_eq_nsmul (n : ℕ) [n.AtLeastTwo] (b : M) :
    (ofNat(n) : R) • b = OfNat.ofNat n • b := Nat.cast_smul_eq_nsmul ..


/-- `nsmul` is equal to any other module structure via a cast. -/
@[deprecated Nat.cast_smul_eq_nsmul (since := "2024-07-23")]
lemma nsmul_eq_smul_cast (n : ℕ) (b : M) : n • b = (n : R) • b := (Nat.cast_smul_eq_nsmul ..).symm


/-- Convert back any exotic `ℕ`-smul to the canonical instance. This should not be needed since in
mathlib all `AddCommMonoid`s should normally have exactly one `ℕ`-module structure by design.
-/
theorem nat_smul_eq_nsmul (h : Module ℕ M) (n : ℕ) (x : M) : @SMul.smul ℕ M h.toSMul n x = n • x :=
  Nat.cast_smul_eq_nsmul ..


/-- All `ℕ`-module structures are equal. Not an instance since in mathlib all `AddCommMonoid`
should normally have exactly one `ℕ`-module structure by design. -/
def AddCommMonoid.uniqueNatModule : Unique (Module ℕ M) where
                /-
                  R : Type u_1
                  S : Type u_2
                  M : Type u_3
                  M₂ : Type u_4
                  inst✝² : Semiring R
                  inst✝¹ : AddCommMonoid M
                  inst✝ : Module R M
                  ⊢ Module Nat M
                -/
  default := by infer_instance
                /-
                  🎉 no goals
                -/
                                          /-
                                            R : Type u_1
                                            S : Type u_2
                                            M : Type u_3
                                            M₂ : Type u_4
                                            inst✝² : Semiring R
                                            inst✝¹ : AddCommMonoid M
                                            inst✝ : Module R M
                                            P : Module Nat M
                                            n : Nat
                                            ⊢ ∀ (m : M), Eq (HSMul.hSMul n m) (HSMul.hSMul n m)
                                          -/
  uniq P := (Module.ext' P _) fun n => by convert nat_smul_eq_nsmul P n
                                          /-
                                            🎉 no goals
                                          -/


instance AddCommMonoid.nat_isScalarTower : IsScalarTower ℕ R M where
  smul_assoc n x y := by
    induction n with
    | zero => simp only [zero_smul]
    | succ n ih => simp only [add_smul, one_smul, ih]


theorem map_natCast_smul [AddCommMonoid M] [AddCommMonoid M₂] {F : Type*} [FunLike F M M₂]
    [AddMonoidHomClass F M M₂] (f : F) (R S : Type*) [Semiring R] [Semiring S] [Module R M]
    [Module S M₂] (x : ℕ) (a : M) : f ((x : R) • a) = (x : S) • f a := by
  /-
    M : Type u_3
    M₂ : Type u_4
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid M₂
    F : Type u_5
    inst✝⁵ : FunLike F M M₂
    inst✝⁴ : AddMonoidHomClass F M M₂
    f : F
    R : Type u_6
    S : Type u_7
    inst✝³ : Semiring R
    inst✝² : Semiring S
    inst✝¹ : Module R M
    inst✝ : Module S M₂
    x : Nat
    a : M
    ⊢ Eq (f (HSMul.hSMul (↑x) a)) (HSMul.hSMul (↑x) (f a))
  -/
  simp only [Nat.cast_smul_eq_nsmul, AddMonoidHom.map_nsmul, map_nsmul]
  /-
    🎉 no goals
  -/


theorem Nat.smul_one_eq_cast {R : Type*} [NonAssocSemiring R] (m : ℕ) : m • (1 : R) = ↑m := by
  /-
    R : Type u_5
    inst✝ : NonAssocSemiring R
    m : Nat
    ⊢ Eq (HSMul.hSMul m 1) ↑m
  -/
  rw [nsmul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


theorem Int.smul_one_eq_cast {R : Type*} [NonAssocRing R] (m : ℤ) : m • (1 : R) = ↑m := by
  /-
    R : Type u_5
    inst✝ : NonAssocRing R
    m : Int
    ⊢ Eq (HSMul.hSMul m 1) ↑m
  -/
  rw [zsmul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-05-03")] alias Nat.smul_one_eq_coe := Nat.smul_one_eq_cast

@[deprecated (since := "2024-05-03")] alias Int.smul_one_eq_coe := Int.smul_one_eq_cast

