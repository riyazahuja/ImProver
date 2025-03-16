theorem Nat.noZeroSMulDivisors
    (R) (M) [Semiring R] [CharZero R] [AddCommMonoid M] [Module R M] [NoZeroSMulDivisors R M] :
    NoZeroSMulDivisors ℕ M where
                                                 /-
                                                   R : Type u_3
                                                   M : Type u_4
                                                   inst✝⁴ : Semiring R
                                                   inst✝³ : CharZero R
                                                   inst✝² : AddCommMonoid M
                                                   inst✝¹ : Module R M
                                                   inst✝ : NoZeroSMulDivisors R M
                                                   c : Nat
                                                   x : M
                                                   ⊢ Eq (HSMul.hSMul c x) 0 → Or (Eq c 0) (Eq x 0)
                                                 -/
  eq_zero_or_eq_zero_of_smul_eq_zero {c x} := by rw [← Nat.cast_smul_eq_nsmul R, smul_eq_zero]; simp
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


theorem two_nsmul_eq_zero
    (R) (M) [Semiring R] [CharZero R] [AddCommMonoid M] [Module R M] [NoZeroSMulDivisors R M]
    {v : M} : 2 • v = 0 ↔ v = 0 := by
  /-
    R : Type u_3
    M : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : CharZero R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    v : M
    ⊢ Iff (Eq (HSMul.hSMul 2 v) 0) (Eq v 0)
  -/
  haveI := Nat.noZeroSMulDivisors R M
  /-
    R : Type u_3
    M : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : CharZero R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    v : M
    this : NoZeroSMulDivisors Nat M
    ⊢ Iff (Eq (HSMul.hSMul 2 v) 0) (Eq v 0)
  -/
  simp [smul_eq_zero]
  /-
    🎉 no goals
  -/


/-- If `M` is an `R`-module with one and `M` has characteristic zero, then `R` has characteristic
zero as well. Usually `M` is an `R`-algebra. -/
theorem CharZero.of_module (M) [AddCommMonoidWithOne M] [CharZero M] [Module R M] : CharZero R := by
  /-
    R : Type u_1
    inst✝³ : Semiring R
    M : Type u_3
    inst✝² : AddCommMonoidWithOne M
    inst✝¹ : CharZero M
    inst✝ : Module R M
    ⊢ CharZero R
  -/
  refine ⟨fun m n h => @Nat.cast_injective M _ _ _ _ ?_⟩
  /-
    R : Type u_1
    inst✝³ : Semiring R
    M : Type u_3
    inst✝² : AddCommMonoidWithOne M
    inst✝¹ : CharZero M
    inst✝ : Module R M
    m n : Nat
    h : Eq ↑m ↑n
    ⊢ Eq ↑m ↑n
  -/
  rw [← nsmul_one, ← nsmul_one, ← Nat.cast_smul_eq_nsmul R, ← Nat.cast_smul_eq_nsmul R, h]
  /-
    🎉 no goals
  -/


theorem smul_right_injective [NoZeroSMulDivisors R M] {c : R} (hc : c ≠ 0) :
    Function.Injective (c • · : M → M) :=
  (injective_iff_map_eq_zero (smulAddHom R M c)).2 fun _ ha => (smul_eq_zero.mp ha).resolve_left hc


theorem smul_right_inj [NoZeroSMulDivisors R M] {c : R} (hc : c ≠ 0) {x y : M} :
    c • x = c • y ↔ x = y :=
  (smul_right_injective M hc).eq_iff


theorem self_eq_neg
    (R) (M) [Semiring R] [CharZero R] [AddCommGroup M] [Module R M] [NoZeroSMulDivisors R M]
    {v : M} : v = -v ↔ v = 0 := by
  /-
    R : Type u_3
    M : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : CharZero R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    v : M
    ⊢ Iff (Eq v (Neg.neg v)) (Eq v 0)
  -/
  rw [← two_nsmul_eq_zero R M, two_smul, add_eq_zero_iff_eq_neg]
  /-
    🎉 no goals
  -/


theorem neg_eq_self
    (R) (M) [Semiring R] [CharZero R] [AddCommGroup M] [Module R M] [NoZeroSMulDivisors R M]
    {v : M} : -v = v ↔ v = 0 := by
  /-
    R : Type u_3
    M : Type u_4
    inst✝⁴ : Semiring R
    inst✝³ : CharZero R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroSMulDivisors R M
    v : M
    ⊢ Iff (Eq (Neg.neg v) v) (Eq v 0)
  -/
  rw [eq_comm, self_eq_neg R M]
  /-
    🎉 no goals
  -/


theorem self_ne_neg
    (R) (M) [Semiring R] [CharZero R] [AddCommGroup M] [Module R M] [NoZeroSMulDivisors R M]
    {v : M} : v ≠ -v ↔ v ≠ 0 :=
  (self_eq_neg R M).not


theorem neg_ne_self
    (R) (M) [Semiring R] [CharZero R] [AddCommGroup M] [Module R M] [NoZeroSMulDivisors R M]
    {v : M} : -v ≠ v ↔ v ≠ 0 :=
  (neg_eq_self R M).not


theorem smul_left_injective {x : M} (hx : x ≠ 0) : Function.Injective fun c : R => c • x :=
  fun c d h =>
  sub_eq_zero.mp
    ((smul_eq_zero.mp
          (calc
            (c - d) • x = c • x - d • x := sub_smul c d x
            _ = 0 := sub_eq_zero.mpr h
            )).resolve_right
      hx)


instance [NoZeroSMulDivisors ℤ M] : NoZeroSMulDivisors ℕ M :=
                      /-
                        R : Type u_1
                        M : Type u_2
                        inst✝³ : Ring R
                        inst✝² : AddCommGroup M
                        inst✝¹ : Module R M
                        inst✝ : NoZeroSMulDivisors Int M
                        c : Nat
                        x : M
                        hcx : Eq (HSMul.hSMul c x) 0
                        ⊢ Or (Eq c 0) (Eq x 0)
                      -/
  ⟨fun {c x} hcx ↦ by rwa [← Nat.cast_smul_eq_nsmul ℤ, smul_eq_zero, Nat.cast_eq_zero] at hcx⟩
                      /-
                        🎉 no goals
                      -/


theorem NoZeroSMulDivisors.int_of_charZero
    (R) (M) [Ring R] [AddCommGroup M] [Module R M] [NoZeroSMulDivisors R M] [CharZero R] :
    NoZeroSMulDivisors ℤ M :=
                    /-
                      R : Type u_3
                      M : Type u_4
                      inst✝⁴ : Ring R
                      inst✝³ : AddCommGroup M
                      inst✝² : Module R M
                      inst✝¹ : NoZeroSMulDivisors R M
                      inst✝ : CharZero R
                      z : Int
                      x : M
                      h : Eq (HSMul.hSMul z x) 0
                      ⊢ Or (Eq z 0) (Eq x 0)
                    -/
  ⟨fun {z x} h ↦ by simpa [← smul_one_smul R z x] using h⟩
                    /-
                      🎉 no goals
                    -/


/-- Only a ring of characteristic zero can have a non-trivial module without additive or
scalar torsion. -/
theorem CharZero.of_noZeroSMulDivisors [Nontrivial M] [NoZeroSMulDivisors ℤ M] : CharZero R := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Nontrivial M
    inst✝ : NoZeroSMulDivisors Int M
    ⊢ CharZero R
  -/
  refine ⟨fun {n m h} ↦ ?_⟩
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Nontrivial M
    inst✝ : NoZeroSMulDivisors Int M
    n m : Nat
    h : Eq ↑n ↑m
    ⊢ Eq n m
  -/
  obtain ⟨x, hx⟩ := exists_ne (0 : M)
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Nontrivial M
    inst✝ : NoZeroSMulDivisors Int M
    n m : Nat
    h : Eq ↑n ↑m
    x : M
    hx : Ne x 0
    ⊢ Eq n m
  -/
  replace h : (n : ℤ) • x = (m : ℤ) • x := by simp [← Nat.cast_smul_eq_nsmul R, h]
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Nontrivial M
    inst✝ : NoZeroSMulDivisors Int M
    n m : Nat
    x : M
    hx : Ne x 0
    h : Eq (HSMul.hSMul (↑n) x) (HSMul.hSMul (↑m) x)
    ⊢ Eq n m
  -/
  simpa using smul_left_injective ℤ hx h
  /-
    🎉 no goals
  -/


instance [AddCommGroup M] [NoZeroSMulDivisors ℤ M] : NoZeroSMulDivisors ℕ M :=
                      /-
                        R : Type u_1
                        M : Type u_2
                        inst✝⁴ : Ring R
                        inst✝³ : AddCommGroup M
                        inst✝² : Module R M
                        inst✝¹ : AddCommGroup M
                        inst✝ : NoZeroSMulDivisors Int M
                        c : Nat
                        x : M
                        hcx : Eq (HSMul.hSMul c x) 0
                        ⊢ Or (Eq c 0) (Eq x 0)
                      -/
  ⟨fun {c x} hcx ↦ by rwa [← Nat.cast_smul_eq_nsmul ℤ c x, smul_eq_zero, Nat.cast_eq_zero] at hcx⟩
                      /-
                        🎉 no goals
                      -/


/-- This instance applies to `DivisionSemiring`s, in particular `NNReal` and `NNRat`. -/
instance (priority := 100) GroupWithZero.toNoZeroSMulDivisors : NoZeroSMulDivisors R M :=
  ⟨fun {a _} h ↦ or_iff_not_imp_left.2 fun ha ↦ (smul_eq_zero_iff_eq <| Units.mk0 a ha).1 h⟩


