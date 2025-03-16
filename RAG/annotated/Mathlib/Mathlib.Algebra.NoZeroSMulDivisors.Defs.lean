/-- `NoZeroSMulDivisors R M` states that a scalar multiple is `0` only if either argument is `0`.
This is a version of saying that `M` is torsion free, without assuming `R` is zero-divisor free.

The main application of `NoZeroSMulDivisors R M`, when `M` is a module,
is the result `smul_eq_zero`: a scalar multiple is `0` iff either argument is `0`.

It is a generalization of the `NoZeroDivisors` class to heterogeneous multiplication.
-/
@[mk_iff]
class NoZeroSMulDivisors (R M : Type*) [Zero R] [Zero M] [SMul R M] : Prop where
  /-- If scalar multiplication yields zero, either the scalar or the vector was zero. -/
  eq_zero_or_eq_zero_of_smul_eq_zero : ∀ {c : R} {x : M}, c • x = 0 → c = 0 ∨ x = 0


/-- Pullback a `NoZeroSMulDivisors` instance along an injective function. -/
theorem Function.Injective.noZeroSMulDivisors {R M N : Type*} [Zero R] [Zero M] [Zero N]
    [SMul R M] [SMul R N] [NoZeroSMulDivisors R N] (f : M → N) (hf : Function.Injective f)
    (h0 : f 0 = 0) (hs : ∀ (c : R) (x : M), f (c • x) = c • f x) : NoZeroSMulDivisors R M :=
  ⟨fun {_ _} h =>
                                                                               /-
                                                                                 R : Type u_3
                                                                                 M : Type u_4
                                                                                 N : Type u_5
                                                                                 inst✝⁵ : Zero R
                                                                                 inst✝⁴ : Zero M
                                                                                 inst✝³ : Zero N
                                                                                 inst✝² : SMul R M
                                                                                 inst✝¹ : SMul R N
                                                                                 inst✝ : NoZeroSMulDivisors R N
                                                                                 f : M → N
                                                                                 hf : Function.Injective f
                                                                                 h0 : Eq (f 0) 0
                                                                                 hs : ∀ (c : R) (x : M), Eq (f (HSMul.hSMul c x)) (HSMul.hSMul c (f x))
                                                                                 x✝¹ : R
                                                                                 x✝ : M
                                                                                 h : Eq (HSMul.hSMul x✝¹ x✝) 0
                                                                                 ⊢ Eq (HSMul.hSMul x✝¹ (f x✝)) 0
                                                                               -/
    Or.imp_right (@hf _ _) <| h0.symm ▸ eq_zero_or_eq_zero_of_smul_eq_zero (by rw [← hs, h, h0])⟩
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/

-- See note [lower instance priority]

instance (priority := 100) NoZeroDivisors.toNoZeroSMulDivisors [Zero R] [Mul R]
    [NoZeroDivisors R] : NoZeroSMulDivisors R R :=
  ⟨fun {_ _} => eq_zero_or_eq_zero_of_mul_eq_zero⟩


theorem smul_ne_zero [Zero R] [Zero M] [SMul R M] [NoZeroSMulDivisors R M] {c : R} {x : M}
    (hc : c ≠ 0) (hx : x ≠ 0) : c • x ≠ 0 := fun h =>
  (eq_zero_or_eq_zero_of_smul_eq_zero h).elim hc hx


@[simp]
theorem smul_eq_zero : c • x = 0 ↔ c = 0 ∨ x = 0 :=
  ⟨eq_zero_or_eq_zero_of_smul_eq_zero, fun h =>
    h.elim (fun h => h.symm ▸ zero_smul R x) fun h => h.symm ▸ smul_zero c⟩


                                                           /-
                                                             R : Type u_1
                                                             M : Type u_2
                                                             inst✝³ : Zero R
                                                             inst✝² : Zero M
                                                             inst✝¹ : SMulWithZero R M
                                                             inst✝ : NoZeroSMulDivisors R M
                                                             c : R
                                                             x : M
                                                             ⊢ Iff (Ne (HSMul.hSMul c x) 0) (And (Ne c 0) (Ne x 0))
                                                           -/
theorem smul_ne_zero_iff : c • x ≠ 0 ↔ c ≠ 0 ∧ x ≠ 0 := by rw [Ne, smul_eq_zero, not_or]
                                                           /-
                                                             🎉 no goals
                                                           -/


                                                                   /-
                                                                     R : Type u_1
                                                                     M : Type u_2
                                                                     inst✝³ : Zero R
                                                                     inst✝² : Zero M
                                                                     inst✝¹ : SMulWithZero R M
                                                                     inst✝ : NoZeroSMulDivisors R M
                                                                     c : R
                                                                     x : M
                                                                     hx : Ne x 0
                                                                     ⊢ Iff (Eq (HSMul.hSMul c x) 0) (Eq c 0)
                                                                   -/
lemma smul_eq_zero_iff_left (hx : x ≠ 0) : c • x = 0 ↔ c = 0 := by simp [hx]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

                                                                    /-
                                                                      R : Type u_1
                                                                      M : Type u_2
                                                                      inst✝³ : Zero R
                                                                      inst✝² : Zero M
                                                                      inst✝¹ : SMulWithZero R M
                                                                      inst✝ : NoZeroSMulDivisors R M
                                                                      c : R
                                                                      x : M
                                                                      hc : Ne c 0
                                                                      ⊢ Iff (Eq (HSMul.hSMul c x) 0) (Eq x 0)
                                                                    -/
lemma smul_eq_zero_iff_right (hc : c ≠ 0) : c • x = 0 ↔ x = 0 := by simp [hc]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/

                                                                   /-
                                                                     R : Type u_1
                                                                     M : Type u_2
                                                                     inst✝³ : Zero R
                                                                     inst✝² : Zero M
                                                                     inst✝¹ : SMulWithZero R M
                                                                     inst✝ : NoZeroSMulDivisors R M
                                                                     c : R
                                                                     x : M
                                                                     hx : Ne x 0
                                                                     ⊢ Iff (Ne (HSMul.hSMul c x) 0) (Ne c 0)
                                                                   -/
lemma smul_ne_zero_iff_left (hx : x ≠ 0) : c • x ≠ 0 ↔ c ≠ 0 := by simp [hx]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

                                                                    /-
                                                                      R : Type u_1
                                                                      M : Type u_2
                                                                      inst✝³ : Zero R
                                                                      inst✝² : Zero M
                                                                      inst✝¹ : SMulWithZero R M
                                                                      inst✝ : NoZeroSMulDivisors R M
                                                                      c : R
                                                                      x : M
                                                                      hc : Ne c 0
                                                                      ⊢ Iff (Ne (HSMul.hSMul c x) 0) (Ne x 0)
                                                                    -/
lemma smul_ne_zero_iff_right (hc : c ≠ 0) : c • x ≠ 0 ↔ x ≠ 0 := by simp [hc]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


