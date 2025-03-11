/-- The `weight` of the finitely supported function `f : σ →₀ ℕ`
with respect to `w : σ → M` is the sum `∑ i, f i • w i`. -/
noncomputable def weight : (σ →₀ ℕ) →+ M :=
  (Finsupp.linearCombination ℕ w).toAddMonoidHom


@[deprecated weight (since := "2024-07-20")]
alias _root_.MvPolynomial.weightedDegree := weight


theorem weight_apply (f : σ →₀ ℕ) :
    weight w f = Finsupp.sum f (fun i c => c • w i) := rfl


@[deprecated weight_apply (since := "2024-07-20")]
alias _root_.MvPolynomial.weightedDegree_apply := weight_apply


/-- A weight function is nontorsion if its values are not torsion. -/
class NonTorsionWeight (w : σ → M) : Prop where
  eq_zero_of_smul_eq_zero {n : ℕ} {s : σ} (h : n • w s = 0)  : n = 0


/-- Without zero divisors, nonzero weight is a `NonTorsionWeight` -/
theorem nonTorsionWeight_of [NoZeroSMulDivisors ℕ M] (hw : ∀ i : σ, w i ≠ 0) :
    NonTorsionWeight w where
  eq_zero_of_smul_eq_zero {n s} h := by
    /-
      σ : Type u_1
      M : Type u_2
      w : σ → M
      inst✝¹ : AddCommMonoid M
      inst✝ : NoZeroSMulDivisors Nat M
      hw : ∀ (i : σ), Ne (w i) 0
      n : Nat
      s : σ
      h : Eq (HSMul.hSMul n (w s)) 0
      ⊢ Eq n 0
    -/
    rw [smul_eq_zero, or_iff_not_imp_right] at h
    /-
      σ : Type u_1
      M : Type u_2
      w : σ → M
      inst✝¹ : AddCommMonoid M
      inst✝ : NoZeroSMulDivisors Nat M
      hw : ∀ (i : σ), Ne (w i) 0
      n : Nat
      s : σ
      h : Not (Eq (w s) 0) → Eq n 0
      ⊢ Eq n 0
    -/
    exact h (hw s)
    /-
      🎉 no goals
    -/


theorem NonTorsionWeight.ne_zero [NonTorsionWeight w] (s : σ) :
    w s ≠ 0 := fun h ↦ by
  /-
    σ : Type u_1
    M : Type u_2
    w : σ → M
    inst✝¹ : AddCommMonoid M
    inst✝ : Finsupp.NonTorsionWeight w
    s : σ
    h : Eq (w s) 0
    ⊢ False
  -/
  rw [← one_smul ℕ (w s)] at h
  /-
    σ : Type u_1
    M : Type u_2
    w : σ → M
    inst✝¹ : AddCommMonoid M
    inst✝ : Finsupp.NonTorsionWeight w
    s : σ
    h : Eq (HSMul.hSMul 1 (w s)) 0
    ⊢ False
  -/
  apply Nat.zero_ne_one.symm
  /-
    σ : Type u_1
    M : Type u_2
    w : σ → M
    inst✝¹ : AddCommMonoid M
    inst✝ : Finsupp.NonTorsionWeight w
    s : σ
    h : Eq (HSMul.hSMul 1 (w s)) 0
    ⊢ Eq 1 0
  -/
  exact NonTorsionWeight.eq_zero_of_smul_eq_zero h
  /-
    🎉 no goals
  -/


variable {w} in
lemma weight_sub_single_add {f : σ →₀ ℕ} {i : σ} (hi : f i ≠ 0) :
    (f - single i 1).weight w + w i = f.weight w := by
  /-
    σ : Type u_1
    M : Type u_2
    w : σ → M
    inst✝ : AddCommMonoid M
    f : Finsupp σ Nat
    i : σ
    hi : Ne (f i) 0
    ⊢ Eq (HAdd.hAdd ((Finsupp.weight w) (HSub.hSub f (Finsupp.single i 1))) (w i)) …
  -/
  conv_rhs => rw [← sub_add_single_one_cancel hi, weight_apply]
  /-
    σ : Type u_1
    M : Type u_2
    w : σ → M
    inst✝ : AddCommMonoid M
    f : Finsupp σ Nat
    i : σ
    hi : Ne (f i) 0
    ⊢ Eq (HAdd.hAdd ((Finsupp.weight w) (HSub.hSub f (Finsupp.single i 1))) (w i)) …
  -/
  rw [sum_add_index', sum_single_index, one_smul, weight_apply]
  /-
    σ : Type u_1
    M : Type u_2
    w : σ → M
    inst✝ : AddCommMonoid M
    f : Finsupp σ Nat
    i : σ
    hi : Ne (f i) 0
    ⊢ Eq (HSMul.hSMul 0 (w i)) 0
  -/
  exacts [zero_smul .., fun _ ↦ zero_smul .., fun _ _ _ ↦ add_smul ..]
  /-
    🎉 no goals
  -/


theorem le_weight (w : σ → ℕ) {s : σ} (hs : w s ≠ 0) (f : σ →₀ ℕ) :
    f s ≤ weight w f := by
  classical
  simp only [weight_apply, Finsupp.sum]
  by_cases h : s ∈ f.support
  · rw [Finset.sum_eq_add_sum_diff_singleton h]
    refine le_trans ?_ (Nat.le_add_right _ _)
    apply Nat.le_mul_of_pos_right
    exact Nat.zero_lt_of_ne_zero hs
  · simp only [not_mem_support_iff] at h
    rw [h]
    apply zero_le


instance : SMulPosMono ℕ M :=
  ⟨fun b hb m m' h ↦ by
    /-
      σ : Type u_1
      M : Type u_2
      w✝ : σ → M
      inst✝ : OrderedAddCommMonoid M
      w : σ → M
      b : M
      hb : LE.le 0 b
      m m' : Nat
      h : LE.le m m'
      ⊢ LE.le (HSMul.hSMul m b) (HSMul.hSMul m' b)
    -/
    rw [← Nat.add_sub_of_le h, add_smul]
    /-
      σ : Type u_1
      M : Type u_2
      w✝ : σ → M
      inst✝ : OrderedAddCommMonoid M
      w : σ → M
      b : M
      hb : LE.le 0 b
      m m' : Nat
      h : LE.le m m'
      ⊢ LE.le (HSMul.hSMul m b) (HAdd.hAdd (HSMul.hSMul m b) (HSMul.hSMul (HSub.hSub …
    -/
    exact le_add_of_nonneg_right (nsmul_nonneg hb (m' - m))⟩
    /-
      🎉 no goals
    -/


variable {w} in
theorem le_weight_of_ne_zero (hw : ∀ s, 0 ≤ w s) {s : σ} {f : σ →₀ ℕ} (hs : f s ≠ 0) :
    w s ≤ weight w f := by
  classical
  simp only [weight_apply, Finsupp.sum]
  trans f s • w s
  · apply le_smul_of_one_le_left (hw s)
    exact Nat.one_le_iff_ne_zero.mpr hs
  · rw [← Finsupp.mem_support_iff] at hs
    rw [Finset.sum_eq_add_sum_diff_singleton hs]
    exact le_add_of_nonneg_right <| Finset.sum_nonneg <|
      fun i _ ↦ nsmul_nonneg (hw i) (f i)


theorem le_weight_of_ne_zero' {s : σ} {f : σ →₀ ℕ} (hs : f s ≠ 0) :
    w s ≤ weight w f :=
  le_weight_of_ne_zero (fun _ ↦ zero_le _) hs


/-- If `M` is a `CanonicallyOrderedAddCommMonoid`, then `weight f` is zero iff `f=0. -/
theorem weight_eq_zero_iff_eq_zero
    (w : σ → M) [NonTorsionWeight w] {f : σ →₀ ℕ} :
    weight w f = 0 ↔ f = 0 := by
  classical
  constructor
  · intro h
    ext s
    simp only [Finsupp.coe_zero, Pi.zero_apply]
    by_contra hs
    apply NonTorsionWeight.ne_zero w s
    rw [← nonpos_iff_eq_zero, ← h]
    exact le_weight_of_ne_zero' w hs
  · intro h
    rw [h, map_zero]


theorem finite_of_nat_weight_le [Finite σ] (w : σ → ℕ) (hw : ∀ x, w x ≠ 0) (n : ℕ) :
    {d : σ →₀ ℕ | weight w d ≤ n}.Finite := by
  classical
  set fg := Finset.antidiagonal (Finsupp.equivFunOnFinite.symm (Function.const σ n)) with hfg
  suffices {d : σ →₀ ℕ | weight w d ≤ n} ⊆ ↑(fg.image fun uv => uv.fst) by
    exact Set.Finite.subset (Finset.finite_toSet _) this
  intro d hd
  rw [hfg]
  simp only [Finset.coe_image, Set.mem_image, Finset.mem_coe,
    Finset.mem_antidiagonal, Prod.exists, exists_and_right, exists_eq_right]
  use Finsupp.equivFunOnFinite.symm (Function.const σ n) - d
  ext x
  simp only [Finsupp.coe_add, Finsupp.coe_tsub, Pi.add_apply, Pi.sub_apply,
    Finsupp.equivFunOnFinite_symm_apply_toFun, Function.const_apply]
  rw [add_comm]
  apply Nat.sub_add_cancel
  apply le_trans (le_weight w (hw x) d)
  simpa only [Set.mem_setOf_eq] using hd


/-- The degree of a finsupp function. -/
def degree (d : σ →₀ ℕ) := ∑ i ∈ d.support, d i


@[deprecated degree (since := "2024-07-20")]
alias _root_.MvPolynomial.degree := degree


@[simp]
theorem degree_add (a b : σ →₀ ℕ) : (a + b).degree = a.degree + b.degree :=
  sum_add_index' (h := fun _ ↦ id) (congrFun rfl) fun _ _ ↦ congrFun rfl


@[simp]
theorem degree_single (a : σ) (m : ℕ) : (Finsupp.single a m).degree = m := by
  /-
    σ : Type u_1
    a : σ
    m : Nat
    ⊢ Eq (Finsupp.single a m).degree m
  -/
  rw [degree, Finset.sum_eq_single a]
    /-
      σ : Type u_1
      a : σ
      m : Nat
      ⊢ Eq ((Finsupp.single a m) a) m
    -/
  · simp only [single_eq_same]
    /-
      🎉 no goals
    -/
    /-
      case h₀
      σ : Type u_1
      a : σ
      m : Nat
      ⊢ ∀ (b : σ), Membership.mem (Finsupp.single a m).support b → Ne b a → Eq ((Fin …
    -/
  · intro b _ hba
    /-
      case h₀
      σ : Type u_1
      a : σ
      m : Nat
      b : σ
      a✝ : Membership.mem (Finsupp.single a m).support b
      hba : Ne b a
      ⊢ Eq ((Finsupp.single a m) b) 0
    -/
    exact single_eq_of_ne hba.symm
    /-
      🎉 no goals
    -/
    /-
      case h₁
      σ : Type u_1
      a : σ
      m : Nat
      ⊢ Not (Membership.mem (Finsupp.single a m).support a) → Eq ((Finsupp.single a  …
    -/
  · intro ha
    /-
      case h₁
      σ : Type u_1
      a : σ
      m : Nat
      ha : Not (Membership.mem (Finsupp.single a m).support a)
      ⊢ Eq ((Finsupp.single a m) a) 0
    -/
    simp only [mem_support_iff, single_eq_same, ne_eq, Decidable.not_not] at ha
    /-
      case h₁
      σ : Type u_1
      a : σ
      m : Nat
      ha : Eq m 0
      ⊢ Eq ((Finsupp.single a m) a) 0
    -/
    rw [single_eq_same, ha]
    /-
      🎉 no goals
    -/


lemma degree_eq_zero_iff (d : σ →₀ ℕ) : degree d = 0 ↔ d = 0 := by
  simp only [degree, Finset.sum_eq_zero_iff, Finsupp.mem_support_iff, ne_eq, Decidable.not_imp_self,
    DFunLike.ext_iff, Finsupp.coe_zero, Pi.zero_apply]


@[deprecated degree_eq_zero_iff (since := "2024-07-20")]
alias _root_.MvPolynomial.degree_eq_zero_iff := degree_eq_zero_iff


@[simp]
                                                    /-
                                                      σ : Type u_1
                                                      ⊢ Eq (Finsupp.degree 0) 0
                                                    -/
theorem degree_zero : degree (0 : σ →₀ ℕ) = 0 := by rw [degree_eq_zero_iff]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem degree_eq_weight_one :
    degree (σ := σ) = weight 1 := by
  /-
    σ : Type u_1
    ⊢ Eq Finsupp.degree ⇑(Finsupp.weight 1)
  -/
  ext d
  /-
    case h
    σ : Type u_1
    d : Finsupp σ Nat
    ⊢ Eq d.degree ((Finsupp.weight 1) d)
  -/
  simp only [degree, weight_apply, Pi.one_apply, smul_eq_mul, mul_one, Finsupp.sum]
  /-
    🎉 no goals
  -/


@[deprecated degree_eq_weight_one (since := "2024-07-20")]
alias _root_.MvPolynomial.weightedDegree_one := degree_eq_weight_one


theorem le_degree (s : σ) (f : σ →₀ ℕ) : f s ≤ degree f  := by
  /-
    σ : Type u_1
    s : σ
    f : Finsupp σ Nat
    ⊢ LE.le (f s) f.degree
  -/
  rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    s : σ
    f : Finsupp σ Nat
    ⊢ LE.le (f s) ((Finsupp.weight 1) f)
  -/
  apply le_weight
  /-
    case hs
    σ : Type u_1
    s : σ
    f : Finsupp σ Nat
    ⊢ Ne (1 s) 0
  -/
  simp only [Pi.one_apply, ne_eq, one_ne_zero, not_false_eq_true]
  /-
    🎉 no goals
  -/


theorem finite_of_degree_le [Finite σ] (n : ℕ) :
    {f : σ →₀ ℕ | degree f ≤ n}.Finite := by
  /-
    σ : Type u_1
    inst✝ : Finite σ
    n : Nat
    ⊢ (setOf fun f => LE.le f.degree n).Finite
  -/
  simp_rw [degree_eq_weight_one]
  /-
    σ : Type u_1
    inst✝ : Finite σ
    n : Nat
    ⊢ (setOf fun f => LE.le ((Finsupp.weight 1) f) n).Finite
  -/
  refine finite_of_nat_weight_le (Function.const σ 1) ?_ n
  /-
    σ : Type u_1
    inst✝ : Finite σ
    n : Nat
    ⊢ ∀ (x : σ), Ne (Function.const σ 1 x) 0
  -/
  intro _
  /-
    σ : Type u_1
    inst✝ : Finite σ
    n : Nat
    x✝ : σ
    ⊢ Ne (Function.const σ 1 x✝) 0
  -/
  simp only [Function.const_apply, ne_eq, one_ne_zero, not_false_eq_true]
  /-
    🎉 no goals
  -/


