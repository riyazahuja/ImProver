instance [Zero R] [One R] : One (HahnSeries Γ R) :=
  ⟨single 0 1⟩


open Classical in
@[simp]
theorem one_coeff [Zero R] [One R] {a : Γ} :
    (1 : HahnSeries Γ R).coeff a = if a = 0 then 1 else 0 :=
  single_coeff


@[simp]
theorem single_zero_one [Zero R] [One R] : single (0 : Γ) (1 : R) = 1 :=
  rfl


@[simp]
theorem support_one [MulZeroOneClass R] [Nontrivial R] : support (1 : HahnSeries Γ R) = {0} :=
  support_single_of_ne one_ne_zero


@[simp]
theorem orderTop_one [MulZeroOneClass R] [Nontrivial R] : orderTop (1 : HahnSeries Γ R) = 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝³ : Zero Γ
    inst✝² : PartialOrder Γ
    inst✝¹ : MulZeroOneClass R
    inst✝ : Nontrivial R
    ⊢ Eq (HahnSeries.orderTop 1) 0
  -/
  rw [← single_zero_one, orderTop_single one_ne_zero, WithTop.coe_eq_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem order_one [MulZeroOneClass R] : order (1 : HahnSeries Γ R) = 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : Zero Γ
    inst✝¹ : PartialOrder Γ
    inst✝ : MulZeroOneClass R
    ⊢ Eq (HahnSeries.order 1) 0
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      Γ : Type u_1
      R : Type u_3
      inst✝² : Zero Γ
      inst✝¹ : PartialOrder Γ
      inst✝ : MulZeroOneClass R
      h✝ : Subsingleton R
      ⊢ Eq (HahnSeries.order 1) 0
    -/
  · rw [Subsingleton.elim (1 : HahnSeries Γ R) 0, order_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      Γ : Type u_1
      R : Type u_3
      inst✝² : Zero Γ
      inst✝¹ : PartialOrder Γ
      inst✝ : MulZeroOneClass R
      h✝ : Nontrivial R
      ⊢ Eq (HahnSeries.order 1) 0
    -/
  · exact order_single one_ne_zero
    /-
      🎉 no goals
    -/


@[simp]
theorem leadingCoeff_one [MulZeroOneClass R] : (1 : HahnSeries Γ R).leadingCoeff = 1 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : Zero Γ
    inst✝¹ : PartialOrder Γ
    inst✝ : MulZeroOneClass R
    ⊢ Eq (HahnSeries.leadingCoeff 1) 1
  -/
  simp [leadingCoeff_eq]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma map_one [MonoidWithZero R] [MonoidWithZero S] (f : R →*₀ S) :
    (1 : HahnSeries Γ R).map f = (1 : HahnSeries Γ S) := by
  /-
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝³ : Zero Γ
    inst✝² : PartialOrder Γ
    inst✝¹ : MonoidWithZero R
    inst✝ : MonoidWithZero S
    f : MonoidWithZeroHom R S
    ⊢ Eq (HahnSeries.map 1 f) 1
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝³ : Zero Γ
    inst✝² : PartialOrder Γ
    inst✝¹ : MonoidWithZero R
    inst✝ : MonoidWithZero S
    f : MonoidWithZeroHom R S
    g : Γ
    ⊢ Eq ((HahnSeries.map 1 f).coeff g) (HahnSeries.coeff 1 g)
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : g = 0 <;> simp [h]
                         /-
                           🎉 no goals
                         -/


/-- We introduce a type alias for `HahnSeries` in order to work with scalar multiplication by
series. If we wrote a `SMul (HahnSeries Γ R) (HahnSeries Γ V)` instance, then when
`V = HahnSeries Γ R`, we would have two different actions of `HahnSeries Γ R` on `HahnSeries Γ V`.
See `Mathlib.Algebra.Polynomial.Module` for more discussion on this problem. -/
@[nolint unusedArguments]
def HahnModule (Γ R V : Type*) [PartialOrder Γ] [Zero V] [SMul R V] :=
  HahnSeries Γ V


/-- The casting function to the type synonym. -/
def of (R : Type*) [SMul R V] : HahnSeries Γ V ≃ HahnModule Γ R V :=
  Equiv.refl _


/-- Recursion principle to reduce a result about the synonym to the original type. -/
@[elab_as_elim]
def rec {motive : HahnModule Γ R V → Sort*} (h : ∀ x : HahnSeries Γ V, motive (of R x)) :
    ∀ x, motive x :=
  fun x => h <| (of R).symm x


@[ext]
theorem ext (x y : HahnModule Γ R V) (h : ((of R).symm x).coeff = ((of R).symm y).coeff) : x = y :=
  (of R).symm.injective <| HahnSeries.coeff_inj.1 h


instance instAddCommMonoid : AddCommMonoid (HahnModule Γ R V) :=
  inferInstanceAs <| AddCommMonoid (HahnSeries Γ V)

instance instBaseSMul {V} [Monoid R] [AddMonoid V] [DistribMulAction R V] :
    SMul R (HahnModule Γ R V) :=
  inferInstanceAs <| SMul R (HahnSeries Γ V)


@[simp] theorem of_zero : of R (0 : HahnSeries Γ V) = 0 := rfl

@[simp] theorem of_add (x y : HahnSeries Γ V) : of R (x + y) = of R x + of R y := rfl


@[simp] theorem of_symm_zero : (of R).symm (0 : HahnModule Γ R V) = 0 := rfl

@[simp] theorem of_symm_add (x y : HahnModule Γ R V) :
  (of R).symm (x + y) = (of R).symm x + (of R).symm y := rfl


instance instSMul [Zero R] : SMul (HahnSeries Γ R) (HahnModule Γ' R V) where
  smul x y := (of R) {
    coeff := fun a =>
      ∑ ij ∈ VAddAntidiagonal x.isPWO_support ((of R).symm y).isPWO_support a,
        x.coeff ij.fst • ((of R).symm y).coeff ij.snd
    isPWO_support' :=
        haveI h :
          { a : Γ' |
              (∑ ij ∈ VAddAntidiagonal x.isPWO_support ((of R).symm y).isPWO_support a,
                  x.coeff ij.fst • ((of R).symm y).coeff ij.snd) ≠ 0 } ⊆
            { a : Γ' | (VAddAntidiagonal x.isPWO_support
              ((of R).symm y).isPWO_support a).Nonempty } := by
          /-
            Γ : Type u_1
            Γ' : Type u_2
            R : Type u_3
            S : Type u_4
            V : Type u_5
            inst✝⁶ : PartialOrder Γ
            inst✝⁵ : AddCommMonoid V
            inst✝⁴ : SMul R V
            inst✝³ : PartialOrder Γ'
            inst✝² : VAdd Γ Γ'
            inst✝¹ : IsOrderedCancelVAdd Γ Γ'
            inst✝ : Zero R
            x : HahnSeries Γ R
            y : HahnModule Γ' R V
            ⊢ HasSubset.Subset (setOf fun a => Ne ((Finset.VAddAntidiagonal ⋯ ⋯ a).sum fun …
          -/
          intro a ha
          /-
            Γ : Type u_1
            Γ' : Type u_2
            R : Type u_3
            S : Type u_4
            V : Type u_5
            inst✝⁶ : PartialOrder Γ
            inst✝⁵ : AddCommMonoid V
            inst✝⁴ : SMul R V
            inst✝³ : PartialOrder Γ'
            inst✝² : VAdd Γ Γ'
            inst✝¹ : IsOrderedCancelVAdd Γ Γ'
            inst✝ : Zero R
            x : HahnSeries Γ R
            y : HahnModule Γ' R V
            a : Γ'
            ha : Membership.mem (setOf fun a => Ne ((Finset.VAddAntidiagonal ⋯ ⋯ a).sum fu …
            ⊢ Membership.mem (setOf fun a => (Finset.VAddAntidiagonal ⋯ ⋯ a).Nonempty) a
          -/
          contrapose! ha
          /-
            Γ : Type u_1
            Γ' : Type u_2
            R : Type u_3
            S : Type u_4
            V : Type u_5
            inst✝⁶ : PartialOrder Γ
            inst✝⁵ : AddCommMonoid V
            inst✝⁴ : SMul R V
            inst✝³ : PartialOrder Γ'
            inst✝² : VAdd Γ Γ'
            inst✝¹ : IsOrderedCancelVAdd Γ Γ'
            inst✝ : Zero R
            x : HahnSeries Γ R
            y : HahnModule Γ' R V
            a : Γ'
            ha : Not (Membership.mem (setOf fun a => (Finset.VAddAntidiagonal ⋯ ⋯ a).Nonem …
            ⊢ Not (Membership.mem (setOf fun a => Ne ((Finset.VAddAntidiagonal ⋯ ⋯ a).sum  …
          -/
          simp [not_nonempty_iff_eq_empty.1 ha]
          /-
            🎉 no goals
          -/
        isPWO_support_vaddAntidiagonal.mono h }


theorem smul_coeff [Zero R] (x : HahnSeries Γ R) (y : HahnModule Γ' R V) (a : Γ') :
    ((of R).symm <| x • y).coeff a =
      ∑ ij ∈ VAddAntidiagonal x.isPWO_support ((of R).symm y).isPWO_support a,
        x.coeff ij.fst • ((of R).symm y).coeff ij.snd :=
  rfl


instance instBaseSMulZeroClass [SMulZeroClass R V] :
    SMulZeroClass R (HahnModule Γ R V) :=
  inferInstanceAs <| SMulZeroClass R (HahnSeries Γ V)


@[simp] theorem of_smul [SMulZeroClass R V] (r : R) (x : HahnSeries Γ V) :
  (of R) (r • x) = r • (of R) x := rfl

@[simp] theorem of_symm_smul [SMulZeroClass R V] (r : R) (x : HahnModule Γ R V) :
  (of R).symm (r • x) = r • (of R).symm x := rfl


instance instSMulZeroClass [SMulZeroClass R V] :
    SMulZeroClass (HahnSeries Γ R) (HahnModule Γ' R V) where
  smul_zero x := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : Zero R
      inst✝ : SMulZeroClass R V
      x : HahnSeries Γ R
      ⊢ Eq (HSMul.hSMul x 0) 0
    -/
    ext
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : Zero R
      inst✝ : SMulZeroClass R V
      x : HahnSeries Γ R
      x✝ : Γ'
      ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul x 0)).coeff x✝) (((HahnModule.of R) …
    -/
    simp [smul_coeff]
    /-
      🎉 no goals
    -/


theorem smul_coeff_right [SMulZeroClass R V] {x : HahnSeries Γ R} {y : HahnModule Γ' R V} {a : Γ'}
    {s : Set Γ'} (hs : s.IsPWO) (hys : ((of R).symm y).support ⊆ s) :
    ((of R).symm <| x • y).coeff a =
      ∑ ij ∈ VAddAntidiagonal x.isPWO_support hs a,
        x.coeff ij.fst • ((of R).symm y).coeff ij.snd := by
  classical
  rw [smul_coeff]
  apply sum_subset_zero_on_sdiff (vaddAntidiagonal_mono_right hys) _ fun _ _ => rfl
  intro b hb
  simp only [not_and, mem_sdiff, mem_vaddAntidiagonal, HahnSeries.mem_support, not_imp_not] at hb
  rw [hb.2 hb.1.1 hb.1.2.2, smul_zero]


theorem smul_coeff_left [SMulWithZero R V] {x : HahnSeries Γ R}
    {y : HahnModule Γ' R V} {a : Γ'} {s : Set Γ}
    (hs : s.IsPWO) (hxs : x.support ⊆ s) :
    ((of R).symm <| x • y).coeff a =
      ∑ ij ∈ VAddAntidiagonal hs ((of R).symm y).isPWO_support a,
        x.coeff ij.fst • ((of R).symm y).coeff ij.snd := by
  classical
  rw [smul_coeff]
  apply sum_subset_zero_on_sdiff (vaddAntidiagonal_mono_left hxs) _ fun _ _ => rfl
  intro b hb
  simp only [not_and', mem_sdiff, mem_vaddAntidiagonal, HahnSeries.mem_support, not_ne_iff] at hb
  rw [hb.2 ⟨hb.1.2.1, hb.1.2.2⟩, zero_smul]


theorem smul_add [Zero R] [DistribSMul R V] (x : HahnSeries Γ R) (y z : HahnModule Γ' R V) :
    x • (y + z) = x • y + x • z := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : Zero R
    inst✝ : DistribSMul R V
    x : HahnSeries Γ R
    y z : HahnModule Γ' R V
    ⊢ Eq (HSMul.hSMul x (HAdd.hAdd y z)) (HAdd.hAdd (HSMul.hSMul x y) (HSMul.hSMul …
  -/
  ext k
  /-
    case h.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : Zero R
    inst✝ : DistribSMul R V
    x : HahnSeries Γ R
    y z : HahnModule Γ' R V
    k : Γ'
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul x (HAdd.hAdd y z))).coeff k) (((Hah …
  -/
  have hwf := ((of R).symm y).isPWO_support.union ((of R).symm z).isPWO_support
  /-
    case h.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : Zero R
    inst✝ : DistribSMul R V
    x : HahnSeries Γ R
    y z : HahnModule Γ' R V
    k : Γ'
    hwf : (Union.union ((HahnModule.of R).symm y).support ((HahnModule.of R).symm  …
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul x (HAdd.hAdd y z))).coeff k) (((Hah …
  -/
  rw [smul_coeff_right hwf, of_symm_add]
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : Zero R
      inst✝ : DistribSMul R V
      x : HahnSeries Γ R
      y z : HahnModule Γ' R V
      k : Γ'
      hwf : (Union.union ((HahnModule.of R).symm y).support ((HahnModule.of R).symm  …
      ⊢ Eq ((Finset.VAddAntidiagonal ⋯ hwf k).sum fun ij => HSMul.hSMul (x.coeff ij. …
    -/
  · simp_all only [HahnSeries.add_coeff', Pi.add_apply, smul_add, of_symm_add]
    rw [smul_coeff_right hwf Set.subset_union_right,
      smul_coeff_right hwf Set.subset_union_left]
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : Zero R
      inst✝ : DistribSMul R V
      x : HahnSeries Γ R
      y z : HahnModule Γ' R V
      k : Γ'
      hwf : (Union.union ((HahnModule.of R).symm y).support ((HahnModule.of R).symm  …
      ⊢ Eq ((Finset.VAddAntidiagonal ⋯ hwf k).sum fun x_1 => HSMul.hSMul (x.coeff x_ …
    -/
    simp_all [sum_add_distrib]
    /-
      🎉 no goals
    -/
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : Zero R
      inst✝ : DistribSMul R V
      x : HahnSeries Γ R
      y z : HahnModule Γ' R V
      k : Γ'
      hwf : (Union.union ((HahnModule.of R).symm y).support ((HahnModule.of R).symm  …
      ⊢ HasSubset.Subset ((HahnModule.of R).symm (HAdd.hAdd y z)).support (Union.uni …
    -/
  · intro b
    simp_all only [Set.isPWO_union, HahnSeries.isPWO_support, and_self, of_symm_add,
      HahnSeries.add_coeff', Pi.add_apply, ne_eq, Set.mem_union, HahnSeries.mem_support]
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : Zero R
      inst✝ : DistribSMul R V
      x : HahnSeries Γ R
      y z : HahnModule Γ' R V
      k b : Γ'
      ⊢ Not (Eq (HAdd.hAdd (((HahnModule.of R).symm y).coeff b) (((HahnModule.of R). …
    -/
    contrapose!
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : Zero R
      inst✝ : DistribSMul R V
      x : HahnSeries Γ R
      y z : HahnModule Γ' R V
      k b : Γ'
      ⊢ And (Eq (((HahnModule.of R).symm y).coeff b) 0) (Eq (((HahnModule.of R).symm …
    -/
    intro h
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : Zero R
      inst✝ : DistribSMul R V
      x : HahnSeries Γ R
      y z : HahnModule Γ' R V
      k b : Γ'
      h : And (Eq (((HahnModule.of R).symm y).coeff b) 0) (Eq (((HahnModule.of R).sy …
      ⊢ Eq (HAdd.hAdd (((HahnModule.of R).symm y).coeff b) (((HahnModule.of R).symm  …
    -/
    rw [h.1, h.2, add_zero]
    /-
      🎉 no goals
    -/


instance instDistribSMul [MonoidWithZero R] [DistribSMul R V] : DistribSMul (HahnSeries Γ R)
    (HahnModule Γ' R V) where
  smul_add := smul_add


theorem add_smul [AddCommMonoid R] [SMulWithZero R V] {x y : HahnSeries Γ R}
    {z : HahnModule Γ' R V} (h : ∀ (r s : R) (u : V), (r + s) • u = r • u + s • u) :
    (x + y) • z = x • z + y • z := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : AddCommMonoid R
    inst✝ : SMulWithZero R V
    x y : HahnSeries Γ R
    z : HahnModule Γ' R V
    h : ∀ (r s : R) (u : V), Eq (HSMul.hSMul (HAdd.hAdd r s) u) (HAdd.hAdd (HSMul. …
    ⊢ Eq (HSMul.hSMul (HAdd.hAdd x y) z) (HAdd.hAdd (HSMul.hSMul x z) (HSMul.hSMul …
  -/
  ext a
  /-
    case h.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : AddCommMonoid R
    inst✝ : SMulWithZero R V
    x y : HahnSeries Γ R
    z : HahnModule Γ' R V
    h : ∀ (r s : R) (u : V), Eq (HSMul.hSMul (HAdd.hAdd r s) u) (HAdd.hAdd (HSMul. …
    a : Γ'
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul (HAdd.hAdd x y) z)).coeff a) (((Hah …
  -/
  have hwf := x.isPWO_support.union y.isPWO_support
  /-
    case h.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : AddCommMonoid R
    inst✝ : SMulWithZero R V
    x y : HahnSeries Γ R
    z : HahnModule Γ' R V
    h : ∀ (r s : R) (u : V), Eq (HSMul.hSMul (HAdd.hAdd r s) u) (HAdd.hAdd (HSMul. …
    a : Γ'
    hwf : (Union.union x.support y.support).IsPWO
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul (HAdd.hAdd x y) z)).coeff a) (((Hah …
  -/
  rw [smul_coeff_left hwf, HahnSeries.add_coeff', of_symm_add]
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : AddCommMonoid R
      inst✝ : SMulWithZero R V
      x y : HahnSeries Γ R
      z : HahnModule Γ' R V
      h : ∀ (r s : R) (u : V), Eq (HSMul.hSMul (HAdd.hAdd r s) u) (HAdd.hAdd (HSMul. …
      a : Γ'
      hwf : (Union.union x.support y.support).IsPWO
      ⊢ Eq ((Finset.VAddAntidiagonal hwf ⋯ a).sum fun ij => HSMul.hSMul (HAdd.hAdd x …
    -/
  · simp_all only [Pi.add_apply, HahnSeries.add_coeff']
    rw [smul_coeff_left hwf Set.subset_union_right,
      smul_coeff_left hwf Set.subset_union_left]
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : AddCommMonoid R
      inst✝ : SMulWithZero R V
      x y : HahnSeries Γ R
      z : HahnModule Γ' R V
      h : ∀ (r s : R) (u : V), Eq (HSMul.hSMul (HAdd.hAdd r s) u) (HAdd.hAdd (HSMul. …
      a : Γ'
      hwf : (Union.union x.support y.support).IsPWO
      ⊢ Eq ((Finset.VAddAntidiagonal hwf ⋯ a).sum fun x_1 => HAdd.hAdd (HSMul.hSMul  …
    -/
    simp only [HahnSeries.add_coeff, h, sum_add_distrib]
    /-
      🎉 no goals
    -/
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : AddCommMonoid R
      inst✝ : SMulWithZero R V
      x y : HahnSeries Γ R
      z : HahnModule Γ' R V
      h : ∀ (r s : R) (u : V), Eq (HSMul.hSMul (HAdd.hAdd r s) u) (HAdd.hAdd (HSMul. …
      a : Γ'
      hwf : (Union.union x.support y.support).IsPWO
      ⊢ HasSubset.Subset (HAdd.hAdd x y).support (Union.union x.support y.support)
    -/
  · intro b
    simp_all only [Set.isPWO_union, HahnSeries.isPWO_support, and_self, HahnSeries.mem_support,
      HahnSeries.add_coeff, ne_eq, Set.mem_union, Set.mem_setOf_eq, mem_support]
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : AddCommMonoid R
      inst✝ : SMulWithZero R V
      x y : HahnSeries Γ R
      z : HahnModule Γ' R V
      h : ∀ (r s : R) (u : V), Eq (HSMul.hSMul (HAdd.hAdd r s) u) (HAdd.hAdd (HSMul. …
      a : Γ'
      b : Γ
      ⊢ Not (Eq (HAdd.hAdd (x.coeff b) (y.coeff b)) 0) → Or (Not (Eq (x.coeff b) 0)) …
    -/
    contrapose!
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : AddCommMonoid R
      inst✝ : SMulWithZero R V
      x y : HahnSeries Γ R
      z : HahnModule Γ' R V
      h : ∀ (r s : R) (u : V), Eq (HSMul.hSMul (HAdd.hAdd r s) u) (HAdd.hAdd (HSMul. …
      a : Γ'
      b : Γ
      ⊢ And (Eq (x.coeff b) 0) (Eq (y.coeff b) 0) → Eq (HAdd.hAdd (x.coeff b) (y.coe …
    -/
    intro h
    /-
      case h.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : AddCommMonoid R
      inst✝ : SMulWithZero R V
      x y : HahnSeries Γ R
      z : HahnModule Γ' R V
      h✝ : ∀ (r s : R) (u : V), Eq (HSMul.hSMul (HAdd.hAdd r s) u) (HAdd.hAdd (HSMul …
      a : Γ'
      b : Γ
      h : And (Eq (x.coeff b) 0) (Eq (y.coeff b) 0)
      ⊢ Eq (HAdd.hAdd (x.coeff b) (y.coeff b)) 0
    -/
    rw [h.1, h.2, add_zero]
    /-
      🎉 no goals
    -/


theorem single_smul_coeff_add [MulZeroClass R] [SMulWithZero R V] {r : R} {x : HahnModule Γ' R V}
    {a : Γ'} {b : Γ} :
    ((of R).symm (HahnSeries.single b r • x)).coeff (b +ᵥ a) = r • ((of R).symm x).coeff a := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    r : R
    x : HahnModule Γ' R V
    a : Γ'
    b : Γ
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul ((HahnSeries.single b) r) x)).coeff …
  -/
  by_cases hr : r = 0
  · simp_all only [map_zero, zero_smul, smul_coeff, HahnSeries.support_zero, HahnSeries.zero_coeff,
    sum_const_zero]
  simp only [hr, smul_coeff, smul_coeff, HahnSeries.support_single_of_ne, ne_eq, not_false_iff,
    smul_eq_mul]
  /-
    case neg
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    r : R
    x : HahnModule Γ' R V
    a : Γ'
    b : Γ
    hr : Not (Eq r 0)
    ⊢ Eq ((Finset.VAddAntidiagonal ⋯ ⋯ (HVAdd.hVAdd b a)).sum fun x_1 => HSMul.hSM …
  -/
  by_cases hx : ((of R).symm x).coeff a = 0
    /-
      case pos
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      r : R
      x : HahnModule Γ' R V
      a : Γ'
      b : Γ
      hr : Not (Eq r 0)
      hx : Eq (((HahnModule.of R).symm x).coeff a) 0
      ⊢ Eq ((Finset.VAddAntidiagonal ⋯ ⋯ (HVAdd.hVAdd b a)).sum fun x_1 => HSMul.hSM …
    -/
  · simp only [hx, smul_zero]
    /-
      case pos
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      r : R
      x : HahnModule Γ' R V
      a : Γ'
      b : Γ
      hr : Not (Eq r 0)
      hx : Eq (((HahnModule.of R).symm x).coeff a) 0
      ⊢ Eq ((Finset.VAddAntidiagonal ⋯ ⋯ (HVAdd.hVAdd b a)).sum fun x_1 => HSMul.hSM …
    -/
    rw [sum_congr _ fun _ _ => rfl, sum_empty]
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      r : R
      x : HahnModule Γ' R V
      a : Γ'
      b : Γ
      hr : Not (Eq r 0)
      hx : Eq (((HahnModule.of R).symm x).coeff a) 0
      ⊢ Eq (Finset.VAddAntidiagonal ⋯ ⋯ (HVAdd.hVAdd b a)) EmptyCollection.emptyColl …
    -/
    ext ⟨a1, a2⟩
    simp only [not_mem_empty, not_and, Set.mem_singleton_iff, Classical.not_not,
      mem_vaddAntidiagonal, Set.mem_setOf_eq, iff_false]
    /-
      case h.mk
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      r : R
      x : HahnModule Γ' R V
      a : Γ'
      b : Γ
      hr : Not (Eq r 0)
      hx : Eq (((HahnModule.of R).symm x).coeff a) 0
      a1 : Γ
      a2 : Γ'
      ⊢ Eq a1 b → Membership.mem ((HahnModule.of R).symm x).support a2 → Not (Eq (HV …
    -/
    rintro rfl h2 h1
    /-
      case h.mk
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      r : R
      x : HahnModule Γ' R V
      a : Γ'
      hr : Not (Eq r 0)
      hx : Eq (((HahnModule.of R).symm x).coeff a) 0
      a1 : Γ
      a2 : Γ'
      h2 : Membership.mem ((HahnModule.of R).symm x).support a2
      h1 : Eq (HVAdd.hVAdd a1 a2) (HVAdd.hVAdd a1 a)
      ⊢ False
    -/
    rw [IsCancelVAdd.left_cancel a1 a2 a h1] at h2
    /-
      case h.mk
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      r : R
      x : HahnModule Γ' R V
      a : Γ'
      hr : Not (Eq r 0)
      hx : Eq (((HahnModule.of R).symm x).coeff a) 0
      a1 : Γ
      a2 : Γ'
      h2 : Membership.mem ((HahnModule.of R).symm x).support a
      h1 : Eq (HVAdd.hVAdd a1 a2) (HVAdd.hVAdd a1 a)
      ⊢ False
    -/
    exact h2 hx
    /-
      🎉 no goals
    -/
  trans ∑ ij ∈ {(b, a)},
    (HahnSeries.single b r).coeff ij.fst • ((of R).symm x).coeff ij.snd
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      r : R
      x : HahnModule Γ' R V
      a : Γ'
      b : Γ
      hr : Not (Eq r 0)
      hx : Not (Eq (((HahnModule.of R).symm x).coeff a) 0)
      ⊢ Eq ((Finset.VAddAntidiagonal ⋯ ⋯ (HVAdd.hVAdd b a)).sum fun x_1 => HSMul.hSM …
    -/
  · apply sum_congr _ fun _ _ => rfl
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      r : R
      x : HahnModule Γ' R V
      a : Γ'
      b : Γ
      hr : Not (Eq r 0)
      hx : Not (Eq (((HahnModule.of R).symm x).coeff a) 0)
      ⊢ Eq (Finset.VAddAntidiagonal ⋯ ⋯ (HVAdd.hVAdd b a)) (Singleton.singleton { fs …
    -/
    ext ⟨a1, a2⟩
    simp only [Set.mem_singleton_iff, Prod.mk.inj_iff, mem_vaddAntidiagonal, mem_singleton,
      Set.mem_setOf_eq]
    /-
      case h.mk
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      r : R
      x : HahnModule Γ' R V
      a : Γ'
      b : Γ
      hr : Not (Eq r 0)
      hx : Not (Eq (((HahnModule.of R).symm x).coeff a) 0)
      a1 : Γ
      a2 : Γ'
      ⊢ Iff (And (Eq a1 b) (And (Membership.mem ((HahnModule.of R).symm x).support a …
    -/
    constructor
      /-
        case h.mk.mp
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_5
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : VAdd Γ Γ'
        inst✝³ : IsOrderedCancelVAdd Γ Γ'
        inst✝² : AddCommMonoid V
        inst✝¹ : MulZeroClass R
        inst✝ : SMulWithZero R V
        r : R
        x : HahnModule Γ' R V
        a : Γ'
        b : Γ
        hr : Not (Eq r 0)
        hx : Not (Eq (((HahnModule.of R).symm x).coeff a) 0)
        a1 : Γ
        a2 : Γ'
        ⊢ And (Eq a1 b) (And (Membership.mem ((HahnModule.of R).symm x).support a2) (E …
      -/
    · rintro ⟨rfl, _, h1⟩
      /-
        case h.mk.mp.intro.intro
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_5
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : VAdd Γ Γ'
        inst✝³ : IsOrderedCancelVAdd Γ Γ'
        inst✝² : AddCommMonoid V
        inst✝¹ : MulZeroClass R
        inst✝ : SMulWithZero R V
        r : R
        x : HahnModule Γ' R V
        a : Γ'
        hr : Not (Eq r 0)
        hx : Not (Eq (((HahnModule.of R).symm x).coeff a) 0)
        a1 : Γ
        a2 : Γ'
        left✝ : Membership.mem ((HahnModule.of R).symm x).support a2
        h1 : Eq (HVAdd.hVAdd a1 a2) (HVAdd.hVAdd a1 a)
        ⊢ And (Eq a1 a1) (Eq a2 a)
      -/
      exact ⟨rfl, IsCancelVAdd.left_cancel a1 a2 a h1⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mk.mpr
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_5
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : VAdd Γ Γ'
        inst✝³ : IsOrderedCancelVAdd Γ Γ'
        inst✝² : AddCommMonoid V
        inst✝¹ : MulZeroClass R
        inst✝ : SMulWithZero R V
        r : R
        x : HahnModule Γ' R V
        a : Γ'
        b : Γ
        hr : Not (Eq r 0)
        hx : Not (Eq (((HahnModule.of R).symm x).coeff a) 0)
        a1 : Γ
        a2 : Γ'
        ⊢ And (Eq a1 b) (Eq a2 a) → And (Eq a1 b) (And (Membership.mem ((HahnModule.of …
      -/
    · rintro ⟨rfl, rfl⟩
      /-
        case h.mk.mpr.intro
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        V : Type u_5
        inst✝⁶ : PartialOrder Γ
        inst✝⁵ : PartialOrder Γ'
        inst✝⁴ : VAdd Γ Γ'
        inst✝³ : IsOrderedCancelVAdd Γ Γ'
        inst✝² : AddCommMonoid V
        inst✝¹ : MulZeroClass R
        inst✝ : SMulWithZero R V
        r : R
        x : HahnModule Γ' R V
        hr : Not (Eq r 0)
        a1 : Γ
        a2 : Γ'
        hx : Not (Eq (((HahnModule.of R).symm x).coeff a2) 0)
        ⊢ And (Eq a1 a1) (And (Membership.mem ((HahnModule.of R).symm x).support a2) ( …
      -/
      exact ⟨rfl, by exact hx, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      r : R
      x : HahnModule Γ' R V
      a : Γ'
      b : Γ
      hr : Not (Eq r 0)
      hx : Not (Eq (((HahnModule.of R).symm x).coeff a) 0)
      ⊢ Eq ((Singleton.singleton { fst := b, snd := a }).sum fun ij => HSMul.hSMul ( …
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem single_zero_smul_coeff {Γ} [OrderedAddCommMonoid Γ] [AddAction Γ Γ']
    [IsOrderedCancelVAdd Γ Γ'] [MulZeroClass R] [SMulWithZero R V] {r : R}
    {x : HahnModule Γ' R V} {a : Γ'} :
    ((of R).symm ((HahnSeries.single 0 r : HahnSeries Γ R) • x)).coeff a =
    r • ((of R).symm x).coeff a := by
  /-
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ'
    inst✝⁵ : AddCommMonoid V
    Γ : Type u_6
    inst✝⁴ : OrderedAddCommMonoid Γ
    inst✝³ : AddAction Γ Γ'
    inst✝² : IsOrderedCancelVAdd Γ Γ'
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    r : R
    x : HahnModule Γ' R V
    a : Γ'
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul ((HahnSeries.single 0) r) x)).coeff …
  -/
  nth_rw 1 [← zero_vadd Γ a]
  /-
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ'
    inst✝⁵ : AddCommMonoid V
    Γ : Type u_6
    inst✝⁴ : OrderedAddCommMonoid Γ
    inst✝³ : AddAction Γ Γ'
    inst✝² : IsOrderedCancelVAdd Γ Γ'
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    r : R
    x : HahnModule Γ' R V
    a : Γ'
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul ((HahnSeries.single 0) r) x)).coeff …
  -/
  exact single_smul_coeff_add
  /-
    🎉 no goals
  -/


@[simp]
theorem single_zero_smul_eq_smul (Γ) [OrderedAddCommMonoid Γ] [AddAction Γ Γ']
    [IsOrderedCancelVAdd Γ Γ'] [MulZeroClass R] [SMulWithZero R V] {r : R}
    {x : HahnModule Γ' R V} :
    (HahnSeries.single (0 : Γ) r) • x = r • x := by
  /-
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ'
    inst✝⁵ : AddCommMonoid V
    Γ : Type u_6
    inst✝⁴ : OrderedAddCommMonoid Γ
    inst✝³ : AddAction Γ Γ'
    inst✝² : IsOrderedCancelVAdd Γ Γ'
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    r : R
    x : HahnModule Γ' R V
    ⊢ Eq (HSMul.hSMul ((HahnSeries.single 0) r) x) (HSMul.hSMul r x)
  -/
  ext
  /-
    case h.h
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ'
    inst✝⁵ : AddCommMonoid V
    Γ : Type u_6
    inst✝⁴ : OrderedAddCommMonoid Γ
    inst✝³ : AddAction Γ Γ'
    inst✝² : IsOrderedCancelVAdd Γ Γ'
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    r : R
    x : HahnModule Γ' R V
    x✝ : Γ'
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul ((HahnSeries.single 0) r) x)).coeff …
  -/
  exact single_zero_smul_coeff
  /-
    🎉 no goals
  -/


@[simp]
theorem zero_smul' [Zero R] [SMulWithZero R V] {x : HahnModule Γ' R V} :
    (0 : HahnSeries Γ R) • x = 0 := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : Zero R
    inst✝ : SMulWithZero R V
    x : HahnModule Γ' R V
    ⊢ Eq (HSMul.hSMul 0 x) 0
  -/
  ext
  /-
    case h.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : Zero R
    inst✝ : SMulWithZero R V
    x : HahnModule Γ' R V
    x✝ : Γ'
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul 0 x)).coeff x✝) (((HahnModule.of R) …
  -/
  simp [smul_coeff]
  /-
    🎉 no goals
  -/


@[simp]
theorem one_smul' {Γ} [OrderedAddCommMonoid Γ] [AddAction Γ Γ'] [IsOrderedCancelVAdd Γ Γ']
    [MonoidWithZero R] [MulActionWithZero R V] {x : HahnModule Γ' R V} :
    (1 : HahnSeries Γ R) • x = x := by
  /-
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ'
    inst✝⁵ : AddCommMonoid V
    Γ : Type u_6
    inst✝⁴ : OrderedAddCommMonoid Γ
    inst✝³ : AddAction Γ Γ'
    inst✝² : IsOrderedCancelVAdd Γ Γ'
    inst✝¹ : MonoidWithZero R
    inst✝ : MulActionWithZero R V
    x : HahnModule Γ' R V
    ⊢ Eq (HSMul.hSMul 1 x) x
  -/
  ext g
  /-
    case h.h
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ'
    inst✝⁵ : AddCommMonoid V
    Γ : Type u_6
    inst✝⁴ : OrderedAddCommMonoid Γ
    inst✝³ : AddAction Γ Γ'
    inst✝² : IsOrderedCancelVAdd Γ Γ'
    inst✝¹ : MonoidWithZero R
    inst✝ : MulActionWithZero R V
    x : HahnModule Γ' R V
    g : Γ'
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul 1 x)).coeff g) (((HahnModule.of R). …
  -/
  exact single_zero_smul_coeff.trans (one_smul R (x.coeff g))
  /-
    🎉 no goals
  -/


theorem support_smul_subset_vadd_support' [MulZeroClass R] [SMulWithZero R V] {x : HahnSeries Γ R}
    {y : HahnModule Γ' R V} :
    ((of R).symm (x • y)).support ⊆ x.support +ᵥ ((of R).symm y).support := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    x : HahnSeries Γ R
    y : HahnModule Γ' R V
    ⊢ HasSubset.Subset ((HahnModule.of R).symm (HSMul.hSMul x y)).support (HVAdd.h …
  -/
  apply Set.Subset.trans (fun x hx => _) support_vaddAntidiagonal_subset_vadd
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      x : HahnSeries Γ R
      y : HahnModule Γ' R V
      ⊢ x.support.IsPWO
    -/
  · exact x.isPWO_support
    /-
      🎉 no goals
    -/
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : PartialOrder Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : VAdd Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : MulZeroClass R
      inst✝ : SMulWithZero R V
      x : HahnSeries Γ R
      y : HahnModule Γ' R V
      ⊢ ((HahnModule.of R).symm y).support.IsPWO
    -/
  · exact y.isPWO_support
    /-
      🎉 no goals
    -/
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    x : HahnSeries Γ R
    y : HahnModule Γ' R V
    ⊢ ∀ (x_1 : Γ'), Membership.mem ((HahnModule.of R).symm (HSMul.hSMul x y)).supp …
  -/
  intro x hx
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    x✝ : HahnSeries Γ R
    y : HahnModule Γ' R V
    x : Γ'
    hx : Membership.mem ((HahnModule.of R).symm (HSMul.hSMul x✝ y)).support x
    ⊢ Membership.mem (setOf fun a => (Finset.VAddAntidiagonal ⋯ ⋯ a).Nonempty) x
  -/
  contrapose! hx
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    x✝ : HahnSeries Γ R
    y : HahnModule Γ' R V
    x : Γ'
    hx : Not (Membership.mem (setOf fun a => (Finset.VAddAntidiagonal ⋯ ⋯ a).Nonem …
    ⊢ Not (Membership.mem ((HahnModule.of R).symm (HSMul.hSMul x✝ y)).support x)
  -/
  simp only [Set.mem_setOf_eq, not_nonempty_iff_eq_empty] at hx
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    x✝ : HahnSeries Γ R
    y : HahnModule Γ' R V
    x : Γ'
    hx : Eq (Finset.VAddAntidiagonal ⋯ ⋯ x) EmptyCollection.emptyCollection
    ⊢ Not (Membership.mem ((HahnModule.of R).symm (HSMul.hSMul x✝ y)).support x)
  -/
  simp [hx, smul_coeff]
  /-
    🎉 no goals
  -/


theorem support_smul_subset_vadd_support [MulZeroClass R] [SMulWithZero R V] {x : HahnSeries Γ R}
    {y : HahnModule Γ' R V} :
    ((of R).symm (x • y)).support ⊆ x.support +ᵥ ((of R).symm y).support := by
  have h : x.support +ᵥ ((of R).symm y).support =
      x.support +ᵥ ((of R).symm y).support := by
    exact rfl
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    x : HahnSeries Γ R
    y : HahnModule Γ' R V
    h : Eq (HVAdd.hVAdd x.support ((HahnModule.of R).symm y).support) (HVAdd.hVAdd …
    ⊢ HasSubset.Subset ((HahnModule.of R).symm (HSMul.hSMul x y)).support (HVAdd.h …
  -/
  rw [h]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : PartialOrder Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : VAdd Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : MulZeroClass R
    inst✝ : SMulWithZero R V
    x : HahnSeries Γ R
    y : HahnModule Γ' R V
    h : Eq (HVAdd.hVAdd x.support ((HahnModule.of R).symm y).support) (HVAdd.hVAdd …
    ⊢ HasSubset.Subset ((HahnModule.of R).symm (HSMul.hSMul x y)).support (HVAdd.h …
  -/
  exact support_smul_subset_vadd_support'
  /-
    🎉 no goals
  -/


theorem smul_coeff_order_add_order {Γ} [LinearOrderedCancelAddCommMonoid Γ] [Zero R]
    [SMulWithZero R V] (x : HahnSeries Γ R) (y : HahnModule Γ R V) :
    ((of R).symm (x • y)).coeff (x.order + ((of R).symm y).order) =
    x.leadingCoeff • ((of R).symm y).leadingCoeff := by
  /-
    R : Type u_3
    V : Type u_5
    inst✝³ : AddCommMonoid V
    Γ : Type u_6
    inst✝² : LinearOrderedCancelAddCommMonoid Γ
    inst✝¹ : Zero R
    inst✝ : SMulWithZero R V
    x : HahnSeries Γ R
    y : HahnModule Γ R V
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul x y)).coeff (HAdd.hAdd x.order ((Ha …
  -/
  by_cases hx : x = (0 : HahnSeries Γ R); · simp [HahnSeries.zero_coeff, hx]
                                            /-
                                              🎉 no goals
                                            -/
  /-
    case neg
    R : Type u_3
    V : Type u_5
    inst✝³ : AddCommMonoid V
    Γ : Type u_6
    inst✝² : LinearOrderedCancelAddCommMonoid Γ
    inst✝¹ : Zero R
    inst✝ : SMulWithZero R V
    x : HahnSeries Γ R
    y : HahnModule Γ R V
    hx : Not (Eq x 0)
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul x y)).coeff (HAdd.hAdd x.order ((Ha …
  -/
  by_cases hy : (of R).symm y = 0; · simp [hy, smul_coeff]
                                     /-
                                       🎉 no goals
                                     -/
  rw [HahnSeries.order_of_ne hx, HahnSeries.order_of_ne hy, smul_coeff,
    HahnSeries.leadingCoeff_of_ne hx, HahnSeries.leadingCoeff_of_ne hy]
  /-
    case neg
    R : Type u_3
    V : Type u_5
    inst✝³ : AddCommMonoid V
    Γ : Type u_6
    inst✝² : LinearOrderedCancelAddCommMonoid Γ
    inst✝¹ : Zero R
    inst✝ : SMulWithZero R V
    x : HahnSeries Γ R
    y : HahnModule Γ R V
    hx : Not (Eq x 0)
    hy : Not (Eq ((HahnModule.of R).symm y) 0)
    ⊢ Eq ((Finset.VAddAntidiagonal ⋯ ⋯ (HAdd.hAdd (⋯.min ⋯) (⋯.min ⋯))).sum fun ij …
  -/
  erw [Finset.vaddAntidiagonal_min_vadd_min, Finset.sum_singleton]
  /-
    🎉 no goals
  -/


instance [NonUnitalNonAssocSemiring R] : Mul (HahnSeries Γ R) where
  mul x y := (HahnModule.of R).symm (x • HahnModule.of R y)


theorem of_symm_smul_of_eq_mul [NonUnitalNonAssocSemiring R] {x y : HahnSeries Γ R} :
    (HahnModule.of R).symm (x • HahnModule.of R y) = x * y := rfl


theorem mul_coeff [NonUnitalNonAssocSemiring R] {x y : HahnSeries Γ R} {a : Γ} :
    (x * y).coeff a =
      ∑ ij ∈ addAntidiagonal x.isPWO_support y.isPWO_support a, x.coeff ij.fst * y.coeff ij.snd :=
  rfl


protected lemma map_mul [NonUnitalNonAssocSemiring R] [NonUnitalNonAssocSemiring S] (f : R →ₙ+* S)
    {x y : HahnSeries Γ R} : (x * y).map f = (x.map f : HahnSeries Γ S) * (y.map f) := by
  /-
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCancelAddCommMonoid Γ
    inst✝¹ : NonUnitalNonAssocSemiring R
    inst✝ : NonUnitalNonAssocSemiring S
    f : NonUnitalRingHom R S
    x y : HahnSeries Γ R
    ⊢ Eq ((HMul.hMul x y).map f) (HMul.hMul (x.map f) (y.map f))
  -/
  ext
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCancelAddCommMonoid Γ
    inst✝¹ : NonUnitalNonAssocSemiring R
    inst✝ : NonUnitalNonAssocSemiring S
    f : NonUnitalRingHom R S
    x y : HahnSeries Γ R
    x✝ : Γ
    ⊢ Eq (((HMul.hMul x y).map f).coeff x✝) ((HMul.hMul (x.map f) (y.map f)).coeff …
  -/
  simp only [map_coeff, mul_coeff, ZeroHom.coe_coe, map_sum, map_mul]
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCancelAddCommMonoid Γ
    inst✝¹ : NonUnitalNonAssocSemiring R
    inst✝ : NonUnitalNonAssocSemiring S
    f : NonUnitalRingHom R S
    x y : HahnSeries Γ R
    x✝ : Γ
    ⊢ Eq ((Finset.addAntidiagonal ⋯ ⋯ x✝).sum fun x_1 => HMul.hMul (f (x.coeff x_1 …
  -/
  refine Eq.symm (sum_subset (fun gh hgh => ?_) (fun gh hgh hz => ?_))
    /-
      case coeff.h.refine_1
      Γ : Type u_1
      R : Type u_3
      S : Type u_4
      inst✝² : OrderedCancelAddCommMonoid Γ
      inst✝¹ : NonUnitalNonAssocSemiring R
      inst✝ : NonUnitalNonAssocSemiring S
      f : NonUnitalRingHom R S
      x y : HahnSeries Γ R
      x✝ : Γ
      gh : Prod Γ Γ
      hgh : Membership.mem (Finset.addAntidiagonal ⋯ ⋯ x✝) gh
      ⊢ Membership.mem (Finset.addAntidiagonal ⋯ ⋯ x✝) gh
    -/
  · simp_all only [mem_addAntidiagonal, mem_support, map_coeff, ZeroHom.coe_coe, ne_eq, and_true]
    /-
      case coeff.h.refine_1
      Γ : Type u_1
      R : Type u_3
      S : Type u_4
      inst✝² : OrderedCancelAddCommMonoid Γ
      inst✝¹ : NonUnitalNonAssocSemiring R
      inst✝ : NonUnitalNonAssocSemiring S
      f : NonUnitalRingHom R S
      x y : HahnSeries Γ R
      x✝ : Γ
      gh : Prod Γ Γ
      hgh : And (Not (Eq (f (x.coeff gh.1)) 0)) (And (Not (Eq (f (y.coeff gh.2)) 0)) …
      ⊢ And (Not (Eq (x.coeff gh.1) 0)) (Not (Eq (y.coeff gh.2) 0))
    -/
    exact ⟨fun h => hgh.1 (map_zero f ▸ congrArg f h), fun h => hgh.2.1 (map_zero f ▸ congrArg f h)⟩
    /-
      🎉 no goals
    -/
  · simp_all only [mem_addAntidiagonal, mem_support, ne_eq, map_coeff, ZeroHom.coe_coe, and_true,
      not_and, not_not]
    /-
      case coeff.h.refine_2
      Γ : Type u_1
      R : Type u_3
      S : Type u_4
      inst✝² : OrderedCancelAddCommMonoid Γ
      inst✝¹ : NonUnitalNonAssocSemiring R
      inst✝ : NonUnitalNonAssocSemiring S
      f : NonUnitalRingHom R S
      x y : HahnSeries Γ R
      x✝ : Γ
      gh : Prod Γ Γ
      hgh : And (Not (Eq (x.coeff gh.1) 0)) (And (Not (Eq (y.coeff gh.2) 0)) (Eq (HA …
      hz : Not (Eq (f (x.coeff gh.1)) 0) → Eq (f (y.coeff gh.2)) 0
      ⊢ Eq (HMul.hMul (f (x.coeff gh.1)) (f (y.coeff gh.2))) 0
    -/
    by_cases h : f (x.coeff gh.1) = 0
      /-
        case pos
        Γ : Type u_1
        R : Type u_3
        S : Type u_4
        inst✝² : OrderedCancelAddCommMonoid Γ
        inst✝¹ : NonUnitalNonAssocSemiring R
        inst✝ : NonUnitalNonAssocSemiring S
        f : NonUnitalRingHom R S
        x y : HahnSeries Γ R
        x✝ : Γ
        gh : Prod Γ Γ
        hgh : And (Not (Eq (x.coeff gh.1) 0)) (And (Not (Eq (y.coeff gh.2) 0)) (Eq (HA …
        hz : Not (Eq (f (x.coeff gh.1)) 0) → Eq (f (y.coeff gh.2)) 0
        h : Eq (f (x.coeff gh.1)) 0
        ⊢ Eq (HMul.hMul (f (x.coeff gh.1)) (f (y.coeff gh.2))) 0
      -/
    · exact mul_eq_zero_of_left h (f (y.coeff gh.2))
      /-
        🎉 no goals
      -/
      /-
        case neg
        Γ : Type u_1
        R : Type u_3
        S : Type u_4
        inst✝² : OrderedCancelAddCommMonoid Γ
        inst✝¹ : NonUnitalNonAssocSemiring R
        inst✝ : NonUnitalNonAssocSemiring S
        f : NonUnitalRingHom R S
        x y : HahnSeries Γ R
        x✝ : Γ
        gh : Prod Γ Γ
        hgh : And (Not (Eq (x.coeff gh.1) 0)) (And (Not (Eq (y.coeff gh.2) 0)) (Eq (HA …
        hz : Not (Eq (f (x.coeff gh.1)) 0) → Eq (f (y.coeff gh.2)) 0
        h : Not (Eq (f (x.coeff gh.1)) 0)
        ⊢ Eq (HMul.hMul (f (x.coeff gh.1)) (f (y.coeff gh.2))) 0
      -/
    · exact mul_eq_zero_of_right (f (x.coeff gh.1)) (hz h)
      /-
        🎉 no goals
      -/


theorem mul_coeff_left' [NonUnitalNonAssocSemiring R] {x y : HahnSeries Γ R} {a : Γ} {s : Set Γ}
    (hs : s.IsPWO) (hxs : x.support ⊆ s) :
    (x * y).coeff a =
      ∑ ij ∈ addAntidiagonal hs y.isPWO_support a, x.coeff ij.fst * y.coeff ij.snd :=
  HahnModule.smul_coeff_left hs hxs


theorem mul_coeff_right' [NonUnitalNonAssocSemiring R] {x y : HahnSeries Γ R} {a : Γ} {s : Set Γ}
    (hs : s.IsPWO) (hys : y.support ⊆ s) :
    (x * y).coeff a =
      ∑ ij ∈ addAntidiagonal x.isPWO_support hs a, x.coeff ij.fst * y.coeff ij.snd :=
  HahnModule.smul_coeff_right hs hys


instance [NonUnitalNonAssocSemiring R] : Distrib (HahnSeries Γ R) :=
  { inferInstanceAs (Mul (HahnSeries Γ R)),
    inferInstanceAs (Add (HahnSeries Γ R)) with
    left_distrib := fun x y z => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        x y z : HahnSeries Γ R
        ⊢ Eq (HMul.hMul x (HAdd.hAdd y z)) (HAdd.hAdd (HMul.hMul x y) (HMul.hMul x z))
      -/
      simp only [← of_symm_smul_of_eq_mul]
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        x y z : HahnSeries Γ R
        ⊢ Eq ((HahnModule.of R).symm (HSMul.hSMul x ((HahnModule.of R) (HAdd.hAdd y z) …
      -/
      exact HahnModule.smul_add x y z
      /-
        🎉 no goals
      -/
    right_distrib := fun x y z => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        x y z : HahnSeries Γ R
        ⊢ Eq (HMul.hMul (HAdd.hAdd x y) z) (HAdd.hAdd (HMul.hMul x z) (HMul.hMul y z))
      -/
      simp only [← of_symm_smul_of_eq_mul]
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        x y z : HahnSeries Γ R
        ⊢ Eq ((HahnModule.of R).symm (HSMul.hSMul (HAdd.hAdd x y) ((HahnModule.of R) z …
      -/
      refine HahnModule.add_smul ?_
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        x y z : HahnSeries Γ R
        ⊢ ∀ (r s u : R), Eq (HSMul.hSMul (HAdd.hAdd r s) u) (HAdd.hAdd (HSMul.hSMul r  …
      -/
      simp only [smul_eq_mul]
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        x y z : HahnSeries Γ R
        ⊢ ∀ (r s u : R), Eq (HMul.hMul (HAdd.hAdd r s) u) (HAdd.hAdd (HMul.hMul r u) ( …
      -/
      exact add_mul }
      /-
        🎉 no goals
      -/


theorem single_mul_coeff_add [NonUnitalNonAssocSemiring R] {r : R} {x : HahnSeries Γ R} {a : Γ}
    {b : Γ} : (single b r * x).coeff (a + b) = r * x.coeff a := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    r : R
    x : HahnSeries Γ R
    a b : Γ
    ⊢ Eq ((HMul.hMul ((HahnSeries.single b) r) x).coeff (HAdd.hAdd a b)) (HMul.hMu …
  -/
  rw [← of_symm_smul_of_eq_mul, add_comm, ← vadd_eq_add]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    r : R
    x : HahnSeries Γ R
    a b : Γ
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul ((HahnSeries.single b) r) ((HahnMod …
  -/
  exact HahnModule.single_smul_coeff_add
  /-
    🎉 no goals
  -/


theorem mul_single_coeff_add [NonUnitalNonAssocSemiring R] {r : R} {x : HahnSeries Γ R} {a : Γ}
    {b : Γ} : (x * single b r).coeff (a + b) = x.coeff a * r := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    r : R
    x : HahnSeries Γ R
    a b : Γ
    ⊢ Eq ((HMul.hMul x ((HahnSeries.single b) r)).coeff (HAdd.hAdd a b)) (HMul.hMu …
  -/
  by_cases hr : r = 0
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      r : R
      x : HahnSeries Γ R
      a b : Γ
      hr : Eq r 0
      ⊢ Eq ((HMul.hMul x ((HahnSeries.single b) r)).coeff (HAdd.hAdd a b)) (HMul.hMu …
    -/
  · simp [hr, mul_coeff]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    r : R
    x : HahnSeries Γ R
    a b : Γ
    hr : Not (Eq r 0)
    ⊢ Eq ((HMul.hMul x ((HahnSeries.single b) r)).coeff (HAdd.hAdd a b)) (HMul.hMu …
  -/
  simp only [hr, smul_coeff, mul_coeff, support_single_of_ne, Ne, not_false_iff, smul_eq_mul]
  /-
    case neg
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    r : R
    x : HahnSeries Γ R
    a b : Γ
    hr : Not (Eq r 0)
    ⊢ Eq ((Finset.addAntidiagonal ⋯ ⋯ (HAdd.hAdd a b)).sum fun x_1 => HMul.hMul (x …
  -/
  by_cases hx : x.coeff a = 0
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      r : R
      x : HahnSeries Γ R
      a b : Γ
      hr : Not (Eq r 0)
      hx : Eq (x.coeff a) 0
      ⊢ Eq ((Finset.addAntidiagonal ⋯ ⋯ (HAdd.hAdd a b)).sum fun x_1 => HMul.hMul (x …
    -/
  · simp only [hx, zero_mul]
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      r : R
      x : HahnSeries Γ R
      a b : Γ
      hr : Not (Eq r 0)
      hx : Eq (x.coeff a) 0
      ⊢ Eq ((Finset.addAntidiagonal ⋯ ⋯ (HAdd.hAdd a b)).sum fun x_1 => HMul.hMul (x …
    -/
    rw [sum_congr _ fun _ _ => rfl, sum_empty]
    /-
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      r : R
      x : HahnSeries Γ R
      a b : Γ
      hr : Not (Eq r 0)
      hx : Eq (x.coeff a) 0
      ⊢ Eq (Finset.addAntidiagonal ⋯ ⋯ (HAdd.hAdd a b)) EmptyCollection.emptyCollect …
    -/
    ext ⟨a1, a2⟩
    simp only [not_mem_empty, not_and, Set.mem_singleton_iff, Classical.not_not,
      mem_addAntidiagonal, Set.mem_setOf_eq, iff_false]
    /-
      case h.mk
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      r : R
      x : HahnSeries Γ R
      a b : Γ
      hr : Not (Eq r 0)
      hx : Eq (x.coeff a) 0
      a1 a2 : Γ
      ⊢ Membership.mem x.support a1 → Eq a2 b → Not (Eq (HAdd.hAdd a1 a2) (HAdd.hAdd …
    -/
    rintro h2 rfl h1
    /-
      case h.mk
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      r : R
      x : HahnSeries Γ R
      a : Γ
      hr : Not (Eq r 0)
      hx : Eq (x.coeff a) 0
      a1 a2 : Γ
      h2 : Membership.mem x.support a1
      h1 : Eq (HAdd.hAdd a1 a2) (HAdd.hAdd a a2)
      ⊢ False
    -/
    rw [← add_right_cancel h1] at hx
    /-
      case h.mk
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      r : R
      x : HahnSeries Γ R
      a : Γ
      hr : Not (Eq r 0)
      a1 : Γ
      hx : Eq (x.coeff a1) 0
      a2 : Γ
      h2 : Membership.mem x.support a1
      h1 : Eq (HAdd.hAdd a1 a2) (HAdd.hAdd a a2)
      ⊢ False
    -/
    exact h2 hx
    /-
      🎉 no goals
    -/
  /-
    case neg
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    r : R
    x : HahnSeries Γ R
    a b : Γ
    hr : Not (Eq r 0)
    hx : Not (Eq (x.coeff a) 0)
    ⊢ Eq ((Finset.addAntidiagonal ⋯ ⋯ (HAdd.hAdd a b)).sum fun x_1 => HMul.hMul (x …
  -/
  trans ∑ ij ∈ {(a, b)}, x.coeff ij.fst * (single b r).coeff ij.snd
    /-
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      r : R
      x : HahnSeries Γ R
      a b : Γ
      hr : Not (Eq r 0)
      hx : Not (Eq (x.coeff a) 0)
      ⊢ Eq ((Finset.addAntidiagonal ⋯ ⋯ (HAdd.hAdd a b)).sum fun x_1 => HMul.hMul (x …
    -/
  · apply sum_congr _ fun _ _ => rfl
    /-
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      r : R
      x : HahnSeries Γ R
      a b : Γ
      hr : Not (Eq r 0)
      hx : Not (Eq (x.coeff a) 0)
      ⊢ Eq (Finset.addAntidiagonal ⋯ ⋯ (HAdd.hAdd a b)) (Singleton.singleton { fst : …
    -/
    ext ⟨a1, a2⟩
    simp only [Set.mem_singleton_iff, Prod.mk.inj_iff, mem_addAntidiagonal, mem_singleton,
      Set.mem_setOf_eq]
    /-
      case h.mk
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      r : R
      x : HahnSeries Γ R
      a b : Γ
      hr : Not (Eq r 0)
      hx : Not (Eq (x.coeff a) 0)
      a1 a2 : Γ
      ⊢ Iff (And (Membership.mem x.support a1) (And (Eq a2 b) (Eq (HAdd.hAdd a1 a2)  …
    -/
    constructor
      /-
        case h.mk.mp
        Γ : Type u_1
        R : Type u_3
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        r : R
        x : HahnSeries Γ R
        a b : Γ
        hr : Not (Eq r 0)
        hx : Not (Eq (x.coeff a) 0)
        a1 a2 : Γ
        ⊢ And (Membership.mem x.support a1) (And (Eq a2 b) (Eq (HAdd.hAdd a1 a2) (HAdd …
      -/
    · rintro ⟨_, rfl, h1⟩
      /-
        case h.mk.mp.intro.intro
        Γ : Type u_1
        R : Type u_3
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        r : R
        x : HahnSeries Γ R
        a : Γ
        hr : Not (Eq r 0)
        hx : Not (Eq (x.coeff a) 0)
        a1 a2 : Γ
        left✝ : Membership.mem x.support a1
        h1 : Eq (HAdd.hAdd a1 a2) (HAdd.hAdd a a2)
        ⊢ And (Eq a1 a) (Eq a2 a2)
      -/
      exact ⟨add_right_cancel h1, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mk.mpr
        Γ : Type u_1
        R : Type u_3
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        r : R
        x : HahnSeries Γ R
        a b : Γ
        hr : Not (Eq r 0)
        hx : Not (Eq (x.coeff a) 0)
        a1 a2 : Γ
        ⊢ And (Eq a1 a) (Eq a2 b) → And (Membership.mem x.support a1) (And (Eq a2 b) ( …
      -/
    · rintro ⟨rfl, rfl⟩
      /-
        case h.mk.mpr.intro
        Γ : Type u_1
        R : Type u_3
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        r : R
        x : HahnSeries Γ R
        hr : Not (Eq r 0)
        a1 a2 : Γ
        hx : Not (Eq (x.coeff a1) 0)
        ⊢ And (Membership.mem x.support a1) (And (Eq a2 a2) (Eq (HAdd.hAdd a1 a2) (HAd …
      -/
      simp [hx]
      /-
        🎉 no goals
      -/
    /-
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      r : R
      x : HahnSeries Γ R
      a b : Γ
      hr : Not (Eq r 0)
      hx : Not (Eq (x.coeff a) 0)
      ⊢ Eq ((Singleton.singleton { fst := a, snd := b }).sum fun ij => HMul.hMul (x. …
    -/
  · simp
    /-
      🎉 no goals
    -/


@[simp]
theorem mul_single_zero_coeff [NonUnitalNonAssocSemiring R] {r : R} {x : HahnSeries Γ R} {a : Γ} :
                                                   /-
                                                     Γ : Type u_1
                                                     R : Type u_3
                                                     inst✝¹ : OrderedCancelAddCommMonoid Γ
                                                     inst✝ : NonUnitalNonAssocSemiring R
                                                     r : R
                                                     x : HahnSeries Γ R
                                                     a : Γ
                                                     ⊢ Eq ((HMul.hMul x ((HahnSeries.single 0) r)).coeff a) (HMul.hMul (x.coeff a) r)
                                                   -/
    (x * single 0 r).coeff a = x.coeff a * r := by rw [← add_zero a, mul_single_coeff_add, add_zero]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem single_zero_mul_coeff [NonUnitalNonAssocSemiring R] {r : R} {x : HahnSeries Γ R} {a : Γ} :
    ((single 0 r : HahnSeries Γ R) * x).coeff a = r * x.coeff a := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    r : R
    x : HahnSeries Γ R
    a : Γ
    ⊢ Eq ((HMul.hMul ((HahnSeries.single 0) r) x).coeff a) (HMul.hMul r (x.coeff a))
  -/
  rw [← add_zero a, single_mul_coeff_add, add_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem single_zero_mul_eq_smul [Semiring R] {r : R} {x : HahnSeries Γ R} :
    single 0 r * x = r • x := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    r : R
    x : HahnSeries Γ R
    ⊢ Eq (HMul.hMul ((HahnSeries.single 0) r) x) (HSMul.hSMul r x)
  -/
  ext
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    r : R
    x : HahnSeries Γ R
    x✝ : Γ
    ⊢ Eq ((HMul.hMul ((HahnSeries.single 0) r) x).coeff x✝) ((HSMul.hSMul r x).coe …
  -/
  exact single_zero_mul_coeff
  /-
    🎉 no goals
  -/


theorem support_mul_subset_add_support [NonUnitalNonAssocSemiring R] {x y : HahnSeries Γ R} :
    support (x * y) ⊆ support x + support y := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    x y : HahnSeries Γ R
    ⊢ HasSubset.Subset (HMul.hMul x y).support (HAdd.hAdd x.support y.support)
  -/
  rw [← of_symm_smul_of_eq_mul, ← vadd_eq_add]
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    x y : HahnSeries Γ R
    ⊢ HasSubset.Subset ((HahnModule.of R).symm (HSMul.hSMul x ((HahnModule.of R) y …
  -/
  exact HahnModule.support_smul_subset_vadd_support
  /-
    🎉 no goals
  -/


theorem mul_coeff_order_add_order {Γ} [LinearOrderedCancelAddCommMonoid Γ]
    [NonUnitalNonAssocSemiring R] (x y : HahnSeries Γ R) :
    (x * y).coeff (x.order + y.order) = x.leadingCoeff * y.leadingCoeff := by
  /-
    R : Type u_3
    Γ : Type u_6
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    x y : HahnSeries Γ R
    ⊢ Eq ((HMul.hMul x y).coeff (HAdd.hAdd x.order y.order)) (HMul.hMul x.leadingC …
  -/
  simp only [← of_symm_smul_of_eq_mul]
  /-
    R : Type u_3
    Γ : Type u_6
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    x y : HahnSeries Γ R
    ⊢ Eq (((HahnModule.of R).symm (HSMul.hSMul x ((HahnModule.of R) y))).coeff (HA …
  -/
  exact HahnModule.smul_coeff_order_add_order x y
  /-
    🎉 no goals
  -/


private theorem mul_assoc' [NonUnitalSemiring R] (x y z : HahnSeries Γ R) :
    x * y * z = x * (y * z) := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalSemiring R
    x y z : HahnSeries Γ R
    ⊢ Eq (HMul.hMul (HMul.hMul x y) z) (HMul.hMul x (HMul.hMul y z))
  -/
  ext b
  rw [mul_coeff_left' (x.isPWO_support.add y.isPWO_support) support_mul_subset_add_support,
    mul_coeff_right' (y.isPWO_support.add z.isPWO_support) support_mul_subset_add_support]
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalSemiring R
    x y z : HahnSeries Γ R
    b : Γ
    ⊢ Eq ((Finset.addAntidiagonal ⋯ ⋯ b).sum fun ij => HMul.hMul ((HMul.hMul x y). …
  -/
  simp only [mul_coeff, add_coeff, sum_mul, mul_sum, sum_sigma']
  apply Finset.sum_nbij' (fun ⟨⟨_i, j⟩, ⟨k, l⟩⟩ ↦ ⟨(k, l + j), (l, j)⟩)
    (fun ⟨⟨i, _j⟩, ⟨k, l⟩⟩ ↦ ⟨(i + k, l), (i, k)⟩) <;>
    /-
      case coeff.h.hi
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalSemiring R
      x y z : HahnSeries Γ R
      b : Γ
      ⊢ ∀ (a : Sigma fun i => Prod Γ Γ), Membership.mem ((Finset.addAntidiagonal ⋯ ⋯ …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    aesop (add safe Set.add_mem_add) (add simp [add_assoc, mul_assoc])
    /-
      🎉 no goals
    -/


instance [NonUnitalNonAssocSemiring R] : NonUnitalNonAssocSemiring (HahnSeries Γ R) :=
  { inferInstanceAs (AddCommMonoid (HahnSeries Γ R)),
    inferInstanceAs (Distrib (HahnSeries Γ R)) with
    zero_mul := fun _ => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        x✝ : HahnSeries Γ R
        ⊢ Eq (HMul.hMul 0 x✝) 0
      -/
      ext
      /-
        case coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        x✝¹ : HahnSeries Γ R
        x✝ : Γ
        ⊢ Eq ((HMul.hMul 0 x✝¹).coeff x✝) (HahnSeries.coeff 0 x✝)
      -/
      simp [mul_coeff]
      /-
        🎉 no goals
      -/
    mul_zero := fun _ => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        x✝ : HahnSeries Γ R
        ⊢ Eq (HMul.hMul x✝ 0) 0
      -/
      ext
      /-
        case coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonUnitalNonAssocSemiring R
        x✝¹ : HahnSeries Γ R
        x✝ : Γ
        ⊢ Eq ((HMul.hMul x✝¹ 0).coeff x✝) (HahnSeries.coeff 0 x✝)
      -/
      simp [mul_coeff] }
      /-
        🎉 no goals
      -/


instance [NonUnitalSemiring R] : NonUnitalSemiring (HahnSeries Γ R) :=
  { inferInstanceAs (NonUnitalNonAssocSemiring (HahnSeries Γ R)) with
    mul_assoc := mul_assoc' }


instance [NonAssocSemiring R] : NonAssocSemiring (HahnSeries Γ R) :=
  { AddMonoidWithOne.unary,
    inferInstanceAs (NonUnitalNonAssocSemiring (HahnSeries Γ R)) with
    one_mul := fun x => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonAssocSemiring R
        x : HahnSeries Γ R
        ⊢ Eq (HMul.hMul 1 x) x
      -/
      ext
      /-
        case coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonAssocSemiring R
        x : HahnSeries Γ R
        x✝ : Γ
        ⊢ Eq ((HMul.hMul 1 x).coeff x✝) (x.coeff x✝)
      -/
      exact single_zero_mul_coeff.trans (one_mul _)
      /-
        🎉 no goals
      -/
    mul_one := fun x => by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonAssocSemiring R
        x : HahnSeries Γ R
        ⊢ Eq (HMul.hMul x 1) x
      -/
      ext
      /-
        case coeff.h
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝¹ : OrderedCancelAddCommMonoid Γ
        inst✝ : NonAssocSemiring R
        x : HahnSeries Γ R
        x✝ : Γ
        ⊢ Eq ((HMul.hMul x 1).coeff x✝) (x.coeff x✝)
      -/
      exact mul_single_zero_coeff.trans (mul_one _) }
      /-
        🎉 no goals
      -/


instance [Semiring R] : Semiring (HahnSeries Γ R) :=
  { inferInstanceAs (NonAssocSemiring (HahnSeries Γ R)),
    inferInstanceAs (NonUnitalSemiring (HahnSeries Γ R)) with }


instance [NonUnitalCommSemiring R] : NonUnitalCommSemiring (HahnSeries Γ R) where
  __ : NonUnitalSemiring (HahnSeries Γ R) := inferInstance
  mul_comm x y := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalCommSemiring R
      x y : HahnSeries Γ R
      ⊢ Eq (HMul.hMul x y) (HMul.hMul y x)
    -/
    ext
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalCommSemiring R
      x y : HahnSeries Γ R
      x✝ : Γ
      ⊢ Eq ((HMul.hMul x y).coeff x✝) ((HMul.hMul y x).coeff x✝)
    -/
    simp_rw [mul_coeff, mul_comm]
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalCommSemiring R
      x y : HahnSeries Γ R
      x✝ : Γ
      ⊢ Eq ((Finset.addAntidiagonal ⋯ ⋯ x✝).sum fun ij => HMul.hMul (x.coeff ij.1) ( …
    -/
    exact Finset.sum_equiv (Equiv.prodComm _ _) (fun _ ↦ swap_mem_addAntidiagonal.symm) <| by simp
    /-
      🎉 no goals
    -/


instance [CommSemiring R] : CommSemiring (HahnSeries Γ R) :=
  { inferInstanceAs (NonUnitalCommSemiring (HahnSeries Γ R)),
    inferInstanceAs (Semiring (HahnSeries Γ R)) with }


instance [NonUnitalNonAssocRing R] : NonUnitalNonAssocRing (HahnSeries Γ R) :=
  { inferInstanceAs (NonUnitalNonAssocSemiring (HahnSeries Γ R)),
    inferInstanceAs (AddGroup (HahnSeries Γ R)) with }


instance [NonUnitalRing R] : NonUnitalRing (HahnSeries Γ R) :=
  { inferInstanceAs (NonUnitalNonAssocRing (HahnSeries Γ R)),
    inferInstanceAs (NonUnitalSemiring (HahnSeries Γ R)) with }


instance [NonAssocRing R] : NonAssocRing (HahnSeries Γ R) :=
  { inferInstanceAs (NonUnitalNonAssocRing (HahnSeries Γ R)),
    inferInstanceAs (NonAssocSemiring (HahnSeries Γ R)) with }


instance [Ring R] : Ring (HahnSeries Γ R) :=
  { inferInstanceAs (Semiring (HahnSeries Γ R)),
    inferInstanceAs (AddCommGroup (HahnSeries Γ R)) with }


instance [NonUnitalCommRing R] : NonUnitalCommRing (HahnSeries Γ R) :=
  { inferInstanceAs (NonUnitalCommSemiring (HahnSeries Γ R)),
    inferInstanceAs (NonUnitalRing (HahnSeries Γ R)) with }


instance [CommRing R] : CommRing (HahnSeries Γ R) :=
  { inferInstanceAs (CommSemiring (HahnSeries Γ R)),
    inferInstanceAs (Ring (HahnSeries Γ R)) with }


private theorem mul_smul' [Semiring R] [Module R V] (x y : HahnSeries Γ R)
    (z : HahnModule Γ' R V) : (x * y) • z = x • (y • z) := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : OrderedCancelAddCommMonoid Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : AddAction Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : Semiring R
    inst✝ : Module R V
    x y : HahnSeries Γ R
    z : HahnModule Γ' R V
    ⊢ Eq (HSMul.hSMul (HMul.hMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
  -/
  ext b
  rw [smul_coeff_left (x.isPWO_support.add y.isPWO_support)
    HahnSeries.support_mul_subset_add_support, smul_coeff_right
    (y.isPWO_support.vadd ((of R).symm z).isPWO_support) support_smul_subset_vadd_support]
  /-
    case h.h
    Γ : Type u_1
    Γ' : Type u_2
    R : Type u_3
    V : Type u_5
    inst✝⁶ : OrderedCancelAddCommMonoid Γ
    inst✝⁵ : PartialOrder Γ'
    inst✝⁴ : AddAction Γ Γ'
    inst✝³ : IsOrderedCancelVAdd Γ Γ'
    inst✝² : AddCommMonoid V
    inst✝¹ : Semiring R
    inst✝ : Module R V
    x y : HahnSeries Γ R
    z : HahnModule Γ' R V
    b : Γ'
    ⊢ Eq ((Finset.VAddAntidiagonal ⋯ ⋯ b).sum fun ij => HSMul.hSMul ((HMul.hMul x  …
  -/
  simp only [HahnSeries.mul_coeff, smul_coeff, HahnSeries.add_coeff, sum_smul, smul_sum, sum_sigma']
  apply Finset.sum_nbij' (fun ⟨⟨_i, j⟩, ⟨k, l⟩⟩ ↦ ⟨(k, l +ᵥ j), (l, j)⟩)
    (fun ⟨⟨i, _j⟩, ⟨k, l⟩⟩ ↦ ⟨(i + k, l), (i, k)⟩) <;>
    /-
      case h.h.hi
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      V : Type u_5
      inst✝⁶ : OrderedCancelAddCommMonoid Γ
      inst✝⁵ : PartialOrder Γ'
      inst✝⁴ : AddAction Γ Γ'
      inst✝³ : IsOrderedCancelVAdd Γ Γ'
      inst✝² : AddCommMonoid V
      inst✝¹ : Semiring R
      inst✝ : Module R V
      x y : HahnSeries Γ R
      z : HahnModule Γ' R V
      b : Γ'
      ⊢ ∀ (a : Sigma fun i => Prod Γ Γ), Membership.mem ((Finset.VAddAntidiagonal ⋯  …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    aesop (add safe [Set.vadd_mem_vadd, Set.add_mem_add]) (add simp [add_vadd, mul_smul])
    /-
      🎉 no goals
    -/


instance instBaseModule [Semiring R] [Module R V] : Module R (HahnModule Γ' R V) :=
  inferInstanceAs <| Module R (HahnSeries Γ' V)


instance instModule [Semiring R] [Module R V] : Module (HahnSeries Γ R)
    (HahnModule Γ' R V) := {
  inferInstanceAs (DistribSMul (HahnSeries Γ R) (HahnModule Γ' R V)) with
  mul_smul := mul_smul'
  one_smul := fun _ => one_smul'
  add_smul := fun _ _ _ => add_smul Module.add_smul
  zero_smul := fun _ => zero_smul' }


instance instNoZeroSMulDivisors {Γ} [LinearOrderedCancelAddCommMonoid Γ] [Zero R]
    [SMulWithZero R V] [NoZeroSMulDivisors R V] :
    NoZeroSMulDivisors (HahnSeries Γ R) (HahnModule Γ R V) where
  eq_zero_or_eq_zero_of_smul_eq_zero {x y} hxy := by
    /-
      Γ✝ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝⁸ : OrderedCancelAddCommMonoid Γ✝
      inst✝⁷ : PartialOrder Γ'
      inst✝⁶ : AddAction Γ✝ Γ'
      inst✝⁵ : IsOrderedCancelVAdd Γ✝ Γ'
      inst✝⁴ : AddCommMonoid V
      Γ : Type u_6
      inst✝³ : LinearOrderedCancelAddCommMonoid Γ
      inst✝² : Zero R
      inst✝¹ : SMulWithZero R V
      inst✝ : NoZeroSMulDivisors R V
      x : HahnSeries Γ R
      y : HahnModule Γ R V
      hxy : Eq (HSMul.hSMul x y) 0
      ⊢ Or (Eq x 0) (Eq y 0)
    -/
    contrapose! hxy
    /-
      Γ✝ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝⁸ : OrderedCancelAddCommMonoid Γ✝
      inst✝⁷ : PartialOrder Γ'
      inst✝⁶ : AddAction Γ✝ Γ'
      inst✝⁵ : IsOrderedCancelVAdd Γ✝ Γ'
      inst✝⁴ : AddCommMonoid V
      Γ : Type u_6
      inst✝³ : LinearOrderedCancelAddCommMonoid Γ
      inst✝² : Zero R
      inst✝¹ : SMulWithZero R V
      inst✝ : NoZeroSMulDivisors R V
      x : HahnSeries Γ R
      y : HahnModule Γ R V
      hxy : And (Ne x 0) (Ne y 0)
      ⊢ Ne (HSMul.hSMul x y) 0
    -/
    simp only [ne_eq]
    /-
      Γ✝ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝⁸ : OrderedCancelAddCommMonoid Γ✝
      inst✝⁷ : PartialOrder Γ'
      inst✝⁶ : AddAction Γ✝ Γ'
      inst✝⁵ : IsOrderedCancelVAdd Γ✝ Γ'
      inst✝⁴ : AddCommMonoid V
      Γ : Type u_6
      inst✝³ : LinearOrderedCancelAddCommMonoid Γ
      inst✝² : Zero R
      inst✝¹ : SMulWithZero R V
      inst✝ : NoZeroSMulDivisors R V
      x : HahnSeries Γ R
      y : HahnModule Γ R V
      hxy : And (Ne x 0) (Ne y 0)
      ⊢ Not (Eq (HSMul.hSMul x y) 0)
    -/
    rw [HahnModule.ext_iff, funext_iff, not_forall]
    /-
      Γ✝ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝⁸ : OrderedCancelAddCommMonoid Γ✝
      inst✝⁷ : PartialOrder Γ'
      inst✝⁶ : AddAction Γ✝ Γ'
      inst✝⁵ : IsOrderedCancelVAdd Γ✝ Γ'
      inst✝⁴ : AddCommMonoid V
      Γ : Type u_6
      inst✝³ : LinearOrderedCancelAddCommMonoid Γ
      inst✝² : Zero R
      inst✝¹ : SMulWithZero R V
      inst✝ : NoZeroSMulDivisors R V
      x : HahnSeries Γ R
      y : HahnModule Γ R V
      hxy : And (Ne x 0) (Ne y 0)
      ⊢ Exists fun x_1 => Not (Eq (((HahnModule.of R).symm (HSMul.hSMul x y)).coeff  …
    -/
    refine ⟨x.order + ((of R).symm y).order, ?_⟩
    /-
      Γ✝ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝⁸ : OrderedCancelAddCommMonoid Γ✝
      inst✝⁷ : PartialOrder Γ'
      inst✝⁶ : AddAction Γ✝ Γ'
      inst✝⁵ : IsOrderedCancelVAdd Γ✝ Γ'
      inst✝⁴ : AddCommMonoid V
      Γ : Type u_6
      inst✝³ : LinearOrderedCancelAddCommMonoid Γ
      inst✝² : Zero R
      inst✝¹ : SMulWithZero R V
      inst✝ : NoZeroSMulDivisors R V
      x : HahnSeries Γ R
      y : HahnModule Γ R V
      hxy : And (Ne x 0) (Ne y 0)
      ⊢ Not (Eq (((HahnModule.of R).symm (HSMul.hSMul x y)).coeff (HAdd.hAdd x.order …
    -/
    rw [smul_coeff_order_add_order x y, of_symm_zero, HahnSeries.zero_coeff, smul_eq_zero, not_or]
    /-
      Γ✝ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝⁸ : OrderedCancelAddCommMonoid Γ✝
      inst✝⁷ : PartialOrder Γ'
      inst✝⁶ : AddAction Γ✝ Γ'
      inst✝⁵ : IsOrderedCancelVAdd Γ✝ Γ'
      inst✝⁴ : AddCommMonoid V
      Γ : Type u_6
      inst✝³ : LinearOrderedCancelAddCommMonoid Γ
      inst✝² : Zero R
      inst✝¹ : SMulWithZero R V
      inst✝ : NoZeroSMulDivisors R V
      x : HahnSeries Γ R
      y : HahnModule Γ R V
      hxy : And (Ne x 0) (Ne y 0)
      ⊢ And (Not (Eq x.leadingCoeff 0)) (Not (Eq ((HahnModule.of R).symm y).leadingC …
    -/
    constructor
      /-
        case left
        Γ✝ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝⁸ : OrderedCancelAddCommMonoid Γ✝
        inst✝⁷ : PartialOrder Γ'
        inst✝⁶ : AddAction Γ✝ Γ'
        inst✝⁵ : IsOrderedCancelVAdd Γ✝ Γ'
        inst✝⁴ : AddCommMonoid V
        Γ : Type u_6
        inst✝³ : LinearOrderedCancelAddCommMonoid Γ
        inst✝² : Zero R
        inst✝¹ : SMulWithZero R V
        inst✝ : NoZeroSMulDivisors R V
        x : HahnSeries Γ R
        y : HahnModule Γ R V
        hxy : And (Ne x 0) (Ne y 0)
        ⊢ Not (Eq x.leadingCoeff 0)
      -/
    · exact HahnSeries.leadingCoeff_ne_iff.mpr hxy.1
      /-
        🎉 no goals
      -/
      /-
        case right
        Γ✝ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝⁸ : OrderedCancelAddCommMonoid Γ✝
        inst✝⁷ : PartialOrder Γ'
        inst✝⁶ : AddAction Γ✝ Γ'
        inst✝⁵ : IsOrderedCancelVAdd Γ✝ Γ'
        inst✝⁴ : AddCommMonoid V
        Γ : Type u_6
        inst✝³ : LinearOrderedCancelAddCommMonoid Γ
        inst✝² : Zero R
        inst✝¹ : SMulWithZero R V
        inst✝ : NoZeroSMulDivisors R V
        x : HahnSeries Γ R
        y : HahnModule Γ R V
        hxy : And (Ne x 0) (Ne y 0)
        ⊢ Not (Eq ((HahnModule.of R).symm y).leadingCoeff 0)
      -/
    · exact HahnSeries.leadingCoeff_ne_iff.mpr hxy.2
      /-
        🎉 no goals
      -/


instance {Γ} [LinearOrderedCancelAddCommMonoid Γ] [NonUnitalNonAssocSemiring R] [NoZeroDivisors R] :
    NoZeroDivisors (HahnSeries Γ R) where
    eq_zero_or_eq_zero_of_mul_eq_zero {x y} xy := by
      haveI : NoZeroSMulDivisors (HahnSeries Γ R) (HahnSeries Γ R) :=
        HahnModule.instNoZeroSMulDivisors
      /-
        Γ✝ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝³ : OrderedCancelAddCommMonoid Γ✝
        Γ : Type u_6
        inst✝² : LinearOrderedCancelAddCommMonoid Γ
        inst✝¹ : NonUnitalNonAssocSemiring R
        inst✝ : NoZeroDivisors R
        x y : HahnSeries Γ R
        xy : Eq (HMul.hMul x y) 0
        this : NoZeroSMulDivisors (HahnSeries Γ R) (HahnSeries Γ R)
        ⊢ Or (Eq x 0) (Eq y 0)
      -/
      exact eq_zero_or_eq_zero_of_smul_eq_zero xy
      /-
        🎉 no goals
      -/


instance {Γ} [LinearOrderedCancelAddCommMonoid Γ] [Ring R] [IsDomain R] :
    IsDomain (HahnSeries Γ R) :=
  NoZeroDivisors.to_isDomain _


theorem orderTop_add_orderTop_le_orderTop_mul {Γ} [LinearOrderedCancelAddCommMonoid Γ]
    [NonUnitalNonAssocSemiring R] {x y : HahnSeries Γ R} :
    x.orderTop + y.orderTop ≤ (x * y).orderTop := by
  /-
    R : Type u_3
    Γ : Type u_6
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    x y : HahnSeries Γ R
    ⊢ LE.le (HAdd.hAdd x.orderTop y.orderTop) (HMul.hMul x y).orderTop
  -/
  by_cases hx : x = 0; · simp [hx]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_3
    Γ : Type u_6
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    x y : HahnSeries Γ R
    hx : Not (Eq x 0)
    ⊢ LE.le (HAdd.hAdd x.orderTop y.orderTop) (HMul.hMul x y).orderTop
  -/
  by_cases hy : y = 0; · simp [hy]
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    R : Type u_3
    Γ : Type u_6
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    x y : HahnSeries Γ R
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    ⊢ LE.le (HAdd.hAdd x.orderTop y.orderTop) (HMul.hMul x y).orderTop
  -/
  by_cases hxy : x * y = 0
    /-
      case pos
      R : Type u_3
      Γ : Type u_6
      inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      x y : HahnSeries Γ R
      hx : Not (Eq x 0)
      hy : Not (Eq y 0)
      hxy : Eq (HMul.hMul x y) 0
      ⊢ LE.le (HAdd.hAdd x.orderTop y.orderTop) (HMul.hMul x y).orderTop
    -/
  · simp [hxy]
    /-
      🎉 no goals
    -/
  rw [orderTop_of_ne hx, orderTop_of_ne hy, orderTop_of_ne hxy, ← WithTop.coe_add,
    WithTop.coe_le_coe, ← Set.IsWF.min_add]
  /-
    case neg
    R : Type u_3
    Γ : Type u_6
    inst✝¹ : LinearOrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    x y : HahnSeries Γ R
    hx : Not (Eq x 0)
    hy : Not (Eq y 0)
    hxy : Not (Eq (HMul.hMul x y) 0)
    ⊢ LE.le (⋯.min ⋯) (⋯.min ⋯)
  -/
  exact Set.IsWF.min_le_min_of_subset support_mul_subset_add_support
  /-
    🎉 no goals
  -/


@[simp]
theorem order_mul {Γ} [LinearOrderedCancelAddCommMonoid Γ] [NonUnitalNonAssocSemiring R]
    [NoZeroDivisors R] {x y : HahnSeries Γ R} (hx : x ≠ 0) (hy : y ≠ 0) :
    (x * y).order = x.order + y.order := by
  /-
    R : Type u_3
    Γ : Type u_6
    inst✝² : LinearOrderedCancelAddCommMonoid Γ
    inst✝¹ : NonUnitalNonAssocSemiring R
    inst✝ : NoZeroDivisors R
    x y : HahnSeries Γ R
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (HMul.hMul x y).order (HAdd.hAdd x.order y.order)
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_3
      Γ : Type u_6
      inst✝² : LinearOrderedCancelAddCommMonoid Γ
      inst✝¹ : NonUnitalNonAssocSemiring R
      inst✝ : NoZeroDivisors R
      x y : HahnSeries Γ R
      hx : Ne x 0
      hy : Ne y 0
      ⊢ LE.le (HMul.hMul x y).order (HAdd.hAdd x.order y.order)
    -/
  · apply order_le_of_coeff_ne_zero
    /-
      case a.h
      R : Type u_3
      Γ : Type u_6
      inst✝² : LinearOrderedCancelAddCommMonoid Γ
      inst✝¹ : NonUnitalNonAssocSemiring R
      inst✝ : NoZeroDivisors R
      x y : HahnSeries Γ R
      hx : Ne x 0
      hy : Ne y 0
      ⊢ Ne ((HMul.hMul x y).coeff (HAdd.hAdd x.order y.order)) 0
    -/
    rw [mul_coeff_order_add_order x y]
    /-
      case a.h
      R : Type u_3
      Γ : Type u_6
      inst✝² : LinearOrderedCancelAddCommMonoid Γ
      inst✝¹ : NonUnitalNonAssocSemiring R
      inst✝ : NoZeroDivisors R
      x y : HahnSeries Γ R
      hx : Ne x 0
      hy : Ne y 0
      ⊢ Ne (HMul.hMul x.leadingCoeff y.leadingCoeff) 0
    -/
    exact mul_ne_zero (leadingCoeff_ne_iff.mpr hx) (leadingCoeff_ne_iff.mpr hy)
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_3
      Γ : Type u_6
      inst✝² : LinearOrderedCancelAddCommMonoid Γ
      inst✝¹ : NonUnitalNonAssocSemiring R
      inst✝ : NoZeroDivisors R
      x y : HahnSeries Γ R
      hx : Ne x 0
      hy : Ne y 0
      ⊢ LE.le (HAdd.hAdd x.order y.order) (HMul.hMul x y).order
    -/
  · rw [order_of_ne hx, order_of_ne hy, order_of_ne (mul_ne_zero hx hy), ← Set.IsWF.min_add]
    /-
      case a
      R : Type u_3
      Γ : Type u_6
      inst✝² : LinearOrderedCancelAddCommMonoid Γ
      inst✝¹ : NonUnitalNonAssocSemiring R
      inst✝ : NoZeroDivisors R
      x y : HahnSeries Γ R
      hx : Ne x 0
      hy : Ne y 0
      ⊢ LE.le (⋯.min ⋯) (⋯.min ⋯)
    -/
    exact Set.IsWF.min_le_min_of_subset support_mul_subset_add_support
    /-
      🎉 no goals
    -/


@[simp]
theorem order_pow {Γ} [LinearOrderedCancelAddCommMonoid Γ] [Semiring R] [NoZeroDivisors R]
    (x : HahnSeries Γ R) (n : ℕ) : (x ^ n).order = n • x.order := by
  /-
    R : Type u_3
    Γ : Type u_6
    inst✝² : LinearOrderedCancelAddCommMonoid Γ
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    x : HahnSeries Γ R
    n : Nat
    ⊢ Eq (HPow.hPow x n).order (HSMul.hSMul n x.order)
  -/
  induction' n with h IH
    /-
      case zero
      R : Type u_3
      Γ : Type u_6
      inst✝² : LinearOrderedCancelAddCommMonoid Γ
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      x : HahnSeries Γ R
      ⊢ Eq (HPow.hPow x 0).order (HSMul.hSMul 0 x.order)
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    R : Type u_3
    Γ : Type u_6
    inst✝² : LinearOrderedCancelAddCommMonoid Γ
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    x : HahnSeries Γ R
    h : Nat
    IH : Eq (HPow.hPow x h).order (HSMul.hSMul h x.order)
    ⊢ Eq (HPow.hPow x (HAdd.hAdd h 1)).order (HSMul.hSMul (HAdd.hAdd h 1) x.order)
  -/
  rcases eq_or_ne x 0 with (rfl | hx)
    /-
      case succ.inl
      R : Type u_3
      Γ : Type u_6
      inst✝² : LinearOrderedCancelAddCommMonoid Γ
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      h : Nat
      IH : Eq (HPow.hPow 0 h).order (HSMul.hSMul h (HahnSeries.order 0))
      ⊢ Eq (HPow.hPow 0 (HAdd.hAdd h 1)).order (HSMul.hSMul (HAdd.hAdd h 1) (HahnSer …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ.inr
    R : Type u_3
    Γ : Type u_6
    inst✝² : LinearOrderedCancelAddCommMonoid Γ
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    x : HahnSeries Γ R
    h : Nat
    IH : Eq (HPow.hPow x h).order (HSMul.hSMul h x.order)
    hx : Ne x 0
    ⊢ Eq (HPow.hPow x (HAdd.hAdd h 1)).order (HSMul.hSMul (HAdd.hAdd h 1) x.order)
  -/
  rw [pow_succ, order_mul (pow_ne_zero _ hx) hx, succ_nsmul, IH]
  /-
    🎉 no goals
  -/


@[simp]
theorem single_mul_single {a b : Γ} {r s : R} :
    single a r * single b s = single (a + b) (r * s) := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    a b : Γ
    r s : R
    ⊢ Eq (HMul.hMul ((HahnSeries.single a) r) ((HahnSeries.single b) s)) ((HahnSer …
  -/
  ext x
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonUnitalNonAssocSemiring R
    a b : Γ
    r s : R
    x : Γ
    ⊢ Eq ((HMul.hMul ((HahnSeries.single a) r) ((HahnSeries.single b) s)).coeff x) …
  -/
  by_cases h : x = a + b
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      a b : Γ
      r s : R
      x : Γ
      h : Eq x (HAdd.hAdd a b)
      ⊢ Eq ((HMul.hMul ((HahnSeries.single a) r) ((HahnSeries.single b) s)).coeff x) …
    -/
  · rw [h, mul_single_coeff_add]
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      a b : Γ
      r s : R
      x : Γ
      h : Eq x (HAdd.hAdd a b)
      ⊢ Eq (HMul.hMul (((HahnSeries.single a) r).coeff a) s) (((HahnSeries.single (H …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      a b : Γ
      r s : R
      x : Γ
      h : Not (Eq x (HAdd.hAdd a b))
      ⊢ Eq ((HMul.hMul ((HahnSeries.single a) r) ((HahnSeries.single b) s)).coeff x) …
    -/
  · rw [single_coeff_of_ne h, mul_coeff, sum_eq_zero]
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      a b : Γ
      r s : R
      x : Γ
      h : Not (Eq x (HAdd.hAdd a b))
      ⊢ ∀ (x_1 : Prod Γ Γ), Membership.mem (Finset.addAntidiagonal ⋯ ⋯ x) x_1 → Eq ( …
    -/
    simp_rw [mem_addAntidiagonal]
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      a b : Γ
      r s : R
      x : Γ
      h : Not (Eq x (HAdd.hAdd a b))
      ⊢ ∀ (x_1 : Prod Γ Γ), And (Membership.mem ((HahnSeries.single a) r).support x_ …
    -/
    rintro ⟨y, z⟩ ⟨hy, hz, rfl⟩
    /-
      case neg.mk.intro.intro
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      a b : Γ
      r s : R
      y z : Γ
      hy : Membership.mem ((HahnSeries.single a) r).support { fst := y, snd := z }.1
      hz : Membership.mem ((HahnSeries.single b) s).support { fst := y, snd := z }.2
      h : Not (Eq (HAdd.hAdd { fst := y, snd := z }.1 { fst := y, snd := z }.2) (HAd …
      ⊢ Eq (HMul.hMul (((HahnSeries.single a) r).coeff { fst := y, snd := z }.1) ((( …
    -/
    rw [eq_of_mem_support_single hy, eq_of_mem_support_single hz] at h
    /-
      case neg.mk.intro.intro
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonUnitalNonAssocSemiring R
      a b : Γ
      r s : R
      y z : Γ
      hy : Membership.mem ((HahnSeries.single a) r).support { fst := y, snd := z }.1
      hz : Membership.mem ((HahnSeries.single b) s).support { fst := y, snd := z }.2
      h : Not (Eq (HAdd.hAdd a b) (HAdd.hAdd a b))
      ⊢ Eq (HMul.hMul (((HahnSeries.single a) r).coeff { fst := y, snd := z }.1) ((( …
    -/
    exact (h rfl).elim
    /-
      🎉 no goals
    -/


@[simp]
theorem single_pow (a : Γ) (n : ℕ) (r : R) : single a r ^ n = single (n • a) (r ^ n) := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : Semiring R
    a : Γ
    n : Nat
    r : R
    ⊢ Eq (HPow.hPow ((HahnSeries.single a) r) n) ((HahnSeries.single (HSMul.hSMul  …
  -/
  induction' n with n IH
    /-
      case zero
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : Semiring R
      a : Γ
      r : R
      ⊢ Eq (HPow.hPow ((HahnSeries.single a) r) 0) ((HahnSeries.single (HSMul.hSMul  …
    -/
  · ext; simp only [pow_zero, one_coeff, zero_smul, single_coeff]
         /-
           🎉 no goals
         -/
    /-
      case succ
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : Semiring R
      a : Γ
      r : R
      n : Nat
      IH : Eq (HPow.hPow ((HahnSeries.single a) r) n) ((HahnSeries.single (HSMul.hSM …
      ⊢ Eq (HPow.hPow ((HahnSeries.single a) r) (HAdd.hAdd n 1)) ((HahnSeries.single …
    -/
  · rw [pow_succ, pow_succ, IH, single_mul_single, succ_nsmul]
    /-
      🎉 no goals
    -/


/-- `C a` is the constant Hahn Series `a`. `C` is provided as a ring homomorphism. -/
@[simps]
def C : R →+* HahnSeries Γ R where
  toFun := single 0
  map_zero' := single_eq_zero
  map_one' := rfl
  map_add' x y := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonAssocSemiring R
      x y : R
      ⊢ Eq ((↑{ toFun := ⇑(HahnSeries.single 0), map_one' := ⋯, map_mul' := ⋯ }).toF …
    -/
    ext a
                     /-
                       Γ : Type u_1
                       Γ' : Type u_2
                       R : Type u_3
                       S : Type u_4
                       V : Type u_5
                       inst✝¹ : OrderedCancelAddCommMonoid Γ
                       inst✝ : NonAssocSemiring R
                       x y : R
                       ⊢ Eq ({ toFun := ⇑(HahnSeries.single 0), map_one' := ⋯ }.toFun (HMul.hMul x y) …
                     -/
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonAssocSemiring R
      x y : R
      a : Γ
      ⊢ Eq (((↑{ toFun := ⇑(HahnSeries.single 0), map_one' := ⋯, map_mul' := ⋯ }).to …
    -/
                     /-
                       🎉 no goals
                     -/
                           /-
                             🎉 no goals
                           -/
    by_cases h : a = 0 <;> simp [h]
                           /-
                             🎉 no goals
                           -/
  map_mul' x y := by rw [single_mul_single, zero_add]


theorem C_zero : C (0 : R) = (0 : HahnSeries Γ R) :=
  C.map_zero


theorem C_one : C (1 : R) = (1 : HahnSeries Γ R) :=
  C.map_one


theorem map_C [NonAssocSemiring S] (a : R) (f : R →+* S) :
    ((C a).map f : HahnSeries Γ S) = C (f a) := by
  /-
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCancelAddCommMonoid Γ
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    a : R
    f : RingHom R S
    ⊢ Eq ((HahnSeries.C a).map f) (HahnSeries.C (f a))
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝² : OrderedCancelAddCommMonoid Γ
    inst✝¹ : NonAssocSemiring R
    inst✝ : NonAssocSemiring S
    a : R
    f : RingHom R S
    g : Γ
    ⊢ Eq (((HahnSeries.C a).map f).coeff g) ((HahnSeries.C (f a)).coeff g)
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : g = 0 <;> simp [h]
                         /-
                           🎉 no goals
                         -/


theorem C_injective : Function.Injective (C : R → HahnSeries Γ R) := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonAssocSemiring R
    ⊢ Function.Injective ⇑HahnSeries.C
  -/
  intro r s rs
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonAssocSemiring R
    r s : R
    rs : Eq (HahnSeries.C r) (HahnSeries.C s)
    ⊢ Eq r s
  -/
  rw [HahnSeries.ext_iff, funext_iff] at rs
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonAssocSemiring R
    r s : R
    rs : ∀ (x : Γ), Eq ((HahnSeries.C r).coeff x) ((HahnSeries.C s).coeff x)
    ⊢ Eq r s
  -/
  have h := rs 0
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonAssocSemiring R
    r s : R
    rs : ∀ (x : Γ), Eq ((HahnSeries.C r).coeff x) ((HahnSeries.C s).coeff x)
    h : Eq ((HahnSeries.C r).coeff 0) ((HahnSeries.C s).coeff 0)
    ⊢ Eq r s
  -/
  rwa [C_apply, single_coeff_same, C_apply, single_coeff_same] at h
  /-
    🎉 no goals
  -/


theorem C_ne_zero {r : R} (h : r ≠ 0) : (C r : HahnSeries Γ R) ≠ 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonAssocSemiring R
    r : R
    h : Ne r 0
    ⊢ Ne (HahnSeries.C r) 0
  -/
  contrapose! h
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonAssocSemiring R
    r : R
    h : Eq (HahnSeries.C r) 0
    ⊢ Eq r 0
  -/
  rw [← C_zero] at h
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonAssocSemiring R
    r : R
    h : Eq (HahnSeries.C r) (HahnSeries.C 0)
    ⊢ Eq r 0
  -/
  exact C_injective h
  /-
    🎉 no goals
  -/


theorem order_C {r : R} : order (C r : HahnSeries Γ R) = 0 := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝¹ : OrderedCancelAddCommMonoid Γ
    inst✝ : NonAssocSemiring R
    r : R
    ⊢ Eq (HahnSeries.C r).order 0
  -/
  by_cases h : r = 0
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonAssocSemiring R
      r : R
      h : Eq r 0
      ⊢ Eq (HahnSeries.C r).order 0
    -/
  · rw [h, C_zero, order_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝¹ : OrderedCancelAddCommMonoid Γ
      inst✝ : NonAssocSemiring R
      r : R
      h : Not (Eq r 0)
      ⊢ Eq (HahnSeries.C r).order 0
    -/
  · exact order_single h
    /-
      🎉 no goals
    -/


theorem C_mul_eq_smul {r : R} {x : HahnSeries Γ R} : C r * x = r • x :=
  single_zero_mul_eq_smul


theorem embDomain_mul [NonUnitalNonAssocSemiring R] (f : Γ ↪o Γ')
    (hf : ∀ x y, f (x + y) = f x + f y) (x y : HahnSeries Γ R) :
    embDomain f (x * y) = embDomain f x * embDomain f y := by
  /-
    Γ : Type u_1
    R : Type u_3
    inst✝² : OrderedCancelAddCommMonoid Γ
    Γ' : Type u_6
    inst✝¹ : OrderedCancelAddCommMonoid Γ'
    inst✝ : NonUnitalNonAssocSemiring R
    f : OrderEmbedding Γ Γ'
    hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    x y : HahnSeries Γ R
    ⊢ Eq (HahnSeries.embDomain f (HMul.hMul x y)) (HMul.hMul (HahnSeries.embDomain …
  -/
  ext g
  /-
    case coeff.h
    Γ : Type u_1
    R : Type u_3
    inst✝² : OrderedCancelAddCommMonoid Γ
    Γ' : Type u_6
    inst✝¹ : OrderedCancelAddCommMonoid Γ'
    inst✝ : NonUnitalNonAssocSemiring R
    f : OrderEmbedding Γ Γ'
    hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
    x y : HahnSeries Γ R
    g : Γ'
    ⊢ Eq ((HahnSeries.embDomain f (HMul.hMul x y)).coeff g) ((HMul.hMul (HahnSerie …
  -/
  by_cases hg : g ∈ Set.range f
    /-
      case pos
      Γ : Type u_1
      R : Type u_3
      inst✝² : OrderedCancelAddCommMonoid Γ
      Γ' : Type u_6
      inst✝¹ : OrderedCancelAddCommMonoid Γ'
      inst✝ : NonUnitalNonAssocSemiring R
      f : OrderEmbedding Γ Γ'
      hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      x y : HahnSeries Γ R
      g : Γ'
      hg : Membership.mem (Set.range ⇑f) g
      ⊢ Eq ((HahnSeries.embDomain f (HMul.hMul x y)).coeff g) ((HMul.hMul (HahnSerie …
    -/
  · obtain ⟨g, rfl⟩ := hg
    /-
      case pos.intro
      Γ : Type u_1
      R : Type u_3
      inst✝² : OrderedCancelAddCommMonoid Γ
      Γ' : Type u_6
      inst✝¹ : OrderedCancelAddCommMonoid Γ'
      inst✝ : NonUnitalNonAssocSemiring R
      f : OrderEmbedding Γ Γ'
      hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      x y : HahnSeries Γ R
      g : Γ
      ⊢ Eq ((HahnSeries.embDomain f (HMul.hMul x y)).coeff (f g)) ((HMul.hMul (HahnS …
    -/
    simp only [mul_coeff, embDomain_coeff]
    trans
      ∑ ij in
        (addAntidiagonal x.isPWO_support y.isPWO_support g).map
          (Function.Embedding.prodMap f.toEmbedding f.toEmbedding),
        (embDomain f x).coeff ij.1 * (embDomain f y).coeff ij.2
      /-
        Γ : Type u_1
        R : Type u_3
        inst✝² : OrderedCancelAddCommMonoid Γ
        Γ' : Type u_6
        inst✝¹ : OrderedCancelAddCommMonoid Γ'
        inst✝ : NonUnitalNonAssocSemiring R
        f : OrderEmbedding Γ Γ'
        hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        x y : HahnSeries Γ R
        g : Γ
        ⊢ Eq ((Finset.addAntidiagonal ⋯ ⋯ g).sum fun ij => HMul.hMul (x.coeff ij.1) (y …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      Γ : Type u_1
      R : Type u_3
      inst✝² : OrderedCancelAddCommMonoid Γ
      Γ' : Type u_6
      inst✝¹ : OrderedCancelAddCommMonoid Γ'
      inst✝ : NonUnitalNonAssocSemiring R
      f : OrderEmbedding Γ Γ'
      hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      x y : HahnSeries Γ R
      g : Γ
      ⊢ Eq ((Finset.map (f.prodMap f.toEmbedding) (Finset.addAntidiagonal ⋯ ⋯ g)).su …
    -/
    apply sum_subset
      /-
        case h
        Γ : Type u_1
        R : Type u_3
        inst✝² : OrderedCancelAddCommMonoid Γ
        Γ' : Type u_6
        inst✝¹ : OrderedCancelAddCommMonoid Γ'
        inst✝ : NonUnitalNonAssocSemiring R
        f : OrderEmbedding Γ Γ'
        hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        x y : HahnSeries Γ R
        g : Γ
        ⊢ HasSubset.Subset (Finset.map (f.prodMap f.toEmbedding) (Finset.addAntidiagon …
      -/
    · rintro ⟨i, j⟩ hij
      simp only [exists_prop, mem_map, Prod.mk.inj_iff, mem_addAntidiagonal,
        Function.Embedding.coe_prodMap, mem_support, Prod.exists] at hij
      /-
        case h.mk
        Γ : Type u_1
        R : Type u_3
        inst✝² : OrderedCancelAddCommMonoid Γ
        Γ' : Type u_6
        inst✝¹ : OrderedCancelAddCommMonoid Γ'
        inst✝ : NonUnitalNonAssocSemiring R
        f : OrderEmbedding Γ Γ'
        hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        x y : HahnSeries Γ R
        g : Γ
        i j : Γ'
        hij : Exists fun a => Exists fun b => And (And (Ne (x.coeff a) 0) (And (Ne (y. …
        ⊢ Membership.mem (Finset.addAntidiagonal ⋯ ⋯ (f g)) { fst := i, snd := j }
      -/
      obtain ⟨i, j, ⟨hx, hy, rfl⟩, rfl, rfl⟩ := hij
      /-
        case h.mk.intro.intro.intro.intro.intro.refl
        Γ : Type u_1
        R : Type u_3
        inst✝² : OrderedCancelAddCommMonoid Γ
        Γ' : Type u_6
        inst✝¹ : OrderedCancelAddCommMonoid Γ'
        inst✝ : NonUnitalNonAssocSemiring R
        f : OrderEmbedding Γ Γ'
        hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        x y : HahnSeries Γ R
        i j : Γ
        hx : Ne (x.coeff i) 0
        hy : Ne (y.coeff j) 0
        ⊢ Membership.mem (Finset.addAntidiagonal ⋯ ⋯ (f (HAdd.hAdd i j))) { fst := f.t …
      -/
      simp [hx, hy, hf]
      /-
        🎉 no goals
      -/
      /-
        case hf
        Γ : Type u_1
        R : Type u_3
        inst✝² : OrderedCancelAddCommMonoid Γ
        Γ' : Type u_6
        inst✝¹ : OrderedCancelAddCommMonoid Γ'
        inst✝ : NonUnitalNonAssocSemiring R
        f : OrderEmbedding Γ Γ'
        hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        x y : HahnSeries Γ R
        g : Γ
        ⊢ ∀ (x_1 : Prod Γ' Γ'), Membership.mem (Finset.addAntidiagonal ⋯ ⋯ (f g)) x_1  …
      -/
    · rintro ⟨_, _⟩ h1 h2
      /-
        case hf.mk
        Γ : Type u_1
        R : Type u_3
        inst✝² : OrderedCancelAddCommMonoid Γ
        Γ' : Type u_6
        inst✝¹ : OrderedCancelAddCommMonoid Γ'
        inst✝ : NonUnitalNonAssocSemiring R
        f : OrderEmbedding Γ Γ'
        hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        x y : HahnSeries Γ R
        g : Γ
        fst✝ snd✝ : Γ'
        h1 : Membership.mem (Finset.addAntidiagonal ⋯ ⋯ (f g)) { fst := fst✝, snd := s …
        h2 : Not (Membership.mem (Finset.map (f.prodMap f.toEmbedding) (Finset.addAnti …
        ⊢ Eq (HMul.hMul ((HahnSeries.embDomain f x).coeff { fst := fst✝, snd := snd✝ } …
      -/
      contrapose! h2
      /-
        case hf.mk
        Γ : Type u_1
        R : Type u_3
        inst✝² : OrderedCancelAddCommMonoid Γ
        Γ' : Type u_6
        inst✝¹ : OrderedCancelAddCommMonoid Γ'
        inst✝ : NonUnitalNonAssocSemiring R
        f : OrderEmbedding Γ Γ'
        hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        x y : HahnSeries Γ R
        g : Γ
        fst✝ snd✝ : Γ'
        h1 : Membership.mem (Finset.addAntidiagonal ⋯ ⋯ (f g)) { fst := fst✝, snd := s …
        h2 : Ne (HMul.hMul ((HahnSeries.embDomain f x).coeff { fst := fst✝, snd := snd …
        ⊢ Membership.mem (Finset.map (f.prodMap f.toEmbedding) (Finset.addAntidiagonal …
      -/
      obtain ⟨i, _, rfl⟩ := support_embDomain_subset (ne_zero_and_ne_zero_of_mul h2).1
      /-
        case hf.mk.intro.intro
        Γ : Type u_1
        R : Type u_3
        inst✝² : OrderedCancelAddCommMonoid Γ
        Γ' : Type u_6
        inst✝¹ : OrderedCancelAddCommMonoid Γ'
        inst✝ : NonUnitalNonAssocSemiring R
        f : OrderEmbedding Γ Γ'
        hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        x y : HahnSeries Γ R
        g : Γ
        snd✝ : Γ'
        i : Γ
        left✝ : Membership.mem x.support i
        h1 : Membership.mem (Finset.addAntidiagonal ⋯ ⋯ (f g)) { fst := f i, snd := sn …
        h2 : Ne (HMul.hMul ((HahnSeries.embDomain f x).coeff { fst := f i, snd := snd✝ …
        ⊢ Membership.mem (Finset.map (f.prodMap f.toEmbedding) (Finset.addAntidiagonal …
      -/
      obtain ⟨j, _, rfl⟩ := support_embDomain_subset (ne_zero_and_ne_zero_of_mul h2).2
      simp only [exists_prop, mem_map, Prod.mk.inj_iff, mem_addAntidiagonal,
        Function.Embedding.coe_prodMap, mem_support, Prod.exists]
      simp only [mem_addAntidiagonal, embDomain_coeff, mem_support, ← hf,
        OrderEmbedding.eq_iff_eq] at h1
      /-
        case hf.mk.intro.intro.intro.intro
        Γ : Type u_1
        R : Type u_3
        inst✝² : OrderedCancelAddCommMonoid Γ
        Γ' : Type u_6
        inst✝¹ : OrderedCancelAddCommMonoid Γ'
        inst✝ : NonUnitalNonAssocSemiring R
        f : OrderEmbedding Γ Γ'
        hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
        x y : HahnSeries Γ R
        g i : Γ
        left✝¹ : Membership.mem x.support i
        j : Γ
        left✝ : Membership.mem y.support j
        h2 : Ne (HMul.hMul ((HahnSeries.embDomain f x).coeff { fst := f i, snd := f j  …
        h1 : And (Ne (x.coeff i) 0) (And (Ne (y.coeff j) 0) (Eq (HAdd.hAdd i j) g))
        ⊢ Exists fun a => Exists fun b => And (And (Ne (x.coeff a) 0) (And (Ne (y.coef …
      -/
      exact ⟨i, j, h1, rfl⟩
      /-
        🎉 no goals
      -/
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝² : OrderedCancelAddCommMonoid Γ
      Γ' : Type u_6
      inst✝¹ : OrderedCancelAddCommMonoid Γ'
      inst✝ : NonUnitalNonAssocSemiring R
      f : OrderEmbedding Γ Γ'
      hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      x y : HahnSeries Γ R
      g : Γ'
      hg : Not (Membership.mem (Set.range ⇑f) g)
      ⊢ Eq ((HahnSeries.embDomain f (HMul.hMul x y)).coeff g) ((HMul.hMul (HahnSerie …
    -/
  · rw [embDomain_notin_range hg, eq_comm]
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝² : OrderedCancelAddCommMonoid Γ
      Γ' : Type u_6
      inst✝¹ : OrderedCancelAddCommMonoid Γ'
      inst✝ : NonUnitalNonAssocSemiring R
      f : OrderEmbedding Γ Γ'
      hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      x y : HahnSeries Γ R
      g : Γ'
      hg : Not (Membership.mem (Set.range ⇑f) g)
      ⊢ Eq ((HMul.hMul (HahnSeries.embDomain f x) (HahnSeries.embDomain f y)).coeff  …
    -/
    contrapose! hg
    /-
      case neg
      Γ : Type u_1
      R : Type u_3
      inst✝² : OrderedCancelAddCommMonoid Γ
      Γ' : Type u_6
      inst✝¹ : OrderedCancelAddCommMonoid Γ'
      inst✝ : NonUnitalNonAssocSemiring R
      f : OrderEmbedding Γ Γ'
      hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      x y : HahnSeries Γ R
      g : Γ'
      hg : Ne ((HMul.hMul (HahnSeries.embDomain f x) (HahnSeries.embDomain f y)).coe …
      ⊢ Membership.mem (Set.range ⇑f) g
    -/
    obtain ⟨_, hi, _, hj, rfl⟩ := support_mul_subset_add_support ((mem_support _ _).2 hg)
    /-
      case neg.intro.intro.intro.intro
      Γ : Type u_1
      R : Type u_3
      inst✝² : OrderedCancelAddCommMonoid Γ
      Γ' : Type u_6
      inst✝¹ : OrderedCancelAddCommMonoid Γ'
      inst✝ : NonUnitalNonAssocSemiring R
      f : OrderEmbedding Γ Γ'
      hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      x y : HahnSeries Γ R
      w✝¹ : Γ'
      hi : Membership.mem (HahnSeries.embDomain f x).support w✝¹
      w✝ : Γ'
      hj : Membership.mem (HahnSeries.embDomain f y).support w✝
      hg : Ne ((HMul.hMul (HahnSeries.embDomain f x) (HahnSeries.embDomain f y)).coe …
      ⊢ Membership.mem (Set.range ⇑f) ((fun x1 x2 => HAdd.hAdd x1 x2) w✝¹ w✝)
    -/
    obtain ⟨i, _, rfl⟩ := support_embDomain_subset hi
    /-
      case neg.intro.intro.intro.intro.intro.intro
      Γ : Type u_1
      R : Type u_3
      inst✝² : OrderedCancelAddCommMonoid Γ
      Γ' : Type u_6
      inst✝¹ : OrderedCancelAddCommMonoid Γ'
      inst✝ : NonUnitalNonAssocSemiring R
      f : OrderEmbedding Γ Γ'
      hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      x y : HahnSeries Γ R
      w✝ : Γ'
      hj : Membership.mem (HahnSeries.embDomain f y).support w✝
      i : Γ
      left✝ : Membership.mem x.support i
      hi : Membership.mem (HahnSeries.embDomain f x).support (f i)
      hg : Ne ((HMul.hMul (HahnSeries.embDomain f x) (HahnSeries.embDomain f y)).coe …
      ⊢ Membership.mem (Set.range ⇑f) ((fun x1 x2 => HAdd.hAdd x1 x2) (f i) w✝)
    -/
    obtain ⟨j, _, rfl⟩ := support_embDomain_subset hj
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro
      Γ : Type u_1
      R : Type u_3
      inst✝² : OrderedCancelAddCommMonoid Γ
      Γ' : Type u_6
      inst✝¹ : OrderedCancelAddCommMonoid Γ'
      inst✝ : NonUnitalNonAssocSemiring R
      f : OrderEmbedding Γ Γ'
      hf : ∀ (x y : Γ), Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
      x y : HahnSeries Γ R
      i : Γ
      left✝¹ : Membership.mem x.support i
      hi : Membership.mem (HahnSeries.embDomain f x).support (f i)
      j : Γ
      left✝ : Membership.mem y.support j
      hj : Membership.mem (HahnSeries.embDomain f y).support (f j)
      hg : Ne ((HMul.hMul (HahnSeries.embDomain f x) (HahnSeries.embDomain f y)).coe …
      ⊢ Membership.mem (Set.range ⇑f) ((fun x1 x2 => HAdd.hAdd x1 x2) (f i) (f j))
    -/
    exact ⟨i + j, hf i j⟩
    /-
      🎉 no goals
    -/


theorem embDomain_one [NonAssocSemiring R] (f : Γ ↪o Γ') (hf : f 0 = 0) :
    embDomain f (1 : HahnSeries Γ R) = (1 : HahnSeries Γ' R) :=
  embDomain_single.trans <| hf.symm ▸ rfl


/-- Extending the domain of Hahn series is a ring homomorphism. -/
@[simps]
def embDomainRingHom [NonAssocSemiring R] (f : Γ →+ Γ') (hfi : Function.Injective f)
    (hf : ∀ g g' : Γ, f g ≤ f g' ↔ g ≤ g') : HahnSeries Γ R →+* HahnSeries Γ' R where
  toFun := embDomain ⟨⟨f, hfi⟩, hf _ _⟩
  map_one' := embDomain_one _ f.map_zero
  map_mul' := embDomain_mul _ f.map_add
  map_zero' := embDomain_zero
  map_add' := embDomain_add _


theorem embDomainRingHom_C [NonAssocSemiring R] {f : Γ →+ Γ'} {hfi : Function.Injective f}
    {hf : ∀ g g' : Γ, f g ≤ f g' ↔ g ≤ g'} {r : R} : embDomainRingHom f hfi hf (C r) = C r :=
                             /-
                               Γ : Type u_1
                               R : Type u_3
                               inst✝² : OrderedCancelAddCommMonoid Γ
                               Γ' : Type u_6
                               inst✝¹ : OrderedCancelAddCommMonoid Γ'
                               inst✝ : NonAssocSemiring R
                               f : AddMonoidHom Γ Γ'
                               hfi : Function.Injective ⇑f
                               hf : ∀ (g g' : Γ), Iff (LE.le (f g) (f g')) (LE.le g g')
                               r : R
                               ⊢ Eq ((HahnSeries.single ({ toFun := ⇑f, inj' := hfi, map_rel_iff' := ⋯ } 0))  …
                             -/
  embDomain_single.trans (by simp)
                             /-
                               🎉 no goals
                             -/


instance : Algebra R (HahnSeries Γ A) where
  toRingHom := C.comp (algebraMap R A)
  smul_def' r x := by
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝³ : OrderedCancelAddCommMonoid Γ
      inst✝² : CommSemiring R
      A : Type u_6
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      r : R
      x : HahnSeries Γ A
      ⊢ Eq (HSMul.hSMul r x) (HMul.hMul ((HahnSeries.C.comp (algebraMap R A)) r) x)
    -/
    ext
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝³ : OrderedCancelAddCommMonoid Γ
      inst✝² : CommSemiring R
      A : Type u_6
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      r : R
      x : HahnSeries Γ A
      x✝ : Γ
      ⊢ Eq ((HSMul.hSMul r x).coeff x✝) ((HMul.hMul ((HahnSeries.C.comp (algebraMap  …
    -/
    /-
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝³ : OrderedCancelAddCommMonoid Γ
      inst✝² : CommSemiring R
      A : Type u_6
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      r : R
      x : HahnSeries Γ A
      ⊢ Eq (HMul.hMul ((HahnSeries.C.comp (algebraMap R A)) r) x) (HMul.hMul x ((Hah …
    -/
    simp
    /-
      🎉 no goals
    -/
  commutes' r x := by
    /-
      case coeff.h
      Γ : Type u_1
      Γ' : Type u_2
      R : Type u_3
      S : Type u_4
      V : Type u_5
      inst✝³ : OrderedCancelAddCommMonoid Γ
      inst✝² : CommSemiring R
      A : Type u_6
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      r : R
      x : HahnSeries Γ A
      x✝ : Γ
      ⊢ Eq (HSMul.hSMul r (x.coeff x✝)) (HMul.hMul (x.coeff x✝) ((algebraMap R A) r))
    -/
    ext
    /-
      🎉 no goals
    -/
    simp only [smul_coeff, single_zero_mul_eq_smul, RingHom.coe_comp, RingHom.toFun_eq_coe, C_apply,
      Function.comp_apply, algebraMap_smul, mul_single_zero_coeff]
    rw [← Algebra.commutes, Algebra.smul_def]


theorem C_eq_algebraMap : C = algebraMap R (HahnSeries Γ R) :=
  rfl


theorem algebraMap_apply {r : R} : algebraMap R (HahnSeries Γ A) r = C (algebraMap R A r) :=
  rfl


instance [Nontrivial Γ] [Nontrivial R] : Nontrivial (Subalgebra R (HahnSeries Γ R)) :=
  ⟨⟨⊥, ⊤, by
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝⁵ : OrderedCancelAddCommMonoid Γ
        inst✝⁴ : CommSemiring R
        A : Type u_6
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial Γ
        inst✝ : Nontrivial R
        ⊢ Ne Bot.bot Top.top
      -/
      rw [Ne, SetLike.ext_iff, not_forall]
      /-
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝⁵ : OrderedCancelAddCommMonoid Γ
        inst✝⁴ : CommSemiring R
        A : Type u_6
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial Γ
        inst✝ : Nontrivial R
        ⊢ Exists fun x => Not (Iff (Membership.mem Bot.bot x) (Membership.mem Top.top  …
      -/
      obtain ⟨a, ha⟩ := exists_ne (0 : Γ)
      /-
        case intro
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝⁵ : OrderedCancelAddCommMonoid Γ
        inst✝⁴ : CommSemiring R
        A : Type u_6
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial Γ
        inst✝ : Nontrivial R
        a : Γ
        ha : Ne a 0
        ⊢ Exists fun x => Not (Iff (Membership.mem Bot.bot x) (Membership.mem Top.top  …
      -/
      refine ⟨single a 1, ?_⟩
      /-
        case intro
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝⁵ : OrderedCancelAddCommMonoid Γ
        inst✝⁴ : CommSemiring R
        A : Type u_6
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial Γ
        inst✝ : Nontrivial R
        a : Γ
        ha : Ne a 0
        ⊢ Not (Iff (Membership.mem Bot.bot ((HahnSeries.single a) 1)) (Membership.mem  …
      -/
      simp only [Algebra.mem_bot, not_exists, Set.mem_range, iff_true, Algebra.mem_top]
      /-
        case intro
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝⁵ : OrderedCancelAddCommMonoid Γ
        inst✝⁴ : CommSemiring R
        A : Type u_6
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial Γ
        inst✝ : Nontrivial R
        a : Γ
        ha : Ne a 0
        ⊢ ∀ (x : R), Not (Eq ((algebraMap R (HahnSeries Γ R)) x) ((HahnSeries.single a …
      -/
      intro x
      /-
        case intro
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝⁵ : OrderedCancelAddCommMonoid Γ
        inst✝⁴ : CommSemiring R
        A : Type u_6
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial Γ
        inst✝ : Nontrivial R
        a : Γ
        ha : Ne a 0
        x : R
        ⊢ Not (Eq ((algebraMap R (HahnSeries Γ R)) x) ((HahnSeries.single a) 1))
      -/
      rw [HahnSeries.ext_iff, funext_iff, not_forall]
      /-
        case intro
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝⁵ : OrderedCancelAddCommMonoid Γ
        inst✝⁴ : CommSemiring R
        A : Type u_6
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial Γ
        inst✝ : Nontrivial R
        a : Γ
        ha : Ne a 0
        x : R
        ⊢ Exists fun x_1 => Not (Eq (((algebraMap R (HahnSeries Γ R)) x).coeff x_1) (( …
      -/
      refine ⟨a, ?_⟩
      /-
        case intro
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝⁵ : OrderedCancelAddCommMonoid Γ
        inst✝⁴ : CommSemiring R
        A : Type u_6
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial Γ
        inst✝ : Nontrivial R
        a : Γ
        ha : Ne a 0
        x : R
        ⊢ Not (Eq (((algebraMap R (HahnSeries Γ R)) x).coeff a) (((HahnSeries.single a …
      -/
      rw [single_coeff_same, algebraMap_apply, C_apply, single_coeff_of_ne ha]
      /-
        case intro
        Γ : Type u_1
        Γ' : Type u_2
        R : Type u_3
        S : Type u_4
        V : Type u_5
        inst✝⁵ : OrderedCancelAddCommMonoid Γ
        inst✝⁴ : CommSemiring R
        A : Type u_6
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        inst✝¹ : Nontrivial Γ
        inst✝ : Nontrivial R
        a : Γ
        ha : Ne a 0
        x : R
        ⊢ Not (Eq 0 1)
      -/
      exact zero_ne_one⟩⟩
      /-
        🎉 no goals
      -/


/-- Extending the domain of Hahn series is an algebra homomorphism. -/
@[simps!]
def embDomainAlgHom (f : Γ →+ Γ') (hfi : Function.Injective f)
    (hf : ∀ g g' : Γ, f g ≤ f g' ↔ g ≤ g') : HahnSeries Γ A →ₐ[R] HahnSeries Γ' A :=
  { embDomainRingHom f hfi hf with commutes' := fun _ => embDomainRingHom_C (hf := hf) }


