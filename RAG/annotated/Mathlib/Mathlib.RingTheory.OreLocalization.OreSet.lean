/-- Cancellability in monoids with zeros can act as a replacement for the `ore_right_cancel`
condition of an ore set. -/
def oreSetOfCancelMonoidWithZero {R : Type*} [CancelMonoidWithZero R] {S : Submonoid R}
    (oreNum : R → S → R) (oreDenom : R → S → S)
    (ore_eq : ∀ (r : R) (s : S), oreDenom r s * r = oreNum r s * s) : OreSet S :=
  { ore_right_cancel := fun _ _ s h => ⟨s, mul_eq_mul_left_iff.mpr (mul_eq_mul_right_iff.mp h)⟩
    oreNum
    oreDenom
    ore_eq }


/-- In rings without zero divisors, the first (cancellability) condition is always fulfilled,
it suffices to give a proof for the Ore condition itself. -/
def oreSetOfNoZeroDivisors {R : Type*} [Ring R] [NoZeroDivisors R] {S : Submonoid R}
    (oreNum : R → S → R) (oreDenom : R → S → S)
    (ore_eq : ∀ (r : R) (s : S), oreDenom r s * r = oreNum r s * s) : OreSet S :=
  letI : CancelMonoidWithZero R := NoZeroDivisors.toCancelMonoidWithZero
  oreSetOfCancelMonoidWithZero oreNum oreDenom ore_eq


lemma nonempty_oreSet_iff {R : Type*} [Ring R] {S : Submonoid R} :
    Nonempty (OreSet S) ↔ (∀ (r₁ r₂ : R) (s : S), r₁ * s = r₂ * s → ∃ s' : S, s' * r₁ = s' * r₂) ∧
      (∀ (r : R) (s : S), ∃ (r' : R) (s' : S), s' * r = r' * s) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    S : Submonoid R
    ⊢ Iff (Nonempty (OreLocalization.OreSet S)) (And (∀ (r₁ r₂ : R) (s : Subtype f …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝ : Ring R
      S : Submonoid R
      ⊢ Nonempty (OreLocalization.OreSet S) → And (∀ (r₁ r₂ : R) (s : Subtype fun x  …
    -/
  · exact fun ⟨_⟩ ↦ ⟨ore_right_cancel, fun r s ↦ ⟨oreNum r s, oreDenom r s, ore_eq r s⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝ : Ring R
      S : Submonoid R
      ⊢ And (∀ (r₁ r₂ : R) (s : Subtype fun x => Membership.mem S x), Eq (HMul.hMul  …
    -/
  · intro ⟨H, H'⟩
    /-
      case mpr
      R : Type u_1
      inst✝ : Ring R
      S : Submonoid R
      H : ∀ (r₁ r₂ : R) (s : Subtype fun x => Membership.mem S x), Eq (HMul.hMul r₁  …
      H' : ∀ (r : R) (s : Subtype fun x => Membership.mem S x), Exists fun r' => Exi …
      ⊢ Nonempty (OreLocalization.OreSet S)
    -/
    choose r' s' h using H'
    /-
      case mpr
      R : Type u_1
      inst✝ : Ring R
      S : Submonoid R
      H : ∀ (r₁ r₂ : R) (s : Subtype fun x => Membership.mem S x), Eq (HMul.hMul r₁  …
      r' : R → (Subtype fun x => Membership.mem S x) → R
      s' : R → (Subtype fun x => Membership.mem S x) → Subtype fun x => Membership.m …
      h : ∀ (r : R) (s : Subtype fun x => Membership.mem S x), Eq (HMul.hMul (↑(s' r …
      ⊢ Nonempty (OreLocalization.OreSet S)
    -/
    exact ⟨H, r', s', h⟩
    /-
      🎉 no goals
    -/


lemma nonempty_oreSet_iff_of_noZeroDivisors {R : Type*} [Ring R] [NoZeroDivisors R]
    {S : Submonoid R} :
    Nonempty (OreSet S) ↔ ∀ (r : R) (s : S), ∃ (r' : R) (s' : S), s' * r = r' * s := by
  /-
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : NoZeroDivisors R
    S : Submonoid R
    ⊢ Iff (Nonempty (OreLocalization.OreSet S)) (∀ (r : R) (s : Subtype fun x => M …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : NoZeroDivisors R
      S : Submonoid R
      ⊢ Nonempty (OreLocalization.OreSet S) → ∀ (r : R) (s : Subtype fun x => Member …
    -/
  · exact fun ⟨_⟩ ↦ fun r s ↦ ⟨oreNum r s, oreDenom r s, ore_eq r s⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : NoZeroDivisors R
      S : Submonoid R
      ⊢ (∀ (r : R) (s : Subtype fun x => Membership.mem S x), Exists fun r' => Exist …
    -/
  · intro H
    /-
      case mpr
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : NoZeroDivisors R
      S : Submonoid R
      H : ∀ (r : R) (s : Subtype fun x => Membership.mem S x), Exists fun r' => Exis …
      ⊢ Nonempty (OreLocalization.OreSet S)
    -/
    choose r' s' h using H
    /-
      case mpr
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : NoZeroDivisors R
      S : Submonoid R
      r' : R → (Subtype fun x => Membership.mem S x) → R
      s' : R → (Subtype fun x => Membership.mem S x) → Subtype fun x => Membership.m …
      h : ∀ (r : R) (s : Subtype fun x => Membership.mem S x), Eq (HMul.hMul (↑(s' r …
      ⊢ Nonempty (OreLocalization.OreSet S)
    -/
    exact ⟨oreSetOfNoZeroDivisors r' s' h⟩
    /-
      🎉 no goals
    -/


