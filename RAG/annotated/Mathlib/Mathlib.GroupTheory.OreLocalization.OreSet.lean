/-- A submonoid `S` of an additive monoid `R` is (left) Ore if common summands on the right can be
turned into common summands on the left, and if each pair of `r : R` and `s : S` admits an Ore
minuend `v : R` and an Ore subtrahend `u : S` such that `u + r = v + s`. -/
class AddOreSet {R : Type*} [AddMonoid R] (S : AddSubmonoid R) where
  /-- Common summands on the right can be turned into common summands on the left, a weak form of
cancellability. -/
  ore_right_cancel : ∀ (r₁ r₂ : R) (s : S), r₁ + s = r₂ + s → ∃ s' : S, s' + r₁ = s' + r₂
  /-- The Ore minuend of a difference. -/
  oreMin : R → S → R
  /-- The Ore subtrahend of a difference. -/
  oreSubtra : R → S → S
  /-- The Ore condition of a difference, expressed in terms of `oreMin` and `oreSubtra`. -/
  ore_eq : ∀ (r : R) (s : S), oreSubtra r s + r = oreMin r s + s


/-- A submonoid `S` of a monoid `R` is (left) Ore if common factors on the right can be turned
into common factors on the left, and if each pair of `r : R` and `s : S` admits an Ore numerator
`v : R` and an Ore denominator `u : S` such that `u * r = v * s`. -/
@[to_additive AddOreLocalization.AddOreSet]
class OreSet {R : Type*} [Monoid R] (S : Submonoid R) where
  /-- Common factors on the right can be turned into common factors on the left, a weak form of
cancellability. -/
  ore_right_cancel : ∀ (r₁ r₂ : R) (s : S), r₁ * s = r₂ * s → ∃ s' : S, s' * r₁ = s' * r₂
  /-- The Ore numerator of a fraction. -/
  oreNum : R → S → R
  /-- The Ore denominator of a fraction. -/
  oreDenom : R → S → S
  /-- The Ore condition of a fraction, expressed in terms of `oreNum` and `oreDenom`. -/
  ore_eq : ∀ (r : R) (s : S), oreDenom r s * r = oreNum r s * s

-- TODO: use this once it's available.
-- run_cmd to_additive.map_namespace `OreLocalization `AddOreLocalization


/-- Common factors on the right can be turned into common factors on the left, a weak form of
cancellability. -/
@[to_additive AddOreLocalization.ore_right_cancel]
theorem ore_right_cancel (r₁ r₂ : R) (s : S) (h : r₁ * s = r₂ * s) : ∃ s' : S, s' * r₁ = s' * r₂ :=
  OreSet.ore_right_cancel r₁ r₂ s h


/-- The Ore numerator of a fraction. -/
@[to_additive AddOreLocalization.oreMin "The Ore minuend of a difference."]
def oreNum (r : R) (s : S) : R :=
  OreSet.oreNum r s


/-- The Ore denominator of a fraction. -/
@[to_additive AddOreLocalization.oreSubtra "The Ore subtrahend of a difference."]
def oreDenom (r : R) (s : S) : S :=
  OreSet.oreDenom r s


/-- The Ore condition of a fraction, expressed in terms of `oreNum` and `oreDenom`. -/
@[to_additive AddOreLocalization.add_ore_eq
  "The Ore condition of a difference, expressed in terms of `oreMin` and `oreSubtra`."]
theorem ore_eq (r : R) (s : S) : oreDenom r s * r = oreNum r s * s :=
  OreSet.ore_eq r s


/-- The Ore condition bundled in a sigma type. This is useful in situations where we want to obtain
both witnesses and the condition for a given fraction. -/
@[to_additive AddOreLocalization.addOreCondition
  "The Ore condition bundled in a sigma type. This is useful in situations where we want to obtain
both witnesses and the condition for a given difference."]
def oreCondition (r : R) (s : S) : Σ'r' : R, Σ's' : S, s' * r = r' * s :=
  ⟨oreNum r s, oreDenom r s, ore_eq r s⟩


/-- The trivial submonoid is an Ore set. -/
@[to_additive AddOreLocalization.addOreSetBot]
instance oreSetBot : OreSet (⊥ : Submonoid R) where
  ore_right_cancel _ _ s h :=
    ⟨s, by
      /-
        R : Type u_1
        inst✝¹ : Monoid R
        S : Submonoid R
        inst✝ : OreLocalization.OreSet S
        x✝¹ x✝ : R
        s : Subtype fun x => Membership.mem Bot.bot x
        h : Eq (HMul.hMul x✝¹ ↑s) (HMul.hMul x✝ ↑s)
        ⊢ Eq (HMul.hMul (↑s) x✝¹) (HMul.hMul (↑s) x✝)
      -/
      rcases s with ⟨s, hs⟩
      /-
        case mk
        R : Type u_1
        inst✝¹ : Monoid R
        S : Submonoid R
        inst✝ : OreLocalization.OreSet S
        x✝¹ x✝ s : R
        hs : Membership.mem Bot.bot s
        h : Eq (HMul.hMul x✝¹ ↑⟨s, hs⟩) (HMul.hMul x✝ ↑⟨s, hs⟩)
        ⊢ Eq (HMul.hMul (↑⟨s, hs⟩) x✝¹) (HMul.hMul (↑⟨s, hs⟩) x✝)
      -/
      rw [Submonoid.mem_bot] at hs
      /-
        case mk
        R : Type u_1
        inst✝¹ : Monoid R
        S : Submonoid R
        inst✝ : OreLocalization.OreSet S
        x✝¹ x✝ s : R
        hs✝ : Membership.mem Bot.bot s
        hs : Eq s 1
        h : Eq (HMul.hMul x✝¹ ↑⟨s, hs✝⟩) (HMul.hMul x✝ ↑⟨s, hs✝⟩)
        ⊢ Eq (HMul.hMul (↑⟨s, hs✝⟩) x✝¹) (HMul.hMul (↑⟨s, hs✝⟩) x✝)
      -/
      subst hs
      /-
        case mk
        R : Type u_1
        inst✝¹ : Monoid R
        S : Submonoid R
        inst✝ : OreLocalization.OreSet S
        x✝¹ x✝ : R
        hs : Membership.mem Bot.bot 1
        h : Eq (HMul.hMul x✝¹ ↑⟨1, hs⟩) (HMul.hMul x✝ ↑⟨1, hs⟩)
        ⊢ Eq (HMul.hMul (↑⟨1, hs⟩) x✝¹) (HMul.hMul (↑⟨1, hs⟩) x✝)
      -/
      rw [mul_one, mul_one] at h
      /-
        case mk
        R : Type u_1
        inst✝¹ : Monoid R
        S : Submonoid R
        inst✝ : OreLocalization.OreSet S
        x✝¹ x✝ : R
        hs : Membership.mem Bot.bot 1
        h : Eq x✝¹ x✝
        ⊢ Eq (HMul.hMul (↑⟨1, hs⟩) x✝¹) (HMul.hMul (↑⟨1, hs⟩) x✝)
      -/
      subst h
      /-
        case mk
        R : Type u_1
        inst✝¹ : Monoid R
        S : Submonoid R
        inst✝ : OreLocalization.OreSet S
        x✝ : R
        hs : Membership.mem Bot.bot 1
        ⊢ Eq (HMul.hMul (↑⟨1, hs⟩) x✝) (HMul.hMul (↑⟨1, hs⟩) x✝)
      -/
      rfl⟩
      /-
        🎉 no goals
      -/
  oreNum r _ := r
  oreDenom _ s := s
  ore_eq _ s := by
    /-
      R : Type u_1
      inst✝¹ : Monoid R
      S : Submonoid R
      inst✝ : OreLocalization.OreSet S
      x✝ : R
      s : Subtype fun x => Membership.mem Bot.bot x
      ⊢ Eq (HMul.hMul (↑((fun x s => s) x✝ s)) x✝) (HMul.hMul ((fun r x => r) x✝ s)  …
    -/
    rcases s with ⟨s, hs⟩
    /-
      case mk
      R : Type u_1
      inst✝¹ : Monoid R
      S : Submonoid R
      inst✝ : OreLocalization.OreSet S
      x✝ s : R
      hs : Membership.mem Bot.bot s
      ⊢ Eq (HMul.hMul (↑((fun x s => s) x✝ ⟨s, hs⟩)) x✝) (HMul.hMul ((fun r x => r)  …
    -/
    rw [Submonoid.mem_bot] at hs
    /-
      case mk
      R : Type u_1
      inst✝¹ : Monoid R
      S : Submonoid R
      inst✝ : OreLocalization.OreSet S
      x✝ s : R
      hs✝ : Membership.mem Bot.bot s
      hs : Eq s 1
      ⊢ Eq (HMul.hMul (↑((fun x s => s) x✝ ⟨s, hs✝⟩)) x✝) (HMul.hMul ((fun r x => r) …
    -/
    simp [hs]
    /-
      🎉 no goals
    -/


/-- Every submonoid of a commutative monoid is an Ore set. -/
@[to_additive AddOreLocalization.addOreSetComm]
instance (priority := 100) oreSetComm {R} [CommMonoid R] (S : Submonoid R) : OreSet S where
                                     /-
                                       R✝ : Type u_1
                                       inst✝² : Monoid R✝
                                       S✝ : Submonoid R✝
                                       inst✝¹ : OreLocalization.OreSet S✝
                                       R : Type ?u.8049
                                       inst✝ : CommMonoid R
                                       S : Submonoid R
                                       m n : R
                                       s : Subtype fun x => Membership.mem S x
                                       h : Eq (HMul.hMul m ↑s) (HMul.hMul n ↑s)
                                       ⊢ Eq (HMul.hMul (↑s) m) (HMul.hMul (↑s) n)
                                     -/
  ore_right_cancel m n s h := ⟨s, by rw [mul_comm (s : R) n, mul_comm (s : R) m, h]⟩
                                     /-
                                       🎉 no goals
                                     -/
  oreNum r _ := r
  oreDenom _ s := s
                   /-
                     R✝ : Type u_1
                     inst✝² : Monoid R✝
                     S✝ : Submonoid R✝
                     inst✝¹ : OreLocalization.OreSet S✝
                     R : Type ?u.8049
                     inst✝ : CommMonoid R
                     S : Submonoid R
                     r : R
                     s : Subtype fun x => Membership.mem S x
                     ⊢ Eq (HMul.hMul (↑((fun x s => s) r s)) r) (HMul.hMul ((fun r x => r) r s) ↑s)
                   -/
  ore_eq r s := by rw [mul_comm]
                   /-
                     🎉 no goals
                   -/


