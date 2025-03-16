/-- The center of a semigroup `M` is the set of elements that commute with everything in `M` -/
@[to_additive
      "The center of a semigroup `M` is the set of elements that commute with everything in `M`"]
def center : Subsemigroup M where
  carrier := Set.center M
  mul_mem' := Set.mul_mem_center

-- Porting note: `coe_center` is now redundant


/-- The center of a magma is commutative and associative. -/
@[to_additive "The center of an additive magma is commutative and associative."]
instance center.commSemigroup : CommSemigroup (center M) where
  mul_assoc _ b _ := Subtype.ext <| b.2.mid_assoc _ _
  mul_comm a _ := Subtype.ext <| a.2.comm _


@[to_additive]
theorem mem_center_iff {z : M} : z ∈ center M ↔ ∀ g, g * z = z * g := by
  /-
    M : Type u_1
    inst✝ : Semigroup M
    z : M
    ⊢ Iff (Membership.mem (Subsemigroup.center M) z) (∀ (g : M), Eq (HMul.hMul g z …
  -/
  rw [← Semigroup.mem_center_iff]
  /-
    M : Type u_1
    inst✝ : Semigroup M
    z : M
    ⊢ Iff (Membership.mem (Subsemigroup.center M) z) (Membership.mem (Set.center M …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


@[to_additive]
instance decidableMemCenter (a) [Decidable <| ∀ b : M, b * a = a * b] :
    Decidable (a ∈ center M) :=
  decidable_of_iff' _ Semigroup.mem_center_iff


@[to_additive (attr := simp)]
theorem center_eq_top : center M = ⊤ :=
  SetLike.coe_injective (Set.center_eq_univ M)


