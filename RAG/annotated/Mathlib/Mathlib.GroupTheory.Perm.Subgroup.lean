instance sumCongrHom.decidableMemRange {α β : Type*} [DecidableEq α] [DecidableEq β] [Fintype α]
    [Fintype β] : DecidablePred (· ∈ (sumCongrHom α β).range) := fun _ => inferInstance


@[simp]
theorem sumCongrHom.card_range {α β : Type*} [Fintype (sumCongrHom α β).range]
    [Fintype (Perm α × Perm β)] :
    Fintype.card (sumCongrHom α β).range = Fintype.card (Perm α × Perm β) :=
  Fintype.card_eq.mpr ⟨(ofInjective (sumCongrHom α β) sumCongrHom_injective).symm⟩


instance sigmaCongrRightHom.decidableMemRange {α : Type*} {β : α → Type*} [DecidableEq α]
    [∀ a, DecidableEq (β a)] [Fintype α] [∀ a, Fintype (β a)] :
    DecidablePred (· ∈ (sigmaCongrRightHom β).range) := fun _ => inferInstance


@[simp]
theorem sigmaCongrRightHom.card_range {α : Type*} {β : α → Type*}
    [Fintype (sigmaCongrRightHom β).range] [Fintype (∀ a, Perm (β a))] :
    Fintype.card (sigmaCongrRightHom β).range = Fintype.card (∀ a, Perm (β a)) :=
  Fintype.card_eq.mpr ⟨(ofInjective (sigmaCongrRightHom β) sigmaCongrRightHom_injective).symm⟩


instance subtypeCongrHom.decidableMemRange {α : Type*} (p : α → Prop) [DecidablePred p]
    [Fintype (Perm { a // p a } × Perm { a // ¬p a })] [DecidableEq (Perm α)] :
    DecidablePred (· ∈ (subtypeCongrHom p).range) := fun _ => inferInstance


@[simp]
theorem subtypeCongrHom.card_range {α : Type*} (p : α → Prop) [DecidablePred p]
    [Fintype (subtypeCongrHom p).range] [Fintype (Perm { a // p a } × Perm { a // ¬p a })] :
    Fintype.card (subtypeCongrHom p).range =
      Fintype.card (Perm { a // p a } × Perm { a // ¬p a }) :=
  Fintype.card_eq.mpr ⟨(ofInjective (subtypeCongrHom p) (subtypeCongrHom_injective p)).symm⟩


/-- **Cayley's theorem**: Every group G is isomorphic to a subgroup of the symmetric group acting on
`G`. Note that we generalize this to an arbitrary "faithful" group action by `G`. Setting `H = G`
recovers the usual statement of Cayley's theorem via `RightCancelMonoid.faithfulSMul` -/
noncomputable def subgroupOfMulAction (G H : Type*) [Group G] [MulAction G H] [FaithfulSMul G H] :
    G ≃* (MulAction.toPermHom G H).range :=
  MulEquiv.ofLeftInverse' _ (Classical.choose_spec MulAction.toPerm_injective.hasLeftInverse)


