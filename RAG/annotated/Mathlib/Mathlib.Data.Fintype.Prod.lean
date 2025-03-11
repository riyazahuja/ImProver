theorem toFinset_prod (s : Set α) (t : Set β) [Fintype s] [Fintype t] [Fintype (s ×ˢ t)] :
    (s ×ˢ t).toFinset = s.toFinset ×ˢ t.toFinset := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype ↑t
    inst✝ : Fintype ↑(SProd.sprod s t)
    ⊢ Eq (SProd.sprod s t).toFinset (SProd.sprod s.toFinset t.toFinset)
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype ↑t
    inst✝ : Fintype ↑(SProd.sprod s t)
    a✝ : Prod α β
    ⊢ Iff (Membership.mem (SProd.sprod s t).toFinset a✝) (Membership.mem (SProd.sp …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem toFinset_off_diag {s : Set α} [DecidableEq α] [Fintype s] [Fintype s.offDiag] :
    s.offDiag.toFinset = s.toFinset.offDiag :=
                   /-
                     α : Type u_1
                     s : Set α
                     inst✝² : DecidableEq α
                     inst✝¹ : Fintype ↑s
                     inst✝ : Fintype ↑s.offDiag
                     ⊢ ∀ (a : Prod α α), Iff (Membership.mem s.offDiag.toFinset a) (Membership.mem  …
                   -/
  Finset.ext <| by simp
                   /-
                     🎉 no goals
                   -/


instance instFintypeProd (α β : Type*) [Fintype α] [Fintype β] : Fintype (α × β) :=
                                  /-
                                    α✝ : Type u_1
                                    β✝ : Type u_2
                                    γ : Type u_3
                                    α : Type u_4
                                    β : Type u_5
                                    inst✝¹ : Fintype α
                                    inst✝ : Fintype β
                                    x✝ : Prod α β
                                    a : α
                                    b : β
                                    ⊢ Membership.mem (SProd.sprod Finset.univ Finset.univ) { fst := a, snd := b }
                                  -/
  ⟨univ ×ˢ univ, fun ⟨a, b⟩ => by simp⟩
                                  /-
                                    🎉 no goals
                                  -/


@[simp] lemma univ_product_univ : univ ×ˢ univ = (univ : Finset (α × β)) := rfl


@[simp] lemma product_eq_univ [Nonempty α] [Nonempty β] : s ×ˢ t = univ ↔ s = univ ∧ t = univ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    inst✝² : Fintype β
    s : Finset α
    t : Finset β
    inst✝¹ : Nonempty α
    inst✝ : Nonempty β
    ⊢ Iff (Eq (SProd.sprod s t) Finset.univ) (And (Eq s Finset.univ) (Eq t Finset. …
  -/
  simp [eq_univ_iff_forall, forall_and]
  /-
    🎉 no goals
  -/


@[simp]
theorem Fintype.card_prod (α β : Type*) [Fintype α] [Fintype β] :
    Fintype.card (α × β) = Fintype.card α * Fintype.card β :=
  card_product _ _


@[simp]
theorem infinite_prod : Infinite (α × β) ↔ Infinite α ∧ Nonempty β ∨ Nonempty α ∧ Infinite β := by
  refine
    ⟨fun H => ?_, fun H =>
      H.elim (and_imp.2 <| @Prod.infinite_of_left α β) (and_imp.2 <| @Prod.infinite_of_right α β)⟩
  /-
    α : Type u_1
    β : Type u_2
    H : Infinite (Prod α β)
    ⊢ Or (And (Infinite α) (Nonempty β)) (And (Nonempty α) (Infinite β))
  -/
  rw [and_comm]; contrapose! H; intro H'
  /-
    α : Type u_1
    β : Type u_2
    H : And (Nonempty β → Not (Infinite α)) (Nonempty α → Not (Infinite β))
    H' : Infinite (Prod α β)
    ⊢ False
  -/
  rcases Infinite.nonempty (α × β) with ⟨a, b⟩
  /-
    case intro.mk
    α : Type u_1
    β : Type u_2
    H : And (Nonempty β → Not (Infinite α)) (Nonempty α → Not (Infinite β))
    H' : Infinite (Prod α β)
    a : α
    b : β
    ⊢ False
  -/
  haveI := fintypeOfNotInfinite (H.1 ⟨b⟩); haveI := fintypeOfNotInfinite (H.2 ⟨a⟩)
  /-
    case intro.mk
    α : Type u_1
    β : Type u_2
    H : And (Nonempty β → Not (Infinite α)) (Nonempty α → Not (Infinite β))
    H' : Infinite (Prod α β)
    a : α
    b : β
    this✝ : Fintype α
    this : Fintype β
    ⊢ False
  -/
  exact H'.false
  /-
    🎉 no goals
  -/


instance Pi.infinite_of_left {ι : Sort*} {π : ι → Type*} [∀ i, Nontrivial <| π i] [Infinite ι] :
    Infinite (∀ i : ι, π i) := by
  classical
  choose m n hm using fun i => exists_pair_ne (π i)
  refine Infinite.of_injective (fun i => update m i (n i)) fun x y h => of_not_not fun hne => ?_
  simp_rw [update_eq_iff, update_of_ne hne] at h
  exact (hm x h.1.symm).elim


/-- If at least one `π i` is infinite and the rest nonempty, the pi type of all `π` is infinite. -/
theorem Pi.infinite_of_exists_right {ι : Sort*} {π : ι → Sort*} (i : ι) [Infinite <| π i]
    [∀ i, Nonempty <| π i] : Infinite (∀ i : ι, π i) := by
  classical
  let ⟨m⟩ := @Pi.instNonempty ι π _
  exact Infinite.of_injective _ (update_injective m i)


/-- See `Pi.infinite_of_exists_right` for the case that only one `π i` is infinite. -/
instance Pi.infinite_of_right {ι : Sort*} {π : ι → Type*} [∀ i, Infinite <| π i] [Nonempty ι] :
    Infinite (∀ i : ι, π i) :=
  Pi.infinite_of_exists_right (Classical.arbitrary ι)


/-- Non-dependent version of `Pi.infinite_of_left`. -/
instance Function.infinite_of_left {ι : Sort*} {π : Type*} [Nontrivial π] [Infinite ι] :
    Infinite (ι → π) :=
  Pi.infinite_of_left


/-- Non-dependent version of `Pi.infinite_of_exists_right` and `Pi.infinite_of_right`. -/
instance Function.infinite_of_right {ι : Sort*} {π : Type*} [Infinite π] [Nonempty ι] :
    Infinite (ι → π) :=
  Pi.infinite_of_right


