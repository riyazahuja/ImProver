/-- Given a self-map `f : α → α`,
`invariants f` is the σ-algebra of measurable sets that are invariant under `f`.

A set `s` is `(invariants f)`-measurable
iff it is meaurable w.r.t. the canonical σ-algebra on `α` and `f ⁻¹' s = s`. -/
def invariants [m : MeasurableSpace α] (f : α → α) : MeasurableSpace α :=
                                 /-
                                   α : Type u_1
                                   m : MeasurableSpace α
                                   f : α → α
                                   ⊢ (fun s => Eq (Set.preimage f s) s) EmptyCollection.emptyCollection
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
                                          /-
                                            🎉 no goals
                                          -/
  { m ⊓ ⟨fun s ↦ f ⁻¹' s = s, by simp, by simp, fun f hf ↦ by simp [hf]⟩ with
                                                              /-
                                                                🎉 no goals
                                                              -/
    MeasurableSet' := fun s ↦ MeasurableSet[m] s ∧ f ⁻¹' s = s }


/-- A set `s` is `(invariants f)`-measurable
iff it is meaurable w.r.t. the canonical σ-algebra on `α` and `f ⁻¹' s = s`. -/
theorem measurableSet_invariants {f : α → α} {s : Set α} :
    MeasurableSet[invariants f] s ↔ MeasurableSet s ∧ f ⁻¹' s = s :=
  .rfl


@[simp]
theorem invariants_id : invariants (id : α → α) = ‹MeasurableSpace α› :=
  ext fun _ ↦ ⟨And.left, fun h ↦ ⟨h, rfl⟩⟩


theorem invariants_le (f : α → α) : invariants f ≤ ‹MeasurableSpace α› := fun _ ↦ And.left


theorem inf_le_invariants_comp (f g : α → α) :
    invariants f ⊓ invariants g ≤ invariants (f ∘ g) := fun s hs ↦
              /-
                α : Type u_1
                inst✝ : MeasurableSpace α
                f g : α → α
                s : Set α
                hs : MeasurableSet s
                ⊢ Eq (Set.preimage (Function.comp f g) s) s
              -/
  ⟨hs.1.1, by rw [preimage_comp, hs.1.2, hs.2.2]⟩
              /-
                🎉 no goals
              -/


theorem le_invariants_iterate (f : α → α) (n : ℕ) :
    invariants f ≤ invariants (f^[n]) := by
  induction n with
  | zero => simp [invariants_le]
  | succ n ihn => exact le_trans (le_inf ihn le_rfl) (inf_le_invariants_comp _ _)


theorem measurable_invariants_dom {f : α → α} {g : α → β} :
    Measurable[invariants f] g ↔ Measurable g ∧ ∀ s, MeasurableSet s → (g ∘ f) ⁻¹' s = g ⁻¹' s := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    β : Type u_2
    inst✝ : MeasurableSpace β
    f : α → α
    g : α → β
    ⊢ Iff (Measurable g) (And (Measurable g) (∀ (s : Set β), MeasurableSet s → Eq  …
  -/
  simp only [Measurable, ← forall_and]; rfl
                                        /-
                                          🎉 no goals
                                        -/


theorem measurable_invariants_of_semiconj {fa : α → α} {fb : β → β} {g : α → β} (hg : Measurable g)
    (hfg : Semiconj g fa fb) : @Measurable _ _ (invariants fa) (invariants fb) g := fun s hs ↦
               /-
                 α : Type u_1
                 inst✝¹ : MeasurableSpace α
                 β : Type u_2
                 inst✝ : MeasurableSpace β
                 fa : α → α
                 fb : β → β
                 g : α → β
                 hg : Measurable g
                 hfg : Function.Semiconj g fa fb
                 s : Set β
                 hs : MeasurableSet s
                 ⊢ Eq (Set.preimage fa (Set.preimage g s)) (Set.preimage g s)
               -/
  ⟨hg hs.1, by rw [← preimage_comp, hfg.comp_eq, preimage_comp, hs.2]⟩
               /-
                 🎉 no goals
               -/


