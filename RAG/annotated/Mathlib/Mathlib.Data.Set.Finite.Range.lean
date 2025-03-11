instance fintypeRange [DecidableEq α] (f : ι → α) [Fintype (PLift ι)] : Fintype (range f) :=
                                                               /-
                                                                 α : Type u
                                                                 β : Type v
                                                                 ι : Sort w
                                                                 γ : Type x
                                                                 inst✝¹ : DecidableEq α
                                                                 f : ι → α
                                                                 inst✝ : Fintype (PLift ι)
                                                                 ⊢ ∀ (x : α), Iff (Membership.mem (Finset.image (Function.comp f PLift.down) Fi …
                                                               -/
  Fintype.ofFinset (Finset.univ.image <| f ∘ PLift.down) <| by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


instance finite_range (f : ι → α) [Finite ι] : Finite (range f) := by
  classical
  haveI := Fintype.ofFinite (PLift ι)
  infer_instance


instance finite_replacement [Finite α] (f : α → β) :
    Finite {f x | x : α} :=
  Finite.Set.finite_range f


theorem finite_range (f : ι → α) [Finite ι] : (range f).Finite :=
  toFinite _


theorem Finite.dependent_image {s : Set α} (hs : s.Finite) (F : ∀ i ∈ s, β) :
    {y : β | ∃ x hx, F x hx = y}.Finite := by
  /-
    α : Type u
    β : Type v
    s : Set α
    hs : s.Finite
    F : (i : α) → Membership.mem s i → β
    ⊢ (setOf fun y => Exists fun x => Exists fun hx => Eq (F x hx) y).Finite
  -/
  have := hs.to_subtype
  /-
    α : Type u
    β : Type v
    s : Set α
    hs : s.Finite
    F : (i : α) → Membership.mem s i → β
    this : Finite ↑s
    ⊢ (setOf fun y => Exists fun x => Exists fun hx => Eq (F x hx) y).Finite
  -/
  simpa [range] using finite_range fun x : s => F x x.2
  /-
    🎉 no goals
  -/


