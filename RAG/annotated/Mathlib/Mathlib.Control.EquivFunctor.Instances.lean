instance EquivFunctorUnique : EquivFunctor Unique where
  map e := Equiv.uniqueCongr e
                    /-
                      α : Type ?u.3
                      ⊢ Eq ((fun {α β} e => ⇑e.uniqueCongr) (Equiv.refl α)) id
                    -/
  map_refl' α := by simp [eq_iff_true_of_subsingleton]
                    /-
                      🎉 no goals
                    -/
                   /-
                     ⊢ ∀ {α β γ : Type ?u.3} (k : Equiv α β) (h : Equiv β γ), Eq ((fun {α β} e => ⇑ …
                   -/
  map_trans' := by simp [eq_iff_true_of_subsingleton]
                   /-
                     🎉 no goals
                   -/


instance EquivFunctorPerm : EquivFunctor Perm where
  map e p := (e.symm.trans p).trans e
                    /-
                      α : Type ?u.545
                      ⊢ Eq ((fun {α β} e p => (e.symm.trans p).trans e) (Equiv.refl α)) id
                    -/
  map_refl' α := by ext; simp
                         /-
                           🎉 no goals
                         -/
                       /-
                         α✝ β✝ γ✝ : Type ?u.545
                         x✝¹ : Equiv α✝ β✝
                         x✝ : Equiv β✝ γ✝
                         ⊢ Eq ((fun {α β} e p => (e.symm.trans p).trans e) (x✝¹.trans x✝)) (Function.co …
                       -/
  map_trans' _ _ := by ext; simp
                            /-
                              🎉 no goals
                            -/

-- There is a classical instance of `LawfulFunctor Finset` available,
-- but we provide this computable alternative separately.

instance EquivFunctorFinset : EquivFunctor Finset where
  map e s := s.map e.toEmbedding
                    /-
                      α : Type ?u.1564
                      ⊢ Eq ((fun {α β} e s => Finset.map e.toEmbedding s) (Equiv.refl α)) id
                    -/
  map_refl' α := by ext; simp
                         /-
                           🎉 no goals
                         -/
  map_trans' k h := by
    /-
      α✝ β✝ γ✝ : Type ?u.1564
      k : Equiv α✝ β✝
      h : Equiv β✝ γ✝
      ⊢ Eq ((fun {α β} e s => Finset.map e.toEmbedding s) (k.trans h)) (Function.com …
    -/
    ext _ a; simp; constructor <;> intro h'
      /-
        case h.h.mp
        α✝ β✝ γ✝ : Type ?u.1564
        k : Equiv α✝ β✝
        h : Equiv β✝ γ✝
        x✝ : Finset α✝
        a : γ✝
        h' : Exists fun a_1 => And (Membership.mem x✝ a_1) (Eq (h (k a_1)) a)
        ⊢ Membership.mem x✝ (k.symm (h.symm a))
      -/
    · let ⟨a, ha₁, ha₂⟩ := h'
      /-
        case h.h.mp
        α✝ β✝ γ✝ : Type ?u.1564
        k : Equiv α✝ β✝
        h : Equiv β✝ γ✝
        x✝ : Finset α✝
        a✝ : γ✝
        h' : Exists fun a => And (Membership.mem x✝ a) (Eq (h (k a)) a✝)
        a : α✝
        ha₁ : Membership.mem x✝ a
        ha₂ : Eq (h (k a)) a✝
        ⊢ Membership.mem x✝ (k.symm (h.symm a✝))
      -/
      rw [← ha₂]; simp; apply ha₁
                        /-
                          🎉 no goals
                        -/
      /-
        case h.h.mpr
        α✝ β✝ γ✝ : Type ?u.1564
        k : Equiv α✝ β✝
        h : Equiv β✝ γ✝
        x✝ : Finset α✝
        a : γ✝
        h' : Membership.mem x✝ (k.symm (h.symm a))
        ⊢ Exists fun a_1 => And (Membership.mem x✝ a_1) (Eq (h (k a_1)) a)
      -/
    · exists (Equiv.symm k) ((Equiv.symm h) a)
      /-
        case h.h.mpr
        α✝ β✝ γ✝ : Type ?u.1564
        k : Equiv α✝ β✝
        h : Equiv β✝ γ✝
        x✝ : Finset α✝
        a : γ✝
        h' : Membership.mem x✝ (k.symm (h.symm a))
        ⊢ And (Membership.mem x✝ (k.symm (h.symm a))) (Eq (h (k (k.symm (h.symm a)))) a)
      -/
      simp [h']
      /-
        🎉 no goals
      -/


instance EquivFunctorFintype : EquivFunctor Fintype where
  map e _ := Fintype.ofBijective e e.bijective
                    /-
                      α : Type ?u.4116
                      ⊢ Eq ((fun {α β} e x => Fintype.ofBijective ⇑e ⋯) (Equiv.refl α)) id
                    -/
  map_refl' α := by ext; simp [eq_iff_true_of_subsingleton]
                         /-
                           🎉 no goals
                         -/
                   /-
                     ⊢ ∀ {α β γ : Type ?u.4116} (k : Equiv α β) (h : Equiv β γ), Eq ((fun {α β} e x …
                   -/
  map_trans' := by simp [eq_iff_true_of_subsingleton]
                   /-
                     🎉 no goals
                   -/

