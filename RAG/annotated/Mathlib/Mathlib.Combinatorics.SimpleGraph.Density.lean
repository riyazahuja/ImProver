/-- Finset of edges of a relation between two finsets of vertices. -/
def interedges (s : Finset α) (t : Finset β) : Finset (α × β) := {e ∈ s ×ˢ t | r e.1 e.2}


/-- Edge density of a relation between two finsets of vertices. -/
def edgeDensity (s : Finset α) (t : Finset β) : ℚ := #(interedges r s t) / (#s * #t)


theorem mem_interedges_iff {x : α × β} : x ∈ interedges r s t ↔ x.1 ∈ s ∧ x.2 ∈ t ∧ r x.1 x.2 := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s : Finset α
    t : Finset β
    x : Prod α β
    ⊢ Iff (Membership.mem (Rel.interedges r s t) x) (And (Membership.mem s x.1) (A …
  -/
  rw [interedges, mem_filter, Finset.mem_product, and_assoc]
  /-
    🎉 no goals
  -/


theorem mk_mem_interedges_iff : (a, b) ∈ interedges r s t ↔ a ∈ s ∧ b ∈ t ∧ r a b :=
  mem_interedges_iff


@[simp]
theorem interedges_empty_left (t : Finset β) : interedges r ∅ t = ∅ := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    t : Finset β
    ⊢ Eq (Rel.interedges r EmptyCollection.emptyCollection t) EmptyCollection.empt …
  -/
  rw [interedges, Finset.empty_product, filter_empty]
  /-
    🎉 no goals
  -/


theorem interedges_mono (hs : s₂ ⊆ s₁) (ht : t₂ ⊆ t₁) : interedges r s₂ t₂ ⊆ interedges r s₁ t₁ :=
  fun x ↦ by
    /-
      α : Type u_4
      β : Type u_5
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      hs : HasSubset.Subset s₂ s₁
      ht : HasSubset.Subset t₂ t₁
      x : Prod α β
      ⊢ Membership.mem (Rel.interedges r s₂ t₂) x → Membership.mem (Rel.interedges r …
    -/
    simp_rw [mem_interedges_iff]
    /-
      α : Type u_4
      β : Type u_5
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      hs : HasSubset.Subset s₂ s₁
      ht : HasSubset.Subset t₂ t₁
      x : Prod α β
      ⊢ And (Membership.mem s₂ x.1) (And (Membership.mem t₂ x.2) (r x.1 x.2)) → And  …
    -/
    exact fun h ↦ ⟨hs h.1, ht h.2.1, h.2.2⟩
    /-
      🎉 no goals
    -/


theorem card_interedges_add_card_interedges_compl (s : Finset α) (t : Finset β) :
    #(interedges r s t) + #(interedges (fun x y ↦ ¬r x y) s t) = #s * #t := by
  classical
  rw [← card_product, interedges, interedges, ← card_union_of_disjoint, filter_union_filter_neg_eq]
  exact disjoint_filter.2 fun _ _ ↦ Classical.not_not.2


theorem interedges_disjoint_left {s s' : Finset α} (hs : Disjoint s s') (t : Finset β) :
    Disjoint (interedges r s t) (interedges r s' t) := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s s' : Finset α
    hs : Disjoint s s'
    t : Finset β
    ⊢ Disjoint (Rel.interedges r s t) (Rel.interedges r s' t)
  -/
  rw [Finset.disjoint_left] at hs ⊢
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s s' : Finset α
    hs : ∀ ⦃a : α⦄, Membership.mem s a → Not (Membership.mem s' a)
    t : Finset β
    ⊢ ∀ ⦃a : Prod α β⦄, Membership.mem (Rel.interedges r s t) a → Not (Membership. …
  -/
  intro _ hx hy
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s s' : Finset α
    hs : ∀ ⦃a : α⦄, Membership.mem s a → Not (Membership.mem s' a)
    t : Finset β
    a✝ : Prod α β
    hx : Membership.mem (Rel.interedges r s t) a✝
    hy : Membership.mem (Rel.interedges r s' t) a✝
    ⊢ False
  -/
  rw [mem_interedges_iff] at hx hy
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s s' : Finset α
    hs : ∀ ⦃a : α⦄, Membership.mem s a → Not (Membership.mem s' a)
    t : Finset β
    a✝ : Prod α β
    hx : And (Membership.mem s a✝.1) (And (Membership.mem t a✝.2) (r a✝.1 a✝.2))
    hy : And (Membership.mem s' a✝.1) (And (Membership.mem t a✝.2) (r a✝.1 a✝.2))
    ⊢ False
  -/
  exact hs hx.1 hy.1
  /-
    🎉 no goals
  -/


theorem interedges_disjoint_right (s : Finset α) {t t' : Finset β} (ht : Disjoint t t') :
    Disjoint (interedges r s t) (interedges r s t') := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s : Finset α
    t t' : Finset β
    ht : Disjoint t t'
    ⊢ Disjoint (Rel.interedges r s t) (Rel.interedges r s t')
  -/
  rw [Finset.disjoint_left] at ht ⊢
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s : Finset α
    t t' : Finset β
    ht : ∀ ⦃a : β⦄, Membership.mem t a → Not (Membership.mem t' a)
    ⊢ ∀ ⦃a : Prod α β⦄, Membership.mem (Rel.interedges r s t) a → Not (Membership. …
  -/
  intro _ hx hy
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s : Finset α
    t t' : Finset β
    ht : ∀ ⦃a : β⦄, Membership.mem t a → Not (Membership.mem t' a)
    a✝ : Prod α β
    hx : Membership.mem (Rel.interedges r s t) a✝
    hy : Membership.mem (Rel.interedges r s t') a✝
    ⊢ False
  -/
  rw [mem_interedges_iff] at hx hy
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s : Finset α
    t t' : Finset β
    ht : ∀ ⦃a : β⦄, Membership.mem t a → Not (Membership.mem t' a)
    a✝ : Prod α β
    hx : And (Membership.mem s a✝.1) (And (Membership.mem t a✝.2) (r a✝.1 a✝.2))
    hy : And (Membership.mem s a✝.1) (And (Membership.mem t' a✝.2) (r a✝.1 a✝.2))
    ⊢ False
  -/
  exact ht hx.2.1 hy.2.1
  /-
    🎉 no goals
  -/


lemma interedges_eq_biUnion :
    interedges r s t = s.biUnion fun x ↦ {y ∈ t | r x y}.map ⟨(x, ·), Prod.mk.inj_left x⟩ := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝² : (a : α) → DecidablePred (r a)
    s : Finset α
    t : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    ⊢ Eq (Rel.interedges r s t) (s.biUnion fun x => Finset.map { toFun := fun x_1  …
  -/
  ext ⟨x, y⟩; simp [mem_interedges_iff]
              /-
                🎉 no goals
              -/


theorem interedges_biUnion_left (s : Finset ι) (t : Finset β) (f : ι → Finset α) :
    interedges r (s.biUnion f) t = s.biUnion fun a ↦ interedges r (f a) t := by
  /-
    ι : Type u_2
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝² : (a : α) → DecidablePred (r a)
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Finset ι
    t : Finset β
    f : ι → Finset α
    ⊢ Eq (Rel.interedges r (s.biUnion f) t) (s.biUnion fun a => Rel.interedges r ( …
  -/
  ext
  /-
    case h
    ι : Type u_2
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝² : (a : α) → DecidablePred (r a)
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Finset ι
    t : Finset β
    f : ι → Finset α
    a✝ : Prod α β
    ⊢ Iff (Membership.mem (Rel.interedges r (s.biUnion f) t) a✝) (Membership.mem ( …
  -/
  simp only [mem_biUnion, mem_interedges_iff, exists_and_right, ← and_assoc]
  /-
    🎉 no goals
  -/


theorem interedges_biUnion_right (s : Finset α) (t : Finset ι) (f : ι → Finset β) :
    interedges r s (t.biUnion f) = t.biUnion fun b ↦ interedges r s (f b) := by
  /-
    ι : Type u_2
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝² : (a : α) → DecidablePred (r a)
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Finset α
    t : Finset ι
    f : ι → Finset β
    ⊢ Eq (Rel.interedges r s (t.biUnion f)) (t.biUnion fun b => Rel.interedges r s …
  -/
  ext a
  /-
    case h
    ι : Type u_2
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝² : (a : α) → DecidablePred (r a)
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Finset α
    t : Finset ι
    f : ι → Finset β
    a : Prod α β
    ⊢ Iff (Membership.mem (Rel.interedges r s (t.biUnion f)) a) (Membership.mem (t …
  -/
  simp only [mem_interedges_iff, mem_biUnion]
  exact ⟨fun ⟨x₁, ⟨x₂, x₃, x₄⟩, x₅⟩ ↦ ⟨x₂, x₃, x₁, x₄, x₅⟩,
    fun ⟨x₂, x₃, x₁, x₄, x₅⟩ ↦ ⟨x₁, ⟨x₂, x₃, x₄⟩, x₅⟩⟩


theorem interedges_biUnion (s : Finset ι) (t : Finset κ) (f : ι → Finset α) (g : κ → Finset β) :
    interedges r (s.biUnion f) (t.biUnion g) =
      (s ×ˢ t).biUnion fun ab ↦ interedges r (f ab.1) (g ab.2) := by
  /-
    ι : Type u_2
    κ : Type u_3
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝² : (a : α) → DecidablePred (r a)
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    s : Finset ι
    t : Finset κ
    f : ι → Finset α
    g : κ → Finset β
    ⊢ Eq (Rel.interedges r (s.biUnion f) (t.biUnion g)) ((SProd.sprod s t).biUnion …
  -/
  simp_rw [product_biUnion, interedges_biUnion_left, interedges_biUnion_right]
  /-
    🎉 no goals
  -/


theorem card_interedges_le_mul (s : Finset α) (t : Finset β) :
    #(interedges r s t) ≤ #s * #t :=
  (card_filter_le _ _).trans (card_product _ _).le


theorem edgeDensity_nonneg (s : Finset α) (t : Finset β) : 0 ≤ edgeDensity r s t := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s : Finset α
    t : Finset β
    ⊢ LE.le 0 (Rel.edgeDensity r s t)
  -/
                       /-
                         🎉 no goals
                       -/
  apply div_nonneg <;> exact mod_cast Nat.zero_le _
                       /-
                         🎉 no goals
                       -/


theorem edgeDensity_le_one (s : Finset α) (t : Finset β) : edgeDensity r s t ≤ 1 := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s : Finset α
    t : Finset β
    ⊢ LE.le (Rel.edgeDensity r s t) 1
  -/
  apply div_le_one_of_le₀
    /-
      case h
      α : Type u_4
      β : Type u_5
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s : Finset α
      t : Finset β
      ⊢ LE.le (↑(Rel.interedges r s t).card) (HMul.hMul ↑s.card ↑t.card)
    -/
  · exact mod_cast card_interedges_le_mul r s t
    /-
      🎉 no goals
    -/
    /-
      case hb
      α : Type u_4
      β : Type u_5
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s : Finset α
      t : Finset β
      ⊢ LE.le 0 (HMul.hMul ↑s.card ↑t.card)
    -/
  · exact mod_cast Nat.zero_le _
    /-
      🎉 no goals
    -/


theorem edgeDensity_add_edgeDensity_compl (hs : s.Nonempty) (ht : t.Nonempty) :
    edgeDensity r s t + edgeDensity (fun x y ↦ ¬r x y) s t = 1 := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s : Finset α
    t : Finset β
    hs : s.Nonempty
    ht : t.Nonempty
    ⊢ Eq (HAdd.hAdd (Rel.edgeDensity r s t) (Rel.edgeDensity (fun x y => Not (r x  …
  -/
  rw [edgeDensity, edgeDensity, div_add_div_same, div_eq_one_iff_eq]
    /-
      α : Type u_4
      β : Type u_5
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s : Finset α
      t : Finset β
      hs : s.Nonempty
      ht : t.Nonempty
      ⊢ Eq (HAdd.hAdd ↑(Rel.interedges r s t).card ↑(Rel.interedges (fun x y => Not  …
    -/
  · exact mod_cast card_interedges_add_card_interedges_compl r s t
    /-
      🎉 no goals
    -/
    /-
      α : Type u_4
      β : Type u_5
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s : Finset α
      t : Finset β
      hs : s.Nonempty
      ht : t.Nonempty
      ⊢ Ne (HMul.hMul ↑s.card ↑t.card) 0
    -/
  · exact mod_cast (mul_pos hs.card_pos ht.card_pos).ne'
    /-
      🎉 no goals
    -/


@[simp]
theorem edgeDensity_empty_left (t : Finset β) : edgeDensity r ∅ t = 0 := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    t : Finset β
    ⊢ Eq (Rel.edgeDensity r EmptyCollection.emptyCollection t) 0
  -/
  rw [edgeDensity, Finset.card_empty, Nat.cast_zero, zero_mul, div_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem edgeDensity_empty_right (s : Finset α) : edgeDensity r s ∅ = 0 := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s : Finset α
    ⊢ Eq (Rel.edgeDensity r s EmptyCollection.emptyCollection) 0
  -/
  rw [edgeDensity, Finset.card_empty, Nat.cast_zero, mul_zero, div_zero]
  /-
    🎉 no goals
  -/


theorem card_interedges_finpartition_left [DecidableEq α] (P : Finpartition s) (t : Finset β) :
    #(interedges r s t) = ∑ a ∈ P.parts, #(interedges r a t) := by
  classical
  simp_rw [← P.biUnion_parts, interedges_biUnion_left, id]
  rw [card_biUnion]
  exact fun x hx y hy h ↦ interedges_disjoint_left r (P.disjoint hx hy h) _


theorem card_interedges_finpartition_right [DecidableEq β] (s : Finset α) (P : Finpartition t) :
    #(interedges r s t) = ∑ b ∈ P.parts, #(interedges r s b) := by
  classical
  simp_rw [← P.biUnion_parts, interedges_biUnion_right, id]
  rw [card_biUnion]
  exact fun x hx y hy h ↦ interedges_disjoint_right r _ (P.disjoint hx hy h)


theorem card_interedges_finpartition [DecidableEq α] [DecidableEq β] (P : Finpartition s)
    (Q : Finpartition t) :
    #(interedges r s t) = ∑ ab ∈ P.parts ×ˢ Q.parts, #(interedges r ab.1 ab.2) := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝² : (a : α) → DecidablePred (r a)
    s : Finset α
    t : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    P : Finpartition s
    Q : Finpartition t
    ⊢ Eq (Rel.interedges r s t).card ((SProd.sprod P.parts Q.parts).sum fun ab =>  …
  -/
  rw [card_interedges_finpartition_left _ P, sum_product]
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝² : (a : α) → DecidablePred (r a)
    s : Finset α
    t : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    P : Finpartition s
    Q : Finpartition t
    ⊢ Eq (P.parts.sum fun a => (Rel.interedges r a t).card) (P.parts.sum fun x =>  …
  -/
  congr; ext
  /-
    case e_f.h
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝² : (a : α) → DecidablePred (r a)
    s : Finset α
    t : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    P : Finpartition s
    Q : Finpartition t
    x✝ : Finset α
    ⊢ Eq (Rel.interedges r x✝ t).card (Q.parts.sum fun y => (Rel.interedges r { fs …
  -/
  rw [card_interedges_finpartition_right]
  /-
    🎉 no goals
  -/


theorem mul_edgeDensity_le_edgeDensity (hs : s₂ ⊆ s₁) (ht : t₂ ⊆ t₁) (hs₂ : s₂.Nonempty)
    (ht₂ : t₂.Nonempty) :
    (#s₂ : ℚ) / #s₁ * (#t₂ / #t₁) * edgeDensity r s₂ t₂ ≤ edgeDensity r s₁ t₁ := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hs₂ : s₂.Nonempty
    ht₂ : t₂.Nonempty
    ⊢ LE.le (HMul.hMul (HMul.hMul (HDiv.hDiv ↑s₂.card ↑s₁.card) (HDiv.hDiv ↑t₂.car …
  -/
  have hst : (#s₂ : ℚ) * #t₂ ≠ 0 := by simp [hs₂.ne_empty, ht₂.ne_empty]
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hs₂ : s₂.Nonempty
    ht₂ : t₂.Nonempty
    hst : Ne (HMul.hMul ↑s₂.card ↑t₂.card) 0
    ⊢ LE.le (HMul.hMul (HMul.hMul (HDiv.hDiv ↑s₂.card ↑s₁.card) (HDiv.hDiv ↑t₂.car …
  -/
  rw [edgeDensity, edgeDensity, div_mul_div_comm, mul_comm, div_mul_div_cancel₀ hst]
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hs₂ : s₂.Nonempty
    ht₂ : t₂.Nonempty
    hst : Ne (HMul.hMul ↑s₂.card ↑t₂.card) 0
    ⊢ LE.le (HDiv.hDiv (↑(Rel.interedges r s₂ t₂).card) (HMul.hMul ↑s₁.card ↑t₁.ca …
  -/
  gcongr
  /-
    case hab.h.a
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hs₂ : s₂.Nonempty
    ht₂ : t₂.Nonempty
    hst : Ne (HMul.hMul ↑s₂.card ↑t₂.card) 0
    ⊢ HasSubset.Subset (Rel.interedges r s₂ t₂) (Rel.interedges r s₁ t₁)
  -/
  exact interedges_mono hs ht
  /-
    🎉 no goals
  -/


theorem edgeDensity_sub_edgeDensity_le_one_sub_mul (hs : s₂ ⊆ s₁) (ht : t₂ ⊆ t₁) (hs₂ : s₂.Nonempty)
    (ht₂ : t₂.Nonempty) :
    edgeDensity r s₂ t₂ - edgeDensity r s₁ t₁ ≤ 1 - #s₂ / #s₁ * (#t₂ / #t₁) := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hs₂ : s₂.Nonempty
    ht₂ : t₂.Nonempty
    ⊢ LE.le (HSub.hSub (Rel.edgeDensity r s₂ t₂) (Rel.edgeDensity r s₁ t₁)) (HSub. …
  -/
  refine (sub_le_sub_left (mul_edgeDensity_le_edgeDensity r hs ht hs₂ ht₂) _).trans ?_
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hs₂ : s₂.Nonempty
    ht₂ : t₂.Nonempty
    ⊢ LE.le (HSub.hSub (Rel.edgeDensity r s₂ t₂) (HMul.hMul (HMul.hMul (HDiv.hDiv  …
  -/
  refine le_trans ?_ (mul_le_of_le_one_right ?_ (edgeDensity_le_one r s₂ t₂))
    /-
      case refine_1
      α : Type u_4
      β : Type u_5
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      hs : HasSubset.Subset s₂ s₁
      ht : HasSubset.Subset t₂ t₁
      hs₂ : s₂.Nonempty
      ht₂ : t₂.Nonempty
      ⊢ LE.le (HSub.hSub (Rel.edgeDensity r s₂ t₂) (HMul.hMul (HMul.hMul (HDiv.hDiv  …
    -/
  · rw [sub_mul, one_mul]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hs₂ : s₂.Nonempty
    ht₂ : t₂.Nonempty
    ⊢ LE.le 0 (HSub.hSub 1 (HMul.hMul (HDiv.hDiv ↑s₂.card ↑s₁.card) (HDiv.hDiv ↑t₂ …
  -/
  refine sub_nonneg_of_le (mul_le_one₀ ?_ ?_ ?_)
    /-
      case refine_2.refine_1
      α : Type u_4
      β : Type u_5
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      hs : HasSubset.Subset s₂ s₁
      ht : HasSubset.Subset t₂ t₁
      hs₂ : s₂.Nonempty
      ht₂ : t₂.Nonempty
      ⊢ LE.le (HDiv.hDiv ↑s₂.card ↑s₁.card) 1
    -/
  · exact div_le_one_of_le₀ ((@Nat.cast_le ℚ).2 (card_le_card hs)) (Nat.cast_nonneg _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2.refine_2
      α : Type u_4
      β : Type u_5
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      hs : HasSubset.Subset s₂ s₁
      ht : HasSubset.Subset t₂ t₁
      hs₂ : s₂.Nonempty
      ht₂ : t₂.Nonempty
      ⊢ LE.le 0 (HDiv.hDiv ↑t₂.card ↑t₁.card)
    -/
                         /-
                           🎉 no goals
                         -/
  · apply div_nonneg <;> exact mod_cast Nat.zero_le _
                         /-
                           🎉 no goals
                         -/
    /-
      case refine_2.refine_3
      α : Type u_4
      β : Type u_5
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      hs : HasSubset.Subset s₂ s₁
      ht : HasSubset.Subset t₂ t₁
      hs₂ : s₂.Nonempty
      ht₂ : t₂.Nonempty
      ⊢ LE.le (HDiv.hDiv ↑t₂.card ↑t₁.card) 1
    -/
  · exact div_le_one_of_le₀ ((@Nat.cast_le ℚ).2 (card_le_card ht)) (Nat.cast_nonneg _)
    /-
      🎉 no goals
    -/


theorem abs_edgeDensity_sub_edgeDensity_le_one_sub_mul (hs : s₂ ⊆ s₁) (ht : t₂ ⊆ t₁)
    (hs₂ : s₂.Nonempty) (ht₂ : t₂.Nonempty) :
    |edgeDensity r s₂ t₂ - edgeDensity r s₁ t₁| ≤ 1 - #s₂ / #s₁ * (#t₂ / #t₁) := by
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hs₂ : s₂.Nonempty
    ht₂ : t₂.Nonempty
    ⊢ LE.le (abs (HSub.hSub (Rel.edgeDensity r s₂ t₂) (Rel.edgeDensity r s₁ t₁)))  …
  -/
  refine abs_sub_le_iff.2 ⟨edgeDensity_sub_edgeDensity_le_one_sub_mul r hs ht hs₂ ht₂, ?_⟩
  rw [← add_sub_cancel_right (edgeDensity r s₁ t₁) (edgeDensity (fun x y ↦ ¬r x y) s₁ t₁),
    ← add_sub_cancel_right (edgeDensity r s₂ t₂) (edgeDensity (fun x y ↦ ¬r x y) s₂ t₂),
    edgeDensity_add_edgeDensity_compl _ (hs₂.mono hs) (ht₂.mono ht),
    edgeDensity_add_edgeDensity_compl _ hs₂ ht₂, sub_sub_sub_cancel_left]
  /-
    α : Type u_4
    β : Type u_5
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hs₂ : s₂.Nonempty
    ht₂ : t₂.Nonempty
    ⊢ LE.le (HSub.hSub (Rel.edgeDensity (fun x y => Not (r x y)) s₂ t₂) (Rel.edgeD …
  -/
  exact edgeDensity_sub_edgeDensity_le_one_sub_mul _ hs ht hs₂ ht₂
  /-
    🎉 no goals
  -/


theorem abs_edgeDensity_sub_edgeDensity_le_two_mul_sub_sq (hs : s₂ ⊆ s₁) (ht : t₂ ⊆ t₁)
    (hδ₀ : 0 ≤ δ) (hδ₁ : δ < 1) (hs₂ : (1 - δ) * #s₁ ≤ #s₂)
    (ht₂ : (1 - δ) * #t₁ ≤ #t₂) :
    |(edgeDensity r s₂ t₂ : 𝕜) - edgeDensity r s₁ t₁| ≤ 2 * δ - δ ^ 2 := by
  have hδ' : 0 ≤ 2 * δ - δ ^ 2 := by
    rw [sub_nonneg, sq]
    gcongr
    exact hδ₁.le.trans (by norm_num)
  /-
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ₀ : LE.le 0 δ
    hδ₁ : LT.lt δ 1
    hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
    ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r s₂ t₂) ↑(Rel.edgeDensity r s₁ t₁)) …
  -/
  rw [← sub_pos] at hδ₁
  /-
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ₀ : LE.le 0 δ
    hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
    hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
    ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r s₂ t₂) ↑(Rel.edgeDensity r s₁ t₁)) …
  -/
  obtain rfl | hs₂' := s₂.eq_empty_or_nonempty
    /-
      case inl
      𝕜 : Type u_1
      α : Type u_4
      β : Type u_5
      inst✝¹ : LinearOrderedField 𝕜
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ : Finset α
      t₁ t₂ : Finset β
      δ : 𝕜
      ht : HasSubset.Subset t₂ t₁
      hδ₀ : LE.le 0 δ
      hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
      ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
      hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
      hs : HasSubset.Subset EmptyCollection.emptyCollection s₁
      hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑EmptyCollection.emptyCollect …
      ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r EmptyCollection.emptyCollection t₂ …
    -/
  · rw [Finset.card_empty, Nat.cast_zero] at hs₂
    /-
      case inl
      𝕜 : Type u_1
      α : Type u_4
      β : Type u_5
      inst✝¹ : LinearOrderedField 𝕜
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ : Finset α
      t₁ t₂ : Finset β
      δ : 𝕜
      ht : HasSubset.Subset t₂ t₁
      hδ₀ : LE.le 0 δ
      hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
      ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
      hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
      hs : HasSubset.Subset EmptyCollection.emptyCollection s₁
      hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) 0
      ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r EmptyCollection.emptyCollection t₂ …
    -/
    simpa [edgeDensity, (nonpos_of_mul_nonpos_right hs₂ hδ₁).antisymm (Nat.cast_nonneg _)] using hδ'
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ₀ : LE.le 0 δ
    hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
    hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
    hs₂' : s₂.Nonempty
    ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r s₂ t₂) ↑(Rel.edgeDensity r s₁ t₁)) …
  -/
  obtain rfl | ht₂' := t₂.eq_empty_or_nonempty
    /-
      case inr.inl
      𝕜 : Type u_1
      α : Type u_4
      β : Type u_5
      inst✝¹ : LinearOrderedField 𝕜
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ : Finset β
      δ : 𝕜
      hs : HasSubset.Subset s₂ s₁
      hδ₀ : LE.le 0 δ
      hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
      hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
      hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
      hs₂' : s₂.Nonempty
      ht : HasSubset.Subset EmptyCollection.emptyCollection t₁
      ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑EmptyCollection.emptyCollect …
      ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r s₂ EmptyCollection.emptyCollection …
    -/
  · rw [Finset.card_empty, Nat.cast_zero] at ht₂
    /-
      case inr.inl
      𝕜 : Type u_1
      α : Type u_4
      β : Type u_5
      inst✝¹ : LinearOrderedField 𝕜
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ : Finset β
      δ : 𝕜
      hs : HasSubset.Subset s₂ s₁
      hδ₀ : LE.le 0 δ
      hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
      hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
      hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
      hs₂' : s₂.Nonempty
      ht : HasSubset.Subset EmptyCollection.emptyCollection t₁
      ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) 0
      ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r s₂ EmptyCollection.emptyCollection …
    -/
    simpa [edgeDensity, (nonpos_of_mul_nonpos_right ht₂ hδ₁).antisymm (Nat.cast_nonneg _)] using hδ'
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ₀ : LE.le 0 δ
    hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
    hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
    hs₂' : s₂.Nonempty
    ht₂' : t₂.Nonempty
    ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r s₂ t₂) ↑(Rel.edgeDensity r s₁ t₁)) …
  -/
  have hr : 2 * δ - δ ^ 2 = 1 - (1 - δ) * (1 - δ) := by ring
  /-
    case inr.inr
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ₀ : LE.le 0 δ
    hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
    hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
    hs₂' : s₂.Nonempty
    ht₂' : t₂.Nonempty
    hr : Eq (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2)) (HSub.hSub 1 (HMul.hMul (H …
    ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r s₂ t₂) ↑(Rel.edgeDensity r s₁ t₁)) …
  -/
  rw [hr]
  /-
    case inr.inr
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ₀ : LE.le 0 δ
    hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
    hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
    hs₂' : s₂.Nonempty
    ht₂' : t₂.Nonempty
    hr : Eq (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2)) (HSub.hSub 1 (HMul.hMul (H …
    ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r s₂ t₂) ↑(Rel.edgeDensity r s₁ t₁)) …
  -/
  norm_cast
  refine
    (Rat.cast_le.2 <| abs_edgeDensity_sub_edgeDensity_le_one_sub_mul r hs ht hs₂' ht₂').trans ?_
  /-
    case inr.inr
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ₀ : LE.le 0 δ
    hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
    hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
    hs₂' : s₂.Nonempty
    ht₂' : t₂.Nonempty
    hr : Eq (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2)) (HSub.hSub 1 (HMul.hMul (H …
    ⊢ LE.le (↑(HSub.hSub 1 (HMul.hMul (HDiv.hDiv ↑s₂.card ↑s₁.card) (HDiv.hDiv ↑t₂ …
  -/
  push_cast
  /-
    case inr.inr
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ₀ : LE.le 0 δ
    hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
    hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
    hs₂' : s₂.Nonempty
    ht₂' : t₂.Nonempty
    hr : Eq (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2)) (HSub.hSub 1 (HMul.hMul (H …
    ⊢ LE.le (HSub.hSub 1 (HMul.hMul (HDiv.hDiv ↑s₂.card ↑s₁.card) (HDiv.hDiv ↑t₂.c …
  -/
  have h₁ := hs₂'.mono hs
  /-
    case inr.inr
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ₀ : LE.le 0 δ
    hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
    hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
    hs₂' : s₂.Nonempty
    ht₂' : t₂.Nonempty
    hr : Eq (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2)) (HSub.hSub 1 (HMul.hMul (H …
    h₁ : s₁.Nonempty
    ⊢ LE.le (HSub.hSub 1 (HMul.hMul (HDiv.hDiv ↑s₂.card ↑s₁.card) (HDiv.hDiv ↑t₂.c …
  -/
  have h₂ := ht₂'.mono ht
  /-
    case inr.inr
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ₀ : LE.le 0 δ
    hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
    hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
    hs₂' : s₂.Nonempty
    ht₂' : t₂.Nonempty
    hr : Eq (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2)) (HSub.hSub 1 (HMul.hMul (H …
    h₁ : s₁.Nonempty
    h₂ : t₁.Nonempty
    ⊢ LE.le (HSub.hSub 1 (HMul.hMul (HDiv.hDiv ↑s₂.card ↑s₁.card) (HDiv.hDiv ↑t₂.c …
  -/
  gcongr
    /-
      case inr.inr.h.h₁
      𝕜 : Type u_1
      α : Type u_4
      β : Type u_5
      inst✝¹ : LinearOrderedField 𝕜
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      δ : 𝕜
      hs : HasSubset.Subset s₂ s₁
      ht : HasSubset.Subset t₂ t₁
      hδ₀ : LE.le 0 δ
      hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
      hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
      ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
      hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
      hs₂' : s₂.Nonempty
      ht₂' : t₂.Nonempty
      hr : Eq (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2)) (HSub.hSub 1 (HMul.hMul (H …
      h₁ : s₁.Nonempty
      h₂ : t₁.Nonempty
      ⊢ LE.le (HSub.hSub 1 δ) (HDiv.hDiv ↑s₂.card ↑s₁.card)
    -/
  · refine (le_div_iff₀ ?_).2 hs₂
    /-
      case inr.inr.h.h₁
      𝕜 : Type u_1
      α : Type u_4
      β : Type u_5
      inst✝¹ : LinearOrderedField 𝕜
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      δ : 𝕜
      hs : HasSubset.Subset s₂ s₁
      ht : HasSubset.Subset t₂ t₁
      hδ₀ : LE.le 0 δ
      hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
      hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
      ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
      hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
      hs₂' : s₂.Nonempty
      ht₂' : t₂.Nonempty
      hr : Eq (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2)) (HSub.hSub 1 (HMul.hMul (H …
      h₁ : s₁.Nonempty
      h₂ : t₁.Nonempty
      ⊢ LT.lt 0 ↑s₁.card
    -/
    exact mod_cast h₁.card_pos
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.h.h₂
      𝕜 : Type u_1
      α : Type u_4
      β : Type u_5
      inst✝¹ : LinearOrderedField 𝕜
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      δ : 𝕜
      hs : HasSubset.Subset s₂ s₁
      ht : HasSubset.Subset t₂ t₁
      hδ₀ : LE.le 0 δ
      hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
      hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
      ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
      hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
      hs₂' : s₂.Nonempty
      ht₂' : t₂.Nonempty
      hr : Eq (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2)) (HSub.hSub 1 (HMul.hMul (H …
      h₁ : s₁.Nonempty
      h₂ : t₁.Nonempty
      ⊢ LE.le (HSub.hSub 1 δ) (HDiv.hDiv ↑t₂.card ↑t₁.card)
    -/
  · refine (le_div_iff₀ ?_).2 ht₂
    /-
      case inr.inr.h.h₂
      𝕜 : Type u_1
      α : Type u_4
      β : Type u_5
      inst✝¹ : LinearOrderedField 𝕜
      r : α → β → Prop
      inst✝ : (a : α) → DecidablePred (r a)
      s₁ s₂ : Finset α
      t₁ t₂ : Finset β
      δ : 𝕜
      hs : HasSubset.Subset s₂ s₁
      ht : HasSubset.Subset t₂ t₁
      hδ₀ : LE.le 0 δ
      hδ₁ : LT.lt 0 (HSub.hSub 1 δ)
      hs₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
      ht₂ : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
      hδ' : LE.le 0 (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2))
      hs₂' : s₂.Nonempty
      ht₂' : t₂.Nonempty
      hr : Eq (HSub.hSub (HMul.hMul 2 δ) (HPow.hPow δ 2)) (HSub.hSub 1 (HMul.hMul (H …
      h₁ : s₁.Nonempty
      h₂ : t₁.Nonempty
      ⊢ LT.lt 0 ↑t₁.card
    -/
    exact mod_cast h₂.card_pos
    /-
      🎉 no goals
    -/


/-- If `s₂ ⊆ s₁`, `t₂ ⊆ t₁` and they take up all but a `δ`-proportion, then the difference in edge
densities is at most `2 * δ`. -/
theorem abs_edgeDensity_sub_edgeDensity_le_two_mul (hs : s₂ ⊆ s₁) (ht : t₂ ⊆ t₁) (hδ : 0 ≤ δ)
    (hscard : (1 - δ) * #s₁ ≤ #s₂) (htcard : (1 - δ) * #t₁ ≤ #t₂) :
    |(edgeDensity r s₂ t₂ : 𝕜) - edgeDensity r s₁ t₁| ≤ 2 * δ := by
  /-
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ : LE.le 0 δ
    hscard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    htcard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r s₂ t₂) ↑(Rel.edgeDensity r s₁ t₁)) …
  -/
  cases' lt_or_le δ 1 with h h
  · exact (abs_edgeDensity_sub_edgeDensity_le_two_mul_sub_sq r hs ht hδ h hscard htcard).trans
      ((sub_le_self_iff _).2 <| sq_nonneg δ)
  /-
    case inr
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ : LE.le 0 δ
    hscard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    htcard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    h : LE.le 1 δ
    ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r s₂ t₂) ↑(Rel.edgeDensity r s₁ t₁)) …
  -/
  rw [two_mul]
  /-
    case inr
    𝕜 : Type u_1
    α : Type u_4
    β : Type u_5
    inst✝¹ : LinearOrderedField 𝕜
    r : α → β → Prop
    inst✝ : (a : α) → DecidablePred (r a)
    s₁ s₂ : Finset α
    t₁ t₂ : Finset β
    δ : 𝕜
    hs : HasSubset.Subset s₂ s₁
    ht : HasSubset.Subset t₂ t₁
    hδ : LE.le 0 δ
    hscard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
    htcard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
    h : LE.le 1 δ
    ⊢ LE.le (abs (HSub.hSub ↑(Rel.edgeDensity r s₂ t₂) ↑(Rel.edgeDensity r s₁ t₁)) …
  -/
  refine (abs_sub _ _).trans (add_le_add (le_trans ?_ h) (le_trans ?_ h)) <;>
      /-
        case inr.refine_1
        𝕜 : Type u_1
        α : Type u_4
        β : Type u_5
        inst✝¹ : LinearOrderedField 𝕜
        r : α → β → Prop
        inst✝ : (a : α) → DecidablePred (r a)
        s₁ s₂ : Finset α
        t₁ t₂ : Finset β
        δ : 𝕜
        hs : HasSubset.Subset s₂ s₁
        ht : HasSubset.Subset t₂ t₁
        hδ : LE.le 0 δ
        hscard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
        htcard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
        h : LE.le 1 δ
        ⊢ LE.le (abs ↑(Rel.edgeDensity r s₂ t₂)) 1
      -/
        /-
          case inr.refine_1
          𝕜 : Type u_1
          α : Type u_4
          β : Type u_5
          inst✝¹ : LinearOrderedField 𝕜
          r : α → β → Prop
          inst✝ : (a : α) → DecidablePred (r a)
          s₁ s₂ : Finset α
          t₁ t₂ : Finset β
          δ : 𝕜
          hs : HasSubset.Subset s₂ s₁
          ht : HasSubset.Subset t₂ t₁
          hδ : LE.le 0 δ
          hscard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
          htcard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
          h : LE.le 1 δ
          ⊢ LE.le (↑(Rel.edgeDensity r s₂ t₂)) 1
        -/
        /-
          🎉 no goals
        -/
        /-
          case inr.refine_1
          𝕜 : Type u_1
          α : Type u_4
          β : Type u_5
          inst✝¹ : LinearOrderedField 𝕜
          r : α → β → Prop
          inst✝ : (a : α) → DecidablePred (r a)
          s₁ s₂ : Finset α
          t₁ t₂ : Finset β
          δ : 𝕜
          hs : HasSubset.Subset s₂ s₁
          ht : HasSubset.Subset t₂ t₁
          hδ : LE.le 0 δ
          hscard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
          htcard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
          h : LE.le 1 δ
          ⊢ LE.le 0 ↑(Rel.edgeDensity r s₂ t₂)
        -/
        /-
          🎉 no goals
        -/
      · exact mod_cast edgeDensity_le_one r _ _
        /-
          🎉 no goals
        -/
        /-
          case inr.refine_2
          𝕜 : Type u_1
          α : Type u_4
          β : Type u_5
          inst✝¹ : LinearOrderedField 𝕜
          r : α → β → Prop
          inst✝ : (a : α) → DecidablePred (r a)
          s₁ s₂ : Finset α
          t₁ t₂ : Finset β
          δ : 𝕜
          hs : HasSubset.Subset s₂ s₁
          ht : HasSubset.Subset t₂ t₁
          hδ : LE.le 0 δ
          hscard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑s₁.card) ↑s₂.card
          htcard : LE.le (HMul.hMul (HSub.hSub 1 δ) ↑t₁.card) ↑t₂.card
          h : LE.le 1 δ
          ⊢ LE.le 0 ↑(Rel.edgeDensity r s₁ t₁)
        -/
      · exact mod_cast edgeDensity_nonneg r _ _
        /-
          🎉 no goals
        -/


@[simp]
theorem swap_mem_interedges_iff (hr : Symmetric r) {x : α × α} :
    x.swap ∈ interedges r s t ↔ x ∈ interedges r t s := by
  /-
    α : Type u_4
    r : α → α → Prop
    inst✝ : DecidableRel r
    s t : Finset α
    hr : Symmetric r
    x : Prod α α
    ⊢ Iff (Membership.mem (Rel.interedges r s t) x.swap) (Membership.mem (Rel.inte …
  -/
  rw [mem_interedges_iff, mem_interedges_iff, hr.iff]
  /-
    α : Type u_4
    r : α → α → Prop
    inst✝ : DecidableRel r
    s t : Finset α
    hr : Symmetric r
    x : Prod α α
    ⊢ Iff (And (Membership.mem s x.swap.1) (And (Membership.mem t x.swap.2) (r x.s …
  -/
  exact and_left_comm
  /-
    🎉 no goals
  -/


theorem mk_mem_interedges_comm (hr : Symmetric r) :
    (a, b) ∈ interedges r s t ↔ (b, a) ∈ interedges r t s :=
  @swap_mem_interedges_iff _ _ _ _ _ hr (b, a)


theorem card_interedges_comm (hr : Symmetric r) (s t : Finset α) :
    #(interedges r s t) = #(interedges r t s) :=
  Finset.card_bij (fun (x : α × α) _ ↦ x.swap) (fun _ ↦ (swap_mem_interedges_iff hr).2)
    (fun _ _ _ _ h ↦ Prod.swap_injective h) fun x h ↦
    ⟨x.swap, (swap_mem_interedges_iff hr).2 h, x.swap_swap⟩


theorem edgeDensity_comm (hr : Symmetric r) (s t : Finset α) :
    edgeDensity r s t = edgeDensity r t s := by
  /-
    α : Type u_4
    r : α → α → Prop
    inst✝ : DecidableRel r
    hr : Symmetric r
    s t : Finset α
    ⊢ Eq (Rel.edgeDensity r s t) (Rel.edgeDensity r t s)
  -/
  rw [edgeDensity, mul_comm, card_interedges_comm hr, edgeDensity]
  /-
    🎉 no goals
  -/


/-- Finset of edges of a relation between two finsets of vertices. -/
def interedges (s t : Finset α) : Finset (α × α) :=
  Rel.interedges G.Adj s t


/-- Density of edges of a graph between two finsets of vertices. -/
def edgeDensity : Finset α → Finset α → ℚ :=
  Rel.edgeDensity G.Adj


lemma interedges_def (s t : Finset α) : G.interedges s t = {e ∈ s ×ˢ t | G.Adj e.1 e.2} := rfl


lemma edgeDensity_def (s t : Finset α) : G.edgeDensity s t = #(G.interedges s t) / (#s * #t) := rfl


theorem card_interedges_div_card (s t : Finset α) :
    (#(G.interedges s t) : ℚ) / (#s * #t) = G.edgeDensity s t :=
  rfl


theorem mem_interedges_iff {x : α × α} : x ∈ G.interedges s t ↔ x.1 ∈ s ∧ x.2 ∈ t ∧ G.Adj x.1 x.2 :=
  Rel.mem_interedges_iff


theorem mk_mem_interedges_iff : (a, b) ∈ G.interedges s t ↔ a ∈ s ∧ b ∈ t ∧ G.Adj a b :=
  Rel.mk_mem_interedges_iff


@[simp]
theorem interedges_empty_left (t : Finset α) : G.interedges ∅ t = ∅ :=
  Rel.interedges_empty_left _


theorem interedges_mono : s₂ ⊆ s₁ → t₂ ⊆ t₁ → G.interedges s₂ t₂ ⊆ G.interedges s₁ t₁ :=
  Rel.interedges_mono


theorem interedges_disjoint_left (hs : Disjoint s₁ s₂) (t : Finset α) :
    Disjoint (G.interedges s₁ t) (G.interedges s₂ t) :=
  Rel.interedges_disjoint_left _ hs _


theorem interedges_disjoint_right (s : Finset α) (ht : Disjoint t₁ t₂) :
    Disjoint (G.interedges s t₁) (G.interedges s t₂) :=
  Rel.interedges_disjoint_right _ _ ht


theorem interedges_biUnion_left (s : Finset ι) (t : Finset α) (f : ι → Finset α) :
    G.interedges (s.biUnion f) t = s.biUnion fun a ↦ G.interedges (f a) t :=
  Rel.interedges_biUnion_left _ _ _ _


theorem interedges_biUnion_right (s : Finset α) (t : Finset ι) (f : ι → Finset α) :
    G.interedges s (t.biUnion f) = t.biUnion fun b ↦ G.interedges s (f b) :=
  Rel.interedges_biUnion_right _ _ _ _


theorem interedges_biUnion (s : Finset ι) (t : Finset κ) (f : ι → Finset α) (g : κ → Finset α) :
    G.interedges (s.biUnion f) (t.biUnion g) =
      (s ×ˢ t).biUnion fun ab ↦ G.interedges (f ab.1) (g ab.2) :=
  Rel.interedges_biUnion _ _ _ _ _


theorem card_interedges_add_card_interedges_compl (h : Disjoint s t) :
    #(G.interedges s t) + #(Gᶜ.interedges s t) = #s * #t := by
  /-
    α : Type u_4
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    s t : Finset α
    inst✝ : DecidableEq α
    h : Disjoint s t
    ⊢ Eq (HAdd.hAdd (G.interedges s t).card ((HasCompl.compl G).interedges s t).ca …
  -/
  rw [← card_product, interedges_def, interedges_def]
  have : {e ∈ s ×ˢ t | Gᶜ.Adj e.1 e.2} = {e ∈ s ×ˢ t | ¬G.Adj e.1 e.2} := by
    refine filter_congr fun x hx ↦ ?_
    rw [mem_product] at hx
    rw [compl_adj, and_iff_right (h.forall_ne_finset hx.1 hx.2)]
  /-
    α : Type u_4
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    s t : Finset α
    inst✝ : DecidableEq α
    h : Disjoint s t
    this : Eq (Finset.filter (fun e => (HasCompl.compl G).Adj e.1 e.2) (SProd.spro …
    ⊢ Eq (HAdd.hAdd (Finset.filter (fun e => G.Adj e.1 e.2) (SProd.sprod s t)).car …
  -/
  rw [this, ← card_union_of_disjoint, filter_union_filter_neg_eq]
  /-
    α : Type u_4
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    s t : Finset α
    inst✝ : DecidableEq α
    h : Disjoint s t
    this : Eq (Finset.filter (fun e => (HasCompl.compl G).Adj e.1 e.2) (SProd.spro …
    ⊢ Disjoint (Finset.filter (fun e => G.Adj e.1 e.2) (SProd.sprod s t)) (Finset. …
  -/
  exact disjoint_filter.2 fun _ _ ↦ Classical.not_not.2
  /-
    🎉 no goals
  -/


theorem edgeDensity_add_edgeDensity_compl (hs : s.Nonempty) (ht : t.Nonempty) (h : Disjoint s t) :
    G.edgeDensity s t + Gᶜ.edgeDensity s t = 1 := by
  /-
    α : Type u_4
    G : SimpleGraph α
    inst✝¹ : DecidableRel G.Adj
    s t : Finset α
    inst✝ : DecidableEq α
    hs : s.Nonempty
    ht : t.Nonempty
    h : Disjoint s t
    ⊢ Eq (HAdd.hAdd (G.edgeDensity s t) ((HasCompl.compl G).edgeDensity s t)) 1
  -/
  rw [edgeDensity_def, edgeDensity_def, div_add_div_same, div_eq_one_iff_eq]
    /-
      α : Type u_4
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      s t : Finset α
      inst✝ : DecidableEq α
      hs : s.Nonempty
      ht : t.Nonempty
      h : Disjoint s t
      ⊢ Eq (HAdd.hAdd ↑(G.interedges s t).card ↑((HasCompl.compl G).interedges s t). …
    -/
  · exact mod_cast card_interedges_add_card_interedges_compl _ h
    /-
      🎉 no goals
    -/
  -- Porting note: Wrote a workaround for `positivity` tactic.
    /-
      α : Type u_4
      G : SimpleGraph α
      inst✝¹ : DecidableRel G.Adj
      s t : Finset α
      inst✝ : DecidableEq α
      hs : s.Nonempty
      ht : t.Nonempty
      h : Disjoint s t
      ⊢ Ne (HMul.hMul ↑s.card ↑t.card) 0
    -/
                          /-
                            🎉 no goals
                          -/
  · apply mul_ne_zero <;> exact mod_cast Nat.pos_iff_ne_zero.1 (Nonempty.card_pos ‹_›)
                          /-
                            🎉 no goals
                          -/


theorem card_interedges_le_mul (s t : Finset α) : #(G.interedges s t) ≤ #s * #t :=
  Rel.card_interedges_le_mul _ _ _


theorem edgeDensity_nonneg (s t : Finset α) : 0 ≤ G.edgeDensity s t :=
  Rel.edgeDensity_nonneg _ _ _


theorem edgeDensity_le_one (s t : Finset α) : G.edgeDensity s t ≤ 1 :=
  Rel.edgeDensity_le_one _ _ _


@[simp]
theorem edgeDensity_empty_left (t : Finset α) : G.edgeDensity ∅ t = 0 :=
  Rel.edgeDensity_empty_left _ _


@[simp]
theorem edgeDensity_empty_right (s : Finset α) : G.edgeDensity s ∅ = 0 :=
  Rel.edgeDensity_empty_right _ _


@[simp]
theorem swap_mem_interedges_iff {x : α × α} : x.swap ∈ G.interedges s t ↔ x ∈ G.interedges t s :=
  Rel.swap_mem_interedges_iff G.symm


theorem mk_mem_interedges_comm : (a, b) ∈ G.interedges s t ↔ (b, a) ∈ G.interedges t s :=
  Rel.mk_mem_interedges_comm G.symm


theorem edgeDensity_comm (s t : Finset α) : G.edgeDensity s t = G.edgeDensity t s :=
  Rel.edgeDensity_comm G.symm s t


