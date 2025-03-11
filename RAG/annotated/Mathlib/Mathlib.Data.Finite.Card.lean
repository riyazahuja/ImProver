/-- There is (noncomputably) an equivalence between a finite type `α` and `Fin (Nat.card α)`. -/
def Finite.equivFin (α : Type*) [Finite α] : α ≃ Fin (Nat.card α) := by
  /-
    α✝ : Type u_1
    β : Type u_2
    γ : Type u_3
    α : Type u_4
    inst✝ : Finite α
    ⊢ Equiv α (Fin (Nat.card α))
  -/
  have := (Finite.exists_equiv_fin α).choose_spec.some
  /-
    α✝ : Type u_1
    β : Type u_2
    γ : Type u_3
    α : Type u_4
    inst✝ : Finite α
    this : Equiv α (Fin ⋯.choose)
    ⊢ Equiv α (Fin (Nat.card α))
  -/
  rwa [Nat.card_eq_of_equiv_fin this]
  /-
    🎉 no goals
  -/


/-- Similar to `Finite.equivFin` but with control over the term used for the cardinality. -/
def Finite.equivFinOfCardEq [Finite α] {n : ℕ} (h : Nat.card α = n) : α ≃ Fin n := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Finite α
    n : Nat
    h : Eq (Nat.card α) n
    ⊢ Equiv α (Fin n)
  -/
  subst h
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : Finite α
    ⊢ Equiv α (Fin (Nat.card α))
  -/
  apply Finite.equivFin
  /-
    🎉 no goals
  -/


open scoped Classical in
theorem Nat.card_eq (α : Type*) :
    Nat.card α = if _ : Finite α then @Fintype.card α (Fintype.ofFinite α) else 0 := by
  /-
    α : Type u_4
    ⊢ Eq (Nat.card α) (dite (Finite α) (fun x => Fintype.card α) fun x => 0)
  -/
  cases finite_or_infinite α
    /-
      case inl
      α : Type u_4
      h✝ : Finite α
      ⊢ Eq (Nat.card α) (dite (Finite α) (fun x => Fintype.card α) fun x => 0)
    -/
  · letI := Fintype.ofFinite α
    /-
      case inl
      α : Type u_4
      h✝ : Finite α
      this : Fintype α := Fintype.ofFinite α
      ⊢ Eq (Nat.card α) (dite (Finite α) (fun x => Fintype.card α) fun x => 0)
    -/
    simp only [this, *, Nat.card_eq_fintype_card, dif_pos]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_4
      h✝ : Infinite α
      ⊢ Eq (Nat.card α) (dite (Finite α) (fun x => Fintype.card α) fun x => 0)
    -/
  · simp only [*, card_eq_zero_of_infinite, not_finite_iff_infinite.mpr, dite_false]
    /-
      🎉 no goals
    -/


theorem Finite.card_pos_iff [Finite α] : 0 < Nat.card α ↔ Nonempty α := by
  /-
    α : Type u_1
    inst✝ : Finite α
    ⊢ Iff (LT.lt 0 (Nat.card α)) (Nonempty α)
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    inst✝ : Finite α
    this : Fintype α
    ⊢ Iff (LT.lt 0 (Nat.card α)) (Nonempty α)
  -/
  rw [Nat.card_eq_fintype_card, Fintype.card_pos_iff]
  /-
    🎉 no goals
  -/


theorem Finite.card_pos [Finite α] [h : Nonempty α] : 0 < Nat.card α :=
  Finite.card_pos_iff.mpr h


theorem cast_card_eq_mk {α : Type*} [Finite α] : ↑(Nat.card α) = Cardinal.mk α :=
  Cardinal.cast_toNat_of_lt_aleph0 (Cardinal.lt_aleph0_of_finite α)


theorem card_eq [Finite α] [Finite β] : Nat.card α = Nat.card β ↔ Nonempty (α ≃ β) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    ⊢ Iff (Eq (Nat.card α) (Nat.card β)) (Nonempty (Equiv α β))
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    this : Fintype α
    ⊢ Iff (Eq (Nat.card α) (Nat.card β)) (Nonempty (Equiv α β))
  -/
  haveI := Fintype.ofFinite β
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    this✝ : Fintype α
    this : Fintype β
    ⊢ Iff (Eq (Nat.card α) (Nat.card β)) (Nonempty (Equiv α β))
  -/
  simp only [Nat.card_eq_fintype_card, Fintype.card_eq]
  /-
    🎉 no goals
  -/


theorem card_le_one_iff_subsingleton [Finite α] : Nat.card α ≤ 1 ↔ Subsingleton α := by
  /-
    α : Type u_1
    inst✝ : Finite α
    ⊢ Iff (LE.le (Nat.card α) 1) (Subsingleton α)
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    inst✝ : Finite α
    this : Fintype α
    ⊢ Iff (LE.le (Nat.card α) 1) (Subsingleton α)
  -/
  simp only [Nat.card_eq_fintype_card, Fintype.card_le_one_iff_subsingleton]
  /-
    🎉 no goals
  -/


theorem one_lt_card_iff_nontrivial [Finite α] : 1 < Nat.card α ↔ Nontrivial α := by
  /-
    α : Type u_1
    inst✝ : Finite α
    ⊢ Iff (LT.lt 1 (Nat.card α)) (Nontrivial α)
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    inst✝ : Finite α
    this : Fintype α
    ⊢ Iff (LT.lt 1 (Nat.card α)) (Nontrivial α)
  -/
  simp only [Nat.card_eq_fintype_card, Fintype.one_lt_card_iff_nontrivial]
  /-
    🎉 no goals
  -/


theorem one_lt_card [Finite α] [h : Nontrivial α] : 1 < Nat.card α :=
  one_lt_card_iff_nontrivial.mpr h


@[simp]
theorem card_option [Finite α] : Nat.card (Option α) = Nat.card α + 1 := by
  /-
    α : Type u_1
    inst✝ : Finite α
    ⊢ Eq (Nat.card (Option α)) (HAdd.hAdd (Nat.card α) 1)
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    inst✝ : Finite α
    this : Fintype α
    ⊢ Eq (Nat.card (Option α)) (HAdd.hAdd (Nat.card α) 1)
  -/
  simp only [Nat.card_eq_fintype_card, Fintype.card_option]
  /-
    🎉 no goals
  -/


theorem card_le_of_injective [Finite β] (f : α → β) (hf : Function.Injective f) :
    Nat.card α ≤ Nat.card β := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Finite β
    f : α → β
    hf : Function.Injective f
    ⊢ LE.le (Nat.card α) (Nat.card β)
  -/
  haveI := Fintype.ofFinite β
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Finite β
    f : α → β
    hf : Function.Injective f
    this : Fintype β
    ⊢ LE.le (Nat.card α) (Nat.card β)
  -/
  haveI := Fintype.ofInjective f hf
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Finite β
    f : α → β
    hf : Function.Injective f
    this✝ : Fintype β
    this : Fintype α
    ⊢ LE.le (Nat.card α) (Nat.card β)
  -/
  simpa only [Nat.card_eq_fintype_card] using Fintype.card_le_of_injective f hf
  /-
    🎉 no goals
  -/


theorem card_le_of_embedding [Finite β] (f : α ↪ β) : Nat.card α ≤ Nat.card β :=
  card_le_of_injective _ f.injective


theorem card_le_of_surjective [Finite α] (f : α → β) (hf : Function.Surjective f) :
    Nat.card β ≤ Nat.card α := by
  classical
  haveI := Fintype.ofFinite α
  haveI := Fintype.ofSurjective f hf
  simpa only [Nat.card_eq_fintype_card] using Fintype.card_le_of_surjective f hf


theorem card_eq_zero_iff [Finite α] : Nat.card α = 0 ↔ IsEmpty α := by
  /-
    α : Type u_1
    inst✝ : Finite α
    ⊢ Iff (Eq (Nat.card α) 0) (IsEmpty α)
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    inst✝ : Finite α
    this : Fintype α
    ⊢ Iff (Eq (Nat.card α) 0) (IsEmpty α)
  -/
  simp only [Nat.card_eq_fintype_card, Fintype.card_eq_zero_iff]
  /-
    🎉 no goals
  -/


/-- If `f` is injective, then `Nat.card α ≤ Nat.card β`. We must also assume
  `Nat.card β = 0 → Nat.card α = 0` since `Nat.card` is defined to be `0` for infinite types. -/
theorem card_le_of_injective' {f : α → β} (hf : Function.Injective f)
    (h : Nat.card β = 0 → Nat.card α = 0) : Nat.card α ≤ Nat.card β :=
  (or_not_of_imp h).casesOn (fun h => le_of_eq_of_le h zero_le') fun h =>
    @card_le_of_injective α β (Nat.finite_of_card_ne_zero h) f hf


/-- If `f` is an embedding, then `Nat.card α ≤ Nat.card β`. We must also assume
  `Nat.card β = 0 → Nat.card α = 0` since `Nat.card` is defined to be `0` for infinite types. -/
theorem card_le_of_embedding' (f : α ↪ β) (h : Nat.card β = 0 → Nat.card α = 0) :
    Nat.card α ≤ Nat.card β :=
  card_le_of_injective' f.2 h


/-- If `f` is surjective, then `Nat.card β ≤ Nat.card α`. We must also assume
  `Nat.card α = 0 → Nat.card β = 0` since `Nat.card` is defined to be `0` for infinite types. -/
theorem card_le_of_surjective' {f : α → β} (hf : Function.Surjective f)
    (h : Nat.card α = 0 → Nat.card β = 0) : Nat.card β ≤ Nat.card α :=
  (or_not_of_imp h).casesOn (fun h => le_of_eq_of_le h zero_le') fun h =>
    @card_le_of_surjective α β (Nat.finite_of_card_ne_zero h) f hf


/-- NB: `Nat.card` is defined to be `0` for infinite types. -/
theorem card_eq_zero_of_surjective {f : α → β} (hf : Function.Surjective f) (h : Nat.card β = 0) :
    Nat.card α = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Surjective f
    h : Eq (Nat.card β) 0
    ⊢ Eq (Nat.card α) 0
  -/
  cases finite_or_infinite β
    /-
      case inl
      α : Type u_1
      β : Type u_2
      f : α → β
      hf : Function.Surjective f
      h : Eq (Nat.card β) 0
      h✝ : Finite β
      ⊢ Eq (Nat.card α) 0
    -/
  · haveI := card_eq_zero_iff.mp h
    /-
      case inl
      α : Type u_1
      β : Type u_2
      f : α → β
      hf : Function.Surjective f
      h : Eq (Nat.card β) 0
      h✝ : Finite β
      this : IsEmpty β
      ⊢ Eq (Nat.card α) 0
    -/
    haveI := Function.isEmpty f
    /-
      case inl
      α : Type u_1
      β : Type u_2
      f : α → β
      hf : Function.Surjective f
      h : Eq (Nat.card β) 0
      h✝ : Finite β
      this✝ : IsEmpty β
      this : IsEmpty α
      ⊢ Eq (Nat.card α) 0
    -/
    exact Nat.card_of_isEmpty
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      f : α → β
      hf : Function.Surjective f
      h : Eq (Nat.card β) 0
      h✝ : Infinite β
      ⊢ Eq (Nat.card α) 0
    -/
  · haveI := Infinite.of_surjective f hf
    /-
      case inr
      α : Type u_1
      β : Type u_2
      f : α → β
      hf : Function.Surjective f
      h : Eq (Nat.card β) 0
      h✝ : Infinite β
      this : Infinite α
      ⊢ Eq (Nat.card α) 0
    -/
    exact Nat.card_eq_zero_of_infinite
    /-
      🎉 no goals
    -/


/-- NB: `Nat.card` is defined to be `0` for infinite types. -/
theorem card_eq_zero_of_injective [Nonempty α] {f : α → β} (hf : Function.Injective f)
    (h : Nat.card α = 0) : Nat.card β = 0 :=
  card_eq_zero_of_surjective (Function.invFun_surjective hf) h


/-- NB: `Nat.card` is defined to be `0` for infinite types. -/
theorem card_eq_zero_of_embedding [Nonempty α] (f : α ↪ β) (h : Nat.card α = 0) : Nat.card β = 0 :=
  card_eq_zero_of_injective f.2 h


theorem card_sum [Finite α] [Finite β] : Nat.card (α ⊕ β) = Nat.card α + Nat.card β := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    ⊢ Eq (Nat.card (Sum α β)) (HAdd.hAdd (Nat.card α) (Nat.card β))
  -/
  haveI := Fintype.ofFinite α
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    this : Fintype α
    ⊢ Eq (Nat.card (Sum α β)) (HAdd.hAdd (Nat.card α) (Nat.card β))
  -/
  haveI := Fintype.ofFinite β
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Finite α
    inst✝ : Finite β
    this✝ : Fintype α
    this : Fintype β
    ⊢ Eq (Nat.card (Sum α β)) (HAdd.hAdd (Nat.card α) (Nat.card β))
  -/
  simp only [Nat.card_eq_fintype_card, Fintype.card_sum]
  /-
    🎉 no goals
  -/


theorem card_image_le {s : Set α} [Finite s] (f : α → β) : Nat.card (f '' s) ≤ Nat.card s :=
  card_le_of_surjective _ Set.surjective_onto_image


theorem card_range_le [Finite α] (f : α → β) : Nat.card (Set.range f) ≤ Nat.card α :=
  card_le_of_surjective _ Set.surjective_onto_range


theorem card_subtype_le [Finite α] (p : α → Prop) : Nat.card { x // p x } ≤ Nat.card α := by
  classical
  haveI := Fintype.ofFinite α
  simpa only [Nat.card_eq_fintype_card] using Fintype.card_subtype_le p


theorem card_subtype_lt [Finite α] {p : α → Prop} {x : α} (hx : ¬p x) :
    Nat.card { x // p x } < Nat.card α := by
  classical
  haveI := Fintype.ofFinite α
  simpa only [Nat.card_eq_fintype_card, gt_iff_lt] using Fintype.card_subtype_lt hx


theorem card_eq_coe_natCard (α : Type*) [Finite α] : card α = Nat.card α := by
  /-
    α : Type u_4
    inst✝ : Finite α
    ⊢ Eq (ENat.card α) ↑(Nat.card α)
  -/
  unfold ENat.card
  /-
    α : Type u_4
    inst✝ : Finite α
    ⊢ Eq (Cardinal.toENat (Cardinal.mk α)) ↑(Nat.card α)
  -/
  apply symm
  /-
    case a
    α : Type u_4
    inst✝ : Finite α
    ⊢ Eq (↑(Nat.card α)) (Cardinal.toENat (Cardinal.mk α))
  -/
  rw [Cardinal.natCast_eq_toENat_iff]
  /-
    case a
    α : Type u_4
    inst✝ : Finite α
    ⊢ Eq (↑(Nat.card α)) (Cardinal.mk α)
  -/
  exact Finite.cast_card_eq_mk
  /-
    🎉 no goals
  -/


theorem card_union_le (s t : Set α) : Nat.card (↥(s ∪ t)) ≤ Nat.card s + Nat.card t := by
  /-
    α : Type u_1
    s t : Set α
    ⊢ LE.le (Nat.card ↑(Union.union s t)) (HAdd.hAdd (Nat.card ↑s) (Nat.card ↑t))
  -/
  cases' _root_.finite_or_infinite (↥(s ∪ t)) with h h
    /-
      case inl
      α : Type u_1
      s t : Set α
      h : Finite ↑(Union.union s t)
      ⊢ LE.le (Nat.card ↑(Union.union s t)) (HAdd.hAdd (Nat.card ↑s) (Nat.card ↑t))
    -/
  · rw [finite_coe_iff, finite_union, ← finite_coe_iff, ← finite_coe_iff] at h
    /-
      case inl
      α : Type u_1
      s t : Set α
      h : And (Finite ↑s) (Finite ↑t)
      ⊢ LE.le (Nat.card ↑(Union.union s t)) (HAdd.hAdd (Nat.card ↑s) (Nat.card ↑t))
    -/
    cases h
    rw [← @Nat.cast_le Cardinal, Nat.cast_add, Finite.cast_card_eq_mk, Finite.cast_card_eq_mk,
      Finite.cast_card_eq_mk]
    /-
      case inl.intro
      α : Type u_1
      s t : Set α
      left✝ : Finite ↑s
      right✝ : Finite ↑t
      ⊢ LE.le (Cardinal.mk ↑(Union.union s t)) (HAdd.hAdd (Cardinal.mk ↑s) (Cardinal …
    -/
    exact Cardinal.mk_union_le s t
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      s t : Set α
      h : Infinite ↑(Union.union s t)
      ⊢ LE.le (Nat.card ↑(Union.union s t)) (HAdd.hAdd (Nat.card ↑s) (Nat.card ↑t))
    -/
  · exact Nat.card_eq_zero_of_infinite.trans_le (zero_le _)
    /-
      🎉 no goals
    -/


theorem card_lt_card (ht : t.Finite) (hsub : s ⊂ t) : Nat.card s < Nat.card t := by
  /-
    α : Type u_1
    s t : Set α
    ht : t.Finite
    hsub : HasSSubset.SSubset s t
    ⊢ LT.lt (Nat.card ↑s) (Nat.card ↑t)
  -/
  have : Fintype t := Finite.fintype ht
  /-
    α : Type u_1
    s t : Set α
    ht : t.Finite
    hsub : HasSSubset.SSubset s t
    this : Fintype ↑t
    ⊢ LT.lt (Nat.card ↑s) (Nat.card ↑t)
  -/
  have : Fintype s := Finite.fintype (subset ht (subset_of_ssubset hsub))
  /-
    α : Type u_1
    s t : Set α
    ht : t.Finite
    hsub : HasSSubset.SSubset s t
    this✝ : Fintype ↑t
    this : Fintype ↑s
    ⊢ LT.lt (Nat.card ↑s) (Nat.card ↑t)
  -/
  simp only [Nat.card_eq_fintype_card]
  /-
    α : Type u_1
    s t : Set α
    ht : t.Finite
    hsub : HasSSubset.SSubset s t
    this✝ : Fintype ↑t
    this : Fintype ↑s
    ⊢ LT.lt (Fintype.card ↑s) (Fintype.card ↑t)
  -/
  exact Set.card_lt_card hsub
  /-
    🎉 no goals
  -/


theorem eq_of_subset_of_card_le (ht : t.Finite) (hsub : s ⊆ t) (hcard : Nat.card t ≤ Nat.card s) :
    s = t :=
  (eq_or_ssubset_of_subset hsub).elim id fun h ↦ absurd hcard <| not_le_of_lt <| ht.card_lt_card h


theorem equiv_image_eq_iff_subset (e : α ≃ α) (hs : s.Finite) : e '' s = s ↔ e '' s ⊆ s :=
              /-
                α : Type u_1
                s : Set α
                e : Equiv α α
                hs : s.Finite
                h : Eq (Set.image (⇑e) s) s
                ⊢ HasSubset.Subset (Set.image (⇑e) s) s
              -/
  ⟨fun h ↦ by rw [h], fun h ↦ hs.eq_of_subset_of_card_le h <|
              /-
                🎉 no goals
              -/
    ge_of_eq (Nat.card_congr (e.image s).symm)⟩


theorem eq_top_of_card_le_of_finite [Finite α] {s : Set α} (h : Nat.card α ≤ Nat.card s) : s = ⊤ :=
  Set.Finite.eq_of_subset_of_card_le univ.toFinite (subset_univ s) <|
    Nat.card_congr (Equiv.Set.univ α) ▸ h


