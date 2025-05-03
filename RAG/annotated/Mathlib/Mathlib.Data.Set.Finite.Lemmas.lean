theorem Finite.fin_embedding {s : Set α} (h : s.Finite) :
    ∃ (n : ℕ) (f : Fin n ↪ α), range f = s :=
  ⟨_, (Fintype.equivFin (h.toFinset : Set α)).symm.asEmbedding, by
    /-
      α : Type u
      s : Set α
      h : s.Finite
      ⊢ Eq (Set.range ⇑(Fintype.equivFin ↑↑h.toFinset).symm.asEmbedding) s
    -/
    simp only [Finset.coe_sort_coe, Equiv.asEmbedding_range, Finite.coe_toFinset, setOf_mem_eq]⟩
    /-
      🎉 no goals
    -/


theorem Finite.fin_param {s : Set α} (h : s.Finite) :
    ∃ (n : ℕ) (f : Fin n → α), Injective f ∧ range f = s :=
  let ⟨n, f, hf⟩ := h.fin_embedding
  ⟨n, f, f.injective, hf⟩


/-- Induction up to a finite set `S`. -/
theorem Finite.induction_to {C : Set α → Prop} {S : Set α} (h : S.Finite)
    (S0 : Set α) (hS0 : S0 ⊆ S) (H0 : C S0) (H1 : ∀ s ⊂ S, C s → ∃ a ∈ S \ s, C (insert a s)) :
    C S := by
  /-
    α : Type u
    C : Set α → Prop
    S : Set α
    h : S.Finite
    S0 : Set α
    hS0 : HasSubset.Subset S0 S
    H0 : C S0
    H1 : ∀ (s : Set α), HasSSubset.SSubset s S → C s → Exists fun a => And (Member …
    ⊢ C S
  -/
  have : Finite S := Finite.to_subtype h
  /-
    α : Type u
    C : Set α → Prop
    S : Set α
    h : S.Finite
    S0 : Set α
    hS0 : HasSubset.Subset S0 S
    H0 : C S0
    H1 : ∀ (s : Set α), HasSSubset.SSubset s S → C s → Exists fun a => And (Member …
    this : Finite ↑S
    ⊢ C S
  -/
  have : Finite {T : Set α // T ⊆ S} := Finite.of_equiv (Set S) (Equiv.Set.powerset S).symm
  /-
    α : Type u
    C : Set α → Prop
    S : Set α
    h : S.Finite
    S0 : Set α
    hS0 : HasSubset.Subset S0 S
    H0 : C S0
    H1 : ∀ (s : Set α), HasSSubset.SSubset s S → C s → Exists fun a => And (Member …
    this✝ : Finite ↑S
    this : Finite (Subtype fun T => HasSubset.Subset T S)
    ⊢ C S
  -/
  rw [← Subtype.coe_mk (p := (· ⊆ S)) _ le_rfl]
  /-
    α : Type u
    C : Set α → Prop
    S : Set α
    h : S.Finite
    S0 : Set α
    hS0 : HasSubset.Subset S0 S
    H0 : C S0
    H1 : ∀ (s : Set α), HasSSubset.SSubset s S → C s → Exists fun a => And (Member …
    this✝ : Finite ↑S
    this : Finite (Subtype fun T => HasSubset.Subset T S)
    ⊢ C ↑⟨S, ⋯⟩
  -/
  rw [← Subtype.coe_mk (p := (· ⊆ S)) _ hS0] at H0
  /-
    α : Type u
    C : Set α → Prop
    S : Set α
    h : S.Finite
    S0 : Set α
    hS0 : HasSubset.Subset S0 S
    H0 : C ↑⟨S0, hS0⟩
    H1 : ∀ (s : Set α), HasSSubset.SSubset s S → C s → Exists fun a => And (Member …
    this✝ : Finite ↑S
    this : Finite (Subtype fun T => HasSubset.Subset T S)
    ⊢ C ↑⟨S, ⋯⟩
  -/
  refine Finite.to_wellFoundedGT.wf.induction_bot' (fun s hs hs' ↦ ?_) H0
  /-
    α : Type u
    C : Set α → Prop
    S : Set α
    h : S.Finite
    S0 : Set α
    hS0 : HasSubset.Subset S0 S
    H0 : C ↑⟨S0, hS0⟩
    H1 : ∀ (s : Set α), HasSSubset.SSubset s S → C s → Exists fun a => And (Member …
    this✝ : Finite ↑S
    this : Finite (Subtype fun T => HasSubset.Subset T S)
    s : Subtype fun x => HasSubset.Subset x S
    hs : Ne ↑s ↑⟨S, ⋯⟩
    hs' : C ↑s
    ⊢ Exists fun c => And (GT.gt c s) (C ↑c)
  -/
  obtain ⟨a, ⟨ha1, ha2⟩, ha'⟩ := H1 s (ssubset_of_ne_of_subset hs s.2) hs'
  /-
    case intro.intro.intro
    α : Type u
    C : Set α → Prop
    S : Set α
    h : S.Finite
    S0 : Set α
    hS0 : HasSubset.Subset S0 S
    H0 : C ↑⟨S0, hS0⟩
    H1 : ∀ (s : Set α), HasSSubset.SSubset s S → C s → Exists fun a => And (Member …
    this✝ : Finite ↑S
    this : Finite (Subtype fun T => HasSubset.Subset T S)
    s : Subtype fun x => HasSubset.Subset x S
    hs : Ne ↑s ↑⟨S, ⋯⟩
    hs' : C ↑s
    a : α
    ha' : C (Insert.insert a ↑s)
    ha1 : Membership.mem S a
    ha2 : Not (Membership.mem (↑s) a)
    ⊢ Exists fun c => And (GT.gt c s) (C ↑c)
  -/
  exact ⟨⟨insert a s.1, insert_subset ha1 s.2⟩, Set.ssubset_insert ha2, ha'⟩
  /-
    🎉 no goals
  -/


/-- Induction up to `univ`. -/
theorem Finite.induction_to_univ [Finite α] {C : Set α → Prop} (S0 : Set α)
    (H0 : C S0) (H1 : ∀ S ≠ univ, C S → ∃ a ∉ S, C (insert a S)) : C univ :=
                                                      /-
                                                        α : Type u
                                                        inst✝ : Finite α
                                                        C : Set α → Prop
                                                        S0 : Set α
                                                        H0 : C S0
                                                        H1 : ∀ (S : Set α), Ne S Set.univ → C S → Exists fun a => And (Not (Membership …
                                                        ⊢ ∀ (s : Set α), HasSSubset.SSubset s Set.univ → C s → Exists fun a => And (Me …
                                                      -/
  finite_univ.induction_to S0 (subset_univ S0) H0 (by simpa [ssubset_univ_iff])
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem exists_min_image [LinearOrder β] (s : Set α) (f : α → β) (h1 : s.Finite) :
    s.Nonempty → ∃ a ∈ s, ∀ b ∈ s, f a ≤ f b
  | ⟨x, hx⟩ => by
    simpa only [exists_prop, Finite.mem_toFinset] using
      h1.toFinset.exists_min_image f ⟨x, h1.mem_toFinset.2 hx⟩


theorem exists_max_image [LinearOrder β] (s : Set α) (f : α → β) (h1 : s.Finite) :
    s.Nonempty → ∃ a ∈ s, ∀ b ∈ s, f b ≤ f a
  | ⟨x, hx⟩ => by
    simpa only [exists_prop, Finite.mem_toFinset] using
      h1.toFinset.exists_max_image f ⟨x, h1.mem_toFinset.2 hx⟩


theorem exists_lower_bound_image [Nonempty α] [LinearOrder β] (s : Set α) (f : α → β)
    (h : s.Finite) : ∃ a : α, ∀ b ∈ s, f a ≤ f b := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Nonempty α
    inst✝ : LinearOrder β
    s : Set α
    f : α → β
    h : s.Finite
    ⊢ Exists fun a => ∀ (b : α), Membership.mem s b → LE.le (f a) (f b)
  -/
  rcases s.eq_empty_or_nonempty with rfl | hs
    /-
      case inl
      α : Type u
      β : Type v
      inst✝¹ : Nonempty α
      inst✝ : LinearOrder β
      f : α → β
      h : EmptyCollection.emptyCollection.Finite
      ⊢ Exists fun a => ∀ (b : α), Membership.mem EmptyCollection.emptyCollection b  …
    -/
  · exact ‹Nonempty α›.elim fun a => ⟨a, fun _ => False.elim⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      β : Type v
      inst✝¹ : Nonempty α
      inst✝ : LinearOrder β
      s : Set α
      f : α → β
      h : s.Finite
      hs : s.Nonempty
      ⊢ Exists fun a => ∀ (b : α), Membership.mem s b → LE.le (f a) (f b)
    -/
  · rcases Set.exists_min_image s f h hs with ⟨x₀, _, hx₀⟩
    /-
      case inr.intro.intro
      α : Type u
      β : Type v
      inst✝¹ : Nonempty α
      inst✝ : LinearOrder β
      s : Set α
      f : α → β
      h : s.Finite
      hs : s.Nonempty
      x₀ : α
      left✝ : Membership.mem s x₀
      hx₀ : ∀ (b : α), Membership.mem s b → LE.le (f x₀) (f b)
      ⊢ Exists fun a => ∀ (b : α), Membership.mem s b → LE.le (f a) (f b)
    -/
    exact ⟨x₀, fun x hx => hx₀ x hx⟩
    /-
      🎉 no goals
    -/


theorem exists_upper_bound_image [Nonempty α] [LinearOrder β] (s : Set α) (f : α → β)
    (h : s.Finite) : ∃ a : α, ∀ b ∈ s, f b ≤ f a :=
  exists_lower_bound_image (β := βᵒᵈ) s f h


