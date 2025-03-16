@[simp]
theorem Equiv.asEmbedding_range {α β : Sort _} {p : β → Prop} (e : α ≃ Subtype p) :
    Set.range e.asEmbedding = setOf p :=
                                                                                         /-
                                                                                           α : Sort u_1
                                                                                           β : Type u_2
                                                                                           p : β → Prop
                                                                                           e : Equiv α (Subtype p)
                                                                                           x : β
                                                                                           hs : Membership.mem (setOf p) x
                                                                                           ⊢ Eq (e.asEmbedding (e.symm ⟨x, hs⟩)) x
                                                                                         -/
  Set.ext fun x ↦ ⟨fun ⟨y, h⟩ ↦ h ▸ Subtype.coe_prop (e y), fun hs ↦ ⟨e.symm ⟨x, hs⟩, by simp⟩⟩
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


/-- Embedding into `WithTop α`. -/
@[simps]
def coeWithTop {α} : α ↪ WithTop α :=
  { Embedding.some with toFun := WithTop.some }


/-- Given an embedding `f : α ↪ β` and a point outside of `Set.range f`, construct an embedding
`Option α ↪ β`. -/
@[simps]
def optionElim {α β} (f : α ↪ β) (x : β) (h : x ∉ Set.range f) : Option α ↪ β :=
  ⟨Option.elim' x f, Option.injective_iff.2 ⟨f.2, h⟩⟩


/-- Equivalence between embeddings of `Option α` and a sigma type over the embeddings of `α`. -/
@[simps]
def optionEmbeddingEquiv (α β) : (Option α ↪ β) ≃ Σ f : α ↪ β, ↥(Set.range f)ᶜ where
  toFun f := ⟨coeWithTop.trans f, f none, fun ⟨x, hx⟩ ↦ Option.some_ne_none x <| f.injective hx⟩
  invFun f := f.1.optionElim f.2 f.2.2
                          /-
                            α : Type ?u.1094
                            β : Type ?u.1095
                            f : Function.Embedding (Option α) β
                            ⊢ ∀ (x : Option α), Eq (((fun f => f.fst.optionElim ↑f.snd ⋯) ((fun f => ⟨Func …
                          -/
                                             /-
                                               🎉 no goals
                                             -/
  left_inv f := ext <| by rintro (_ | _) <;> simp [Option.coe_def]; rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                   /-
                                     α : Type ?u.1094
                                     β : Type ?u.1095
                                     x✝ : Sigma fun f => ↑(HasCompl.compl (Set.range ⇑f))
                                     f : Function.Embedding α β
                                     y : β
                                     hy : Membership.mem (HasCompl.compl (Set.range ⇑f)) y
                                     ⊢ Eq ((fun f => ⟨Function.Embedding.coeWithTop.trans f, ⟨f Option.none, ⋯⟩⟩) ( …
                                   -/
                                           /-
                                             🎉 no goals
                                           -/
  right_inv := fun ⟨f, y, hy⟩ ↦ by ext <;> simp [Option.coe_def]; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Restrict the codomain of an embedding. -/
def codRestrict {α β} (p : Set β) (f : α ↪ β) (H : ∀ a, f a ∈ p) : α ↪ p :=
  ⟨fun a ↦ ⟨f a, H a⟩, fun _ _ h ↦ f.injective (congr_arg Subtype.val h)⟩


@[simp]
theorem codRestrict_apply {α β} (p) (f : α ↪ β) (H a) : codRestrict p f H a = ⟨f a, H a⟩ :=
  rfl


/-- `Set.image` as an embedding `Set α ↪ Set β`. -/
@[simps apply]
protected def image {α β} (f : α ↪ β) : Set α ↪ Set β :=
  ⟨image f, f.2.image_injective⟩


/-- The injection map is an embedding between subsets. -/
@[simps apply_coe]
def embeddingOfSubset {α} (s t : Set α) (h : s ⊆ t) : s ↪ t :=
  ⟨fun x ↦ ⟨x.1, h x.2⟩, fun ⟨x, hx⟩ ⟨y, hy⟩ h ↦ by
    /-
      α : Type ?u.3673
      s t : Set α
      h✝ : HasSubset.Subset s t
      x✝¹ x✝ : ↑s
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      h : Eq ((fun x => ⟨↑x, ⋯⟩) ⟨x, hx⟩) ((fun x => ⟨↑x, ⋯⟩) ⟨y, hy⟩)
      ⊢ Eq ⟨x, hx⟩ ⟨y, hy⟩
    -/
    congr
    /-
      case e_val
      α : Type ?u.3673
      s t : Set α
      h✝ : HasSubset.Subset s t
      x✝¹ x✝ : ↑s
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      h : Eq ((fun x => ⟨↑x, ⋯⟩) ⟨x, hx⟩) ((fun x => ⟨↑x, ⋯⟩) ⟨y, hy⟩)
      ⊢ Eq x y
    -/
    injection h⟩
    /-
      🎉 no goals
    -/


/-- A subtype `{x // p x ∨ q x}` over a disjunction of `p q : α → Prop` is equivalent to a sum of
subtypes `{x // p x} ⊕ {x // q x}` such that `¬ p x` is sent to the right, when
`Disjoint p q`.

See also `Equiv.sumCompl`, for when `IsCompl p q`. -/
@[simps apply]
def subtypeOrEquiv (p q : α → Prop) [DecidablePred p] (h : Disjoint p q) :
    { x // p x ∨ q x } ≃ { x // p x } ⊕ { x // q x } where
  toFun := subtypeOrLeftEmbedding p q
  invFun :=
    Sum.elim (Subtype.impEmbedding _ _ fun x hx ↦ (Or.inl hx : p x ∨ q x))
      (Subtype.impEmbedding _ _ fun x hx ↦ (Or.inr hx : p x ∨ q x))
  left_inv x := by
    /-
      α : Type u_1
      p q : α → Prop
      inst✝ : DecidablePred p
      h : Disjoint p q
      x : Subtype fun x => Or (p x) (q x)
      ⊢ Eq (Sum.elim (⇑(Subtype.impEmbedding p (fun x => Or (p x) (q x)) ⋯)) (⇑(Subt …
    -/
    by_cases hx : p x
      /-
        case pos
        α : Type u_1
        p q : α → Prop
        inst✝ : DecidablePred p
        h : Disjoint p q
        x : Subtype fun x => Or (p x) (q x)
        hx : p ↑x
        ⊢ Eq (Sum.elim (⇑(Subtype.impEmbedding p (fun x => Or (p x) (q x)) ⋯)) (⇑(Subt …
      -/
    · rw [subtypeOrLeftEmbedding_apply_left _ hx]
      /-
        case pos
        α : Type u_1
        p q : α → Prop
        inst✝ : DecidablePred p
        h : Disjoint p q
        x : Subtype fun x => Or (p x) (q x)
        hx : p ↑x
        ⊢ Eq (Sum.elim (⇑(Subtype.impEmbedding p (fun x => Or (p x) (q x)) ⋯)) (⇑(Subt …
      -/
      simp [Subtype.ext_iff]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        p q : α → Prop
        inst✝ : DecidablePred p
        h : Disjoint p q
        x : Subtype fun x => Or (p x) (q x)
        hx : Not (p ↑x)
        ⊢ Eq (Sum.elim (⇑(Subtype.impEmbedding p (fun x => Or (p x) (q x)) ⋯)) (⇑(Subt …
      -/
    · rw [subtypeOrLeftEmbedding_apply_right _ hx]
      /-
        case neg
        α : Type u_1
        p q : α → Prop
        inst✝ : DecidablePred p
        h : Disjoint p q
        x : Subtype fun x => Or (p x) (q x)
        hx : Not (p ↑x)
        ⊢ Eq (Sum.elim (⇑(Subtype.impEmbedding p (fun x => Or (p x) (q x)) ⋯)) (⇑(Subt …
      -/
      simp [Subtype.ext_iff]
      /-
        🎉 no goals
      -/
  right_inv x := by
    cases x with
    | inl x =>
        simp only [Sum.elim_inl]
        rw [subtypeOrLeftEmbedding_apply_left]
        · simp
        · simpa using x.prop
    | inr x =>
        simp only [Sum.elim_inr]
        rw [subtypeOrLeftEmbedding_apply_right]
        · simp
        · suffices ¬p x by simpa
          intro hp
          simpa using h.le_bot x ⟨hp, x.prop⟩


@[simp]
theorem subtypeOrEquiv_symm_inl (p q : α → Prop) [DecidablePred p] (h : Disjoint p q)
    (x : { x // p x }) : (subtypeOrEquiv p q h).symm (Sum.inl x) = ⟨x, Or.inl x.prop⟩ :=
  rfl


@[simp]
theorem subtypeOrEquiv_symm_inr (p q : α → Prop) [DecidablePred p] (h : Disjoint p q)
    (x : { x // q x }) : (subtypeOrEquiv p q h).symm (Sum.inr x) = ⟨x, Or.inr x.prop⟩ :=
  rfl


/-- For disjoint `s t : Set α`, the natural injection from `↑s ⊕ ↑t` to `α`. -/
@[simps] def Function.Embedding.sumSet (h : Disjoint s t) : s ⊕ t ↪ α where
  toFun := Sum.elim (↑) (↑)
  inj' := by
    /-
      α : Type u_1
      ι : Type u_2
      s t r : Set α
      h : Disjoint s t
      ⊢ Function.Injective (Sum.elim Subtype.val Subtype.val)
    -/
    rintro (⟨a, ha⟩ | ⟨a, ha⟩) (⟨b, hb⟩ | ⟨b, hb⟩)
      /-
        case inl.mk.inl.mk
        α : Type u_1
        ι : Type u_2
        s t r : Set α
        h : Disjoint s t
        a : α
        ha : Membership.mem s a
        b : α
        hb : Membership.mem s b
        ⊢ Eq (Sum.elim Subtype.val Subtype.val (Sum.inl ⟨a, ha⟩)) (Sum.elim Subtype.va …
      -/
    · simp [Subtype.val_inj]
      /-
        🎉 no goals
      -/
      /-
        case inl.mk.inr.mk
        α : Type u_1
        ι : Type u_2
        s t r : Set α
        h : Disjoint s t
        a : α
        ha : Membership.mem s a
        b : α
        hb : Membership.mem t b
        ⊢ Eq (Sum.elim Subtype.val Subtype.val (Sum.inl ⟨a, ha⟩)) (Sum.elim Subtype.va …
      -/
    · simpa using h.ne_of_mem ha hb
      /-
        🎉 no goals
      -/
      /-
        case inr.mk.inl.mk
        α : Type u_1
        ι : Type u_2
        s t r : Set α
        h : Disjoint s t
        a : α
        ha : Membership.mem t a
        b : α
        hb : Membership.mem s b
        ⊢ Eq (Sum.elim Subtype.val Subtype.val (Sum.inr ⟨a, ha⟩)) (Sum.elim Subtype.va …
      -/
    · simpa using h.symm.ne_of_mem ha hb
      /-
        🎉 no goals
      -/
    /-
      case inr.mk.inr.mk
      α : Type u_1
      ι : Type u_2
      s t r : Set α
      h : Disjoint s t
      a : α
      ha : Membership.mem t a
      b : α
      hb : Membership.mem t b
      ⊢ Eq (Sum.elim Subtype.val Subtype.val (Sum.inr ⟨a, ha⟩)) (Sum.elim Subtype.va …
    -/
    simp [Subtype.val_inj]
    /-
      🎉 no goals
    -/


@[norm_cast] lemma Function.Embedding.coe_sumSet (h : Disjoint s t) :
    (Function.Embedding.sumSet h : s ⊕ t → α) = Sum.elim (↑) (↑) := rfl


@[simp] theorem Function.Embedding.sumSet_preimage_inl (h : Disjoint s t) :
    .inl ⁻¹' (Function.Embedding.sumSet h ⁻¹' r) = r ∩ s := by
  /-
    α : Type u_1
    s t r : Set α
    h : Disjoint s t
    ⊢ Eq (Set.image Subtype.val (Set.preimage Sum.inl (Set.preimage (⇑(Function.Em …
  -/
  simp [Set.ext_iff]
  /-
    🎉 no goals
  -/


@[simp] theorem Function.Embedding.sumSet_preimage_inr (h : Disjoint s t) :
    .inr ⁻¹' (Function.Embedding.sumSet h ⁻¹' r) = r ∩ t := by
  /-
    α : Type u_1
    s t r : Set α
    h : Disjoint s t
    ⊢ Eq (Set.image Subtype.val (Set.preimage Sum.inr (Set.preimage (⇑(Function.Em …
  -/
  simp [Set.ext_iff]
  /-
    🎉 no goals
  -/


@[simp] theorem Function.Embedding.sumSet_range {s t : Set α} (h : Disjoint s t) :
    range (Function.Embedding.sumSet h) = s ∪ t := by
  /-
    α : Type u_1
    s t : Set α
    h : Disjoint s t
    ⊢ Eq (Set.range ⇑(Function.Embedding.sumSet h)) (Union.union s t)
  -/
  simp [Set.ext_iff]
  /-
    🎉 no goals
  -/


/-- For an indexed family `s : ι → Set α` of disjoint sets,
the natural injection from the sigma-type `(i : ι) × ↑(s i)` to `α`. -/
@[simps] def Function.Embedding.sigmaSet {s : ι → Set α} (h : Pairwise (Disjoint on s)) :
    (i : ι) × s i ↪ α where
  toFun x := x.2.1
  inj' := by
    /-
      α : Type u_1
      ι : Type u_2
      s✝ t r : Set α
      s : ι → Set α
      h : Pairwise (Function.onFun Disjoint s)
      ⊢ Function.Injective fun x => ↑x.snd
    -/
    rintro ⟨i, x, hx⟩ ⟨j, -, hx'⟩ rfl
    /-
      case mk.mk.mk.mk
      α : Type u_1
      ι : Type u_2
      s✝ t r : Set α
      s : ι → Set α
      h : Pairwise (Function.onFun Disjoint s)
      i : ι
      x : α
      hx : Membership.mem (s i) x
      j : ι
      hx' : Membership.mem (s j) ((fun x => ↑x.snd) ⟨i, ⟨x, hx⟩⟩)
      ⊢ Eq ⟨i, ⟨x, hx⟩⟩ ⟨j, ⟨(fun x => ↑x.snd) ⟨i, ⟨x, hx⟩⟩, hx'⟩⟩
    -/
    obtain rfl : i = j := h.eq (not_disjoint_iff.2 ⟨_, hx, hx'⟩)
    /-
      case mk.mk.mk.mk
      α : Type u_1
      ι : Type u_2
      s✝ t r : Set α
      s : ι → Set α
      h : Pairwise (Function.onFun Disjoint s)
      i : ι
      x : α
      hx : Membership.mem (s i) x
      hx' : Membership.mem (s i) ((fun x => ↑x.snd) ⟨i, ⟨x, hx⟩⟩)
      ⊢ Eq ⟨i, ⟨x, hx⟩⟩ ⟨i, ⟨(fun x => ↑x.snd) ⟨i, ⟨x, hx⟩⟩, hx'⟩⟩
    -/
    rfl
    /-
      🎉 no goals
    -/


@[norm_cast] lemma Function.Embedding.coe_sigmaSet {s : ι → Set α} (h) :
    (Function.Embedding.sigmaSet h : ((i : ι) × s i) → α) = fun x ↦ x.2.1 := rfl


@[simp] theorem Function.Embedding.sigmaSet_preimage {s : ι → Set α}
    (h : Pairwise (Disjoint on s)) (i : ι) (r : Set α) :
    Sigma.mk i ⁻¹' (Function.Embedding.sigmaSet h ⁻¹' r) = r ∩ s i := by
  /-
    α : Type u_1
    ι : Type u_2
    s : ι → Set α
    h : Pairwise (Function.onFun Disjoint s)
    i : ι
    r : Set α
    ⊢ Eq (Set.image Subtype.val (Set.preimage (Sigma.mk i) (Set.preimage (⇑(Functi …
  -/
  simp [Set.ext_iff]
  /-
    🎉 no goals
  -/


@[simp] theorem Function.Embedding.sigmaSet_range {s : ι → Set α}
    (h : Pairwise (Disjoint on s)) : Set.range (Function.Embedding.sigmaSet h) = ⋃ i, s i := by
  /-
    α : Type u_1
    ι : Type u_2
    s : ι → Set α
    h : Pairwise (Function.onFun Disjoint s)
    ⊢ Eq (Set.range ⇑(Function.Embedding.sigmaSet h)) (Set.iUnion fun i => s i)
  -/
  simp [Set.ext_iff]
  /-
    🎉 no goals
  -/


