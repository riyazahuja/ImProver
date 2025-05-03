/-- The intent closure of `s : Set α` along a relation `r : α → β → Prop` is the set of all elements
which `r` relates to all elements of `s`. -/
def intentClosure (s : Set α) : Set β :=
  { b | ∀ ⦃a⦄, a ∈ s → r a b }


/-- The extent closure of `t : Set β` along a relation `r : α → β → Prop` is the set of all elements
which `r` relates to all elements of `t`. -/
def extentClosure (t : Set β) : Set α :=
  { a | ∀ ⦃b⦄, b ∈ t → r a b }


theorem subset_intentClosure_iff_subset_extentClosure :
    t ⊆ intentClosure r s ↔ s ⊆ extentClosure r t :=
  ⟨fun h _ ha _ hb => h hb ha, fun h _ hb _ ha => h ha hb⟩


theorem gc_intentClosure_extentClosure :
    GaloisConnection (toDual ∘ intentClosure r) (extentClosure r ∘ ofDual) := fun _ _ =>
  subset_intentClosure_iff_subset_extentClosure


theorem intentClosure_swap (t : Set β) : intentClosure (swap r) t = extentClosure r t :=
  rfl


theorem extentClosure_swap (s : Set α) : extentClosure (swap r) s = intentClosure r s :=
  rfl


@[simp]
theorem intentClosure_empty : intentClosure r ∅ = univ :=
  eq_univ_of_forall fun _ _ => False.elim


@[simp]
theorem extentClosure_empty : extentClosure r ∅ = univ :=
  intentClosure_empty _


@[simp]
theorem intentClosure_union (s₁ s₂ : Set α) :
    intentClosure r (s₁ ∪ s₂) = intentClosure r s₁ ∩ intentClosure r s₂ :=
  Set.ext fun _ => forall₂_or_left


@[simp]
theorem extentClosure_union (t₁ t₂ : Set β) :
    extentClosure r (t₁ ∪ t₂) = extentClosure r t₁ ∩ extentClosure r t₂ :=
  intentClosure_union _ _ _


@[simp]
theorem intentClosure_iUnion (f : ι → Set α) :
    intentClosure r (⋃ i, f i) = ⋂ i, intentClosure r (f i) :=
  (gc_intentClosure_extentClosure r).l_iSup


@[simp]
theorem extentClosure_iUnion (f : ι → Set β) :
    extentClosure r (⋃ i, f i) = ⋂ i, extentClosure r (f i) :=
  intentClosure_iUnion _ _


theorem intentClosure_iUnion₂ (f : ∀ i, κ i → Set α) :
    intentClosure r (⋃ (i) (j), f i j) = ⋂ (i) (j), intentClosure r (f i j) :=
  (gc_intentClosure_extentClosure r).l_iSup₂


theorem extentClosure_iUnion₂ (f : ∀ i, κ i → Set β) :
    extentClosure r (⋃ (i) (j), f i j) = ⋂ (i) (j), extentClosure r (f i j) :=
  intentClosure_iUnion₂ _ _


theorem subset_extentClosure_intentClosure (s : Set α) :
    s ⊆ extentClosure r (intentClosure r s) :=
  (gc_intentClosure_extentClosure r).le_u_l _


theorem subset_intentClosure_extentClosure (t : Set β) :
    t ⊆ intentClosure r (extentClosure r t) :=
  subset_extentClosure_intentClosure _ t


@[simp]
theorem intentClosure_extentClosure_intentClosure (s : Set α) :
    intentClosure r (extentClosure r <| intentClosure r s) = intentClosure r s :=
  (gc_intentClosure_extentClosure r).l_u_l_eq_l _


@[simp]
theorem extentClosure_intentClosure_extentClosure (t : Set β) :
    extentClosure r (intentClosure r <| extentClosure r t) = extentClosure r t :=
  intentClosure_extentClosure_intentClosure _ t


theorem intentClosure_anti : Antitone (intentClosure r) :=
  (gc_intentClosure_extentClosure r).monotone_l


theorem extentClosure_anti : Antitone (extentClosure r) :=
  intentClosure_anti _


/-- The formal concepts of a relation. A concept of `r : α → β → Prop` is a pair of sets `s`, `t`
such that `s` is the set of all elements that are `r`-related to all of `t` and `t` is the set of
all elements that are `r`-related to all of `s`. -/
structure Concept extends Set α × Set β where
  /-- The axiom of a `Concept` stating that the closure of the first set is the second set. -/
  closure_fst : intentClosure r fst = snd
  /-- The axiom of a `Concept` stating that the closure of the second set is the first set. -/
  closure_snd : extentClosure r snd = fst


@[ext]
theorem ext (h : c.fst = d.fst) : c = d := by
  /-
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    c d : Concept α β r
    h : Eq c.toProd.1 d.toProd.1
    ⊢ Eq c d
  -/
  obtain ⟨⟨s₁, t₁⟩, h₁, _⟩ := c
  /-
    case mk.mk
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    d : Concept α β r
    s₁ : Set α
    t₁ : Set β
    h₁ : Eq (intentClosure r { fst := s₁, snd := t₁ }.1) { fst := s₁, snd := t₁ }.2
    closure_snd✝ : Eq (extentClosure r { fst := s₁, snd := t₁ }.2) { fst := s₁, sn …
    h : Eq { fst := s₁, snd := t₁, closure_fst := h₁, closure_snd := closure_snd✝  …
    ⊢ Eq { fst := s₁, snd := t₁, closure_fst := h₁, closure_snd := closure_snd✝ } d
  -/
  obtain ⟨⟨s₂, t₂⟩, h₂, _⟩ := d
  /-
    case mk.mk.mk.mk
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    s₁ : Set α
    t₁ : Set β
    h₁ : Eq (intentClosure r { fst := s₁, snd := t₁ }.1) { fst := s₁, snd := t₁ }.2
    closure_snd✝¹ : Eq (extentClosure r { fst := s₁, snd := t₁ }.2) { fst := s₁, s …
    s₂ : Set α
    t₂ : Set β
    h₂ : Eq (intentClosure r { fst := s₂, snd := t₂ }.1) { fst := s₂, snd := t₂ }.2
    closure_snd✝ : Eq (extentClosure r { fst := s₂, snd := t₂ }.2) { fst := s₂, sn …
    h : Eq { fst := s₁, snd := t₁, closure_fst := h₁, closure_snd := closure_snd✝¹ …
    ⊢ Eq { fst := s₁, snd := t₁, closure_fst := h₁, closure_snd := closure_snd✝¹ } …
  -/
  dsimp at h₁ h₂ h
  /-
    case mk.mk.mk.mk
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    s₁ : Set α
    t₁ : Set β
    h₁ : Eq (intentClosure r s₁) t₁
    closure_snd✝¹ : Eq (extentClosure r { fst := s₁, snd := t₁ }.2) { fst := s₁, s …
    s₂ : Set α
    t₂ : Set β
    h₂ : Eq (intentClosure r s₂) t₂
    closure_snd✝ : Eq (extentClosure r { fst := s₂, snd := t₂ }.2) { fst := s₂, sn …
    h : Eq s₁ s₂
    ⊢ Eq { fst := s₁, snd := t₁, closure_fst := h₁, closure_snd := closure_snd✝¹ } …
  -/
  substs h h₁ h₂
  /-
    case mk.mk.mk.mk
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    s₁ : Set α
    closure_snd✝¹ closure_snd✝ : Eq (extentClosure r { fst := s₁, snd := intentClo …
    ⊢ Eq { fst := s₁, snd := intentClosure r s₁, closure_fst := ⋯, closure_snd :=  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem ext' (h : c.snd = d.snd) : c = d := by
  /-
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    c d : Concept α β r
    h : Eq c.toProd.2 d.toProd.2
    ⊢ Eq c d
  -/
  obtain ⟨⟨s₁, t₁⟩, _, h₁⟩ := c
  /-
    case mk.mk
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    d : Concept α β r
    s₁ : Set α
    t₁ : Set β
    closure_fst✝ : Eq (intentClosure r { fst := s₁, snd := t₁ }.1) { fst := s₁, sn …
    h₁ : Eq (extentClosure r { fst := s₁, snd := t₁ }.2) { fst := s₁, snd := t₁ }.1
    h : Eq { fst := s₁, snd := t₁, closure_fst := closure_fst✝, closure_snd := h₁  …
    ⊢ Eq { fst := s₁, snd := t₁, closure_fst := closure_fst✝, closure_snd := h₁ } d
  -/
  obtain ⟨⟨s₂, t₂⟩, _, h₂⟩ := d
  /-
    case mk.mk.mk.mk
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    s₁ : Set α
    t₁ : Set β
    closure_fst✝¹ : Eq (intentClosure r { fst := s₁, snd := t₁ }.1) { fst := s₁, s …
    h₁ : Eq (extentClosure r { fst := s₁, snd := t₁ }.2) { fst := s₁, snd := t₁ }.1
    s₂ : Set α
    t₂ : Set β
    closure_fst✝ : Eq (intentClosure r { fst := s₂, snd := t₂ }.1) { fst := s₂, sn …
    h₂ : Eq (extentClosure r { fst := s₂, snd := t₂ }.2) { fst := s₂, snd := t₂ }.1
    h : Eq { fst := s₁, snd := t₁, closure_fst := closure_fst✝¹, closure_snd := h₁ …
    ⊢ Eq { fst := s₁, snd := t₁, closure_fst := closure_fst✝¹, closure_snd := h₁ } …
  -/
  dsimp at h₁ h₂ h
  /-
    case mk.mk.mk.mk
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    s₁ : Set α
    t₁ : Set β
    closure_fst✝¹ : Eq (intentClosure r { fst := s₁, snd := t₁ }.1) { fst := s₁, s …
    h₁ : Eq (extentClosure r t₁) s₁
    s₂ : Set α
    t₂ : Set β
    closure_fst✝ : Eq (intentClosure r { fst := s₂, snd := t₂ }.1) { fst := s₂, sn …
    h₂ : Eq (extentClosure r t₂) s₂
    h : Eq t₁ t₂
    ⊢ Eq { fst := s₁, snd := t₁, closure_fst := closure_fst✝¹, closure_snd := h₁ } …
  -/
  substs h h₁ h₂
  /-
    case mk.mk.mk.mk
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    t₁ : Set β
    closure_fst✝¹ closure_fst✝ : Eq (intentClosure r { fst := extentClosure r t₁,  …
    ⊢ Eq { fst := extentClosure r t₁, snd := t₁, closure_fst := closure_fst✝¹, clo …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem fst_injective : Injective fun c : Concept α β r => c.fst := fun _ _ => ext


theorem snd_injective : Injective fun c : Concept α β r => c.snd := fun _ _ => ext'


instance instSupConcept : Max (Concept α β r) :=
  ⟨fun c d =>
    { fst := extentClosure r (c.snd ∩ d.snd)
      snd := c.snd ∩ d.snd
      closure_fst := by
        rw [← c.closure_fst, ← d.closure_fst, ← intentClosure_union,
          intentClosure_extentClosure_intentClosure]
      closure_snd := rfl }⟩


instance instInfConcept : Min (Concept α β r) :=
  ⟨fun c d =>
    { fst := c.fst ∩ d.fst
      snd := intentClosure r (c.fst ∩ d.fst)
      closure_fst := rfl
      closure_snd := by
        rw [← c.closure_snd, ← d.closure_snd, ← extentClosure_union,
          extentClosure_intentClosure_extentClosure] }⟩


instance instSemilatticeInfConcept : SemilatticeInf (Concept α β r) :=
  (fst_injective.semilatticeInf _) fun _ _ => rfl


@[simp]
theorem fst_subset_fst_iff : c.fst ⊆ d.fst ↔ c ≤ d :=
  Iff.rfl


@[simp]
theorem fst_ssubset_fst_iff : c.fst ⊂ d.fst ↔ c < d :=
  Iff.rfl


@[simp]
theorem snd_subset_snd_iff : c.snd ⊆ d.snd ↔ d ≤ c := by
  /-
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    c d : Concept α β r
    ⊢ Iff (HasSubset.Subset c.toProd.2 d.toProd.2) (LE.le d c)
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      r : α → β → Prop
      c d : Concept α β r
      h : HasSubset.Subset c.toProd.2 d.toProd.2
      ⊢ LE.le d c
    -/
  · rw [← fst_subset_fst_iff, ← c.closure_snd, ← d.closure_snd]
    /-
      case refine_1
      α : Type u_2
      β : Type u_3
      r : α → β → Prop
      c d : Concept α β r
      h : HasSubset.Subset c.toProd.2 d.toProd.2
      ⊢ HasSubset.Subset (extentClosure r d.toProd.2) (extentClosure r c.toProd.2)
    -/
    exact extentClosure_anti _ h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_2
      β : Type u_3
      r : α → β → Prop
      c d : Concept α β r
      h : LE.le d c
      ⊢ HasSubset.Subset c.toProd.2 d.toProd.2
    -/
  · rw [← c.closure_fst, ← d.closure_fst]
    /-
      case refine_2
      α : Type u_2
      β : Type u_3
      r : α → β → Prop
      c d : Concept α β r
      h : LE.le d c
      ⊢ HasSubset.Subset (intentClosure r c.toProd.1) (intentClosure r d.toProd.1)
    -/
    exact intentClosure_anti _ h
    /-
      🎉 no goals
    -/


@[simp]
theorem snd_ssubset_snd_iff : c.snd ⊂ d.snd ↔ d < c := by
  /-
    α : Type u_2
    β : Type u_3
    r : α → β → Prop
    c d : Concept α β r
    ⊢ Iff (HasSSubset.SSubset c.toProd.2 d.toProd.2) (LT.lt d c)
  -/
  rw [ssubset_iff_subset_not_subset, lt_iff_le_not_le, snd_subset_snd_iff, snd_subset_snd_iff]
  /-
    🎉 no goals
  -/


theorem strictMono_fst : StrictMono (Prod.fst ∘ toProd : Concept α β r → Set α) := fun _ _ =>
  fst_ssubset_fst_iff.2


theorem strictAnti_snd : StrictAnti (Prod.snd ∘ toProd : Concept α β r → Set β) := fun _ _ =>
  snd_ssubset_snd_iff.2


instance instLatticeConcept : Lattice (Concept α β r) :=
  { Concept.instSemilatticeInfConcept with
    sup := (· ⊔ ·)
    le_sup_left := fun _ _ => snd_subset_snd_iff.1 inter_subset_left
    le_sup_right := fun _ _ => snd_subset_snd_iff.1 inter_subset_right
    sup_le := fun c d e => by
      /-
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        κ : ι → Sort u_4
        r : α → β → Prop
        s : Set α
        t : Set β
        c✝ d✝ c d e : Concept α β r
        ⊢ LE.le c e → LE.le d e → LE.le ((fun x1 x2 => Max.max x1 x2) c d) e
      -/
      simp_rw [← snd_subset_snd_iff]
      /-
        ι : Sort u_1
        α : Type u_2
        β : Type u_3
        κ : ι → Sort u_4
        r : α → β → Prop
        s : Set α
        t : Set β
        c✝ d✝ c d e : Concept α β r
        ⊢ HasSubset.Subset e.toProd.2 c.toProd.2 → HasSubset.Subset e.toProd.2 d.toPro …
      -/
      exact subset_inter }
      /-
        🎉 no goals
      -/


instance instBoundedOrderConcept : BoundedOrder (Concept α β r) where
  top := ⟨⟨univ, intentClosure r univ⟩, rfl, eq_univ_of_forall fun _ _ hb => hb trivial⟩
  le_top _ := subset_univ _
  bot := ⟨⟨extentClosure r univ, univ⟩, eq_univ_of_forall fun _ _ ha => ha trivial, rfl⟩
  bot_le _ := snd_subset_snd_iff.1 <| subset_univ _


instance : SupSet (Concept α β r) :=
  ⟨fun S =>
    { fst := extentClosure r (⋂ c ∈ S, (c : Concept _ _ _).snd)
      snd := ⋂ c ∈ S, (c : Concept _ _ _).snd
      closure_fst := by
        simp_rw [← closure_fst, ← intentClosure_iUnion₂,
          intentClosure_extentClosure_intentClosure]
      closure_snd := rfl }⟩


instance : InfSet (Concept α β r) :=
  ⟨fun S =>
    { fst := ⋂ c ∈ S, (c : Concept _ _ _).fst
      snd := intentClosure r (⋂ c ∈ S, (c : Concept _ _ _).fst)
      closure_fst := rfl
      closure_snd := by
        simp_rw [← closure_snd, ← extentClosure_iUnion₂,
          extentClosure_intentClosure_extentClosure] }⟩


instance : CompleteLattice (Concept α β r) :=
  { Concept.instLatticeConcept,
    Concept.instBoundedOrderConcept with
    sup := Concept.instSupConcept.max
    le_sSup := fun _ _ hc => snd_subset_snd_iff.1 <| biInter_subset_of_mem hc
    sSup_le := fun _ _ hc =>
      snd_subset_snd_iff.1 <| subset_iInter₂ fun d hd => snd_subset_snd_iff.2 <| hc d hd
    inf := Concept.instInfConcept.min
    sInf_le := fun _ _ => biInter_subset_of_mem
    le_sInf := fun _ _ => subset_iInter₂ }


@[simp]
theorem top_fst : (⊤ : Concept α β r).fst = univ :=
  rfl


@[simp]
theorem top_snd : (⊤ : Concept α β r).snd = intentClosure r univ :=
  rfl


@[simp]
theorem bot_fst : (⊥ : Concept α β r).fst = extentClosure r univ :=
  rfl


@[simp]
theorem bot_snd : (⊥ : Concept α β r).snd = univ :=
  rfl


@[simp]
theorem sup_fst (c d : Concept α β r) : (c ⊔ d).fst = extentClosure r (c.snd ∩ d.snd) :=
  rfl


@[simp]
theorem sup_snd (c d : Concept α β r) : (c ⊔ d).snd = c.snd ∩ d.snd :=
  rfl


@[simp]
theorem inf_fst (c d : Concept α β r) : (c ⊓ d).fst = c.fst ∩ d.fst :=
  rfl


@[simp]
theorem inf_snd (c d : Concept α β r) : (c ⊓ d).snd = intentClosure r (c.fst ∩ d.fst) :=
  rfl


@[simp]
theorem sSup_fst (S : Set (Concept α β r)) :
    (sSup S).fst = extentClosure r (⋂ c ∈ S, (c : Concept _ _ _).snd) :=
  rfl


@[simp]
theorem sSup_snd (S : Set (Concept α β r)) : (sSup S).snd = ⋂ c ∈ S, (c : Concept _ _ _).snd :=
  rfl


@[simp]
theorem sInf_fst (S : Set (Concept α β r)) : (sInf S).fst = ⋂ c ∈ S, (c : Concept _ _ _).fst :=
  rfl


@[simp]
theorem sInf_snd (S : Set (Concept α β r)) :
    (sInf S).snd = intentClosure r (⋂ c ∈ S, (c : Concept _ _ _).fst) :=
  rfl


instance : Inhabited (Concept α β r) :=
  ⟨⊥⟩


/-- Swap the sets of a concept to make it a concept of the dual context. -/
@[simps]
def swap (c : Concept α β r) : Concept β α (swap r) :=
  ⟨c.toProd.swap, c.closure_snd, c.closure_fst⟩


@[simp]
theorem swap_swap (c : Concept α β r) : c.swap.swap = c :=
  ext rfl


@[simp]
theorem swap_le_swap_iff : c.swap ≤ d.swap ↔ d ≤ c :=
  snd_subset_snd_iff


@[simp]
theorem swap_lt_swap_iff : c.swap < d.swap ↔ d < c :=
  snd_ssubset_snd_iff


/-- The dual of a concept lattice is isomorphic to the concept lattice of the dual context. -/
@[simps]
def swapEquiv : (Concept α β r)ᵒᵈ ≃o Concept β α (Function.swap r) where
  toFun := swap ∘ ofDual
  invFun := toDual ∘ swap
  left_inv := swap_swap
  right_inv := swap_swap
  map_rel_iff' := swap_le_swap_iff


