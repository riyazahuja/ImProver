/-- Auxiliary definition for the proof of the shrinking lemma. A partial refinement of a covering
`⋃ i, u i` of a set `s` is a map `v : ι → Set X` and a set `carrier : Set ι` such that

* `s ⊆ ⋃ i, v i`;
* all `v i` are open;
* if `i ∈ carrier v`, then `closure (v i) ⊆ u i`;
* if `i ∉ carrier`, then `v i = u i`.

This type is equipped with the following partial order: `v ≤ v'` if `v.carrier ⊆ v'.carrier`
and `v i = v' i` for `i ∈ v.carrier`. We will use Zorn's lemma to prove that this type has
a maximal element, then show that the maximal element must have `carrier = univ`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet. @[nolint has_nonempty_instance]
@[ext] structure PartialRefinement (u : ι → Set X) (s : Set X) (p : Set X → Prop) where
  /-- A family of sets that form a partial refinement of `u`. -/
  toFun : ι → Set X
  /-- The set of indexes `i` such that `i`-th set is already shrunk. -/
  carrier : Set ι
  /-- Each set from the partially refined family is open. -/
  protected isOpen : ∀ i, IsOpen (toFun i)
  /-- The partially refined family still covers the set. -/
  subset_iUnion : s ⊆ ⋃ i, toFun i
  /-- For each `i ∈ carrier`, the original set includes the closure of the refined set. -/
  closure_subset : ∀ {i}, i ∈ carrier → closure (toFun i) ⊆ u i
  /-- For each `i ∈ carrier`, the refined set satisfies `p`. -/
  pred_of_mem {i} (hi : i ∈ carrier) : p (toFun i)
  /-- Sets that correspond to `i ∉ carrier` are not modified. -/
  apply_eq : ∀ {i}, i ∉ carrier → toFun i = u i


instance : CoeFun (PartialRefinement u s p) fun _ => ι → Set X := ⟨toFun⟩


protected theorem subset (v : PartialRefinement u s p) (i : ι) : v i ⊆ u i := by
  classical
  exact if h : i ∈ v.carrier then subset_closure.trans (v.closure_subset h) else (v.apply_eq h).le


open Classical in
instance : PartialOrder (PartialRefinement u s p) where
  le v₁ v₂ := v₁.carrier ⊆ v₂.carrier ∧ ∀ i ∈ v₁.carrier, v₁ i = v₂ i
  le_refl _ := ⟨Subset.refl _, fun _ _ => rfl⟩
  le_trans _ _ _ h₁₂ h₂₃ :=
    ⟨Subset.trans h₁₂.1 h₂₃.1, fun i hi => (h₁₂.2 i hi).trans (h₂₃.2 i <| h₁₂.1 hi)⟩
  le_antisymm v₁ v₂ h₁₂ h₂₁ :=
    have hc : v₁.carrier = v₂.carrier := Subset.antisymm h₁₂.1 h₂₁.1
    PartialRefinement.ext
      (funext fun x =>
        if hx : x ∈ v₁.carrier then h₁₂.2 _ hx
        else (v₁.apply_eq hx).trans (Eq.symm <| v₂.apply_eq <| hc ▸ hx))
      hc


/-- If two partial refinements `v₁`, `v₂` belong to a chain (hence, they are comparable)
and `i` belongs to the carriers of both partial refinements, then `v₁ i = v₂ i`. -/
theorem apply_eq_of_chain {c : Set (PartialRefinement u s p)} (hc : IsChain (· ≤ ·) c) {v₁ v₂}
    (h₁ : v₁ ∈ c) (h₂ : v₂ ∈ c) {i} (hi₁ : i ∈ v₁.carrier) (hi₂ : i ∈ v₂.carrier) :
    v₁ i = v₂ i :=
  (hc.total h₁ h₂).elim (fun hle => hle.2 _ hi₁) (fun hle => (hle.2 _ hi₂).symm)


/-- The carrier of the least upper bound of a non-empty chain of partial refinements is the union of
their carriers. -/
def chainSupCarrier (c : Set (PartialRefinement u s p)) : Set ι :=
  ⋃ v ∈ c, carrier v


open Classical in
/-- Choice of an element of a nonempty chain of partial refinements. If `i` belongs to one of
`carrier v`, `v ∈ c`, then `find c ne i` is one of these partial refinements. -/
def find (c : Set (PartialRefinement u s p)) (ne : c.Nonempty) (i : ι) : PartialRefinement u s p :=
  if hi : ∃ v ∈ c, i ∈ carrier v then hi.choose else ne.some


theorem find_mem {c : Set (PartialRefinement u s p)} (i : ι) (ne : c.Nonempty) :
    find c ne i ∈ c := by
  /-
    ι : Type u_1
    X : Type u_2
    inst✝ : TopologicalSpace X
    u : ι → Set X
    s : Set X
    p : Set X → Prop
    c : Set (ShrinkingLemma.PartialRefinement u s p)
    i : ι
    ne : c.Nonempty
    ⊢ Membership.mem c (ShrinkingLemma.PartialRefinement.find c ne i)
  -/
  rw [find]
  /-
    ι : Type u_1
    X : Type u_2
    inst✝ : TopologicalSpace X
    u : ι → Set X
    s : Set X
    p : Set X → Prop
    c : Set (ShrinkingLemma.PartialRefinement u s p)
    i : ι
    ne : c.Nonempty
    ⊢ Membership.mem c (dite (Exists fun v => And (Membership.mem c v) (Membership …
  -/
  split_ifs with h
  /-
    case pos
    ι : Type u_1
    X : Type u_2
    inst✝ : TopologicalSpace X
    u : ι → Set X
    s : Set X
    p : Set X → Prop
    c : Set (ShrinkingLemma.PartialRefinement u s p)
    i : ι
    ne : c.Nonempty
    h : Exists fun v => And (Membership.mem c v) (Membership.mem v.carrier i)
    ⊢ Membership.mem c h.choose
  -/
  exacts [h.choose_spec.1, ne.some_mem]
  /-
    🎉 no goals
  -/


theorem mem_find_carrier_iff {c : Set (PartialRefinement u s p)} {i : ι} (ne : c.Nonempty) :
    i ∈ (find c ne i).carrier ↔ i ∈ chainSupCarrier c := by
  /-
    ι : Type u_1
    X : Type u_2
    inst✝ : TopologicalSpace X
    u : ι → Set X
    s : Set X
    p : Set X → Prop
    c : Set (ShrinkingLemma.PartialRefinement u s p)
    i : ι
    ne : c.Nonempty
    ⊢ Iff (Membership.mem (ShrinkingLemma.PartialRefinement.find c ne i).carrier i …
  -/
  rw [find]
  /-
    ι : Type u_1
    X : Type u_2
    inst✝ : TopologicalSpace X
    u : ι → Set X
    s : Set X
    p : Set X → Prop
    c : Set (ShrinkingLemma.PartialRefinement u s p)
    i : ι
    ne : c.Nonempty
    ⊢ Iff (Membership.mem (dite (Exists fun v => And (Membership.mem c v) (Members …
  -/
  split_ifs with h
    /-
      case pos
      ι : Type u_1
      X : Type u_2
      inst✝ : TopologicalSpace X
      u : ι → Set X
      s : Set X
      p : Set X → Prop
      c : Set (ShrinkingLemma.PartialRefinement u s p)
      i : ι
      ne : c.Nonempty
      h : Exists fun v => And (Membership.mem c v) (Membership.mem v.carrier i)
      ⊢ Iff (Membership.mem h.choose.carrier i) (Membership.mem (ShrinkingLemma.Part …
    -/
  · have := h.choose_spec
    /-
      case pos
      ι : Type u_1
      X : Type u_2
      inst✝ : TopologicalSpace X
      u : ι → Set X
      s : Set X
      p : Set X → Prop
      c : Set (ShrinkingLemma.PartialRefinement u s p)
      i : ι
      ne : c.Nonempty
      h : Exists fun v => And (Membership.mem c v) (Membership.mem v.carrier i)
      this : And (Membership.mem c h.choose) (Membership.mem h.choose.carrier i)
      ⊢ Iff (Membership.mem h.choose.carrier i) (Membership.mem (ShrinkingLemma.Part …
    -/
    exact iff_of_true this.2 (mem_iUnion₂.2 ⟨_, this.1, this.2⟩)
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      X : Type u_2
      inst✝ : TopologicalSpace X
      u : ι → Set X
      s : Set X
      p : Set X → Prop
      c : Set (ShrinkingLemma.PartialRefinement u s p)
      i : ι
      ne : c.Nonempty
      h : Not (Exists fun v => And (Membership.mem c v) (Membership.mem v.carrier i))
      ⊢ Iff (Membership.mem ne.some.carrier i) (Membership.mem (ShrinkingLemma.Parti …
    -/
  · push_neg at h
    /-
      case neg
      ι : Type u_1
      X : Type u_2
      inst✝ : TopologicalSpace X
      u : ι → Set X
      s : Set X
      p : Set X → Prop
      c : Set (ShrinkingLemma.PartialRefinement u s p)
      i : ι
      ne : c.Nonempty
      h : ∀ (v : ShrinkingLemma.PartialRefinement u s p), Membership.mem c v → Not ( …
      ⊢ Iff (Membership.mem ne.some.carrier i) (Membership.mem (ShrinkingLemma.Parti …
    -/
    refine iff_of_false (h _ ne.some_mem) ?_
    /-
      case neg
      ι : Type u_1
      X : Type u_2
      inst✝ : TopologicalSpace X
      u : ι → Set X
      s : Set X
      p : Set X → Prop
      c : Set (ShrinkingLemma.PartialRefinement u s p)
      i : ι
      ne : c.Nonempty
      h : ∀ (v : ShrinkingLemma.PartialRefinement u s p), Membership.mem c v → Not ( …
      ⊢ Not (Membership.mem (ShrinkingLemma.PartialRefinement.chainSupCarrier c) i)
    -/
    simpa only [chainSupCarrier, mem_iUnion₂, not_exists]
    /-
      🎉 no goals
    -/


theorem find_apply_of_mem {c : Set (PartialRefinement u s p)} (hc : IsChain (· ≤ ·) c)
    (ne : c.Nonempty) {i v} (hv : v ∈ c) (hi : i ∈ carrier v) : find c ne i i = v i :=
  apply_eq_of_chain hc (find_mem _ _) hv ((mem_find_carrier_iff _).2 <| mem_iUnion₂.2 ⟨v, hv, hi⟩)
    hi


/-- Least upper bound of a nonempty chain of partial refinements. -/
def chainSup (c : Set (PartialRefinement u s p)) (hc : IsChain (· ≤ ·) c) (ne : c.Nonempty)
    (hfin : ∀ x ∈ s, { i | x ∈ u i }.Finite) (hU : s ⊆ ⋃ i, u i) : PartialRefinement u s p where
  toFun i := find c ne i i
  carrier := chainSupCarrier c
  isOpen i := (find _ _ _).isOpen i
  subset_iUnion x hxs := mem_iUnion.2 <| by
    /-
      ι : Type u_1
      X : Type u_2
      inst✝ : TopologicalSpace X
      u : ι → Set X
      s : Set X
      p : Set X → Prop
      c : Set (ShrinkingLemma.PartialRefinement u s p)
      hc : IsChain (fun x1 x2 => LE.le x1 x2) c
      ne : c.Nonempty
      hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
      hU : HasSubset.Subset s (Set.iUnion fun i => u i)
      x : X
      hxs : Membership.mem s x
      ⊢ Exists fun i => Membership.mem ((ShrinkingLemma.PartialRefinement.find c ne  …
    -/
    rcases em (∃ i, i ∉ chainSupCarrier c ∧ x ∈ u i) with (⟨i, hi, hxi⟩ | hx)
      /-
        case inl.intro.intro
        ι : Type u_1
        X : Type u_2
        inst✝ : TopologicalSpace X
        u : ι → Set X
        s : Set X
        p : Set X → Prop
        c : Set (ShrinkingLemma.PartialRefinement u s p)
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        ne : c.Nonempty
        hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
        hU : HasSubset.Subset s (Set.iUnion fun i => u i)
        x : X
        hxs : Membership.mem s x
        i : ι
        hi : Not (Membership.mem (ShrinkingLemma.PartialRefinement.chainSupCarrier c) i)
        hxi : Membership.mem (u i) x
        ⊢ Exists fun i => Membership.mem ((ShrinkingLemma.PartialRefinement.find c ne  …
      -/
    · use i
      /-
        case h
        ι : Type u_1
        X : Type u_2
        inst✝ : TopologicalSpace X
        u : ι → Set X
        s : Set X
        p : Set X → Prop
        c : Set (ShrinkingLemma.PartialRefinement u s p)
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        ne : c.Nonempty
        hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
        hU : HasSubset.Subset s (Set.iUnion fun i => u i)
        x : X
        hxs : Membership.mem s x
        i : ι
        hi : Not (Membership.mem (ShrinkingLemma.PartialRefinement.chainSupCarrier c) i)
        hxi : Membership.mem (u i) x
        ⊢ Membership.mem ((ShrinkingLemma.PartialRefinement.find c ne i).toFun i) x
      -/
      simpa only [(find c ne i).apply_eq (mt (mem_find_carrier_iff _).1 hi)]
      /-
        🎉 no goals
      -/
      /-
        case inr
        ι : Type u_1
        X : Type u_2
        inst✝ : TopologicalSpace X
        u : ι → Set X
        s : Set X
        p : Set X → Prop
        c : Set (ShrinkingLemma.PartialRefinement u s p)
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        ne : c.Nonempty
        hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
        hU : HasSubset.Subset s (Set.iUnion fun i => u i)
        x : X
        hxs : Membership.mem s x
        hx : Not (Exists fun i => And (Not (Membership.mem (ShrinkingLemma.PartialRefi …
        ⊢ Exists fun i => Membership.mem ((ShrinkingLemma.PartialRefinement.find c ne  …
      -/
    · simp_rw [not_exists, not_and, not_imp_not, chainSupCarrier, mem_iUnion₂] at hx
      /-
        case inr
        ι : Type u_1
        X : Type u_2
        inst✝ : TopologicalSpace X
        u : ι → Set X
        s : Set X
        p : Set X → Prop
        c : Set (ShrinkingLemma.PartialRefinement u s p)
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        ne : c.Nonempty
        hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
        hU : HasSubset.Subset s (Set.iUnion fun i => u i)
        x : X
        hxs : Membership.mem s x
        hx : ∀ (x_1 : ι), Membership.mem (u x_1) x → Exists fun i => Exists fun j => M …
        ⊢ Exists fun i => Membership.mem ((ShrinkingLemma.PartialRefinement.find c ne  …
      -/
      haveI : Nonempty (PartialRefinement u s p) := ⟨ne.some⟩
      /-
        case inr
        ι : Type u_1
        X : Type u_2
        inst✝ : TopologicalSpace X
        u : ι → Set X
        s : Set X
        p : Set X → Prop
        c : Set (ShrinkingLemma.PartialRefinement u s p)
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        ne : c.Nonempty
        hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
        hU : HasSubset.Subset s (Set.iUnion fun i => u i)
        x : X
        hxs : Membership.mem s x
        hx : ∀ (x_1 : ι), Membership.mem (u x_1) x → Exists fun i => Exists fun j => M …
        this : Nonempty (ShrinkingLemma.PartialRefinement u s p)
        ⊢ Exists fun i => Membership.mem ((ShrinkingLemma.PartialRefinement.find c ne  …
      -/
      choose! v hvc hiv using hx
      rcases (hfin x hxs).exists_maximal_wrt v _ (mem_iUnion.1 (hU hxs)) with
        ⟨i, hxi : x ∈ u i, hmax : ∀ j, x ∈ u j → v i ≤ v j → v i = v j⟩
      /-
        case inr.intro.intro
        ι : Type u_1
        X : Type u_2
        inst✝ : TopologicalSpace X
        u : ι → Set X
        s : Set X
        p : Set X → Prop
        c : Set (ShrinkingLemma.PartialRefinement u s p)
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        ne : c.Nonempty
        hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
        hU : HasSubset.Subset s (Set.iUnion fun i => u i)
        x : X
        hxs : Membership.mem s x
        this : Nonempty (ShrinkingLemma.PartialRefinement u s p)
        v : ι → ShrinkingLemma.PartialRefinement u s p
        hvc : ∀ (x_1 : ι), Membership.mem (u x_1) x → Membership.mem c (v x_1)
        hiv : ∀ (x_1 : ι), Membership.mem (u x_1) x → Membership.mem (v x_1).carrier x_1
        i : ι
        hxi : Membership.mem (u i) x
        hmax : ∀ (j : ι), Membership.mem (u j) x → LE.le (v i) (v j) → Eq (v i) (v j)
        ⊢ Exists fun i => Membership.mem ((ShrinkingLemma.PartialRefinement.find c ne  …
      -/
      rcases mem_iUnion.1 ((v i).subset_iUnion hxs) with ⟨j, hj⟩
      /-
        case inr.intro.intro.intro
        ι : Type u_1
        X : Type u_2
        inst✝ : TopologicalSpace X
        u : ι → Set X
        s : Set X
        p : Set X → Prop
        c : Set (ShrinkingLemma.PartialRefinement u s p)
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        ne : c.Nonempty
        hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
        hU : HasSubset.Subset s (Set.iUnion fun i => u i)
        x : X
        hxs : Membership.mem s x
        this : Nonempty (ShrinkingLemma.PartialRefinement u s p)
        v : ι → ShrinkingLemma.PartialRefinement u s p
        hvc : ∀ (x_1 : ι), Membership.mem (u x_1) x → Membership.mem c (v x_1)
        hiv : ∀ (x_1 : ι), Membership.mem (u x_1) x → Membership.mem (v x_1).carrier x_1
        i : ι
        hxi : Membership.mem (u i) x
        hmax : ∀ (j : ι), Membership.mem (u j) x → LE.le (v i) (v j) → Eq (v i) (v j)
        j : ι
        hj : Membership.mem ((v i).toFun j) x
        ⊢ Exists fun i => Membership.mem ((ShrinkingLemma.PartialRefinement.find c ne  …
      -/
      use j
      /-
        case h
        ι : Type u_1
        X : Type u_2
        inst✝ : TopologicalSpace X
        u : ι → Set X
        s : Set X
        p : Set X → Prop
        c : Set (ShrinkingLemma.PartialRefinement u s p)
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        ne : c.Nonempty
        hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
        hU : HasSubset.Subset s (Set.iUnion fun i => u i)
        x : X
        hxs : Membership.mem s x
        this : Nonempty (ShrinkingLemma.PartialRefinement u s p)
        v : ι → ShrinkingLemma.PartialRefinement u s p
        hvc : ∀ (x_1 : ι), Membership.mem (u x_1) x → Membership.mem c (v x_1)
        hiv : ∀ (x_1 : ι), Membership.mem (u x_1) x → Membership.mem (v x_1).carrier x_1
        i : ι
        hxi : Membership.mem (u i) x
        hmax : ∀ (j : ι), Membership.mem (u j) x → LE.le (v i) (v j) → Eq (v i) (v j)
        j : ι
        hj : Membership.mem ((v i).toFun j) x
        ⊢ Membership.mem ((ShrinkingLemma.PartialRefinement.find c ne j).toFun j) x
      -/
      have hj' : x ∈ u j := (v i).subset _ hj
      /-
        case h
        ι : Type u_1
        X : Type u_2
        inst✝ : TopologicalSpace X
        u : ι → Set X
        s : Set X
        p : Set X → Prop
        c : Set (ShrinkingLemma.PartialRefinement u s p)
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        ne : c.Nonempty
        hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
        hU : HasSubset.Subset s (Set.iUnion fun i => u i)
        x : X
        hxs : Membership.mem s x
        this : Nonempty (ShrinkingLemma.PartialRefinement u s p)
        v : ι → ShrinkingLemma.PartialRefinement u s p
        hvc : ∀ (x_1 : ι), Membership.mem (u x_1) x → Membership.mem c (v x_1)
        hiv : ∀ (x_1 : ι), Membership.mem (u x_1) x → Membership.mem (v x_1).carrier x_1
        i : ι
        hxi : Membership.mem (u i) x
        hmax : ∀ (j : ι), Membership.mem (u j) x → LE.le (v i) (v j) → Eq (v i) (v j)
        j : ι
        hj : Membership.mem ((v i).toFun j) x
        hj' : Membership.mem (u j) x
        ⊢ Membership.mem ((ShrinkingLemma.PartialRefinement.find c ne j).toFun j) x
      -/
      have : v j ≤ v i := (hc.total (hvc _ hxi) (hvc _ hj')).elim (fun h => (hmax j hj' h).ge) id
      /-
        case h
        ι : Type u_1
        X : Type u_2
        inst✝ : TopologicalSpace X
        u : ι → Set X
        s : Set X
        p : Set X → Prop
        c : Set (ShrinkingLemma.PartialRefinement u s p)
        hc : IsChain (fun x1 x2 => LE.le x1 x2) c
        ne : c.Nonempty
        hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
        hU : HasSubset.Subset s (Set.iUnion fun i => u i)
        x : X
        hxs : Membership.mem s x
        this✝ : Nonempty (ShrinkingLemma.PartialRefinement u s p)
        v : ι → ShrinkingLemma.PartialRefinement u s p
        hvc : ∀ (x_1 : ι), Membership.mem (u x_1) x → Membership.mem c (v x_1)
        hiv : ∀ (x_1 : ι), Membership.mem (u x_1) x → Membership.mem (v x_1).carrier x_1
        i : ι
        hxi : Membership.mem (u i) x
        hmax : ∀ (j : ι), Membership.mem (u j) x → LE.le (v i) (v j) → Eq (v i) (v j)
        j : ι
        hj : Membership.mem ((v i).toFun j) x
        hj' : Membership.mem (u j) x
        this : LE.le (v j) (v i)
        ⊢ Membership.mem ((ShrinkingLemma.PartialRefinement.find c ne j).toFun j) x
      -/
      simpa only [find_apply_of_mem hc ne (hvc _ hxi) (this.1 <| hiv _ hj')]
      /-
        🎉 no goals
      -/
  closure_subset hi := (find c ne _).closure_subset ((mem_find_carrier_iff _).2 hi)
  pred_of_mem {i} hi := by
    /-
      ι : Type u_1
      X : Type u_2
      inst✝ : TopologicalSpace X
      u : ι → Set X
      s : Set X
      p : Set X → Prop
      c : Set (ShrinkingLemma.PartialRefinement u s p)
      hc : IsChain (fun x1 x2 => LE.le x1 x2) c
      ne : c.Nonempty
      hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
      hU : HasSubset.Subset s (Set.iUnion fun i => u i)
      i : ι
      hi : Membership.mem (ShrinkingLemma.PartialRefinement.chainSupCarrier c) i
      ⊢ p ((fun i => (ShrinkingLemma.PartialRefinement.find c ne i).toFun i) i)
    -/
    obtain ⟨v, hv⟩ := Set.mem_iUnion.mp hi
    /-
      case intro
      ι : Type u_1
      X : Type u_2
      inst✝ : TopologicalSpace X
      u : ι → Set X
      s : Set X
      p : Set X → Prop
      c : Set (ShrinkingLemma.PartialRefinement u s p)
      hc : IsChain (fun x1 x2 => LE.le x1 x2) c
      ne : c.Nonempty
      hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
      hU : HasSubset.Subset s (Set.iUnion fun i => u i)
      i : ι
      hi : Membership.mem (ShrinkingLemma.PartialRefinement.chainSupCarrier c) i
      v : ShrinkingLemma.PartialRefinement u s p
      hv : Membership.mem (Set.iUnion fun h => v.carrier) i
      ⊢ p ((fun i => (ShrinkingLemma.PartialRefinement.find c ne i).toFun i) i)
    -/
    simp only [mem_iUnion, exists_prop] at hv
    /-
      case intro
      ι : Type u_1
      X : Type u_2
      inst✝ : TopologicalSpace X
      u : ι → Set X
      s : Set X
      p : Set X → Prop
      c : Set (ShrinkingLemma.PartialRefinement u s p)
      hc : IsChain (fun x1 x2 => LE.le x1 x2) c
      ne : c.Nonempty
      hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
      hU : HasSubset.Subset s (Set.iUnion fun i => u i)
      i : ι
      hi : Membership.mem (ShrinkingLemma.PartialRefinement.chainSupCarrier c) i
      v : ShrinkingLemma.PartialRefinement u s p
      hv : And (Membership.mem c v) (Membership.mem v.carrier i)
      ⊢ p ((fun i => (ShrinkingLemma.PartialRefinement.find c ne i).toFun i) i)
    -/
    simp only
    /-
      case intro
      ι : Type u_1
      X : Type u_2
      inst✝ : TopologicalSpace X
      u : ι → Set X
      s : Set X
      p : Set X → Prop
      c : Set (ShrinkingLemma.PartialRefinement u s p)
      hc : IsChain (fun x1 x2 => LE.le x1 x2) c
      ne : c.Nonempty
      hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
      hU : HasSubset.Subset s (Set.iUnion fun i => u i)
      i : ι
      hi : Membership.mem (ShrinkingLemma.PartialRefinement.chainSupCarrier c) i
      v : ShrinkingLemma.PartialRefinement u s p
      hv : And (Membership.mem c v) (Membership.mem v.carrier i)
      ⊢ p ((ShrinkingLemma.PartialRefinement.find c ne i).toFun i)
    -/
    rw [find_apply_of_mem hc ne hv.1 hv.2]
    /-
      case intro
      ι : Type u_1
      X : Type u_2
      inst✝ : TopologicalSpace X
      u : ι → Set X
      s : Set X
      p : Set X → Prop
      c : Set (ShrinkingLemma.PartialRefinement u s p)
      hc : IsChain (fun x1 x2 => LE.le x1 x2) c
      ne : c.Nonempty
      hfin : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x) …
      hU : HasSubset.Subset s (Set.iUnion fun i => u i)
      i : ι
      hi : Membership.mem (ShrinkingLemma.PartialRefinement.chainSupCarrier c) i
      v : ShrinkingLemma.PartialRefinement u s p
      hv : And (Membership.mem c v) (Membership.mem v.carrier i)
      ⊢ p (v.toFun i)
    -/
    exact v.pred_of_mem hv.2
    /-
      🎉 no goals
    -/
  apply_eq hi := (find c ne _).apply_eq (mt (mem_find_carrier_iff _).1 hi)


/-- `chainSup hu c hc ne hfin hU` is an upper bound of the chain `c`. -/
theorem le_chainSup {c : Set (PartialRefinement u s p)} (hc : IsChain (· ≤ ·) c) (ne : c.Nonempty)
    (hfin : ∀ x ∈ s, { i | x ∈ u i }.Finite) (hU : s ⊆ ⋃ i, u i) {v} (hv : v ∈ c) :
    v ≤ chainSup c hc ne hfin hU :=
  ⟨fun _ hi => mem_biUnion hv hi, fun _ hi => (find_apply_of_mem hc _ hv hi).symm⟩


/-- If `s` is a closed set, `v` is a partial refinement, and `i` is an index such that
`i ∉ v.carrier`, then there exists a partial refinement that is strictly greater than `v`. -/
theorem exists_gt [NormalSpace X] (v : PartialRefinement u s ⊤) (hs : IsClosed s)
    (i : ι) (hi : i ∉ v.carrier) :
    ∃ v' : PartialRefinement u s ⊤, v < v' := by
  have I : (s ∩ ⋂ (j) (_ : j ≠ i), (v j)ᶜ) ⊆ v i := by
    simp only [subset_def, mem_inter_iff, mem_iInter, and_imp]
    intro x hxs H
    rcases mem_iUnion.1 (v.subset_iUnion hxs) with ⟨j, hj⟩
    exact (em (j = i)).elim (fun h => h ▸ hj) fun h => (H j h hj).elim
  have C : IsClosed (s ∩ ⋂ (j) (_ : j ≠ i), (v j)ᶜ) :=
    IsClosed.inter hs (isClosed_biInter fun _ _ => isClosed_compl_iff.2 <| v.isOpen _)
  /-
    ι : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝ : NormalSpace X
    v : ShrinkingLemma.PartialRefinement u s Top.top
    hs : IsClosed s
    i : ι
    hi : Not (Membership.mem v.carrier i)
    I : HasSubset.Subset (Inter.inter s (Set.iInter fun j => Set.iInter fun x => H …
    C : IsClosed (Inter.inter s (Set.iInter fun j => Set.iInter fun x => HasCompl. …
    ⊢ Exists fun v' => LT.lt v v'
  -/
  rcases normal_exists_closure_subset C (v.isOpen i) I with ⟨vi, ovi, hvi, cvi⟩
  classical
  refine ⟨⟨update v i vi, insert i v.carrier, ?_, ?_, ?_, ?_, ?_⟩, ?_, ?_⟩
  · intro j
    rcases eq_or_ne j i with (rfl| hne) <;> simp [*, v.isOpen]
  · refine fun x hx => mem_iUnion.2 ?_
    rcases em (∃ j ≠ i, x ∈ v j) with (⟨j, hji, hj⟩ | h)
    · use j
      rwa [update_of_ne hji]
    · push_neg at h
      use i
      rw [update_self]
      exact hvi ⟨hx, mem_biInter h⟩
  · rintro j (rfl | hj)
    · rwa [update_self, ← v.apply_eq hi]
    · rw [update_of_ne (ne_of_mem_of_not_mem hj hi)]
      exact v.closure_subset hj
  · exact fun _ => trivial
  · intro j hj
    rw [mem_insert_iff, not_or] at hj
    rw [update_of_ne hj.1, v.apply_eq hj.2]
  · refine ⟨subset_insert _ _, fun j hj => ?_⟩
    exact (update_of_ne (ne_of_mem_of_not_mem hj hi) _ _).symm
  · exact fun hle => hi (hle.1 <| mem_insert _ _)


/-- **Shrinking lemma**. A point-finite open cover of a closed subset of a normal space can be
"shrunk" to a new open cover so that the closure of each new open set is contained in the
corresponding original open set. -/
theorem exists_subset_iUnion_closure_subset (hs : IsClosed s) (uo : ∀ i, IsOpen (u i))
    (uf : ∀ x ∈ s, { i | x ∈ u i }.Finite) (us : s ⊆ ⋃ i, u i) :
    ∃ v : ι → Set X, s ⊆ iUnion v ∧ (∀ i, IsOpen (v i)) ∧ ∀ i, closure (v i) ⊆ u i := by
  haveI : Nonempty (PartialRefinement u s ⊤) :=
    ⟨⟨u, ∅, uo, us, False.elim, False.elim, fun _ => rfl⟩⟩
  have : ∀ c : Set (PartialRefinement u s ⊤),
      IsChain (· ≤ ·) c → c.Nonempty → ∃ ub, ∀ v ∈ c, v ≤ ub :=
    fun c hc ne => ⟨.chainSup c hc ne uf us, fun v hv => PartialRefinement.le_chainSup _ _ _ _ hv⟩
  /-
    ι : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝ : NormalSpace X
    hs : IsClosed s
    uo : ∀ (i : ι), IsOpen (u i)
    uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
    us : HasSubset.Subset s (Set.iUnion fun i => u i)
    this✝ : Nonempty (ShrinkingLemma.PartialRefinement u s Top.top)
    this : ∀ (c : Set (ShrinkingLemma.PartialRefinement u s Top.top)), IsChain (fu …
    ⊢ Exists fun v => And (HasSubset.Subset s (Set.iUnion v)) (And (∀ (i : ι), IsO …
  -/
  rcases zorn_le_nonempty this with ⟨v, hv⟩
  suffices ∀ i, i ∈ v.carrier from
    ⟨v, v.subset_iUnion, fun i => v.isOpen _, fun i => v.closure_subset (this i)⟩
  /-
    case intro
    ι : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝ : NormalSpace X
    hs : IsClosed s
    uo : ∀ (i : ι), IsOpen (u i)
    uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
    us : HasSubset.Subset s (Set.iUnion fun i => u i)
    this✝ : Nonempty (ShrinkingLemma.PartialRefinement u s Top.top)
    this : ∀ (c : Set (ShrinkingLemma.PartialRefinement u s Top.top)), IsChain (fu …
    v : ShrinkingLemma.PartialRefinement u s Top.top
    hv : IsMax v
    ⊢ ∀ (i : ι), Membership.mem v.carrier i
  -/
  refine fun i ↦ by_contra fun hi ↦ ?_
  /-
    case intro
    ι : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝ : NormalSpace X
    hs : IsClosed s
    uo : ∀ (i : ι), IsOpen (u i)
    uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
    us : HasSubset.Subset s (Set.iUnion fun i => u i)
    this✝ : Nonempty (ShrinkingLemma.PartialRefinement u s Top.top)
    this : ∀ (c : Set (ShrinkingLemma.PartialRefinement u s Top.top)), IsChain (fu …
    v : ShrinkingLemma.PartialRefinement u s Top.top
    hv : IsMax v
    i : ι
    hi : Not (Membership.mem v.carrier i)
    ⊢ False
  -/
  rcases v.exists_gt hs i hi with ⟨v', hlt⟩
  /-
    case intro.intro
    ι : Type u_1
    X : Type u_2
    inst✝¹ : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝ : NormalSpace X
    hs : IsClosed s
    uo : ∀ (i : ι), IsOpen (u i)
    uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
    us : HasSubset.Subset s (Set.iUnion fun i => u i)
    this✝ : Nonempty (ShrinkingLemma.PartialRefinement u s Top.top)
    this : ∀ (c : Set (ShrinkingLemma.PartialRefinement u s Top.top)), IsChain (fu …
    v : ShrinkingLemma.PartialRefinement u s Top.top
    hv : IsMax v
    i : ι
    hi : Not (Membership.mem v.carrier i)
    v' : ShrinkingLemma.PartialRefinement u s Top.top
    hlt : LT.lt v v'
    ⊢ False
  -/
  exact hv.not_lt hlt
  /-
    🎉 no goals
  -/


/-- **Shrinking lemma**. A point-finite open cover of a closed subset of a normal space can be
"shrunk" to a new closed cover so that each new closed set is contained in the corresponding
original open set. See also `exists_subset_iUnion_closure_subset` for a stronger statement. -/
theorem exists_subset_iUnion_closed_subset (hs : IsClosed s) (uo : ∀ i, IsOpen (u i))
    (uf : ∀ x ∈ s, { i | x ∈ u i }.Finite) (us : s ⊆ ⋃ i, u i) :
    ∃ v : ι → Set X, s ⊆ iUnion v ∧ (∀ i, IsClosed (v i)) ∧ ∀ i, v i ⊆ u i :=
  let ⟨v, hsv, _, hv⟩ := exists_subset_iUnion_closure_subset hs uo uf us
  ⟨fun i => closure (v i), Subset.trans hsv (iUnion_mono fun _ => subset_closure),
    fun _ => isClosed_closure, hv⟩


/-- Shrinking lemma. A point-finite open cover of a closed subset of a normal space can be "shrunk"
to a new open cover so that the closure of each new open set is contained in the corresponding
original open set. -/
theorem exists_iUnion_eq_closure_subset (uo : ∀ i, IsOpen (u i)) (uf : ∀ x, { i | x ∈ u i }.Finite)
    (uU : ⋃ i, u i = univ) :
    ∃ v : ι → Set X, iUnion v = univ ∧ (∀ i, IsOpen (v i)) ∧ ∀ i, closure (v i) ⊆ u i :=
  let ⟨v, vU, hv⟩ := exists_subset_iUnion_closure_subset isClosed_univ uo (fun x _ => uf x) uU.ge
  ⟨v, univ_subset_iff.1 vU, hv⟩


/-- Shrinking lemma. A point-finite open cover of a closed subset of a normal space can be "shrunk"
to a new closed cover so that each of the new closed sets is contained in the corresponding
original open set. See also `exists_iUnion_eq_closure_subset` for a stronger statement. -/
theorem exists_iUnion_eq_closed_subset (uo : ∀ i, IsOpen (u i)) (uf : ∀ x, { i | x ∈ u i }.Finite)
    (uU : ⋃ i, u i = univ) :
    ∃ v : ι → Set X, iUnion v = univ ∧ (∀ i, IsClosed (v i)) ∧ ∀ i, v i ⊆ u i :=
  let ⟨v, vU, hv⟩ := exists_subset_iUnion_closed_subset isClosed_univ uo (fun x _ => uf x) uU.ge
  ⟨v, univ_subset_iff.1 vU, hv⟩


/-- In a locally compact Hausdorff space `X`, if `s` is a compact set, `v` is a partial refinement,
and `i` is an index such that `i ∉ v.carrier`, then there exists a partial refinement that is
strictly greater than `v`. -/
theorem exists_gt_t2space (v : PartialRefinement u s (fun w => IsCompact (closure w)))
    (hs : IsCompact s) (i : ι) (hi : i ∉ v.carrier) :
    ∃ v' : PartialRefinement u s (fun w => IsCompact (closure w)),
      v < v' ∧ IsCompact (closure (v' i)) := by
  -- take `v i` such that `closure (v i)` is compact
  /-
    ι : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝¹ : T2Space X
    inst✝ : LocallyCompactSpace X
    v : ShrinkingLemma.PartialRefinement u s fun w => IsCompact (closure w)
    hs : IsCompact s
    i : ι
    hi : Not (Membership.mem v.carrier i)
    ⊢ Exists fun v' => And (LT.lt v v') (IsCompact (closure (v'.toFun i)))
  -/
  set si := s ∩ (⋃ j ≠ i, v j)ᶜ with hsi
  /-
    ι : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝¹ : T2Space X
    inst✝ : LocallyCompactSpace X
    v : ShrinkingLemma.PartialRefinement u s fun w => IsCompact (closure w)
    hs : IsCompact s
    i : ι
    hi : Not (Membership.mem v.carrier i)
    si : Set X := Inter.inter s (HasCompl.compl (Set.iUnion fun j => Set.iUnion fu …
    hsi : Eq si (Inter.inter s (HasCompl.compl (Set.iUnion fun j => Set.iUnion fun …
    ⊢ Exists fun v' => And (LT.lt v v') (IsCompact (closure (v'.toFun i)))
  -/
  simp only [ne_eq, compl_iUnion] at hsi
  have hsic : IsCompact si := by
    apply IsCompact.of_isClosed_subset hs _ Set.inter_subset_left
    · have : IsOpen (⋃ j ≠ i, v j) := by
        apply isOpen_biUnion
        intro j _
        exact v.isOpen j
      exact IsClosed.inter (IsCompact.isClosed hs) (IsOpen.isClosed_compl this)
  have : si ⊆ v i := by
    intro x hx
    have (j) (hj : j ≠ i) : x ∉ v j := by
      rw [hsi] at hx
      apply Set.not_mem_of_mem_compl
      have hsi' : x ∈ (⋂ i_1, ⋂ (_ : ¬i_1 = i), (v.toFun i_1)ᶜ) := Set.mem_of_mem_inter_right hx
      rw [ne_eq] at hj
      rw [Set.mem_iInter₂] at hsi'
      exact hsi' j hj
    obtain ⟨j, hj⟩ := Set.mem_iUnion.mp
      (v.subset_iUnion (Set.mem_of_mem_inter_left hx))
    obtain rfl : j = i := by
      by_contra! h
      exact this j h hj
    exact hj
  /-
    ι : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝¹ : T2Space X
    inst✝ : LocallyCompactSpace X
    v : ShrinkingLemma.PartialRefinement u s fun w => IsCompact (closure w)
    hs : IsCompact s
    i : ι
    hi : Not (Membership.mem v.carrier i)
    si : Set X := Inter.inter s (HasCompl.compl (Set.iUnion fun j => Set.iUnion fu …
    hsi : Eq si (Inter.inter s (Set.iInter fun i_1 => Set.iInter fun i => HasCompl …
    hsic : IsCompact si
    this : HasSubset.Subset si (v.toFun i)
    ⊢ Exists fun v' => And (LT.lt v v') (IsCompact (closure (v'.toFun i)))
  -/
  obtain ⟨vi, hvi⟩ := exists_open_between_and_isCompact_closure hsic (v.isOpen i) this
  classical
  refine ⟨⟨update v i vi, insert i v.carrier, ?_, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_⟩, ?_⟩
  · intro j
    rcases eq_or_ne j i with (rfl| hne) <;> simp [*, v.isOpen]
  · refine fun x hx => mem_iUnion.2 ?_
    rcases em (∃ j ≠ i, x ∈ v j) with (⟨j, hji, hj⟩ | h)
    · use j
      rwa [update_of_ne hji]
    · push_neg at h
      use i
      rw [update_self]
      apply hvi.2.1
      rw [hsi]
      exact ⟨hx, mem_iInter₂_of_mem h⟩
  · rintro j (rfl | hj)
    · rw [update_self]
      exact subset_trans hvi.2.2.1 <| PartialRefinement.subset v j
    · rw [update_of_ne (ne_of_mem_of_not_mem hj hi)]
      exact v.closure_subset hj
  · intro j hj
    rw [mem_insert_iff] at hj
    by_cases h : j = i
    · rw [← h]
      simp only [update_self]
      exact hvi.2.2.2
    · apply hj.elim
      · intro hji
        exact False.elim (h hji)
      · intro hjmemv
        rw [update_of_ne h]
        exact v.pred_of_mem hjmemv
  · intro j hj
    rw [mem_insert_iff, not_or] at hj
    rw [update_of_ne hj.1, v.apply_eq hj.2]
  · refine ⟨subset_insert _ _, fun j hj => ?_⟩
    exact (update_of_ne (ne_of_mem_of_not_mem hj hi) _ _).symm
  · exact fun hle => hi (hle.1 <| mem_insert _ _)
  · simp only [update_self]
    exact hvi.2.2.2


/-- **Shrinking lemma** . A point-finite open cover of a compact subset of a `T2Space`
`LocallyCompactSpace` can be "shrunk" to a new open cover so that the closure of each new open set
is contained in the corresponding original open set. -/
theorem exists_subset_iUnion_closure_subset_t2space (hs : IsCompact s) (uo : ∀ i, IsOpen (u i))
    (uf : ∀ x ∈ s, { i | x ∈ u i }.Finite) (us : s ⊆ ⋃ i, u i) :
    ∃ v : ι → Set X, s ⊆ iUnion v ∧ (∀ i, IsOpen (v i)) ∧ (∀ i, closure (v i) ⊆ u i)
      ∧ (∀ i, IsCompact (closure (v i))) := by
  haveI : Nonempty (PartialRefinement u s (fun w => IsCompact (closure w))) :=
    ⟨⟨u, ∅, uo, us, False.elim, False.elim, fun _ => rfl⟩⟩
  have : ∀ c : Set (PartialRefinement u s (fun w => IsCompact (closure w))),
      IsChain (· ≤ ·) c → c.Nonempty → ∃ ub, ∀ v ∈ c, v ≤ ub :=
    fun c hc ne => ⟨.chainSup c hc ne uf us, fun v hv => PartialRefinement.le_chainSup _ _ _ _ hv⟩
  /-
    ι : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝¹ : T2Space X
    inst✝ : LocallyCompactSpace X
    hs : IsCompact s
    uo : ∀ (i : ι), IsOpen (u i)
    uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
    us : HasSubset.Subset s (Set.iUnion fun i => u i)
    this✝ : Nonempty (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (clo …
    this : ∀ (c : Set (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (cl …
    ⊢ Exists fun v => And (HasSubset.Subset s (Set.iUnion v)) (And (∀ (i : ι), IsO …
  -/
  rcases zorn_le_nonempty this with ⟨v, hv⟩
  suffices ∀ i, i ∈ v.carrier from
    ⟨v, v.subset_iUnion, fun i => v.isOpen _, fun i => v.closure_subset (this i), ?_⟩
    /-
      case intro.refine_2
      ι : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      u : ι → Set X
      s : Set X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      hs : IsCompact s
      uo : ∀ (i : ι), IsOpen (u i)
      uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
      us : HasSubset.Subset s (Set.iUnion fun i => u i)
      this✝¹ : Nonempty (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (cl …
      this✝ : ∀ (c : Set (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (c …
      v : ShrinkingLemma.PartialRefinement u s fun w => IsCompact (closure w)
      hv : IsMax v
      this : ∀ (i : ι), Membership.mem v.carrier i
      ⊢ ∀ (i : ι), IsCompact (closure (v.toFun i))
    -/
  · intro i
    /-
      case intro.refine_2
      ι : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      u : ι → Set X
      s : Set X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      hs : IsCompact s
      uo : ∀ (i : ι), IsOpen (u i)
      uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
      us : HasSubset.Subset s (Set.iUnion fun i => u i)
      this✝¹ : Nonempty (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (cl …
      this✝ : ∀ (c : Set (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (c …
      v : ShrinkingLemma.PartialRefinement u s fun w => IsCompact (closure w)
      hv : IsMax v
      this : ∀ (i : ι), Membership.mem v.carrier i
      i : ι
      ⊢ IsCompact (closure (v.toFun i))
    -/
    exact v.pred_of_mem (this i)
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_1
      ι : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      u : ι → Set X
      s : Set X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      hs : IsCompact s
      uo : ∀ (i : ι), IsOpen (u i)
      uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
      us : HasSubset.Subset s (Set.iUnion fun i => u i)
      this✝ : Nonempty (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (clo …
      this : ∀ (c : Set (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (cl …
      v : ShrinkingLemma.PartialRefinement u s fun w => IsCompact (closure w)
      hv : IsMax v
      ⊢ ∀ (i : ι), Membership.mem v.carrier i
    -/
  · intro i
    /-
      case intro.refine_1
      ι : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      u : ι → Set X
      s : Set X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      hs : IsCompact s
      uo : ∀ (i : ι), IsOpen (u i)
      uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
      us : HasSubset.Subset s (Set.iUnion fun i => u i)
      this✝ : Nonempty (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (clo …
      this : ∀ (c : Set (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (cl …
      v : ShrinkingLemma.PartialRefinement u s fun w => IsCompact (closure w)
      hv : IsMax v
      i : ι
      ⊢ Membership.mem v.carrier i
    -/
    by_contra! hi
    /-
      case intro.refine_1
      ι : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      u : ι → Set X
      s : Set X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      hs : IsCompact s
      uo : ∀ (i : ι), IsOpen (u i)
      uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
      us : HasSubset.Subset s (Set.iUnion fun i => u i)
      this✝ : Nonempty (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (clo …
      this : ∀ (c : Set (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (cl …
      v : ShrinkingLemma.PartialRefinement u s fun w => IsCompact (closure w)
      hv : IsMax v
      i : ι
      hi : Not (Membership.mem v.carrier i)
      ⊢ False
    -/
    rcases exists_gt_t2space v hs i hi with ⟨v', hlt, _⟩
    /-
      case intro.refine_1.intro.intro
      ι : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      u : ι → Set X
      s : Set X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      hs : IsCompact s
      uo : ∀ (i : ι), IsOpen (u i)
      uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
      us : HasSubset.Subset s (Set.iUnion fun i => u i)
      this✝ : Nonempty (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (clo …
      this : ∀ (c : Set (ShrinkingLemma.PartialRefinement u s fun w => IsCompact (cl …
      v : ShrinkingLemma.PartialRefinement u s fun w => IsCompact (closure w)
      hv : IsMax v
      i : ι
      hi : Not (Membership.mem v.carrier i)
      v' : ShrinkingLemma.PartialRefinement u s fun w => IsCompact (closure w)
      hlt : LT.lt v v'
      right✝ : IsCompact (closure (v'.toFun i))
      ⊢ False
    -/
    exact hv.not_lt hlt
    /-
      🎉 no goals
    -/


/-- **Shrinking lemma**. A point-finite open cover of a compact subset of a locally compact T2 space
can be "shrunk" to a new closed cover so that each new closed set is contained in the corresponding
original open set. See also `exists_subset_iUnion_closure_subset_t2space` for a stronger statement.
-/
theorem exists_subset_iUnion_compact_subset_t2space (hs : IsCompact s) (uo : ∀ i, IsOpen (u i))
    (uf : ∀ x ∈ s, { i | x ∈ u i }.Finite) (us : s ⊆ ⋃ i, u i) :
    ∃ v : ι → Set X, s ⊆ iUnion v ∧ (∀ i, IsClosed (v i)) ∧ (∀ i, v i ⊆ u i)
      ∧ ∀ i, IsCompact (v i) := by
  /-
    ι : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝¹ : T2Space X
    inst✝ : LocallyCompactSpace X
    hs : IsCompact s
    uo : ∀ (i : ι), IsOpen (u i)
    uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
    us : HasSubset.Subset s (Set.iUnion fun i => u i)
    ⊢ Exists fun v => And (HasSubset.Subset s (Set.iUnion v)) (And (∀ (i : ι), IsC …
  -/
  let ⟨v, hsv, _, hv⟩ := exists_subset_iUnion_closure_subset_t2space hs uo uf us
  /-
    ι : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝¹ : T2Space X
    inst✝ : LocallyCompactSpace X
    hs : IsCompact s
    uo : ∀ (i : ι), IsOpen (u i)
    uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
    us : HasSubset.Subset s (Set.iUnion fun i => u i)
    v : ι → Set X
    hsv : HasSubset.Subset s (Set.iUnion v)
    left✝ : ∀ (i : ι), IsOpen (v i)
    hv : And (∀ (i : ι), HasSubset.Subset (closure (v i)) (u i)) (∀ (i : ι), IsCom …
    ⊢ Exists fun v => And (HasSubset.Subset s (Set.iUnion v)) (And (∀ (i : ι), IsC …
  -/
  use fun i => closure (v i)
  /-
    case h
    ι : Type u_1
    X : Type u_2
    inst✝² : TopologicalSpace X
    u : ι → Set X
    s : Set X
    inst✝¹ : T2Space X
    inst✝ : LocallyCompactSpace X
    hs : IsCompact s
    uo : ∀ (i : ι), IsOpen (u i)
    uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
    us : HasSubset.Subset s (Set.iUnion fun i => u i)
    v : ι → Set X
    hsv : HasSubset.Subset s (Set.iUnion v)
    left✝ : ∀ (i : ι), IsOpen (v i)
    hv : And (∀ (i : ι), HasSubset.Subset (closure (v i)) (u i)) (∀ (i : ι), IsCom …
    ⊢ And (HasSubset.Subset s (Set.iUnion fun i => closure (v i))) (And (∀ (i : ι) …
  -/
  refine ⟨?_, ?_, ?_⟩
    /-
      case h.refine_1
      ι : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      u : ι → Set X
      s : Set X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      hs : IsCompact s
      uo : ∀ (i : ι), IsOpen (u i)
      uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
      us : HasSubset.Subset s (Set.iUnion fun i => u i)
      v : ι → Set X
      hsv : HasSubset.Subset s (Set.iUnion v)
      left✝ : ∀ (i : ι), IsOpen (v i)
      hv : And (∀ (i : ι), HasSubset.Subset (closure (v i)) (u i)) (∀ (i : ι), IsCom …
      ⊢ HasSubset.Subset s (Set.iUnion fun i => closure (v i))
    -/
  · exact Subset.trans hsv (iUnion_mono fun _ => subset_closure)
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      ι : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      u : ι → Set X
      s : Set X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      hs : IsCompact s
      uo : ∀ (i : ι), IsOpen (u i)
      uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
      us : HasSubset.Subset s (Set.iUnion fun i => u i)
      v : ι → Set X
      hsv : HasSubset.Subset s (Set.iUnion v)
      left✝ : ∀ (i : ι), IsOpen (v i)
      hv : And (∀ (i : ι), HasSubset.Subset (closure (v i)) (u i)) (∀ (i : ι), IsCom …
      ⊢ ∀ (i : ι), IsClosed ((fun i => closure (v i)) i)
    -/
  · simp only [isClosed_closure, implies_true]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_3
      ι : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      u : ι → Set X
      s : Set X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      hs : IsCompact s
      uo : ∀ (i : ι), IsOpen (u i)
      uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
      us : HasSubset.Subset s (Set.iUnion fun i => u i)
      v : ι → Set X
      hsv : HasSubset.Subset s (Set.iUnion v)
      left✝ : ∀ (i : ι), IsOpen (v i)
      hv : And (∀ (i : ι), HasSubset.Subset (closure (v i)) (u i)) (∀ (i : ι), IsCom …
      ⊢ And (∀ (i : ι), HasSubset.Subset ((fun i => closure (v i)) i) (u i)) (∀ (i : …
    -/
  · simp only
    /-
      case h.refine_3
      ι : Type u_1
      X : Type u_2
      inst✝² : TopologicalSpace X
      u : ι → Set X
      s : Set X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      hs : IsCompact s
      uo : ∀ (i : ι), IsOpen (u i)
      uf : ∀ (x : X), Membership.mem s x → (setOf fun i => Membership.mem (u i) x).F …
      us : HasSubset.Subset s (Set.iUnion fun i => u i)
      v : ι → Set X
      hsv : HasSubset.Subset s (Set.iUnion v)
      left✝ : ∀ (i : ι), IsOpen (v i)
      hv : And (∀ (i : ι), HasSubset.Subset (closure (v i)) (u i)) (∀ (i : ι), IsCom …
      ⊢ And (∀ (i : ι), HasSubset.Subset (closure (v i)) (u i)) (∀ (i : ι), IsCompac …
    -/
    exact And.intro (fun i => hv.1 i) (fun i => hv.2 i)
    /-
      🎉 no goals
    -/


