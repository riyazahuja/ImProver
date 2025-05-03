/-- Given any function `m` assigning measures to sets satisfying `m ∅ = 0`, there is
  a unique maximal outer measure `μ` satisfying `μ s ≤ m s` for all `s : Set α`. -/
protected def ofFunction (m : Set α → ℝ≥0∞) (m_empty : m ∅ = 0) : OuterMeasure α :=
  let μ s := ⨅ (f : ℕ → Set α) (_ : s ⊆ ⋃ i, f i), ∑' i, m (f i)
  { measureOf := μ
    empty :=
      le_antisymm
                                                                            /-
                                                                              α : Type u_1
                                                                              m : Set α → ENNReal
                                                                              m_empty : Eq (m EmptyCollection.emptyCollection) 0
                                                                              μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
                                                                              ⊢ LE.le (tsum fun i => m ((fun x => EmptyCollection.emptyCollection) i)) 0
                                                                            -/
        ((iInf_le_of_le fun _ => ∅) <| iInf_le_of_le (empty_subset _) <| by simp [m_empty])
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
        (zero_le _)
    mono := fun {_ _} hs => iInf_mono fun _ => iInf_mono' fun hb => ⟨hs.trans hb, le_rfl⟩
    iUnion_nat := fun s _ =>
      ENNReal.le_of_forall_pos_le_add <| by
        /-
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ⊢ ∀ (ε : NNReal), LT.lt 0 ε → LT.lt (tsum fun i => μ (s i)) Top.top → LE.le (μ …
        -/
        intro ε hε (hb : (∑' i, μ (s i)) < ∞)
        /-
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ⊢ LE.le (μ (Set.iUnion fun i => s i)) (HAdd.hAdd (tsum fun i => μ (s i)) ↑ε)
        -/
        rcases ENNReal.exists_pos_sum_of_countable (ENNReal.coe_pos.2 hε).ne' ℕ with ⟨ε', hε', hl⟩
        /-
          case intro.intro
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          ⊢ LE.le (μ (Set.iUnion fun i => s i)) (HAdd.hAdd (tsum fun i => μ (s i)) ↑ε)
        -/
        refine le_trans ?_ (add_le_add_left (le_of_lt hl) _)
        /-
          case intro.intro
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          ⊢ LE.le (μ (Set.iUnion fun i => s i)) (HAdd.hAdd (tsum fun i => μ (s i)) (tsum …
        -/
        rw [← ENNReal.tsum_add]
        choose f hf using
          show ∀ i, ∃ f : ℕ → Set α, (s i ⊆ ⋃ i, f i) ∧ (∑' i, m (f i)) < μ (s i) + ε' i by
            intro i
            have : μ (s i) < μ (s i) + ε' i :=
              ENNReal.lt_add_right (ne_top_of_le_ne_top hb.ne <| ENNReal.le_tsum _)
                (by simpa using (hε' i).ne')
            rcases iInf_lt_iff.mp this with ⟨t, ht⟩
            exists t
            contrapose! ht
            exact le_iInf ht
        /-
          case intro.intro
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          f : Nat → Nat → Set α
          hf : ∀ (i : Nat), And (HasSubset.Subset (s i) (Set.iUnion fun i_1 => f i i_1)) …
          ⊢ LE.le (μ (Set.iUnion fun i => s i)) (tsum fun a => HAdd.hAdd (μ (s a)) ↑(ε'  …
        -/
        refine le_trans ?_ (ENNReal.tsum_le_tsum fun i => le_of_lt (hf i).2)
        /-
          case intro.intro
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          f : Nat → Nat → Set α
          hf : ∀ (i : Nat), And (HasSubset.Subset (s i) (Set.iUnion fun i_1 => f i i_1)) …
          ⊢ LE.le (μ (Set.iUnion fun i => s i)) (tsum fun a => tsum fun i => m (f a i))
        -/
        rw [← ENNReal.tsum_prod, ← Nat.pairEquiv.symm.tsum_eq]
        /-
          case intro.intro
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          f : Nat → Nat → Set α
          hf : ∀ (i : Nat), And (HasSubset.Subset (s i) (Set.iUnion fun i_1 => f i i_1)) …
          ⊢ LE.le (μ (Set.iUnion fun i => s i)) (tsum fun c => m (f (Nat.pairEquiv.symm  …
        -/
        refine iInf_le_of_le _ (iInf_le _ ?_)
        /-
          case intro.intro
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          f : Nat → Nat → Set α
          hf : ∀ (i : Nat), And (HasSubset.Subset (s i) (Set.iUnion fun i_1 => f i i_1)) …
          ⊢ HasSubset.Subset (Set.iUnion fun i => s i) (Set.iUnion fun i => f (Nat.pairE …
        -/
        apply iUnion_subset
        /-
          case intro.intro.h
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          f : Nat → Nat → Set α
          hf : ∀ (i : Nat), And (HasSubset.Subset (s i) (Set.iUnion fun i_1 => f i i_1)) …
          ⊢ ∀ (i : Nat), HasSubset.Subset (s i) (Set.iUnion fun i => f (Nat.pairEquiv.sy …
        -/
        intro i
        /-
          case intro.intro.h
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          f : Nat → Nat → Set α
          hf : ∀ (i : Nat), And (HasSubset.Subset (s i) (Set.iUnion fun i_1 => f i i_1)) …
          i : Nat
          ⊢ HasSubset.Subset (s i) (Set.iUnion fun i => f (Nat.pairEquiv.symm i).1 (Nat. …
        -/
        apply Subset.trans (hf i).1
        /-
          case intro.intro.h
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          f : Nat → Nat → Set α
          hf : ∀ (i : Nat), And (HasSubset.Subset (s i) (Set.iUnion fun i_1 => f i i_1)) …
          i : Nat
          ⊢ HasSubset.Subset (Set.iUnion fun i_1 => f i i_1) (Set.iUnion fun i => f (Nat …
        -/
        apply iUnion_subset
        /-
          case intro.intro.h.h
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          f : Nat → Nat → Set α
          hf : ∀ (i : Nat), And (HasSubset.Subset (s i) (Set.iUnion fun i_1 => f i i_1)) …
          i : Nat
          ⊢ ∀ (i_1 : Nat), HasSubset.Subset (f i i_1) (Set.iUnion fun i => f (Nat.pairEq …
        -/
        simp only [Nat.pairEquiv_symm_apply]
        /-
          case intro.intro.h.h
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          f : Nat → Nat → Set α
          hf : ∀ (i : Nat), And (HasSubset.Subset (s i) (Set.iUnion fun i_1 => f i i_1)) …
          i : Nat
          ⊢ ∀ (i_1 : Nat), HasSubset.Subset (f i i_1) (Set.iUnion fun i => f (Nat.unpair …
        -/
        rw [iUnion_unpair]
        /-
          case intro.intro.h.h
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          f : Nat → Nat → Set α
          hf : ∀ (i : Nat), And (HasSubset.Subset (s i) (Set.iUnion fun i_1 => f i i_1)) …
          i : Nat
          ⊢ ∀ (i_1 : Nat), HasSubset.Subset (f i i_1) (Set.iUnion fun i => Set.iUnion fu …
        -/
        intro j
        /-
          case intro.intro.h.h
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          μ : Set α → ENNReal := fun s => iInf fun f => iInf fun x => tsum fun i => m (f …
          s : Nat → Set α
          x✝ : Pairwise (Function.onFun Disjoint s)
          ε : NNReal
          hε : LT.lt 0 ε
          hb : LT.lt (tsum fun i => μ (s i)) Top.top
          ε' : Nat → NNReal
          hε' : ∀ (i : Nat), LT.lt 0 (ε' i)
          hl : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
          f : Nat → Nat → Set α
          hf : ∀ (i : Nat), And (HasSubset.Subset (s i) (Set.iUnion fun i_1 => f i i_1)) …
          i j : Nat
          ⊢ HasSubset.Subset (f i j) (Set.iUnion fun i => Set.iUnion fun j => f i j)
        -/
        apply subset_iUnion₂ i }
        /-
          🎉 no goals
        -/


/-- `ofFunction` of a set `s` is the infimum of `∑ᵢ, m (tᵢ)` for all collections of sets
`tᵢ` that cover `s`. -/
theorem ofFunction_apply (s : Set α) :
    OuterMeasure.ofFunction m m_empty s = ⨅ (t : ℕ → Set α) (_ : s ⊆ iUnion t), ∑' n, m (t n) :=
  rfl


/-- `ofFunction` of a set `s` is the infimum of `∑ᵢ, m (tᵢ)` for all collections of sets
`tᵢ` that cover `s`, with all `tᵢ` satisfying a predicate `P` such that `m` is infinite for sets
that don't satisfy `P`.
This is similar to `ofFunction_apply`, except that the sets `tᵢ` satisfy `P`.
The hypothesis `m_top` applies in particular to a function of the form `extend m'`. -/
theorem ofFunction_eq_iInf_mem {P : Set α → Prop} (m_top : ∀ s, ¬ P s → m s = ∞) (s : Set α) :
    OuterMeasure.ofFunction m m_empty s =
      ⨅ (t : ℕ → Set α) (_ : ∀ i, P (t i)) (_ : s ⊆ ⋃ i, t i), ∑' i, m (t i) := by
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    P : Set α → Prop
    m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
    s : Set α
    ⊢ Eq ((MeasureTheory.OuterMeasure.ofFunction m m_empty) s) (iInf fun t => iInf …
  -/
  rw [OuterMeasure.ofFunction_apply]
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    P : Set α → Prop
    m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
    s : Set α
    ⊢ Eq (iInf fun t => iInf fun x => tsum fun n => m (t n)) (iInf fun t => iInf f …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      P : Set α → Prop
      m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
      s : Set α
      ⊢ LE.le (iInf fun t => iInf fun x => tsum fun n => m (t n)) (iInf fun t => iIn …
    -/
  · exact le_iInf fun t ↦ le_iInf fun _ ↦ le_iInf fun h ↦ iInf₂_le _ (by exact h)
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      P : Set α → Prop
      m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
      s : Set α
      ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => tsum fun i => m (t i)) (iIn …
    -/
  · simp_rw [le_iInf_iff]
    /-
      case a
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      P : Set α → Prop
      m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
      s : Set α
      ⊢ ∀ (i : Nat → Set α), HasSubset.Subset s (Set.iUnion i) → LE.le (iInf fun t = …
    -/
    refine fun t ht_subset ↦ iInf_le_of_le t ?_
    /-
      case a
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      P : Set α → Prop
      m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
      s : Set α
      t : Nat → Set α
      ht_subset : HasSubset.Subset s (Set.iUnion t)
      ⊢ LE.le (iInf fun x => iInf fun x => tsum fun i => m (t i)) (tsum fun n => m ( …
    -/
    by_cases ht : ∀ i, P (t i)
      /-
        case pos
        α : Type u_1
        m : Set α → ENNReal
        m_empty : Eq (m EmptyCollection.emptyCollection) 0
        P : Set α → Prop
        m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
        s : Set α
        t : Nat → Set α
        ht_subset : HasSubset.Subset s (Set.iUnion t)
        ht : ∀ (i : Nat), P (t i)
        ⊢ LE.le (iInf fun x => iInf fun x => tsum fun i => m (t i)) (tsum fun n => m ( …
      -/
    · exact iInf_le_of_le ht (iInf_le_of_le ht_subset le_rfl)
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        m : Set α → ENNReal
        m_empty : Eq (m EmptyCollection.emptyCollection) 0
        P : Set α → Prop
        m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
        s : Set α
        t : Nat → Set α
        ht_subset : HasSubset.Subset s (Set.iUnion t)
        ht : Not (∀ (i : Nat), P (t i))
        ⊢ LE.le (iInf fun x => iInf fun x => tsum fun i => m (t i)) (tsum fun n => m ( …
      -/
    · simp only [ht, not_false_eq_true, iInf_neg, top_le_iff]
      /-
        case neg
        α : Type u_1
        m : Set α → ENNReal
        m_empty : Eq (m EmptyCollection.emptyCollection) 0
        P : Set α → Prop
        m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
        s : Set α
        t : Nat → Set α
        ht_subset : HasSubset.Subset s (Set.iUnion t)
        ht : Not (∀ (i : Nat), P (t i))
        ⊢ Eq (tsum fun i => m (t i)) Top.top
      -/
      push_neg at ht
      /-
        case neg
        α : Type u_1
        m : Set α → ENNReal
        m_empty : Eq (m EmptyCollection.emptyCollection) 0
        P : Set α → Prop
        m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
        s : Set α
        t : Nat → Set α
        ht_subset : HasSubset.Subset s (Set.iUnion t)
        ht : Exists fun i => Not (P (t i))
        ⊢ Eq (tsum fun i => m (t i)) Top.top
      -/
      obtain ⟨i, hti_not_mem⟩ := ht
      /-
        case neg.intro
        α : Type u_1
        m : Set α → ENNReal
        m_empty : Eq (m EmptyCollection.emptyCollection) 0
        P : Set α → Prop
        m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
        s : Set α
        t : Nat → Set α
        ht_subset : HasSubset.Subset s (Set.iUnion t)
        i : Nat
        hti_not_mem : Not (P (t i))
        ⊢ Eq (tsum fun i => m (t i)) Top.top
      -/
      have hfi_top : m (t i) = ∞ := m_top _ hti_not_mem
      /-
        case neg.intro
        α : Type u_1
        m : Set α → ENNReal
        m_empty : Eq (m EmptyCollection.emptyCollection) 0
        P : Set α → Prop
        m_top : ∀ (s : Set α), Not (P s) → Eq (m s) Top.top
        s : Set α
        t : Nat → Set α
        ht_subset : HasSubset.Subset s (Set.iUnion t)
        i : Nat
        hti_not_mem : Not (P (t i))
        hfi_top : Eq (m (t i)) Top.top
        ⊢ Eq (tsum fun i => m (t i)) Top.top
      -/
      exact ENNReal.tsum_eq_top_of_eq_top ⟨i, hfi_top⟩
      /-
        🎉 no goals
      -/


theorem ofFunction_le (s : Set α) : OuterMeasure.ofFunction m m_empty s ≤ m s :=
  let f : ℕ → Set α := fun i => Nat.casesOn i s fun _ => ∅
  iInf_le_of_le f <|
    iInf_le_of_le (subset_iUnion f 0) <|
      le_of_eq <| tsum_eq_single 0 <| by
        /-
          α : Type u_1
          m : Set α → ENNReal
          m_empty : Eq (m EmptyCollection.emptyCollection) 0
          s : Set α
          f : Nat → Set α := fun i => Nat.casesOn i s fun x => EmptyCollection.emptyColl …
          ⊢ ∀ (b' : Nat), Ne b' 0 → Eq (m (f b')) 0
        -/
        rintro (_ | i)
          /-
            case zero
            α : Type u_1
            m : Set α → ENNReal
            m_empty : Eq (m EmptyCollection.emptyCollection) 0
            s : Set α
            f : Nat → Set α := fun i => Nat.casesOn i s fun x => EmptyCollection.emptyColl …
            ⊢ Ne 0 0 → Eq (m (f 0)) 0
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case succ
            α : Type u_1
            m : Set α → ENNReal
            m_empty : Eq (m EmptyCollection.emptyCollection) 0
            s : Set α
            f : Nat → Set α := fun i => Nat.casesOn i s fun x => EmptyCollection.emptyColl …
            i : Nat
            ⊢ Ne (HAdd.hAdd i 1) 0 → Eq (m (f (HAdd.hAdd i 1))) 0
          -/
        · simp [f, m_empty]
          /-
            🎉 no goals
          -/


theorem ofFunction_eq (s : Set α) (m_mono : ∀ ⦃t : Set α⦄, s ⊆ t → m s ≤ m t)
    (m_subadd : ∀ s : ℕ → Set α, m (⋃ i, s i) ≤ ∑' i, m (s i)) :
    OuterMeasure.ofFunction m m_empty s = m s :=
  le_antisymm (ofFunction_le s) <|
    le_iInf fun f => le_iInf fun hf => le_trans (m_mono hf) (m_subadd f)


theorem le_ofFunction {μ : OuterMeasure α} :
    μ ≤ OuterMeasure.ofFunction m m_empty ↔ ∀ s, μ s ≤ m s :=
  ⟨fun H s => le_trans (H s) (ofFunction_le s), fun H _ =>
    le_iInf fun f =>
      le_iInf fun hs =>
        le_trans (μ.mono hs) <| le_trans (measure_iUnion_le f) <| ENNReal.tsum_le_tsum fun _ => H _⟩


theorem isGreatest_ofFunction :
    IsGreatest { μ : OuterMeasure α | ∀ s, μ s ≤ m s } (OuterMeasure.ofFunction m m_empty) :=
  ⟨fun _ => ofFunction_le _, fun _ => le_ofFunction.2⟩


theorem ofFunction_eq_sSup : OuterMeasure.ofFunction m m_empty = sSup { μ | ∀ s, μ s ≤ m s } :=
  (@isGreatest_ofFunction α m m_empty).isLUB.sSup_eq.symm


/-- If `m u = ∞` for any set `u` that has nonempty intersection both with `s` and `t`, then
`μ (s ∪ t) = μ s + μ t`, where `μ = MeasureTheory.OuterMeasure.ofFunction m m_empty`.

E.g., if `α` is an (e)metric space and `m u = ∞` on any set of diameter `≥ r`, then this lemma
implies that `μ (s ∪ t) = μ s + μ t` on any two sets such that `r ≤ edist x y` for all `x ∈ s`
and `y ∈ t`. -/
theorem ofFunction_union_of_top_of_nonempty_inter {s t : Set α}
    (h : ∀ u, (s ∩ u).Nonempty → (t ∩ u).Nonempty → m u = ∞) :
    OuterMeasure.ofFunction m m_empty (s ∪ t) =
      OuterMeasure.ofFunction m m_empty s + OuterMeasure.ofFunction m m_empty t := by
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    s t : Set α
    h : ∀ (u : Set α), (Inter.inter s u).Nonempty → (Inter.inter t u).Nonempty → E …
    ⊢ Eq ((MeasureTheory.OuterMeasure.ofFunction m m_empty) (Union.union s t)) (HA …
  -/
  refine le_antisymm (measure_union_le _ _) (le_iInf₂ fun f hf ↦ ?_)
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    s t : Set α
    h : ∀ (u : Set α), (Inter.inter s u).Nonempty → (Inter.inter t u).Nonempty → E …
    f : Nat → Set α
    hf : HasSubset.Subset (Union.union s t) (Set.iUnion fun i => f i)
    ⊢ LE.le (HAdd.hAdd ((MeasureTheory.OuterMeasure.ofFunction m m_empty) s) ((Mea …
  -/
  set μ := OuterMeasure.ofFunction m m_empty
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    s t : Set α
    h : ∀ (u : Set α), (Inter.inter s u).Nonempty → (Inter.inter t u).Nonempty → E …
    f : Nat → Set α
    hf : HasSubset.Subset (Union.union s t) (Set.iUnion fun i => f i)
    μ : MeasureTheory.OuterMeasure α := MeasureTheory.OuterMeasure.ofFunction m m_ …
    ⊢ LE.le (HAdd.hAdd (μ s) (μ t)) (tsum fun i => m (f i))
  -/
  rcases Classical.em (∃ i, (s ∩ f i).Nonempty ∧ (t ∩ f i).Nonempty) with (⟨i, hs, ht⟩ | he)
  · calc
      μ s + μ t ≤ ∞ := le_top
      _ = m (f i) := (h (f i) hs ht).symm
      _ ≤ ∑' i, m (f i) := ENNReal.le_tsum i

  /-
    case inr
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    s t : Set α
    h : ∀ (u : Set α), (Inter.inter s u).Nonempty → (Inter.inter t u).Nonempty → E …
    f : Nat → Set α
    hf : HasSubset.Subset (Union.union s t) (Set.iUnion fun i => f i)
    μ : MeasureTheory.OuterMeasure α := MeasureTheory.OuterMeasure.ofFunction m m_ …
    he : Not (Exists fun i => And (Inter.inter s (f i)).Nonempty (Inter.inter t (f …
    ⊢ LE.le (HAdd.hAdd (μ s) (μ t)) (tsum fun i => m (f i))
  -/
  set I := fun s => { i : ℕ | (s ∩ f i).Nonempty }
  /-
    case inr
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    s t : Set α
    h : ∀ (u : Set α), (Inter.inter s u).Nonempty → (Inter.inter t u).Nonempty → E …
    f : Nat → Set α
    hf : HasSubset.Subset (Union.union s t) (Set.iUnion fun i => f i)
    μ : MeasureTheory.OuterMeasure α := MeasureTheory.OuterMeasure.ofFunction m m_ …
    he : Not (Exists fun i => And (Inter.inter s (f i)).Nonempty (Inter.inter t (f …
    I : Set α → Set Nat := fun s => setOf fun i => (Inter.inter s (f i)).Nonempty
    ⊢ LE.le (HAdd.hAdd (μ s) (μ t)) (tsum fun i => m (f i))
  -/
  have hd : Disjoint (I s) (I t) := disjoint_iff_inf_le.mpr fun i hi => he ⟨i, hi⟩
  have hI : ∀ u ⊆ s ∪ t, μ u ≤ ∑' i : I u, μ (f i) := fun u hu =>
    calc
      μ u ≤ μ (⋃ i : I u, f i) :=
        μ.mono fun x hx =>
          let ⟨i, hi⟩ := mem_iUnion.1 (hf (hu hx))
          mem_iUnion.2 ⟨⟨i, ⟨x, hx, hi⟩⟩, hi⟩
      _ ≤ ∑' i : I u, μ (f i) := measure_iUnion_le _

  calc
    μ s + μ t ≤ (∑' i : I s, μ (f i)) + ∑' i : I t, μ (f i) :=
      add_le_add (hI _ subset_union_left) (hI _ subset_union_right)
    _ = ∑' i : ↑(I s ∪ I t), μ (f i) :=
      (tsum_union_disjoint (f := fun i => μ (f i)) hd ENNReal.summable ENNReal.summable).symm
    _ ≤ ∑' i, μ (f i) :=
      (tsum_le_tsum_of_inj (↑) Subtype.coe_injective (fun _ _ => zero_le _) (fun _ => le_rfl)
        ENNReal.summable ENNReal.summable)
    _ ≤ ∑' i, m (f i) := ENNReal.tsum_le_tsum fun i => ofFunction_le _


theorem comap_ofFunction {β} (f : β → α) (h : Monotone m ∨ Surjective f) :
    comap f (OuterMeasure.ofFunction m m_empty) =
                                                        /-
                                                          α : Type u_1
                                                          m : Set α → ENNReal
                                                          m_empty : Eq (m EmptyCollection.emptyCollection) 0
                                                          β : Type ?u.27416
                                                          f : β → α
                                                          h : Or (Monotone m) (Function.Surjective f)
                                                          ⊢ Eq ((fun s => m (Set.image f s)) EmptyCollection.emptyCollection) 0
                                                        -/
      OuterMeasure.ofFunction (fun s => m (f '' s)) (by simp; simp [m_empty]) := by
                                                              /-
                                                                🎉 no goals
                                                              -/
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    β : Type u_2
    f : β → α
    h : Or (Monotone m) (Function.Surjective f)
    ⊢ Eq ((MeasureTheory.OuterMeasure.comap f) (MeasureTheory.OuterMeasure.ofFunct …
  -/
  refine le_antisymm (le_ofFunction.2 fun s => ?_) fun s => ?_
    /-
      case refine_1
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : β → α
      h : Or (Monotone m) (Function.Surjective f)
      s : Set β
      ⊢ LE.le (((MeasureTheory.OuterMeasure.comap f) (MeasureTheory.OuterMeasure.ofF …
    -/
  · rw [comap_apply]
    /-
      case refine_1
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : β → α
      h : Or (Monotone m) (Function.Surjective f)
      s : Set β
      ⊢ LE.le ((MeasureTheory.OuterMeasure.ofFunction m m_empty) (Set.image f s)) (m …
    -/
    apply ofFunction_le
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : β → α
      h : Or (Monotone m) (Function.Surjective f)
      s : Set β
      ⊢ LE.le ((MeasureTheory.OuterMeasure.ofFunction (fun s => m (Set.image f s)) ⋯ …
    -/
  · rw [comap_apply, ofFunction_apply, ofFunction_apply]
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : β → α
      h : Or (Monotone m) (Function.Surjective f)
      s : Set β
      ⊢ LE.le (iInf fun t => iInf fun x => tsum fun n => m (Set.image f (t n))) (iIn …
    -/
    refine iInf_mono' fun t => ⟨fun k => f ⁻¹' t k, ?_⟩
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : β → α
      h : Or (Monotone m) (Function.Surjective f)
      s : Set β
      t : Nat → Set α
      ⊢ LE.le (iInf fun x => tsum fun n => m (Set.image f ((fun k => Set.preimage f  …
    -/
    refine iInf_mono' fun ht => ?_
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : β → α
      h : Or (Monotone m) (Function.Surjective f)
      s : Set β
      t : Nat → Set α
      ht : HasSubset.Subset (Set.image f s) (Set.iUnion t)
      ⊢ Exists fun i => LE.le (tsum fun n => m (Set.image f ((fun k => Set.preimage  …
    -/
    rw [Set.image_subset_iff, preimage_iUnion] at ht
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : β → α
      h : Or (Monotone m) (Function.Surjective f)
      s : Set β
      t : Nat → Set α
      ht : HasSubset.Subset s (Set.iUnion fun i => Set.preimage f (t i))
      ⊢ Exists fun i => LE.le (tsum fun n => m (Set.image f ((fun k => Set.preimage  …
    -/
    refine ⟨ht, ENNReal.tsum_le_tsum fun n => ?_⟩
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : β → α
      h : Or (Monotone m) (Function.Surjective f)
      s : Set β
      t : Nat → Set α
      ht : HasSubset.Subset s (Set.iUnion fun i => Set.preimage f (t i))
      n : Nat
      ⊢ LE.le (m (Set.image f ((fun k => Set.preimage f (t k)) n))) (m (t n))
    -/
    cases' h with hl hr
    /-
      case refine_2.inl
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : β → α
      s : Set β
      t : Nat → Set α
      ht : HasSubset.Subset s (Set.iUnion fun i => Set.preimage f (t i))
      n : Nat
      hl : Monotone m
      ⊢ LE.le (m (Set.image f ((fun k => Set.preimage f (t k)) n))) (m (t n))
    -/
    exacts [hl (image_preimage_subset _ _), (congr_arg m (hr.image_preimage (t n))).le]
    /-
      🎉 no goals
    -/


theorem map_ofFunction_le {β} (f : α → β) :
    map f (OuterMeasure.ofFunction m m_empty) ≤
      OuterMeasure.ofFunction (fun s => m (f ⁻¹' s)) m_empty :=
  le_ofFunction.2 fun s => by
    /-
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : α → β
      s : Set β
      ⊢ LE.le (((MeasureTheory.OuterMeasure.map f) (MeasureTheory.OuterMeasure.ofFun …
    -/
    rw [map_apply]
    /-
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : α → β
      s : Set β
      ⊢ LE.le ((MeasureTheory.OuterMeasure.ofFunction m m_empty) (Set.preimage f s)) …
    -/
    apply ofFunction_le
    /-
      🎉 no goals
    -/


theorem map_ofFunction {β} {f : α → β} (hf : Injective f) :
    map f (OuterMeasure.ofFunction m m_empty) =
      OuterMeasure.ofFunction (fun s => m (f ⁻¹' s)) m_empty := by
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    ⊢ Eq ((MeasureTheory.OuterMeasure.map f) (MeasureTheory.OuterMeasure.ofFunctio …
  -/
  refine (map_ofFunction_le _).antisymm fun s => ?_
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    s : Set β
    ⊢ LE.le ((MeasureTheory.OuterMeasure.ofFunction (fun s => m (Set.preimage f s) …
  -/
  simp only [ofFunction_apply, map_apply, le_iInf_iff]
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    s : Set β
    ⊢ ∀ (i : Nat → Set α), HasSubset.Subset (Set.preimage f s) (Set.iUnion i) → LE …
  -/
  intro t ht
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    s : Set β
    t : Nat → Set α
    ht : HasSubset.Subset (Set.preimage f s) (Set.iUnion t)
    ⊢ LE.le (iInf fun t => iInf fun x => tsum fun n => m (Set.preimage f (t n))) ( …
  -/
  refine iInf_le_of_le (fun n => (range f)ᶜ ∪ f '' t n) (iInf_le_of_le ?_ ?_)
    /-
      case refine_1
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : α → β
      hf : Function.Injective f
      s : Set β
      t : Nat → Set α
      ht : HasSubset.Subset (Set.preimage f s) (Set.iUnion t)
      ⊢ HasSubset.Subset s (Set.iUnion fun n => Union.union (HasCompl.compl (Set.ran …
    -/
  · rw [← union_iUnion, ← inter_subset, ← image_preimage_eq_inter_range, ← image_iUnion]
    /-
      case refine_1
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : α → β
      hf : Function.Injective f
      s : Set β
      t : Nat → Set α
      ht : HasSubset.Subset (Set.preimage f s) (Set.iUnion t)
      ⊢ HasSubset.Subset (Set.image f (Set.preimage f s)) (Set.image f (Set.iUnion f …
    -/
    exact image_subset _ ht
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : α → β
      hf : Function.Injective f
      s : Set β
      t : Nat → Set α
      ht : HasSubset.Subset (Set.preimage f s) (Set.iUnion t)
      ⊢ LE.le (tsum fun n => m (Set.preimage f ((fun n => Union.union (HasCompl.comp …
    -/
  · refine ENNReal.tsum_le_tsum fun n => le_of_eq ?_
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      m_empty : Eq (m EmptyCollection.emptyCollection) 0
      β : Type u_2
      f : α → β
      hf : Function.Injective f
      s : Set β
      t : Nat → Set α
      ht : HasSubset.Subset (Set.preimage f s) (Set.iUnion t)
      n : Nat
      ⊢ Eq (m (Set.preimage f ((fun n => Union.union (HasCompl.compl (Set.range f))  …
    -/
    simp [hf.preimage_image]
    /-
      🎉 no goals
    -/

-- TODO (kmill): change `m (t ∩ s)` to `m (s ∩ t)`

theorem restrict_ofFunction (s : Set α) (hm : Monotone m) :
    restrict s (OuterMeasure.ofFunction m m_empty) =
                                                       /-
                                                         α : Type u_1
                                                         m : Set α → ENNReal
                                                         m_empty : Eq (m EmptyCollection.emptyCollection) 0
                                                         s : Set α
                                                         hm : Monotone m
                                                         ⊢ Eq ((fun t => m (Inter.inter t s)) EmptyCollection.emptyCollection) 0
                                                       -/
      OuterMeasure.ofFunction (fun t => m (t ∩ s)) (by simp; simp [m_empty]) := by
                                                             /-
                                                               🎉 no goals
                                                             -/
      /-
        α : Type u_1
        m : Set α → ENNReal
        m_empty : Eq (m EmptyCollection.emptyCollection) 0
        s : Set α
        hm : Monotone m
        ⊢ Eq ((MeasureTheory.OuterMeasure.restrict s) (MeasureTheory.OuterMeasure.ofFu …
      -/
      rw [restrict]
      /-
        α : Type u_1
        m : Set α → ENNReal
        m_empty : Eq (m EmptyCollection.emptyCollection) 0
        s : Set α
        hm : Monotone m
        ⊢ Eq (((MeasureTheory.OuterMeasure.map Subtype.val).comp (MeasureTheory.OuterM …
      -/
      simp only [inter_comm _ s, LinearMap.comp_apply]
      /-
        α : Type u_1
        m : Set α → ENNReal
        m_empty : Eq (m EmptyCollection.emptyCollection) 0
        s : Set α
        hm : Monotone m
        ⊢ Eq ((MeasureTheory.OuterMeasure.map Subtype.val) ((MeasureTheory.OuterMeasur …
      -/
      rw [comap_ofFunction _ (Or.inl hm)]
      /-
        α : Type u_1
        m : Set α → ENNReal
        m_empty : Eq (m EmptyCollection.emptyCollection) 0
        s : Set α
        hm : Monotone m
        ⊢ Eq ((MeasureTheory.OuterMeasure.map Subtype.val) (MeasureTheory.OuterMeasure …
      -/
      simp only [map_ofFunction Subtype.coe_injective, Subtype.image_preimage_coe]
      /-
        🎉 no goals
      -/


theorem smul_ofFunction {c : ℝ≥0∞} (hc : c ≠ ∞) : c • OuterMeasure.ofFunction m m_empty =
                                        /-
                                          α : Type u_1
                                          m : Set α → ENNReal
                                          m_empty : Eq (m EmptyCollection.emptyCollection) 0
                                          c : ENNReal
                                          hc : Ne c Top.top
                                          ⊢ Eq (HSMul.hSMul c m EmptyCollection.emptyCollection) 0
                                        -/
    OuterMeasure.ofFunction (c • m) (by simp [m_empty]) := by
                                        /-
                                          🎉 no goals
                                        -/
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    c : ENNReal
    hc : Ne c Top.top
    ⊢ Eq (HSMul.hSMul c (MeasureTheory.OuterMeasure.ofFunction m m_empty)) (Measur …
  -/
  ext1 s
  /-
    case h
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    c : ENNReal
    hc : Ne c Top.top
    s : Set α
    ⊢ Eq ((HSMul.hSMul c (MeasureTheory.OuterMeasure.ofFunction m m_empty)) s) ((M …
  -/
  haveI : Nonempty { t : ℕ → Set α // s ⊆ ⋃ i, t i } := ⟨⟨fun _ => s, subset_iUnion (fun _ => s) 0⟩⟩
  simp only [smul_apply, ofFunction_apply, ENNReal.tsum_mul_left, Pi.smul_apply, smul_eq_mul,
  iInf_subtype']
  /-
    case h
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    c : ENNReal
    hc : Ne c Top.top
    s : Set α
    this : Nonempty (Subtype fun t => HasSubset.Subset s (Set.iUnion fun i => t i))
    ⊢ Eq (HMul.hMul c (iInf fun x => tsum fun n => m (↑x n))) (iInf fun x => HMul. …
  -/
  rw [ENNReal.mul_iInf fun h => (hc h).elim]
  /-
    🎉 no goals
  -/


/-- Given any function `m` assigning measures to sets, there is a unique maximal outer measure `μ`
  satisfying `μ s ≤ m s` for all `s : Set α`. This is the same as `OuterMeasure.ofFunction`,
  except that it doesn't require `m ∅ = 0`. -/
def boundedBy : OuterMeasure α :=
                                                               /-
                                                                 α : Type u_1
                                                                 m : Set α → ENNReal
                                                                 ⊢ Eq ((fun s => iSup fun x => m s) EmptyCollection.emptyCollection) 0
                                                               -/
  OuterMeasure.ofFunction (fun s => ⨆ _ : s.Nonempty, m s) (by simp [Set.not_nonempty_empty])
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem boundedBy_le (s : Set α) : boundedBy m s ≤ m s :=
  (ofFunction_le _).trans iSup_const_le


theorem boundedBy_eq_ofFunction (m_empty : m ∅ = 0) (s : Set α) :
    boundedBy m s = OuterMeasure.ofFunction m m_empty s := by
  have : (fun s : Set α => ⨆ _ : s.Nonempty, m s) = m := by
    ext1 t
    rcases t.eq_empty_or_nonempty with h | h <;> simp [h, Set.not_nonempty_empty, m_empty]
  /-
    α : Type u_1
    m : Set α → ENNReal
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    s : Set α
    this : Eq (fun s => iSup fun x => m s) m
    ⊢ Eq ((MeasureTheory.OuterMeasure.boundedBy m) s) ((MeasureTheory.OuterMeasure …
  -/
  simp [boundedBy, this]
  /-
    🎉 no goals
  -/


theorem boundedBy_apply (s : Set α) :
    boundedBy m s = ⨅ (t : ℕ → Set α) (_ : s ⊆ iUnion t),
                      ∑' n, ⨆ _ : (t n).Nonempty, m (t n) := by
  /-
    α : Type u_1
    m : Set α → ENNReal
    s : Set α
    ⊢ Eq ((MeasureTheory.OuterMeasure.boundedBy m) s) (iInf fun t => iInf fun x => …
  -/
  simp [boundedBy, ofFunction_apply]
  /-
    🎉 no goals
  -/


theorem boundedBy_eq (s : Set α) (m_empty : m ∅ = 0) (m_mono : ∀ ⦃t : Set α⦄, s ⊆ t → m s ≤ m t)
    (m_subadd : ∀ s : ℕ → Set α, m (⋃ i, s i) ≤ ∑' i, m (s i)) : boundedBy m s = m s := by
  /-
    α : Type u_1
    m : Set α → ENNReal
    s : Set α
    m_empty : Eq (m EmptyCollection.emptyCollection) 0
    m_mono : ∀ ⦃t : Set α⦄, HasSubset.Subset s t → LE.le (m s) (m t)
    m_subadd : ∀ (s : Nat → Set α), LE.le (m (Set.iUnion fun i => s i)) (tsum fun  …
    ⊢ Eq ((MeasureTheory.OuterMeasure.boundedBy m) s) (m s)
  -/
  rw [boundedBy_eq_ofFunction m_empty, ofFunction_eq s m_mono m_subadd]
  /-
    🎉 no goals
  -/


@[simp]
theorem boundedBy_eq_self (m : OuterMeasure α) : boundedBy m = m :=
  ext fun _ => boundedBy_eq _ measure_empty (fun _ ht => measure_mono ht) measure_iUnion_le


theorem le_boundedBy {μ : OuterMeasure α} : μ ≤ boundedBy m ↔ ∀ s, μ s ≤ m s := by
  /-
    α : Type u_1
    m : Set α → ENNReal
    μ : MeasureTheory.OuterMeasure α
    ⊢ Iff (LE.le μ (MeasureTheory.OuterMeasure.boundedBy m)) (∀ (s : Set α), LE.le …
  -/
  rw [boundedBy , le_ofFunction, forall_congr']; intro s
  /-
    α : Type u_1
    m : Set α → ENNReal
    μ : MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ Iff (LE.le (μ s) (iSup fun x => m s)) (LE.le (μ s) (m s))
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  rcases s.eq_empty_or_nonempty with h | h <;> simp [h, Set.not_nonempty_empty]
                                               /-
                                                 🎉 no goals
                                               -/


theorem le_boundedBy' {μ : OuterMeasure α} :
    μ ≤ boundedBy m ↔ ∀ s : Set α, s.Nonempty → μ s ≤ m s := by
  /-
    α : Type u_1
    m : Set α → ENNReal
    μ : MeasureTheory.OuterMeasure α
    ⊢ Iff (LE.le μ (MeasureTheory.OuterMeasure.boundedBy m)) (∀ (s : Set α), s.Non …
  -/
  rw [le_boundedBy, forall_congr']
  /-
    α : Type u_1
    m : Set α → ENNReal
    μ : MeasureTheory.OuterMeasure α
    ⊢ ∀ (a : Set α), Iff (LE.le (μ a) (m a)) (a.Nonempty → LE.le (μ a) (m a))
  -/
  intro s
  /-
    α : Type u_1
    m : Set α → ENNReal
    μ : MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ Iff (LE.le (μ s) (m s)) (s.Nonempty → LE.le (μ s) (m s))
  -/
                                               /-
                                                 🎉 no goals
                                               -/
  rcases s.eq_empty_or_nonempty with h | h <;> simp [h]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem boundedBy_top : boundedBy (⊤ : Set α → ℝ≥0∞) = ⊤ := by
  /-
    α : Type u_1
    ⊢ Eq (MeasureTheory.OuterMeasure.boundedBy Top.top) Top.top
  -/
  rw [eq_top_iff, le_boundedBy']
  /-
    α : Type u_1
    ⊢ ∀ (s : Set α), s.Nonempty → LE.le (Top.top s) (Top.top s)
  -/
  intro s hs
  /-
    α : Type u_1
    s : Set α
    hs : s.Nonempty
    ⊢ LE.le (Top.top s) (Top.top s)
  -/
  rw [top_apply hs]
  /-
    α : Type u_1
    s : Set α
    hs : s.Nonempty
    ⊢ LE.le Top.top (Top.top s)
  -/
  exact le_rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem boundedBy_zero : boundedBy (0 : Set α → ℝ≥0∞) = 0 := by
  /-
    α : Type u_1
    ⊢ Eq (MeasureTheory.OuterMeasure.boundedBy 0) 0
  -/
  rw [← coe_bot, eq_bot_iff]
  /-
    α : Type u_1
    ⊢ LE.le (MeasureTheory.OuterMeasure.boundedBy 0) Bot.bot
  -/
  apply boundedBy_le
  /-
    🎉 no goals
  -/


theorem smul_boundedBy {c : ℝ≥0∞} (hc : c ≠ ∞) : c • boundedBy m = boundedBy (c • m) := by
  /-
    α : Type u_1
    m : Set α → ENNReal
    c : ENNReal
    hc : Ne c Top.top
    ⊢ Eq (HSMul.hSMul c (MeasureTheory.OuterMeasure.boundedBy m)) (MeasureTheory.O …
  -/
  simp only [boundedBy , smul_ofFunction hc]
  /-
    α : Type u_1
    m : Set α → ENNReal
    c : ENNReal
    hc : Ne c Top.top
    ⊢ Eq (MeasureTheory.OuterMeasure.ofFunction (HSMul.hSMul c fun s => iSup fun x …
  -/
  congr 1 with s : 1
  /-
    case e_m.h
    α : Type u_1
    m : Set α → ENNReal
    c : ENNReal
    hc : Ne c Top.top
    s : Set α
    ⊢ Eq (HSMul.hSMul c (fun s => iSup fun x => m s) s) (iSup fun x => HSMul.hSMul …
  -/
                                                    /-
                                                      🎉 no goals
                                                    -/
  rcases s.eq_empty_or_nonempty with (rfl | hs) <;> simp [*]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem comap_boundedBy {β} (f : β → α)
    (h : (Monotone fun s : { s : Set α // s.Nonempty } => m s) ∨ Surjective f) :
    comap f (boundedBy m) = boundedBy fun s => m (f '' s) := by
  /-
    α : Type u_1
    m : Set α → ENNReal
    β : Type u_2
    f : β → α
    h : Or (Monotone fun s => m ↑s) (Function.Surjective f)
    ⊢ Eq ((MeasureTheory.OuterMeasure.comap f) (MeasureTheory.OuterMeasure.bounded …
  -/
  refine (comap_ofFunction _ ?_).trans ?_
    /-
      case refine_1
      α : Type u_1
      m : Set α → ENNReal
      β : Type u_2
      f : β → α
      h : Or (Monotone fun s => m ↑s) (Function.Surjective f)
      ⊢ Or (Monotone fun s => iSup fun x => m s) (Function.Surjective f)
    -/
  · refine h.imp (fun H s t hst => iSup_le fun hs => ?_) id
    /-
      case refine_1
      α : Type u_1
      m : Set α → ENNReal
      β : Type u_2
      f : β → α
      h : Or (Monotone fun s => m ↑s) (Function.Surjective f)
      H : Monotone fun s => m ↑s
      s t : Set α
      hst : LE.le s t
      hs : s.Nonempty
      ⊢ LE.le (m s) ((fun s => iSup fun x => m s) t)
    -/
    have ht : t.Nonempty := hs.mono hst
    /-
      case refine_1
      α : Type u_1
      m : Set α → ENNReal
      β : Type u_2
      f : β → α
      h : Or (Monotone fun s => m ↑s) (Function.Surjective f)
      H : Monotone fun s => m ↑s
      s t : Set α
      hst : LE.le s t
      hs : s.Nonempty
      ht : t.Nonempty
      ⊢ LE.le (m s) ((fun s => iSup fun x => m s) t)
    -/
    exact (@H ⟨s, hs⟩ ⟨t, ht⟩ hst).trans (le_iSup (fun _ : t.Nonempty => m t) ht)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      β : Type u_2
      f : β → α
      h : Or (Monotone fun s => m ↑s) (Function.Surjective f)
      ⊢ Eq (MeasureTheory.OuterMeasure.ofFunction (fun s => iSup fun x => m (Set.ima …
    -/
  · dsimp only [boundedBy]
    /-
      case refine_2
      α : Type u_1
      m : Set α → ENNReal
      β : Type u_2
      f : β → α
      h : Or (Monotone fun s => m ↑s) (Function.Surjective f)
      ⊢ Eq (MeasureTheory.OuterMeasure.ofFunction (fun s => iSup fun x => m (Set.ima …
    -/
    congr with s : 1
    /-
      case refine_2.e_m.h
      α : Type u_1
      m : Set α → ENNReal
      β : Type u_2
      f : β → α
      h : Or (Monotone fun s => m ↑s) (Function.Surjective f)
      s : Set β
      ⊢ Eq (iSup fun x => m (Set.image f s)) (iSup fun x => m (Set.image f s))
    -/
    rw [image_nonempty]
    /-
      🎉 no goals
    -/


/-- If `m u = ∞` for any set `u` that has nonempty intersection both with `s` and `t`, then
`μ (s ∪ t) = μ s + μ t`, where `μ = MeasureTheory.OuterMeasure.boundedBy m`.

E.g., if `α` is an (e)metric space and `m u = ∞` on any set of diameter `≥ r`, then this lemma
implies that `μ (s ∪ t) = μ s + μ t` on any two sets such that `r ≤ edist x y` for all `x ∈ s`
and `y ∈ t`. -/
theorem boundedBy_union_of_top_of_nonempty_inter {s t : Set α}
    (h : ∀ u, (s ∩ u).Nonempty → (t ∩ u).Nonempty → m u = ∞) :
    boundedBy m (s ∪ t) = boundedBy m s + boundedBy m t :=
  ofFunction_union_of_top_of_nonempty_inter fun u hs ht =>
    top_unique <| (h u hs ht).ge.trans <| le_iSup (fun _ => m u) (hs.mono inter_subset_right)


/-- Given a set of outer measures, we define a new function that on a set `s` is defined to be the
  infimum of `μ(s)` for the outer measures `μ` in the collection. We ensure that this
  function is defined to be `0` on `∅`, even if the collection of outer measures is empty.
  The outer measure generated by this function is the infimum of the given outer measures. -/
def sInfGen (m : Set (OuterMeasure α)) (s : Set α) : ℝ≥0∞ :=
  ⨅ (μ : OuterMeasure α) (_ : μ ∈ m), μ s


theorem sInfGen_def (m : Set (OuterMeasure α)) (t : Set α) :
    sInfGen m t = ⨅ (μ : OuterMeasure α) (_ : μ ∈ m), μ t :=
  rfl


theorem sInf_eq_boundedBy_sInfGen (m : Set (OuterMeasure α)) :
    sInf m = OuterMeasure.boundedBy (sInfGen m) := by
  /-
    α : Type u_1
    m : Set (MeasureTheory.OuterMeasure α)
    ⊢ Eq (InfSet.sInf m) (MeasureTheory.OuterMeasure.boundedBy (MeasureTheory.Oute …
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      α : Type u_1
      m : Set (MeasureTheory.OuterMeasure α)
      ⊢ LE.le (InfSet.sInf m) (MeasureTheory.OuterMeasure.boundedBy (MeasureTheory.O …
    -/
  · refine le_boundedBy.2 fun s => le_iInf₂ fun μ hμ => ?_
    /-
      case refine_1
      α : Type u_1
      m : Set (MeasureTheory.OuterMeasure α)
      s : Set α
      μ : MeasureTheory.OuterMeasure α
      hμ : Membership.mem m μ
      ⊢ LE.le ((InfSet.sInf m) s) (μ s)
    -/
    apply sInf_le hμ
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : Set (MeasureTheory.OuterMeasure α)
      ⊢ LE.le (MeasureTheory.OuterMeasure.boundedBy (MeasureTheory.OuterMeasure.sInf …
    -/
  · refine le_sInf ?_
    /-
      case refine_2
      α : Type u_1
      m : Set (MeasureTheory.OuterMeasure α)
      ⊢ ∀ (b : MeasureTheory.OuterMeasure α), Membership.mem m b → LE.le (MeasureThe …
    -/
    intro μ hμ t
    /-
      case refine_2
      α : Type u_1
      m : Set (MeasureTheory.OuterMeasure α)
      μ : MeasureTheory.OuterMeasure α
      hμ : Membership.mem m μ
      t : Set α
      ⊢ LE.le ((MeasureTheory.OuterMeasure.boundedBy (MeasureTheory.OuterMeasure.sIn …
    -/
    exact le_trans (boundedBy_le t) (iInf₂_le μ hμ)
    /-
      🎉 no goals
    -/


theorem iSup_sInfGen_nonempty {m : Set (OuterMeasure α)} (h : m.Nonempty) (t : Set α) :
    ⨆ _ : t.Nonempty, sInfGen m t = ⨅ (μ : OuterMeasure α) (_ : μ ∈ m), μ t := by
  /-
    α : Type u_1
    m : Set (MeasureTheory.OuterMeasure α)
    h : m.Nonempty
    t : Set α
    ⊢ Eq (iSup fun x => MeasureTheory.OuterMeasure.sInfGen m t) (iInf fun μ => iIn …
  -/
  rcases t.eq_empty_or_nonempty with (rfl | ht)
    /-
      case inl
      α : Type u_1
      m : Set (MeasureTheory.OuterMeasure α)
      h : m.Nonempty
      ⊢ Eq (iSup fun x => MeasureTheory.OuterMeasure.sInfGen m EmptyCollection.empty …
    -/
  · simp [biInf_const h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      m : Set (MeasureTheory.OuterMeasure α)
      h : m.Nonempty
      t : Set α
      ht : t.Nonempty
      ⊢ Eq (iSup fun x => MeasureTheory.OuterMeasure.sInfGen m t) (iInf fun μ => iIn …
    -/
  · simp [ht, sInfGen_def]
    /-
      🎉 no goals
    -/


/-- The value of the Infimum of a nonempty set of outer measures on a set is not simply
the minimum value of a measure on that set: it is the infimum sum of measures of countable set of
sets that covers that set, where a different measure can be used for each set in the cover. -/
theorem sInf_apply {m : Set (OuterMeasure α)} {s : Set α} (h : m.Nonempty) :
    sInf m s =
      ⨅ (t : ℕ → Set α) (_ : s ⊆ iUnion t), ∑' n, ⨅ (μ : OuterMeasure α) (_ : μ ∈ m), μ (t n) := by
  /-
    α : Type u_1
    m : Set (MeasureTheory.OuterMeasure α)
    s : Set α
    h : m.Nonempty
    ⊢ Eq ((InfSet.sInf m) s) (iInf fun t => iInf fun x => tsum fun n => iInf fun μ …
  -/
  simp_rw [sInf_eq_boundedBy_sInfGen, boundedBy_apply, iSup_sInfGen_nonempty h]
  /-
    🎉 no goals
  -/


/-- The value of the Infimum of a set of outer measures on a nonempty set is not simply
the minimum value of a measure on that set: it is the infimum sum of measures of countable set of
sets that covers that set, where a different measure can be used for each set in the cover. -/
theorem sInf_apply' {m : Set (OuterMeasure α)} {s : Set α} (h : s.Nonempty) :
    sInf m s =
      ⨅ (t : ℕ → Set α) (_ : s ⊆ iUnion t), ∑' n, ⨅ (μ : OuterMeasure α) (_ : μ ∈ m), μ (t n) :=
                                            /-
                                              α : Type u_1
                                              m : Set (MeasureTheory.OuterMeasure α)
                                              s : Set α
                                              h : s.Nonempty
                                              hm : Eq m EmptyCollection.emptyCollection
                                              ⊢ Eq ((InfSet.sInf m) s) (iInf fun t => iInf fun x => tsum fun n => iInf fun μ …
                                            -/
  m.eq_empty_or_nonempty.elim (fun hm => by simp [hm, h]) sInf_apply
                                            /-
                                              🎉 no goals
                                            -/


/-- The value of the Infimum of a nonempty family of outer measures on a set is not simply
the minimum value of a measure on that set: it is the infimum sum of measures of countable set of
sets that covers that set, where a different measure can be used for each set in the cover. -/
theorem iInf_apply {ι} [Nonempty ι] (m : ι → OuterMeasure α) (s : Set α) :
    (⨅ i, m i) s = ⨅ (t : ℕ → Set α) (_ : s ⊆ iUnion t), ∑' n, ⨅ i, m i (t n) := by
  /-
    α : Type u_1
    ι : Sort u_2
    inst✝ : Nonempty ι
    m : ι → MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ Eq ((iInf fun i => m i) s) (iInf fun t => iInf fun x => tsum fun n => iInf f …
  -/
  rw [iInf, sInf_apply (range_nonempty m)]
  /-
    α : Type u_1
    ι : Sort u_2
    inst✝ : Nonempty ι
    m : ι → MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ Eq (iInf fun t => iInf fun x => tsum fun n => iInf fun μ => iInf fun x => μ  …
  -/
  simp only [iInf_range]
  /-
    🎉 no goals
  -/


/-- The value of the Infimum of a family of outer measures on a nonempty set is not simply
the minimum value of a measure on that set: it is the infimum sum of measures of countable set of
sets that covers that set, where a different measure can be used for each set in the cover. -/
theorem iInf_apply' {ι} (m : ι → OuterMeasure α) {s : Set α} (hs : s.Nonempty) :
    (⨅ i, m i) s = ⨅ (t : ℕ → Set α) (_ : s ⊆ iUnion t), ∑' n, ⨅ i, m i (t n) := by
  /-
    α : Type u_1
    ι : Sort u_2
    m : ι → MeasureTheory.OuterMeasure α
    s : Set α
    hs : s.Nonempty
    ⊢ Eq ((iInf fun i => m i) s) (iInf fun t => iInf fun x => tsum fun n => iInf f …
  -/
  rw [iInf, sInf_apply' hs]
  /-
    α : Type u_1
    ι : Sort u_2
    m : ι → MeasureTheory.OuterMeasure α
    s : Set α
    hs : s.Nonempty
    ⊢ Eq (iInf fun t => iInf fun x => tsum fun n => iInf fun μ => iInf fun x => μ  …
  -/
  simp only [iInf_range]
  /-
    🎉 no goals
  -/


/-- The value of the Infimum of a nonempty family of outer measures on a set is not simply
the minimum value of a measure on that set: it is the infimum sum of measures of countable set of
sets that covers that set, where a different measure can be used for each set in the cover. -/
theorem biInf_apply {ι} {I : Set ι} (hI : I.Nonempty) (m : ι → OuterMeasure α) (s : Set α) :
    (⨅ i ∈ I, m i) s = ⨅ (t : ℕ → Set α) (_ : s ⊆ iUnion t), ∑' n, ⨅ i ∈ I, m i (t n) := by
  /-
    α : Type u_1
    ι : Type u_2
    I : Set ι
    hI : I.Nonempty
    m : ι → MeasureTheory.OuterMeasure α
    s : Set α
    ⊢ Eq ((iInf fun i => iInf fun h => m i) s) (iInf fun t => iInf fun x => tsum f …
  -/
  haveI := hI.to_subtype
  /-
    α : Type u_1
    ι : Type u_2
    I : Set ι
    hI : I.Nonempty
    m : ι → MeasureTheory.OuterMeasure α
    s : Set α
    this : Nonempty ↑I
    ⊢ Eq ((iInf fun i => iInf fun h => m i) s) (iInf fun t => iInf fun x => tsum f …
  -/
  simp only [← iInf_subtype'', iInf_apply]
  /-
    🎉 no goals
  -/


/-- The value of the Infimum of a nonempty family of outer measures on a set is not simply
the minimum value of a measure on that set: it is the infimum sum of measures of countable set of
sets that covers that set, where a different measure can be used for each set in the cover. -/
theorem biInf_apply' {ι} (I : Set ι) (m : ι → OuterMeasure α) {s : Set α} (hs : s.Nonempty) :
    (⨅ i ∈ I, m i) s = ⨅ (t : ℕ → Set α) (_ : s ⊆ iUnion t), ∑' n, ⨅ i ∈ I, m i (t n) := by
  /-
    α : Type u_1
    ι : Type u_2
    I : Set ι
    m : ι → MeasureTheory.OuterMeasure α
    s : Set α
    hs : s.Nonempty
    ⊢ Eq ((iInf fun i => iInf fun h => m i) s) (iInf fun t => iInf fun x => tsum f …
  -/
  simp only [← iInf_subtype'', iInf_apply' _ hs]
  /-
    🎉 no goals
  -/


theorem map_iInf_le {ι β} (f : α → β) (m : ι → OuterMeasure α) :
    map f (⨅ i, m i) ≤ ⨅ i, map f (m i) :=
  (map_mono f).map_iInf_le


theorem comap_iInf {ι β} (f : α → β) (m : ι → OuterMeasure β) :
    comap f (⨅ i, m i) = ⨅ i, comap f (m i) := by
  /-
    α : Type u_1
    ι : Sort u_2
    β : Type u_3
    f : α → β
    m : ι → MeasureTheory.OuterMeasure β
    ⊢ Eq ((MeasureTheory.OuterMeasure.comap f) (iInf fun i => m i)) (iInf fun i => …
  -/
  refine ext_nonempty fun s hs => ?_
  /-
    α : Type u_1
    ι : Sort u_2
    β : Type u_3
    f : α → β
    m : ι → MeasureTheory.OuterMeasure β
    s : Set α
    hs : s.Nonempty
    ⊢ Eq (((MeasureTheory.OuterMeasure.comap f) (iInf fun i => m i)) s) ((iInf fun …
  -/
  refine ((comap_mono f).map_iInf_le s).antisymm ?_
  simp only [comap_apply, iInf_apply' _ hs, iInf_apply' _ (hs.image _), le_iInf_iff,
    Set.image_subset_iff, preimage_iUnion]
  /-
    α : Type u_1
    ι : Sort u_2
    β : Type u_3
    f : α → β
    m : ι → MeasureTheory.OuterMeasure β
    s : Set α
    hs : s.Nonempty
    ⊢ ∀ (i : Nat → Set β), HasSubset.Subset s (Set.iUnion fun i_1 => Set.preimage  …
  -/
  refine fun t ht => iInf_le_of_le _ (iInf_le_of_le ht <| ENNReal.tsum_le_tsum fun k => ?_)
  /-
    α : Type u_1
    ι : Sort u_2
    β : Type u_3
    f : α → β
    m : ι → MeasureTheory.OuterMeasure β
    s : Set α
    hs : s.Nonempty
    t : Nat → Set β
    ht : HasSubset.Subset s (Set.iUnion fun i => Set.preimage f (t i))
    k : Nat
    ⊢ LE.le (iInf fun i => (m i) (Set.image f (Set.preimage f (t k)))) (iInf fun i …
  -/
  exact iInf_mono fun i => (m i).mono (image_preimage_subset _ _)
  /-
    🎉 no goals
  -/


theorem map_iInf {ι β} {f : α → β} (hf : Injective f) (m : ι → OuterMeasure α) :
    map f (⨅ i, m i) = restrict (range f) (⨅ i, map f (m i)) := by
  /-
    α : Type u_1
    ι : Sort u_2
    β : Type u_3
    f : α → β
    hf : Function.Injective f
    m : ι → MeasureTheory.OuterMeasure α
    ⊢ Eq ((MeasureTheory.OuterMeasure.map f) (iInf fun i => m i)) ((MeasureTheory. …
  -/
  refine Eq.trans ?_ (map_comap _ _)
  /-
    α : Type u_1
    ι : Sort u_2
    β : Type u_3
    f : α → β
    hf : Function.Injective f
    m : ι → MeasureTheory.OuterMeasure α
    ⊢ Eq ((MeasureTheory.OuterMeasure.map f) (iInf fun i => m i)) ((MeasureTheory. …
  -/
  simp only [comap_iInf, comap_map hf]
  /-
    🎉 no goals
  -/


theorem map_iInf_comap {ι β} [Nonempty ι] {f : α → β} (m : ι → OuterMeasure β) :
    map f (⨅ i, comap f (m i)) = ⨅ i, map f (comap f (m i)) := by
  /-
    α : Type u_1
    ι : Sort u_2
    β : Type u_3
    inst✝ : Nonempty ι
    f : α → β
    m : ι → MeasureTheory.OuterMeasure β
    ⊢ Eq ((MeasureTheory.OuterMeasure.map f) (iInf fun i => (MeasureTheory.OuterMe …
  -/
  refine (map_iInf_le _ _).antisymm fun s => ?_
  /-
    α : Type u_1
    ι : Sort u_2
    β : Type u_3
    inst✝ : Nonempty ι
    f : α → β
    m : ι → MeasureTheory.OuterMeasure β
    s : Set β
    ⊢ LE.le ((iInf fun i => (MeasureTheory.OuterMeasure.map f) ((MeasureTheory.Out …
  -/
  simp only [map_apply, comap_apply, iInf_apply, le_iInf_iff]
  /-
    α : Type u_1
    ι : Sort u_2
    β : Type u_3
    inst✝ : Nonempty ι
    f : α → β
    m : ι → MeasureTheory.OuterMeasure β
    s : Set β
    ⊢ ∀ (i : Nat → Set α), HasSubset.Subset (Set.preimage f s) (Set.iUnion i) → LE …
  -/
  refine fun t ht => iInf_le_of_le (fun n => f '' t n ∪ (range f)ᶜ) (iInf_le_of_le ?_ ?_)
  · rw [← iUnion_union, Set.union_comm, ← inter_subset, ← image_iUnion, ←
      image_preimage_eq_inter_range]
    /-
      case refine_1
      α : Type u_1
      ι : Sort u_2
      β : Type u_3
      inst✝ : Nonempty ι
      f : α → β
      m : ι → MeasureTheory.OuterMeasure β
      s : Set β
      t : Nat → Set α
      ht : HasSubset.Subset (Set.preimage f s) (Set.iUnion t)
      ⊢ HasSubset.Subset (Set.image f (Set.preimage f s)) (Set.image f (Set.iUnion f …
    -/
    exact image_subset _ ht
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      ι : Sort u_2
      β : Type u_3
      inst✝ : Nonempty ι
      f : α → β
      m : ι → MeasureTheory.OuterMeasure β
      s : Set β
      t : Nat → Set α
      ht : HasSubset.Subset (Set.preimage f s) (Set.iUnion t)
      ⊢ LE.le (tsum fun n => iInf fun i => (m i) (Set.image f (Set.preimage f ((fun  …
    -/
  · refine ENNReal.tsum_le_tsum fun n => iInf_mono fun i => (m i).mono ?_
    simp only [preimage_union, preimage_compl, preimage_range, compl_univ, union_empty,
      image_subset_iff]
    /-
      case refine_2
      α : Type u_1
      ι : Sort u_2
      β : Type u_3
      inst✝ : Nonempty ι
      f : α → β
      m : ι → MeasureTheory.OuterMeasure β
      s : Set β
      t : Nat → Set α
      ht : HasSubset.Subset (Set.preimage f s) (Set.iUnion t)
      n : Nat
      i : ι
      ⊢ HasSubset.Subset (Set.preimage f (Set.image f (t n))) (Set.preimage f (Set.i …
    -/
    exact subset_refl _
    /-
      🎉 no goals
    -/


theorem map_biInf_comap {ι β} {I : Set ι} (hI : I.Nonempty) {f : α → β} (m : ι → OuterMeasure β) :
    map f (⨅ i ∈ I, comap f (m i)) = ⨅ i ∈ I, map f (comap f (m i)) := by
  /-
    α : Type u_1
    ι : Type u_2
    β : Type u_3
    I : Set ι
    hI : I.Nonempty
    f : α → β
    m : ι → MeasureTheory.OuterMeasure β
    ⊢ Eq ((MeasureTheory.OuterMeasure.map f) (iInf fun i => iInf fun h => (Measure …
  -/
  haveI := hI.to_subtype
  /-
    α : Type u_1
    ι : Type u_2
    β : Type u_3
    I : Set ι
    hI : I.Nonempty
    f : α → β
    m : ι → MeasureTheory.OuterMeasure β
    this : Nonempty ↑I
    ⊢ Eq ((MeasureTheory.OuterMeasure.map f) (iInf fun i => iInf fun h => (Measure …
  -/
  rw [← iInf_subtype'', ← iInf_subtype'']
  /-
    α : Type u_1
    ι : Type u_2
    β : Type u_3
    I : Set ι
    hI : I.Nonempty
    f : α → β
    m : ι → MeasureTheory.OuterMeasure β
    this : Nonempty ↑I
    ⊢ Eq ((MeasureTheory.OuterMeasure.map f) (iInf fun i => (MeasureTheory.OuterMe …
  -/
  exact map_iInf_comap _
  /-
    🎉 no goals
  -/


theorem restrict_iInf_restrict {ι} (s : Set α) (m : ι → OuterMeasure α) :
    restrict s (⨅ i, restrict s (m i)) = restrict s (⨅ i, m i) :=
  calc restrict s (⨅ i, restrict s (m i))
                                                                     /-
                                                                       α : Type u_1
                                                                       ι : Sort u_2
                                                                       s : Set α
                                                                       m : ι → MeasureTheory.OuterMeasure α
                                                                       ⊢ Eq ((MeasureTheory.OuterMeasure.restrict s) (iInf fun i => (MeasureTheory.Ou …
                                                                     -/
    _ = restrict (range ((↑) : s → α)) (⨅ i, restrict s (m i)) := by rw [Subtype.range_coe]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    _ = map ((↑) : s → α) (⨅ i, comap (↑) (m i)) := (map_iInf Subtype.coe_injective _).symm
    _ = restrict s (⨅ i, m i) := congr_arg (map ((↑) : s → α)) (comap_iInf _ _).symm


theorem restrict_iInf {ι} [Nonempty ι] (s : Set α) (m : ι → OuterMeasure α) :
    restrict s (⨅ i, m i) = ⨅ i, restrict s (m i) :=
  (congr_arg (map ((↑) : s → α)) (comap_iInf _ _)).trans (map_iInf_comap _)


theorem restrict_biInf {ι} {I : Set ι} (hI : I.Nonempty) (s : Set α) (m : ι → OuterMeasure α) :
    restrict s (⨅ i ∈ I, m i) = ⨅ i ∈ I, restrict s (m i) := by
  /-
    α : Type u_1
    ι : Type u_2
    I : Set ι
    hI : I.Nonempty
    s : Set α
    m : ι → MeasureTheory.OuterMeasure α
    ⊢ Eq ((MeasureTheory.OuterMeasure.restrict s) (iInf fun i => iInf fun h => m i …
  -/
  haveI := hI.to_subtype
  /-
    α : Type u_1
    ι : Type u_2
    I : Set ι
    hI : I.Nonempty
    s : Set α
    m : ι → MeasureTheory.OuterMeasure α
    this : Nonempty ↑I
    ⊢ Eq ((MeasureTheory.OuterMeasure.restrict s) (iInf fun i => iInf fun h => m i …
  -/
  rw [← iInf_subtype'', ← iInf_subtype'']
  /-
    α : Type u_1
    ι : Type u_2
    I : Set ι
    hI : I.Nonempty
    s : Set α
    m : ι → MeasureTheory.OuterMeasure α
    this : Nonempty ↑I
    ⊢ Eq ((MeasureTheory.OuterMeasure.restrict s) (iInf fun i => m ↑i)) (iInf fun  …
  -/
  exact restrict_iInf _ _
  /-
    🎉 no goals
  -/


/-- This proves that Inf and restrict commute for outer measures, so long as the set of
outer measures is nonempty. -/
theorem restrict_sInf_eq_sInf_restrict (m : Set (OuterMeasure α)) {s : Set α} (hm : m.Nonempty) :
    restrict s (sInf m) = sInf (restrict s '' m) := by
  /-
    α : Type u_1
    m : Set (MeasureTheory.OuterMeasure α)
    s : Set α
    hm : m.Nonempty
    ⊢ Eq ((MeasureTheory.OuterMeasure.restrict s) (InfSet.sInf m)) (InfSet.sInf (S …
  -/
  simp only [sInf_eq_iInf, restrict_biInf, hm, iInf_image]
  /-
    🎉 no goals
  -/


