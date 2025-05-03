theorem pairwise_iUnion {f : κ → Set α} (h : Directed (· ⊆ ·) f) :
    (⋃ n, f n).Pairwise r ↔ ∀ n, (f n).Pairwise r := by
  /-
    α : Type u_1
    κ : Sort u_4
    r : α → α → Prop
    f : κ → Set α
    h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) f
    ⊢ Iff ((Set.iUnion fun n => f n).Pairwise r) (∀ (n : κ), (f n).Pairwise r)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      κ : Sort u_4
      r : α → α → Prop
      f : κ → Set α
      h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) f
      ⊢ (Set.iUnion fun n => f n).Pairwise r → ∀ (n : κ), (f n).Pairwise r
    -/
  · intro H n
    /-
      case mp
      α : Type u_1
      κ : Sort u_4
      r : α → α → Prop
      f : κ → Set α
      h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) f
      H : (Set.iUnion fun n => f n).Pairwise r
      n : κ
      ⊢ (f n).Pairwise r
    -/
    exact Pairwise.mono (subset_iUnion _ _) H
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      κ : Sort u_4
      r : α → α → Prop
      f : κ → Set α
      h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) f
      ⊢ (∀ (n : κ), (f n).Pairwise r) → (Set.iUnion fun n => f n).Pairwise r
    -/
  · intro H i hi j hj hij
    /-
      case mpr
      α : Type u_1
      κ : Sort u_4
      r : α → α → Prop
      f : κ → Set α
      h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) f
      H : ∀ (n : κ), (f n).Pairwise r
      i : α
      hi : Membership.mem (Set.iUnion fun n => f n) i
      j : α
      hj : Membership.mem (Set.iUnion fun n => f n) j
      hij : Ne i j
      ⊢ r i j
    -/
    rcases mem_iUnion.1 hi with ⟨m, hm⟩
    /-
      case mpr.intro
      α : Type u_1
      κ : Sort u_4
      r : α → α → Prop
      f : κ → Set α
      h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) f
      H : ∀ (n : κ), (f n).Pairwise r
      i : α
      hi : Membership.mem (Set.iUnion fun n => f n) i
      j : α
      hj : Membership.mem (Set.iUnion fun n => f n) j
      hij : Ne i j
      m : κ
      hm : Membership.mem (f m) i
      ⊢ r i j
    -/
    rcases mem_iUnion.1 hj with ⟨n, hn⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      κ : Sort u_4
      r : α → α → Prop
      f : κ → Set α
      h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) f
      H : ∀ (n : κ), (f n).Pairwise r
      i : α
      hi : Membership.mem (Set.iUnion fun n => f n) i
      j : α
      hj : Membership.mem (Set.iUnion fun n => f n) j
      hij : Ne i j
      m : κ
      hm : Membership.mem (f m) i
      n : κ
      hn : Membership.mem (f n) j
      ⊢ r i j
    -/
    rcases h m n with ⟨p, mp, np⟩
    /-
      case mpr.intro.intro.intro.intro
      α : Type u_1
      κ : Sort u_4
      r : α → α → Prop
      f : κ → Set α
      h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) f
      H : ∀ (n : κ), (f n).Pairwise r
      i : α
      hi : Membership.mem (Set.iUnion fun n => f n) i
      j : α
      hj : Membership.mem (Set.iUnion fun n => f n) j
      hij : Ne i j
      m : κ
      hm : Membership.mem (f m) i
      n : κ
      hn : Membership.mem (f n) j
      p : κ
      mp : HasSubset.Subset (f m) (f p)
      np : HasSubset.Subset (f n) (f p)
      ⊢ r i j
    -/
    exact H p (mp hm) (np hn) hij
    /-
      🎉 no goals
    -/


theorem pairwise_sUnion {r : α → α → Prop} {s : Set (Set α)} (h : DirectedOn (· ⊆ ·) s) :
    (⋃₀ s).Pairwise r ↔ ∀ a ∈ s, Set.Pairwise a r := by
  /-
    α : Type u_1
    r : α → α → Prop
    s : Set (Set α)
    h : DirectedOn (fun x1 x2 => HasSubset.Subset x1 x2) s
    ⊢ Iff (s.sUnion.Pairwise r) (∀ (a : Set α), Membership.mem s a → a.Pairwise r)
  -/
  rw [sUnion_eq_iUnion, pairwise_iUnion h.directed_val, SetCoe.forall]
  /-
    🎉 no goals
  -/


theorem pairwiseDisjoint_iUnion {g : ι' → Set ι} (h : Directed (· ⊆ ·) g) :
    (⋃ n, g n).PairwiseDisjoint f ↔ ∀ ⦃n⦄, (g n).PairwiseDisjoint f :=
  pairwise_iUnion h


theorem pairwiseDisjoint_sUnion {s : Set (Set ι)} (h : DirectedOn (· ⊆ ·) s) :
    (⋃₀ s).PairwiseDisjoint f ↔ ∀ ⦃a⦄, a ∈ s → Set.PairwiseDisjoint a f :=
  pairwise_sUnion h


/-- Bind operation for `Set.PairwiseDisjoint`. If you want to only consider finsets of indices, you
can use `Set.PairwiseDisjoint.biUnion_finset`. -/
theorem PairwiseDisjoint.biUnion {s : Set ι'} {g : ι' → Set ι} {f : ι → α}
    (hs : s.PairwiseDisjoint fun i' : ι' => ⨆ i ∈ g i', f i)
    (hg : ∀ i ∈ s, (g i).PairwiseDisjoint f) : (⋃ i ∈ s, g i).PairwiseDisjoint f := by
  /-
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝ : CompleteLattice α
    s : Set ι'
    g : ι' → Set ι
    f : ι → α
    hs : s.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f i
    hg : ∀ (i : ι'), Membership.mem s i → (g i).PairwiseDisjoint f
    ⊢ (Set.iUnion fun i => Set.iUnion fun h => g i).PairwiseDisjoint f
  -/
  rintro a ha b hb hab
  /-
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝ : CompleteLattice α
    s : Set ι'
    g : ι' → Set ι
    f : ι → α
    hs : s.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f i
    hg : ∀ (i : ι'), Membership.mem s i → (g i).PairwiseDisjoint f
    a : ι
    ha : Membership.mem (Set.iUnion fun i => Set.iUnion fun h => g i) a
    b : ι
    hb : Membership.mem (Set.iUnion fun i => Set.iUnion fun h => g i) b
    hab : Ne a b
    ⊢ Function.onFun Disjoint f a b
  -/
  simp_rw [Set.mem_iUnion] at ha hb
  /-
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝ : CompleteLattice α
    s : Set ι'
    g : ι' → Set ι
    f : ι → α
    hs : s.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f i
    hg : ∀ (i : ι'), Membership.mem s i → (g i).PairwiseDisjoint f
    a b : ι
    hab : Ne a b
    ha : Exists fun i => Exists fun i_1 => Membership.mem (g i) a
    hb : Exists fun i => Exists fun i_1 => Membership.mem (g i) b
    ⊢ Function.onFun Disjoint f a b
  -/
  obtain ⟨c, hc, ha⟩ := ha
  /-
    case intro.intro
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝ : CompleteLattice α
    s : Set ι'
    g : ι' → Set ι
    f : ι → α
    hs : s.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f i
    hg : ∀ (i : ι'), Membership.mem s i → (g i).PairwiseDisjoint f
    a b : ι
    hab : Ne a b
    hb : Exists fun i => Exists fun i_1 => Membership.mem (g i) b
    c : ι'
    hc : Membership.mem s c
    ha : Membership.mem (g c) a
    ⊢ Function.onFun Disjoint f a b
  -/
  obtain ⟨d, hd, hb⟩ := hb
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝ : CompleteLattice α
    s : Set ι'
    g : ι' → Set ι
    f : ι → α
    hs : s.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f i
    hg : ∀ (i : ι'), Membership.mem s i → (g i).PairwiseDisjoint f
    a b : ι
    hab : Ne a b
    c : ι'
    hc : Membership.mem s c
    ha : Membership.mem (g c) a
    d : ι'
    hd : Membership.mem s d
    hb : Membership.mem (g d) b
    ⊢ Function.onFun Disjoint f a b
  -/
  obtain hcd | hcd := eq_or_ne (g c) (g d)
    /-
      case intro.intro.intro.intro.inl
      α : Type u_1
      ι : Type u_2
      ι' : Type u_3
      inst✝ : CompleteLattice α
      s : Set ι'
      g : ι' → Set ι
      f : ι → α
      hs : s.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f i
      hg : ∀ (i : ι'), Membership.mem s i → (g i).PairwiseDisjoint f
      a b : ι
      hab : Ne a b
      c : ι'
      hc : Membership.mem s c
      ha : Membership.mem (g c) a
      d : ι'
      hd : Membership.mem s d
      hb : Membership.mem (g d) b
      hcd : Eq (g c) (g d)
      ⊢ Function.onFun Disjoint f a b
    -/
  · exact hg d hd (hcd ▸ ha) hb hab
    /-
      🎉 no goals
    -/
  -- Porting note: the elaborator couldn't figure out `f` here.
  · exact (hs hc hd <| ne_of_apply_ne _ hcd).mono
      (le_iSup₂ (f := fun i (_ : i ∈ g c) => f i) a ha)
      (le_iSup₂ (f := fun i (_ : i ∈ g d) => f i) b hb)


/-- If the suprema of columns are pairwise disjoint and suprema of rows as well, then everything is
pairwise disjoint. Not to be confused with `Set.PairwiseDisjoint.prod`. -/
theorem PairwiseDisjoint.prod_left {f : ι × ι' → α}
    (hs : s.PairwiseDisjoint fun i => ⨆ i' ∈ t, f (i, i'))
    (ht : t.PairwiseDisjoint fun i' => ⨆ i ∈ s, f (i, i')) :
    (s ×ˢ t : Set (ι × ι')).PairwiseDisjoint f := by
  /-
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝ : CompleteLattice α
    s : Set ι
    t : Set ι'
    f : Prod ι ι' → α
    hs : s.PairwiseDisjoint fun i => iSup fun i' => iSup fun h => f { fst := i, sn …
    ht : t.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f { fst := i, sn …
    ⊢ (SProd.sprod s t).PairwiseDisjoint f
  -/
  rintro ⟨i, i'⟩ hi ⟨j, j'⟩ hj h
  /-
    case mk.mk
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝ : CompleteLattice α
    s : Set ι
    t : Set ι'
    f : Prod ι ι' → α
    hs : s.PairwiseDisjoint fun i => iSup fun i' => iSup fun h => f { fst := i, sn …
    ht : t.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f { fst := i, sn …
    i : ι
    i' : ι'
    hi : Membership.mem (SProd.sprod s t) { fst := i, snd := i' }
    j : ι
    j' : ι'
    hj : Membership.mem (SProd.sprod s t) { fst := j, snd := j' }
    h : Ne { fst := i, snd := i' } { fst := j, snd := j' }
    ⊢ Function.onFun Disjoint f { fst := i, snd := i' } { fst := j, snd := j' }
  -/
  rw [mem_prod] at hi hj
  /-
    case mk.mk
    α : Type u_1
    ι : Type u_2
    ι' : Type u_3
    inst✝ : CompleteLattice α
    s : Set ι
    t : Set ι'
    f : Prod ι ι' → α
    hs : s.PairwiseDisjoint fun i => iSup fun i' => iSup fun h => f { fst := i, sn …
    ht : t.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f { fst := i, sn …
    i : ι
    i' : ι'
    hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
    j : ι
    j' : ι'
    hj : And (Membership.mem s { fst := j, snd := j' }.1) (Membership.mem t { fst  …
    h : Ne { fst := i, snd := i' } { fst := j, snd := j' }
    ⊢ Function.onFun Disjoint f { fst := i, snd := i' } { fst := j, snd := j' }
  -/
  obtain rfl | hij := eq_or_ne i j
    /-
      case mk.mk.inl
      α : Type u_1
      ι : Type u_2
      ι' : Type u_3
      inst✝ : CompleteLattice α
      s : Set ι
      t : Set ι'
      f : Prod ι ι' → α
      hs : s.PairwiseDisjoint fun i => iSup fun i' => iSup fun h => f { fst := i, sn …
      ht : t.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f { fst := i, sn …
      i : ι
      i' : ι'
      hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
      j' : ι'
      hj : And (Membership.mem s { fst := i, snd := j' }.1) (Membership.mem t { fst  …
      h : Ne { fst := i, snd := i' } { fst := i, snd := j' }
      ⊢ Function.onFun Disjoint f { fst := i, snd := i' } { fst := i, snd := j' }
    -/
  · refine (ht hi.2 hj.2 <| (Prod.mk.inj_left _).ne_iff.1 h).mono ?_ ?_
      /-
        case mk.mk.inl.refine_1
        α : Type u_1
        ι : Type u_2
        ι' : Type u_3
        inst✝ : CompleteLattice α
        s : Set ι
        t : Set ι'
        f : Prod ι ι' → α
        hs : s.PairwiseDisjoint fun i => iSup fun i' => iSup fun h => f { fst := i, sn …
        ht : t.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f { fst := i, sn …
        i : ι
        i' : ι'
        hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
        j' : ι'
        hj : And (Membership.mem s { fst := i, snd := j' }.1) (Membership.mem t { fst  …
        h : Ne { fst := i, snd := i' } { fst := i, snd := j' }
        ⊢ LE.le (f { fst := i, snd := i' }) ((fun i' => iSup fun i => iSup fun h => f  …
      -/
    · convert le_iSup₂ (α := α) i hi.1; rfl
                                        /-
                                          🎉 no goals
                                        -/
      /-
        case mk.mk.inl.refine_2
        α : Type u_1
        ι : Type u_2
        ι' : Type u_3
        inst✝ : CompleteLattice α
        s : Set ι
        t : Set ι'
        f : Prod ι ι' → α
        hs : s.PairwiseDisjoint fun i => iSup fun i' => iSup fun h => f { fst := i, sn …
        ht : t.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f { fst := i, sn …
        i : ι
        i' : ι'
        hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
        j' : ι'
        hj : And (Membership.mem s { fst := i, snd := j' }.1) (Membership.mem t { fst  …
        h : Ne { fst := i, snd := i' } { fst := i, snd := j' }
        ⊢ LE.le (f { fst := i, snd := j' }) ((fun i' => iSup fun i => iSup fun h => f  …
      -/
    · convert le_iSup₂ (α := α) i hj.1; rfl
                                        /-
                                          🎉 no goals
                                        -/
    /-
      case mk.mk.inr
      α : Type u_1
      ι : Type u_2
      ι' : Type u_3
      inst✝ : CompleteLattice α
      s : Set ι
      t : Set ι'
      f : Prod ι ι' → α
      hs : s.PairwiseDisjoint fun i => iSup fun i' => iSup fun h => f { fst := i, sn …
      ht : t.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f { fst := i, sn …
      i : ι
      i' : ι'
      hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
      j : ι
      j' : ι'
      hj : And (Membership.mem s { fst := j, snd := j' }.1) (Membership.mem t { fst  …
      h : Ne { fst := i, snd := i' } { fst := j, snd := j' }
      hij : Ne i j
      ⊢ Function.onFun Disjoint f { fst := i, snd := i' } { fst := j, snd := j' }
    -/
  · refine (hs hi.1 hj.1 hij).mono ?_ ?_
      /-
        case mk.mk.inr.refine_1
        α : Type u_1
        ι : Type u_2
        ι' : Type u_3
        inst✝ : CompleteLattice α
        s : Set ι
        t : Set ι'
        f : Prod ι ι' → α
        hs : s.PairwiseDisjoint fun i => iSup fun i' => iSup fun h => f { fst := i, sn …
        ht : t.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f { fst := i, sn …
        i : ι
        i' : ι'
        hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
        j : ι
        j' : ι'
        hj : And (Membership.mem s { fst := j, snd := j' }.1) (Membership.mem t { fst  …
        h : Ne { fst := i, snd := i' } { fst := j, snd := j' }
        hij : Ne i j
        ⊢ LE.le (f { fst := i, snd := i' }) ((fun i => iSup fun i' => iSup fun h => f  …
      -/
    · convert le_iSup₂ (α := α) i' hi.2; rfl
                                         /-
                                           🎉 no goals
                                         -/
      /-
        case mk.mk.inr.refine_2
        α : Type u_1
        ι : Type u_2
        ι' : Type u_3
        inst✝ : CompleteLattice α
        s : Set ι
        t : Set ι'
        f : Prod ι ι' → α
        hs : s.PairwiseDisjoint fun i => iSup fun i' => iSup fun h => f { fst := i, sn …
        ht : t.PairwiseDisjoint fun i' => iSup fun i => iSup fun h => f { fst := i, sn …
        i : ι
        i' : ι'
        hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
        j : ι
        j' : ι'
        hj : And (Membership.mem s { fst := j, snd := j' }.1) (Membership.mem t { fst  …
        h : Ne { fst := i, snd := i' } { fst := j, snd := j' }
        hij : Ne i j
        ⊢ LE.le (f { fst := j, snd := j' }) ((fun i => iSup fun i' => iSup fun h => f  …
      -/
    · convert le_iSup₂ (α := α) j' hj.2; rfl
                                         /-
                                           🎉 no goals
                                         -/


theorem pairwiseDisjoint_prod_left {s : Set ι} {t : Set ι'} {f : ι × ι' → α} :
    (s ×ˢ t : Set (ι × ι')).PairwiseDisjoint f ↔
      (s.PairwiseDisjoint fun i => ⨆ i' ∈ t, f (i, i')) ∧
        t.PairwiseDisjoint fun i' => ⨆ i ∈ s, f (i, i') := by
  refine
      ⟨fun h => ⟨fun i hi j hj hij => ?_, fun i hi j hj hij => ?_⟩, fun h => h.1.prod_left h.2⟩ <;>
    /-
      case refine_1
      α : Type u_1
      ι : Type u_2
      ι' : Type u_3
      inst✝ : Order.Frame α
      s : Set ι
      t : Set ι'
      f : Prod ι ι' → α
      h : (SProd.sprod s t).PairwiseDisjoint f
      i : ι
      hi : Membership.mem s i
      j : ι
      hj : Membership.mem s j
      hij : Ne i j
      ⊢ Function.onFun Disjoint (fun i => iSup fun i' => iSup fun h => f { fst := i, …
    -/
    simp_rw [Function.onFun, iSup_disjoint_iff, disjoint_iSup_iff] <;>
    /-
      case refine_1
      α : Type u_1
      ι : Type u_2
      ι' : Type u_3
      inst✝ : Order.Frame α
      s : Set ι
      t : Set ι'
      f : Prod ι ι' → α
      h : (SProd.sprod s t).PairwiseDisjoint f
      i : ι
      hi : Membership.mem s i
      j : ι
      hj : Membership.mem s j
      hij : Ne i j
      ⊢ ∀ (i_1 : ι'), Membership.mem t i_1 → ∀ (i_3 : ι'), Membership.mem t i_3 → Di …
    -/
    intro i' hi' j' hj'
    /-
      case refine_1
      α : Type u_1
      ι : Type u_2
      ι' : Type u_3
      inst✝ : Order.Frame α
      s : Set ι
      t : Set ι'
      f : Prod ι ι' → α
      h : (SProd.sprod s t).PairwiseDisjoint f
      i : ι
      hi : Membership.mem s i
      j : ι
      hj : Membership.mem s j
      hij : Ne i j
      i' : ι'
      hi' : Membership.mem t i'
      j' : ι'
      hj' : Membership.mem t j'
      ⊢ Disjoint (f { fst := i, snd := i' }) (f { fst := j, snd := j' })
    -/
  · exact h (mk_mem_prod hi hi') (mk_mem_prod hj hj') (ne_of_apply_ne Prod.fst hij)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      ι : Type u_2
      ι' : Type u_3
      inst✝ : Order.Frame α
      s : Set ι
      t : Set ι'
      f : Prod ι ι' → α
      h : (SProd.sprod s t).PairwiseDisjoint f
      i : ι'
      hi : Membership.mem t i
      j : ι'
      hj : Membership.mem t j
      hij : Ne i j
      i' : ι
      hi' : Membership.mem s i'
      j' : ι
      hj' : Membership.mem s j'
      ⊢ Disjoint (f { fst := i', snd := i }) (f { fst := j', snd := j })
    -/
  · exact h (mk_mem_prod hi' hi) (mk_mem_prod hj' hj) (ne_of_apply_ne Prod.snd hij)
    /-
      🎉 no goals
    -/


theorem biUnion_diff_biUnion_eq {s t : Set ι} {f : ι → Set α} (h : (s ∪ t).PairwiseDisjoint f) :
    ((⋃ i ∈ s, f i) \ ⋃ i ∈ t, f i) = ⋃ i ∈ s \ t, f i := by
  refine
    (biUnion_diff_biUnion_subset f s t).antisymm
      (iUnion₂_subset fun i hi a ha => (mem_diff _).2 ⟨mem_biUnion hi.1 ha, ?_⟩)
  /-
    α : Type u_1
    ι : Type u_2
    s t : Set ι
    f : ι → Set α
    h : (Union.union s t).PairwiseDisjoint f
    i : ι
    hi : Membership.mem (SDiff.sdiff s t) i
    a : α
    ha : Membership.mem (f i) a
    ⊢ Not (Membership.mem (Set.iUnion fun x => Set.iUnion fun h => f x) a)
  -/
  rw [mem_iUnion₂]; rintro ⟨j, hj, haj⟩
  /-
    case intro.intro
    α : Type u_1
    ι : Type u_2
    s t : Set ι
    f : ι → Set α
    h : (Union.union s t).PairwiseDisjoint f
    i : ι
    hi : Membership.mem (SDiff.sdiff s t) i
    a : α
    ha : Membership.mem (f i) a
    j : ι
    hj : Membership.mem t j
    haj : Membership.mem (f j) a
    ⊢ False
  -/
  exact (h (Or.inl hi.1) (Or.inr hj) (ne_of_mem_of_not_mem hj hi.2).symm).le_bot ⟨ha, haj⟩
  /-
    🎉 no goals
  -/



/-- Equivalence between a disjoint bounded union and a dependent sum. -/
noncomputable def biUnionEqSigmaOfDisjoint {s : Set ι} {f : ι → Set α} (h : s.PairwiseDisjoint f) :
    (⋃ i ∈ s, f i) ≃ Σi : s, f i :=
  (Equiv.setCongr (biUnion_eq_iUnion _ _)).trans <|
    unionEqSigmaOfDisjoint fun ⟨_i, hi⟩ ⟨_j, hj⟩ ne => h hi hj fun eq => ne <| Subtype.eq eq


theorem Set.PairwiseDisjoint.subset_of_biUnion_subset_biUnion (h₀ : (s ∪ t).PairwiseDisjoint f)
    (h₁ : ∀ i ∈ s, (f i).Nonempty) (h : ⋃ i ∈ s, f i ⊆ ⋃ i ∈ t, f i) : s ⊆ t := by
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → Set α
    s t : Set ι
    h₀ : (Union.union s t).PairwiseDisjoint f
    h₁ : ∀ (i : ι), Membership.mem s i → (f i).Nonempty
    h : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set.iUnion …
    ⊢ HasSubset.Subset s t
  -/
  rintro i hi
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → Set α
    s t : Set ι
    h₀ : (Union.union s t).PairwiseDisjoint f
    h₁ : ∀ (i : ι), Membership.mem s i → (f i).Nonempty
    h : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set.iUnion …
    i : ι
    hi : Membership.mem s i
    ⊢ Membership.mem t i
  -/
  obtain ⟨a, hai⟩ := h₁ i hi
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    f : ι → Set α
    s t : Set ι
    h₀ : (Union.union s t).PairwiseDisjoint f
    h₁ : ∀ (i : ι), Membership.mem s i → (f i).Nonempty
    h : HasSubset.Subset (Set.iUnion fun i => Set.iUnion fun h => f i) (Set.iUnion …
    i : ι
    hi : Membership.mem s i
    a : α
    hai : Membership.mem (f i) a
    ⊢ Membership.mem t i
  -/
  obtain ⟨j, hj, haj⟩ := mem_iUnion₂.1 (h <| mem_iUnion₂_of_mem hi hai)
  rwa [h₀.eq (subset_union_left hi) (subset_union_right hj)
      (not_disjoint_iff.2 ⟨a, hai, haj⟩)]


theorem Pairwise.subset_of_biUnion_subset_biUnion (h₀ : Pairwise (Disjoint on f))
    (h₁ : ∀ i ∈ s, (f i).Nonempty) (h : ⋃ i ∈ s, f i ⊆ ⋃ i ∈ t, f i) : s ⊆ t :=
  Set.PairwiseDisjoint.subset_of_biUnion_subset_biUnion (h₀.set_pairwise _) h₁ h


theorem Pairwise.biUnion_injective (h₀ : Pairwise (Disjoint on f)) (h₁ : ∀ i, (f i).Nonempty) :
    Injective fun s : Set ι => ⋃ i ∈ s, f i := fun _s _t h =>
  ((h₀.subset_of_biUnion_subset_biUnion fun _ _ => h₁ _) <| h.subset).antisymm <|
    (h₀.subset_of_biUnion_subset_biUnion fun _ _ => h₁ _) <| h.superset


/-- In a disjoint union we can identify the unique set an element belongs to. -/
theorem pairwiseDisjoint_unique {y : α}
    (h_disjoint : PairwiseDisjoint s f)
    (hy : y ∈ (⋃ i ∈ s, f i)) : ∃! i, i ∈ s ∧ y ∈ f i := by
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → Set α
    s : Set ι
    y : α
    h_disjoint : s.PairwiseDisjoint f
    hy : Membership.mem (Set.iUnion fun i => Set.iUnion fun h => f i) y
    ⊢ ExistsUnique fun i => And (Membership.mem s i) (Membership.mem (f i) y)
  -/
  refine existsUnique_of_exists_of_unique ?ex ?unique
    /-
      case ex
      α : Type u_1
      ι : Type u_2
      f : ι → Set α
      s : Set ι
      y : α
      h_disjoint : s.PairwiseDisjoint f
      hy : Membership.mem (Set.iUnion fun i => Set.iUnion fun h => f i) y
      ⊢ Exists fun x => And (Membership.mem s x) (Membership.mem (f x) y)
    -/
  · simpa only [mem_iUnion, exists_prop] using hy
    /-
      🎉 no goals
    -/
    /-
      case unique
      α : Type u_1
      ι : Type u_2
      f : ι → Set α
      s : Set ι
      y : α
      h_disjoint : s.PairwiseDisjoint f
      hy : Membership.mem (Set.iUnion fun i => Set.iUnion fun h => f i) y
      ⊢ ∀ (y₁ y₂ : ι), And (Membership.mem s y₁) (Membership.mem (f y₁) y) → And (Me …
    -/
  · rintro i j ⟨his, hi⟩ ⟨hjs, hj⟩
    /-
      case unique.intro.intro
      α : Type u_1
      ι : Type u_2
      f : ι → Set α
      s : Set ι
      y : α
      h_disjoint : s.PairwiseDisjoint f
      hy : Membership.mem (Set.iUnion fun i => Set.iUnion fun h => f i) y
      i j : ι
      his : Membership.mem s i
      hi : Membership.mem (f i) y
      hjs : Membership.mem s j
      hj : Membership.mem (f j) y
      ⊢ Eq i j
    -/
    exact h_disjoint.elim his hjs <| not_disjoint_iff.mpr ⟨y, ⟨hi, hj⟩⟩
    /-
      🎉 no goals
    -/


