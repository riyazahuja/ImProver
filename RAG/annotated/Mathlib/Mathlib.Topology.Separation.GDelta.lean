theorem IsGδ.compl_singleton (x : X) [T1Space X] : IsGδ ({x}ᶜ : Set X) :=
  isOpen_compl_singleton.isGδ


@[deprecated (since := "2024-02-15")] alias isGδ_compl_singleton := IsGδ.compl_singleton


theorem Set.Countable.isGδ_compl {s : Set X} [T1Space X] (hs : s.Countable) : IsGδ sᶜ := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : T1Space X
    hs : s.Countable
    ⊢ IsGδ (HasCompl.compl s)
  -/
  rw [← biUnion_of_singleton s, compl_iUnion₂]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : T1Space X
    hs : s.Countable
    ⊢ IsGδ (Set.iInter fun i => Set.iInter fun j => HasCompl.compl (Singleton.sing …
  -/
  exact .biInter hs fun x _ => .compl_singleton x
  /-
    🎉 no goals
  -/


theorem Set.Finite.isGδ_compl {s : Set X} [T1Space X] (hs : s.Finite) : IsGδ sᶜ :=
  hs.countable.isGδ_compl


theorem Set.Subsingleton.isGδ_compl {s : Set X} [T1Space X] (hs : s.Subsingleton) : IsGδ sᶜ :=
  hs.finite.isGδ_compl


theorem Finset.isGδ_compl [T1Space X] (s : Finset X) : IsGδ (sᶜ : Set X) :=
  s.finite_toSet.isGδ_compl


protected theorem IsGδ.singleton [FirstCountableTopology X] [T1Space X] (x : X) :
    IsGδ ({x} : Set X) := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : FirstCountableTopology X
    inst✝ : T1Space X
    x : X
    ⊢ IsGδ (Singleton.singleton x)
  -/
  rcases (nhds_basis_opens x).exists_antitone_subbasis with ⟨U, hU, h_basis⟩
  /-
    case intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : FirstCountableTopology X
    inst✝ : T1Space X
    x : X
    U : Nat → Set X
    hU : ∀ (i : Nat), And (Membership.mem (U i) x) (IsOpen (U i))
    h_basis : (nhds x).HasAntitoneBasis fun i => U i
    ⊢ IsGδ (Singleton.singleton x)
  -/
  rw [← biInter_basis_nhds h_basis.toHasBasis]
  /-
    case intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : FirstCountableTopology X
    inst✝ : T1Space X
    x : X
    U : Nat → Set X
    hU : ∀ (i : Nat), And (Membership.mem (U i) x) (IsOpen (U i))
    h_basis : (nhds x).HasAntitoneBasis fun i => U i
    ⊢ IsGδ (Set.iInter fun i => Set.iInter fun x => U i)
  -/
  exact .biInter (to_countable _) fun n _ => (hU n).2.isGδ
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-15")] alias isGδ_singleton := IsGδ.singleton


theorem Set.Finite.isGδ [FirstCountableTopology X] {s : Set X} [T1Space X] (hs : s.Finite) :
    IsGδ s :=
  Finite.induction_on hs .empty fun _ _ ↦ .union (.singleton _)



/-- A topological space `X` is a *perfectly normal space* provided it is normal and
closed sets are Gδ. -/
class PerfectlyNormalSpace (X : Type u) [TopologicalSpace X] extends NormalSpace X : Prop where
    closed_gdelta : ∀ ⦃h : Set X⦄, IsClosed h → IsGδ h


/-- Lemma that allows the easy conclusion that perfectly normal spaces are completely normal. -/
theorem Disjoint.hasSeparatingCover_closed_gdelta_right {s t : Set X} [NormalSpace X]
    (st_dis : Disjoint s t) (t_cl : IsClosed t) (t_gd : IsGδ t) : HasSeparatingCover s t := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s t : Set X
    inst✝ : NormalSpace X
    st_dis : Disjoint s t
    t_cl : IsClosed t
    t_gd : IsGδ t
    ⊢ HasSeparatingCover s t
  -/
  obtain ⟨T, T_open, T_count, T_int⟩ := t_gd
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s t : Set X
    inst✝ : NormalSpace X
    st_dis : Disjoint s t
    t_cl : IsClosed t
    T : Set (Set X)
    T_open : ∀ (t : Set X), Membership.mem T t → IsOpen t
    T_count : T.Countable
    T_int : Eq t T.sInter
    ⊢ HasSeparatingCover s t
  -/
  rcases T.eq_empty_or_nonempty with rfl | T_nonempty
    /-
      case intro.intro.intro.inl
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s t : Set X
      inst✝ : NormalSpace X
      st_dis : Disjoint s t
      t_cl : IsClosed t
      T_open : ∀ (t : Set X), Membership.mem EmptyCollection.emptyCollection t → IsO …
      T_count : EmptyCollection.emptyCollection.Countable
      T_int : Eq t EmptyCollection.emptyCollection.sInter
      ⊢ HasSeparatingCover s t
    -/
  · rw [T_int, sInter_empty] at st_dis
    /-
      case intro.intro.intro.inl
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s t : Set X
      inst✝ : NormalSpace X
      st_dis : Disjoint s Set.univ
      t_cl : IsClosed t
      T_open : ∀ (t : Set X), Membership.mem EmptyCollection.emptyCollection t → IsO …
      T_count : EmptyCollection.emptyCollection.Countable
      T_int : Eq t EmptyCollection.emptyCollection.sInter
      ⊢ HasSeparatingCover s t
    -/
    rw [(s.disjoint_univ).mp st_dis]
    /-
      case intro.intro.intro.inl
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s t : Set X
      inst✝ : NormalSpace X
      st_dis : Disjoint s Set.univ
      t_cl : IsClosed t
      T_open : ∀ (t : Set X), Membership.mem EmptyCollection.emptyCollection t → IsO …
      T_count : EmptyCollection.emptyCollection.Countable
      T_int : Eq t EmptyCollection.emptyCollection.sInter
      ⊢ HasSeparatingCover EmptyCollection.emptyCollection t
    -/
    exact t.hasSeparatingCover_empty_left
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.inr
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s t : Set X
    inst✝ : NormalSpace X
    st_dis : Disjoint s t
    t_cl : IsClosed t
    T : Set (Set X)
    T_open : ∀ (t : Set X), Membership.mem T t → IsOpen t
    T_count : T.Countable
    T_int : Eq t T.sInter
    T_nonempty : T.Nonempty
    ⊢ HasSeparatingCover s t
  -/
  obtain ⟨g, g_surj⟩ := T_count.exists_surjective T_nonempty
  choose g' g'_open clt_sub_g' clg'_sub_g using fun n ↦ by
    apply normal_exists_closure_subset t_cl (T_open (g n).1 (g n).2)
    rw [T_int]
    exact sInter_subset_of_mem (g n).2
  have clg'_int : t = ⋂ i, closure (g' i) := by
    apply (subset_iInter fun n ↦ (clt_sub_g' n).trans subset_closure).antisymm
    rw [T_int]
    refine subset_sInter fun t tinT ↦ ?_
    obtain ⟨n, gn⟩ := g_surj ⟨t, tinT⟩
    refine iInter_subset_of_subset n <| (clg'_sub_g n).trans ?_
    rw [gn]
  /-
    case intro.intro.intro.inr.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s t : Set X
    inst✝ : NormalSpace X
    st_dis : Disjoint s t
    t_cl : IsClosed t
    T : Set (Set X)
    T_open : ∀ (t : Set X), Membership.mem T t → IsOpen t
    T_count : T.Countable
    T_int : Eq t T.sInter
    T_nonempty : T.Nonempty
    g : Nat → ↑T
    g_surj : Function.Surjective g
    g' : Nat → Set X
    g'_open : ∀ (n : Nat), IsOpen (g' n)
    clt_sub_g' : ∀ (n : Nat), HasSubset.Subset t (g' n)
    clg'_sub_g : ∀ (n : Nat), HasSubset.Subset (closure (g' n)) ↑(g n)
    clg'_int : Eq t (Set.iInter fun i => closure (g' i))
    ⊢ HasSeparatingCover s t
  -/
  use fun n ↦ (closure (g' n))ᶜ
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    s t : Set X
    inst✝ : NormalSpace X
    st_dis : Disjoint s t
    t_cl : IsClosed t
    T : Set (Set X)
    T_open : ∀ (t : Set X), Membership.mem T t → IsOpen t
    T_count : T.Countable
    T_int : Eq t T.sInter
    T_nonempty : T.Nonempty
    g : Nat → ↑T
    g_surj : Function.Surjective g
    g' : Nat → Set X
    g'_open : ∀ (n : Nat), IsOpen (g' n)
    clt_sub_g' : ∀ (n : Nat), HasSubset.Subset t (g' n)
    clg'_sub_g : ∀ (n : Nat), HasSubset.Subset (closure (g' n)) ↑(g n)
    clg'_int : Eq t (Set.iInter fun i => closure (g' i))
    ⊢ And (HasSubset.Subset s (Set.iUnion fun n => (fun n => HasCompl.compl (closu …
  -/
  constructor
    /-
      case h.left
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s t : Set X
      inst✝ : NormalSpace X
      st_dis : Disjoint s t
      t_cl : IsClosed t
      T : Set (Set X)
      T_open : ∀ (t : Set X), Membership.mem T t → IsOpen t
      T_count : T.Countable
      T_int : Eq t T.sInter
      T_nonempty : T.Nonempty
      g : Nat → ↑T
      g_surj : Function.Surjective g
      g' : Nat → Set X
      g'_open : ∀ (n : Nat), IsOpen (g' n)
      clt_sub_g' : ∀ (n : Nat), HasSubset.Subset t (g' n)
      clg'_sub_g : ∀ (n : Nat), HasSubset.Subset (closure (g' n)) ↑(g n)
      clg'_int : Eq t (Set.iInter fun i => closure (g' i))
      ⊢ HasSubset.Subset s (Set.iUnion fun n => HasCompl.compl (closure (g' n)))
    -/
  · rw [← compl_iInter, subset_compl_comm, ← clg'_int]
    /-
      case h.left
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s t : Set X
      inst✝ : NormalSpace X
      st_dis : Disjoint s t
      t_cl : IsClosed t
      T : Set (Set X)
      T_open : ∀ (t : Set X), Membership.mem T t → IsOpen t
      T_count : T.Countable
      T_int : Eq t T.sInter
      T_nonempty : T.Nonempty
      g : Nat → ↑T
      g_surj : Function.Surjective g
      g' : Nat → Set X
      g'_open : ∀ (n : Nat), IsOpen (g' n)
      clt_sub_g' : ∀ (n : Nat), HasSubset.Subset t (g' n)
      clg'_sub_g : ∀ (n : Nat), HasSubset.Subset (closure (g' n)) ↑(g n)
      clg'_int : Eq t (Set.iInter fun i => closure (g' i))
      ⊢ HasSubset.Subset t (HasCompl.compl s)
    -/
    exact st_dis.subset_compl_left
    /-
      🎉 no goals
    -/
    /-
      case h.right
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s t : Set X
      inst✝ : NormalSpace X
      st_dis : Disjoint s t
      t_cl : IsClosed t
      T : Set (Set X)
      T_open : ∀ (t : Set X), Membership.mem T t → IsOpen t
      T_count : T.Countable
      T_int : Eq t T.sInter
      T_nonempty : T.Nonempty
      g : Nat → ↑T
      g_surj : Function.Surjective g
      g' : Nat → Set X
      g'_open : ∀ (n : Nat), IsOpen (g' n)
      clt_sub_g' : ∀ (n : Nat), HasSubset.Subset t (g' n)
      clg'_sub_g : ∀ (n : Nat), HasSubset.Subset (closure (g' n)) ↑(g n)
      clg'_int : Eq t (Set.iInter fun i => closure (g' i))
      ⊢ ∀ (n : Nat), And (IsOpen (HasCompl.compl (closure (g' n)))) (Disjoint (closu …
    -/
  · refine fun n ↦ ⟨isOpen_compl_iff.mpr isClosed_closure, ?_⟩
    /-
      case h.right
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s t : Set X
      inst✝ : NormalSpace X
      st_dis : Disjoint s t
      t_cl : IsClosed t
      T : Set (Set X)
      T_open : ∀ (t : Set X), Membership.mem T t → IsOpen t
      T_count : T.Countable
      T_int : Eq t T.sInter
      T_nonempty : T.Nonempty
      g : Nat → ↑T
      g_surj : Function.Surjective g
      g' : Nat → Set X
      g'_open : ∀ (n : Nat), IsOpen (g' n)
      clt_sub_g' : ∀ (n : Nat), HasSubset.Subset t (g' n)
      clg'_sub_g : ∀ (n : Nat), HasSubset.Subset (closure (g' n)) ↑(g n)
      clg'_int : Eq t (Set.iInter fun i => closure (g' i))
      n : Nat
      ⊢ Disjoint (closure (HasCompl.compl (closure (g' n)))) t
    -/
    simp only [closure_compl, disjoint_compl_left_iff_subset]
    /-
      case h.right
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s t : Set X
      inst✝ : NormalSpace X
      st_dis : Disjoint s t
      t_cl : IsClosed t
      T : Set (Set X)
      T_open : ∀ (t : Set X), Membership.mem T t → IsOpen t
      T_count : T.Countable
      T_int : Eq t T.sInter
      T_nonempty : T.Nonempty
      g : Nat → ↑T
      g_surj : Function.Surjective g
      g' : Nat → Set X
      g'_open : ∀ (n : Nat), IsOpen (g' n)
      clt_sub_g' : ∀ (n : Nat), HasSubset.Subset t (g' n)
      clg'_sub_g : ∀ (n : Nat), HasSubset.Subset (closure (g' n)) ↑(g n)
      clg'_int : Eq t (Set.iInter fun i => closure (g' i))
      n : Nat
      ⊢ HasSubset.Subset t (interior (closure (g' n)))
    -/
    rw [← closure_eq_iff_isClosed.mpr t_cl] at clt_sub_g'
    /-
      case h.right
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      s t : Set X
      inst✝ : NormalSpace X
      st_dis : Disjoint s t
      t_cl : IsClosed t
      T : Set (Set X)
      T_open : ∀ (t : Set X), Membership.mem T t → IsOpen t
      T_count : T.Countable
      T_int : Eq t T.sInter
      T_nonempty : T.Nonempty
      g : Nat → ↑T
      g_surj : Function.Surjective g
      g' : Nat → Set X
      g'_open : ∀ (n : Nat), IsOpen (g' n)
      clt_sub_g' : ∀ (n : Nat), HasSubset.Subset (closure t) (g' n)
      clg'_sub_g : ∀ (n : Nat), HasSubset.Subset (closure (g' n)) ↑(g n)
      clg'_int : Eq t (Set.iInter fun i => closure (g' i))
      n : Nat
      ⊢ HasSubset.Subset t (interior (closure (g' n)))
    -/
    exact subset_closure.trans <| (clt_sub_g' n).trans <| (g'_open n).subset_interior_closure
    /-
      🎉 no goals
    -/


instance (priority := 100) PerfectlyNormalSpace.toCompletelyNormalSpace
    [PerfectlyNormalSpace X] : CompletelyNormalSpace X where
  completely_normal _ _ hd₁ hd₂ := separatedNhds_iff_disjoint.mp <|
    hasSeparatingCovers_iff_separatedNhds.mp
      ⟨(hd₂.hasSeparatingCover_closed_gdelta_right isClosed_closure <|
         closed_gdelta isClosed_closure).mono (fun ⦃_⦄ a ↦ a) subset_closure,
       ((Disjoint.symm hd₁).hasSeparatingCover_closed_gdelta_right isClosed_closure <|
         closed_gdelta isClosed_closure).mono (fun ⦃_⦄ a ↦ a) subset_closure⟩


/-- A T₆ space is a perfectly normal T₁ space. -/
class T6Space (X : Type u) [TopologicalSpace X] extends T1Space X, PerfectlyNormalSpace X : Prop

-- see Note [lower instance priority]

/-- A `T₆` space is a `T₅` space. -/
instance (priority := 100) T6Space.toT5Space [T6Space X] : T5Space X where
  -- follows from type-class inference

