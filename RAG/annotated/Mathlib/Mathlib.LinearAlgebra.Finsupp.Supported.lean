/-- `Finsupp.supported M R s` is the `R`-submodule of all `p : α →₀ M` such that `p.support ⊆ s`. -/
def supported (s : Set α) : Submodule R (α →₀ M) where
  carrier := { p | ↑p.support ⊆ s }
  add_mem' {p q} hp hq := by
    classical
    refine Subset.trans (Subset.trans (Finset.coe_subset.2 support_add) ?_) (union_subset hp hq)
    rw [Finset.coe_union]
  zero_mem' := by
    /-
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁷ : Semiring R
      inst✝⁶ : Semiring S
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : AddCommMonoid P
      inst✝ : Module R P
      s : Set α
      ⊢ Membership.mem { carrier := setOf fun p => HasSubset.Subset (↑p.support) s,  …
    -/
    simp only [subset_def, Finset.mem_coe, Set.mem_setOf_eq, mem_support_iff, zero_apply]
    /-
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁷ : Semiring R
      inst✝⁶ : Semiring S
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : AddCommMonoid P
      inst✝ : Module R P
      s : Set α
      ⊢ ∀ (x : α), Ne 0 0 → Membership.mem s x
    -/
    intro h ha
    /-
      α : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      R : Type u_5
      S : Type u_6
      inst✝⁷ : Semiring R
      inst✝⁶ : Semiring S
      inst✝⁵ : AddCommMonoid M
      inst✝⁴ : Module R M
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : AddCommMonoid P
      inst✝ : Module R P
      s : Set α
      h : α
      ha : Ne 0 0
      ⊢ Membership.mem s h
    -/
    exact (ha rfl).elim
    /-
      🎉 no goals
    -/
  smul_mem' _ _ hp := Subset.trans (Finset.coe_subset.2 support_smul) hp


theorem mem_supported {s : Set α} (p : α →₀ M) : p ∈ supported M R s ↔ ↑p.support ⊆ s :=
  Iff.rfl


theorem mem_supported' {s : Set α} (p : α →₀ M) :
    p ∈ supported M R s ↔ ∀ x ∉ s, p x = 0 := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Set α
    p : Finsupp α M
    ⊢ Iff (Membership.mem (Finsupp.supported M R s) p) (∀ (x : α), Not (Membership …
  -/
  haveI := Classical.decPred fun x : α => x ∈ s; simp [mem_supported, Set.subset_def, not_imp_comm]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem mem_supported_support (p : α →₀ M) : p ∈ Finsupp.supported M R (p.support : Set α) := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    p : Finsupp α M
    ⊢ Membership.mem (Finsupp.supported M R ↑p.support) p
  -/
  rw [Finsupp.mem_supported]
  /-
    🎉 no goals
  -/


theorem single_mem_supported {s : Set α} {a : α} (b : M) (h : a ∈ s) :
    single a b ∈ supported M R s :=
  Set.Subset.trans support_single_subset (Finset.singleton_subset_set_iff.2 h)


theorem supported_eq_span_single (s : Set α) :
    supported R R s = span R ((fun i => single i 1) '' s) := by
  /-
    α : Type u_1
    R : Type u_5
    inst✝ : Semiring R
    s : Set α
    ⊢ Eq (Finsupp.supported R R s) (Submodule.span R (Set.image (fun i => Finsupp. …
  -/
  refine (span_eq_of_le _ ?_ (SetLike.le_def.2 fun l hl => ?_)).symm
    /-
      case refine_1
      α : Type u_1
      R : Type u_5
      inst✝ : Semiring R
      s : Set α
      ⊢ HasSubset.Subset (Set.image (fun i => Finsupp.single i 1) s) ↑(Finsupp.suppo …
    -/
  · rintro _ ⟨_, hp, rfl⟩
    /-
      case refine_1.intro.intro
      α : Type u_1
      R : Type u_5
      inst✝ : Semiring R
      s : Set α
      w✝ : α
      hp : Membership.mem s w✝
      ⊢ Membership.mem (↑(Finsupp.supported R R s)) ((fun i => Finsupp.single i 1) w✝)
    -/
    exact single_mem_supported R 1 hp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      R : Type u_5
      inst✝ : Semiring R
      s : Set α
      l : Finsupp α R
      hl : Membership.mem (Finsupp.supported R R s) l
      ⊢ Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1) s) …
    -/
  · rw [← l.sum_single]
    /-
      case refine_2
      α : Type u_1
      R : Type u_5
      inst✝ : Semiring R
      s : Set α
      l : Finsupp α R
      hl : Membership.mem (Finsupp.supported R R s) l
      ⊢ Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1) s) …
    -/
    refine sum_mem fun i il => ?_
  -- Porting note: Needed to help this convert quite a bit replacing underscores
    /-
      case refine_2
      α : Type u_1
      R : Type u_5
      inst✝ : Semiring R
      s : Set α
      l : Finsupp α R
      hl : Membership.mem (Finsupp.supported R R s) l
      i : α
      il : Membership.mem l.support i
      ⊢ Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1) s) …
    -/
    convert smul_mem (M := α →₀ R) (x := single i 1) (span R ((fun i => single i 1) '' s)) (l i) ?_
      /-
        case h.e'_5
        α : Type u_1
        R : Type u_5
        inst✝ : Semiring R
        s : Set α
        l : Finsupp α R
        hl : Membership.mem (Finsupp.supported R R s) l
        i : α
        il : Membership.mem l.support i
        ⊢ Eq (Finsupp.single i (l i)) (HSMul.hSMul (l i) (Finsupp.single i 1))
      -/
    · simp [span]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u_1
        R : Type u_5
        inst✝ : Semiring R
        s : Set α
        l : Finsupp α R
        hl : Membership.mem (Finsupp.supported R R s) l
        i : α
        il : Membership.mem l.support i
        ⊢ Membership.mem (Submodule.span R (Set.image (fun i => Finsupp.single i 1) s) …
      -/
    · apply subset_span
      /-
        case refine_2.a
        α : Type u_1
        R : Type u_5
        inst✝ : Semiring R
        s : Set α
        l : Finsupp α R
        hl : Membership.mem (Finsupp.supported R R s) l
        i : α
        il : Membership.mem l.support i
        ⊢ Membership.mem (Set.image (fun i => Finsupp.single i 1) s) (Finsupp.single i …
      -/
      apply Set.mem_image_of_mem _ (hl il)
      /-
        🎉 no goals
      -/


/-- Interpret `Finsupp.filter s` as a linear map from `α →₀ M` to `supported M R s`. -/
def restrictDom (s : Set α) [DecidablePred (· ∈ s)] : (α →₀ M) →ₗ[R] supported M R s :=
  LinearMap.codRestrict _
    { toFun := filter (· ∈ s)
      map_add' := fun _ _ => filter_add
      map_smul' := fun _ _ => filter_smul } fun l =>
    (mem_supported' _ _).2 fun _ => filter_apply_neg (· ∈ s) l


@[simp]
theorem restrictDom_apply (s : Set α) (l : α →₀ M) [DecidablePred (· ∈ s)] :
    (restrictDom M R s l : α →₀ M) = Finsupp.filter (· ∈ s) l := rfl


theorem restrictDom_comp_subtype (s : Set α) [DecidablePred (· ∈ s)] :
    (restrictDom M R s).comp (Submodule.subtype _) = LinearMap.id := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    ⊢ Eq ((Finsupp.restrictDom M R s).comp (Finsupp.supported M R s).subtype) Line …
  -/
  ext l a
  /-
    case h.a.h
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    l : Subtype fun x => Membership.mem (Finsupp.supported M R s) x
    a : α
    ⊢ Eq (↑(((Finsupp.restrictDom M R s).comp (Finsupp.supported M R s).subtype) l …
  -/
  by_cases h : a ∈ s
    /-
      case pos
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝³ : Semiring R
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R M
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      l : Subtype fun x => Membership.mem (Finsupp.supported M R s) x
      a : α
      h : Membership.mem s a
      ⊢ Eq (↑(((Finsupp.restrictDom M R s).comp (Finsupp.supported M R s).subtype) l …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    l : Subtype fun x => Membership.mem (Finsupp.supported M R s) x
    a : α
    h : Not (Membership.mem s a)
    ⊢ Eq (↑(((Finsupp.restrictDom M R s).comp (Finsupp.supported M R s).subtype) l …
  -/
  simpa [h] using ((mem_supported' R l.1).1 l.2 a h).symm
  /-
    🎉 no goals
  -/


theorem range_restrictDom (s : Set α) [DecidablePred (· ∈ s)] :
    LinearMap.range (restrictDom M R s) = ⊤ :=
  range_eq_top.2 <|
    Function.RightInverse.surjective <| LinearMap.congr_fun (restrictDom_comp_subtype s)


theorem supported_mono {s t : Set α} (st : s ⊆ t) : supported M R s ≤ supported M R t := fun _ h =>
  Set.Subset.trans h st


@[simp]
theorem supported_empty : supported M R (∅ : Set α) = ⊥ :=
                                                        /-
                                                          α : Type u_1
                                                          M : Type u_2
                                                          R : Type u_5
                                                          inst✝² : Semiring R
                                                          inst✝¹ : AddCommMonoid M
                                                          inst✝ : Module R M
                                                          l : Finsupp α M
                                                          h : Membership.mem (Finsupp.supported M R EmptyCollection.emptyCollection) l
                                                          ⊢ Eq l 0
                                                        -/
  eq_bot_iff.2 fun l h => (Submodule.mem_bot R).2 <| by ext; simp_all [mem_supported']
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem supported_univ : supported M R (Set.univ : Set α) = ⊤ :=
  eq_top_iff.2 fun _ _ => Set.subset_univ _


theorem supported_iUnion {δ : Type*} (s : δ → Set α) :
    supported M R (⋃ i, s i) = ⨆ i, supported M R (s i) := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    δ : Type u_7
    s : δ → Set α
    ⊢ Eq (Finsupp.supported M R (Set.iUnion fun i => s i)) (iSup fun i => Finsupp. …
  -/
  refine le_antisymm ?_ (iSup_le fun i => supported_mono <| Set.subset_iUnion _ _)
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    δ : Type u_7
    s : δ → Set α
    ⊢ LE.le (Finsupp.supported M R (Set.iUnion fun i => s i)) (iSup fun i => Finsu …
  -/
  haveI := Classical.decPred fun x => x ∈ ⋃ i, s i
  suffices
    LinearMap.range ((Submodule.subtype _).comp (restrictDom M R (⋃ i, s i))) ≤
      ⨆ i, supported M R (s i) by
    rwa [LinearMap.range_comp, range_restrictDom, Submodule.map_top, range_subtype] at this
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    δ : Type u_7
    s : δ → Set α
    this : DecidablePred fun x => Membership.mem (Set.iUnion fun i => s i) x
    ⊢ LE.le (LinearMap.range ((Finsupp.supported M R (Set.iUnion fun i => s i)).su …
  -/
  rw [range_le_iff_comap, eq_top_iff]
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    δ : Type u_7
    s : δ → Set α
    this : DecidablePred fun x => Membership.mem (Set.iUnion fun i => s i) x
    ⊢ LE.le Top.top (Submodule.comap ((Finsupp.supported M R (Set.iUnion fun i =>  …
  -/
  rintro l ⟨⟩
  -- Porting note: Was ported as `induction l using Finsupp.induction`
  /-
    case intro
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    δ : Type u_7
    s : δ → Set α
    this : DecidablePred fun x => Membership.mem (Set.iUnion fun i => s i) x
    l : Finsupp α M
    ⊢ Membership.mem (Submodule.comap ((Finsupp.supported M R (Set.iUnion fun i => …
  -/
  refine Finsupp.induction l ?_ ?_
    /-
      case intro.refine_1
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      δ : Type u_7
      s : δ → Set α
      this : DecidablePred fun x => Membership.mem (Set.iUnion fun i => s i) x
      l : Finsupp α M
      ⊢ Membership.mem (Submodule.comap ((Finsupp.supported M R (Set.iUnion fun i => …
    -/
  · exact zero_mem _
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      δ : Type u_7
      s : δ → Set α
      this : DecidablePred fun x => Membership.mem (Set.iUnion fun i => s i) x
      l : Finsupp α M
      ⊢ ∀ (a : α) (b : M) (f : Finsupp α M), Not (Membership.mem f.support a) → Ne b …
    -/
  · refine fun x a l _ _ => add_mem ?_
    /-
      case intro.refine_2
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      δ : Type u_7
      s : δ → Set α
      this : DecidablePred fun x => Membership.mem (Set.iUnion fun i => s i) x
      l✝ : Finsupp α M
      x : α
      a : M
      l : Finsupp α M
      x✝¹ : Not (Membership.mem l.support x)
      x✝ : Ne a 0
      ⊢ Membership.mem (Submodule.comap ((Finsupp.supported M R (Set.iUnion fun i => …
    -/
    by_cases h : ∃ i, x ∈ s i
    · simp only [mem_comap, coe_comp, coe_subtype, Function.comp_apply, restrictDom_apply,
        mem_iUnion, h, filter_single_of_pos]
      /-
        case pos
        α : Type u_1
        M : Type u_2
        R : Type u_5
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        δ : Type u_7
        s : δ → Set α
        this : DecidablePred fun x => Membership.mem (Set.iUnion fun i => s i) x
        l✝ : Finsupp α M
        x : α
        a : M
        l : Finsupp α M
        x✝¹ : Not (Membership.mem l.support x)
        x✝ : Ne a 0
        h : Exists fun i => Membership.mem (s i) x
        ⊢ Membership.mem (iSup fun i => Finsupp.supported M R (s i)) (Finsupp.single x …
      -/
      cases' h with i hi
      /-
        case pos.intro
        α : Type u_1
        M : Type u_2
        R : Type u_5
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        δ : Type u_7
        s : δ → Set α
        this : DecidablePred fun x => Membership.mem (Set.iUnion fun i => s i) x
        l✝ : Finsupp α M
        x : α
        a : M
        l : Finsupp α M
        x✝¹ : Not (Membership.mem l.support x)
        x✝ : Ne a 0
        i : δ
        hi : Membership.mem (s i) x
        ⊢ Membership.mem (iSup fun i => Finsupp.supported M R (s i)) (Finsupp.single x …
      -/
      exact le_iSup (fun i => supported M R (s i)) i (single_mem_supported R _ hi)
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        M : Type u_2
        R : Type u_5
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        δ : Type u_7
        s : δ → Set α
        this : DecidablePred fun x => Membership.mem (Set.iUnion fun i => s i) x
        l✝ : Finsupp α M
        x : α
        a : M
        l : Finsupp α M
        x✝¹ : Not (Membership.mem l.support x)
        x✝ : Ne a 0
        h : Not (Exists fun i => Membership.mem (s i) x)
        ⊢ Membership.mem (Submodule.comap ((Finsupp.supported M R (Set.iUnion fun i => …
      -/
    · simp [h]
      /-
        🎉 no goals
      -/


theorem supported_union (s t : Set α) :
    supported M R (s ∪ t) = supported M R s ⊔ supported M R t := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s t : Set α
    ⊢ Eq (Finsupp.supported M R (Union.union s t)) (Max.max (Finsupp.supported M R …
  -/
  rw [Set.union_eq_iUnion, supported_iUnion, iSup_bool_eq, cond_true, cond_false]
  /-
    🎉 no goals
  -/


theorem supported_iInter {ι : Type*} (s : ι → Set α) :
    supported M R (⋂ i, s i) = ⨅ i, supported M R (s i) :=
                            /-
                              α : Type u_1
                              M : Type u_2
                              R : Type u_5
                              inst✝² : Semiring R
                              inst✝¹ : AddCommMonoid M
                              inst✝ : Module R M
                              ι : Type u_7
                              s : ι → Set α
                              x : Finsupp α M
                              ⊢ Iff (Membership.mem (Finsupp.supported M R (Set.iInter fun i => s i)) x) (Me …
                            -/
  Submodule.ext fun x => by simp [mem_supported, subset_iInter_iff]
                            /-
                              🎉 no goals
                            -/


theorem supported_inter (s t : Set α) :
    supported M R (s ∩ t) = supported M R s ⊓ supported M R t := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s t : Set α
    ⊢ Eq (Finsupp.supported M R (Inter.inter s t)) (Min.min (Finsupp.supported M R …
  -/
  rw [Set.inter_eq_iInter, supported_iInter, iInf_bool_eq]; rfl
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem disjoint_supported_supported {s t : Set α} (h : Disjoint s t) :
    Disjoint (supported M R s) (supported M R t) :=
                       /-
                         α : Type u_1
                         M : Type u_2
                         R : Type u_5
                         inst✝² : Semiring R
                         inst✝¹ : AddCommMonoid M
                         inst✝ : Module R M
                         s t : Set α
                         h : Disjoint s t
                         ⊢ Eq (Min.min (Finsupp.supported M R s) (Finsupp.supported M R t)) Bot.bot
                       -/
  disjoint_iff.2 <| by rw [← supported_inter, disjoint_iff_inter_eq_empty.1 h, supported_empty]
                       /-
                         🎉 no goals
                       -/


theorem disjoint_supported_supported_iff [Nontrivial M] {s t : Set α} :
    Disjoint (supported M R s) (supported M R t) ↔ Disjoint s t := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Nontrivial M
    s t : Set α
    ⊢ Iff (Disjoint (Finsupp.supported M R s) (Finsupp.supported M R t)) (Disjoint …
  -/
  refine ⟨fun h => Set.disjoint_left.mpr fun x hx1 hx2 => ?_, disjoint_supported_supported⟩
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Nontrivial M
    s t : Set α
    h : Disjoint (Finsupp.supported M R s) (Finsupp.supported M R t)
    x : α
    hx1 : Membership.mem s x
    hx2 : Membership.mem t x
    ⊢ False
  -/
  rcases exists_ne (0 : M) with ⟨y, hy⟩
  /-
    case intro
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Nontrivial M
    s t : Set α
    h : Disjoint (Finsupp.supported M R s) (Finsupp.supported M R t)
    x : α
    hx1 : Membership.mem s x
    hx2 : Membership.mem t x
    y : M
    hy : Ne y 0
    ⊢ False
  -/
  have := h.le_bot ⟨single_mem_supported R y hx1, single_mem_supported R y hx2⟩
  /-
    case intro
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Nontrivial M
    s t : Set α
    h : Disjoint (Finsupp.supported M R s) (Finsupp.supported M R t)
    x : α
    hx1 : Membership.mem s x
    hx2 : Membership.mem t x
    y : M
    hy : Ne y 0
    this : Membership.mem Bot.bot (Finsupp.single x y)
    ⊢ False
  -/
  rw [mem_bot, single_eq_zero] at this
  /-
    case intro
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Nontrivial M
    s t : Set α
    h : Disjoint (Finsupp.supported M R s) (Finsupp.supported M R t)
    x : α
    hx1 : Membership.mem s x
    hx2 : Membership.mem t x
    y : M
    hy : Ne y 0
    this : Eq y 0
    ⊢ False
  -/
  exact hy this
  /-
    🎉 no goals
  -/


/-- Interpret `Finsupp.restrictSupportEquiv` as a linear equivalence between
`supported M R s` and `s →₀ M`. -/
def supportedEquivFinsupp (s : Set α) : supported M R s ≃ₗ[R] s →₀ M := by
  /-
    α : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    R : Type u_5
    S : Type u_6
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    s : Set α
    ⊢ LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Finsupp.support …
  -/
  let F : supported M R s ≃ (s →₀ M) := restrictSupportEquiv s M
  /-
    α : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    R : Type u_5
    S : Type u_6
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    s : Set α
    F : Equiv (Subtype fun x => Membership.mem (Finsupp.supported M R s) x) (Finsu …
    ⊢ LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Finsupp.support …
  -/
  refine F.toLinearEquiv ?_
  have :
    (F : supported M R s → ↥s →₀ M) =
      (lsubtypeDomain s : (α →₀ M) →ₗ[R] s →₀ M).comp (Submodule.subtype (supported M R s)) :=
    rfl
  /-
    α : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    R : Type u_5
    S : Type u_6
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    s : Set α
    F : Equiv (Subtype fun x => Membership.mem (Finsupp.supported M R s) x) (Finsu …
    this : Eq ⇑F ⇑((Finsupp.lsubtypeDomain s).comp (Finsupp.supported M R s).subty …
    ⊢ IsLinearMap R ⇑F
  -/
  rw [this]
  /-
    α : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    R : Type u_5
    S : Type u_6
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    s : Set α
    F : Equiv (Subtype fun x => Membership.mem (Finsupp.supported M R s) x) (Finsu …
    this : Eq ⇑F ⇑((Finsupp.lsubtypeDomain s).comp (Finsupp.supported M R s).subty …
    ⊢ IsLinearMap R ⇑((Finsupp.lsubtypeDomain s).comp (Finsupp.supported M R s).su …
  -/
  exact LinearMap.isLinear _
  /-
    🎉 no goals
  -/


theorem supported_comap_lmapDomain (f : α → α') (s : Set α') :
    supported M R (f ⁻¹' s) ≤ (supported M R s).comap (lmapDomain M R f) := by
  classical
  intro l (hl : (l.support : Set α) ⊆ f ⁻¹' s)
  show ↑(mapDomain f l).support ⊆ s
  rw [← Set.image_subset_iff, ← Finset.coe_image] at hl
  exact Set.Subset.trans mapDomain_support hl


theorem lmapDomain_supported (f : α → α') (s : Set α) :
    (supported M R s).map (lmapDomain M R f) = supported M R (f '' s) := by
  classical
  cases isEmpty_or_nonempty α
  · simp [s.eq_empty_of_isEmpty]
  refine
    le_antisymm
      (map_le_iff_le_comap.2 <|
        le_trans (supported_mono <| Set.subset_preimage_image _ _)
          (supported_comap_lmapDomain M R _ _))
      ?_
  intro l hl
  refine ⟨(lmapDomain M R (Function.invFunOn f s) : (α' →₀ M) →ₗ[R] α →₀ M) l, fun x hx => ?_, ?_⟩
  · rcases Finset.mem_image.1 (mapDomain_support hx) with ⟨c, hc, rfl⟩
    exact Function.invFunOn_mem (by simpa using hl hc)
  · rw [← LinearMap.comp_apply, ← lmapDomain_comp]
    refine (mapDomain_congr fun c hc => ?_).trans mapDomain_id
    exact Function.invFunOn_eq (by simpa using hl hc)


theorem lmapDomain_disjoint_ker (f : α → α') {s : Set α}
    (H : ∀ a ∈ s, ∀ b ∈ s, f a = f b → a = b) :
    Disjoint (supported M R s) (ker (lmapDomain M R f)) := by
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α' : Type u_7
    f : α → α'
    s : Set α
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
    ⊢ Disjoint (Finsupp.supported M R s) (LinearMap.ker (Finsupp.lmapDomain M R f))
  -/
  rw [disjoint_iff_inf_le]
  /-
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α' : Type u_7
    f : α → α'
    s : Set α
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
    ⊢ LE.le (Min.min (Finsupp.supported M R s) (LinearMap.ker (Finsupp.lmapDomain  …
  -/
  rintro l ⟨h₁, h₂⟩
  /-
    case intro
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α' : Type u_7
    f : α → α'
    s : Set α
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
    l : Finsupp α M
    h₁ : Membership.mem (↑(Finsupp.supported M R s)) l
    h₂ : Membership.mem (↑(LinearMap.ker (Finsupp.lmapDomain M R f))) l
    ⊢ Membership.mem Bot.bot l
  -/
  rw [SetLike.mem_coe, mem_ker, lmapDomain_apply, mapDomain] at h₂
  /-
    case intro
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α' : Type u_7
    f : α → α'
    s : Set α
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
    l : Finsupp α M
    h₁ : Membership.mem (↑(Finsupp.supported M R s)) l
    h₂ : Eq (l.sum fun a => Finsupp.single (f a)) 0
    ⊢ Membership.mem Bot.bot l
  -/
  simp only [mem_bot]; ext x
  /-
    case intro.h
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α' : Type u_7
    f : α → α'
    s : Set α
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
    l : Finsupp α M
    h₁ : Membership.mem (↑(Finsupp.supported M R s)) l
    h₂ : Eq (l.sum fun a => Finsupp.single (f a)) 0
    x : α
    ⊢ Eq (l x) (0 x)
  -/
  haveI := Classical.decPred fun x => x ∈ s
  /-
    case intro.h
    α : Type u_1
    M : Type u_2
    R : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    α' : Type u_7
    f : α → α'
    s : Set α
    H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
    l : Finsupp α M
    h₁ : Membership.mem (↑(Finsupp.supported M R s)) l
    h₂ : Eq (l.sum fun a => Finsupp.single (f a)) 0
    x : α
    this : DecidablePred fun x => Membership.mem s x
    ⊢ Eq (l x) (0 x)
  -/
  by_cases xs : x ∈ s
  · have : Finsupp.sum l (fun a => Finsupp.single (f a)) (f x) = 0 := by
      rw [h₂]
      rfl
    /-
      case pos
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α' : Type u_7
      f : α → α'
      s : Set α
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
      l : Finsupp α M
      h₁ : Membership.mem (↑(Finsupp.supported M R s)) l
      h₂ : Eq (l.sum fun a => Finsupp.single (f a)) 0
      x : α
      this✝ : DecidablePred fun x => Membership.mem s x
      xs : Membership.mem s x
      this : Eq ((l.sum fun a => Finsupp.single (f a)) (f x)) 0
      ⊢ Eq (l x) (0 x)
    -/
    rw [Finsupp.sum_apply, Finsupp.sum_eq_single x, single_eq_same] at this
      /-
        case pos
        α : Type u_1
        M : Type u_2
        R : Type u_5
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        α' : Type u_7
        f : α → α'
        s : Set α
        H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
        l : Finsupp α M
        h₁ : Membership.mem (↑(Finsupp.supported M R s)) l
        h₂ : Eq (l.sum fun a => Finsupp.single (f a)) 0
        x : α
        this✝ : DecidablePred fun x => Membership.mem s x
        xs : Membership.mem s x
        this : Eq (l x) 0
        ⊢ Eq (l x) (0 x)
      -/
    · simpa
      /-
        🎉 no goals
      -/
      /-
        case pos.h₀
        α : Type u_1
        M : Type u_2
        R : Type u_5
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        α' : Type u_7
        f : α → α'
        s : Set α
        H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
        l : Finsupp α M
        h₁ : Membership.mem (↑(Finsupp.supported M R s)) l
        h₂ : Eq (l.sum fun a => Finsupp.single (f a)) 0
        x : α
        this✝ : DecidablePred fun x => Membership.mem s x
        xs : Membership.mem s x
        this : Eq (l.sum fun a₁ b => (Finsupp.single (f a₁) b) (f x)) 0
        ⊢ ∀ (b : α), Ne (l b) 0 → Ne b x → Eq ((Finsupp.single (f b) (l b)) (f x)) 0
      -/
    · intro y hy xy
      /-
        case pos.h₀
        α : Type u_1
        M : Type u_2
        R : Type u_5
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        α' : Type u_7
        f : α → α'
        s : Set α
        H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
        l : Finsupp α M
        h₁ : Membership.mem (↑(Finsupp.supported M R s)) l
        h₂ : Eq (l.sum fun a => Finsupp.single (f a)) 0
        x : α
        this✝ : DecidablePred fun x => Membership.mem s x
        xs : Membership.mem s x
        this : Eq (l.sum fun a₁ b => (Finsupp.single (f a₁) b) (f x)) 0
        y : α
        hy : Ne (l y) 0
        xy : Ne y x
        ⊢ Eq ((Finsupp.single (f y) (l y)) (f x)) 0
      -/
      simp only [SetLike.mem_coe, mem_supported, subset_def, Finset.mem_coe, mem_support_iff] at h₁
      /-
        case pos.h₀
        α : Type u_1
        M : Type u_2
        R : Type u_5
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        α' : Type u_7
        f : α → α'
        s : Set α
        H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
        l : Finsupp α M
        h₂ : Eq (l.sum fun a => Finsupp.single (f a)) 0
        x : α
        this✝ : DecidablePred fun x => Membership.mem s x
        xs : Membership.mem s x
        this : Eq (l.sum fun a₁ b => (Finsupp.single (f a₁) b) (f x)) 0
        y : α
        hy : Ne (l y) 0
        xy : Ne y x
        h₁ : ∀ (x : α), Ne (l x) 0 → Membership.mem s x
        ⊢ Eq ((Finsupp.single (f y) (l y)) (f x)) 0
      -/
      simp [mt (H _ (h₁ _ hy) _ xs) xy]
      /-
        🎉 no goals
      -/
      /-
        case pos.h₁
        α : Type u_1
        M : Type u_2
        R : Type u_5
        inst✝² : Semiring R
        inst✝¹ : AddCommMonoid M
        inst✝ : Module R M
        α' : Type u_7
        f : α → α'
        s : Set α
        H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
        l : Finsupp α M
        h₁ : Membership.mem (↑(Finsupp.supported M R s)) l
        h₂ : Eq (l.sum fun a => Finsupp.single (f a)) 0
        x : α
        this✝ : DecidablePred fun x => Membership.mem s x
        xs : Membership.mem s x
        this : Eq (l.sum fun a₁ b => (Finsupp.single (f a₁) b) (f x)) 0
        ⊢ Eq (l x) 0 → Eq ((Finsupp.single (f x) 0) (f x)) 0
      -/
    · simp +contextual
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α' : Type u_7
      f : α → α'
      s : Set α
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
      l : Finsupp α M
      h₁ : Membership.mem (↑(Finsupp.supported M R s)) l
      h₂ : Eq (l.sum fun a => Finsupp.single (f a)) 0
      x : α
      this : DecidablePred fun x => Membership.mem s x
      xs : Not (Membership.mem s x)
      ⊢ Eq (l x) (0 x)
    -/
  · by_contra h
    /-
      case neg
      α : Type u_1
      M : Type u_2
      R : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      α' : Type u_7
      f : α → α'
      s : Set α
      H : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem s b → Eq (f a) ( …
      l : Finsupp α M
      h₁ : Membership.mem (↑(Finsupp.supported M R s)) l
      h₂ : Eq (l.sum fun a => Finsupp.single (f a)) 0
      x : α
      this : DecidablePred fun x => Membership.mem s x
      xs : Not (Membership.mem s x)
      h : Not (Eq (l x) (0 x))
      ⊢ False
    -/
    exact xs (h₁ <| Finsupp.mem_support_iff.2 h)
    /-
      🎉 no goals
    -/


/-- An equivalence of sets induces a linear equivalence of `Finsupp`s supported on those sets. -/
noncomputable def congr {α' : Type*} (s : Set α) (t : Set α') (e : s ≃ t) :
    supported M R s ≃ₗ[R] supported M R t := by
  /-
    α : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    R : Type u_5
    S : Type u_6
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    α' : Type u_7
    s : Set α
    t : Set α'
    e : Equiv ↑s ↑t
    ⊢ LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Finsupp.support …
  -/
  haveI := Classical.decPred fun x => x ∈ s
  /-
    α : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    R : Type u_5
    S : Type u_6
    inst✝⁷ : Semiring R
    inst✝⁶ : Semiring S
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : Module R M
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : AddCommMonoid P
    inst✝ : Module R P
    α' : Type u_7
    s : Set α
    t : Set α'
    e : Equiv ↑s ↑t
    this : DecidablePred fun x => Membership.mem s x
    ⊢ LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (Finsupp.support …
  -/
  haveI := Classical.decPred fun x => x ∈ t
  exact Finsupp.supportedEquivFinsupp s ≪≫ₗ
    (Finsupp.domLCongr e ≪≫ₗ (Finsupp.supportedEquivFinsupp t).symm)


