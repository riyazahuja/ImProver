/-- A complete type over a given theory in a certain type of variables is a maximally
  consistent (with the theory) set of formulas in that type. -/
structure CompleteType where
  toTheory : L[[α]].Theory
  subset' : (L.lhomWithConstants α).onTheory T ⊆ toTheory
  isMaximal' : toTheory.IsMaximal


instance Sentence.instSetLike : SetLike (T.CompleteType α) (L[[α]].Sentence) :=
  ⟨fun p => p.toTheory, fun p q h => by
    /-
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      p q : T.CompleteType α
      h : Eq ((fun p => ↑p) p) ((fun p => ↑p) q)
      ⊢ Eq p q
    -/
    cases p
    /-
      case mk
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      q : T.CompleteType α
      toTheory✝ : (L.withConstants α).Theory
      subset'✝ : HasSubset.Subset ((L.lhomWithConstants α).onTheory T) toTheory✝
      isMaximal'✝ : toTheory✝.IsMaximal
      h : Eq ((fun p => ↑p) { toTheory := toTheory✝, subset' := subset'✝, isMaximal' …
      ⊢ Eq { toTheory := toTheory✝, subset' := subset'✝, isMaximal' := isMaximal'✝ } q
    -/
    cases q
    /-
      case mk.mk
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      toTheory✝¹ : (L.withConstants α).Theory
      subset'✝¹ : HasSubset.Subset ((L.lhomWithConstants α).onTheory T) toTheory✝¹
      isMaximal'✝¹ : toTheory✝¹.IsMaximal
      toTheory✝ : (L.withConstants α).Theory
      subset'✝ : HasSubset.Subset ((L.lhomWithConstants α).onTheory T) toTheory✝
      isMaximal'✝ : toTheory✝.IsMaximal
      h : Eq ((fun p => ↑p) { toTheory := toTheory✝¹, subset' := subset'✝¹, isMaxima …
      ⊢ Eq { toTheory := toTheory✝¹, subset' := subset'✝¹, isMaximal' := isMaximal'✝ …
    -/
    congr ⟩
    /-
      🎉 no goals
    -/


theorem isMaximal (p : T.CompleteType α) : IsMaximal (p : L[[α]].Theory) :=
  p.isMaximal'


theorem subset (p : T.CompleteType α) : (L.lhomWithConstants α).onTheory T ⊆ (p : L[[α]].Theory) :=
  p.subset'


theorem mem_or_not_mem (p : T.CompleteType α) (φ : L[[α]].Sentence) : φ ∈ p ∨ φ.not ∈ p :=
  p.isMaximal.mem_or_not_mem φ


theorem mem_of_models (p : T.CompleteType α) {φ : L[[α]].Sentence}
    (h : (L.lhomWithConstants α).onTheory T ⊨ᵇ φ) : φ ∈ p :=
  (p.mem_or_not_mem φ).resolve_right fun con =>
    ((models_iff_not_satisfiable _).1 h)
      (p.isMaximal.1.mono (union_subset p.subset (singleton_subset_iff.2 con)))


theorem not_mem_iff (p : T.CompleteType α) (φ : L[[α]].Sentence) : φ.not ∈ p ↔ ¬φ ∈ p :=
  ⟨fun hf ht => by
    have h : ¬IsSatisfiable ({φ, φ.not} : L[[α]].Theory) := by
      rintro ⟨@⟨_, _, h, _⟩⟩
      simp only [model_iff, mem_insert_iff, mem_singleton_iff, forall_eq_or_imp, forall_eq] at h
      exact h.2 h.1
    /-
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      p : T.CompleteType α
      φ : (L.withConstants α).Sentence
      hf : Membership.mem p (FirstOrder.Language.Formula.not φ)
      ht : Membership.mem p φ
      h : Not (Insert.insert φ (Singleton.singleton (FirstOrder.Language.Formula.not …
      ⊢ False
    -/
    refine h (p.isMaximal.1.mono ?_)
    /-
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      p : T.CompleteType α
      φ : (L.withConstants α).Sentence
      hf : Membership.mem p (FirstOrder.Language.Formula.not φ)
      ht : Membership.mem p φ
      h : Not (Insert.insert φ (Singleton.singleton (FirstOrder.Language.Formula.not …
      ⊢ HasSubset.Subset (Insert.insert φ (Singleton.singleton (FirstOrder.Language. …
    -/
    rw [insert_subset_iff, singleton_subset_iff]
    /-
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      p : T.CompleteType α
      φ : (L.withConstants α).Sentence
      hf : Membership.mem p (FirstOrder.Language.Formula.not φ)
      ht : Membership.mem p φ
      h : Not (Insert.insert φ (Singleton.singleton (FirstOrder.Language.Formula.not …
      ⊢ And (Membership.mem (↑p) φ) (Membership.mem (↑p) (FirstOrder.Language.Formul …
    -/
    exact ⟨ht, hf⟩, (p.mem_or_not_mem φ).resolve_left⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem compl_setOf_mem {φ : L[[α]].Sentence} :
    { p : T.CompleteType α | φ ∈ p }ᶜ = { p : T.CompleteType α | φ.not ∈ p } :=
  ext fun _ => (not_mem_iff _ _).symm


theorem setOf_subset_eq_empty_iff (S : L[[α]].Theory) :
    { p : T.CompleteType α | S ⊆ ↑p } = ∅ ↔
      ¬((L.lhomWithConstants α).onTheory T ∪ S).IsSatisfiable := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    S : (L.withConstants α).Theory
    ⊢ Iff (Eq (setOf fun p => HasSubset.Subset S ↑p) EmptyCollection.emptyCollecti …
  -/
  rw [iff_not_comm, ← not_nonempty_iff_eq_empty, Classical.not_not, Set.Nonempty]
  refine
    ⟨fun h =>
      ⟨⟨L[[α]].completeTheory h.some, (subset_union_left (t := S)).trans completeTheory.subset,
          completeTheory.isMaximal (L[[α]]) h.some⟩,
        (((L.lhomWithConstants α).onTheory T).subset_union_right).trans completeTheory.subset⟩,
      ?_⟩
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    S : (L.withConstants α).Theory
    ⊢ (Exists fun x => Membership.mem (setOf fun p => HasSubset.Subset S ↑p) x) →  …
  -/
  rintro ⟨p, hp⟩
  /-
    case intro
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    S : (L.withConstants α).Theory
    p : T.CompleteType α
    hp : Membership.mem (setOf fun p => HasSubset.Subset S ↑p) p
    ⊢ (Union.union ((L.lhomWithConstants α).onTheory T) S).IsSatisfiable
  -/
  exact p.isMaximal.1.mono (union_subset p.subset hp)
  /-
    🎉 no goals
  -/


theorem setOf_mem_eq_univ_iff (φ : L[[α]].Sentence) :
    { p : T.CompleteType α | φ ∈ p } = Set.univ ↔ (L.lhomWithConstants α).onTheory T ⊨ᵇ φ := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    φ : (L.withConstants α).Sentence
    ⊢ Iff (Eq (setOf fun p => Membership.mem p φ) Set.univ) (((L.lhomWithConstants …
  -/
  rw [models_iff_not_satisfiable, ← compl_empty_iff, compl_setOf_mem, ← setOf_subset_eq_empty_iff]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    φ : (L.withConstants α).Sentence
    ⊢ Iff (Eq (setOf fun p => Membership.mem p (FirstOrder.Language.Formula.not φ) …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem setOf_subset_eq_univ_iff (S : L[[α]].Theory) :
    { p : T.CompleteType α | S ⊆ ↑p } = Set.univ ↔
      ∀ φ, φ ∈ S → (L.lhomWithConstants α).onTheory T ⊨ᵇ φ := by
  have h : { p : T.CompleteType α | S ⊆ ↑p } = ⋂₀ ((fun φ => { p | φ ∈ p }) '' S) := by
    ext
    simp [subset_def]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    S : (L.withConstants α).Theory
    h : Eq (setOf fun p => HasSubset.Subset S ↑p) (Set.image (fun φ => setOf fun p …
    ⊢ Iff (Eq (setOf fun p => HasSubset.Subset S ↑p) Set.univ) (∀ (φ : (L.withCons …
  -/
  simp_rw [h, sInter_eq_univ, ← setOf_mem_eq_univ_iff]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    S : (L.withConstants α).Theory
    h : Eq (setOf fun p => HasSubset.Subset S ↑p) (Set.image (fun φ => setOf fun p …
    ⊢ Iff (∀ (s : Set (T.CompleteType α)), Membership.mem (Set.image (fun φ => set …
  -/
  refine ⟨fun h φ φS => h _ ⟨_, φS, rfl⟩, ?_⟩
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    S : (L.withConstants α).Theory
    h : Eq (setOf fun p => HasSubset.Subset S ↑p) (Set.image (fun φ => setOf fun p …
    ⊢ (∀ (φ : (L.withConstants α).Sentence), Membership.mem S φ → Eq (setOf fun p  …
  -/
  rintro h _ ⟨φ, h1, rfl⟩
  /-
    case intro.intro
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    S : (L.withConstants α).Theory
    h✝ : Eq (setOf fun p => HasSubset.Subset S ↑p) (Set.image (fun φ => setOf fun  …
    h : ∀ (φ : (L.withConstants α).Sentence), Membership.mem S φ → Eq (setOf fun p …
    φ : (L.withConstants α).Sentence
    h1 : Membership.mem S φ
    ⊢ Eq ((fun φ => setOf fun p => Membership.mem p φ) φ) Set.univ
  -/
  exact h _ h1
  /-
    🎉 no goals
  -/


theorem nonempty_iff : Nonempty (T.CompleteType α) ↔ T.IsSatisfiable := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    ⊢ Iff (Nonempty (T.CompleteType α)) T.IsSatisfiable
  -/
  rw [← isSatisfiable_onTheory_iff (lhomWithConstants_injective L α)]
  rw [nonempty_iff_univ_nonempty, nonempty_iff_ne_empty, Ne, not_iff_comm,
    ← union_empty ((L.lhomWithConstants α).onTheory T), ← setOf_subset_eq_empty_iff]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    ⊢ Iff (Eq (setOf fun p => HasSubset.Subset EmptyCollection.emptyCollection ↑p) …
  -/
  simp
  /-
    🎉 no goals
  -/


instance instNonempty : Nonempty (CompleteType (∅ : L.Theory) α) :=
  nonempty_iff.2 (isSatisfiable_empty L)


theorem iInter_setOf_subset {ι : Type*} (S : ι → L[[α]].Theory) :
    ⋂ i : ι, { p : T.CompleteType α | S i ⊆ p } =
      { p : T.CompleteType α | ⋃ i : ι, S i ⊆ p } := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    ι : Type u_1
    S : ι → (L.withConstants α).Theory
    ⊢ Eq (Set.iInter fun i => setOf fun p => HasSubset.Subset (S i) ↑p) (setOf fun …
  -/
  ext
  /-
    case h
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    ι : Type u_1
    S : ι → (L.withConstants α).Theory
    x✝ : T.CompleteType α
    ⊢ Iff (Membership.mem (Set.iInter fun i => setOf fun p => HasSubset.Subset (S  …
  -/
  simp only [mem_iInter, mem_setOf_eq, iUnion_subset_iff]
  /-
    🎉 no goals
  -/


theorem toList_foldr_inf_mem {p : T.CompleteType α} {t : Finset (L[[α]]).Sentence} :
    t.toList.foldr (· ⊓ ·) ⊤ ∈ p ↔ (t : L[[α]].Theory) ⊆ ↑p := by
  simp_rw [subset_def, ← SetLike.mem_coe, p.isMaximal.mem_iff_models, models_sentence_iff,
    Sentence.Realize, Formula.Realize, BoundedFormula.realize_foldr_inf, Finset.mem_toList]
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    p : T.CompleteType α
    t : Finset (L.withConstants α).Sentence
    ⊢ Iff (∀ (M : FirstOrder.Language.Theory.ModelType ↑p) (φ : (L.withConstants α …
  -/
  exact ⟨fun h φ hφ M => h _ _ hφ, fun h M φ hφ => h _ hφ _⟩
  /-
    🎉 no goals
  -/


/-- The set of all formulas true at a tuple in a structure forms a complete type. -/
def typeOf (v : α → M) : T.CompleteType α :=
  haveI : (constantsOn α).Structure M := constantsOn.structure v
  { toTheory := L[[α]].completeTheory M
    subset' := model_iff_subset_completeTheory.1 ((LHom.onTheory_model _ T).2 inferInstance)
    isMaximal' := completeTheory.isMaximal _ _ }


@[simp]
theorem mem_typeOf {φ : L[[α]].Sentence} :
    φ ∈ T.typeOf v ↔ (Formula.equivSentence.symm φ).Realize v :=
  letI : (constantsOn α).Structure M := constantsOn.structure v
  mem_completeTheory.trans (Formula.realize_equivSentence_symm _ _ _).symm


theorem formula_mem_typeOf {φ : L.Formula α} :
                                                             /-
                                                               L : FirstOrder.Language
                                                               T : L.Theory
                                                               α : Type w
                                                               M : Type w'
                                                               inst✝² : L.Structure M
                                                               inst✝¹ : Nonempty M
                                                               inst✝ : FirstOrder.Language.Theory.Model M T
                                                               v : α → M
                                                               φ : L.Formula α
                                                               ⊢ Iff (Membership.mem (T.typeOf v) (FirstOrder.Language.Formula.equivSentence  …
                                                             -/
    Formula.equivSentence φ ∈ T.typeOf v ↔ φ.Realize v := by simp
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- A complete type `p` is realized in a particular structure when there is some
  tuple `v` whose type is `p`. -/
@[simp]
def realizedTypes (α : Type w) : Set (T.CompleteType α) :=
  Set.range (T.typeOf : (α → M) → T.CompleteType α)


theorem exists_modelType_is_realized_in (p : T.CompleteType α) :
    ∃ M : Theory.ModelType.{u, v, max u v w} T, p ∈ T.realizedTypes M α := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    p : T.CompleteType α
    ⊢ Exists fun M => Membership.mem (T.realizedTypes (↑M) α) p
  -/
  obtain ⟨M⟩ := p.isMaximal.1
  /-
    case intro
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    p : T.CompleteType α
    M : FirstOrder.Language.Theory.ModelType ↑p
    ⊢ Exists fun M => Membership.mem (T.realizedTypes (↑M) α) p
  -/
  refine ⟨(M.subtheoryModel p.subset).reduct (L.lhomWithConstants α), fun a => (L.con a : M), ?_⟩
  /-
    case intro
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    p : T.CompleteType α
    M : FirstOrder.Language.Theory.ModelType ↑p
    ⊢ Eq (T.typeOf fun a => ↑(L.con a)) p
  -/
  refine SetLike.ext fun φ => ?_
  /-
    case intro
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    p : T.CompleteType α
    M : FirstOrder.Language.Theory.ModelType ↑p
    φ : (L.withConstants α).Sentence
    ⊢ Iff (Membership.mem (T.typeOf fun a => ↑(L.con a)) φ) (Membership.mem p φ)
  -/
  simp only [CompleteType.mem_typeOf]
  refine
    (@Formula.realize_equivSentence_symm_con _
      ((M.subtheoryModel p.subset).reduct (L.lhomWithConstants α)) _ _ M.struc _ φ).trans
      (_root_.trans (_root_.trans ?_ (p.isMaximal.isComplete.realize_sentence_iff φ M))
        (p.isMaximal.mem_iff_models φ).symm)
  /-
    case intro
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    p : T.CompleteType α
    M : FirstOrder.Language.Theory.ModelType ↑p
    φ : (L.withConstants α).Sentence
    ⊢ Iff (FirstOrder.Language.Sentence.Realize (↑(FirstOrder.Language.Theory.Mode …
  -/
  rfl
  /-
    🎉 no goals
  -/


