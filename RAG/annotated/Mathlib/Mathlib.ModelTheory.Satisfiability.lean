/-- A theory is satisfiable if a structure models it. -/
def IsSatisfiable : Prop :=
  Nonempty (ModelType.{u, v, max u v} T)


/-- A theory is finitely satisfiable if all of its finite subtheories are satisfiable. -/
def IsFinitelySatisfiable : Prop :=
  ∀ T0 : Finset L.Sentence, (T0 : L.Theory) ⊆ T → IsSatisfiable (T0 : L.Theory)


theorem Model.isSatisfiable (M : Type w) [Nonempty M] [L.Structure M] [M ⊨ T] :
    T.IsSatisfiable :=
  ⟨((⊥ : Substructure _ (ModelType.of T M)).elementarySkolem₁Reduct.toModel T).shrink⟩


theorem IsSatisfiable.mono (h : T'.IsSatisfiable) (hs : T ⊆ T') : T.IsSatisfiable :=
  ⟨(Theory.Model.mono (ModelType.is_model h.some) hs).bundled⟩


theorem isSatisfiable_empty (L : Language.{u, v}) : IsSatisfiable (∅ : L.Theory) :=
  ⟨default⟩


theorem isSatisfiable_of_isSatisfiable_onTheory {L' : Language.{w, w'}} (φ : L →ᴸ L')
    (h : (φ.onTheory T).IsSatisfiable) : T.IsSatisfiable :=
  Model.isSatisfiable (h.some.reduct φ)


theorem isSatisfiable_onTheory_iff {L' : Language.{w, w'}} {φ : L →ᴸ L'} (h : φ.Injective) :
    (φ.onTheory T).IsSatisfiable ↔ T.IsSatisfiable := by
  classical
    refine ⟨isSatisfiable_of_isSatisfiable_onTheory φ, fun h' => ?_⟩
    haveI : Inhabited h'.some := Classical.inhabited_of_nonempty'
    exact Model.isSatisfiable (h'.some.defaultExpansion h)


theorem IsSatisfiable.isFinitelySatisfiable (h : T.IsSatisfiable) : T.IsFinitelySatisfiable :=
  fun _ => h.mono


/-- The **Compactness Theorem of first-order logic**: A theory is satisfiable if and only if it is
finitely satisfiable. -/
theorem isSatisfiable_iff_isFinitelySatisfiable {T : L.Theory} :
    T.IsSatisfiable ↔ T.IsFinitelySatisfiable :=
  ⟨Theory.IsSatisfiable.isFinitelySatisfiable, fun h => by
    classical
      set M : Finset T → Type max u v := fun T0 : Finset T =>
        (h (T0.map (Function.Embedding.subtype fun x => x ∈ T)) T0.map_subtype_subset).some.Carrier
      let M' := Filter.Product (Ultrafilter.of (Filter.atTop : Filter (Finset T))) M
      have h' : M' ⊨ T := by
        refine ⟨fun φ hφ => ?_⟩
        rw [Ultraproduct.sentence_realize]
        refine
          Filter.Eventually.filter_mono (Ultrafilter.of_le _)
            (Filter.eventually_atTop.2
              ⟨{⟨φ, hφ⟩}, fun s h' =>
                Theory.realize_sentence_of_mem (s.map (Function.Embedding.subtype fun x => x ∈ T))
                  ?_⟩)
        simp only [Finset.coe_map, Function.Embedding.coe_subtype, Set.mem_image, Finset.mem_coe,
          Subtype.exists, Subtype.coe_mk, exists_and_right, exists_eq_right]
        exact ⟨hφ, h' (Finset.mem_singleton_self _)⟩
      exact ⟨ModelType.of T M'⟩⟩


theorem isSatisfiable_directed_union_iff {ι : Type*} [Nonempty ι] {T : ι → L.Theory}
    (h : Directed (· ⊆ ·) T) : Theory.IsSatisfiable (⋃ i, T i) ↔ ∀ i, (T i).IsSatisfiable := by
  /-
    L : FirstOrder.Language
    ι : Type u_1
    inst✝ : Nonempty ι
    T : ι → L.Theory
    h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) T
    ⊢ Iff (FirstOrder.Language.Theory.IsSatisfiable (Set.iUnion fun i => T i)) (∀  …
  -/
  refine ⟨fun h' i => h'.mono (Set.subset_iUnion _ _), fun h' => ?_⟩
  /-
    L : FirstOrder.Language
    ι : Type u_1
    inst✝ : Nonempty ι
    T : ι → L.Theory
    h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) T
    h' : ∀ (i : ι), (T i).IsSatisfiable
    ⊢ FirstOrder.Language.Theory.IsSatisfiable (Set.iUnion fun i => T i)
  -/
  rw [isSatisfiable_iff_isFinitelySatisfiable, IsFinitelySatisfiable]
  /-
    L : FirstOrder.Language
    ι : Type u_1
    inst✝ : Nonempty ι
    T : ι → L.Theory
    h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) T
    h' : ∀ (i : ι), (T i).IsSatisfiable
    ⊢ ∀ (T0 : Finset L.Sentence), HasSubset.Subset (↑T0) (Set.iUnion fun i => T i) …
  -/
  intro T0 hT0
  /-
    L : FirstOrder.Language
    ι : Type u_1
    inst✝ : Nonempty ι
    T : ι → L.Theory
    h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) T
    h' : ∀ (i : ι), (T i).IsSatisfiable
    T0 : Finset L.Sentence
    hT0 : HasSubset.Subset (↑T0) (Set.iUnion fun i => T i)
    ⊢ FirstOrder.Language.Theory.IsSatisfiable ↑T0
  -/
  obtain ⟨i, hi⟩ := h.exists_mem_subset_of_finset_subset_biUnion hT0
  /-
    case intro
    L : FirstOrder.Language
    ι : Type u_1
    inst✝ : Nonempty ι
    T : ι → L.Theory
    h : Directed (fun x1 x2 => HasSubset.Subset x1 x2) T
    h' : ∀ (i : ι), (T i).IsSatisfiable
    T0 : Finset L.Sentence
    hT0 : HasSubset.Subset (↑T0) (Set.iUnion fun i => T i)
    i : ι
    hi : HasSubset.Subset (↑T0) (T i)
    ⊢ FirstOrder.Language.Theory.IsSatisfiable ↑T0
  -/
  exact (h' i).mono hi
  /-
    🎉 no goals
  -/


theorem isSatisfiable_union_distinctConstantsTheory_of_card_le (T : L.Theory) (s : Set α)
    (M : Type w') [Nonempty M] [L.Structure M] [M ⊨ T]
    (h : Cardinal.lift.{w'} #s ≤ Cardinal.lift.{w} #M) :
    ((L.lhomWithConstants α).onTheory T ∪ L.distinctConstantsTheory s).IsSatisfiable := by
  /-
    L : FirstOrder.Language
    α : Type w
    T : L.Theory
    s : Set α
    M : Type w'
    inst✝² : Nonempty M
    inst✝¹ : L.Structure M
    inst✝ : FirstOrder.Language.Theory.Model M T
    h : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.lift.{w, w'} (Car …
    ⊢ (Union.union ((L.lhomWithConstants α).onTheory T) (L.distinctConstantsTheory …
  -/
  haveI : Inhabited M := Classical.inhabited_of_nonempty inferInstance
  /-
    L : FirstOrder.Language
    α : Type w
    T : L.Theory
    s : Set α
    M : Type w'
    inst✝² : Nonempty M
    inst✝¹ : L.Structure M
    inst✝ : FirstOrder.Language.Theory.Model M T
    h : LE.le (Cardinal.lift.{w', w} (Cardinal.mk ↑s)) (Cardinal.lift.{w, w'} (Car …
    this : Inhabited M
    ⊢ (Union.union ((L.lhomWithConstants α).onTheory T) (L.distinctConstantsTheory …
  -/
  rw [Cardinal.lift_mk_le'] at h
  /-
    L : FirstOrder.Language
    α : Type w
    T : L.Theory
    s : Set α
    M : Type w'
    inst✝² : Nonempty M
    inst✝¹ : L.Structure M
    inst✝ : FirstOrder.Language.Theory.Model M T
    h : Nonempty (Function.Embedding (↑s) M)
    this : Inhabited M
    ⊢ (Union.union ((L.lhomWithConstants α).onTheory T) (L.distinctConstantsTheory …
  -/
  letI : (constantsOn α).Structure M := constantsOn.structure (Function.extend (↑) h.some default)
  have : M ⊨ (L.lhomWithConstants α).onTheory T ∪ L.distinctConstantsTheory s := by
    refine ((LHom.onTheory_model _ _).2 inferInstance).union ?_
    rw [model_distinctConstantsTheory]
    refine fun a as b bs ab => ?_
    rw [← Subtype.coe_mk a as, ← Subtype.coe_mk b bs, ← Subtype.ext_iff]
    exact
      h.some.injective
        ((Subtype.coe_injective.extend_apply h.some default ⟨a, as⟩).symm.trans
          (ab.trans (Subtype.coe_injective.extend_apply h.some default ⟨b, bs⟩)))
  /-
    L : FirstOrder.Language
    α : Type w
    T : L.Theory
    s : Set α
    M : Type w'
    inst✝² : Nonempty M
    inst✝¹ : L.Structure M
    inst✝ : FirstOrder.Language.Theory.Model M T
    h : Nonempty (Function.Embedding (↑s) M)
    this✝¹ : Inhabited M
    this✝ : (FirstOrder.Language.constantsOn α).Structure M := FirstOrder.Language …
    this : FirstOrder.Language.Theory.Model M (Union.union ((L.lhomWithConstants α …
    ⊢ (Union.union ((L.lhomWithConstants α).onTheory T) (L.distinctConstantsTheory …
  -/
  exact Model.isSatisfiable M
  /-
    🎉 no goals
  -/


theorem isSatisfiable_union_distinctConstantsTheory_of_infinite (T : L.Theory) (s : Set α)
    (M : Type w') [L.Structure M] [M ⊨ T] [Infinite M] :
    ((L.lhomWithConstants α).onTheory T ∪ L.distinctConstantsTheory s).IsSatisfiable := by
  classical
    rw [distinctConstantsTheory_eq_iUnion, Set.union_iUnion, isSatisfiable_directed_union_iff]
    · exact fun t =>
        isSatisfiable_union_distinctConstantsTheory_of_card_le T _ M
          ((lift_le_aleph0.2 (finset_card_lt_aleph0 _).le).trans
            (aleph0_le_lift.2 (aleph0_le_mk M)))
    · apply Monotone.directed_le
      refine monotone_const.union (monotone_distinctConstantsTheory.comp ?_)
      simp only [Finset.coe_map, Function.Embedding.coe_subtype]
      exact Monotone.comp (g := Set.image ((↑) : s → α)) (f := ((↑) : Finset s → Set s))
        Set.monotone_image fun _ _ => Finset.coe_subset.2


/-- Any theory with an infinite model has arbitrarily large models. -/
theorem exists_large_model_of_infinite_model (T : L.Theory) (κ : Cardinal.{w}) (M : Type w')
    [L.Structure M] [M ⊨ T] [Infinite M] :
    ∃ N : ModelType.{_, _, max u v w} T, Cardinal.lift.{max u v w} κ ≤ #N := by
  obtain ⟨N⟩ :=
    isSatisfiable_union_distinctConstantsTheory_of_infinite T (Set.univ : Set κ.out) M
  /-
    case intro
    L : FirstOrder.Language
    T : L.Theory
    κ : Cardinal.{w}
    M : Type w'
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Infinite M
    N : (Union.union ((L.lhomWithConstants (Quotient.out κ)).onTheory T) (L.distin …
    ⊢ Exists fun N => LE.le (Cardinal.lift.{max u v w, w} κ) (Cardinal.mk ↑N)
  -/
  refine ⟨(N.is_model.mono Set.subset_union_left).bundled.reduct _, ?_⟩
  /-
    case intro
    L : FirstOrder.Language
    T : L.Theory
    κ : Cardinal.{w}
    M : Type w'
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Infinite M
    N : (Union.union ((L.lhomWithConstants (Quotient.out κ)).onTheory T) (L.distin …
    ⊢ LE.le (Cardinal.lift.{max u v w, w} κ) (Cardinal.mk ↑(FirstOrder.Language.Th …
  -/
  haveI : N ⊨ distinctConstantsTheory _ _ := N.is_model.mono Set.subset_union_right
  /-
    case intro
    L : FirstOrder.Language
    T : L.Theory
    κ : Cardinal.{w}
    M : Type w'
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Infinite M
    N : (Union.union ((L.lhomWithConstants (Quotient.out κ)).onTheory T) (L.distin …
    this : FirstOrder.Language.Theory.Model (↑N) (L.distinctConstantsTheory Set.un …
    ⊢ LE.le (Cardinal.lift.{max u v w, w} κ) (Cardinal.mk ↑(FirstOrder.Language.Th …
  -/
  rw [ModelType.reduct_Carrier, coe_of]
  /-
    case intro
    L : FirstOrder.Language
    T : L.Theory
    κ : Cardinal.{w}
    M : Type w'
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Infinite M
    N : (Union.union ((L.lhomWithConstants (Quotient.out κ)).onTheory T) (L.distin …
    this : FirstOrder.Language.Theory.Model (↑N) (L.distinctConstantsTheory Set.un …
    ⊢ LE.le (Cardinal.lift.{max u v w, w} κ) (Cardinal.mk ↑N)
  -/
  refine _root_.trans (lift_le.2 (le_of_eq (Cardinal.mk_out κ).symm)) ?_
  /-
    case intro
    L : FirstOrder.Language
    T : L.Theory
    κ : Cardinal.{w}
    M : Type w'
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Infinite M
    N : (Union.union ((L.lhomWithConstants (Quotient.out κ)).onTheory T) (L.distin …
    this : FirstOrder.Language.Theory.Model (↑N) (L.distinctConstantsTheory Set.un …
    ⊢ LE.le (Cardinal.lift.{max (max u v) w, w} (Cardinal.mk (Quotient.out κ))) (C …
  -/
  rw [← mk_univ]
  refine
    (card_le_of_model_distinctConstantsTheory L Set.univ N).trans (lift_le.{max u v w}.1 ?_)
  /-
    case intro
    L : FirstOrder.Language
    T : L.Theory
    κ : Cardinal.{w}
    M : Type w'
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Infinite M
    N : (Union.union ((L.lhomWithConstants (Quotient.out κ)).onTheory T) (L.distin …
    this : FirstOrder.Language.Theory.Model (↑N) (L.distinctConstantsTheory Set.un …
    ⊢ LE.le (Cardinal.lift.{max u v w, max (max u v) w} (Cardinal.lift.{w, max (ma …
  -/
  rw [lift_lift]
  /-
    🎉 no goals
  -/


theorem isSatisfiable_iUnion_iff_isSatisfiable_iUnion_finset {ι : Type*} (T : ι → L.Theory) :
    IsSatisfiable (⋃ i, T i) ↔ ∀ s : Finset ι, IsSatisfiable (⋃ i ∈ s, T i) := by
  classical
    refine
      ⟨fun h s => h.mono (Set.iUnion_mono fun _ => Set.iUnion_subset_iff.2 fun _ => refl _),
        fun h => ?_⟩
    rw [isSatisfiable_iff_isFinitelySatisfiable]
    intro s hs
    rw [Set.iUnion_eq_iUnion_finset] at hs
    obtain ⟨t, ht⟩ := Directed.exists_mem_subset_of_finset_subset_biUnion (by
      exact Monotone.directed_le fun t1 t2 (h : ∀ ⦃x⦄, x ∈ t1 → x ∈ t2) =>
        Set.iUnion_mono fun _ => Set.iUnion_mono' fun h1 => ⟨h h1, refl _⟩) hs
    exact (h t).mono ht


/-- A version of The Downward Löwenheim–Skolem theorem where the structure `N` elementarily embeds
into `M`, but is not by type a substructure of `M`, and thus can be chosen to belong to the universe
of the cardinal `κ`.
 -/
theorem exists_elementaryEmbedding_card_eq_of_le (M : Type w') [L.Structure M] [Nonempty M]
    (κ : Cardinal.{w}) (h1 : ℵ₀ ≤ κ) (h2 : lift.{w} L.card ≤ Cardinal.lift.{max u v} κ)
    (h3 : lift.{w'} κ ≤ Cardinal.lift.{w} #M) :
    ∃ N : Bundled L.Structure, Nonempty (N ↪ₑ[L] M) ∧ #N = κ := by
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝¹ : L.Structure M
    inst✝ : Nonempty M
    κ : Cardinal.{w}
    h1 : LE.le Cardinal.aleph0 κ
    h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
    h3 : LE.le (Cardinal.lift.{w', w} κ) (Cardinal.lift.{w, w'} (Cardinal.mk M))
    ⊢ Exists fun N => And (Nonempty (L.ElementaryEmbedding (↑N) M)) (Eq (Cardinal. …
  -/
  obtain ⟨S, _, hS⟩ := exists_elementarySubstructure_card_eq L ∅ κ h1 (by simp) h2 h3
  have : Small.{w} S := by
    rw [← lift_inj.{_, w + 1}, lift_lift, lift_lift] at hS
    exact small_iff_lift_mk_lt_univ.2 (lt_of_eq_of_lt hS κ.lift_lt_univ')
  refine
    ⟨(equivShrink S).bundledInduced L,
      ⟨S.subtype.comp (Equiv.bundledInducedEquiv L _).symm.toElementaryEmbedding⟩,
      lift_inj.1 (_root_.trans ?_ hS)⟩
  /-
    case intro.intro
    L : FirstOrder.Language
    M : Type w'
    inst✝¹ : L.Structure M
    inst✝ : Nonempty M
    κ : Cardinal.{w}
    h1 : LE.le Cardinal.aleph0 κ
    h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
    h3 : LE.le (Cardinal.lift.{w', w} κ) (Cardinal.lift.{w, w'} (Cardinal.mk M))
    S : L.ElementarySubstructure M
    left✝ : HasSubset.Subset EmptyCollection.emptyCollection ↑S
    hS : Eq (Cardinal.lift.{w, w'} (Cardinal.mk (Subtype fun x => Membership.mem S …
    this : Small.{w, w'} (Subtype fun x => Membership.mem S x)
    ⊢ Eq (Cardinal.lift.{w', w} (Cardinal.mk ↑(Equiv.bundledInduced L (equivShrink …
  -/
  simp only [Equiv.bundledInduced_α, lift_mk_shrink']
  /-
    🎉 no goals
  -/


/-- The **Upward Löwenheim–Skolem Theorem**: If `κ` is a cardinal greater than the cardinalities of
`L` and an infinite `L`-structure `M`, then `M` has an elementary extension of cardinality `κ`. -/
theorem exists_elementaryEmbedding_card_eq_of_ge (M : Type w') [L.Structure M] [iM : Infinite M]
    (κ : Cardinal.{w}) (h1 : Cardinal.lift.{w} L.card ≤ Cardinal.lift.{max u v} κ)
    (h2 : Cardinal.lift.{w} #M ≤ Cardinal.lift.{w'} κ) :
    ∃ N : Bundled L.Structure, Nonempty (M ↪ₑ[L] N) ∧ #N = κ := by
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝ : L.Structure M
    iM : Infinite M
    κ : Cardinal.{w}
    h1 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
    h2 : LE.le (Cardinal.lift.{w, w'} (Cardinal.mk M)) (Cardinal.lift.{w', w} κ)
    ⊢ Exists fun N => And (Nonempty (L.ElementaryEmbedding M ↑N)) (Eq (Cardinal.mk …
  -/
  obtain ⟨N0, hN0⟩ := (L.elementaryDiagram M).exists_large_model_of_infinite_model κ M
  /-
    case intro
    L : FirstOrder.Language
    M : Type w'
    inst✝ : L.Structure M
    iM : Infinite M
    κ : Cardinal.{w}
    h1 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
    h2 : LE.le (Cardinal.lift.{w, w'} (Cardinal.mk M)) (Cardinal.lift.{w', w} κ)
    N0 : (L.elementaryDiagram M).ModelType
    hN0 : LE.le (Cardinal.lift.{max (max u w') v w, w} κ) (Cardinal.mk ↑N0)
    ⊢ Exists fun N => And (Nonempty (L.ElementaryEmbedding M ↑N)) (Eq (Cardinal.mk …
  -/
  rw [← lift_le.{max u v}, lift_lift, lift_lift] at h2
  obtain ⟨N, ⟨NN0⟩, hN⟩ :=
    exists_elementaryEmbedding_card_eq_of_le (L[[M]]) N0 κ
      (aleph0_le_lift.1 ((aleph0_le_lift.2 (aleph0_le_mk M)).trans h2))
      (by
        simp only [card_withConstants, lift_add, lift_lift]
        rw [add_comm, add_eq_max (aleph0_le_lift.2 (infinite_iff.1 iM)), max_le_iff]
        rw [← lift_le.{w'}, lift_lift, lift_lift] at h1
        exact ⟨h2, h1⟩)
      (hN0.trans (by rw [← lift_umax, lift_id]))
  /-
    case intro.intro.intro.intro
    L : FirstOrder.Language
    M : Type w'
    inst✝ : L.Structure M
    iM : Infinite M
    κ : Cardinal.{w}
    h1 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
    h2 : LE.le (Cardinal.lift.{max w u v, w'} (Cardinal.mk M)) (Cardinal.lift.{max …
    N0 : (L.elementaryDiagram M).ModelType
    hN0 : LE.le (Cardinal.lift.{max (max u w') v w, w} κ) (Cardinal.mk ↑N0)
    N : CategoryTheory.Bundled (L.withConstants M).Structure
    hN : Eq (Cardinal.mk ↑N) κ
    NN0 : (L.withConstants M).ElementaryEmbedding ↑N ↑N0
    ⊢ Exists fun N => And (Nonempty (L.ElementaryEmbedding M ↑N)) (Eq (Cardinal.mk …
  -/
  letI := (lhomWithConstants L M).reduct N
  haveI h : N ⊨ L.elementaryDiagram M :=
    (NN0.theory_model_iff (L.elementaryDiagram M)).2 inferInstance
  /-
    case intro.intro.intro.intro
    L : FirstOrder.Language
    M : Type w'
    inst✝ : L.Structure M
    iM : Infinite M
    κ : Cardinal.{w}
    h1 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
    h2 : LE.le (Cardinal.lift.{max w u v, w'} (Cardinal.mk M)) (Cardinal.lift.{max …
    N0 : (L.elementaryDiagram M).ModelType
    hN0 : LE.le (Cardinal.lift.{max (max u w') v w, w} κ) (Cardinal.mk ↑N0)
    N : CategoryTheory.Bundled (L.withConstants M).Structure
    hN : Eq (Cardinal.mk ↑N) κ
    NN0 : (L.withConstants M).ElementaryEmbedding ↑N ↑N0
    this : L.Structure ↑N := (L.lhomWithConstants M).reduct ↑N
    h : FirstOrder.Language.Theory.Model (↑N) (L.elementaryDiagram M)
    ⊢ Exists fun N => And (Nonempty (L.ElementaryEmbedding M ↑N)) (Eq (Cardinal.mk …
  -/
  refine ⟨Bundled.of N, ⟨?_⟩, hN⟩
  /-
    case intro.intro.intro.intro
    L : FirstOrder.Language
    M : Type w'
    inst✝ : L.Structure M
    iM : Infinite M
    κ : Cardinal.{w}
    h1 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
    h2 : LE.le (Cardinal.lift.{max w u v, w'} (Cardinal.mk M)) (Cardinal.lift.{max …
    N0 : (L.elementaryDiagram M).ModelType
    hN0 : LE.le (Cardinal.lift.{max (max u w') v w, w} κ) (Cardinal.mk ↑N0)
    N : CategoryTheory.Bundled (L.withConstants M).Structure
    hN : Eq (Cardinal.mk ↑N) κ
    NN0 : (L.withConstants M).ElementaryEmbedding ↑N ↑N0
    this : L.Structure ↑N := (L.lhomWithConstants M).reduct ↑N
    h : FirstOrder.Language.Theory.Model (↑N) (L.elementaryDiagram M)
    ⊢ L.ElementaryEmbedding M ↑(CategoryTheory.Bundled.of ↑N)
  -/
  apply ElementaryEmbedding.ofModelsElementaryDiagram L M N
  /-
    🎉 no goals
  -/


/-- The Löwenheim–Skolem Theorem: If `κ` is a cardinal greater than the cardinalities of `L`
and an infinite `L`-structure `M`, then there is an elementary embedding in the appropriate
direction between then `M` and a structure of cardinality `κ`. -/
theorem exists_elementaryEmbedding_card_eq (M : Type w') [L.Structure M] [iM : Infinite M]
    (κ : Cardinal.{w}) (h1 : ℵ₀ ≤ κ) (h2 : lift.{w} L.card ≤ Cardinal.lift.{max u v} κ) :
    ∃ N : Bundled L.Structure, (Nonempty (N ↪ₑ[L] M) ∨ Nonempty (M ↪ₑ[L] N)) ∧ #N = κ := by
  cases le_or_gt (lift.{w'} κ) (Cardinal.lift.{w} #M) with
  | inl h =>
    obtain ⟨N, hN1, hN2⟩ := exists_elementaryEmbedding_card_eq_of_le L M κ h1 h2 h
    exact ⟨N, Or.inl hN1, hN2⟩
  | inr h =>
    obtain ⟨N, hN1, hN2⟩ := exists_elementaryEmbedding_card_eq_of_ge L M κ h2 (le_of_lt h)
    exact ⟨N, Or.inr hN1, hN2⟩


/-- A consequence of the Löwenheim–Skolem Theorem: If `κ` is a cardinal greater than the
cardinalities of `L` and an infinite `L`-structure `M`, then there is a structure of cardinality `κ`
elementarily equivalent to `M`. -/
theorem exists_elementarilyEquivalent_card_eq (M : Type w') [L.Structure M] [Infinite M]
    (κ : Cardinal.{w}) (h1 : ℵ₀ ≤ κ) (h2 : lift.{w} L.card ≤ Cardinal.lift.{max u v} κ) :
    ∃ N : CategoryTheory.Bundled L.Structure, (M ≅[L] N) ∧ #N = κ := by
  /-
    L : FirstOrder.Language
    M : Type w'
    inst✝¹ : L.Structure M
    inst✝ : Infinite M
    κ : Cardinal.{w}
    h1 : LE.le Cardinal.aleph0 κ
    h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
    ⊢ Exists fun N => And (L.ElementarilyEquivalent M ↑N) (Eq (Cardinal.mk ↑N) κ)
  -/
  obtain ⟨N, NM | MN, hNκ⟩ := exists_elementaryEmbedding_card_eq L M κ h1 h2
    /-
      case intro.intro.inl
      L : FirstOrder.Language
      M : Type w'
      inst✝¹ : L.Structure M
      inst✝ : Infinite M
      κ : Cardinal.{w}
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      N : CategoryTheory.Bundled L.Structure
      hNκ : Eq (Cardinal.mk ↑N) κ
      NM : Nonempty (L.ElementaryEmbedding (↑N) M)
      ⊢ Exists fun N => And (L.ElementarilyEquivalent M ↑N) (Eq (Cardinal.mk ↑N) κ)
    -/
  · exact ⟨N, NM.some.elementarilyEquivalent.symm, hNκ⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      L : FirstOrder.Language
      M : Type w'
      inst✝¹ : L.Structure M
      inst✝ : Infinite M
      κ : Cardinal.{w}
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      N : CategoryTheory.Bundled L.Structure
      hNκ : Eq (Cardinal.mk ↑N) κ
      MN : Nonempty (L.ElementaryEmbedding M ↑N)
      ⊢ Exists fun N => And (L.ElementarilyEquivalent M ↑N) (Eq (Cardinal.mk ↑N) κ)
    -/
  · exact ⟨N, MN.some.elementarilyEquivalent, hNκ⟩
    /-
      🎉 no goals
    -/


theorem exists_model_card_eq (h : ∃ M : ModelType.{u, v, max u v} T, Infinite M) (κ : Cardinal.{w})
    (h1 : ℵ₀ ≤ κ) (h2 : Cardinal.lift.{w} L.card ≤ Cardinal.lift.{max u v} κ) :
    ∃ N : ModelType.{u, v, w} T, #N = κ := by
  cases h with
  | intro M MI =>
    haveI := MI
    obtain ⟨N, hN, rfl⟩ := exists_elementarilyEquivalent_card_eq L M κ h1 h2
    haveI : Nonempty N := hN.nonempty
    exact ⟨hN.theory_model.bundled, rfl⟩


/-- A theory models a (bounded) formula when any of its nonempty models realizes that formula on all
  inputs. -/
def ModelsBoundedFormula (φ : L.BoundedFormula α n) : Prop :=
  ∀ (M : ModelType.{u, v, max u v w} T) (v : α → M) (xs : Fin n → M), φ.Realize v xs

-- Porting note: In Lean3 it was `⊨` but ambiguous.

@[inherit_doc FirstOrder.Language.Theory.ModelsBoundedFormula]
infixl:51 " ⊨ᵇ " => ModelsBoundedFormula -- input using \|= or \vDash, but not using \models


theorem models_formula_iff {φ : L.Formula α} :
    T ⊨ᵇ φ ↔ ∀ (M : ModelType.{u, v, max u v w} T) (v : α → M), φ.Realize v :=
  forall_congr' fun _ => forall_congr' fun _ => Unique.forall_iff


theorem models_sentence_iff {φ : L.Sentence} : T ⊨ᵇ φ ↔ ∀ M : ModelType.{u, v, max u v} T, M ⊨ φ :=
  models_formula_iff.trans (forall_congr' fun _ => Unique.forall_iff)


theorem models_sentence_of_mem {φ : L.Sentence} (h : φ ∈ T) : T ⊨ᵇ φ :=
  models_sentence_iff.2 fun _ => realize_sentence_of_mem T h


theorem models_iff_not_satisfiable (φ : L.Sentence) : T ⊨ᵇ φ ↔ ¬IsSatisfiable (T ∪ {φ.not}) := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    ⊢ Iff (T.ModelsBoundedFormula φ) (Not (Union.union T (Singleton.singleton (Fir …
  -/
  rw [models_sentence_iff, IsSatisfiable]
  refine
    ⟨fun h1 h2 =>
      (Sentence.realize_not _).1
        (realize_sentence_of_mem (T ∪ {Formula.not φ})
          (Set.subset_union_right (Set.mem_singleton _)))
        (h1 (h2.some.subtheoryModel Set.subset_union_left)),
      fun h M => ?_⟩
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    h : Not (Nonempty (Union.union T (Singleton.singleton (FirstOrder.Language.For …
    M : T.ModelType
    ⊢ FirstOrder.Language.Sentence.Realize (↑M) φ
  -/
  contrapose! h
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    M : T.ModelType
    h : Not (FirstOrder.Language.Sentence.Realize (↑M) φ)
    ⊢ Nonempty (Union.union T (Singleton.singleton (FirstOrder.Language.Formula.no …
  -/
  rw [← Sentence.realize_not] at h
  refine
    ⟨{  Carrier := M
        is_model := ⟨fun ψ hψ => hψ.elim (realize_sentence_of_mem _) fun h' => ?_⟩ }⟩
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    M : T.ModelType
    h : FirstOrder.Language.Sentence.Realize (↑M) (FirstOrder.Language.Formula.not …
    ψ : L.Sentence
    hψ : Membership.mem (Union.union T (Singleton.singleton (FirstOrder.Language.F …
    h' : Membership.mem (Singleton.singleton (FirstOrder.Language.Formula.not φ)) ψ
    ⊢ FirstOrder.Language.Sentence.Realize (↑M) ψ
  -/
  rw [Set.mem_singleton_iff.1 h']
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    M : T.ModelType
    h : FirstOrder.Language.Sentence.Realize (↑M) (FirstOrder.Language.Formula.not …
    ψ : L.Sentence
    hψ : Membership.mem (Union.union T (Singleton.singleton (FirstOrder.Language.F …
    h' : Membership.mem (Singleton.singleton (FirstOrder.Language.Formula.not φ)) ψ
    ⊢ FirstOrder.Language.Sentence.Realize (↑M) (FirstOrder.Language.Formula.not φ)
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem ModelsBoundedFormula.realize_sentence {φ : L.Sentence} (h : T ⊨ᵇ φ) (M : Type*)
    [L.Structure M] [M ⊨ T] [Nonempty M] : M ⊨ φ := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    h : T.ModelsBoundedFormula φ
    M : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Nonempty M
    ⊢ FirstOrder.Language.Sentence.Realize M φ
  -/
  rw [models_iff_not_satisfiable] at h
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    h : Not (Union.union T (Singleton.singleton (FirstOrder.Language.Formula.not φ …
    M : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Nonempty M
    ⊢ FirstOrder.Language.Sentence.Realize M φ
  -/
  contrapose! h
  have : M ⊨ T ∪ {Formula.not φ} := by
    simp only [Set.union_singleton, model_iff, Set.mem_insert_iff, forall_eq_or_imp,
      Sentence.realize_not]
    rw [← model_iff]
    exact ⟨h, inferInstance⟩
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    M : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Nonempty M
    h : Not (FirstOrder.Language.Sentence.Realize M φ)
    this : FirstOrder.Language.Theory.Model M (Union.union T (Singleton.singleton  …
    ⊢ (Union.union T (Singleton.singleton (FirstOrder.Language.Formula.not φ))).Is …
  -/
  exact Model.isSatisfiable M
  /-
    🎉 no goals
  -/


theorem models_formula_iff_onTheory_models_equivSentence {φ : L.Formula α} :
    T ⊨ᵇ φ ↔ (L.lhomWithConstants α).onTheory T ⊨ᵇ Formula.equivSentence φ := by
  refine ⟨fun h => models_sentence_iff.2 (fun M => ?_),
    fun h => models_formula_iff.2 (fun M v => ?_)⟩
    /-
      case refine_1
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      φ : L.Formula α
      h : T.ModelsBoundedFormula φ
      M : ((L.lhomWithConstants α).onTheory T).ModelType
      ⊢ FirstOrder.Language.Sentence.Realize (↑M) (FirstOrder.Language.Formula.equiv …
    -/
  · letI := (L.lhomWithConstants α).reduct M
    /-
      case refine_1
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      φ : L.Formula α
      h : T.ModelsBoundedFormula φ
      M : ((L.lhomWithConstants α).onTheory T).ModelType
      this : L.Structure ↑M := (L.lhomWithConstants α).reduct ↑M
      ⊢ FirstOrder.Language.Sentence.Realize (↑M) (FirstOrder.Language.Formula.equiv …
    -/
    have : (L.lhomWithConstants α).IsExpansionOn M := LHom.isExpansionOn_reduct _ _
      -- why doesn't that instance just work?
    /-
      case refine_1
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      φ : L.Formula α
      h : T.ModelsBoundedFormula φ
      M : ((L.lhomWithConstants α).onTheory T).ModelType
      this✝ : L.Structure ↑M := (L.lhomWithConstants α).reduct ↑M
      this : (L.lhomWithConstants α).IsExpansionOn ↑M
      ⊢ FirstOrder.Language.Sentence.Realize (↑M) (FirstOrder.Language.Formula.equiv …
    -/
    rw [Formula.realize_equivSentence]
    /-
      case refine_1
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      φ : L.Formula α
      h : T.ModelsBoundedFormula φ
      M : ((L.lhomWithConstants α).onTheory T).ModelType
      this✝ : L.Structure ↑M := (L.lhomWithConstants α).reduct ↑M
      this : (L.lhomWithConstants α).IsExpansionOn ↑M
      ⊢ φ.Realize fun a => ↑(L.con a)
    -/
    have : M ⊨ T := (LHom.onTheory_model _ _).1 M.is_model -- why isn't M.is_model inferInstance?
    /-
      case refine_1
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      φ : L.Formula α
      h : T.ModelsBoundedFormula φ
      M : ((L.lhomWithConstants α).onTheory T).ModelType
      this✝¹ : L.Structure ↑M := (L.lhomWithConstants α).reduct ↑M
      this✝ : (L.lhomWithConstants α).IsExpansionOn ↑M
      this : FirstOrder.Language.Theory.Model (↑M) T
      ⊢ φ.Realize fun a => ↑(L.con a)
    -/
    let M' := Theory.ModelType.of T M
    /-
      case refine_1
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      φ : L.Formula α
      h : T.ModelsBoundedFormula φ
      M : ((L.lhomWithConstants α).onTheory T).ModelType
      this✝¹ : L.Structure ↑M := (L.lhomWithConstants α).reduct ↑M
      this✝ : (L.lhomWithConstants α).IsExpansionOn ↑M
      this : FirstOrder.Language.Theory.Model (↑M) T
      M' : T.ModelType := FirstOrder.Language.Theory.ModelType.of T ↑M
      ⊢ φ.Realize fun a => ↑(L.con a)
    -/
    exact h M' (fun a => (L.con a : M)) _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      φ : L.Formula α
      h : ((L.lhomWithConstants α).onTheory T).ModelsBoundedFormula (FirstOrder.Lang …
      M : T.ModelType
      v : α → ↑M
      ⊢ φ.Realize v
    -/
  · letI : (constantsOn α).Structure M := constantsOn.structure v
    /-
      case refine_2
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      φ : L.Formula α
      h : ((L.lhomWithConstants α).onTheory T).ModelsBoundedFormula (FirstOrder.Lang …
      M : T.ModelType
      v : α → ↑M
      this : (FirstOrder.Language.constantsOn α).Structure ↑M := FirstOrder.Language …
      ⊢ φ.Realize v
    -/
    have : M ⊨ (L.lhomWithConstants α).onTheory T := (LHom.onTheory_model _ _).2 inferInstance
    /-
      case refine_2
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      φ : L.Formula α
      h : ((L.lhomWithConstants α).onTheory T).ModelsBoundedFormula (FirstOrder.Lang …
      M : T.ModelType
      v : α → ↑M
      this✝ : (FirstOrder.Language.constantsOn α).Structure ↑M := FirstOrder.Languag …
      this : FirstOrder.Language.Theory.Model (↑M) ((L.lhomWithConstants α).onTheory …
      ⊢ φ.Realize v
    -/
    exact (Formula.realize_equivSentence _ _).1 (h.realize_sentence M)
    /-
      🎉 no goals
    -/


theorem ModelsBoundedFormula.realize_formula {φ : L.Formula α} (h : T ⊨ᵇ φ) (M : Type*)
    [L.Structure M] [M ⊨ T] [Nonempty M] {v : α → M} : φ.Realize v := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    φ : L.Formula α
    h : T.ModelsBoundedFormula φ
    M : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Nonempty M
    v : α → M
    ⊢ φ.Realize v
  -/
  rw [models_formula_iff_onTheory_models_equivSentence] at h
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    φ : L.Formula α
    h : ((L.lhomWithConstants α).onTheory T).ModelsBoundedFormula (FirstOrder.Lang …
    M : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Nonempty M
    v : α → M
    ⊢ φ.Realize v
  -/
  letI : (constantsOn α).Structure M := constantsOn.structure v
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    φ : L.Formula α
    h : ((L.lhomWithConstants α).onTheory T).ModelsBoundedFormula (FirstOrder.Lang …
    M : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Nonempty M
    v : α → M
    this : (FirstOrder.Language.constantsOn α).Structure M := FirstOrder.Language. …
    ⊢ φ.Realize v
  -/
  have : M ⊨ (L.lhomWithConstants α).onTheory T := (LHom.onTheory_model _ _).2 inferInstance
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    φ : L.Formula α
    h : ((L.lhomWithConstants α).onTheory T).ModelsBoundedFormula (FirstOrder.Lang …
    M : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Nonempty M
    v : α → M
    this✝ : (FirstOrder.Language.constantsOn α).Structure M := FirstOrder.Language …
    this : FirstOrder.Language.Theory.Model M ((L.lhomWithConstants α).onTheory T)
    ⊢ φ.Realize v
  -/
  exact (Formula.realize_equivSentence _ _).1 (h.realize_sentence M)
  /-
    🎉 no goals
  -/


theorem models_toFormula_iff {φ : L.BoundedFormula α n} : T ⊨ᵇ φ.toFormula ↔ T ⊨ᵇ φ := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ : L.BoundedFormula α n
    ⊢ Iff (T.ModelsBoundedFormula φ.toFormula) (T.ModelsBoundedFormula φ)
  -/
  refine ⟨fun h M v xs => ?_, ?_⟩
    /-
      case refine_1
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      n : Nat
      φ : L.BoundedFormula α n
      h : T.ModelsBoundedFormula φ.toFormula
      M : T.ModelType
      v : α → ↑M
      xs : Fin n → ↑M
      ⊢ φ.Realize v xs
    -/
  · have h' : φ.toFormula.Realize (Sum.elim v xs) := h.realize_formula M
    /-
      case refine_1
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      n : Nat
      φ : L.BoundedFormula α n
      h : T.ModelsBoundedFormula φ.toFormula
      M : T.ModelType
      v : α → ↑M
      xs : Fin n → ↑M
      h' : φ.toFormula.Realize (Sum.elim v xs)
      ⊢ φ.Realize v xs
    -/
    simp only [BoundedFormula.realize_toFormula, Sum.elim_comp_inl, Sum.elim_comp_inr] at h'
    /-
      case refine_1
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      n : Nat
      φ : L.BoundedFormula α n
      h : T.ModelsBoundedFormula φ.toFormula
      M : T.ModelType
      v : α → ↑M
      xs : Fin n → ↑M
      h' : φ.Realize v xs
      ⊢ φ.Realize v xs
    -/
    exact h'
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      n : Nat
      φ : L.BoundedFormula α n
      ⊢ T.ModelsBoundedFormula φ → T.ModelsBoundedFormula φ.toFormula
    -/
  · simp only [models_formula_iff, BoundedFormula.realize_toFormula]
    /-
      case refine_2
      L : FirstOrder.Language
      T : L.Theory
      α : Type w
      n : Nat
      φ : L.BoundedFormula α n
      ⊢ T.ModelsBoundedFormula φ → ∀ (M : T.ModelType) (v : Sum α (Fin n) → ↑M), φ.R …
    -/
    exact fun h M v => h M _ _
    /-
      🎉 no goals
    -/


theorem ModelsBoundedFormula.realize_boundedFormula
    {φ : L.BoundedFormula α n} (h : T ⊨ᵇ φ) (M : Type*)
    [L.Structure M] [M ⊨ T] [Nonempty M] {v : α → M} {xs : Fin n → M} : φ.Realize v xs := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ : L.BoundedFormula α n
    h : T.ModelsBoundedFormula φ
    M : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Nonempty M
    v : α → M
    xs : Fin n → M
    ⊢ φ.Realize v xs
  -/
  have h' : φ.toFormula.Realize (Sum.elim v xs) := (models_toFormula_iff.2 h).realize_formula M
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ : L.BoundedFormula α n
    h : T.ModelsBoundedFormula φ
    M : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Nonempty M
    v : α → M
    xs : Fin n → M
    h' : φ.toFormula.Realize (Sum.elim v xs)
    ⊢ φ.Realize v xs
  -/
  simp only [BoundedFormula.realize_toFormula, Sum.elim_comp_inl, Sum.elim_comp_inr] at h'
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    n : Nat
    φ : L.BoundedFormula α n
    h : T.ModelsBoundedFormula φ
    M : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Nonempty M
    v : α → M
    xs : Fin n → M
    h' : φ.Realize v xs
    ⊢ φ.Realize v xs
  -/
  exact h'
  /-
    🎉 no goals
  -/


theorem models_of_models_theory {T' : L.Theory}
    (h : ∀ φ : L.Sentence, φ ∈ T' → T ⊨ᵇ φ)
    {φ : L.Formula α} (hφ : T' ⊨ᵇ φ) : T ⊨ᵇ φ := fun M => by
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    T' : L.Theory
    h : ∀ (φ : L.Sentence), Membership.mem T' φ → T.ModelsBoundedFormula φ
    φ : L.Formula α
    hφ : T'.ModelsBoundedFormula φ
    M : T.ModelType
    ⊢ ∀ (v : α → ↑M) (xs : Fin 0 → ↑M), FirstOrder.Language.BoundedFormula.Realize …
  -/
  have hM : M ⊨ T' := T'.model_iff.2 (fun ψ hψ => (h ψ hψ).realize_sentence M)
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    T' : L.Theory
    h : ∀ (φ : L.Sentence), Membership.mem T' φ → T.ModelsBoundedFormula φ
    φ : L.Formula α
    hφ : T'.ModelsBoundedFormula φ
    M : T.ModelType
    hM : FirstOrder.Language.Theory.Model (↑M) T'
    ⊢ ∀ (v : α → ↑M) (xs : Fin 0 → ↑M), FirstOrder.Language.BoundedFormula.Realize …
  -/
  let M' : ModelType T' := ⟨M⟩
  /-
    L : FirstOrder.Language
    T : L.Theory
    α : Type w
    T' : L.Theory
    h : ∀ (φ : L.Sentence), Membership.mem T' φ → T.ModelsBoundedFormula φ
    φ : L.Formula α
    hφ : T'.ModelsBoundedFormula φ
    M : T.ModelType
    hM : FirstOrder.Language.Theory.Model (↑M) T'
    M' : T'.ModelType := FirstOrder.Language.Theory.ModelType.mk ↑M
    ⊢ ∀ (v : α → ↑M) (xs : Fin 0 → ↑M), FirstOrder.Language.BoundedFormula.Realize …
  -/
  exact hφ M'
  /-
    🎉 no goals
  -/


/-- An alternative statement of the Compactness Theorem. A formula `φ` is modeled by a
theory iff there is a finite subset `T0` of the theory such that `φ` is modeled by `T0` -/
theorem models_iff_finset_models {φ : L.Sentence} :
    T ⊨ᵇ φ ↔ ∃ T0 : Finset L.Sentence, (T0 : L.Theory) ⊆ T ∧ (T0 : L.Theory) ⊨ᵇ φ := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    ⊢ Iff (T.ModelsBoundedFormula φ) (Exists fun T0 => And (HasSubset.Subset (↑T0) …
  -/
  simp only [models_iff_not_satisfiable]
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    ⊢ Iff (Not (Union.union T (Singleton.singleton (FirstOrder.Language.Formula.no …
  -/
  rw [← not_iff_not, not_not, isSatisfiable_iff_isFinitelySatisfiable, IsFinitelySatisfiable]
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    ⊢ Iff (∀ (T0 : Finset L.Sentence), HasSubset.Subset (↑T0) (Union.union T (Sing …
  -/
  push_neg
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    ⊢ Iff (∀ (T0 : Finset L.Sentence), HasSubset.Subset (↑T0) (Union.union T (Sing …
  -/
  letI := Classical.decEq (Sentence L)
  /-
    L : FirstOrder.Language
    T : L.Theory
    φ : L.Sentence
    this : DecidableEq L.Sentence := Classical.decEq L.Sentence
    ⊢ Iff (∀ (T0 : Finset L.Sentence), HasSubset.Subset (↑T0) (Union.union T (Sing …
  -/
  constructor
    /-
      case mp
      L : FirstOrder.Language
      T : L.Theory
      φ : L.Sentence
      this : DecidableEq L.Sentence := Classical.decEq L.Sentence
      ⊢ (∀ (T0 : Finset L.Sentence), HasSubset.Subset (↑T0) (Union.union T (Singleto …
    -/
  · intro h T0 hT0
    simpa using h (T0 ∪ {Formula.not φ})
      (by
        simp only [Finset.coe_union, Finset.coe_singleton]
        exact Set.union_subset_union hT0 (Set.Subset.refl _))
    /-
      case mpr
      L : FirstOrder.Language
      T : L.Theory
      φ : L.Sentence
      this : DecidableEq L.Sentence := Classical.decEq L.Sentence
      ⊢ (∀ (T0 : Finset L.Sentence), HasSubset.Subset (↑T0) T → (Union.union (↑T0) ( …
    -/
  · intro h T0 hT0
    exact IsSatisfiable.mono (h (T0.erase (Formula.not φ))
      (by simpa using hT0)) (by simp)


/-- A theory is complete when it is satisfiable and models each sentence or its negation. -/
def IsComplete (T : L.Theory) : Prop :=
  T.IsSatisfiable ∧ ∀ φ : L.Sentence, T ⊨ᵇ φ ∨ T ⊨ᵇ φ.not


theorem models_not_iff (h : T.IsComplete) (φ : L.Sentence) : T ⊨ᵇ φ.not ↔ ¬T ⊨ᵇ φ := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    h : T.IsComplete
    φ : L.Sentence
    ⊢ Iff (T.ModelsBoundedFormula (FirstOrder.Language.Formula.not φ)) (Not (T.Mod …
  -/
  cases' h.2 φ with hφ hφn
    /-
      case inl
      L : FirstOrder.Language
      T : L.Theory
      h : T.IsComplete
      φ : L.Sentence
      hφ : T.ModelsBoundedFormula φ
      ⊢ Iff (T.ModelsBoundedFormula (FirstOrder.Language.Formula.not φ)) (Not (T.Mod …
    -/
  · simp only [hφ, not_true, iff_false]
    /-
      case inl
      L : FirstOrder.Language
      T : L.Theory
      h : T.IsComplete
      φ : L.Sentence
      hφ : T.ModelsBoundedFormula φ
      ⊢ Not (T.ModelsBoundedFormula (FirstOrder.Language.Formula.not φ))
    -/
    rw [models_sentence_iff, not_forall]
    /-
      case inl
      L : FirstOrder.Language
      T : L.Theory
      h : T.IsComplete
      φ : L.Sentence
      hφ : T.ModelsBoundedFormula φ
      ⊢ Exists fun x => Not (FirstOrder.Language.Sentence.Realize (↑x) (FirstOrder.L …
    -/
    refine ⟨h.1.some, ?_⟩
    /-
      case inl
      L : FirstOrder.Language
      T : L.Theory
      h : T.IsComplete
      φ : L.Sentence
      hφ : T.ModelsBoundedFormula φ
      ⊢ Not (FirstOrder.Language.Sentence.Realize (↑(Nonempty.some ⋯)) (FirstOrder.L …
    -/
    simp only [Sentence.realize_not, Classical.not_not]
    /-
      case inl
      L : FirstOrder.Language
      T : L.Theory
      h : T.IsComplete
      φ : L.Sentence
      hφ : T.ModelsBoundedFormula φ
      ⊢ FirstOrder.Language.Sentence.Realize (↑(Nonempty.some ⋯)) φ
    -/
    exact models_sentence_iff.1 hφ _
    /-
      🎉 no goals
    -/
    /-
      case inr
      L : FirstOrder.Language
      T : L.Theory
      h : T.IsComplete
      φ : L.Sentence
      hφn : T.ModelsBoundedFormula (FirstOrder.Language.Formula.not φ)
      ⊢ Iff (T.ModelsBoundedFormula (FirstOrder.Language.Formula.not φ)) (Not (T.Mod …
    -/
  · simp only [hφn, true_iff]
    /-
      case inr
      L : FirstOrder.Language
      T : L.Theory
      h : T.IsComplete
      φ : L.Sentence
      hφn : T.ModelsBoundedFormula (FirstOrder.Language.Formula.not φ)
      ⊢ Not (T.ModelsBoundedFormula φ)
    -/
    intro hφ
    /-
      case inr
      L : FirstOrder.Language
      T : L.Theory
      h : T.IsComplete
      φ : L.Sentence
      hφn : T.ModelsBoundedFormula (FirstOrder.Language.Formula.not φ)
      hφ : T.ModelsBoundedFormula φ
      ⊢ False
    -/
    rw [models_sentence_iff] at *
    /-
      case inr
      L : FirstOrder.Language
      T : L.Theory
      h : T.IsComplete
      φ : L.Sentence
      hφn : ∀ (M : T.ModelType), FirstOrder.Language.Sentence.Realize (↑M) (FirstOrd …
      hφ : ∀ (M : T.ModelType), FirstOrder.Language.Sentence.Realize (↑M) φ
      ⊢ False
    -/
    exact hφn h.1.some (hφ _)
    /-
      🎉 no goals
    -/


theorem realize_sentence_iff (h : T.IsComplete) (φ : L.Sentence) (M : Type*) [L.Structure M]
    [M ⊨ T] [Nonempty M] : M ⊨ φ ↔ T ⊨ᵇ φ := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    h : T.IsComplete
    φ : L.Sentence
    M : Type u_1
    inst✝² : L.Structure M
    inst✝¹ : FirstOrder.Language.Theory.Model M T
    inst✝ : Nonempty M
    ⊢ Iff (FirstOrder.Language.Sentence.Realize M φ) (T.ModelsBoundedFormula φ)
  -/
  cases' h.2 φ with hφ hφn
    /-
      case inl
      L : FirstOrder.Language
      T : L.Theory
      h : T.IsComplete
      φ : L.Sentence
      M : Type u_1
      inst✝² : L.Structure M
      inst✝¹ : FirstOrder.Language.Theory.Model M T
      inst✝ : Nonempty M
      hφ : T.ModelsBoundedFormula φ
      ⊢ Iff (FirstOrder.Language.Sentence.Realize M φ) (T.ModelsBoundedFormula φ)
    -/
  · exact iff_of_true (hφ.realize_sentence M) hφ
    /-
      🎉 no goals
    -/
  · exact
      iff_of_false ((Sentence.realize_not M).1 (hφn.realize_sentence M))
        ((h.models_not_iff φ).1 hφn)


/-- A theory is maximal when it is satisfiable and contains each sentence or its negation.
  Maximal theories are complete. -/
def IsMaximal (T : L.Theory) : Prop :=
  T.IsSatisfiable ∧ ∀ φ : L.Sentence, φ ∈ T ∨ φ.not ∈ T


theorem IsMaximal.isComplete (h : T.IsMaximal) : T.IsComplete :=
  h.imp_right (forall_imp fun _ => Or.imp models_sentence_of_mem models_sentence_of_mem)


theorem IsMaximal.mem_or_not_mem (h : T.IsMaximal) (φ : L.Sentence) : φ ∈ T ∨ φ.not ∈ T :=
  h.2 φ


theorem IsMaximal.mem_of_models (h : T.IsMaximal) {φ : L.Sentence} (hφ : T ⊨ᵇ φ) : φ ∈ T := by
  /-
    L : FirstOrder.Language
    T : L.Theory
    h : T.IsMaximal
    φ : L.Sentence
    hφ : T.ModelsBoundedFormula φ
    ⊢ Membership.mem T φ
  -/
  refine (h.mem_or_not_mem φ).resolve_right fun con => ?_
  /-
    L : FirstOrder.Language
    T : L.Theory
    h : T.IsMaximal
    φ : L.Sentence
    hφ : T.ModelsBoundedFormula φ
    con : Membership.mem T (FirstOrder.Language.Formula.not φ)
    ⊢ False
  -/
  rw [models_iff_not_satisfiable, Set.union_singleton, Set.insert_eq_of_mem con] at hφ
  /-
    L : FirstOrder.Language
    T : L.Theory
    h : T.IsMaximal
    φ : L.Sentence
    hφ : Not T.IsSatisfiable
    con : Membership.mem T (FirstOrder.Language.Formula.not φ)
    ⊢ False
  -/
  exact hφ h.1
  /-
    🎉 no goals
  -/


theorem IsMaximal.mem_iff_models (h : T.IsMaximal) (φ : L.Sentence) : φ ∈ T ↔ T ⊨ᵇ φ :=
  ⟨models_sentence_of_mem, h.mem_of_models⟩


theorem isSatisfiable [Nonempty M] : (L.completeTheory M).IsSatisfiable :=
  Theory.Model.isSatisfiable M


theorem mem_or_not_mem (φ : L.Sentence) : φ ∈ L.completeTheory M ∨ φ.not ∈ L.completeTheory M := by
  /-
    L : FirstOrder.Language
    M : Type w
    inst✝ : L.Structure M
    φ : L.Sentence
    ⊢ Or (Membership.mem (L.completeTheory M) φ) (Membership.mem (L.completeTheory …
  -/
  simp_rw [completeTheory, Set.mem_setOf_eq, Sentence.Realize, Formula.realize_not, or_not]
  /-
    🎉 no goals
  -/


theorem isMaximal [Nonempty M] : (L.completeTheory M).IsMaximal :=
  ⟨isSatisfiable L M, mem_or_not_mem L M⟩


theorem isComplete [Nonempty M] : (L.completeTheory M).IsComplete :=
  (completeTheory.isMaximal L M).isComplete


/-- A theory is `κ`-categorical if all models of size `κ` are isomorphic. -/
def Categorical : Prop :=
  ∀ M N : T.ModelType, #M = κ → #N = κ → Nonempty (M ≃[L] N)


/-- The Łoś–Vaught Test : a criterion for categorical theories to be complete. -/
theorem Categorical.isComplete (h : κ.Categorical T) (h1 : ℵ₀ ≤ κ)
    (h2 : Cardinal.lift.{w} L.card ≤ Cardinal.lift.{max u v} κ) (hS : T.IsSatisfiable)
    (hT : ∀ M : Theory.ModelType.{u, v, max u v} T, Infinite M) : T.IsComplete :=
  ⟨hS, fun φ => by
    /-
      L : FirstOrder.Language
      κ : Cardinal.{w}
      T : L.Theory
      h : κ.Categorical T
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      hS : T.IsSatisfiable
      hT : ∀ (M : T.ModelType), Infinite ↑M
      φ : L.Sentence
      ⊢ Or (T.ModelsBoundedFormula φ) (T.ModelsBoundedFormula (FirstOrder.Language.F …
    -/
    obtain ⟨_, _⟩ := Theory.exists_model_card_eq ⟨hS.some, hT hS.some⟩ κ h1 h2
    /-
      case intro
      L : FirstOrder.Language
      κ : Cardinal.{w}
      T : L.Theory
      h : κ.Categorical T
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      hS : T.IsSatisfiable
      hT : ∀ (M : T.ModelType), Infinite ↑M
      φ : L.Sentence
      w✝ : T.ModelType
      h✝ : Eq (Cardinal.mk ↑w✝) κ
      ⊢ Or (T.ModelsBoundedFormula φ) (T.ModelsBoundedFormula (FirstOrder.Language.F …
    -/
    rw [Theory.models_sentence_iff, Theory.models_sentence_iff]
    /-
      case intro
      L : FirstOrder.Language
      κ : Cardinal.{w}
      T : L.Theory
      h : κ.Categorical T
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      hS : T.IsSatisfiable
      hT : ∀ (M : T.ModelType), Infinite ↑M
      φ : L.Sentence
      w✝ : T.ModelType
      h✝ : Eq (Cardinal.mk ↑w✝) κ
      ⊢ Or (∀ (M : T.ModelType), FirstOrder.Language.Sentence.Realize (↑M) φ) (∀ (M  …
    -/
    by_contra! con
    /-
      case intro
      L : FirstOrder.Language
      κ : Cardinal.{w}
      T : L.Theory
      h : κ.Categorical T
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      hS : T.IsSatisfiable
      hT : ∀ (M : T.ModelType), Infinite ↑M
      φ : L.Sentence
      w✝ : T.ModelType
      h✝ : Eq (Cardinal.mk ↑w✝) κ
      con : And (Exists fun M => Not (FirstOrder.Language.Sentence.Realize (↑M) φ))  …
      ⊢ False
    -/
    obtain ⟨⟨MF, hMF⟩, MT, hMT⟩ := con
    /-
      case intro.intro.intro.intro
      L : FirstOrder.Language
      κ : Cardinal.{w}
      T : L.Theory
      h : κ.Categorical T
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      hS : T.IsSatisfiable
      hT : ∀ (M : T.ModelType), Infinite ↑M
      φ : L.Sentence
      w✝ : T.ModelType
      h✝ : Eq (Cardinal.mk ↑w✝) κ
      MF : T.ModelType
      hMF : Not (FirstOrder.Language.Sentence.Realize (↑MF) φ)
      MT : T.ModelType
      hMT : Not (FirstOrder.Language.Sentence.Realize (↑MT) (FirstOrder.Language.For …
      ⊢ False
    -/
    rw [Sentence.realize_not, Classical.not_not] at hMT
    /-
      case intro.intro.intro.intro
      L : FirstOrder.Language
      κ : Cardinal.{w}
      T : L.Theory
      h : κ.Categorical T
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      hS : T.IsSatisfiable
      hT : ∀ (M : T.ModelType), Infinite ↑M
      φ : L.Sentence
      w✝ : T.ModelType
      h✝ : Eq (Cardinal.mk ↑w✝) κ
      MF : T.ModelType
      hMF : Not (FirstOrder.Language.Sentence.Realize (↑MF) φ)
      MT : T.ModelType
      hMT : FirstOrder.Language.Sentence.Realize (↑MT) φ
      ⊢ False
    -/
    refine hMF ?_
    /-
      case intro.intro.intro.intro
      L : FirstOrder.Language
      κ : Cardinal.{w}
      T : L.Theory
      h : κ.Categorical T
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      hS : T.IsSatisfiable
      hT : ∀ (M : T.ModelType), Infinite ↑M
      φ : L.Sentence
      w✝ : T.ModelType
      h✝ : Eq (Cardinal.mk ↑w✝) κ
      MF : T.ModelType
      hMF : Not (FirstOrder.Language.Sentence.Realize (↑MF) φ)
      MT : T.ModelType
      hMT : FirstOrder.Language.Sentence.Realize (↑MT) φ
      ⊢ FirstOrder.Language.Sentence.Realize (↑MF) φ
    -/
    haveI := hT MT
    /-
      case intro.intro.intro.intro
      L : FirstOrder.Language
      κ : Cardinal.{w}
      T : L.Theory
      h : κ.Categorical T
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      hS : T.IsSatisfiable
      hT : ∀ (M : T.ModelType), Infinite ↑M
      φ : L.Sentence
      w✝ : T.ModelType
      h✝ : Eq (Cardinal.mk ↑w✝) κ
      MF : T.ModelType
      hMF : Not (FirstOrder.Language.Sentence.Realize (↑MF) φ)
      MT : T.ModelType
      hMT : FirstOrder.Language.Sentence.Realize (↑MT) φ
      this : Infinite ↑MT
      ⊢ FirstOrder.Language.Sentence.Realize (↑MF) φ
    -/
    haveI := hT MF
    /-
      case intro.intro.intro.intro
      L : FirstOrder.Language
      κ : Cardinal.{w}
      T : L.Theory
      h : κ.Categorical T
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      hS : T.IsSatisfiable
      hT : ∀ (M : T.ModelType), Infinite ↑M
      φ : L.Sentence
      w✝ : T.ModelType
      h✝ : Eq (Cardinal.mk ↑w✝) κ
      MF : T.ModelType
      hMF : Not (FirstOrder.Language.Sentence.Realize (↑MF) φ)
      MT : T.ModelType
      hMT : FirstOrder.Language.Sentence.Realize (↑MT) φ
      this✝ : Infinite ↑MT
      this : Infinite ↑MF
      ⊢ FirstOrder.Language.Sentence.Realize (↑MF) φ
    -/
    obtain ⟨NT, MNT, hNT⟩ := exists_elementarilyEquivalent_card_eq L MT κ h1 h2
    /-
      case intro.intro.intro.intro.intro.intro
      L : FirstOrder.Language
      κ : Cardinal.{w}
      T : L.Theory
      h : κ.Categorical T
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      hS : T.IsSatisfiable
      hT : ∀ (M : T.ModelType), Infinite ↑M
      φ : L.Sentence
      w✝ : T.ModelType
      h✝ : Eq (Cardinal.mk ↑w✝) κ
      MF : T.ModelType
      hMF : Not (FirstOrder.Language.Sentence.Realize (↑MF) φ)
      MT : T.ModelType
      hMT : FirstOrder.Language.Sentence.Realize (↑MT) φ
      this✝ : Infinite ↑MT
      this : Infinite ↑MF
      NT : CategoryTheory.Bundled L.Structure
      MNT : L.ElementarilyEquivalent ↑MT ↑NT
      hNT : Eq (Cardinal.mk ↑NT) κ
      ⊢ FirstOrder.Language.Sentence.Realize (↑MF) φ
    -/
    obtain ⟨NF, MNF, hNF⟩ := exists_elementarilyEquivalent_card_eq L MF κ h1 h2
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro
      L : FirstOrder.Language
      κ : Cardinal.{w}
      T : L.Theory
      h : κ.Categorical T
      h1 : LE.le Cardinal.aleph0 κ
      h2 : LE.le (Cardinal.lift.{w, max u v} L.card) (Cardinal.lift.{max u v, w} κ)
      hS : T.IsSatisfiable
      hT : ∀ (M : T.ModelType), Infinite ↑M
      φ : L.Sentence
      w✝ : T.ModelType
      h✝ : Eq (Cardinal.mk ↑w✝) κ
      MF : T.ModelType
      hMF : Not (FirstOrder.Language.Sentence.Realize (↑MF) φ)
      MT : T.ModelType
      hMT : FirstOrder.Language.Sentence.Realize (↑MT) φ
      this✝ : Infinite ↑MT
      this : Infinite ↑MF
      NT : CategoryTheory.Bundled L.Structure
      MNT : L.ElementarilyEquivalent ↑MT ↑NT
      hNT : Eq (Cardinal.mk ↑NT) κ
      NF : CategoryTheory.Bundled L.Structure
      MNF : L.ElementarilyEquivalent ↑MF ↑NF
      hNF : Eq (Cardinal.mk ↑NF) κ
      ⊢ FirstOrder.Language.Sentence.Realize (↑MF) φ
    -/
    obtain ⟨TF⟩ := h (MNT.toModel T) (MNF.toModel T) hNT hNF
    exact
      ((MNT.realize_sentence φ).trans
        ((StrongHomClass.realize_sentence TF φ).trans (MNF.realize_sentence φ).symm)).1 hMT⟩


theorem empty_theory_categorical (T : Language.empty.Theory) : κ.Categorical T := fun M N hM hN =>
     /-
       κ : Cardinal.{w}
       T : FirstOrder.Language.empty.Theory
       M N : T.ModelType
       hM : Eq (Cardinal.mk ↑M) κ
       hN : Eq (Cardinal.mk ↑N) κ
       ⊢ Nonempty (FirstOrder.Language.empty.Equiv ↑M ↑N)
     -/
  by rw [empty.nonempty_equiv_iff, hM, hN]
     /-
       🎉 no goals
     -/


theorem empty_infinite_Theory_isComplete : Language.empty.infiniteTheory.IsComplete :=
                                                                 /-
                                                                   ⊢ LE.le (Cardinal.lift.{0, 0} FirstOrder.Language.empty.card) (Cardinal.lift.{ …
                                                                 -/
  (empty_theory_categorical.{0} ℵ₀ _).isComplete ℵ₀ _ le_rfl (by simp)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    ⟨by
      /-
        ⊢ FirstOrder.Language.empty.infiniteTheory.ModelType
      -/
      haveI : Language.empty.Structure ℕ := emptyStructure
      /-
        this : FirstOrder.Language.empty.Structure Nat
        ⊢ FirstOrder.Language.empty.infiniteTheory.ModelType
      -/
      exact ((model_infiniteTheory_iff Language.empty).2 (inferInstanceAs (Infinite ℕ))).bundled⟩
      /-
        🎉 no goals
      -/
    fun M => (model_infiniteTheory_iff Language.empty).1 M.is_model


