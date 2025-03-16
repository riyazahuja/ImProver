/-- Given a finite set `s` of indices, a square cylinder is the product of a set `S` of
`∀ i : s, α i` and of `univ` on the other indices. The set `S` is a product of sets `t i` such that
for all `i : s`, `t i ∈ C i`.
`squareCylinders` is the set of all such squareCylinders. -/
def squareCylinders (C : ∀ i, Set (Set (α i))) : Set (Set (∀ i, α i)) :=
  {S | ∃ s : Finset ι, ∃ t ∈ univ.pi C, S = (s : Set ι).pi t}


theorem squareCylinders_eq_iUnion_image (C : ∀ i, Set (Set (α i))) :
    squareCylinders C = ⋃ s : Finset ι, (fun t ↦ (s : Set ι).pi t) '' univ.pi C := by
  /-
    ι : Type u_2
    α : ι → Type u_1
    C : (i : ι) → Set (Set (α i))
    ⊢ Eq (MeasureTheory.squareCylinders C) (Set.iUnion fun s => Set.image (fun t = …
  -/
  ext1 f
  simp only [squareCylinders, mem_iUnion, mem_image, mem_univ_pi, exists_prop, mem_setOf_eq,
    eq_comm (a := f)]


theorem isPiSystem_squareCylinders {C : ∀ i, Set (Set (α i))} (hC : ∀ i, IsPiSystem (C i))
    (hC_univ : ∀ i, univ ∈ C i) :
    IsPiSystem (squareCylinders C) := by
  /-
    ι : Type u_2
    α : ι → Type u_1
    C : (i : ι) → Set (Set (α i))
    hC : ∀ (i : ι), IsPiSystem (C i)
    hC_univ : ∀ (i : ι), Membership.mem (C i) Set.univ
    ⊢ IsPiSystem (MeasureTheory.squareCylinders C)
  -/
  rintro S₁ ⟨s₁, t₁, h₁, rfl⟩ S₂ ⟨s₂, t₂, h₂, rfl⟩ hst_nonempty
  classical
  let t₁' := s₁.piecewise t₁ (fun i ↦ univ)
  let t₂' := s₂.piecewise t₂ (fun i ↦ univ)
  have h1 : ∀ i ∈ (s₁ : Set ι), t₁ i = t₁' i :=
    fun i hi ↦ (Finset.piecewise_eq_of_mem _ _ _ hi).symm
  have h1' : ∀ i ∉ (s₁ : Set ι), t₁' i = univ :=
    fun i hi ↦ Finset.piecewise_eq_of_not_mem _ _ _ hi
  have h2 : ∀ i ∈ (s₂ : Set ι), t₂ i = t₂' i :=
    fun i hi ↦ (Finset.piecewise_eq_of_mem _ _ _ hi).symm
  have h2' : ∀ i ∉ (s₂ : Set ι), t₂' i = univ :=
    fun i hi ↦ Finset.piecewise_eq_of_not_mem _ _ _ hi
  rw [Set.pi_congr rfl h1, Set.pi_congr rfl h2, ← union_pi_inter h1' h2']
  refine ⟨s₁ ∪ s₂, fun i ↦ t₁' i ∩ t₂' i, ?_, ?_⟩
  · rw [mem_univ_pi]
    intro i
    have : (t₁' i ∩ t₂' i).Nonempty := by
      obtain ⟨f, hf⟩ := hst_nonempty
      rw [Set.pi_congr rfl h1, Set.pi_congr rfl h2, mem_inter_iff, mem_pi, mem_pi] at hf
      refine ⟨f i, ⟨?_, ?_⟩⟩
      · by_cases hi₁ : i ∈ s₁
        · exact hf.1 i hi₁
        · rw [h1' i hi₁]
          exact mem_univ _
      · by_cases hi₂ : i ∈ s₂
        · exact hf.2 i hi₂
        · rw [h2' i hi₂]
          exact mem_univ _
    refine hC i _ ?_ _ ?_ this
    · by_cases hi₁ : i ∈ s₁
      · rw [← h1 i hi₁]
        exact h₁ i (mem_univ _)
      · rw [h1' i hi₁]
        exact hC_univ i
    · by_cases hi₂ : i ∈ s₂
      · rw [← h2 i hi₂]
        exact h₂ i (mem_univ _)
      · rw [h2' i hi₂]
        exact hC_univ i
  · rw [Finset.coe_union]


theorem comap_eval_le_generateFrom_squareCylinders_singleton
    (α : ι → Type*) [m : ∀ i, MeasurableSpace (α i)] (i : ι) :
    MeasurableSpace.comap (Function.eval i) (m i) ≤
      MeasurableSpace.generateFrom
        ((fun t ↦ ({i} : Set ι).pi t) '' univ.pi fun i ↦ {s : Set (α i) | MeasurableSet s}) := by
  /-
    ι : Type u_2
    α : ι → Type u_1
    m : (i : ι) → MeasurableSpace (α i)
    i : ι
    ⊢ LE.le (MeasurableSpace.comap (Function.eval i) (m i)) (MeasurableSpace.gener …
  -/
  simp only [Function.eval, singleton_pi]
  /-
    ι : Type u_2
    α : ι → Type u_1
    m : (i : ι) → MeasurableSpace (α i)
    i : ι
    ⊢ LE.le (MeasurableSpace.comap (Function.eval i) (m i)) (MeasurableSpace.gener …
  -/
  rw [MeasurableSpace.comap_eq_generateFrom]
  /-
    ι : Type u_2
    α : ι → Type u_1
    m : (i : ι) → MeasurableSpace (α i)
    i : ι
    ⊢ LE.le (MeasurableSpace.generateFrom (setOf fun t => Exists fun s => And (Mea …
  -/
  refine MeasurableSpace.generateFrom_mono fun S ↦ ?_
  /-
    ι : Type u_2
    α : ι → Type u_1
    m : (i : ι) → MeasurableSpace (α i)
    i : ι
    S : Set ((x : ι) → α x)
    ⊢ Membership.mem (setOf fun t => Exists fun s => And (MeasurableSet s) (Eq (Se …
  -/
  simp only [mem_setOf_eq, mem_image, mem_univ_pi, forall_exists_index, and_imp]
  /-
    ι : Type u_2
    α : ι → Type u_1
    m : (i : ι) → MeasurableSpace (α i)
    i : ι
    S : Set ((x : ι) → α x)
    ⊢ ∀ (x : Set (α i)), MeasurableSet x → Eq (Set.preimage (Function.eval i) x) S …
  -/
  intro t ht h
  classical
  refine ⟨fun j ↦ if hji : j = i then by convert t else univ, fun j ↦ ?_, ?_⟩
  · by_cases hji : j = i
    · simp only [hji, eq_self_iff_true, eq_mpr_eq_cast, dif_pos]
      convert ht
      simp only [id_eq, cast_heq]
    · simp only [hji, not_false_iff, dif_neg, MeasurableSet.univ]
  · simp only [id_eq, eq_mpr_eq_cast, ← h]
    ext1 x
    simp only [singleton_pi, Function.eval, cast_eq, dite_eq_ite, ite_true, mem_preimage]


/-- The square cylinders formed from measurable sets generate the product σ-algebra. -/
theorem generateFrom_squareCylinders [∀ i, MeasurableSpace (α i)] :
    MeasurableSpace.generateFrom (squareCylinders fun i ↦ {s : Set (α i) | MeasurableSet s}) =
      MeasurableSpace.pi := by
  /-
    ι : Type u_2
    α : ι → Type u_1
    inst✝ : (i : ι) → MeasurableSpace (α i)
    ⊢ Eq (MeasurableSpace.generateFrom (MeasureTheory.squareCylinders fun i => set …
  -/
  apply le_antisymm
    /-
      case a
      ι : Type u_2
      α : ι → Type u_1
      inst✝ : (i : ι) → MeasurableSpace (α i)
      ⊢ LE.le (MeasurableSpace.generateFrom (MeasureTheory.squareCylinders fun i =>  …
    -/
  · rw [MeasurableSpace.generateFrom_le_iff]
    /-
      case a
      ι : Type u_2
      α : ι → Type u_1
      inst✝ : (i : ι) → MeasurableSpace (α i)
      ⊢ HasSubset.Subset (MeasureTheory.squareCylinders fun i => setOf fun s => Meas …
    -/
    rintro S ⟨s, t, h, rfl⟩
    /-
      case a.intro.intro.intro
      ι : Type u_2
      α : ι → Type u_1
      inst✝ : (i : ι) → MeasurableSpace (α i)
      s : Finset ι
      t : (i : ι) → Set (α i)
      h : Membership.mem (Set.univ.pi fun i => setOf fun s => MeasurableSet s) t
      ⊢ Membership.mem (setOf fun t => MeasurableSet t) ((↑s).pi t)
    -/
    simp only [mem_univ_pi, mem_setOf_eq] at h
    /-
      case a.intro.intro.intro
      ι : Type u_2
      α : ι → Type u_1
      inst✝ : (i : ι) → MeasurableSpace (α i)
      s : Finset ι
      t : (i : ι) → Set (α i)
      h : ∀ (i : ι), MeasurableSet (t i)
      ⊢ Membership.mem (setOf fun t => MeasurableSet t) ((↑s).pi t)
    -/
    exact MeasurableSet.pi (Finset.countable_toSet _) (fun i _ ↦ h i)
    /-
      🎉 no goals
    -/
    /-
      case a
      ι : Type u_2
      α : ι → Type u_1
      inst✝ : (i : ι) → MeasurableSpace (α i)
      ⊢ LE.le MeasurableSpace.pi (MeasurableSpace.generateFrom (MeasureTheory.square …
    -/
  · refine iSup_le fun i ↦ ?_
    /-
      case a
      ι : Type u_2
      α : ι → Type u_1
      inst✝ : (i : ι) → MeasurableSpace (α i)
      i : ι
      ⊢ LE.le (MeasurableSpace.comap (fun b => b i) (inst✝ i)) (MeasurableSpace.gene …
    -/
    refine (comap_eval_le_generateFrom_squareCylinders_singleton α i).trans ?_
    /-
      case a
      ι : Type u_2
      α : ι → Type u_1
      inst✝ : (i : ι) → MeasurableSpace (α i)
      i : ι
      ⊢ LE.le (MeasurableSpace.generateFrom (Set.image (fun t => (Singleton.singleto …
    -/
    refine MeasurableSpace.generateFrom_mono ?_
    /-
      case a
      ι : Type u_2
      α : ι → Type u_1
      inst✝ : (i : ι) → MeasurableSpace (α i)
      i : ι
      ⊢ HasSubset.Subset (Set.image (fun t => (Singleton.singleton i).pi t) (Set.uni …
    -/
    rw [← Finset.coe_singleton, squareCylinders_eq_iUnion_image]
    exact subset_iUnion
      (fun (s : Finset ι) ↦
        (fun t : ∀ i, Set (α i) ↦ (s : Set ι).pi t) '' univ.pi (fun i ↦ setOf MeasurableSet))
      ({i} : Finset ι)


/-- Given a finite set `s` of indices, a cylinder is the preimage of a set `S` of `∀ i : s, α i` by
the projection from `∀ i, α i` to `∀ i : s, α i`. -/
def cylinder (s : Finset ι) (S : Set (∀ i : s, α i)) : Set (∀ i, α i) :=
  s.restrict ⁻¹' S


@[simp]
theorem mem_cylinder (s : Finset ι) (S : Set (∀ i : s, α i)) (f : ∀ i, α i) :
    f ∈ cylinder s S ↔ s.restrict f ∈ S :=
  mem_preimage


@[simp]
theorem cylinder_empty (s : Finset ι) : cylinder s (∅ : Set (∀ i : s, α i)) = ∅ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    ⊢ Eq (MeasureTheory.cylinder s EmptyCollection.emptyCollection) EmptyCollectio …
  -/
  rw [cylinder, preimage_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem cylinder_univ (s : Finset ι) : cylinder s (univ : Set (∀ i : s, α i)) = univ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    ⊢ Eq (MeasureTheory.cylinder s Set.univ) Set.univ
  -/
  rw [cylinder, preimage_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem cylinder_eq_empty_iff [h_nonempty : Nonempty (∀ i, α i)] (s : Finset ι)
    (S : Set (∀ i : s, α i)) :
    cylinder s S = ∅ ↔ S = ∅ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    h_nonempty : Nonempty ((i : ι) → α i)
    s : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    ⊢ Iff (Eq (MeasureTheory.cylinder s S) EmptyCollection.emptyCollection) (Eq S  …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ by (rw [h]; exact cylinder_empty _)⟩
  /-
    ι : Type u_1
    α : ι → Type u_2
    h_nonempty : Nonempty ((i : ι) → α i)
    s : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    h : Eq (MeasureTheory.cylinder s S) EmptyCollection.emptyCollection
    ⊢ Eq S EmptyCollection.emptyCollection
  -/
  by_contra hS
  /-
    ι : Type u_1
    α : ι → Type u_2
    h_nonempty : Nonempty ((i : ι) → α i)
    s : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    h : Eq (MeasureTheory.cylinder s S) EmptyCollection.emptyCollection
    hS : Not (Eq S EmptyCollection.emptyCollection)
    ⊢ False
  -/
  rw [← Ne, ← nonempty_iff_ne_empty] at hS
  /-
    ι : Type u_1
    α : ι → Type u_2
    h_nonempty : Nonempty ((i : ι) → α i)
    s : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    h : Eq (MeasureTheory.cylinder s S) EmptyCollection.emptyCollection
    hS : S.Nonempty
    ⊢ False
  -/
  let f := hS.some
  /-
    ι : Type u_1
    α : ι → Type u_2
    h_nonempty : Nonempty ((i : ι) → α i)
    s : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    h : Eq (MeasureTheory.cylinder s S) EmptyCollection.emptyCollection
    hS : S.Nonempty
    f : (i : Subtype fun x => Membership.mem s x) → α ↑i := hS.some
    ⊢ False
  -/
  have hf : f ∈ S := hS.choose_spec
  classical
  let f' : ∀ i, α i := fun i ↦ if hi : i ∈ s then f ⟨i, hi⟩ else h_nonempty.some i
  have hf' : f' ∈ cylinder s S := by
    rw [mem_cylinder]
    simpa only [Finset.restrict_def, Finset.coe_mem, dif_pos, f']
  rw [h] at hf'
  exact not_mem_empty _ hf'


theorem inter_cylinder (s₁ s₂ : Finset ι) (S₁ : Set (∀ i : s₁, α i)) (S₂ : Set (∀ i : s₂, α i))
    [DecidableEq ι] :
    cylinder s₁ S₁ ∩ cylinder s₂ S₂ =
      cylinder (s₁ ∪ s₂)
        (Finset.restrict₂ Finset.subset_union_left ⁻¹' S₁ ∩
          Finset.restrict₂ Finset.subset_union_right ⁻¹' S₂) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s₁ s₂ : Finset ι
    S₁ : Set ((i : Subtype fun x => Membership.mem s₁ x) → α ↑i)
    S₂ : Set ((i : Subtype fun x => Membership.mem s₂ x) → α ↑i)
    inst✝ : DecidableEq ι
    ⊢ Eq (Inter.inter (MeasureTheory.cylinder s₁ S₁) (MeasureTheory.cylinder s₂ S₂ …
  -/
  ext1 f; simp only [mem_inter_iff, mem_cylinder, mem_setOf_eq]; rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem inter_cylinder_same (s : Finset ι) (S₁ : Set (∀ i : s, α i)) (S₂ : Set (∀ i : s, α i)) :
    cylinder s S₁ ∩ cylinder s S₂ = cylinder s (S₁ ∩ S₂) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    S₁ S₂ : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    ⊢ Eq (Inter.inter (MeasureTheory.cylinder s S₁) (MeasureTheory.cylinder s S₂)) …
  -/
  classical rw [inter_cylinder]; rfl
  /-
    🎉 no goals
  -/


theorem union_cylinder (s₁ s₂ : Finset ι) (S₁ : Set (∀ i : s₁, α i)) (S₂ : Set (∀ i : s₂, α i))
    [DecidableEq ι] :
    cylinder s₁ S₁ ∪ cylinder s₂ S₂ =
      cylinder (s₁ ∪ s₂)
        (Finset.restrict₂ Finset.subset_union_left ⁻¹' S₁ ∪
          Finset.restrict₂ Finset.subset_union_right ⁻¹' S₂) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s₁ s₂ : Finset ι
    S₁ : Set ((i : Subtype fun x => Membership.mem s₁ x) → α ↑i)
    S₂ : Set ((i : Subtype fun x => Membership.mem s₂ x) → α ↑i)
    inst✝ : DecidableEq ι
    ⊢ Eq (Union.union (MeasureTheory.cylinder s₁ S₁) (MeasureTheory.cylinder s₂ S₂ …
  -/
  ext1 f; simp only [mem_union, mem_cylinder, mem_setOf_eq]; rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem union_cylinder_same (s : Finset ι) (S₁ : Set (∀ i : s, α i)) (S₂ : Set (∀ i : s, α i)) :
    cylinder s S₁ ∪ cylinder s S₂ = cylinder s (S₁ ∪ S₂) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    S₁ S₂ : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    ⊢ Eq (Union.union (MeasureTheory.cylinder s S₁) (MeasureTheory.cylinder s S₂)) …
  -/
  classical rw [union_cylinder]; rfl
  /-
    🎉 no goals
  -/


theorem compl_cylinder (s : Finset ι) (S : Set (∀ i : s, α i)) :
    (cylinder s S)ᶜ = cylinder s (Sᶜ) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    ⊢ Eq (HasCompl.compl (MeasureTheory.cylinder s S)) (MeasureTheory.cylinder s ( …
  -/
  ext1 f; simp only [mem_compl_iff, mem_cylinder]
          /-
            🎉 no goals
          -/


theorem diff_cylinder_same (s : Finset ι) (S T : Set (∀ i : s, α i)) :
    cylinder s S \ cylinder s T = cylinder s (S \ T) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Finset ι
    S T : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    ⊢ Eq (SDiff.sdiff (MeasureTheory.cylinder s S) (MeasureTheory.cylinder s T)) ( …
  -/
  ext1 f; simp only [mem_diff, mem_cylinder]
          /-
            🎉 no goals
          -/


theorem eq_of_cylinder_eq_of_subset [h_nonempty : Nonempty (∀ i, α i)] {I J : Finset ι}
    {S : Set (∀ i : I, α i)} {T : Set (∀ i : J, α i)} (h_eq : cylinder I S = cylinder J T)
    (hJI : J ⊆ I) :
    S = Finset.restrict₂ hJI ⁻¹' T := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    h_nonempty : Nonempty ((i : ι) → α i)
    I J : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem I x) → α ↑i)
    T : Set ((i : Subtype fun x => Membership.mem J x) → α ↑i)
    h_eq : Eq (MeasureTheory.cylinder I S) (MeasureTheory.cylinder J T)
    hJI : HasSubset.Subset J I
    ⊢ Eq S (Set.preimage (Finset.restrict₂ hJI) T)
  -/
  rw [Set.ext_iff] at h_eq
  /-
    ι : Type u_1
    α : ι → Type u_2
    h_nonempty : Nonempty ((i : ι) → α i)
    I J : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem I x) → α ↑i)
    T : Set ((i : Subtype fun x => Membership.mem J x) → α ↑i)
    h_eq : ∀ (x : (i : ι) → α i), Iff (Membership.mem (MeasureTheory.cylinder I S) …
    hJI : HasSubset.Subset J I
    ⊢ Eq S (Set.preimage (Finset.restrict₂ hJI) T)
  -/
  simp only [mem_cylinder] at h_eq
  /-
    ι : Type u_1
    α : ι → Type u_2
    h_nonempty : Nonempty ((i : ι) → α i)
    I J : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem I x) → α ↑i)
    T : Set ((i : Subtype fun x => Membership.mem J x) → α ↑i)
    hJI : HasSubset.Subset J I
    h_eq : ∀ (x : (i : ι) → α i), Iff (Membership.mem S (I.restrict x)) (Membershi …
    ⊢ Eq S (Set.preimage (Finset.restrict₂ hJI) T)
  -/
  ext1 f
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    h_nonempty : Nonempty ((i : ι) → α i)
    I J : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem I x) → α ↑i)
    T : Set ((i : Subtype fun x => Membership.mem J x) → α ↑i)
    hJI : HasSubset.Subset J I
    h_eq : ∀ (x : (i : ι) → α i), Iff (Membership.mem S (I.restrict x)) (Membershi …
    f : (i : Subtype fun x => Membership.mem I x) → α ↑i
    ⊢ Iff (Membership.mem S f) (Membership.mem (Set.preimage (Finset.restrict₂ hJI …
  -/
  simp only [mem_preimage]
  classical
  specialize h_eq fun i ↦ if hi : i ∈ I then f ⟨i, hi⟩ else h_nonempty.some i
  have h_mem : ∀ j : J, ↑j ∈ I := fun j ↦ hJI j.prop
  simpa only [Finset.restrict_def, Finset.coe_mem, dite_true, h_mem] using h_eq


theorem cylinder_eq_cylinder_union [DecidableEq ι] (I : Finset ι) (S : Set (∀ i : I, α i))
    (J : Finset ι) :
    cylinder I S =
      cylinder (I ∪ J) (Finset.restrict₂ Finset.subset_union_left ⁻¹' S) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : DecidableEq ι
    I : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem I x) → α ↑i)
    J : Finset ι
    ⊢ Eq (MeasureTheory.cylinder I S) (MeasureTheory.cylinder (Union.union I J) (S …
  -/
  ext1 f; simp only [mem_cylinder, Finset.restrict_def, Finset.restrict₂_def, mem_preimage]
          /-
            🎉 no goals
          -/


theorem disjoint_cylinder_iff [Nonempty (∀ i, α i)] {s t : Finset ι} {S : Set (∀ i : s, α i)}
    {T : Set (∀ i : t, α i)} [DecidableEq ι] :
    Disjoint (cylinder s S) (cylinder t T) ↔
      Disjoint
        (Finset.restrict₂ Finset.subset_union_left ⁻¹' S)
        (Finset.restrict₂ Finset.subset_union_right ⁻¹' T) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : Nonempty ((i : ι) → α i)
    s t : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    T : Set ((i : Subtype fun x => Membership.mem t x) → α ↑i)
    inst✝ : DecidableEq ι
    ⊢ Iff (Disjoint (MeasureTheory.cylinder s S) (MeasureTheory.cylinder t T)) (Di …
  -/
  simp_rw [Set.disjoint_iff, subset_empty_iff, inter_cylinder, cylinder_eq_empty_iff]
  /-
    🎉 no goals
  -/


theorem IsClosed.cylinder [∀ i, TopologicalSpace (α i)] (s : Finset ι) {S : Set (∀ i : s, α i)}
    (hs : IsClosed S) : IsClosed (cylinder s S) :=
  hs.preimage (continuous_pi fun _ ↦ continuous_apply _)


theorem _root_.MeasurableSet.cylinder [∀ i, MeasurableSpace (α i)] (s : Finset ι)
    {S : Set (∀ i : s, α i)} (hS : MeasurableSet S) :
    MeasurableSet (cylinder s S) :=
  measurable_pi_lambda _ (fun _ ↦ measurable_pi_apply _) hS


/-- Given a finite set `s` of indices, a cylinder is the preimage of a set `S` of `∀ i : s, α i` by
the projection from `∀ i, α i` to `∀ i : s, α i`.
`measurableCylinders` is the set of all cylinders with measurable base `S`. -/
def measurableCylinders (α : ι → Type*) [∀ i, MeasurableSpace (α i)] : Set (Set (∀ i, α i)) :=
  ⋃ (s) (S) (_ : MeasurableSet S), {cylinder s S}


theorem empty_mem_measurableCylinders (α : ι → Type*) [∀ i, MeasurableSpace (α i)] :
    ∅ ∈ measurableCylinders α := by
  /-
    ι : Type u_2
    α : ι → Type u_1
    inst✝ : (i : ι) → MeasurableSpace (α i)
    ⊢ Membership.mem (MeasureTheory.measurableCylinders α) EmptyCollection.emptyCo …
  -/
  simp_rw [measurableCylinders, mem_iUnion, mem_singleton_iff]
  /-
    ι : Type u_2
    α : ι → Type u_1
    inst✝ : (i : ι) → MeasurableSpace (α i)
    ⊢ Exists fun i => Exists fun i_1 => Exists fun h => Eq EmptyCollection.emptyCo …
  -/
  exact ⟨∅, ∅, MeasurableSet.empty, (cylinder_empty _).symm⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_measurableCylinders (t : Set (∀ i, α i)) :
    t ∈ measurableCylinders α ↔ ∃ s S, MeasurableSet S ∧ t = cylinder s S := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    t : Set ((i : ι) → α i)
    ⊢ Iff (Membership.mem (MeasureTheory.measurableCylinders α) t) (Exists fun s = …
  -/
  simp_rw [measurableCylinders, mem_iUnion, exists_prop, mem_singleton_iff]
  /-
    🎉 no goals
  -/


@[measurability]
theorem _root_.MeasurableSet.of_mem_measurableCylinders {s : Set (Π i, α i)}
    (hs : s ∈ measurableCylinders α) : MeasurableSet s := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    s : Set ((i : ι) → α i)
    hs : Membership.mem (MeasureTheory.measurableCylinders α) s
    ⊢ MeasurableSet s
  -/
  obtain ⟨I, t, mt, rfl⟩ := (mem_measurableCylinders s).1 hs
  /-
    case intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    I : Finset ι
    t : Set ((i : Subtype fun x => Membership.mem I x) → α ↑i)
    mt : MeasurableSet t
    hs : Membership.mem (MeasureTheory.measurableCylinders α) (MeasureTheory.cylin …
    ⊢ MeasurableSet (MeasureTheory.cylinder I t)
  -/
  exact mt.cylinder
  /-
    🎉 no goals
  -/


/-- A finset `s` such that `t = cylinder s S`. `S` is given by `measurableCylinders.set`. -/
noncomputable def measurableCylinders.finset (ht : t ∈ measurableCylinders α) : Finset ι :=
  ((mem_measurableCylinders t).mp ht).choose


/-- A set `S` such that `t = cylinder s S`. `s` is given by `measurableCylinders.finset`. -/
def measurableCylinders.set (ht : t ∈ measurableCylinders α) :
    Set (∀ i : measurableCylinders.finset ht, α i) :=
  ((mem_measurableCylinders t).mp ht).choose_spec.choose


theorem measurableCylinders.measurableSet (ht : t ∈ measurableCylinders α) :
    MeasurableSet (measurableCylinders.set ht) :=
  ((mem_measurableCylinders t).mp ht).choose_spec.choose_spec.left


theorem measurableCylinders.eq_cylinder (ht : t ∈ measurableCylinders α) :
    t = cylinder (measurableCylinders.finset ht) (measurableCylinders.set ht) :=
  ((mem_measurableCylinders t).mp ht).choose_spec.choose_spec.right


theorem cylinder_mem_measurableCylinders (s : Finset ι) (S : Set (∀ i : s, α i))
    (hS : MeasurableSet S) :
    cylinder s S ∈ measurableCylinders α := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    s : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    hS : MeasurableSet S
    ⊢ Membership.mem (MeasureTheory.measurableCylinders α) (MeasureTheory.cylinder …
  -/
  rw [mem_measurableCylinders]; exact ⟨s, S, hS, rfl⟩
                                /-
                                  🎉 no goals
                                -/


theorem inter_mem_measurableCylinders (hs : s ∈ measurableCylinders α)
    (ht : t ∈ measurableCylinders α) :
    s ∩ t ∈ measurableCylinders α := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    s t : Set ((i : ι) → α i)
    hs : Membership.mem (MeasureTheory.measurableCylinders α) s
    ht : Membership.mem (MeasureTheory.measurableCylinders α) t
    ⊢ Membership.mem (MeasureTheory.measurableCylinders α) (Inter.inter s t)
  -/
  rw [mem_measurableCylinders] at *
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    s t : Set ((i : ι) → α i)
    hs : Exists fun s_1 => Exists fun S => And (MeasurableSet S) (Eq s (MeasureThe …
    ht : Exists fun s => Exists fun S => And (MeasurableSet S) (Eq t (MeasureTheor …
    ⊢ Exists fun s_1 => Exists fun S => And (MeasurableSet S) (Eq (Inter.inter s t …
  -/
  obtain ⟨s₁, S₁, hS₁, rfl⟩ := hs
  /-
    case intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    t : Set ((i : ι) → α i)
    ht : Exists fun s => Exists fun S => And (MeasurableSet S) (Eq t (MeasureTheor …
    s₁ : Finset ι
    S₁ : Set ((i : Subtype fun x => Membership.mem s₁ x) → α ↑i)
    hS₁ : MeasurableSet S₁
    ⊢ Exists fun s => Exists fun S => And (MeasurableSet S) (Eq (Inter.inter (Meas …
  -/
  obtain ⟨s₂, S₂, hS₂, rfl⟩ := ht
  classical
  refine ⟨s₁ ∪ s₂,
    Finset.restrict₂ Finset.subset_union_left ⁻¹' S₁ ∩
      {f | Finset.restrict₂ Finset.subset_union_right f ∈ S₂}, ?_, ?_⟩
  · refine MeasurableSet.inter ?_ ?_
    · exact measurable_pi_lambda _ (fun _ ↦ measurable_pi_apply _) hS₁
    · exact measurable_pi_lambda _ (fun _ ↦ measurable_pi_apply _) hS₂
  · exact inter_cylinder _ _ _ _


theorem isPiSystem_measurableCylinders : IsPiSystem (measurableCylinders α) :=
  fun _ hS _ hT _ ↦ inter_mem_measurableCylinders hS hT


theorem compl_mem_measurableCylinders (hs : s ∈ measurableCylinders α) :
    sᶜ ∈ measurableCylinders α := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    s : Set ((i : ι) → α i)
    hs : Membership.mem (MeasureTheory.measurableCylinders α) s
    ⊢ Membership.mem (MeasureTheory.measurableCylinders α) (HasCompl.compl s)
  -/
  rw [mem_measurableCylinders] at hs ⊢
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    s : Set ((i : ι) → α i)
    hs : Exists fun s_1 => Exists fun S => And (MeasurableSet S) (Eq s (MeasureThe …
    ⊢ Exists fun s_1 => Exists fun S => And (MeasurableSet S) (Eq (HasCompl.compl  …
  -/
  obtain ⟨s, S, hS, rfl⟩ := hs
  /-
    case intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    s : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    hS : MeasurableSet S
    ⊢ Exists fun s_1 => Exists fun S_1 => And (MeasurableSet S_1) (Eq (HasCompl.co …
  -/
  refine ⟨s, Sᶜ, hS.compl, ?_⟩
  /-
    case intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    s : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
    hS : MeasurableSet S
    ⊢ Eq (HasCompl.compl (MeasureTheory.cylinder s S)) (MeasureTheory.cylinder s ( …
  -/
  rw [compl_cylinder]
  /-
    🎉 no goals
  -/


theorem univ_mem_measurableCylinders (α : ι → Type*) [∀ i, MeasurableSpace (α i)] :
    Set.univ ∈ measurableCylinders α := by
  /-
    ι : Type u_2
    α : ι → Type u_1
    inst✝ : (i : ι) → MeasurableSpace (α i)
    ⊢ Membership.mem (MeasureTheory.measurableCylinders α) Set.univ
  -/
  rw [← compl_empty]; exact compl_mem_measurableCylinders (empty_mem_measurableCylinders α)
                      /-
                        🎉 no goals
                      -/


theorem union_mem_measurableCylinders (hs : s ∈ measurableCylinders α)
    (ht : t ∈ measurableCylinders α) :
    s ∪ t ∈ measurableCylinders α := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    s t : Set ((i : ι) → α i)
    hs : Membership.mem (MeasureTheory.measurableCylinders α) s
    ht : Membership.mem (MeasureTheory.measurableCylinders α) t
    ⊢ Membership.mem (MeasureTheory.measurableCylinders α) (Union.union s t)
  -/
  rw [union_eq_compl_compl_inter_compl]
  exact compl_mem_measurableCylinders (inter_mem_measurableCylinders
    (compl_mem_measurableCylinders hs) (compl_mem_measurableCylinders ht))


theorem diff_mem_measurableCylinders (hs : s ∈ measurableCylinders α)
    (ht : t ∈ measurableCylinders α) :
    s \ t ∈ measurableCylinders α := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    s t : Set ((i : ι) → α i)
    hs : Membership.mem (MeasureTheory.measurableCylinders α) s
    ht : Membership.mem (MeasureTheory.measurableCylinders α) t
    ⊢ Membership.mem (MeasureTheory.measurableCylinders α) (SDiff.sdiff s t)
  -/
  rw [diff_eq_compl_inter]
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    s t : Set ((i : ι) → α i)
    hs : Membership.mem (MeasureTheory.measurableCylinders α) s
    ht : Membership.mem (MeasureTheory.measurableCylinders α) t
    ⊢ Membership.mem (MeasureTheory.measurableCylinders α) (Inter.inter (HasCompl. …
  -/
  exact inter_mem_measurableCylinders (compl_mem_measurableCylinders ht) hs
  /-
    🎉 no goals
  -/


/-- The measurable cylinders generate the product σ-algebra. -/
theorem generateFrom_measurableCylinders :
    MeasurableSpace.generateFrom (measurableCylinders α) = MeasurableSpace.pi := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    ⊢ Eq (MeasurableSpace.generateFrom (MeasureTheory.measurableCylinders α)) Meas …
  -/
  apply le_antisymm
    /-
      case a
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → MeasurableSpace (α i)
      ⊢ LE.le (MeasurableSpace.generateFrom (MeasureTheory.measurableCylinders α)) M …
    -/
  · refine MeasurableSpace.generateFrom_le (fun S hS ↦ ?_)
    /-
      case a
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → MeasurableSpace (α i)
      S : Set ((i : ι) → α i)
      hS : Membership.mem (MeasureTheory.measurableCylinders α) S
      ⊢ MeasurableSet S
    -/
    obtain ⟨s, S, hSm, rfl⟩ := (mem_measurableCylinders _).mp hS
    /-
      case a.intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → MeasurableSpace (α i)
      s : Finset ι
      S : Set ((i : Subtype fun x => Membership.mem s x) → α ↑i)
      hSm : MeasurableSet S
      hS : Membership.mem (MeasureTheory.measurableCylinders α) (MeasureTheory.cylin …
      ⊢ MeasurableSet (MeasureTheory.cylinder s S)
    -/
    exact hSm.cylinder
    /-
      🎉 no goals
    -/
    /-
      case a
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → MeasurableSpace (α i)
      ⊢ LE.le MeasurableSpace.pi (MeasurableSpace.generateFrom (MeasureTheory.measur …
    -/
  · refine iSup_le fun i ↦ ?_
    /-
      case a
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → MeasurableSpace (α i)
      i : ι
      ⊢ LE.le (MeasurableSpace.comap (fun b => b i) (inst✝ i)) (MeasurableSpace.gene …
    -/
    refine (comap_eval_le_generateFrom_squareCylinders_singleton α i).trans ?_
    /-
      case a
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → MeasurableSpace (α i)
      i : ι
      ⊢ LE.le (MeasurableSpace.generateFrom (Set.image (fun t => (Singleton.singleto …
    -/
    refine MeasurableSpace.generateFrom_mono (fun x ↦ ?_)
    simp only [singleton_pi, Function.eval, mem_image, mem_pi, mem_univ, mem_setOf_eq,
      forall_true_left, mem_measurableCylinders, exists_prop, forall_exists_index, and_imp]
    /-
      case a
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → MeasurableSpace (α i)
      i : ι
      x : Set ((i : ι) → α i)
      ⊢ ∀ (x_1 : (i : ι) → Set (α i)), (∀ (i : ι), MeasurableSet (x_1 i)) → Eq (Set. …
    -/
    rintro t ht rfl
    /-
      case a
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → MeasurableSpace (α i)
      i : ι
      t : (i : ι) → Set (α i)
      ht : ∀ (i : ι), MeasurableSet (t i)
      ⊢ Exists fun s => Exists fun S => And (MeasurableSet S) (Eq (Set.preimage (Fun …
    -/
    refine ⟨{i}, {f | f ⟨i, Finset.mem_singleton_self i⟩ ∈ t i}, measurable_pi_apply _ (ht i), ?_⟩
    /-
      case a
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → MeasurableSpace (α i)
      i : ι
      t : (i : ι) → Set (α i)
      ht : ∀ (i : ι), MeasurableSet (t i)
      ⊢ Eq (Set.preimage (Function.eval i) (t i)) (MeasureTheory.cylinder (Singleton …
    -/
    ext1 x
    /-
      case a.h
      ι : Type u_1
      α : ι → Type u_2
      inst✝ : (i : ι) → MeasurableSpace (α i)
      i : ι
      t : (i : ι) → Set (α i)
      ht : ∀ (i : ι), MeasurableSet (t i)
      x : (x : ι) → α x
      ⊢ Iff (Membership.mem (Set.preimage (Function.eval i) (t i)) x) (Membership.me …
    -/
    simp only [mem_preimage, Function.eval, mem_cylinder, mem_setOf_eq, Finset.restrict]
    /-
      🎉 no goals
    -/


/-- The σ-algebra of cylinder events on `Δ`. It is the smallest σ-algebra making the projections
on the `i`-th coordinate continuous for all `i ∈ Δ`. -/
def cylinderEvents (Δ : Set ι) : MeasurableSpace (∀ i, π i) := ⨆ i ∈ Δ, (m i).comap fun σ ↦ σ i


@[simp] lemma cylinderEvents_univ : cylinderEvents (π := π) univ = MeasurableSpace.pi := by
  /-
    ι : Type u_2
    π : ι → Type u_3
    m : (i : ι) → MeasurableSpace (π i)
    ⊢ Eq (MeasureTheory.cylinderEvents Set.univ) MeasurableSpace.pi
  -/
  simp [cylinderEvents, MeasurableSpace.pi]
  /-
    🎉 no goals
  -/


@[gcongr]
lemma cylinderEvents_mono (h : Δ₁ ⊆ Δ₂) : cylinderEvents (π := π) Δ₁ ≤ cylinderEvents Δ₂ :=
  biSup_mono h


lemma cylinderEvents_le_pi : cylinderEvents (π := π) Δ ≤ MeasurableSpace.pi := by
  /-
    ι : Type u_2
    π : ι → Type u_3
    m : (i : ι) → MeasurableSpace (π i)
    Δ : Set ι
    ⊢ LE.le (MeasureTheory.cylinderEvents Δ) MeasurableSpace.pi
  -/
  simpa using cylinderEvents_mono (subset_univ _)
  /-
    🎉 no goals
  -/


lemma measurable_cylinderEvents_iff {g : α → ∀ i, π i} :
    @Measurable _ _ _ (cylinderEvents Δ) g ↔ ∀ ⦃i⦄, i ∈ Δ → Measurable fun a ↦ g a i := by
  simp_rw [measurable_iff_comap_le, cylinderEvents, MeasurableSpace.comap_iSup,
    MeasurableSpace.comap_comp, Function.comp_def, iSup_le_iff]


@[fun_prop, aesop safe 100 apply (rule_sets := [Measurable])]
lemma measurable_cylinderEvent_apply (hi : i ∈ Δ) :
    Measurable[cylinderEvents Δ] fun f : ∀ i, π i => f i :=
  measurable_cylinderEvents_iff.1 measurable_id hi


@[aesop safe 100 apply (rule_sets := [Measurable])]
lemma Measurable.eval_cylinderEvents {g : α → ∀ i, π i} (hi : i ∈ Δ)
    (hg : @Measurable _ _ _ (cylinderEvents Δ) g) : Measurable fun a ↦ g a i :=
  (measurable_cylinderEvent_apply hi).comp hg


@[fun_prop, aesop safe 100 apply (rule_sets := [Measurable])]
lemma measurable_cylinderEvents_lambda (f : α → ∀ i, π i) (hf : ∀ i, Measurable fun a ↦ f a i) :
    Measurable f :=
  measurable_pi_iff.mpr hf


/-- The function `(f, x) ↦ update f a x : (Π a, π a) × π a → Π a, π a` is measurable. -/
lemma measurable_update_cylinderEvents' [DecidableEq ι] :
    @Measurable _ _ (.prod (cylinderEvents Δ) (m i)) (cylinderEvents Δ)
      (fun p : (∀ i, π i) × π i ↦ update p.1 i p.2) := by
  /-
    ι : Type u_2
    π : ι → Type u_3
    m : (i : ι) → MeasurableSpace (π i)
    Δ : Set ι
    i : ι
    inst✝ : DecidableEq ι
    ⊢ Measurable fun p => Function.update p.1 i p.2
  -/
  rw [measurable_cylinderEvents_iff]
  /-
    ι : Type u_2
    π : ι → Type u_3
    m : (i : ι) → MeasurableSpace (π i)
    Δ : Set ι
    i : ι
    inst✝ : DecidableEq ι
    ⊢ ∀ ⦃i_1 : ι⦄, Membership.mem Δ i_1 → Measurable fun a => Function.update a.1  …
  -/
  intro j hj
  /-
    ι : Type u_2
    π : ι → Type u_3
    m : (i : ι) → MeasurableSpace (π i)
    Δ : Set ι
    i : ι
    inst✝ : DecidableEq ι
    j : ι
    hj : Membership.mem Δ j
    ⊢ Measurable fun a => Function.update a.1 i a.2 j
  -/
  dsimp [update]
  /-
    ι : Type u_2
    π : ι → Type u_3
    m : (i : ι) → MeasurableSpace (π i)
    Δ : Set ι
    i : ι
    inst✝ : DecidableEq ι
    j : ι
    hj : Membership.mem Δ j
    ⊢ Measurable fun a => dite (Eq j i) (fun h => Eq.rec a.2 ⋯) fun h => a.1 j
  -/
  split_ifs with h
    /-
      case pos
      ι : Type u_2
      π : ι → Type u_3
      m : (i : ι) → MeasurableSpace (π i)
      Δ : Set ι
      i : ι
      inst✝ : DecidableEq ι
      j : ι
      hj : Membership.mem Δ j
      h : Eq j i
      ⊢ Measurable fun a => Eq.rec a.2 ⋯
    -/
  · subst h
    /-
      case pos
      ι : Type u_2
      π : ι → Type u_3
      m : (i : ι) → MeasurableSpace (π i)
      Δ : Set ι
      inst✝ : DecidableEq ι
      j : ι
      hj : Membership.mem Δ j
      ⊢ Measurable fun a => Eq.rec a.2 ⋯
    -/
    dsimp
    /-
      case pos
      ι : Type u_2
      π : ι → Type u_3
      m : (i : ι) → MeasurableSpace (π i)
      Δ : Set ι
      inst✝ : DecidableEq ι
      j : ι
      hj : Membership.mem Δ j
      ⊢ Measurable fun a => a.2
    -/
    exact measurable_snd
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_2
      π : ι → Type u_3
      m : (i : ι) → MeasurableSpace (π i)
      Δ : Set ι
      i : ι
      inst✝ : DecidableEq ι
      j : ι
      hj : Membership.mem Δ j
      h : Not (Eq j i)
      ⊢ Measurable fun a => a.1 j
    -/
  · exact measurable_cylinderEvents_iff.1 measurable_fst hj
    /-
      🎉 no goals
    -/


lemma measurable_uniqueElim_cylinderEvents [Unique ι] :
    Measurable (uniqueElim : π (default : ι) → ∀ i, π i) := by
  /-
    ι : Type u_2
    π : ι → Type u_3
    m : (i : ι) → MeasurableSpace (π i)
    inst✝ : Unique ι
    ⊢ Measurable uniqueElim
  -/
  simp_rw [measurable_pi_iff, Unique.forall_iff, uniqueElim_default]; exact measurable_id
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- The function `update f a : π a → Π a, π a` is always measurable.
This doesn't require `f` to be measurable.
This should not be confused with the statement that `update f a x` is measurable. -/
@[measurability]
lemma measurable_update_cylinderEvents (f : ∀ a : ι, π a) {a : ι} [DecidableEq ι] :
    @Measurable _ _ _ (cylinderEvents Δ) (update f a) :=
  measurable_update_cylinderEvents'.comp measurable_prod_mk_left


lemma measurable_update_cylinderEvents_left {a : ι} [DecidableEq ι] {x : π a} :
    @Measurable _ _ (cylinderEvents Δ) (cylinderEvents Δ) (update · a x) :=
  measurable_update_cylinderEvents'.comp measurable_prod_mk_right


lemma measurable_restrict_cylinderEvents (Δ : Set ι) :
    Measurable[cylinderEvents (π := π) Δ] (restrict Δ) := by
  /-
    ι : Type u_2
    π : ι → Type u_3
    m : (i : ι) → MeasurableSpace (π i)
    Δ : Set ι
    ⊢ Measurable Δ.restrict
  -/
  rw [@measurable_pi_iff]; exact fun i ↦ measurable_cylinderEvent_apply i.2
                           /-
                             🎉 no goals
                           -/


