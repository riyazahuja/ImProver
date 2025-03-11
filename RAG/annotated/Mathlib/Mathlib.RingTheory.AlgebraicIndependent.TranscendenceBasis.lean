theorem exists_isTranscendenceBasis (h : Injective (algebraMap R A)) :
    ∃ s : Set A, IsTranscendenceBasis R ((↑) : s → A) := by
  cases' exists_maximal_algebraicIndependent (∅ : Set A) Set.univ (Set.subset_univ _)
      ((algebraicIndependent_empty_iff R A).2 h) with
    s hs
  /-
    case intro
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    h : Function.Injective ⇑(algebraMap R A)
    s : Set A
    hs : And (HasSubset.Subset EmptyCollection.emptyCollection s) (Maximal (fun x  …
    ⊢ Exists fun s => IsTranscendenceBasis R Subtype.val
  -/
  refine ⟨s, hs.2.1.1, fun t ht hst ↦ ?_⟩
  /-
    case intro
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    h : Function.Injective ⇑(algebraMap R A)
    s : Set A
    hs : And (HasSubset.Subset EmptyCollection.emptyCollection s) (Maximal (fun x  …
    t : Set A
    ht : AlgebraicIndependent R Subtype.val
    hst : LE.le (Set.range Subtype.val) t
    ⊢ Eq (Set.range Subtype.val) t
  -/
  simp only [Subtype.range_coe_subtype, setOf_mem_eq] at *
  /-
    case intro
    R : Type u_3
    A : Type u_5
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    h : Function.Injective ⇑(algebraMap R A)
    s : Set A
    hs : And (HasSubset.Subset EmptyCollection.emptyCollection s) (Maximal (fun x  …
    t : Set A
    ht : AlgebraicIndependent R Subtype.val
    hst : LE.le s t
    ⊢ Eq s t
  -/
  exact hs.2.eq_of_le ⟨ht, subset_univ _⟩ hst
  /-
    🎉 no goals
  -/


/-- `Type` version of `exists_isTranscendenceBasis`. -/
theorem exists_isTranscendenceBasis' (R : Type u) {A : Type v} [CommRing R] [CommRing A]
    [Algebra R A] (h : Injective (algebraMap R A)) :
    ∃ (ι : Type v) (x : ι → A), IsTranscendenceBasis R x := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    h : Function.Injective ⇑(algebraMap R A)
    ⊢ Exists fun ι => Exists fun x => IsTranscendenceBasis R x
  -/
  obtain ⟨s, h⟩ := exists_isTranscendenceBasis R h
  /-
    case intro
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    h✝ : Function.Injective ⇑(algebraMap R A)
    s : Set A
    h : IsTranscendenceBasis R Subtype.val
    ⊢ Exists fun ι => Exists fun x => IsTranscendenceBasis R x
  -/
  exact ⟨s, Subtype.val, h⟩
  /-
    🎉 no goals
  -/


theorem AlgebraicIndependent.isTranscendenceBasis_iff {ι : Type w} {R : Type u} [CommRing R]
    [Nontrivial R] {A : Type v} [CommRing A] [Algebra R A] {x : ι → A}
    (i : AlgebraicIndependent R x) :
    IsTranscendenceBasis R x ↔
      ∀ (κ : Type v) (w : κ → A) (_ : AlgebraicIndependent R w) (j : ι → κ) (_ : w ∘ j = x),
        Surjective j := by
  /-
    ι : Type w
    R : Type u
    inst✝³ : CommRing R
    inst✝² : Nontrivial R
    A : Type v
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    x : ι → A
    i : AlgebraicIndependent R x
    ⊢ Iff (IsTranscendenceBasis R x) (∀ (κ : Type v) (w : κ → A), AlgebraicIndepen …
  -/
  fconstructor
    /-
      case mp
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      x : ι → A
      i : AlgebraicIndependent R x
      ⊢ IsTranscendenceBasis R x → ∀ (κ : Type v) (w : κ → A), AlgebraicIndependent  …
    -/
  · rintro p κ w i' j rfl
    /-
      case mp
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      κ : Type v
      w : κ → A
      i' : AlgebraicIndependent R w
      j : ι → κ
      i : AlgebraicIndependent R (Function.comp w j)
      p : IsTranscendenceBasis R (Function.comp w j)
      ⊢ Function.Surjective j
    -/
    have p := p.2 (range w) i'.coe_range (range_comp_subset_range _ _)
    /-
      case mp
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      κ : Type v
      w : κ → A
      i' : AlgebraicIndependent R w
      j : ι → κ
      i : AlgebraicIndependent R (Function.comp w j)
      p✝ : IsTranscendenceBasis R (Function.comp w j)
      p : Eq (Set.range (Function.comp w j)) (Set.range w)
      ⊢ Function.Surjective j
    -/
    rw [range_comp, ← @image_univ _ _ w] at p
    /-
      case mp
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      κ : Type v
      w : κ → A
      i' : AlgebraicIndependent R w
      j : ι → κ
      i : AlgebraicIndependent R (Function.comp w j)
      p✝ : IsTranscendenceBasis R (Function.comp w j)
      p : Eq (Set.image w (Set.range j)) (Set.image w Set.univ)
      ⊢ Function.Surjective j
    -/
    exact range_eq_univ.mp (image_injective.mpr i'.injective p)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      x : ι → A
      i : AlgebraicIndependent R x
      ⊢ (∀ (κ : Type v) (w : κ → A), AlgebraicIndependent R w → ∀ (j : ι → κ), Eq (F …
    -/
  · intro p
    /-
      case mpr
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      x : ι → A
      i : AlgebraicIndependent R x
      p : ∀ (κ : Type v) (w : κ → A), AlgebraicIndependent R w → ∀ (j : ι → κ), Eq ( …
      ⊢ IsTranscendenceBasis R x
    -/
    use i
    /-
      case right
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      x : ι → A
      i : AlgebraicIndependent R x
      p : ∀ (κ : Type v) (w : κ → A), AlgebraicIndependent R w → ∀ (j : ι → κ), Eq ( …
      ⊢ ∀ (s : Set A), AlgebraicIndependent R Subtype.val → LE.le (Set.range x) s →  …
    -/
    intro w i' h
    /-
      case right
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      x : ι → A
      i : AlgebraicIndependent R x
      p : ∀ (κ : Type v) (w : κ → A), AlgebraicIndependent R w → ∀ (j : ι → κ), Eq ( …
      w : Set A
      i' : AlgebraicIndependent R Subtype.val
      h : LE.le (Set.range x) w
      ⊢ Eq (Set.range x) w
    -/
    specialize p w ((↑) : w → A) i' (fun i => ⟨x i, range_subset_iff.mp h i⟩) (by ext; simp)
    /-
      case right
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      x : ι → A
      i : AlgebraicIndependent R x
      w : Set A
      i' : AlgebraicIndependent R Subtype.val
      h : LE.le (Set.range x) w
      p : Function.Surjective fun i => ⟨x i, ⋯⟩
      ⊢ Eq (Set.range x) w
    -/
    have q := congr_arg (fun s => ((↑) : w → A) '' s) p.range_eq
    /-
      case right
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      x : ι → A
      i : AlgebraicIndependent R x
      w : Set A
      i' : AlgebraicIndependent R Subtype.val
      h : LE.le (Set.range x) w
      p : Function.Surjective fun i => ⟨x i, ⋯⟩
      q : Eq ((fun s => Set.image Subtype.val s) (Set.range fun i => ⟨x i, ⋯⟩)) ((fu …
      ⊢ Eq (Set.range x) w
    -/
    dsimp at q
    /-
      case right
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      x : ι → A
      i : AlgebraicIndependent R x
      w : Set A
      i' : AlgebraicIndependent R Subtype.val
      h : LE.le (Set.range x) w
      p : Function.Surjective fun i => ⟨x i, ⋯⟩
      q : Eq (Set.image Subtype.val (Set.range fun i => ⟨x i, ⋯⟩)) (Set.image Subtyp …
      ⊢ Eq (Set.range x) w
    -/
    rw [← image_univ, image_image] at q
    /-
      case right
      ι : Type w
      R : Type u
      inst✝³ : CommRing R
      inst✝² : Nontrivial R
      A : Type v
      inst✝¹ : CommRing A
      inst✝ : Algebra R A
      x : ι → A
      i : AlgebraicIndependent R x
      w : Set A
      i' : AlgebraicIndependent R Subtype.val
      h : LE.le (Set.range x) w
      p : Function.Surjective fun i => ⟨x i, ⋯⟩
      q : Eq (Set.image (fun x_1 => ↑⟨x x_1, ⋯⟩) Set.univ) (Set.image Subtype.val Se …
      ⊢ Eq (Set.range x) w
    -/
    simpa using q
    /-
      🎉 no goals
    -/


theorem IsTranscendenceBasis.isAlgebraic [Nontrivial R] (hx : IsTranscendenceBasis R x) :
    Algebra.IsAlgebraic (adjoin R (range x)) A := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    hx : IsTranscendenceBasis R x
    ⊢ Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Se …
  -/
  constructor
  /-
    case isAlgebraic
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    hx : IsTranscendenceBasis R x
    ⊢ ∀ (x_1 : A), IsAlgebraic (Subtype fun x_2 => Membership.mem (Algebra.adjoin  …
  -/
  intro a
  /-
    case isAlgebraic
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    hx : IsTranscendenceBasis R x
    a : A
    ⊢ IsAlgebraic (Subtype fun x_1 => Membership.mem (Algebra.adjoin R (Set.range  …
  -/
  rw [← not_iff_comm.1 (hx.1.option_iff _).symm]
  /-
    case isAlgebraic
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    hx : IsTranscendenceBasis R x
    a : A
    ⊢ Not (AlgebraicIndependent R fun o => o.elim a x)
  -/
  intro ai
  have h₁ : range x ⊆ range fun o : Option ι => o.elim a x := by
    rintro x ⟨y, rfl⟩
    exact ⟨some y, rfl⟩
  have h₂ : range x ≠ range fun o : Option ι => o.elim a x := by
    intro h
    have : a ∈ range x := by
      rw [h]
      exact ⟨none, rfl⟩
    rcases this with ⟨b, rfl⟩
    have : some b = none := ai.injective rfl
    simpa
  exact h₂ (hx.2 (Set.range fun o : Option ι => o.elim a x)
    ((algebraicIndependent_subtype_range ai.injective).2 ai) h₁)


/-- If `x` is a transcendence basis of `A/R`, then it is empty if and only if
`A/R` is algebraic. -/
theorem IsTranscendenceBasis.isEmpty_iff_isAlgebraic [Nontrivial R]
    (hx : IsTranscendenceBasis R x) :
    IsEmpty ι ↔ Algebra.IsAlgebraic R A := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    hx : IsTranscendenceBasis R x
    ⊢ Iff (IsEmpty ι) (Algebra.IsAlgebraic R A)
  -/
  refine ⟨fun _ ↦ ?_, fun _ ↦ hx.1.isEmpty_of_isAlgebraic⟩
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    hx : IsTranscendenceBasis R x
    x✝ : IsEmpty ι
    ⊢ Algebra.IsAlgebraic R A
  -/
  have := hx.isAlgebraic
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    hx : IsTranscendenceBasis R x
    x✝ : IsEmpty ι
    this : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (Algebra.adjoin  …
    ⊢ Algebra.IsAlgebraic R A
  -/
  rw [Set.range_eq_empty x, adjoin_empty] at this
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    hx : IsTranscendenceBasis R x
    x✝ : IsEmpty ι
    this : Algebra.IsAlgebraic (Subtype fun x => Membership.mem Bot.bot x) A
    ⊢ Algebra.IsAlgebraic R A
  -/
  exact algebra_isAlgebraic_of_algebra_isAlgebraic_bot_left R A
  /-
    🎉 no goals
  -/


/-- If `x` is a transcendence basis of `A/R`, then it is not empty if and only if
`A/R` is transcendental. -/
theorem IsTranscendenceBasis.nonempty_iff_transcendental [Nontrivial R]
    (hx : IsTranscendenceBasis R x) :
    Nonempty ι ↔ Algebra.Transcendental R A := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_5
    x : ι → A
    inst✝³ : CommRing R
    inst✝² : CommRing A
    inst✝¹ : Algebra R A
    inst✝ : Nontrivial R
    hx : IsTranscendenceBasis R x
    ⊢ Iff (Nonempty ι) (Algebra.Transcendental R A)
  -/
  rw [← not_isEmpty_iff, Algebra.transcendental_iff_not_isAlgebraic, hx.isEmpty_iff_isAlgebraic]
  /-
    🎉 no goals
  -/


theorem IsTranscendenceBasis.isAlgebraic_field {F E : Type*} {x : ι → E}
    [Field F] [Field E] [Algebra F E] (hx : IsTranscendenceBasis F x) :
    Algebra.IsAlgebraic (IntermediateField.adjoin F (range x)) E := by
  /-
    ι : Type u_1
    F : Type u_7
    E : Type u_8
    x : ι → E
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    hx : IsTranscendenceBasis F x
    ⊢ Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateField.ad …
  -/
  haveI := hx.isAlgebraic
  /-
    ι : Type u_1
    F : Type u_7
    E : Type u_8
    x : ι → E
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    hx : IsTranscendenceBasis F x
    this : Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (Algebra.adjoin  …
    ⊢ Algebra.IsAlgebraic (Subtype fun x_1 => Membership.mem (IntermediateField.ad …
  -/
  set S := range x
  letI : Algebra (adjoin F S) (IntermediateField.adjoin F S) :=
    (Subalgebra.inclusion (IntermediateField.algebra_adjoin_le_adjoin F S)).toRingHom.toAlgebra
  haveI : IsScalarTower (adjoin F S) (IntermediateField.adjoin F S) E :=
    IsScalarTower.of_algebraMap_eq (congrFun rfl)
  /-
    ι : Type u_1
    F : Type u_7
    E : Type u_8
    x : ι → E
    inst✝² : Field F
    inst✝¹ : Field E
    inst✝ : Algebra F E
    hx : IsTranscendenceBasis F x
    S : Set E := Set.range x
    this✝¹ : Algebra.IsAlgebraic (Subtype fun x => Membership.mem (Algebra.adjoin  …
    this✝ : Algebra (Subtype fun x => Membership.mem (Algebra.adjoin F S) x) (Subt …
    this : IsScalarTower (Subtype fun x => Membership.mem (Algebra.adjoin F S) x)  …
    ⊢ Algebra.IsAlgebraic (Subtype fun x => Membership.mem (IntermediateField.adjo …
  -/
  exact Algebra.IsAlgebraic.extendScalars (R := adjoin F S) (Subalgebra.inclusion_injective _)
  /-
    🎉 no goals
  -/

