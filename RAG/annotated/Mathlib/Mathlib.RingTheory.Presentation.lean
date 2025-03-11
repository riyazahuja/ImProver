/--
A presentation of an `R`-algebra `S` is a family of
generators with
1. `rels`: The type of relations.
2. `relation : relations → MvPolynomial vars R`: The assignment of
each relation to a polynomial in the generators.
-/
@[nolint checkUnivs]
structure Algebra.Presentation extends Algebra.Generators.{w} R S where
  /-- The type of relations. -/
  rels : Type t
  /-- The assignment of each relation to a polynomial in the generators. -/
  relation : rels → toGenerators.Ring
  /-- The relations span the kernel of the canonical map. -/
  span_range_relation_eq_ker :
    Ideal.span (Set.range relation) = toGenerators.ker


@[simp]
lemma aeval_val_relation (i) : aeval P.val (P.relation i) = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Presentation R S
    i : P.rels
    ⊢ Eq ((MvPolynomial.aeval P.val) (P.relation i)) 0
  -/
  rw [← RingHom.mem_ker, ← P.ker_eq_ker_aeval_val, ← P.span_range_relation_eq_ker]
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Algebra.Presentation R S
    i : P.rels
    ⊢ Membership.mem (Ideal.span (Set.range P.relation)) (P.relation i)
  -/
  exact Ideal.subset_span ⟨i, rfl⟩
  /-
    🎉 no goals
  -/


/-- The polynomial algebra wrt a family of generators modulo a family of relations. -/
protected abbrev Quotient : Type (max w u) := P.Ring ⧸ P.ker


/-- `P.Quotient` is `P.Ring`-isomorphic to `S` and in particular `R`-isomorphic to `S`. -/
def quotientEquiv : P.Quotient ≃ₐ[P.Ring] S :=
  Ideal.quotientKerAlgEquivOfRightInverse (f := Algebra.ofId P.Ring S) (g := P.σ) <| fun x ↦ by
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Algebra.Presentation R S
      x : S
      ⊢ Eq ((Algebra.ofId P.Ring S) (P.σ x)) x
    -/
    rw [Algebra.ofId_apply, P.algebraMap_apply, P.aeval_val_σ]
    /-
      🎉 no goals
    -/


@[simp]
lemma quotientEquiv_mk (p : P.Ring) : P.quotientEquiv p = algebraMap P.Ring S p :=
  rfl


@[simp]
lemma quotientEquiv_symm (x : S) : P.quotientEquiv.symm x = P.σ x :=
  rfl


/--
Dimension of a presentation defined as the cardinality of the generators
minus the cardinality of the relations.

Note: this definition is completely non-sensical for non-finite presentations and
even then for this to make sense, you should assume that the presentation
is a complete intersection.
-/
noncomputable def dimension : ℕ :=
  Nat.card P.vars - Nat.card P.rels


/-- A presentation is finite if there are only finitely-many
relations and finitely-many relations. -/
class IsFinite (P : Presentation.{t, w} R S) : Prop where
  finite_vars : Finite P.vars
  finite_rels : Finite P.rels


lemma ideal_fg_of_isFinite [P.IsFinite] : P.ker.FG := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Algebra.Presentation R S
    inst✝ : P.IsFinite
    ⊢ P.ker.FG
  -/
  use (Set.finite_range P.relation).toFinset
  /-
    case h
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    P : Algebra.Presentation R S
    inst✝ : P.IsFinite
    ⊢ Eq (Ideal.span ↑⋯.toFinset) P.ker
  -/
  simp [span_range_relation_eq_ker]
  /-
    🎉 no goals
  -/


/-- If a presentation is finite, the corresponding quotient is
of finite presentation. -/
instance [P.IsFinite] : FinitePresentation R P.Quotient :=
  FinitePresentation.quotient P.ideal_fg_of_isFinite


lemma finitePresentation_of_isFinite [P.IsFinite] :
    FinitePresentation R S :=
  FinitePresentation.equiv (P.quotientEquiv.restrictScalars R)


/-- If `algebraMap R S` is bijective, the empty generators are a presentation with no relations. -/
noncomputable def ofBijectiveAlgebraMap (h : Function.Bijective (algebraMap R S)) :
    Presentation.{t, w} R S where
  __ := Generators.ofSurjectiveAlgebraMap h.surjective
  rels := PEmpty
  relation := PEmpty.elim
  span_range_relation_eq_ker := by
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Algebra.Presentation R S
      h : Function.Bijective ⇑(algebraMap R S)
      ⊢ Eq (Ideal.span (Set.range PEmpty.elim)) __spread✝⁻⁰.ker
    -/
    simp only [Set.range_eq_empty, Ideal.span_empty]
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Algebra.Presentation R S
      h : Function.Bijective ⇑(algebraMap R S)
      ⊢ Eq Bot.bot (Algebra.Generators.ofSurjectiveAlgebraMap ⋯).ker
    -/
    symm
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Algebra.Presentation R S
      h : Function.Bijective ⇑(algebraMap R S)
      ⊢ Eq (Algebra.Generators.ofSurjectiveAlgebraMap ⋯).ker Bot.bot
    -/
    rw [← RingHom.injective_iff_ker_eq_bot]
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Algebra.Presentation R S
      h : Function.Bijective ⇑(algebraMap R S)
      ⊢ Function.Injective ⇑(algebraMap (Algebra.Generators.ofSurjectiveAlgebraMap ⋯ …
    -/
    show Function.Injective (aeval PEmpty.elim)
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Algebra.Presentation R S
      h : Function.Bijective ⇑(algebraMap R S)
      ⊢ Function.Injective ⇑(MvPolynomial.aeval PEmpty.elim)
    -/
    rw [aeval_injective_iff_of_isEmpty]
    /-
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Algebra.Presentation R S
      h : Function.Bijective ⇑(algebraMap R S)
      ⊢ Function.Injective ⇑(algebraMap R S)
    -/
    exact h.injective
    /-
      🎉 no goals
    -/


instance ofBijectiveAlgebraMap_isFinite (h : Function.Bijective (algebraMap R S)) :
    (ofBijectiveAlgebraMap.{t, w} h).IsFinite where
  finite_vars := inferInstanceAs (Finite PEmpty.{w + 1})
  finite_rels := inferInstanceAs (Finite PEmpty.{t + 1})


lemma ofBijectiveAlgebraMap_dimension (h : Function.Bijective (algebraMap R S)) :
    (ofBijectiveAlgebraMap h).dimension = 0 := by
  simp_rw [dimension, ofBijectiveAlgebraMap, Generators.ofSurjectiveAlgebraMap,
    Generators.ofSurjective, Nat.card_eq_fintype_card, Fintype.card_ofIsEmpty]


variable (R) in
/-- The canonical `R`-presentation of `R` with no generators and no relations. -/
noncomputable def id : Presentation.{t, w} R R := ofBijectiveAlgebraMap Function.bijective_id


instance : (id R).IsFinite := ofBijectiveAlgebraMap_isFinite (R := R) Function.bijective_id


lemma id_dimension : (Presentation.id R).dimension = 0 :=
  ofBijectiveAlgebraMap_dimension (R := R) Function.bijective_id


private lemma span_range_relation_eq_ker_localizationAway :
    Ideal.span { C r * X () - 1 } =
      RingHom.ker (aeval (S₁ := S) (Generators.localizationAway r).val) := by
  have : aeval (S₁ := S) (Generators.localizationAway r).val =
      (mvPolynomialQuotientEquiv S r).toAlgHom.comp
        (Ideal.Quotient.mkₐ R (Ideal.span {C r * X () - 1})) := by
    ext x
    simp only [Generators.localizationAway_vars, aeval_X, Generators.localizationAway_val,
      AlgEquiv.toAlgHom_eq_coe, AlgHom.coe_comp, AlgHom.coe_coe, Ideal.Quotient.mkₐ_eq_mk,
      Function.comp_apply]
    rw [IsLocalization.Away.mvPolynomialQuotientEquiv_apply, aeval_X]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    this : Eq (MvPolynomial.aeval (Algebra.Generators.localizationAway r).val) ((↑ …
    ⊢ Eq (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (MvPolynomial.C r) …
  -/
  rw [this]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    this : Eq (MvPolynomial.aeval (Algebra.Generators.localizationAway r).val) ((↑ …
    ⊢ Eq (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (MvPolynomial.C r) …
  -/
  erw [← RingHom.comap_ker]
  simp only [Generators.localizationAway_vars, AlgEquiv.toAlgHom_eq_coe, AlgHom.toRingHom_eq_coe,
    AlgEquiv.toAlgHom_toRingHom]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    this : Eq (MvPolynomial.aeval (Algebra.Generators.localizationAway r).val) ((↑ …
    ⊢ Eq (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (MvPolynomial.C r) …
  -/
  show Ideal.span {C r * X () - 1} = Ideal.comap _ (RingHom.ker (mvPolynomialQuotientEquiv S r))
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    this : Eq (MvPolynomial.aeval (Algebra.Generators.localizationAway r).val) ((↑ …
    ⊢ Eq (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (MvPolynomial.C r) …
  -/
  simp [RingHom.ker_equiv, ← RingHom.ker_eq_comap_bot]
  /-
    🎉 no goals
  -/


variable (S) in
/-- If `S` is the localization of `R` away from `r`, we can construct a natural
presentation of `S` as `R`-algebra with a single generator `X` and the relation `r * X - 1 = 0`. -/
@[simps relation, simps (config := .lemmasOnly) rels]
noncomputable def localizationAway : Presentation R S where
  toGenerators := Generators.localizationAway r
  rels := Unit
  relation _ := C r * X () - 1
  span_range_relation_eq_ker := by
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Algebra.Presentation R S
      r : R
      inst✝ : IsLocalization.Away r S
      ⊢ Eq (Ideal.span (Set.range fun x => HSub.hSub (HMul.hMul (MvPolynomial.C r) ( …
    -/
    simp only [Generators.localizationAway_vars, Set.range_const]
    /-
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      P : Algebra.Presentation R S
      r : R
      inst✝ : IsLocalization.Away r S
      ⊢ Eq (Ideal.span (Singleton.singleton (HSub.hSub (HMul.hMul (MvPolynomial.C r) …
    -/
    apply span_range_relation_eq_ker_localizationAway r
    /-
      🎉 no goals
    -/


instance localizationAway_isFinite : (localizationAway S r).IsFinite where
  finite_vars := inferInstanceAs <| Finite Unit
  finite_rels := inferInstanceAs <| Finite Unit


instance : Fintype (localizationAway S r).rels :=
  inferInstanceAs (Fintype Unit)


@[simp]
lemma localizationAway_dimension_zero : (localizationAway S r).dimension = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Eq (Algebra.Presentation.localizationAway S r).dimension 0
  -/
  simp [Presentation.dimension, localizationAway, Generators.localizationAway_vars]
  /-
    🎉 no goals
  -/


private lemma span_range_relation_eq_ker_baseChange :
    Ideal.span (Set.range fun i ↦ (MvPolynomial.map (algebraMap R T)) (P.relation i)) =
      RingHom.ker (aeval (R := T) (S₁ := T ⊗[R] S) P.baseChange.val) := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_1
    inst✝¹ : CommRing T
    inst✝ : Algebra R T
    P : Algebra.Presentation R S
    ⊢ Eq (Ideal.span (Set.range fun i => (MvPolynomial.map (algebraMap R T)) (P.re …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      ⊢ LE.le (Ideal.span (Set.range fun i => (MvPolynomial.map (algebraMap R T)) (P …
    -/
  · rw [Ideal.span_le]
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      ⊢ HasSubset.Subset (Set.range fun i => (MvPolynomial.map (algebraMap R T)) (P. …
    -/
    intro x ⟨y, hy⟩
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      y : P.rels
      hy : Eq ((fun i => (MvPolynomial.map (algebraMap R T)) (P.relation i)) y) x
      ⊢ Membership.mem (↑(RingHom.ker (MvPolynomial.aeval P.baseChange.val))) x
    -/
    have Z := aeval_val_relation P y
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      y : P.rels
      hy : Eq ((fun i => (MvPolynomial.map (algebraMap R T)) (P.relation i)) y) x
      Z : Eq ((MvPolynomial.aeval P.val) (P.relation y)) 0
      ⊢ Membership.mem (↑(RingHom.ker (MvPolynomial.aeval P.baseChange.val))) x
    -/
    apply_fun TensorProduct.includeRight (R := R) (A := T) at Z
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      y : P.rels
      hy : Eq ((fun i => (MvPolynomial.map (algebraMap R T)) (P.relation i)) y) x
      Z : Eq (Algebra.TensorProduct.includeRight ((MvPolynomial.aeval P.val) (P.rela …
      ⊢ Membership.mem (↑(RingHom.ker (MvPolynomial.aeval P.baseChange.val))) x
    -/
    rw [map_zero] at Z
    simp only [SetLike.mem_coe, RingHom.mem_ker, ← Z, ← hy, algebraMap_apply,
      TensorProduct.includeRight_apply]
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      y : P.rels
      hy : Eq ((fun i => (MvPolynomial.map (algebraMap R T)) (P.relation i)) y) x
      Z : Eq (Algebra.TensorProduct.includeRight ((MvPolynomial.aeval P.val) (P.rela …
      ⊢ Eq ((MvPolynomial.aeval P.baseChange.val) ((MvPolynomial.map (algebraMap R T …
    -/
    erw [aeval_map_algebraMap]
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      y : P.rels
      hy : Eq ((fun i => (MvPolynomial.map (algebraMap R T)) (P.relation i)) y) x
      Z : Eq (Algebra.TensorProduct.includeRight ((MvPolynomial.aeval P.val) (P.rela …
      ⊢ Eq ((MvPolynomial.aeval P.baseChange.val) (P.relation y)) (TensorProduct.tmu …
    -/
    show _ = TensorProduct.includeRight _
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      y : P.rels
      hy : Eq ((fun i => (MvPolynomial.map (algebraMap R T)) (P.relation i)) y) x
      Z : Eq (Algebra.TensorProduct.includeRight ((MvPolynomial.aeval P.val) (P.rela …
      ⊢ Eq ((MvPolynomial.aeval P.baseChange.val) (P.relation y)) (Algebra.TensorPro …
    -/
    erw [map_aeval, TensorProduct.includeRight.comp_algebraMap]
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      y : P.rels
      hy : Eq ((fun i => (MvPolynomial.map (algebraMap R T)) (P.relation i)) y) x
      Z : Eq (Algebra.TensorProduct.includeRight ((MvPolynomial.aeval P.val) (P.rela …
      ⊢ Eq ((MvPolynomial.aeval P.baseChange.val) (P.relation y)) ((MvPolynomial.eva …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      ⊢ LE.le (RingHom.ker (MvPolynomial.aeval P.baseChange.val)) (Ideal.span (Set.r …
    -/
  · intro x hx
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      hx : Membership.mem (RingHom.ker (MvPolynomial.aeval P.baseChange.val)) x
      ⊢ Membership.mem (Ideal.span (Set.range fun i => (MvPolynomial.map (algebraMap …
    -/
    rw [RingHom.mem_ker] at hx
    have H := Algebra.TensorProduct.lTensor_ker (A := T) (IsScalarTower.toAlgHom R P.Ring S)
      P.algebraMap_surjective
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      hx : Eq ((MvPolynomial.aeval P.baseChange.val) x) 0
      H : Eq (RingHom.ker (Algebra.TensorProduct.map (AlgHom.id R T) (IsScalarTower. …
      ⊢ Membership.mem (Ideal.span (Set.range fun i => (MvPolynomial.map (algebraMap …
    -/
    let e := MvPolynomial.algebraTensorAlgEquiv (R := R) (σ := P.vars) (A := T)
    have H' : e.symm x ∈ RingHom.ker (TensorProduct.map (AlgHom.id R T)
        (IsScalarTower.toAlgHom R P.Ring S)) := by
      rw [RingHom.mem_ker, ← hx]
      clear hx
      induction x using MvPolynomial.induction_on with
      | h_C a =>
        simp only [Generators.algebraMap_apply, algHom_C, TensorProduct.algebraMap_apply,
          id.map_eq_id, RingHom.id_apply, e]
        rw [← MvPolynomial.algebraMap_eq, AlgEquiv.commutes]
        simp only [TensorProduct.algebraMap_apply, id.map_eq_id, RingHom.id_apply,
          TensorProduct.map_tmul, AlgHom.coe_id, id_eq, map_one, algebraMap_eq]
        erw [aeval_C]
        simp
      | h_add p q hp hq => simp only [map_add, hp, hq]
      | h_X p i hp =>
        simp only [map_mul, algebraTensorAlgEquiv_symm_X, hp, TensorProduct.map_tmul, map_one,
          IsScalarTower.coe_toAlgHom', Generators.algebraMap_apply, aeval_X, e]
        congr
        erw [aeval_X]
        rw [Generators.baseChange_val]
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      hx : Eq ((MvPolynomial.aeval P.baseChange.val) x) 0
      H : Eq (RingHom.ker (Algebra.TensorProduct.map (AlgHom.id R T) (IsScalarTower. …
      e : AlgEquiv T (TensorProduct R T (MvPolynomial P.vars R)) (MvPolynomial P.var …
      H' : Membership.mem (RingHom.ker (Algebra.TensorProduct.map (AlgHom.id R T) (I …
      ⊢ Membership.mem (Ideal.span (Set.range fun i => (MvPolynomial.map (algebraMap …
    -/
    rw [H] at H'
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      hx : Eq ((MvPolynomial.aeval P.baseChange.val) x) 0
      H : Eq (RingHom.ker (Algebra.TensorProduct.map (AlgHom.id R T) (IsScalarTower. …
      e : AlgEquiv T (TensorProduct R T (MvPolynomial P.vars R)) (MvPolynomial P.var …
      H' : Membership.mem (Ideal.map Algebra.TensorProduct.includeRight (RingHom.ker …
      ⊢ Membership.mem (Ideal.span (Set.range fun i => (MvPolynomial.map (algebraMap …
    -/
    replace H' : e.symm x ∈ Ideal.map TensorProduct.includeRight P.ker := H'
    erw [← P.span_range_relation_eq_ker, ← Ideal.mem_comap, Ideal.comap_symm,
      Ideal.map_map, Ideal.map_span, ← Set.range_comp] at H'
    /-
      case a
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      hx : Eq ((MvPolynomial.aeval P.baseChange.val) x) 0
      H : Eq (RingHom.ker (Algebra.TensorProduct.map (AlgHom.id R T) (IsScalarTower. …
      e : AlgEquiv T (TensorProduct R T (MvPolynomial P.vars R)) (MvPolynomial P.var …
      H' : Membership.mem (Ideal.span (Set.range (Function.comp (⇑((Algebra.TensorPr …
      ⊢ Membership.mem (Ideal.span (Set.range fun i => (MvPolynomial.map (algebraMap …
    -/
    convert H'
    simp only [AlgHom.toRingHom_eq_coe, RingHom.coe_comp, RingHom.coe_coe, Function.comp_apply,
      TensorProduct.includeRight_apply, TensorProduct.lift_tmul, map_one, mapAlgHom_apply, one_mul]
    /-
      case h.e'_4.h.e'_3.h.e'_3.h
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_1
      inst✝¹ : CommRing T
      inst✝ : Algebra R T
      P : Algebra.Presentation R S
      x : MvPolynomial P.vars T
      hx : Eq ((MvPolynomial.aeval P.baseChange.val) x) 0
      H : Eq (RingHom.ker (Algebra.TensorProduct.map (AlgHom.id R T) (IsScalarTower. …
      e : AlgEquiv T (TensorProduct R T (MvPolynomial P.vars R)) (MvPolynomial P.var …
      H' : Membership.mem (Ideal.span (Set.range (Function.comp (⇑((Algebra.TensorPr …
      x✝ : P.rels
      ⊢ Eq ((MvPolynomial.map (algebraMap R T)) (P.relation x✝)) (MvPolynomial.eval₂ …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- If `P` is a presentation of `S` over `R` and `T` is an `R`-algebra, we
obtain a natural presentation of `T ⊗[R] S` over `T`. -/
@[simps relation, simps (config := .lemmasOnly) rels]
noncomputable
def baseChange : Presentation T (T ⊗[R] S) where
  __ := Generators.baseChange P.toGenerators
  rels := P.rels
  relation i := MvPolynomial.map (algebraMap R T) (P.relation i)
  span_range_relation_eq_ker := P.span_range_relation_eq_ker_baseChange T


instance baseChange_isFinite [P.IsFinite] : (P.baseChange T).IsFinite where
  finite_vars := inferInstanceAs <| Finite (P.vars)
  finite_rels := inferInstanceAs <| Finite (P.rels)


/-- The evaluation map `MvPolynomial (Q.vars ⊕ P.vars) →ₐ[R] T` factors via this map. For more
details, see the module docstring at the beginning of the section. -/
private noncomputable def aux : MvPolynomial (Q.vars ⊕ P.vars) R →ₐ[R] MvPolynomial Q.vars S :=
  aeval (Sum.elim X (MvPolynomial.C ∘ P.val))


/-- A choice of pre-image of `Q.relation r` under `aux`. -/
private noncomputable def comp_relation_aux (r : Q.rels) : MvPolynomial (Q.vars ⊕ P.vars) R :=
  Finsupp.sum (Q.relation r)
    (fun x j ↦ (MvPolynomial.rename Sum.inr <| P.σ j) * monomial (x.mapDomain Sum.inl) 1)


@[simp]
private lemma aux_X (i : Q.vars ⊕ P.vars) : (Q.aux P) (X i) = Sum.elim X (C ∘ P.val) i :=
  aeval_X (Sum.elim X (C ∘ P.val)) i


/-- The pre-images constructed in `comp_relation_aux` are indeed pre-images under `aux`. -/
private lemma comp_relation_aux_map (r : Q.rels) :
    (Q.aux P) (Q.comp_relation_aux P r) = Q.relation r := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    r : Q.rels
    ⊢ Eq ((Algebra.Presentation.aux Q P) (Algebra.Presentation.comp_relation_aux Q …
  -/
  simp only [aux, comp_relation_aux, Generators.comp_vars, Sum.elim_inl, map_finsupp_sum]
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    r : Q.rels
    ⊢ Eq (Finsupp.sum (Q.relation r) fun a b => (MvPolynomial.aeval (Sum.elim MvPo …
  -/
  simp only [_root_.map_mul, aeval_rename, aeval_monomial, Sum.elim_comp_inr]
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    r : Q.rels
    ⊢ Eq (Finsupp.sum (Q.relation r) fun a b => HMul.hMul ((MvPolynomial.aeval (Fu …
  -/
  conv_rhs => rw [← Finsupp.sum_single (Q.relation r)]
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    r : Q.rels
    ⊢ Eq (Finsupp.sum (Q.relation r) fun a b => HMul.hMul ((MvPolynomial.aeval (Fu …
  -/
  congr
  /-
    case e_g
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    r : Q.rels
    ⊢ Eq (fun a b => HMul.hMul ((MvPolynomial.aeval (Function.comp (⇑MvPolynomial. …
  -/
  ext u s m
  /-
    case e_g.h.h.a
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    r : Q.rels
    u : Finsupp Q.vars Nat
    s : S
    m : Finsupp Q.vars Nat
    ⊢ Eq (MvPolynomial.coeff m (HMul.hMul ((MvPolynomial.aeval (Function.comp (⇑Mv …
  -/
  simp only [MvPolynomial.single_eq_monomial, aeval, AlgHom.coe_mk, coe_eval₂Hom]
  /-
    case e_g.h.h.a
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    r : Q.rels
    u : Finsupp Q.vars Nat
    s : S
    m : Finsupp Q.vars Nat
    ⊢ Eq (MvPolynomial.coeff m (HMul.hMul (MvPolynomial.eval₂ (algebraMap R (MvPol …
  -/
  rw [monomial_eq, IsScalarTower.algebraMap_eq R S, algebraMap_eq, ← eval₂_comp_left, ← aeval_def]
  /-
    case e_g.h.h.a
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    r : Q.rels
    u : Finsupp Q.vars Nat
    s : S
    m : Finsupp Q.vars Nat
    ⊢ Eq (MvPolynomial.coeff m (HMul.hMul (MvPolynomial.C ((MvPolynomial.aeval P.v …
  -/
  simp [Finsupp.prod_mapDomain_index_inj (Sum.inl_injective)]
  /-
    🎉 no goals
  -/


private lemma aux_surjective : Function.Surjective (Q.aux P) := fun p ↦ by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    p : MvPolynomial Q.vars S
    ⊢ Exists fun a => Eq ((Algebra.Presentation.aux Q P) a) p
  -/
  induction' p using MvPolynomial.induction_on with a p q hp hq p i h
    /-
      case h_C
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_3
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      a : S
      ⊢ Exists fun a_1 => Eq ((Algebra.Presentation.aux Q P) a_1) (MvPolynomial.C a)
    -/
  · use rename Sum.inr <| P.σ a
    /-
      case h
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_3
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      a : S
      ⊢ Eq ((Algebra.Presentation.aux Q P) ((MvPolynomial.rename Sum.inr) (P.σ a)))  …
    -/
    simp only [aux, aeval_rename, Sum.elim_comp_inr]
    have (p : MvPolynomial P.vars R) :
        aeval (C ∘ P.val) p = (C (aeval P.val p) : MvPolynomial Q.vars S) := by
      induction' p using MvPolynomial.induction_on with a p q hp hq p i h
      · simp
      · simp [hp, hq]
      · simp [h]
    /-
      case h
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_3
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      a : S
      this : ∀ (p : MvPolynomial P.vars R), Eq ((MvPolynomial.aeval (Function.comp ( …
      ⊢ Eq ((MvPolynomial.aeval (Function.comp (⇑MvPolynomial.C) P.val)) (P.σ a)) (M …
    -/
    simp [this]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_3
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      p q : MvPolynomial Q.vars S
      hp : Exists fun a => Eq ((Algebra.Presentation.aux Q P) a) p
      hq : Exists fun a => Eq ((Algebra.Presentation.aux Q P) a) q
      ⊢ Exists fun a => Eq ((Algebra.Presentation.aux Q P) a) (HAdd.hAdd p q)
    -/
  · obtain ⟨a, rfl⟩ := hp
    /-
      case h_add.intro
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_3
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      q : MvPolynomial Q.vars S
      hq : Exists fun a => Eq ((Algebra.Presentation.aux Q P) a) q
      a : MvPolynomial (Sum Q.vars P.vars) R
      ⊢ Exists fun a_1 => Eq ((Algebra.Presentation.aux Q P) a_1) (HAdd.hAdd ((Algeb …
    -/
    obtain ⟨b, rfl⟩ := hq
    /-
      case h_add.intro.intro
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_3
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      a b : MvPolynomial (Sum Q.vars P.vars) R
      ⊢ Exists fun a_1 => Eq ((Algebra.Presentation.aux Q P) a_1) (HAdd.hAdd ((Algeb …
    -/
    exact ⟨a + b, map_add _ _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case h_X
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_3
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      p : MvPolynomial Q.vars S
      i : Q.vars
      h : Exists fun a => Eq ((Algebra.Presentation.aux Q P) a) p
      ⊢ Exists fun a => Eq ((Algebra.Presentation.aux Q P) a) (HMul.hMul p (MvPolyno …
    -/
  · obtain ⟨a, rfl⟩ := h
    /-
      case h_X.intro
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_3
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      i : Q.vars
      a : MvPolynomial (Sum Q.vars P.vars) R
      ⊢ Exists fun a_1 => Eq ((Algebra.Presentation.aux Q P) a_1) (HMul.hMul ((Algeb …
    -/
    exact ⟨(a * X (Sum.inl i)), by simp⟩
    /-
      🎉 no goals
    -/


private lemma aux_image_relation :
    Q.aux P '' (Set.range (Algebra.Presentation.comp_relation_aux Q P)) = Set.range Q.relation := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_2
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    ⊢ Eq (Set.image (⇑(Algebra.Presentation.aux Q P)) (Set.range (Algebra.Presenta …
  -/
  ext x
  /-
    case h
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_2
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    x : MvPolynomial Q.vars S
    ⊢ Iff (Membership.mem (Set.image (⇑(Algebra.Presentation.aux Q P)) (Set.range  …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_2
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      x : MvPolynomial Q.vars S
      ⊢ Membership.mem (Set.image (⇑(Algebra.Presentation.aux Q P)) (Set.range (Alge …
    -/
  · rintro ⟨y, ⟨a, rfl⟩, rfl⟩
    /-
      case h.mp.intro.intro.intro
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_2
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      a : Q.rels
      ⊢ Membership.mem (Set.range Q.relation) ((Algebra.Presentation.aux Q P) (Algeb …
    -/
    exact ⟨a, (Q.comp_relation_aux_map P a).symm⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_2
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      x : MvPolynomial Q.vars S
      ⊢ Membership.mem (Set.range Q.relation) x → Membership.mem (Set.image (⇑(Algeb …
    -/
  · rintro ⟨y, rfl⟩
    /-
      case h.mpr.intro
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_2
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      y : Q.rels
      ⊢ Membership.mem (Set.image (⇑(Algebra.Presentation.aux Q P)) (Set.range (Alge …
    -/
    use Q.comp_relation_aux P y
    /-
      case h
      R : Type u
      S : Type v
      inst✝⁴ : CommRing R
      inst✝³ : CommRing S
      inst✝² : Algebra R S
      T : Type u_2
      inst✝¹ : CommRing T
      inst✝ : Algebra S T
      Q : Algebra.Presentation S T
      P : Algebra.Presentation R S
      y : Q.rels
      ⊢ And (Membership.mem (Set.range (Algebra.Presentation.comp_relation_aux Q P)) …
    -/
    simp only [Set.mem_range, exists_apply_eq_apply, true_and, comp_relation_aux_map]
    /-
      🎉 no goals
    -/


private lemma aux_eq_comp : Q.aux P =
    (MvPolynomial.mapAlgHom (aeval P.val)).comp (sumAlgEquiv R Q.vars P.vars).toAlgHom := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    ⊢ Eq (Algebra.Presentation.aux Q P) ((MvPolynomial.mapAlgHom (MvPolynomial.aev …
  -/
  ext i : 1
  /-
    case hf
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    i : Sum Q.vars P.vars
    ⊢ Eq ((Algebra.Presentation.aux Q P) (MvPolynomial.X i)) (((MvPolynomial.mapAl …
  -/
              /-
                🎉 no goals
              -/
  cases i <;> simp
              /-
                🎉 no goals
              -/


private lemma aux_ker :
    RingHom.ker (Q.aux P) = Ideal.map (rename Sum.inr) (RingHom.ker (aeval P.val)) := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    ⊢ Eq (RingHom.ker (Algebra.Presentation.aux Q P)) (Ideal.map (MvPolynomial.ren …
  -/
  rw [aux_eq_comp, ← AlgHom.comap_ker, MvPolynomial.ker_mapAlgHom]
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    ⊢ Eq (Ideal.comap (↑(MvPolynomial.sumAlgEquiv R Q.vars P.vars)) (Ideal.map MvP …
  -/
  show Ideal.comap _ (Ideal.map (IsScalarTower.toAlgHom R (MvPolynomial P.vars R) _) _) = _
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    ⊢ Eq (Ideal.comap (↑(MvPolynomial.sumAlgEquiv R Q.vars P.vars)) (Ideal.map (Is …
  -/
  rw [← sumAlgEquiv_comp_rename_inr, ← Ideal.map_mapₐ, Ideal.comap_map_of_bijective]
  /-
    case hf
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    T : Type u_3
    inst✝¹ : CommRing T
    inst✝ : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    ⊢ Function.Bijective ⇑↑(MvPolynomial.sumAlgEquiv R Q.vars P.vars)
  -/
  simpa using AlgEquiv.bijective (sumAlgEquiv R Q.vars P.vars)
  /-
    🎉 no goals
  -/


private lemma aeval_comp_val_eq :
    (aeval (Q.comp P.toGenerators).val) =
      (aevalTower (IsScalarTower.toAlgHom R S T) Q.val).comp (Q.aux P) := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_1
    inst✝³ : CommRing T
    inst✝² : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    inst✝¹ : Algebra R T
    inst✝ : IsScalarTower R S T
    ⊢ Eq (MvPolynomial.aeval (Q.comp P.toGenerators).val) ((MvPolynomial.aevalTowe …
  -/
  ext i
  /-
    case hf
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_1
    inst✝³ : CommRing T
    inst✝² : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    inst✝¹ : Algebra R T
    inst✝ : IsScalarTower R S T
    i : (Q.comp P.toGenerators).vars
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P.toGenerators).val) (MvPolynomial.X i)) ((( …
  -/
  simp only [AlgHom.coe_comp, Function.comp_apply]
  /-
    case hf
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_1
    inst✝³ : CommRing T
    inst✝² : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    inst✝¹ : Algebra R T
    inst✝ : IsScalarTower R S T
    i : (Q.comp P.toGenerators).vars
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P.toGenerators).val) (MvPolynomial.X i)) ((M …
  -/
  erw [Q.aux_X P i]
  /-
    case hf
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_1
    inst✝³ : CommRing T
    inst✝² : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    inst✝¹ : Algebra R T
    inst✝ : IsScalarTower R S T
    i : (Q.comp P.toGenerators).vars
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P.toGenerators).val) (MvPolynomial.X i)) ((M …
  -/
              /-
                🎉 no goals
              -/
  cases i <;> simp
              /-
                🎉 no goals
              -/


private lemma span_range_relation_eq_ker_comp : Ideal.span
    (Set.range (Sum.elim (Algebra.Presentation.comp_relation_aux Q P)
      fun rp ↦ (rename Sum.inr) (P.relation rp))) = (Q.comp P.toGenerators).ker := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_3
    inst✝³ : CommRing T
    inst✝² : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    inst✝¹ : Algebra R T
    inst✝ : IsScalarTower R S T
    ⊢ Eq (Ideal.span (Set.range (Sum.elim (Algebra.Presentation.comp_relation_aux  …
  -/
  rw [Generators.ker_eq_ker_aeval_val, Q.aeval_comp_val_eq, ← AlgHom.comap_ker]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_3
    inst✝³ : CommRing T
    inst✝² : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    inst✝¹ : Algebra R T
    inst✝ : IsScalarTower R S T
    ⊢ Eq (Ideal.span (Set.range (Sum.elim (Algebra.Presentation.comp_relation_aux  …
  -/
  show _ = Ideal.comap _ (RingHom.ker (aeval Q.val))
  rw [← Q.ker_eq_ker_aeval_val, ← Q.span_range_relation_eq_ker, ← Q.aux_image_relation P,
    ← Ideal.map_span, Ideal.comap_map_of_surjective' _ (Q.aux_surjective P)]
  rw [Set.Sum.elim_range, Ideal.span_union, Q.aux_ker, ← P.ker_eq_ker_aeval_val,
    ← P.span_range_relation_eq_ker, Ideal.map_span]
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_3
    inst✝³ : CommRing T
    inst✝² : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    inst✝¹ : Algebra R T
    inst✝ : IsScalarTower R S T
    ⊢ Eq (Max.max (Ideal.span (Set.range (Algebra.Presentation.comp_relation_aux Q …
  -/
  congr
  /-
    case e_a.e_s
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_3
    inst✝³ : CommRing T
    inst✝² : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    inst✝¹ : Algebra R T
    inst✝ : IsScalarTower R S T
    ⊢ Eq (Set.range fun rp => (MvPolynomial.rename Sum.inr) (P.relation rp)) (Set. …
  -/
  ext
  /-
    case e_a.e_s.h
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_3
    inst✝³ : CommRing T
    inst✝² : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    inst✝¹ : Algebra R T
    inst✝ : IsScalarTower R S T
    x✝ : MvPolynomial (Sum Q.vars P.vars) R
    ⊢ Iff (Membership.mem (Set.range fun rp => (MvPolynomial.rename Sum.inr) (P.re …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given presentations of `T` over `S` and of `S` over `R`,
we may construct a presentation of `T` over `R`. -/
@[simps rels, simps (config := .lemmasOnly) relation]
noncomputable def comp : Presentation R T where
  toGenerators := Q.toGenerators.comp P.toGenerators
  rels := Q.rels ⊕ P.rels
  relation := Sum.elim (Q.comp_relation_aux P)
    (fun rp ↦ MvPolynomial.rename Sum.inr <| P.relation rp)
  span_range_relation_eq_ker := Q.span_range_relation_eq_ker_comp P


@[simp]
lemma comp_relation_inr (r : P.rels) :
    (Q.comp P).relation (Sum.inr r) = rename Sum.inr (P.relation r) :=
  rfl


lemma comp_aeval_relation_inl (r : Q.rels) :
    aeval (Sum.elim X (MvPolynomial.C ∘ P.val)) ((Q.comp P).relation (Sum.inl r)) =
      Q.relation r := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_3
    inst✝³ : CommRing T
    inst✝² : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    inst✝¹ : Algebra R T
    inst✝ : IsScalarTower R S T
    r : Q.rels
    ⊢ Eq ((MvPolynomial.aeval (Sum.elim MvPolynomial.X (Function.comp (⇑MvPolynomi …
  -/
  show (Q.aux P) _ = _
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : Algebra R S
    T : Type u_3
    inst✝³ : CommRing T
    inst✝² : Algebra S T
    Q : Algebra.Presentation S T
    P : Algebra.Presentation R S
    inst✝¹ : Algebra R T
    inst✝ : IsScalarTower R S T
    r : Q.rels
    ⊢ Eq ((Algebra.Presentation.aux Q P) ((Q.comp P).relation (Sum.inl r))) (Q.rel …
  -/
  simp [comp_relation, comp_relation_aux_map]
  /-
    🎉 no goals
  -/


instance comp_isFinite [P.IsFinite] [Q.IsFinite] : (Q.comp P).IsFinite where
  finite_vars := inferInstanceAs <| Finite (Q.vars ⊕ P.vars)
  finite_rels := inferInstanceAs <| Finite (Q.rels ⊕ P.rels)


