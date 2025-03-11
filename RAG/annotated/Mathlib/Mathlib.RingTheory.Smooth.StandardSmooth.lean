/--
A `PreSubmersivePresentation` of an `R`-algebra `S` is a `Presentation`
with finitely-many relations equipped with an injective `map : relations → vars`.

This map determines how the differential of `P` is constructed. See
`PreSubmersivePresentation.differential` for details.
-/
@[nolint checkUnivs]
structure PreSubmersivePresentation extends Algebra.Presentation.{t, w} R S where
  /-- A map from the relations type to the variables type. Used to compute the differential. -/
  map : rels → vars
  map_inj : Function.Injective map
  relations_finite : Finite rels


lemma card_relations_le_card_vars_of_isFinite [P.IsFinite] :
    Nat.card P.rels ≤ Nat.card P.vars :=
  Nat.card_le_card_of_injective P.map P.map_inj


/-- The standard basis of `P.rels → P.ring`. -/
noncomputable abbrev basis : Basis P.rels P.Ring (P.rels → P.Ring) :=
  Pi.basisFun P.Ring P.rels


/--
The differential of a `P : PreSubmersivePresentation` is a `P.Ring`-linear map on
`P.rels → P.Ring`:

The `j`-th standard basis vector, corresponding to the `j`-th relation of `P`, is mapped
to the vector of partial derivatives of `P.relation j` with respect
to the coordinates `P.map i` for all `i : P.rels`.

The determinant of this map is the jacobian of `P` used to define when a `PreSubmersivePresentation`
is submersive. See `PreSubmersivePresentation.jacobian`.
-/
noncomputable def differential : (P.rels → P.Ring) →ₗ[P.Ring] (P.rels → P.Ring) :=
  Basis.constr P.basis P.Ring
    (fun j i : P.rels ↦ MvPolynomial.pderiv (P.map i) (P.relation j))


/-- The jacobian of a `P : PreSubmersivePresentation` is the determinant
of `P.differential` viewed as element of `S`. -/
noncomputable def jacobian : S :=
  algebraMap P.Ring S <| LinearMap.det P.differential


/--
If `P.rels` has a `Fintype` and `DecidableEq` instance, the differential of `P`
can be expressed in matrix form.
-/
noncomputable def jacobiMatrix : Matrix P.rels P.rels P.Ring :=
  LinearMap.toMatrix P.basis P.basis P.differential


lemma jacobian_eq_jacobiMatrix_det : P.jacobian = algebraMap P.Ring S P.jacobiMatrix.det := by
   /-
     R : Type u
     S : Type v
     inst✝⁴ : CommRing R
     inst✝³ : CommRing S
     inst✝² : Algebra R S
     P : Algebra.PreSubmersivePresentation R S
     inst✝¹ : Fintype P.rels
     inst✝ : DecidableEq P.rels
     ⊢ Eq P.jacobian ((algebraMap P.Ring S) P.jacobiMatrix.det)
   -/
   simp [jacobiMatrix, jacobian]
   /-
     🎉 no goals
   -/


lemma jacobiMatrix_apply (i j : P.rels) :
    P.jacobiMatrix i j = MvPolynomial.pderiv (P.map i) (P.relation j) := by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype P.rels
    inst✝ : DecidableEq P.rels
    i j : P.rels
    ⊢ Eq (P.jacobiMatrix i j) ((MvPolynomial.pderiv (P.map i)) (P.relation j))
  -/
  simp [jacobiMatrix, LinearMap.toMatrix, differential, basis]
  /-
    🎉 no goals
  -/


/-- If `algebraMap R S` is bijective, the empty generators are a pre-submersive
presentation with no relations. -/
noncomputable def ofBijectiveAlgebraMap (h : Function.Bijective (algebraMap R S)) :
    PreSubmersivePresentation.{t, w} R S where
  toPresentation := Presentation.ofBijectiveAlgebraMap.{t, w} h
  map := PEmpty.elim
                                 /-
                                   n m : Nat
                                   R : Type u
                                   S : Type v
                                   inst✝² : CommRing R
                                   inst✝¹ : CommRing S
                                   inst✝ : Algebra R S
                                   P : Algebra.PreSubmersivePresentation R S
                                   h✝ : Function.Bijective ⇑(algebraMap R S)
                                   a b : PEmpty.{t + 1}
                                   h : Eq a.elim b.elim
                                   ⊢ Eq a b
                                 -/
  map_inj (a b : PEmpty) h := by contradiction
                                 /-
                                   🎉 no goals
                                 -/
  relations_finite := inferInstanceAs (Finite PEmpty.{t + 1})


instance (h : Function.Bijective (algebraMap R S)) : Fintype (ofBijectiveAlgebraMap h).vars :=
  inferInstanceAs (Fintype PEmpty)


instance (h : Function.Bijective (algebraMap R S)) : Fintype (ofBijectiveAlgebraMap h).rels :=
  inferInstanceAs (Fintype PEmpty)


@[simp]
lemma ofBijectiveAlgebraMap_jacobian (h : Function.Bijective (algebraMap R S)) :
    (ofBijectiveAlgebraMap h).jacobian = 1 := by
  classical
  have : (algebraMap (ofBijectiveAlgebraMap h).Ring S).mapMatrix
      (ofBijectiveAlgebraMap h).jacobiMatrix = 1 := by
    ext (i j : PEmpty)
    contradiction
  rw [jacobian_eq_jacobiMatrix_det, RingHom.map_det, this, Matrix.det_one]


variable (S) in
/-- If `S` is the localization of `R` at `r`, this is the canonical submersive presentation
of `S` as `R`-algebra. -/
@[simps map]
noncomputable def localizationAway : PreSubmersivePresentation R S where
  __ := Presentation.localizationAway S r
  map _ := ()
  map_inj _ _ h := h
  relations_finite := inferInstanceAs <| Finite Unit


instance : Fintype (localizationAway S r).rels :=
  inferInstanceAs (Fintype Unit)


instance : DecidableEq (localizationAway S r).rels :=
  inferInstanceAs (DecidableEq Unit)


@[simp]
lemma localizationAway_jacobiMatrix :
    (localizationAway S r).jacobiMatrix = Matrix.diagonal (fun () ↦ MvPolynomial.C r) := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Eq (Algebra.PreSubmersivePresentation.localizationAway S r).jacobiMatrix (Ma …
  -/
  have h : (pderiv ()) (C r * X () - 1) = C r := by simp
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    h : Eq ((MvPolynomial.pderiv Unit.unit) (HSub.hSub (HMul.hMul (MvPolynomial.C  …
    ⊢ Eq (Algebra.PreSubmersivePresentation.localizationAway S r).jacobiMatrix (Ma …
  -/
  ext (i : Unit) (j : Unit) : 1
  /-
    case a
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    h : Eq ((MvPolynomial.pderiv Unit.unit) (HSub.hSub (HMul.hMul (MvPolynomial.C  …
    i j : Unit
    ⊢ Eq ((Algebra.PreSubmersivePresentation.localizationAway S r).jacobiMatrix i  …
  -/
  rwa [jacobiMatrix_apply]
  /-
    🎉 no goals
  -/


@[simp]
lemma localizationAway_jacobian : (localizationAway S r).jacobian = algebraMap R S r := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Eq (Algebra.PreSubmersivePresentation.localizationAway S r).jacobian ((algeb …
  -/
  rw [jacobian_eq_jacobiMatrix_det, localizationAway_jacobiMatrix]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    r : R
    inst✝ : IsLocalization.Away r S
    ⊢ Eq ((algebraMap (Algebra.PreSubmersivePresentation.localizationAway S r).Rin …
  -/
  simp [show Fintype.card (localizationAway r (S := S)).rels = 1 from rfl]
  /-
    🎉 no goals
  -/


/-- Given an `R`-algebra `S` and an `S`-algebra `T` with pre-submersive presentations,
this is the canonical pre-submersive presentation of `T` as an `R`-algebra. -/
@[simps map]
noncomputable def comp : PreSubmersivePresentation R T where
  __ := Q.toPresentation.comp P.toPresentation
  map := Sum.elim (fun rq ↦ Sum.inl <| Q.map rq) (fun rp ↦ Sum.inr <| P.map rp)
  map_inj := Function.Injective.sum_elim ((Sum.inl_injective).comp (Q.map_inj))
                                                 /-
                                                   n m : Nat
                                                   R : Type u
                                                   S : Type v
                                                   inst✝⁶ : CommRing R
                                                   inst✝⁵ : CommRing S
                                                   inst✝⁴ : Algebra R S
                                                   P✝ : Algebra.PreSubmersivePresentation R S
                                                   T : Type ?u.46737
                                                   inst✝³ : CommRing T
                                                   inst✝² : Algebra R T
                                                   inst✝¹ : Algebra S T
                                                   inst✝ : IsScalarTower R S T
                                                   Q : Algebra.PreSubmersivePresentation S T
                                                   P : Algebra.PreSubmersivePresentation R S
                                                   ⊢ ∀ (a : Q.rels) (b : P.rels), Ne (Sum.inl (Q.map a)) (Sum.inr (P.map b))
                                                 -/
    ((Sum.inr_injective).comp (P.map_inj)) <| by simp
                                                 /-
                                                   🎉 no goals
                                                 -/
  relations_finite := inferInstanceAs <| Finite (Q.rels ⊕ P.rels)


/-- The dimension of the composition of two finite submersive presentations is
the sum of the dimensions. -/
lemma dimension_comp_eq_dimension_add_dimension [Q.IsFinite] [P.IsFinite] :
    (Q.comp P).dimension = Q.dimension + P.dimension := by
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_3
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Q.IsFinite
    inst✝ : P.IsFinite
    ⊢ Eq (Q.comp P).dimension (HAdd.hAdd Q.dimension P.dimension)
  -/
  simp only [Presentation.dimension]
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_3
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Q.IsFinite
    inst✝ : P.IsFinite
    ⊢ Eq (HSub.hSub (Nat.card (Q.comp P).vars) (Nat.card (Q.comp P).rels)) (HAdd.h …
  -/
  erw [Presentation.comp_rels, Generators.comp_vars]
  have : Nat.card P.rels ≤ Nat.card P.vars :=
    card_relations_le_card_vars_of_isFinite P
  have : Nat.card Q.rels ≤ Nat.card Q.vars :=
    card_relations_le_card_vars_of_isFinite Q
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_3
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Q.IsFinite
    inst✝ : P.IsFinite
    this✝ : LE.le (Nat.card P.rels) (Nat.card P.vars)
    this : LE.le (Nat.card Q.rels) (Nat.card Q.vars)
    ⊢ Eq (HSub.hSub (Nat.card (Sum Q.vars P.vars)) (Nat.card (Sum Q.rels P.rels))) …
  -/
  simp only [Nat.card_sum]
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_3
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Q.IsFinite
    inst✝ : P.IsFinite
    this✝ : LE.le (Nat.card P.rels) (Nat.card P.vars)
    this : LE.le (Nat.card Q.rels) (Nat.card Q.vars)
    ⊢ Eq (HSub.hSub (HAdd.hAdd (Nat.card Q.vars) (Nat.card P.vars)) (HAdd.hAdd (Na …
  -/
  omega
  /-
    🎉 no goals
  -/


open scoped Classical in
private lemma jacobiMatrix_comp_inl_inr (i : Q.rels) (j : P.rels) :
    (Q.comp P).jacobiMatrix (Sum.inl i) (Sum.inr j) = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    T : Type u_3
    inst✝⁴ : CommRing T
    inst✝³ : Algebra R T
    inst✝² : Algebra S T
    inst✝¹ : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝ : Fintype (Q.comp P).rels
    i : Q.rels
    j : P.rels
    ⊢ Eq ((Q.comp P).jacobiMatrix (Sum.inl i) (Sum.inr j)) 0
  -/
  rw [jacobiMatrix_apply]
  /-
    R : Type u
    S : Type v
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    T : Type u_3
    inst✝⁴ : CommRing T
    inst✝³ : Algebra R T
    inst✝² : Algebra S T
    inst✝¹ : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝ : Fintype (Q.comp P).rels
    i : Q.rels
    j : P.rels
    ⊢ Eq ((MvPolynomial.pderiv ((Q.comp P).map (Sum.inl i))) ((Q.comp P).relation  …
  -/
  refine MvPolynomial.pderiv_eq_zero_of_not_mem_vars (fun hmem ↦ ?_)
  /-
    R : Type u
    S : Type v
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    T : Type u_3
    inst✝⁴ : CommRing T
    inst✝³ : Algebra R T
    inst✝² : Algebra S T
    inst✝¹ : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝ : Fintype (Q.comp P).rels
    i : Q.rels
    j : P.rels
    hmem : Membership.mem (MvPolynomial.vars ((Q.comp P).relation (Sum.inr j))) (( …
    ⊢ False
  -/
  apply MvPolynomial.vars_rename at hmem
  /-
    R : Type u
    S : Type v
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    T : Type u_3
    inst✝⁴ : CommRing T
    inst✝³ : Algebra R T
    inst✝² : Algebra S T
    inst✝¹ : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝ : Fintype (Q.comp P).rels
    i : Q.rels
    j : P.rels
    hmem : Membership.mem (Finset.image Sum.inr (MvPolynomial.vars (P.relation j)) …
    ⊢ False
  -/
  simp at hmem
  /-
    🎉 no goals
  -/


open scoped Classical in
private lemma jacobiMatrix_comp_₁₂ : (Q.comp P).jacobiMatrix.toBlocks₁₂ = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    T : Type u_5
    inst✝⁴ : CommRing T
    inst✝³ : Algebra R T
    inst✝² : Algebra S T
    inst✝¹ : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝ : Fintype (Q.comp P).rels
    ⊢ Eq (Q.comp P).jacobiMatrix.toBlocks₁₂ 0
  -/
  ext i j : 1
  /-
    case a
    R : Type u
    S : Type v
    inst✝⁷ : CommRing R
    inst✝⁶ : CommRing S
    inst✝⁵ : Algebra R S
    T : Type u_5
    inst✝⁴ : CommRing T
    inst✝³ : Algebra R T
    inst✝² : Algebra S T
    inst✝¹ : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝ : Fintype (Q.comp P).rels
    i : Q.rels
    j : P.rels
    ⊢ Eq ((Q.comp P).jacobiMatrix.toBlocks₁₂ i j) (0 i j)
  -/
  simp [Matrix.toBlocks₁₂, jacobiMatrix_comp_inl_inr]
  /-
    🎉 no goals
  -/


open scoped Classical in
private lemma jacobiMatrix_comp_inl_inl (i j : Q.rels) :
    aeval (Sum.elim X (MvPolynomial.C ∘ P.val))
      ((Q.comp P).jacobiMatrix (Sum.inl j) (Sum.inl i)) = Q.jacobiMatrix j i := by
  rw [jacobiMatrix_apply, jacobiMatrix_apply, comp_map, Sum.elim_inl,
    ← Q.comp_aeval_relation_inl P.toPresentation]
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_3
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype Q.rels
    i j : Q.rels
    ⊢ Eq ((MvPolynomial.aeval (Sum.elim MvPolynomial.X (Function.comp (⇑MvPolynomi …
  -/
  apply aeval_sum_elim_pderiv_inl
  /-
    🎉 no goals
  -/


open scoped Classical in
private lemma jacobiMatrix_comp_₁₁_det :
    (aeval (Q.comp P).val) (Q.comp P).jacobiMatrix.toBlocks₁₁.det = Q.jacobian := by
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype Q.rels
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val) (Q.comp P).jacobiMatrix.toBlocks₁₁.d …
  -/
  rw [jacobian_eq_jacobiMatrix_det, AlgHom.map_det (aeval (Q.comp P).val), RingHom.map_det]
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype Q.rels
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val).mapMatrix (Q.comp P).jacobiMatrix.to …
  -/
  congr
  /-
    case e_M
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype Q.rels
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val).mapMatrix (Q.comp P).jacobiMatrix.to …
  -/
  ext i j : 1
  simp only [Matrix.map_apply, RingHom.mapMatrix_apply, ← Q.jacobiMatrix_comp_inl_inl P,
    Q.algebraMap_apply]
  /-
    case e_M.a
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype Q.rels
    i j : Q.rels
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val).mapMatrix (Q.comp P).jacobiMatrix.to …
  -/
  apply aeval_sum_elim
  /-
    🎉 no goals
  -/


open scoped Classical in
private lemma jacobiMatrix_comp_inr_inr (i j : P.rels) :
    (Q.comp P).jacobiMatrix (Sum.inr i) (Sum.inr j) =
      MvPolynomial.rename Sum.inr (P.jacobiMatrix i j) := by
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_4
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype P.rels
    i j : P.rels
    ⊢ Eq ((Q.comp P).jacobiMatrix (Sum.inr i) (Sum.inr j)) ((MvPolynomial.rename S …
  -/
  rw [jacobiMatrix_apply, jacobiMatrix_apply]
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_4
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype P.rels
    i j : P.rels
    ⊢ Eq ((MvPolynomial.pderiv ((Q.comp P).map (Sum.inr i))) ((Q.comp P).relation  …
  -/
  simp only [comp_map, Sum.elim_inr]
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_4
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype P.rels
    i j : P.rels
    ⊢ Eq ((MvPolynomial.pderiv (Sum.inr (P.map i))) ((Q.comp P).relation (Sum.inr  …
  -/
  apply pderiv_rename Sum.inr_injective
  /-
    🎉 no goals
  -/


open scoped Classical in
private lemma jacobiMatrix_comp_₂₂_det :
    (aeval (Q.comp P).val) (Q.comp P).jacobiMatrix.toBlocks₂₂.det = algebraMap S T P.jacobian := by
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype P.rels
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val) (Q.comp P).jacobiMatrix.toBlocks₂₂.d …
  -/
  rw [jacobian_eq_jacobiMatrix_det]
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype P.rels
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val) (Q.comp P).jacobiMatrix.toBlocks₂₂.d …
  -/
  rw [AlgHom.map_det (aeval (Q.comp P).val), RingHom.map_det, RingHom.map_det]
  /-
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype P.rels
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val).mapMatrix (Q.comp P).jacobiMatrix.to …
  -/
  congr
  /-
    case e_M
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype P.rels
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val).mapMatrix (Q.comp P).jacobiMatrix.to …
  -/
  ext i j : 1
  simp only [Matrix.toBlocks₂₂, AlgHom.mapMatrix_apply, Matrix.map_apply, Matrix.of_apply,
    RingHom.mapMatrix_apply, Generators.algebraMap_apply, map_aeval, coe_eval₂Hom]
  /-
    case e_M.a
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype P.rels
    i j : P.rels
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val) ((Q.comp P).jacobiMatrix (Sum.inr i) …
  -/
  rw [jacobiMatrix_comp_inr_inr, ← IsScalarTower.algebraMap_eq]
  /-
    case e_M.a
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype P.rels
    i j : P.rels
    ⊢ Eq ((MvPolynomial.aeval (Q.comp P).val) ((MvPolynomial.rename Sum.inr) (P.ja …
  -/
  simp only [aeval, AlgHom.coe_mk, coe_eval₂Hom]
  /-
    case e_M.a
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype P.rels
    i j : P.rels
    ⊢ Eq (MvPolynomial.eval₂ (algebraMap R T) (Q.comp P).val ((MvPolynomial.rename …
  -/
  generalize P.jacobiMatrix i j = p
  /-
    case e_M.a
    R : Type u
    S : Type v
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing S
    inst✝⁶ : Algebra R S
    T : Type u_1
    inst✝⁵ : CommRing T
    inst✝⁴ : Algebra R T
    inst✝³ : Algebra S T
    inst✝² : IsScalarTower R S T
    Q : Algebra.PreSubmersivePresentation S T
    P : Algebra.PreSubmersivePresentation R S
    inst✝¹ : Fintype (Q.comp P).rels
    inst✝ : Fintype P.rels
    i j : P.rels
    p : P.Ring
    ⊢ Eq (MvPolynomial.eval₂ (algebraMap R T) (Q.comp P).val ((MvPolynomial.rename …
  -/
  induction' p using MvPolynomial.induction_on with a p q hp hq p i hp
    /-
      case e_M.a.h_C
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      Q : Algebra.PreSubmersivePresentation S T
      P : Algebra.PreSubmersivePresentation R S
      inst✝¹ : Fintype (Q.comp P).rels
      inst✝ : Fintype P.rels
      i j : P.rels
      a : R
      ⊢ Eq (MvPolynomial.eval₂ (algebraMap R T) (Q.comp P).val ((MvPolynomial.rename …
    -/
  · simp only [algHom_C, algebraMap_eq, eval₂_C]
    /-
      case e_M.a.h_C
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      Q : Algebra.PreSubmersivePresentation S T
      P : Algebra.PreSubmersivePresentation R S
      inst✝¹ : Fintype (Q.comp P).rels
      inst✝ : Fintype P.rels
      i j : P.rels
      a : R
      ⊢ Eq (MvPolynomial.eval₂ (algebraMap R T) (Q.comp P).val (MvPolynomial.C a)) ( …
    -/
    erw [MvPolynomial.eval₂_C]
    /-
      🎉 no goals
    -/
    /-
      case e_M.a.h_add
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      Q : Algebra.PreSubmersivePresentation S T
      P : Algebra.PreSubmersivePresentation R S
      inst✝¹ : Fintype (Q.comp P).rels
      inst✝ : Fintype P.rels
      i j : P.rels
      p q : MvPolynomial P.vars R
      hp : Eq (MvPolynomial.eval₂ (algebraMap R T) (Q.comp P).val ((MvPolynomial.ren …
      hq : Eq (MvPolynomial.eval₂ (algebraMap R T) (Q.comp P).val ((MvPolynomial.ren …
      ⊢ Eq (MvPolynomial.eval₂ (algebraMap R T) (Q.comp P).val ((MvPolynomial.rename …
    -/
  · simp [hp, hq]
    /-
      🎉 no goals
    -/
    /-
      case e_M.a.h_X
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      Q : Algebra.PreSubmersivePresentation S T
      P : Algebra.PreSubmersivePresentation R S
      inst✝¹ : Fintype (Q.comp P).rels
      inst✝ : Fintype P.rels
      i✝ j : P.rels
      p : MvPolynomial P.vars R
      i : P.vars
      hp : Eq (MvPolynomial.eval₂ (algebraMap R T) (Q.comp P).val ((MvPolynomial.ren …
      ⊢ Eq (MvPolynomial.eval₂ (algebraMap R T) (Q.comp P).val ((MvPolynomial.rename …
    -/
  · simp only [map_mul, rename_X, eval₂_mul, hp, eval₂_X]
    /-
      case e_M.a.h_X
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      Q : Algebra.PreSubmersivePresentation S T
      P : Algebra.PreSubmersivePresentation R S
      inst✝¹ : Fintype (Q.comp P).rels
      inst✝ : Fintype P.rels
      i✝ j : P.rels
      p : MvPolynomial P.vars R
      i : P.vars
      hp : Eq (MvPolynomial.eval₂ (algebraMap R T) (Q.comp P).val ((MvPolynomial.ren …
      ⊢ Eq (HMul.hMul (MvPolynomial.eval₂ (algebraMap R T) (fun i => (algebraMap S T …
    -/
    erw [Generators.comp_val]
    /-
      case e_M.a.h_X
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      Q : Algebra.PreSubmersivePresentation S T
      P : Algebra.PreSubmersivePresentation R S
      inst✝¹ : Fintype (Q.comp P).rels
      inst✝ : Fintype P.rels
      i✝ j : P.rels
      p : MvPolynomial P.vars R
      i : P.vars
      hp : Eq (MvPolynomial.eval₂ (algebraMap R T) (Q.comp P).val ((MvPolynomial.ren …
      ⊢ Eq (HMul.hMul (MvPolynomial.eval₂ (algebraMap R T) (fun i => (algebraMap S T …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The jacobian of the composition of presentations is the product of the jacobians. -/
@[simp]
lemma comp_jacobian_eq_jacobian_smul_jacobian : (Q.comp P).jacobian = P.jacobian • Q.jacobian := by
  classical
  cases nonempty_fintype Q.rels
  cases nonempty_fintype P.rels
  letI : Fintype (Q.comp P).rels := inferInstanceAs <| Fintype (Q.rels ⊕ P.rels)
  rw [jacobian_eq_jacobiMatrix_det, ← Matrix.fromBlocks_toBlocks ((Q.comp P).jacobiMatrix),
    jacobiMatrix_comp_₁₂]
  convert_to
    (aeval (Q.comp P).val) (Q.comp P).jacobiMatrix.toBlocks₁₁.det *
    (aeval (Q.comp P).val) (Q.comp P).jacobiMatrix.toBlocks₂₂.det = P.jacobian • Q.jacobian
  · simp only [Generators.algebraMap_apply, ← map_mul]
    congr
    convert Matrix.det_fromBlocks_zero₁₂ (Q.comp P).jacobiMatrix.toBlocks₁₁
      (Q.comp P).jacobiMatrix.toBlocks₂₁ (Q.comp P).jacobiMatrix.toBlocks₂₂
  · rw [jacobiMatrix_comp_₁₁_det, jacobiMatrix_comp_₂₂_det, mul_comm, Algebra.smul_def]


/-- If `P` is a pre-submersive presentation of `S` over `R` and `T` is an `R`-algebra, we
obtain a natural pre-submersive presentation of `T ⊗[R] S` over `T`. -/
noncomputable def baseChange : PreSubmersivePresentation T (T ⊗[R] S) where
  __ := P.toPresentation.baseChange T
  map := P.map
  map_inj := P.map_inj
  relations_finite := P.relations_finite


@[simp]
lemma baseChange_jacobian : (P.baseChange T).jacobian = 1 ⊗ₜ P.jacobian := by
  classical
  cases nonempty_fintype P.rels
  letI : Fintype (P.baseChange T).rels := inferInstanceAs <| Fintype P.rels
  simp_rw [jacobian_eq_jacobiMatrix_det]
  have h : (baseChange T P).jacobiMatrix =
      (MvPolynomial.map (algebraMap R T)).mapMatrix P.jacobiMatrix := by
    ext i j : 1
    simp only [baseChange, jacobiMatrix_apply, Presentation.baseChange_relation,
      RingHom.mapMatrix_apply, Matrix.map_apply]
    erw [MvPolynomial.pderiv_map]
    rfl
  rw [h]
  erw [← RingHom.map_det, aeval_map_algebraMap]
  rw [P.algebraMap_apply]
  apply aeval_one_tmul


/--
A `PreSubmersivePresentation` is submersive if its jacobian is a unit in `S`
and the presentation is finite.
-/
@[nolint checkUnivs]
structure SubmersivePresentation extends PreSubmersivePresentation.{t, w} R S where
  jacobian_isUnit : IsUnit toPreSubmersivePresentation.jacobian
  isFinite : toPreSubmersivePresentation.IsFinite := by infer_instance


variable {R S} in
/-- If `algebraMap R S` is bijective, the empty generators are a submersive
presentation with no relations. -/
noncomputable def ofBijectiveAlgebraMap (h : Function.Bijective (algebraMap R S)) :
    SubmersivePresentation.{t, w} R S where
  __ := PreSubmersivePresentation.ofBijectiveAlgebraMap.{t, w} h
  jacobian_isUnit := by
    /-
      n m : Nat
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      h : Function.Bijective ⇑(algebraMap R S)
      ⊢ IsUnit __spread✝⁻⁰.jacobian
    -/
    rw [ofBijectiveAlgebraMap_jacobian]
    /-
      n m : Nat
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      h : Function.Bijective ⇑(algebraMap R S)
      ⊢ IsUnit 1
    -/
    exact isUnit_one
    /-
      🎉 no goals
    -/
  isFinite := Presentation.ofBijectiveAlgebraMap_isFinite h


/-- The canonical submersive `R`-presentation of `R` with no generators and no relations. -/
noncomputable def id : SubmersivePresentation.{t, w} R R :=
  ofBijectiveAlgebraMap Function.bijective_id


/-- Given an `R`-algebra `S` and an `S`-algebra `T` with submersive presentations,
this is the canonical submersive presentation of `T` as an `R`-algebra. -/
noncomputable def comp : SubmersivePresentation R T where
  __ := Q.toPreSubmersivePresentation.comp P.toPreSubmersivePresentation
  jacobian_isUnit := by
    /-
      n m : Nat
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type ?u.284206
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.SubmersivePresentation S T
      P : Algebra.SubmersivePresentation R S
      ⊢ IsUnit __spread✝⁻⁰.jacobian
    -/
    rw [comp_jacobian_eq_jacobian_smul_jacobian, Algebra.smul_def, IsUnit.mul_iff]
    /-
      n m : Nat
      R : Type u
      S : Type v
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : Algebra R S
      T : Type ?u.284206
      inst✝³ : CommRing T
      inst✝² : Algebra R T
      inst✝¹ : Algebra S T
      inst✝ : IsScalarTower R S T
      Q : Algebra.SubmersivePresentation S T
      P : Algebra.SubmersivePresentation R S
      ⊢ And (IsUnit ((algebraMap S T) P.jacobian)) (IsUnit Q.jacobian)
    -/
    exact ⟨RingHom.isUnit_map _ <| P.jacobian_isUnit, Q.jacobian_isUnit⟩
    /-
      🎉 no goals
    -/
  isFinite := Presentation.comp_isFinite Q.toPresentation P.toPresentation


/-- If `S` is the localization of `R` at `r`, this is the canonical submersive presentation
of `S` as `R`-algebra. -/
noncomputable def localizationAway : SubmersivePresentation R S where
  __ := PreSubmersivePresentation.localizationAway S r
  jacobian_isUnit := by
    /-
      n m : Nat
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      ⊢ IsUnit __spread✝⁻⁰.jacobian
    -/
    rw [localizationAway_jacobian]
    /-
      n m : Nat
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      r : R
      inst✝ : IsLocalization.Away r S
      ⊢ IsUnit ((algebraMap R S) r)
    -/
    apply IsLocalization.map_units' (⟨r, 1, by simp⟩ : Submonoid.powers r)
    /-
      🎉 no goals
    -/
  isFinite := Presentation.localizationAway_isFinite r


/-- If `P` is a submersive presentation of `S` over `R` and `T` is an `R`-algebra, we
obtain a natural submersive presentation of `T ⊗[R] S` over `T`. -/
noncomputable def baseChange : SubmersivePresentation T (T ⊗[R] S) where
  toPreSubmersivePresentation := P.toPreSubmersivePresentation.baseChange T
  jacobian_isUnit := P.baseChange_jacobian T ▸ P.jacobian_isUnit.map TensorProduct.includeRight
  isFinite := Presentation.baseChange_isFinite T P.toPresentation


/--
An `R`-algebra `S` is called standard smooth, if there
exists a submersive presentation.
-/
class IsStandardSmooth : Prop where
  out : Nonempty (SubmersivePresentation.{t, w} R S)


/--
The relative dimension of a standard smooth `R`-algebra `S` is
the dimension of an arbitrarily chosen submersive `R`-presentation of `S`.

Note: If `S` is non-trivial, this number is independent of the choice of the presentation as it is
equal to the `S`-rank of `Ω[S/R]` (TODO).
-/
noncomputable def IsStandardSmooth.relativeDimension [IsStandardSmooth R S] : ℕ :=
  ‹IsStandardSmooth R S›.out.some.dimension


/--
An `R`-algebra `S` is called standard smooth of relative dimension `n`, if there exists
a submersive presentation of dimension `n`.
-/
class IsStandardSmoothOfRelativeDimension : Prop where
  out : ∃ P : SubmersivePresentation.{t, w} R S, P.dimension = n


lemma IsStandardSmoothOfRelativeDimension.isStandardSmooth
    [IsStandardSmoothOfRelativeDimension.{t, w} n R S] :
    IsStandardSmooth.{t, w} R S :=
  ⟨‹IsStandardSmoothOfRelativeDimension n R S›.out.nonempty⟩


lemma IsStandardSmoothOfRelativeDimension.of_algebraMap_bijective
    (h : Function.Bijective (algebraMap R S)) :
    IsStandardSmoothOfRelativeDimension.{t, w} 0 R S :=
  ⟨SubmersivePresentation.ofBijectiveAlgebraMap h, Presentation.ofBijectiveAlgebraMap_dimension h⟩


variable (R) in
instance IsStandardSmoothOfRelativeDimension.id :
    IsStandardSmoothOfRelativeDimension.{t, w} 0 R R :=
  IsStandardSmoothOfRelativeDimension.of_algebraMap_bijective Function.bijective_id


instance (priority := 100) IsStandardSmooth.finitePresentation [IsStandardSmooth R S] :
    FinitePresentation R S := by
  /-
    n m : Nat
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsStandardSmooth R S
    ⊢ Algebra.FinitePresentation R S
  -/
  obtain ⟨⟨P⟩⟩ := ‹IsStandardSmooth R S›
  /-
    case mk.intro
    n m : Nat
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : Algebra.IsStandardSmooth R S
    P : Algebra.SubmersivePresentation R S
    ⊢ Algebra.FinitePresentation R S
  -/
  exact P.finitePresentation_of_isFinite
  /-
    🎉 no goals
  -/


lemma IsStandardSmooth.trans [IsStandardSmooth.{t, w} R S] [IsStandardSmooth.{t', w'} S T] :
    IsStandardSmooth.{max t t', max w w'} R T where
  out := by
    /-
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : Algebra.IsStandardSmooth R S
      inst✝ : Algebra.IsStandardSmooth S T
      ⊢ Nonempty (Algebra.SubmersivePresentation R T)
    -/
    obtain ⟨⟨P⟩⟩ := ‹IsStandardSmooth R S›
    /-
      case mk.intro
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : Algebra.IsStandardSmooth R S
      inst✝ : Algebra.IsStandardSmooth S T
      P : Algebra.SubmersivePresentation R S
      ⊢ Nonempty (Algebra.SubmersivePresentation R T)
    -/
    obtain ⟨⟨Q⟩⟩ := ‹IsStandardSmooth S T›
    /-
      case mk.intro.mk.intro
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : Algebra.IsStandardSmooth R S
      inst✝ : Algebra.IsStandardSmooth S T
      P : Algebra.SubmersivePresentation R S
      Q : Algebra.SubmersivePresentation S T
      ⊢ Nonempty (Algebra.SubmersivePresentation R T)
    -/
    exact ⟨Q.comp P⟩
    /-
      🎉 no goals
    -/


lemma IsStandardSmoothOfRelativeDimension.trans [IsStandardSmoothOfRelativeDimension.{t, w} n R S]
    [IsStandardSmoothOfRelativeDimension.{t', w'} m S T] :
    IsStandardSmoothOfRelativeDimension.{max t t', max w w'} (m + n) R T where
  out := by
    /-
      n m : Nat
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : Algebra.IsStandardSmoothOfRelativeDimension n R S
      inst✝ : Algebra.IsStandardSmoothOfRelativeDimension m S T
      ⊢ Exists fun P => Eq P.dimension (HAdd.hAdd m n)
    -/
    obtain ⟨P, hP⟩ := ‹IsStandardSmoothOfRelativeDimension n R S›
    /-
      case mk.intro
      n m : Nat
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : Algebra.IsStandardSmoothOfRelativeDimension n R S
      inst✝ : Algebra.IsStandardSmoothOfRelativeDimension m S T
      P : Algebra.SubmersivePresentation R S
      hP : Eq P.dimension n
      ⊢ Exists fun P => Eq P.dimension (HAdd.hAdd m n)
    -/
    obtain ⟨Q, hQ⟩ := ‹IsStandardSmoothOfRelativeDimension m S T›
    /-
      case mk.intro.mk.intro
      n m : Nat
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : Algebra.IsStandardSmoothOfRelativeDimension n R S
      inst✝ : Algebra.IsStandardSmoothOfRelativeDimension m S T
      P : Algebra.SubmersivePresentation R S
      hP : Eq P.dimension n
      Q : Algebra.SubmersivePresentation S T
      hQ : Eq Q.dimension m
      ⊢ Exists fun P => Eq P.dimension (HAdd.hAdd m n)
    -/
    refine ⟨Q.comp P, hP ▸ hQ ▸ ?_⟩
    /-
      case mk.intro.mk.intro
      n m : Nat
      R : Type u
      S : Type v
      inst✝⁸ : CommRing R
      inst✝⁷ : CommRing S
      inst✝⁶ : Algebra R S
      T : Type u_1
      inst✝⁵ : CommRing T
      inst✝⁴ : Algebra R T
      inst✝³ : Algebra S T
      inst✝² : IsScalarTower R S T
      inst✝¹ : Algebra.IsStandardSmoothOfRelativeDimension n R S
      inst✝ : Algebra.IsStandardSmoothOfRelativeDimension m S T
      P : Algebra.SubmersivePresentation R S
      hP : Eq P.dimension n
      Q : Algebra.SubmersivePresentation S T
      hQ : Eq Q.dimension m
      ⊢ Eq (Q.comp P).dimension (HAdd.hAdd Q.dimension P.dimension)
    -/
    apply PreSubmersivePresentation.dimension_comp_eq_dimension_add_dimension
    /-
      🎉 no goals
    -/


lemma IsStandardSmooth.localization_away (r : R) [IsLocalization.Away r S] :
    IsStandardSmooth.{0, 0} R S where
  out := ⟨SubmersivePresentation.localizationAway S r⟩


lemma IsStandardSmoothOfRelativeDimension.localization_away (r : R) [IsLocalization.Away r S] :
    IsStandardSmoothOfRelativeDimension.{0, 0} 0 R S where
  out := ⟨SubmersivePresentation.localizationAway S r,
    Presentation.localizationAway_dimension_zero r⟩


instance IsStandardSmooth.baseChange [IsStandardSmooth.{t, w} R S] :
    IsStandardSmooth.{t, w} T (T ⊗[R] S) where
  out := by
    /-
      n m : Nat
      R : Type u
      S : Type v
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R S
      T : Type u_1
      inst✝² : CommRing T
      inst✝¹ : Algebra R T
      inst✝ : Algebra.IsStandardSmooth R S
      ⊢ Nonempty (Algebra.SubmersivePresentation T (TensorProduct R T S))
    -/
    obtain ⟨⟨P⟩⟩ := ‹IsStandardSmooth R S›
    /-
      case mk.intro
      n m : Nat
      R : Type u
      S : Type v
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R S
      T : Type u_1
      inst✝² : CommRing T
      inst✝¹ : Algebra R T
      inst✝ : Algebra.IsStandardSmooth R S
      P : Algebra.SubmersivePresentation R S
      ⊢ Nonempty (Algebra.SubmersivePresentation T (TensorProduct R T S))
    -/
    exact ⟨P.baseChange R S T⟩
    /-
      🎉 no goals
    -/


instance IsStandardSmoothOfRelativeDimension.baseChange
    [IsStandardSmoothOfRelativeDimension.{t, w} n R S] :
    IsStandardSmoothOfRelativeDimension.{t, w} n T (T ⊗[R] S) where
  out := by
    /-
      n m : Nat
      R : Type u
      S : Type v
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R S
      T : Type u_1
      inst✝² : CommRing T
      inst✝¹ : Algebra R T
      inst✝ : Algebra.IsStandardSmoothOfRelativeDimension n R S
      ⊢ Exists fun P => Eq P.dimension n
    -/
    obtain ⟨P, hP⟩ := ‹IsStandardSmoothOfRelativeDimension n R S›
    /-
      case mk.intro
      n m : Nat
      R : Type u
      S : Type v
      inst✝⁵ : CommRing R
      inst✝⁴ : CommRing S
      inst✝³ : Algebra R S
      T : Type u_1
      inst✝² : CommRing T
      inst✝¹ : Algebra R T
      inst✝ : Algebra.IsStandardSmoothOfRelativeDimension n R S
      P : Algebra.SubmersivePresentation R S
      hP : Eq P.dimension n
      ⊢ Exists fun P => Eq P.dimension n
    -/
    exact ⟨P.baseChange R S T, hP⟩
    /-
      🎉 no goals
    -/


