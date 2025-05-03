/-- Type synonym for a semiring which depends on an absolute value. This is a function that takes
an absolute value on a semiring and returns the semiring. We use this to assign and infer instances
on a semiring that depend on absolute values. -/
@[nolint unusedArguments]
def WithAbs : AbsoluteValue R S → Type _ := fun _ => R


/-- Canonical equivalence between `WithAbs v` and `R`. -/
def equiv : WithAbs v ≃ R := Equiv.refl (WithAbs v)


instance instNonTrivial [Nontrivial R] : Nontrivial (WithAbs v) := inferInstanceAs (Nontrivial R)


instance instUnique [Unique R] : Unique (WithAbs v) := inferInstanceAs (Unique R)


instance instSemiring : Semiring (WithAbs v) := inferInstanceAs (Semiring R)


instance instRing [Ring R] : Ring (WithAbs v) := inferInstanceAs (Ring R)


instance instInhabited : Inhabited (WithAbs v) := ⟨0⟩


instance normedRing {R : Type*} [Ring R] (v : AbsoluteValue R ℝ) : NormedRing (WithAbs v) :=
  v.toNormedRing


instance normedField (v : AbsoluteValue K ℝ) : NormedField (WithAbs v) :=
  v.toNormedField


@[simp]
theorem equiv_zero : WithAbs.equiv v 0 = 0 := rfl


@[simp]
theorem equiv_symm_zero : (WithAbs.equiv v).symm 0 = 0 := rfl


@[simp]
theorem equiv_add : WithAbs.equiv v (x + y) = WithAbs.equiv v x + WithAbs.equiv v y := rfl


@[simp]
theorem equiv_symm_add :
    (WithAbs.equiv v).symm (r + s) = (WithAbs.equiv v).symm r + (WithAbs.equiv v).symm s :=
  rfl


@[simp]
theorem equiv_sub [Ring R] : WithAbs.equiv v (x - y) = WithAbs.equiv v x - WithAbs.equiv v y := rfl


@[simp]
theorem equiv_symm_sub [Ring R] :
    (WithAbs.equiv v).symm (r - s) = (WithAbs.equiv v).symm r - (WithAbs.equiv v).symm s :=
  rfl


@[simp]
theorem equiv_neg [Ring R] : WithAbs.equiv v (-x) = - WithAbs.equiv v x := rfl


@[simp]
theorem equiv_symm_neg [Ring R] : (WithAbs.equiv v).symm (-r) = - (WithAbs.equiv v).symm r := rfl


@[simp]
theorem equiv_mul : WithAbs.equiv v (x * y) = WithAbs.equiv v x * WithAbs.equiv v y := rfl


@[simp]
theorem equiv_symm_mul :
    (WithAbs.equiv v).symm (x * y) = (WithAbs.equiv v).symm x * (WithAbs.equiv v).symm y :=
  rfl


/-- `WithAbs.equiv` as a ring equivalence. -/
def ringEquiv : WithAbs v ≃+* R := RingEquiv.refl _


/-- If the absolute value `v` factors through an embedding `f` into a normed field, then
`f` is an isometry. -/
theorem isometry_of_comp (h : ∀ x, ‖f x‖ = v x) : Isometry f :=
                                       /-
                                         K : Type u_4
                                         inst✝¹ : Field K
                                         v : AbsoluteValue K Real
                                         L : Type u_5
                                         inst✝ : NormedField L
                                         f : RingHom (WithAbs v) L
                                         h : ∀ (x : WithAbs v), Eq (Norm.norm (f x)) (v x)
                                         x y : WithAbs v
                                         ⊢ Eq (Dist.dist (f x) (f y)) (Dist.dist x y)
                                       -/
  Isometry.of_dist_eq <| fun x y => by simp only [‹NormedField L›.dist_eq, ← f.map_sub, h]; rfl
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


/-- If the absolute value `v` factors through an embedding `f` into a normed field, then
the pseudo metric space associated to the absolute value is the same as the pseudo metric space
induced by `f`. -/
theorem pseudoMetricSpace_induced_of_comp (h : ∀ x, ‖f x‖ = v x) :
    PseudoMetricSpace.induced f inferInstance = (normedField v).toPseudoMetricSpace := by
  /-
    K : Type u_4
    inst✝¹ : Field K
    v : AbsoluteValue K Real
    L : Type u_5
    inst✝ : NormedField L
    f : RingHom (WithAbs v) L
    h : ∀ (x : WithAbs v), Eq (Norm.norm (f x)) (v x)
    ⊢ Eq (PseudoMetricSpace.induced (⇑f) inferInstance) MetricSpace.toPseudoMetric …
  -/
  ext; exact isometry_of_comp h |>.dist_eq _ _
       /-
         🎉 no goals
       -/


/-- If the absolute value `v` factors through an embedding `f` into a normed field, then
the uniform structure associated to the absolute value is the same as the uniform structure
induced by `f`. -/
theorem uniformSpace_comap_eq_of_comp (h : ∀ x, ‖f x‖ = v x) :
    UniformSpace.comap f inferInstance = (normedField v).toUniformSpace := by
  /-
    K : Type u_4
    inst✝¹ : Field K
    v : AbsoluteValue K Real
    L : Type u_5
    inst✝ : NormedField L
    f : RingHom (WithAbs v) L
    h : ∀ (x : WithAbs v), Eq (Norm.norm (f x)) (v x)
    ⊢ Eq (UniformSpace.comap (⇑f) inferInstance) PseudoMetricSpace.toUniformSpace
  -/
  simp only [← pseudoMetricSpace_induced_of_comp h, PseudoMetricSpace.toUniformSpace]
  /-
    🎉 no goals
  -/


/-- If the absolute value `v` factors through an embedding `f` into a normed field, then
`f` is uniform inducing. -/
theorem isUniformInducing_of_comp (h : ∀ x, ‖f x‖ = v x) : IsUniformInducing f :=
  isUniformInducing_iff_uniformSpace.2 <| uniformSpace_comap_eq_of_comp h


/-- The completion of a field with respect to a real absolute value. -/
abbrev Completion := UniformSpace.Completion (WithAbs v)


@[deprecated (since := "2024-12-01")] alias completion := Completion


instance : Coe K v.Completion :=
  inferInstanceAs <| Coe (WithAbs v) (UniformSpace.Completion (WithAbs v))


/-- If the absolute value of a normed field factors through an embedding into another normed field
`L`, then we can extend that embedding to an embedding on the completion `v.Completion →+* L`. -/
abbrev extensionEmbedding_of_comp (h : ∀ x, ‖f x‖ = v x) : v.Completion →+* L :=
  UniformSpace.Completion.extensionHom _
    (WithAbs.isUniformInducing_of_comp h).uniformContinuous.continuous


theorem extensionEmbedding_of_comp_coe (h : ∀ x, ‖f x‖ = v x) (x : K) :
    extensionEmbedding_of_comp h x = f x := by
  rw [← UniformSpace.Completion.extensionHom_coe f
    (WithAbs.isUniformInducing_of_comp h).uniformContinuous.continuous]


/-- If the absolute value of a normed field factors through an embedding into another normed field,
then the extended embedding `v.Completion →+* L` preserves distances. -/
theorem extensionEmbedding_dist_eq_of_comp (h : ∀ x, ‖f x‖ = v x) (x y : v.Completion) :
    dist (extensionEmbedding_of_comp h x) (extensionEmbedding_of_comp h y) =
      dist x y := by
  /-
    K : Type u_4
    inst✝² : Field K
    v : AbsoluteValue K Real
    L : Type u_5
    inst✝¹ : NormedField L
    inst✝ : CompleteSpace L
    f : RingHom (WithAbs v) L
    h : ∀ (x : WithAbs v), Eq (Norm.norm (f x)) (v x)
    x y : v.Completion
    ⊢ Eq (Dist.dist ((AbsoluteValue.Completion.extensionEmbedding_of_comp h) x) (( …
  -/
  refine UniformSpace.Completion.induction_on₂ x y ?_ (fun x y => ?_)
    /-
      case refine_1
      K : Type u_4
      inst✝² : Field K
      v : AbsoluteValue K Real
      L : Type u_5
      inst✝¹ : NormedField L
      inst✝ : CompleteSpace L
      f : RingHom (WithAbs v) L
      h : ∀ (x : WithAbs v), Eq (Norm.norm (f x)) (v x)
      x y : v.Completion
      ⊢ IsClosed (setOf fun x => Eq (Dist.dist ((AbsoluteValue.Completion.extensionE …
    -/
  · refine isClosed_eq ?_ continuous_dist
    /-
      case refine_1
      K : Type u_4
      inst✝² : Field K
      v : AbsoluteValue K Real
      L : Type u_5
      inst✝¹ : NormedField L
      inst✝ : CompleteSpace L
      f : RingHom (WithAbs v) L
      h : ∀ (x : WithAbs v), Eq (Norm.norm (f x)) (v x)
      x y : v.Completion
      ⊢ Continuous fun x => Dist.dist ((AbsoluteValue.Completion.extensionEmbedding_ …
    -/
    exact continuous_iff_continuous_dist.1 UniformSpace.Completion.continuous_extension
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_4
      inst✝² : Field K
      v : AbsoluteValue K Real
      L : Type u_5
      inst✝¹ : NormedField L
      inst✝ : CompleteSpace L
      f : RingHom (WithAbs v) L
      h : ∀ (x : WithAbs v), Eq (Norm.norm (f x)) (v x)
      x✝ y✝ : v.Completion
      x y : WithAbs v
      ⊢ Eq (Dist.dist ((AbsoluteValue.Completion.extensionEmbedding_of_comp h) (↑(Wi …
    -/
  · simp only [extensionEmbedding_of_comp_coe]
    /-
      case refine_2
      K : Type u_4
      inst✝² : Field K
      v : AbsoluteValue K Real
      L : Type u_5
      inst✝¹ : NormedField L
      inst✝ : CompleteSpace L
      f : RingHom (WithAbs v) L
      h : ∀ (x : WithAbs v), Eq (Norm.norm (f x)) (v x)
      x✝ y✝ : v.Completion
      x y : WithAbs v
      ⊢ Eq (Dist.dist (f x) (f y)) (Dist.dist (↑(WithAbs v) x) (↑(WithAbs v) y))
    -/
    exact UniformSpace.Completion.dist_eq x y ▸ (WithAbs.isometry_of_comp h).dist_eq _ _
    /-
      🎉 no goals
    -/


/-- If the absolute value of a normed field factors through an embedding into another normed field,
then the extended embedding `v.Completion →+* L` is an isometry. -/
theorem isometry_extensionEmbedding_of_comp (h : ∀ x, ‖f x‖ = v x) :
    Isometry (extensionEmbedding_of_comp h) :=
  Isometry.of_dist_eq <| extensionEmbedding_dist_eq_of_comp h


/-- If the absolute value of a normed field factors through an embedding into another normed field,
then the extended embedding `v.Completion →+* L` is a closed embedding. -/
theorem isClosedEmbedding_extensionEmbedding_of_comp (h : ∀ x, ‖f x‖ = v x) :
    IsClosedEmbedding (extensionEmbedding_of_comp h) :=
  (isometry_extensionEmbedding_of_comp h).isClosedEmbedding


/-- If the absolute value of a normed field factors through an embedding into another normed field
that is locally compact, then the completion of the first normed field is also locally compact. -/
theorem locallyCompactSpace [LocallyCompactSpace L] (h : ∀ x, ‖f x‖ = v x)  :
    LocallyCompactSpace (v.Completion) :=
  (isClosedEmbedding_extensionEmbedding_of_comp h).locallyCompactSpace


