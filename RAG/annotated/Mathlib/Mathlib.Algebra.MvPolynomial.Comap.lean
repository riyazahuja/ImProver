/-- Given an algebra hom `f : MvPolynomial σ R →ₐ[R] MvPolynomial τ R`
and a variable evaluation `v : τ → R`,
`comap f v` produces a variable evaluation `σ → R`.
-/
noncomputable def comap (f : MvPolynomial σ R →ₐ[R] MvPolynomial τ R) : (τ → R) → σ → R :=
  fun x i => aeval x (f (X i))


@[simp]
theorem comap_apply (f : MvPolynomial σ R →ₐ[R] MvPolynomial τ R) (x : τ → R) (i : σ) :
    comap f x i = aeval x (f (X i)) :=
  rfl


@[simp]
theorem comap_id_apply (x : σ → R) : comap (AlgHom.id R (MvPolynomial σ R)) x = x := by
  /-
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    x : σ → R
    ⊢ Eq (MvPolynomial.comap (AlgHom.id R (MvPolynomial σ R)) x) x
  -/
  funext i
  /-
    case h
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    x : σ → R
    i : σ
    ⊢ Eq (MvPolynomial.comap (AlgHom.id R (MvPolynomial σ R)) x i) (x i)
  -/
  simp only [comap, AlgHom.id_apply, id, aeval_X]
  /-
    🎉 no goals
  -/


theorem comap_id : comap (AlgHom.id R (MvPolynomial σ R)) = id := by
  /-
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    ⊢ Eq (MvPolynomial.comap (AlgHom.id R (MvPolynomial σ R))) id
  -/
  funext x
  /-
    case h
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    x : σ → R
    ⊢ Eq (MvPolynomial.comap (AlgHom.id R (MvPolynomial σ R)) x) (id x)
  -/
  exact comap_id_apply x
  /-
    🎉 no goals
  -/


theorem comap_comp_apply (f : MvPolynomial σ R →ₐ[R] MvPolynomial τ R)
    (g : MvPolynomial τ R →ₐ[R] MvPolynomial υ R) (x : υ → R) :
    comap (g.comp f) x = comap f (comap g x) := by
  /-
    σ : Type u_1
    τ : Type u_2
    υ : Type u_3
    R : Type u_4
    inst✝ : CommSemiring R
    f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
    g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
    x : υ → R
    ⊢ Eq (MvPolynomial.comap (g.comp f) x) (MvPolynomial.comap f (MvPolynomial.com …
  -/
  funext i
  /-
    case h
    σ : Type u_1
    τ : Type u_2
    υ : Type u_3
    R : Type u_4
    inst✝ : CommSemiring R
    f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
    g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
    x : υ → R
    i : σ
    ⊢ Eq (MvPolynomial.comap (g.comp f) x i) (MvPolynomial.comap f (MvPolynomial.c …
  -/
  trans aeval x (aeval (fun i => g (X i)) (f (X i)))
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
      g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
      x : υ → R
      i : σ
      ⊢ Eq (MvPolynomial.comap (g.comp f) x i) ((MvPolynomial.aeval x) ((MvPolynomia …
    -/
  · apply eval₂Hom_congr rfl rfl
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
      g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
      x : υ → R
      i : σ
      ⊢ Eq ((g.comp f) (MvPolynomial.X i)) ((MvPolynomial.aeval fun i => g (MvPolyno …
    -/
    rw [AlgHom.comp_apply]
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
      g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
      x : υ → R
      i : σ
      ⊢ Eq (g (f (MvPolynomial.X i))) ((MvPolynomial.aeval fun i => g (MvPolynomial. …
    -/
    suffices g = aeval fun i => g (X i) by rw [← this]
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
      g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
      x : υ → R
      i : σ
      ⊢ Eq g (MvPolynomial.aeval fun i => g (MvPolynomial.X i))
    -/
    exact aeval_unique g
    /-
      🎉 no goals
    -/
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
      g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
      x : υ → R
      i : σ
      ⊢ Eq ((MvPolynomial.aeval x) ((MvPolynomial.aeval fun i => g (MvPolynomial.X i …
    -/
  · simp only [comap, aeval_eq_eval₂Hom, map_eval₂Hom, AlgHom.comp_apply]
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
      g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
      x : υ → R
      i : σ
      ⊢ Eq ((MvPolynomial.eval₂Hom ((MvPolynomial.eval₂Hom (algebraMap R R) x).comp  …
    -/
    refine eval₂Hom_congr ?_ rfl rfl
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
      g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
      x : υ → R
      i : σ
      ⊢ Eq ((MvPolynomial.eval₂Hom (algebraMap R R) x).comp (algebraMap R (MvPolynom …
    -/
    ext r
    /-
      case a
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
      g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
      x : υ → R
      i : σ
      r : R
      ⊢ Eq (((MvPolynomial.eval₂Hom (algebraMap R R) x).comp (algebraMap R (MvPolyno …
    -/
    apply aeval_C
    /-
      🎉 no goals
    -/


theorem comap_comp (f : MvPolynomial σ R →ₐ[R] MvPolynomial τ R)
    (g : MvPolynomial τ R →ₐ[R] MvPolynomial υ R) : comap (g.comp f) = comap f ∘ comap g := by
  /-
    σ : Type u_1
    τ : Type u_2
    υ : Type u_3
    R : Type u_4
    inst✝ : CommSemiring R
    f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
    g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
    ⊢ Eq (MvPolynomial.comap (g.comp f)) (Function.comp (MvPolynomial.comap f) (Mv …
  -/
  funext x
  /-
    case h
    σ : Type u_1
    τ : Type u_2
    υ : Type u_3
    R : Type u_4
    inst✝ : CommSemiring R
    f : AlgHom R (MvPolynomial σ R) (MvPolynomial τ R)
    g : AlgHom R (MvPolynomial τ R) (MvPolynomial υ R)
    x : υ → R
    ⊢ Eq (MvPolynomial.comap (g.comp f) x) (Function.comp (MvPolynomial.comap f) ( …
  -/
  exact comap_comp_apply _ _ _
  /-
    🎉 no goals
  -/


theorem comap_eq_id_of_eq_id (f : MvPolynomial σ R →ₐ[R] MvPolynomial σ R) (hf : ∀ φ, f φ = φ)
    (x : σ → R) : comap f x = x := by
  /-
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    f : AlgHom R (MvPolynomial σ R) (MvPolynomial σ R)
    hf : ∀ (φ : MvPolynomial σ R), Eq (f φ) φ
    x : σ → R
    ⊢ Eq (MvPolynomial.comap f x) x
  -/
  convert comap_id_apply x
  /-
    case h.e'_2.h.e'_5
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    f : AlgHom R (MvPolynomial σ R) (MvPolynomial σ R)
    hf : ∀ (φ : MvPolynomial σ R), Eq (f φ) φ
    x : σ → R
    ⊢ Eq f (AlgHom.id R (MvPolynomial σ R))
  -/
  ext1 φ
  /-
    case h.e'_2.h.e'_5.hf
    σ : Type u_1
    R : Type u_4
    inst✝ : CommSemiring R
    f : AlgHom R (MvPolynomial σ R) (MvPolynomial σ R)
    hf : ∀ (φ : MvPolynomial σ R), Eq (f φ) φ
    x : σ → R
    φ : σ
    ⊢ Eq (f (MvPolynomial.X φ)) ((AlgHom.id R (MvPolynomial σ R)) (MvPolynomial.X  …
  -/
  simp [hf, AlgHom.id_apply]
  /-
    🎉 no goals
  -/


theorem comap_rename (f : σ → τ) (x : τ → R) : comap (rename f) x = x ∘ f := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    f : σ → τ
    x : τ → R
    ⊢ Eq (MvPolynomial.comap (MvPolynomial.rename f) x) (Function.comp x f)
  -/
  funext
  /-
    case h
    σ : Type u_1
    τ : Type u_2
    R : Type u_4
    inst✝ : CommSemiring R
    f : σ → τ
    x : τ → R
    x✝ : σ
    ⊢ Eq (MvPolynomial.comap (MvPolynomial.rename f) x x✝) (Function.comp x f x✝)
  -/
  simp [rename_X, comap_apply, aeval_X]
  /-
    🎉 no goals
  -/


/-- If two polynomial types over the same coefficient ring `R` are equivalent,
there is a bijection between the types of functions from their variable types to `R`.
-/
noncomputable def comapEquiv (f : MvPolynomial σ R ≃ₐ[R] MvPolynomial τ R) : (τ → R) ≃ (σ → R) where
  toFun := comap f
  invFun := comap f.symm
  left_inv := by
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgEquiv R (MvPolynomial σ R) (MvPolynomial τ R)
      ⊢ Function.LeftInverse (MvPolynomial.comap ↑f.symm) (MvPolynomial.comap ↑f)
    -/
    intro x
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgEquiv R (MvPolynomial σ R) (MvPolynomial τ R)
      x : τ → R
      ⊢ Eq (MvPolynomial.comap (↑f.symm) (MvPolynomial.comap (↑f) x)) x
    -/
    rw [← comap_comp_apply]
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgEquiv R (MvPolynomial σ R) (MvPolynomial τ R)
      x : τ → R
      ⊢ Eq (MvPolynomial.comap ((↑f).comp ↑f.symm) x) x
    -/
    apply comap_eq_id_of_eq_id
    /-
      case hf
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgEquiv R (MvPolynomial σ R) (MvPolynomial τ R)
      x : τ → R
      ⊢ ∀ (φ : MvPolynomial τ R), Eq (((↑f).comp ↑f.symm) φ) φ
    -/
    intro
    /-
      case hf
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgEquiv R (MvPolynomial σ R) (MvPolynomial τ R)
      x : τ → R
      φ✝ : MvPolynomial τ R
      ⊢ Eq (((↑f).comp ↑f.symm) φ✝) φ✝
    -/
    simp only [AlgHom.id_apply, AlgEquiv.comp_symm]
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgEquiv R (MvPolynomial σ R) (MvPolynomial τ R)
      ⊢ Function.RightInverse (MvPolynomial.comap ↑f.symm) (MvPolynomial.comap ↑f)
    -/
    intro x
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgEquiv R (MvPolynomial σ R) (MvPolynomial τ R)
      x : σ → R
      ⊢ Eq (MvPolynomial.comap (↑f) (MvPolynomial.comap (↑f.symm) x)) x
    -/
    rw [← comap_comp_apply]
    /-
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgEquiv R (MvPolynomial σ R) (MvPolynomial τ R)
      x : σ → R
      ⊢ Eq (MvPolynomial.comap ((↑f.symm).comp ↑f) x) x
    -/
    apply comap_eq_id_of_eq_id
    /-
      case hf
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgEquiv R (MvPolynomial σ R) (MvPolynomial τ R)
      x : σ → R
      ⊢ ∀ (φ : MvPolynomial σ R), Eq (((↑f.symm).comp ↑f) φ) φ
    -/
    intro
    /-
      case hf
      σ : Type u_1
      τ : Type u_2
      υ : Type u_3
      R : Type u_4
      inst✝ : CommSemiring R
      f : AlgEquiv R (MvPolynomial σ R) (MvPolynomial τ R)
      x : σ → R
      φ✝ : MvPolynomial σ R
      ⊢ Eq (((↑f.symm).comp ↑f) φ✝) φ✝
    -/
    simp only [AlgHom.id_apply, AlgEquiv.symm_comp]
    /-
      🎉 no goals
    -/


@[simp]
theorem comapEquiv_coe (f : MvPolynomial σ R ≃ₐ[R] MvPolynomial τ R) :
    (comapEquiv f : (τ → R) → σ → R) = comap f :=
  rfl


@[simp]
theorem comapEquiv_symm_coe (f : MvPolynomial σ R ≃ₐ[R] MvPolynomial τ R) :
    ((comapEquiv f).symm : (σ → R) → τ → R) = comap f.symm :=
  rfl


