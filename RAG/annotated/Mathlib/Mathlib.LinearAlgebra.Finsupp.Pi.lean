/-- If `α` has a unique term, then the type of finitely supported functions `α →₀ M` is
`R`-linearly equivalent to `M`. -/
noncomputable def LinearEquiv.finsuppUnique : (α →₀ M) ≃ₗ[R] M :=
  { Finsupp.equivFunOnFinite.trans (Equiv.funUnique α M) with
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }


@[simp]
theorem LinearEquiv.finsuppUnique_apply (f : α →₀ M) :
    LinearEquiv.finsuppUnique R M α f = f default :=
  rfl


@[simp]
theorem LinearEquiv.finsuppUnique_symm_apply (m : M) :
    (LinearEquiv.finsuppUnique R M α).symm m = Finsupp.single default m := by
  /-
    R : Type u_1
    M : Type u_3
    inst✝³ : AddCommMonoid M
    inst✝² : Semiring R
    inst✝¹ : Module R M
    α : Type u_4
    inst✝ : Unique α
    m : M
    ⊢ Eq ((Finsupp.LinearEquiv.finsuppUnique R M α).symm m) (Finsupp.single Inhabi …
  -/
  ext; simp [LinearEquiv.finsuppUnique, Equiv.funUnique, single, Pi.single,
    equivFunOnFinite, Function.update]


/-- Forget that a function is finitely supported.

This is the linear version of `Finsupp.toFun`. -/
@[simps]
def lcoeFun : (α →₀ M) →ₗ[R] α → M where
  toFun := (⇑)
  map_add' x y := by
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
      x y : Finsupp α M
      ⊢ Eq (⇑(HAdd.hAdd x y)) (HAdd.hAdd ⇑x ⇑y)
    -/
    ext
    /-
      case h
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
      x y : Finsupp α M
      x✝ : α
      ⊢ Eq ((HAdd.hAdd x y) x✝) (HAdd.hAdd (⇑x) (⇑y) x✝)
    -/
    simp
    /-
      🎉 no goals
    -/
  map_smul' x y := by
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
      x : R
      y : Finsupp α M
      ⊢ Eq ({ toFun := DFunLike.coe, map_add' := ⋯ }.toFun (HSMul.hSMul x y)) (HSMul …
    -/
    ext
    /-
      case h
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
      x : R
      y : Finsupp α M
      x✝ : α
      ⊢ Eq ({ toFun := DFunLike.coe, map_add' := ⋯ }.toFun (HSMul.hSMul x y) x✝) (HS …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- A surjective linear map to functions on a finite type has a splitting. -/
def splittingOfFunOnFintypeSurjective [Finite α] (f : M →ₗ[R] α → R) (s : Surjective f) :
    (α → R) →ₗ[R] M :=
  (Finsupp.lift _ _ _ fun x : α => (s (Finsupp.single x 1)).choose).comp
    (linearEquivFunOnFinite R R α).symm.toLinearMap


theorem splittingOfFunOnFintypeSurjective_splits [Finite α] (f : M →ₗ[R] α → R)
    (s : Surjective f) : f.comp (splittingOfFunOnFintypeSurjective f s) = LinearMap.id := by
  classical
  ext x y
  dsimp [splittingOfFunOnFintypeSurjective]
  rw [linearEquivFunOnFinite_symm_single, Finsupp.sum_single_index, one_smul,
    (s (Finsupp.single x 1)).choose_spec, Finsupp.single_eq_pi_single]
  rw [zero_smul]


theorem leftInverse_splittingOfFunOnFintypeSurjective [Finite α] (f : M →ₗ[R] α → R)
    (s : Surjective f) : LeftInverse f (splittingOfFunOnFintypeSurjective f s) := fun g =>
  LinearMap.congr_fun (splittingOfFunOnFintypeSurjective_splits f s) g


theorem splittingOfFunOnFintypeSurjective_injective [Finite α] (f : M →ₗ[R] α → R)
    (s : Surjective f) : Injective (splittingOfFunOnFintypeSurjective f s) :=
  (leftInverse_splittingOfFunOnFintypeSurjective f s).injective


