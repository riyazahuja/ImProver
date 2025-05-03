/--
`bind₁` is the "left hand side" bind operation on `MvPolynomial`, operating on the variable type.
Given a polynomial `p : MvPolynomial σ R` and a map `f : σ → MvPolynomial τ R` taking variables
in `p` to polynomials in the variable type `τ`, `bind₁ f p` replaces each variable in `p` with
its value under `f`, producing a new polynomial in `τ`. The coefficient type remains the same.
This operation is an algebra hom.
-/
def bind₁ (f : σ → MvPolynomial τ R) : MvPolynomial σ R →ₐ[R] MvPolynomial τ R :=
  aeval f


/-- `bind₂` is the "right hand side" bind operation on `MvPolynomial`,
operating on the coefficient type.
Given a polynomial `p : MvPolynomial σ R` and
a map `f : R → MvPolynomial σ S` taking coefficients in `p` to polynomials over a new ring `S`,
`bind₂ f p` replaces each coefficient in `p` with its value under `f`,
producing a new polynomial over `S`.
The variable type remains the same. This operation is a ring hom.
-/
def bind₂ (f : R →+* MvPolynomial σ S) : MvPolynomial σ R →+* MvPolynomial σ S :=
  eval₂Hom f X


/--
`join₁` is the monadic join operation corresponding to `MvPolynomial.bind₁`. Given a polynomial `p`
with coefficients in `R` whose variables are polynomials in `σ` with coefficients in `R`,
`join₁ p` collapses `p` to a polynomial with variables in `σ` and coefficients in `R`.
This operation is an algebra hom.
-/
def join₁ : MvPolynomial (MvPolynomial σ R) R →ₐ[R] MvPolynomial σ R :=
  aeval id


/--
`join₂` is the monadic join operation corresponding to `MvPolynomial.bind₂`. Given a polynomial `p`
with variables in `σ` whose coefficients are polynomials in `σ` with coefficients in `R`,
`join₂ p` collapses `p` to a polynomial with variables in `σ` and coefficients in `R`.
This operation is a ring hom.
-/
def join₂ : MvPolynomial σ (MvPolynomial σ R) →+* MvPolynomial σ R :=
  eval₂Hom (RingHom.id _) X


@[simp]
theorem aeval_eq_bind₁ (f : σ → MvPolynomial τ R) : aeval f = bind₁ f :=
  rfl


@[simp]
theorem eval₂Hom_C_eq_bind₁ (f : σ → MvPolynomial τ R) : eval₂Hom C f = bind₁ f :=
  rfl


@[simp]
theorem eval₂Hom_eq_bind₂ (f : R →+* MvPolynomial σ S) : eval₂Hom f X = bind₂ f :=
  rfl


@[simp]
theorem aeval_id_eq_join₁ : aeval id = @join₁ σ R _ :=
  rfl


theorem eval₂Hom_C_id_eq_join₁ (φ : MvPolynomial (MvPolynomial σ R) R) :
    eval₂Hom C id φ = join₁ φ :=
  rfl


@[simp]
theorem eval₂Hom_id_X_eq_join₂ : eval₂Hom (RingHom.id _) X = @join₂ σ R _ :=
  rfl


@[simp]
theorem bind₁_X_right (f : σ → MvPolynomial τ R) (i : σ) : bind₁ f (X i) = f i :=
  aeval_X f i


@[simp]
theorem bind₂_X_right (f : R →+* MvPolynomial σ S) (i : σ) : bind₂ f (X i) = X i :=
  eval₂Hom_X' f X i


@[simp]
theorem bind₁_X_left : bind₁ (X : σ → MvPolynomial σ R) = AlgHom.id R _ := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    ⊢ Eq (MvPolynomial.bind₁ MvPolynomial.X) (AlgHom.id R (MvPolynomial σ R))
  -/
  ext1 i
  /-
    case hf
    σ : Type u_1
    R : Type u_3
    inst✝ : CommSemiring R
    i : σ
    ⊢ Eq ((MvPolynomial.bind₁ MvPolynomial.X) (MvPolynomial.X i)) ((AlgHom.id R (M …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem bind₁_C_right (f : σ → MvPolynomial τ R) (x) : bind₁ f (C x) = C x := algHom_C _ _


@[simp]
theorem bind₂_C_right (f : R →+* MvPolynomial σ S) (r : R) : bind₂ f (C r) = f r :=
  eval₂Hom_C f X r


@[simp]
                                                                               /-
                                                                                 σ : Type u_1
                                                                                 R : Type u_3
                                                                                 inst✝ : CommSemiring R
                                                                                 ⊢ Eq (MvPolynomial.bind₂ MvPolynomial.C) (RingHom.id (MvPolynomial σ R))
                                                                               -/
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
theorem bind₂_C_left : bind₂ (C : R →+* MvPolynomial σ R) = RingHom.id _ := by ext : 2 <;> simp
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


@[simp]
theorem bind₂_comp_C (f : R →+* MvPolynomial σ S) : (bind₂ f).comp C = f :=
  RingHom.ext <| bind₂_C_right _


@[simp]
theorem join₂_map (f : R →+* MvPolynomial σ S) (φ : MvPolynomial σ R) :
                                      /-
                                        σ : Type u_1
                                        R : Type u_3
                                        S : Type u_4
                                        inst✝¹ : CommSemiring R
                                        inst✝ : CommSemiring S
                                        f : RingHom R (MvPolynomial σ S)
                                        φ : MvPolynomial σ R
                                        ⊢ Eq (MvPolynomial.join₂ ((MvPolynomial.map f) φ)) ((MvPolynomial.bind₂ f) φ)
                                      -/
    join₂ (map f φ) = bind₂ f φ := by simp only [join₂, bind₂, eval₂Hom_map_hom, RingHom.id_comp]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem join₂_comp_map (f : R →+* MvPolynomial σ S) : join₂.comp (map f) = bind₂ f :=
  RingHom.ext <| join₂_map _


theorem aeval_id_rename (f : σ → MvPolynomial τ R) (p : MvPolynomial σ R) :
                                            /-
                                              σ : Type u_1
                                              τ : Type u_2
                                              R : Type u_3
                                              inst✝ : CommSemiring R
                                              f : σ → MvPolynomial τ R
                                              p : MvPolynomial σ R
                                              ⊢ Eq ((MvPolynomial.aeval id) ((MvPolynomial.rename f) p)) ((MvPolynomial.aeva …
                                            -/
    aeval id (rename f p) = aeval f p := by rw [aeval_rename, Function.id_comp]
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
theorem join₁_rename (f : σ → MvPolynomial τ R) (φ : MvPolynomial σ R) :
    join₁ (rename f φ) = bind₁ f φ :=
  aeval_id_rename _ _


@[simp]
theorem bind₁_id : bind₁ (@id (MvPolynomial σ R)) = join₁ :=
  rfl


@[simp]
theorem bind₂_id : bind₂ (RingHom.id (MvPolynomial σ R)) = join₂ :=
  rfl


theorem bind₁_bind₁ {υ : Type*} (f : σ → MvPolynomial τ R) (g : τ → MvPolynomial υ R)
    (φ : MvPolynomial σ R) : (bind₁ g) (bind₁ f φ) = bind₁ (fun i => bind₁ g (f i)) φ := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    υ : Type u_6
    f : σ → MvPolynomial τ R
    g : τ → MvPolynomial υ R
    φ : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.bind₁ g) ((MvPolynomial.bind₁ f) φ)) ((MvPolynomial.bind₁  …
  -/
  simp [bind₁, ← comp_aeval]
  /-
    🎉 no goals
  -/


theorem bind₁_comp_bind₁ {υ : Type*} (f : σ → MvPolynomial τ R) (g : τ → MvPolynomial υ R) :
    (bind₁ g).comp (bind₁ f) = bind₁ fun i => bind₁ g (f i) := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    υ : Type u_6
    f : σ → MvPolynomial τ R
    g : τ → MvPolynomial υ R
    ⊢ Eq ((MvPolynomial.bind₁ g).comp (MvPolynomial.bind₁ f)) (MvPolynomial.bind₁  …
  -/
  ext1
  /-
    case hf
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    υ : Type u_6
    f : σ → MvPolynomial τ R
    g : τ → MvPolynomial υ R
    i✝ : σ
    ⊢ Eq (((MvPolynomial.bind₁ g).comp (MvPolynomial.bind₁ f)) (MvPolynomial.X i✝) …
  -/
  apply bind₁_bind₁
  /-
    🎉 no goals
  -/


theorem bind₂_comp_bind₂ (f : R →+* MvPolynomial σ S) (g : S →+* MvPolynomial σ T) :
                                                              /-
                                                                σ : Type u_1
                                                                R : Type u_3
                                                                S : Type u_4
                                                                T : Type u_5
                                                                inst✝² : CommSemiring R
                                                                inst✝¹ : CommSemiring S
                                                                inst✝ : CommSemiring T
                                                                f : RingHom R (MvPolynomial σ S)
                                                                g : RingHom S (MvPolynomial σ T)
                                                                ⊢ Eq ((MvPolynomial.bind₂ g).comp (MvPolynomial.bind₂ f)) (MvPolynomial.bind₂  …
                                                              -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    (bind₂ g).comp (bind₂ f) = bind₂ ((bind₂ g).comp f) := by ext : 2 <;> simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem bind₂_bind₂ (f : R →+* MvPolynomial σ S) (g : S →+* MvPolynomial σ T)
    (φ : MvPolynomial σ R) : (bind₂ g) (bind₂ f φ) = bind₂ ((bind₂ g).comp f) φ :=
  RingHom.congr_fun (bind₂_comp_bind₂ f g) φ


theorem rename_comp_bind₁ {υ : Type*} (f : σ → MvPolynomial τ R) (g : τ → υ) :
    (rename g).comp (bind₁ f) = bind₁ fun i => rename g <| f i := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    υ : Type u_6
    f : σ → MvPolynomial τ R
    g : τ → υ
    ⊢ Eq ((MvPolynomial.rename g).comp (MvPolynomial.bind₁ f)) (MvPolynomial.bind₁ …
  -/
  ext1 i
  /-
    case hf
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    υ : Type u_6
    f : σ → MvPolynomial τ R
    g : τ → υ
    i : σ
    ⊢ Eq (((MvPolynomial.rename g).comp (MvPolynomial.bind₁ f)) (MvPolynomial.X i) …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem rename_bind₁ {υ : Type*} (f : σ → MvPolynomial τ R) (g : τ → υ) (φ : MvPolynomial σ R) :
    rename g (bind₁ f φ) = bind₁ (fun i => rename g <| f i) φ :=
  AlgHom.congr_fun (rename_comp_bind₁ f g) φ


theorem map_bind₂ (f : R →+* MvPolynomial σ S) (g : S →+* T) (φ : MvPolynomial σ R) :
    map g (bind₂ f φ) = bind₂ ((map g).comp f) φ := by
  /-
    σ : Type u_1
    R : Type u_3
    S : Type u_4
    T : Type u_5
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : CommSemiring T
    f : RingHom R (MvPolynomial σ S)
    g : RingHom S T
    φ : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.map g) ((MvPolynomial.bind₂ f) φ)) ((MvPolynomial.bind₂ (( …
  -/
  simp only [bind₂, eval₂_comp_right, coe_eval₂Hom, eval₂_map]
  /-
    σ : Type u_1
    R : Type u_3
    S : Type u_4
    T : Type u_5
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : CommSemiring T
    f : RingHom R (MvPolynomial σ S)
    g : RingHom S T
    φ : MvPolynomial σ R
    ⊢ Eq (MvPolynomial.eval₂ ((MvPolynomial.map g).comp f) (Function.comp (⇑(MvPol …
  -/
  congr 1 with : 1
  /-
    case e_g.h
    σ : Type u_1
    R : Type u_3
    S : Type u_4
    T : Type u_5
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : CommSemiring T
    f : RingHom R (MvPolynomial σ S)
    g : RingHom S T
    φ : MvPolynomial σ R
    x✝ : σ
    ⊢ Eq (Function.comp (⇑(MvPolynomial.map g)) MvPolynomial.X x✝) (MvPolynomial.X …
  -/
  simp only [Function.comp_apply, map_X]
  /-
    🎉 no goals
  -/


theorem bind₁_comp_rename {υ : Type*} (f : τ → MvPolynomial υ R) (g : σ → τ) :
    (bind₁ f).comp (rename g) = bind₁ (f ∘ g) := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    υ : Type u_6
    f : τ → MvPolynomial υ R
    g : σ → τ
    ⊢ Eq ((MvPolynomial.bind₁ f).comp (MvPolynomial.rename g)) (MvPolynomial.bind₁ …
  -/
  ext1 i
  /-
    case hf
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    inst✝ : CommSemiring R
    υ : Type u_6
    f : τ → MvPolynomial υ R
    g : σ → τ
    i : σ
    ⊢ Eq (((MvPolynomial.bind₁ f).comp (MvPolynomial.rename g)) (MvPolynomial.X i) …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem bind₁_rename {υ : Type*} (f : τ → MvPolynomial υ R) (g : σ → τ) (φ : MvPolynomial σ R) :
    bind₁ f (rename g φ) = bind₁ (f ∘ g) φ :=
  AlgHom.congr_fun (bind₁_comp_rename f g) φ


theorem bind₂_map (f : S →+* MvPolynomial σ T) (g : R →+* S) (φ : MvPolynomial σ R) :
                                                 /-
                                                   σ : Type u_1
                                                   R : Type u_3
                                                   S : Type u_4
                                                   T : Type u_5
                                                   inst✝² : CommSemiring R
                                                   inst✝¹ : CommSemiring S
                                                   inst✝ : CommSemiring T
                                                   f : RingHom S (MvPolynomial σ T)
                                                   g : RingHom R S
                                                   φ : MvPolynomial σ R
                                                   ⊢ Eq ((MvPolynomial.bind₂ f) ((MvPolynomial.map g) φ)) ((MvPolynomial.bind₂ (f …
                                                 -/
    bind₂ f (map g φ) = bind₂ (f.comp g) φ := by simp [bind₂]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem map_comp_C (f : R →+* S) : (map f).comp (C : R →+* MvPolynomial σ R) = C.comp f := by
  /-
    σ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    ⊢ Eq ((MvPolynomial.map f).comp MvPolynomial.C) (MvPolynomial.C.comp f)
  -/
  ext1
  /-
    case a
    σ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    x✝ : R
    ⊢ Eq (((MvPolynomial.map f).comp MvPolynomial.C) x✝) ((MvPolynomial.C.comp f)  …
  -/
  apply map_C
  /-
    🎉 no goals
  -/

-- mixing the two monad structures

theorem hom_bind₁ (f : MvPolynomial τ R →+* S) (g : σ → MvPolynomial τ R) (φ : MvPolynomial σ R) :
    f (bind₁ g φ) = eval₂Hom (f.comp C) (fun i => f (g i)) φ := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom (MvPolynomial τ R) S
    g : σ → MvPolynomial τ R
    φ : MvPolynomial σ R
    ⊢ Eq (f ((MvPolynomial.bind₁ g) φ)) ((MvPolynomial.eval₂Hom (f.comp MvPolynomi …
  -/
  rw [bind₁, map_aeval, algebraMap_eq]
  /-
    🎉 no goals
  -/


theorem map_bind₁ (f : R →+* S) (g : σ → MvPolynomial τ R) (φ : MvPolynomial σ R) :
    map f (bind₁ g φ) = bind₁ (fun i : σ => (map f) (g i)) (map f φ) := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    g : σ → MvPolynomial τ R
    φ : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.map f) ((MvPolynomial.bind₁ g) φ)) ((MvPolynomial.bind₁ fu …
  -/
  rw [hom_bind₁, map_comp_C, ← eval₂Hom_map_hom]
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    g : σ → MvPolynomial τ R
    φ : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.eval₂Hom MvPolynomial.C fun i => (MvPolynomial.map f) (g i …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem eval₂Hom_comp_C (f : R →+* S) (g : σ → S) : (eval₂Hom f g).comp C = f := by
  /-
    σ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    g : σ → S
    ⊢ Eq ((MvPolynomial.eval₂Hom f g).comp MvPolynomial.C) f
  -/
  ext1 r
  /-
    case a
    σ : Type u_1
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    g : σ → S
    r : R
    ⊢ Eq (((MvPolynomial.eval₂Hom f g).comp MvPolynomial.C) r) (f r)
  -/
  exact eval₂_C f g r
  /-
    🎉 no goals
  -/


theorem eval₂Hom_bind₁ (f : R →+* S) (g : τ → S) (h : σ → MvPolynomial τ R) (φ : MvPolynomial σ R) :
    eval₂Hom f g (bind₁ h φ) = eval₂Hom f (fun i => eval₂Hom f g (h i)) φ := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    g : τ → S
    h : σ → MvPolynomial τ R
    φ : MvPolynomial σ R
    ⊢ Eq ((MvPolynomial.eval₂Hom f g) ((MvPolynomial.bind₁ h) φ)) ((MvPolynomial.e …
  -/
  rw [hom_bind₁, eval₂Hom_comp_C]
  /-
    🎉 no goals
  -/


theorem aeval_bind₁ [Algebra R S] (f : τ → S) (g : σ → MvPolynomial τ R) (φ : MvPolynomial σ R) :
    aeval f (bind₁ g φ) = aeval (fun i => aeval f (g i)) φ :=
  eval₂Hom_bind₁ _ _ _ _


theorem aeval_comp_bind₁ [Algebra R S] (f : τ → S) (g : σ → MvPolynomial τ R) :
    (aeval f).comp (bind₁ g) = aeval fun i => aeval f (g i) := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    f : τ → S
    g : σ → MvPolynomial τ R
    ⊢ Eq ((MvPolynomial.aeval f).comp (MvPolynomial.bind₁ g)) (MvPolynomial.aeval  …
  -/
  ext1
  /-
    case hf
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    S : Type u_4
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    f : τ → S
    g : σ → MvPolynomial τ R
    i✝ : σ
    ⊢ Eq (((MvPolynomial.aeval f).comp (MvPolynomial.bind₁ g)) (MvPolynomial.X i✝) …
  -/
  apply aeval_bind₁
  /-
    🎉 no goals
  -/


theorem eval₂Hom_comp_bind₂ (f : S →+* T) (g : σ → T) (h : R →+* MvPolynomial σ S) :
                                                                             /-
                                                                               σ : Type u_1
                                                                               R : Type u_3
                                                                               S : Type u_4
                                                                               T : Type u_5
                                                                               inst✝² : CommSemiring R
                                                                               inst✝¹ : CommSemiring S
                                                                               inst✝ : CommSemiring T
                                                                               f : RingHom S T
                                                                               g : σ → T
                                                                               h : RingHom R (MvPolynomial σ S)
                                                                               ⊢ Eq ((MvPolynomial.eval₂Hom f g).comp (MvPolynomial.bind₂ h)) (MvPolynomial.e …
                                                                             -/
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
    (eval₂Hom f g).comp (bind₂ h) = eval₂Hom ((eval₂Hom f g).comp h) g := by ext : 2 <;> simp
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


theorem eval₂Hom_bind₂ (f : S →+* T) (g : σ → T) (h : R →+* MvPolynomial σ S)
    (φ : MvPolynomial σ R) : eval₂Hom f g (bind₂ h φ) = eval₂Hom ((eval₂Hom f g).comp h) g φ :=
  RingHom.congr_fun (eval₂Hom_comp_bind₂ f g h) φ


theorem aeval_bind₂ [Algebra S T] (f : σ → T) (g : R →+* MvPolynomial σ S) (φ : MvPolynomial σ R) :
    aeval f (bind₂ g φ) = eval₂Hom ((↑(aeval f : _ →ₐ[S] _) : _ →+* _).comp g) f φ :=
  eval₂Hom_bind₂ _ _ _ _


alias eval₂Hom_C_left := eval₂Hom_C_eq_bind₁


theorem bind₁_monomial (f : σ → MvPolynomial τ R) (d : σ →₀ ℕ) (r : R) :
    bind₁ f (monomial d r) = C r * ∏ i ∈ d.support, f i ^ d i := by
  simp only [monomial_eq, map_mul, bind₁_C_right, Finsupp.prod, map_prod,
    map_pow, bind₁_X_right]


theorem bind₂_monomial (f : R →+* MvPolynomial σ S) (d : σ →₀ ℕ) (r : R) :
    bind₂ f (monomial d r) = f r * monomial d 1 := by
  simp only [monomial_eq, RingHom.map_mul, bind₂_C_right, Finsupp.prod, map_prod,
    map_pow, bind₂_X_right, C_1, one_mul]


@[simp]
theorem bind₂_monomial_one (f : R →+* MvPolynomial σ S) (d : σ →₀ ℕ) :
                                                /-
                                                  σ : Type u_1
                                                  R : Type u_3
                                                  S : Type u_4
                                                  inst✝¹ : CommSemiring R
                                                  inst✝ : CommSemiring S
                                                  f : RingHom R (MvPolynomial σ S)
                                                  d : Finsupp σ Nat
                                                  ⊢ Eq ((MvPolynomial.bind₂ f) ((MvPolynomial.monomial d) 1)) ((MvPolynomial.mon …
                                                -/
    bind₂ f (monomial d 1) = monomial d 1 := by rw [bind₂_monomial, f.map_one, one_mul]
                                                /-
                                                  🎉 no goals
                                                -/


theorem vars_bind₁ [DecidableEq τ] (f : σ → MvPolynomial τ R) (φ : MvPolynomial σ R) :
    (bind₁ f φ).vars ⊆ φ.vars.biUnion fun i => (f i).vars := by
  calc (bind₁ f φ).vars
    _ = (φ.support.sum fun x : σ →₀ ℕ => (bind₁ f) (monomial x (coeff x φ))).vars := by
      rw [← map_sum, ← φ.as_sum]
    _ ≤ φ.support.biUnion fun i : σ →₀ ℕ => ((bind₁ f) (monomial i (coeff i φ))).vars :=
      (vars_sum_subset _ _)
    _ = φ.support.biUnion fun d : σ →₀ ℕ => vars (C (coeff d φ) * ∏ i ∈ d.support, f i ^ d i) := by
      simp only [bind₁_monomial]
    _ ≤ φ.support.biUnion fun d : σ →₀ ℕ => d.support.biUnion fun i => vars (f i) := ?_
    -- proof below
    _ ≤ φ.vars.biUnion fun i : σ => vars (f i) := ?_
    -- proof below
    /-
      case calc_1
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq τ
      f : σ → MvPolynomial τ R
      φ : MvPolynomial σ R
      ⊢ LE.le (φ.support.biUnion fun d => (HMul.hMul (MvPolynomial.C (MvPolynomial.c …
    -/
  · apply Finset.biUnion_mono
    /-
      case calc_1.h
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq τ
      f : σ → MvPolynomial τ R
      φ : MvPolynomial σ R
      ⊢ ∀ (a : Finsupp σ Nat), Membership.mem φ.support a → HasSubset.Subset (HMul.h …
    -/
    intro d _hd
    calc
      vars (C (coeff d φ) * ∏ i ∈ d.support, f i ^ d i) ≤
          (C (coeff d φ)).vars ∪ (∏ i ∈ d.support, f i ^ d i).vars :=
        vars_mul _ _
      _ ≤ (∏ i ∈ d.support, f i ^ d i).vars := by
        simp only [Finset.empty_union, vars_C, Finset.le_iff_subset, Finset.Subset.refl]
      _ ≤ d.support.biUnion fun i : σ => vars (f i ^ d i) := vars_prod _
      _ ≤ d.support.biUnion fun i : σ => (f i).vars := ?_
    /-
      case calc_1.h
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq τ
      f : σ → MvPolynomial τ R
      φ : MvPolynomial σ R
      d : Finsupp σ Nat
      _hd : Membership.mem φ.support d
      ⊢ LE.le (d.support.biUnion fun i => (HPow.hPow (f i) (d i)).vars) (d.support.b …
    -/
    apply Finset.biUnion_mono
    /-
      case calc_1.h.h
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq τ
      f : σ → MvPolynomial τ R
      φ : MvPolynomial σ R
      d : Finsupp σ Nat
      _hd : Membership.mem φ.support d
      ⊢ ∀ (a : σ), Membership.mem d.support a → HasSubset.Subset (HPow.hPow (f a) (d …
    -/
    intro i _hi
    /-
      case calc_1.h.h
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq τ
      f : σ → MvPolynomial τ R
      φ : MvPolynomial σ R
      d : Finsupp σ Nat
      _hd : Membership.mem φ.support d
      i : σ
      _hi : Membership.mem d.support i
      ⊢ HasSubset.Subset (HPow.hPow (f i) (d i)).vars (f i).vars
    -/
    apply vars_pow
    /-
      🎉 no goals
    -/
    /-
      case calc_2
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq τ
      f : σ → MvPolynomial τ R
      φ : MvPolynomial σ R
      ⊢ LE.le (φ.support.biUnion fun d => d.support.biUnion fun i => (f i).vars) (φ. …
    -/
  · intro j
    /-
      case calc_2
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq τ
      f : σ → MvPolynomial τ R
      φ : MvPolynomial σ R
      j : τ
      ⊢ Membership.mem (φ.support.biUnion fun d => d.support.biUnion fun i => (f i). …
    -/
    simp_rw [Finset.mem_biUnion]
    /-
      case calc_2
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq τ
      f : σ → MvPolynomial τ R
      φ : MvPolynomial σ R
      j : τ
      ⊢ (Exists fun a => And (Membership.mem φ.support a) (Exists fun a_1 => And (Me …
    -/
    rintro ⟨d, hd, ⟨i, hi, hj⟩⟩
    /-
      case calc_2.intro.intro.intro.intro
      σ : Type u_1
      τ : Type u_2
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : DecidableEq τ
      f : σ → MvPolynomial τ R
      φ : MvPolynomial σ R
      j : τ
      d : Finsupp σ Nat
      hd : Membership.mem φ.support d
      i : σ
      hi : Membership.mem d.support i
      hj : Membership.mem (f i).vars j
      ⊢ Exists fun a => And (Membership.mem φ.vars a) (Membership.mem (f a).vars j)
    -/
    exact ⟨i, (mem_vars _).mpr ⟨d, hd, hi⟩, hj⟩
    /-
      🎉 no goals
    -/


theorem mem_vars_bind₁ (f : σ → MvPolynomial τ R) (φ : MvPolynomial σ R) {j : τ}
    (h : j ∈ (bind₁ f φ).vars) : ∃ i : σ, i ∈ φ.vars ∧ j ∈ (f i).vars := by
  classical
  simpa only [exists_prop, Finset.mem_biUnion, mem_support_iff, Ne] using vars_bind₁ f φ h


instance monad : Monad fun σ => MvPolynomial σ R where
  map f p := rename f p
  pure := X
  bind p f := bind₁ f p


instance lawfulFunctor : LawfulFunctor fun σ => MvPolynomial σ R where
                  /-
                    σ : Type u_1
                    τ : Type u_2
                    R : Type u_3
                    S : Type u_4
                    T : Type u_5
                    inst✝² : CommSemiring R
                    inst✝¹ : CommSemiring S
                    inst✝ : CommSemiring T
                    f : σ → MvPolynomial τ R
                    ⊢ ∀ {α β : Type u_6}, Eq Functor.mapConst (Function.comp Functor.map (Function …
                  -/
  map_const := by intros; rfl
                          /-
                            🎉 no goals
                          -/
  -- Porting note: I guess `map_const` no longer has a default implementation?
               /-
                 σ : Type u_1
                 τ : Type u_2
                 R : Type u_3
                 S : Type u_4
                 T : Type u_5
                 inst✝² : CommSemiring R
                 inst✝¹ : CommSemiring S
                 inst✝ : CommSemiring T
                 f : σ → MvPolynomial τ R
                 ⊢ ∀ {α : Type u_6} (x : MvPolynomial α R), Eq (Functor.map id x) x
               -/
  id_map := by intros; simp [(· <$> ·)]
                       /-
                         🎉 no goals
                       -/
                 /-
                   σ : Type u_1
                   τ : Type u_2
                   R : Type u_3
                   S : Type u_4
                   T : Type u_5
                   inst✝² : CommSemiring R
                   inst✝¹ : CommSemiring S
                   inst✝ : CommSemiring T
                   f : σ → MvPolynomial τ R
                   ⊢ ∀ {α β γ : Type u_6} (g : α → β) (h : β → γ) (x : MvPolynomial α R), Eq (Fun …
                 -/
  comp_map := by intros; simp [(· <$> ·)]
                         /-
                           🎉 no goals
                         -/


instance lawfulMonad : LawfulMonad fun σ => MvPolynomial σ R where
                  /-
                    σ : Type u_1
                    τ : Type u_2
                    R : Type u_3
                    S : Type u_4
                    T : Type u_5
                    inst✝² : CommSemiring R
                    inst✝¹ : CommSemiring S
                    inst✝ : CommSemiring T
                    f : σ → MvPolynomial τ R
                    ⊢ ∀ {α β : Type u_6} (x : α) (f : α → MvPolynomial β R), Eq (Bind.bind (Pure.p …
                  -/
  pure_bind := by intros; simp [pure, bind]
                   /-
                     σ : Type u_1
                     τ : Type u_2
                     R : Type u_3
                     S : Type u_4
                     T : Type u_5
                     inst✝² : CommSemiring R
                     inst✝¹ : CommSemiring S
                     inst✝ : CommSemiring T
                     f : σ → MvPolynomial τ R
                     ⊢ ∀ {α β : Type u_6} (x : MvPolynomial α R) (y : MvPolynomial β R), Eq (SeqLef …
                   -/
                          /-
                            🎉 no goals
                          -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
                    /-
                      σ : Type u_1
                      τ : Type u_2
                      R : Type u_3
                      S : Type u_4
                      T : Type u_5
                      inst✝² : CommSemiring R
                      inst✝¹ : CommSemiring S
                      inst✝ : CommSemiring T
                      f : σ → MvPolynomial τ R
                      ⊢ ∀ {α β : Type u_6} (x : MvPolynomial α R) (y : MvPolynomial β R), Eq (SeqRig …
                    -/
                   /-
                     σ : Type u_1
                     τ : Type u_2
                     R : Type u_3
                     S : Type u_4
                     T : Type u_5
                     inst✝² : CommSemiring R
                     inst✝¹ : CommSemiring S
                     inst✝ : CommSemiring T
                     f : σ → MvPolynomial τ R
                     ⊢ ∀ {α β γ : Type u_6} (x : MvPolynomial α R) (f : α → MvPolynomial β R) (g :  …
                   -/
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
                 /-
                   σ : Type u_1
                   τ : Type u_2
                   R : Type u_3
                   S : Type u_4
                   T : Type u_5
                   inst✝² : CommSemiring R
                   inst✝¹ : CommSemiring S
                   inst✝ : CommSemiring T
                   f : σ → MvPolynomial τ R
                   ⊢ ∀ {α β : Type u_6} (g : α → β) (x : MvPolynomial α R), Eq (Seq.seq (Pure.pur …
                 -/
  bind_assoc := by intros; simp [bind, ← bind₁_comp_bind₁]
                         /-
                           🎉 no goals
                         -/
                       /-
                         σ : Type u_1
                         τ : Type u_2
                         R : Type u_3
                         S : Type u_4
                         T : Type u_5
                         inst✝² : CommSemiring R
                         inst✝¹ : CommSemiring S
                         inst✝ : CommSemiring T
                         f : σ → MvPolynomial τ R
                         ⊢ ∀ {α β : Type u_6} (f : α → β) (x : MvPolynomial α R), Eq (Bind.bind x fun a …
                       -/
                           /-
                             🎉 no goals
                           -/
                       /-
                         🎉 no goals
                       -/
                 /-
                   σ : Type u_1
                   τ : Type u_2
                   R : Type u_3
                   S : Type u_4
                   T : Type u_5
                   inst✝² : CommSemiring R
                   inst✝¹ : CommSemiring S
                   inst✝ : CommSemiring T
                   f : σ → MvPolynomial τ R
                   ⊢ ∀ {α β : Type u_6} (f : MvPolynomial (α → β) R) (x : MvPolynomial α R), Eq ( …
                 -/
  seqLeft_eq := by intros; simp [SeqLeft.seqLeft, Seq.seq, (· <$> ·), bind₁_rename]; rfl
                 /-
                   🎉 no goals
                 -/
  seqRight_eq := by intros; simp [SeqRight.seqRight, Seq.seq, (· <$> ·), bind₁_rename]; rfl
  pure_seq := by intros; simp [(· <$> ·), pure, Seq.seq]
  bind_pure_comp := by aesop
  bind_map := by aesop

/-
Possible TODO for the future:
Enable the following definitions, and write a lot of supporting lemmas.

def bind (f : R →+* mv_polynomial τ S) (g : σ → mv_polynomial τ S) :
    mv_polynomial σ R →+* mv_polynomial τ S :=
  eval₂_hom f g

def join (f : R →+* S) : mv_polynomial (mv_polynomial σ R) S →ₐ[S] mv_polynomial σ S :=
  aeval (map f)

def ajoin [algebra R S] : mv_polynomial (mv_polynomial σ R) S →ₐ[S] mv_polynomial σ S :=
  join (algebra_map R S)

-/

