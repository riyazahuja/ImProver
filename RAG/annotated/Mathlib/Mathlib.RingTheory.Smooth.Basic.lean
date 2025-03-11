/-- An `R` algebra `A` is formally smooth if for every `R`-algebra, every square-zero ideal
`I : Ideal B` and `f : A →ₐ[R] B ⧸ I`, there exists at least one lift `A →ₐ[R] B`.

See <https://stacks.math.columbia.edu/tag/00TI>.
-/
@[mk_iff]
class FormallySmooth : Prop where
  comp_surjective :
    ∀ ⦃B : Type u⦄ [CommRing B],
      ∀ [Algebra R B] (I : Ideal B) (_ : I ^ 2 = ⊥),
        Function.Surjective ((Ideal.Quotient.mkₐ R I).comp : (A →ₐ[R] B) → A →ₐ[R] B ⧸ I)


theorem exists_lift {B : Type u} [CommRing B] [_RB : Algebra R B]
    [FormallySmooth R A] (I : Ideal B) (hI : IsNilpotent I) (g : A →ₐ[R] B ⧸ I) :
    ∃ f : A →ₐ[R] B, (Ideal.Quotient.mkₐ R I).comp f = g := by
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type u
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    B : Type u
    inst✝¹ : CommRing B
    _RB : Algebra R B
    inst✝ : Algebra.FormallySmooth R A
    I : Ideal B
    hI : IsNilpotent I
    g : AlgHom R A (HasQuotient.Quotient B I)
    ⊢ Exists fun f => Eq ((Ideal.Quotient.mkₐ R I).comp f) g
  -/
  revert g
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type u
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    B : Type u
    inst✝¹ : CommRing B
    _RB : Algebra R B
    inst✝ : Algebra.FormallySmooth R A
    I : Ideal B
    hI : IsNilpotent I
    ⊢ ∀ (g : AlgHom R A (HasQuotient.Quotient B I)), Exists fun f => Eq ((Ideal.Qu …
  -/
  change Function.Surjective (Ideal.Quotient.mkₐ R I).comp
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type u
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    B : Type u
    inst✝¹ : CommRing B
    _RB : Algebra R B
    inst✝ : Algebra.FormallySmooth R A
    I : Ideal B
    hI : IsNilpotent I
    ⊢ Function.Surjective (Ideal.Quotient.mkₐ R I).comp
  -/
  revert _RB
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type u
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    B : Type u
    inst✝¹ : CommRing B
    inst✝ : Algebra.FormallySmooth R A
    I : Ideal B
    hI : IsNilpotent I
    ⊢ ∀ [_RB : Algebra R B], Function.Surjective (Ideal.Quotient.mkₐ R I).comp
  -/
  apply Ideal.IsNilpotent.induction_on (S := B) I hI
    /-
      case h₁
      R : Type u
      inst✝⁴ : CommSemiring R
      A : Type u
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra.FormallySmooth R A
      I : Ideal B
      hI : IsNilpotent I
      ⊢ ∀ ⦃S : Type u⦄ [inst : CommRing S] (I : Ideal S), Eq (HPow.hPow I 2) Bot.bot …
    -/
  · intro B _ I hI _; exact FormallySmooth.comp_surjective I hI
                      /-
                        🎉 no goals
                      -/
    /-
      case h₂
      R : Type u
      inst✝⁴ : CommSemiring R
      A : Type u
      inst✝³ : Semiring A
      inst✝² : Algebra R A
      B : Type u
      inst✝¹ : CommRing B
      inst✝ : Algebra.FormallySmooth R A
      I : Ideal B
      hI : IsNilpotent I
      ⊢ ∀ ⦃S : Type u⦄ [inst : CommRing S] (I J : Ideal S), LE.le I J → (∀ [_RB : Al …
    -/
  · intro B _ I J hIJ h₁ h₂ _ g
    let this : ((B ⧸ I) ⧸ J.map (Ideal.Quotient.mk I)) ≃ₐ[R] B ⧸ J :=
      {
        (DoubleQuot.quotQuotEquivQuotSup I J).trans
          (Ideal.quotEquivOfEq (sup_eq_right.mpr hIJ)) with
        commutes' := fun x => rfl }
    /-
      case h₂
      R : Type u
      inst✝⁵ : CommSemiring R
      A : Type u
      inst✝⁴ : Semiring A
      inst✝³ : Algebra R A
      B✝ : Type u
      inst✝² : CommRing B✝
      inst✝¹ : Algebra.FormallySmooth R A
      I✝ : Ideal B✝
      hI : IsNilpotent I✝
      B : Type u
      inst✝ : CommRing B
      I J : Ideal B
      hIJ : LE.le I J
      h₁ : ∀ [_RB : Algebra R B], Function.Surjective (Ideal.Quotient.mkₐ R I).comp
      h₂ : ∀ [_RB : Algebra R (HasQuotient.Quotient B I)], Function.Surjective (Idea …
      _RB✝ : Algebra R B
      g : AlgHom R A (HasQuotient.Quotient B J)
      this : AlgEquiv R (HasQuotient.Quotient (HasQuotient.Quotient B I) (Ideal.map  …
        let __src := (DoubleQuot.quotQuotEquivQuotSup I J).trans (Ideal.quotEquivOfE …
        { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯, commutes' := ⋯ }
      ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R J).comp a) g
    -/
    obtain ⟨g', e⟩ := h₂ (this.symm.toAlgHom.comp g)
    /-
      case h₂.intro
      R : Type u
      inst✝⁵ : CommSemiring R
      A : Type u
      inst✝⁴ : Semiring A
      inst✝³ : Algebra R A
      B✝ : Type u
      inst✝² : CommRing B✝
      inst✝¹ : Algebra.FormallySmooth R A
      I✝ : Ideal B✝
      hI : IsNilpotent I✝
      B : Type u
      inst✝ : CommRing B
      I J : Ideal B
      hIJ : LE.le I J
      h₁ : ∀ [_RB : Algebra R B], Function.Surjective (Ideal.Quotient.mkₐ R I).comp
      h₂ : ∀ [_RB : Algebra R (HasQuotient.Quotient B I)], Function.Surjective (Idea …
      _RB✝ : Algebra R B
      g : AlgHom R A (HasQuotient.Quotient B J)
      this : AlgEquiv R (HasQuotient.Quotient (HasQuotient.Quotient B I) (Ideal.map  …
        let __src := (DoubleQuot.quotQuotEquivQuotSup I J).trans (Ideal.quotEquivOfE …
        { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯, commutes' := ⋯ }
      g' : AlgHom R A (HasQuotient.Quotient B I)
      e : Eq ((Ideal.Quotient.mkₐ R (Ideal.map (Ideal.Quotient.mk I) J)).comp g') (( …
      ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R J).comp a) g
    -/
    obtain ⟨g', rfl⟩ := h₁ g'
    /-
      case h₂.intro.intro
      R : Type u
      inst✝⁵ : CommSemiring R
      A : Type u
      inst✝⁴ : Semiring A
      inst✝³ : Algebra R A
      B✝ : Type u
      inst✝² : CommRing B✝
      inst✝¹ : Algebra.FormallySmooth R A
      I✝ : Ideal B✝
      hI : IsNilpotent I✝
      B : Type u
      inst✝ : CommRing B
      I J : Ideal B
      hIJ : LE.le I J
      h₁ : ∀ [_RB : Algebra R B], Function.Surjective (Ideal.Quotient.mkₐ R I).comp
      h₂ : ∀ [_RB : Algebra R (HasQuotient.Quotient B I)], Function.Surjective (Idea …
      _RB✝ : Algebra R B
      g : AlgHom R A (HasQuotient.Quotient B J)
      this : AlgEquiv R (HasQuotient.Quotient (HasQuotient.Quotient B I) (Ideal.map  …
        let __src := (DoubleQuot.quotQuotEquivQuotSup I J).trans (Ideal.quotEquivOfE …
        { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯, commutes' := ⋯ }
      g' : AlgHom R A B
      e : Eq ((Ideal.Quotient.mkₐ R (Ideal.map (Ideal.Quotient.mk I) J)).comp ((Idea …
      ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R J).comp a) g
    -/
    replace e := congr_arg this.toAlgHom.comp e
    conv_rhs at e =>
      rw [← AlgHom.comp_assoc, AlgEquiv.toAlgHom_eq_coe, AlgEquiv.toAlgHom_eq_coe,
        AlgEquiv.comp_symm, AlgHom.id_comp]
    /-
      case h₂.intro.intro
      R : Type u
      inst✝⁵ : CommSemiring R
      A : Type u
      inst✝⁴ : Semiring A
      inst✝³ : Algebra R A
      B✝ : Type u
      inst✝² : CommRing B✝
      inst✝¹ : Algebra.FormallySmooth R A
      I✝ : Ideal B✝
      hI : IsNilpotent I✝
      B : Type u
      inst✝ : CommRing B
      I J : Ideal B
      hIJ : LE.le I J
      h₁ : ∀ [_RB : Algebra R B], Function.Surjective (Ideal.Quotient.mkₐ R I).comp
      h₂ : ∀ [_RB : Algebra R (HasQuotient.Quotient B I)], Function.Surjective (Idea …
      _RB✝ : Algebra R B
      g : AlgHom R A (HasQuotient.Quotient B J)
      this : AlgEquiv R (HasQuotient.Quotient (HasQuotient.Quotient B I) (Ideal.map  …
        let __src := (DoubleQuot.quotQuotEquivQuotSup I J).trans (Ideal.quotEquivOfE …
        { toEquiv := __src.toEquiv, map_mul' := ⋯, map_add' := ⋯, commutes' := ⋯ }
      g' : AlgHom R A B
      e : Eq ((↑this).comp ((Ideal.Quotient.mkₐ R (Ideal.map (Ideal.Quotient.mk I) J …
      ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R J).comp a) g
    -/
    exact ⟨g', e⟩
    /-
      🎉 no goals
    -/


/-- For a formally smooth `R`-algebra `A` and a map `f : A →ₐ[R] B ⧸ I` with `I` square-zero,
this is an arbitrary lift `A →ₐ[R] B`. -/
noncomputable def lift [FormallySmooth R A] (I : Ideal B) (hI : IsNilpotent I)
    (g : A →ₐ[R] B ⧸ I) : A →ₐ[R] B :=
  (FormallySmooth.exists_lift I hI g).choose


@[simp]
theorem comp_lift [FormallySmooth R A] (I : Ideal B) (hI : IsNilpotent I)
    (g : A →ₐ[R] B ⧸ I) : (Ideal.Quotient.mkₐ R I).comp (FormallySmooth.lift I hI g) = g :=
  (FormallySmooth.exists_lift I hI g).choose_spec


@[simp]
theorem mk_lift [FormallySmooth R A] (I : Ideal B) (hI : IsNilpotent I)
    (g : A →ₐ[R] B ⧸ I) (x : A) : Ideal.Quotient.mk I (FormallySmooth.lift I hI g x) = g x :=
  AlgHom.congr_fun (FormallySmooth.comp_lift I hI g : _) x


/-- For a formally smooth `R`-algebra `A` and a map `f : A →ₐ[R] B ⧸ I` with `I` nilpotent,
this is an arbitrary lift `A →ₐ[R] B`. -/
noncomputable def liftOfSurjective [FormallySmooth R A] (f : A →ₐ[R] C)
    (g : B →ₐ[R] C) (hg : Function.Surjective g) (hg' : IsNilpotent <| RingHom.ker (g : B →+* C)) :
    A →ₐ[R] B :=
  FormallySmooth.lift _ hg' ((Ideal.quotientKerAlgEquivOfSurjective hg).symm.toAlgHom.comp f)


@[simp]
theorem liftOfSurjective_apply [FormallySmooth R A] (f : A →ₐ[R] C) (g : B →ₐ[R] C)
    (hg : Function.Surjective g) (hg' : IsNilpotent <| RingHom.ker (g : B →+* C)) (x : A) :
    g (FormallySmooth.liftOfSurjective f g hg hg' x) = f x := by
  /-
    R : Type u
    inst✝⁷ : CommSemiring R
    A : Type u
    inst✝⁶ : Semiring A
    inst✝⁵ : Algebra R A
    B : Type u
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    C : Type u
    inst✝² : CommRing C
    inst✝¹ : Algebra R C
    inst✝ : Algebra.FormallySmooth R A
    f : AlgHom R A C
    g : AlgHom R B C
    hg : Function.Surjective ⇑g
    hg' : IsNilpotent (RingHom.ker ↑g)
    x : A
    ⊢ Eq (g ((Algebra.FormallySmooth.liftOfSurjective f g hg hg') x)) (f x)
  -/
  apply (Ideal.quotientKerAlgEquivOfSurjective hg).symm.injective
  /-
    case a
    R : Type u
    inst✝⁷ : CommSemiring R
    A : Type u
    inst✝⁶ : Semiring A
    inst✝⁵ : Algebra R A
    B : Type u
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    C : Type u
    inst✝² : CommRing C
    inst✝¹ : Algebra R C
    inst✝ : Algebra.FormallySmooth R A
    f : AlgHom R A C
    g : AlgHom R B C
    hg : Function.Surjective ⇑g
    hg' : IsNilpotent (RingHom.ker ↑g)
    x : A
    ⊢ Eq ((Ideal.quotientKerAlgEquivOfSurjective hg).symm (g ((Algebra.FormallySmo …
  -/
  change _ = ((Ideal.quotientKerAlgEquivOfSurjective hg).symm.toAlgHom.comp f) x
  -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
  erw [← FormallySmooth.mk_lift _ hg'
    ((Ideal.quotientKerAlgEquivOfSurjective hg).symm.toAlgHom.comp f)]
  /-
    case a
    R : Type u
    inst✝⁷ : CommSemiring R
    A : Type u
    inst✝⁶ : Semiring A
    inst✝⁵ : Algebra R A
    B : Type u
    inst✝⁴ : CommRing B
    inst✝³ : Algebra R B
    C : Type u
    inst✝² : CommRing C
    inst✝¹ : Algebra R C
    inst✝ : Algebra.FormallySmooth R A
    f : AlgHom R A C
    g : AlgHom R B C
    hg : Function.Surjective ⇑g
    hg' : IsNilpotent (RingHom.ker ↑g)
    x : A
    ⊢ Eq ((Ideal.quotientKerAlgEquivOfSurjective hg).symm (g ((Algebra.FormallySmo …
  -/
  apply (Ideal.quotientKerAlgEquivOfSurjective hg).injective
  simp only [liftOfSurjective, AlgEquiv.apply_symm_apply, AlgEquiv.toAlgHom_eq_coe,
    Ideal.quotientKerAlgEquivOfSurjective_apply, RingHom.kerLift_mk, RingHom.coe_coe]


@[simp]
theorem comp_liftOfSurjective [FormallySmooth R A] (f : A →ₐ[R] C) (g : B →ₐ[R] C)
    (hg : Function.Surjective g) (hg' : IsNilpotent <| RingHom.ker (g : B →+* C)) :
    g.comp (FormallySmooth.liftOfSurjective f g hg hg') = f :=
  AlgHom.ext (FormallySmooth.liftOfSurjective_apply f g hg hg')


theorem of_equiv [FormallySmooth R A] (e : A ≃ₐ[R] B) : FormallySmooth R B := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    A B : Type u
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    inst✝² : Semiring B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallySmooth R A
    e : AlgEquiv R A B
    ⊢ Algebra.FormallySmooth R B
  -/
  constructor
  /-
    case comp_surjective
    R : Type u
    inst✝⁵ : CommSemiring R
    A B : Type u
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    inst✝² : Semiring B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallySmooth R A
    e : AlgEquiv R A B
    ⊢ ∀ ⦃B_1 : Type u⦄ [inst : CommRing B_1] [inst_1 : Algebra R B_1] (I : Ideal B …
  -/
  intro C _ _ I hI f
  /-
    case comp_surjective
    R : Type u
    inst✝⁷ : CommSemiring R
    A B : Type u
    inst✝⁶ : Semiring A
    inst✝⁵ : Algebra R A
    inst✝⁴ : Semiring B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallySmooth R A
    e : AlgEquiv R A B
    C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R B (HasQuotient.Quotient C I)
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R I).comp a) f
  -/
  use (FormallySmooth.lift I ⟨2, hI⟩ (f.comp e : A →ₐ[R] C ⧸ I)).comp e.symm
  rw [← AlgHom.comp_assoc, FormallySmooth.comp_lift, AlgHom.comp_assoc, AlgEquiv.comp_symm,
    AlgHom.comp_id]


theorem iff_of_equiv (e : A ≃ₐ[R] B) : FormallySmooth R A ↔ FormallySmooth R B :=
  ⟨fun _ ↦ of_equiv e, fun _ ↦ of_equiv e.symm⟩


instance mvPolynomial (σ : Type u) : FormallySmooth R (MvPolynomial σ R) := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    σ : Type u
    ⊢ Algebra.FormallySmooth R (MvPolynomial σ R)
  -/
  constructor
  /-
    case comp_surjective
    R : Type u
    inst✝ : CommSemiring R
    σ : Type u
    ⊢ ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), Eq  …
  -/
  intro C _ _ I _ f
  have : ∀ s : σ, ∃ c : C, Ideal.Quotient.mk I c = f (MvPolynomial.X s) := fun s =>
    Ideal.Quotient.mk_surjective _
  /-
    case comp_surjective
    R : Type u
    inst✝² : CommSemiring R
    σ C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    x✝ : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R (MvPolynomial σ R) (HasQuotient.Quotient C I)
    this : ∀ (s : σ), Exists fun c => Eq ((Ideal.Quotient.mk I) c) (f (MvPolynomia …
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R I).comp a) f
  -/
  choose g hg using this
  /-
    case comp_surjective
    R : Type u
    inst✝² : CommSemiring R
    σ C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    x✝ : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R (MvPolynomial σ R) (HasQuotient.Quotient C I)
    g : σ → C
    hg : ∀ (s : σ), Eq ((Ideal.Quotient.mk I) (g s)) (f (MvPolynomial.X s))
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R I).comp a) f
  -/
  refine ⟨MvPolynomial.aeval g, ?_⟩
  /-
    case comp_surjective
    R : Type u
    inst✝² : CommSemiring R
    σ C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    x✝ : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R (MvPolynomial σ R) (HasQuotient.Quotient C I)
    g : σ → C
    hg : ∀ (s : σ), Eq ((Ideal.Quotient.mk I) (g s)) (f (MvPolynomial.X s))
    ⊢ Eq ((Ideal.Quotient.mkₐ R I).comp (MvPolynomial.aeval g)) f
  -/
  ext s
  /-
    case comp_surjective.hf
    R : Type u
    inst✝² : CommSemiring R
    σ C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    x✝ : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R (MvPolynomial σ R) (HasQuotient.Quotient C I)
    g : σ → C
    hg : ∀ (s : σ), Eq ((Ideal.Quotient.mk I) (g s)) (f (MvPolynomial.X s))
    s : σ
    ⊢ Eq (((Ideal.Quotient.mkₐ R I).comp (MvPolynomial.aeval g)) (MvPolynomial.X s …
  -/
  rw [← hg, AlgHom.comp_apply, MvPolynomial.aeval_X]
  /-
    case comp_surjective.hf
    R : Type u
    inst✝² : CommSemiring R
    σ C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    x✝ : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R (MvPolynomial σ R) (HasQuotient.Quotient C I)
    g : σ → C
    hg : ∀ (s : σ), Eq ((Ideal.Quotient.mk I) (g s)) (f (MvPolynomial.X s))
    s : σ
    ⊢ Eq ((Ideal.Quotient.mkₐ R I) (g s)) ((Ideal.Quotient.mk I) (g s))
  -/
  rfl
  /-
    🎉 no goals
  -/


instance polynomial : FormallySmooth R R[X] :=
  FormallySmooth.of_equiv (MvPolynomial.pUnitAlgEquiv R)


theorem comp [FormallySmooth R A] [FormallySmooth A B] : FormallySmooth R B := by
  /-
    R : Type u
    inst✝⁸ : CommSemiring R
    A : Type u
    inst✝⁷ : CommSemiring A
    inst✝⁶ : Algebra R A
    B : Type u
    inst✝⁵ : Semiring B
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : IsScalarTower R A B
    inst✝¹ : Algebra.FormallySmooth R A
    inst✝ : Algebra.FormallySmooth A B
    ⊢ Algebra.FormallySmooth R B
  -/
  constructor
  /-
    case comp_surjective
    R : Type u
    inst✝⁸ : CommSemiring R
    A : Type u
    inst✝⁷ : CommSemiring A
    inst✝⁶ : Algebra R A
    B : Type u
    inst✝⁵ : Semiring B
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : IsScalarTower R A B
    inst✝¹ : Algebra.FormallySmooth R A
    inst✝ : Algebra.FormallySmooth A B
    ⊢ ∀ ⦃B_1 : Type u⦄ [inst : CommRing B_1] [inst_1 : Algebra R B_1] (I : Ideal B …
  -/
  intro C _ _ I hI f
  /-
    case comp_surjective
    R : Type u
    inst✝¹⁰ : CommSemiring R
    A : Type u
    inst✝⁹ : CommSemiring A
    inst✝⁸ : Algebra R A
    B : Type u
    inst✝⁷ : Semiring B
    inst✝⁶ : Algebra R B
    inst✝⁵ : Algebra A B
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra.FormallySmooth R A
    inst✝² : Algebra.FormallySmooth A B
    C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R B (HasQuotient.Quotient C I)
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R I).comp a) f
  -/
  obtain ⟨f', e⟩ := FormallySmooth.comp_surjective I hI (f.comp (IsScalarTower.toAlgHom R A B))
  /-
    case comp_surjective.intro
    R : Type u
    inst✝¹⁰ : CommSemiring R
    A : Type u
    inst✝⁹ : CommSemiring A
    inst✝⁸ : Algebra R A
    B : Type u
    inst✝⁷ : Semiring B
    inst✝⁶ : Algebra R B
    inst✝⁵ : Algebra A B
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra.FormallySmooth R A
    inst✝² : Algebra.FormallySmooth A B
    C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R B (HasQuotient.Quotient C I)
    f' : AlgHom R A C
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f') (f.comp (IsScalarTower.toAlgHom R A  …
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R I).comp a) f
  -/
  letI := f'.toRingHom.toAlgebra
  obtain ⟨f'', e'⟩ :=
    FormallySmooth.comp_surjective I hI { f.toRingHom with commutes' := AlgHom.congr_fun e.symm }
  /-
    case comp_surjective.intro.intro
    R : Type u
    inst✝¹⁰ : CommSemiring R
    A : Type u
    inst✝⁹ : CommSemiring A
    inst✝⁸ : Algebra R A
    B : Type u
    inst✝⁷ : Semiring B
    inst✝⁶ : Algebra R B
    inst✝⁵ : Algebra A B
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra.FormallySmooth R A
    inst✝² : Algebra.FormallySmooth A B
    C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R B (HasQuotient.Quotient C I)
    f' : AlgHom R A C
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f') (f.comp (IsScalarTower.toAlgHom R A  …
    this : Algebra A C := f'.toAlgebra
    f'' : AlgHom A B C
    e' :
      Eq ((Ideal.Quotient.mkₐ A I).comp f'')
        (let __src := f.toRingHom;
        { toRingHom := __src, commutes' := ⋯ })
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R I).comp a) f
  -/
  apply_fun AlgHom.restrictScalars R at e'
  /-
    case comp_surjective.intro.intro
    R : Type u
    inst✝¹⁰ : CommSemiring R
    A : Type u
    inst✝⁹ : CommSemiring A
    inst✝⁸ : Algebra R A
    B : Type u
    inst✝⁷ : Semiring B
    inst✝⁶ : Algebra R B
    inst✝⁵ : Algebra A B
    inst✝⁴ : IsScalarTower R A B
    inst✝³ : Algebra.FormallySmooth R A
    inst✝² : Algebra.FormallySmooth A B
    C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R B (HasQuotient.Quotient C I)
    f' : AlgHom R A C
    e : Eq ((Ideal.Quotient.mkₐ R I).comp f') (f.comp (IsScalarTower.toAlgHom R A  …
    this : Algebra A C := f'.toAlgebra
    f'' : AlgHom A B C
    e' :
      Eq (AlgHom.restrictScalars R ((Ideal.Quotient.mkₐ A I).comp f''))
        (AlgHom.restrictScalars R
          (let __src := f.toRingHom;
          { toRingHom := __src, commutes' := ⋯ }))
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R I).comp a) f
  -/
  exact ⟨f''.restrictScalars _, e'.trans (AlgHom.ext fun _ => rfl)⟩
  /-
    🎉 no goals
  -/



theorem of_split [FormallySmooth R P] (g : A →ₐ[R] P ⧸ (RingHom.ker f.toRingHom) ^ 2)
    (hg : f.kerSquareLift.comp g = AlgHom.id R A) : FormallySmooth R A := by
  /-
    R : Type u
    inst✝⁵ : CommRing R
    P A : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    f : AlgHom R P A
    inst✝ : Algebra.FormallySmooth R P
    g : AlgHom R A (HasQuotient.Quotient P (HPow.hPow (RingHom.ker f.toRingHom) 2))
    hg : Eq (f.kerSquareLift.comp g) (AlgHom.id R A)
    ⊢ Algebra.FormallySmooth R A
  -/
  constructor
  /-
    case comp_surjective
    R : Type u
    inst✝⁵ : CommRing R
    P A : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    f : AlgHom R P A
    inst✝ : Algebra.FormallySmooth R P
    g : AlgHom R A (HasQuotient.Quotient P (HPow.hPow (RingHom.ker f.toRingHom) 2))
    hg : Eq (f.kerSquareLift.comp g) (AlgHom.id R A)
    ⊢ ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), Eq  …
  -/
  intro C _ _ I hI i
  let l : P ⧸ (RingHom.ker f.toRingHom) ^ 2 →ₐ[R] C := by
    refine Ideal.Quotient.liftₐ _ (FormallySmooth.lift I ⟨2, hI⟩ (i.comp f)) ?_
    have : RingHom.ker f ≤ I.comap (FormallySmooth.lift I ⟨2, hI⟩ (i.comp f)) := by
      rintro x (hx : f x = 0)
      have : _ = i (f x) := (FormallySmooth.mk_lift I ⟨2, hI⟩ (i.comp f) x : _)
      rwa [hx, map_zero, ← Ideal.Quotient.mk_eq_mk, Submodule.Quotient.mk_eq_zero] at this
    intro x hx
    have := (Ideal.pow_right_mono this 2).trans (Ideal.le_comap_pow _ 2) hx
    rwa [hI] at this
  have : i.comp f.kerSquareLift = (Ideal.Quotient.mkₐ R _).comp l := by
    apply AlgHom.coe_ringHom_injective
    apply Ideal.Quotient.ringHom_ext
    ext x
    exact (FormallySmooth.mk_lift I ⟨2, hI⟩ (i.comp f) x).symm
  /-
    case comp_surjective
    R : Type u
    inst✝⁷ : CommRing R
    P A : Type u
    inst✝⁶ : CommRing A
    inst✝⁵ : Algebra R A
    inst✝⁴ : CommRing P
    inst✝³ : Algebra R P
    f : AlgHom R P A
    inst✝² : Algebra.FormallySmooth R P
    g : AlgHom R A (HasQuotient.Quotient P (HPow.hPow (RingHom.ker f.toRingHom) 2))
    hg : Eq (f.kerSquareLift.comp g) (AlgHom.id R A)
    C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra R C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    i : AlgHom R A (HasQuotient.Quotient C I)
    l : AlgHom R (HasQuotient.Quotient P (HPow.hPow (RingHom.ker f.toRingHom) 2))  …
    this : Eq (i.comp f.kerSquareLift) ((Ideal.Quotient.mkₐ R I).comp l)
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R I).comp a) i
  -/
  exact ⟨l.comp g, by rw [← AlgHom.comp_assoc, ← this, AlgHom.comp_assoc, hg, AlgHom.comp_id]⟩
  /-
    🎉 no goals
  -/


/-- Let `P →ₐ[R] A` be a surjection with kernel `J`, and `P` a formally smooth `R`-algebra,
then `A` is formally smooth over `R` iff the surjection `P ⧸ J ^ 2 →ₐ[R] A` has a section.

Geometric intuition: we require that a first-order thickening of `Spec A` inside `Spec P` admits
a retraction. -/
theorem iff_split_surjection [FormallySmooth R P] :
    FormallySmooth R A ↔ ∃ g, f.kerSquareLift.comp g = AlgHom.id R A := by
  /-
    R : Type u
    inst✝⁵ : CommRing R
    P A : Type u
    inst✝⁴ : CommRing A
    inst✝³ : Algebra R A
    inst✝² : CommRing P
    inst✝¹ : Algebra R P
    f : AlgHom R P A
    hf : Function.Surjective ⇑f
    inst✝ : Algebra.FormallySmooth R P
    ⊢ Iff (Algebra.FormallySmooth R A) (Exists fun g => Eq (f.kerSquareLift.comp g …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝⁵ : CommRing R
      P A : Type u
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      f : AlgHom R P A
      hf : Function.Surjective ⇑f
      inst✝ : Algebra.FormallySmooth R P
      ⊢ Algebra.FormallySmooth R A → Exists fun g => Eq (f.kerSquareLift.comp g) (Al …
    -/
  · intro
    have surj : Function.Surjective f.kerSquareLift := fun x =>
      ⟨Submodule.Quotient.mk (hf x).choose, (hf x).choose_spec⟩
    have sqz : RingHom.ker f.kerSquareLift.toRingHom ^ 2 = 0 := by
      rw [AlgHom.ker_kerSquareLift, Ideal.cotangentIdeal_square, Ideal.zero_eq_bot]
    refine
      ⟨FormallySmooth.lift _ ⟨2, sqz⟩ (Ideal.quotientKerAlgEquivOfSurjective surj).symm.toAlgHom,
        ?_⟩
    /-
      case mp
      R : Type u
      inst✝⁵ : CommRing R
      P A : Type u
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      f : AlgHom R P A
      hf : Function.Surjective ⇑f
      inst✝ : Algebra.FormallySmooth R P
      a✝ : Algebra.FormallySmooth R A
      surj : Function.Surjective ⇑f.kerSquareLift
      sqz : Eq (HPow.hPow (RingHom.ker f.kerSquareLift.toRingHom) 2) 0
      ⊢ Eq (f.kerSquareLift.comp (Algebra.FormallySmooth.lift (RingHom.ker f.kerSqua …
    -/
    ext x
    have :=
      (Ideal.quotientKerAlgEquivOfSurjective surj).toAlgHom.congr_arg
        (FormallySmooth.mk_lift _ ⟨2, sqz⟩
          (Ideal.quotientKerAlgEquivOfSurjective surj).symm.toAlgHom x)
    -- Porting note: was
    -- dsimp at this
    -- rw [AlgEquiv.apply_symm_apply] at this
    /-
      case mp.H
      R : Type u
      inst✝⁵ : CommRing R
      P A : Type u
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      f : AlgHom R P A
      hf : Function.Surjective ⇑f
      inst✝ : Algebra.FormallySmooth R P
      a✝ : Algebra.FormallySmooth R A
      surj : Function.Surjective ⇑f.kerSquareLift
      sqz : Eq (HPow.hPow (RingHom.ker f.kerSquareLift.toRingHom) 2) 0
      x : A
      this : Eq (↑(Ideal.quotientKerAlgEquivOfSurjective surj) ((Ideal.Quotient.mk ( …
      ⊢ Eq ((f.kerSquareLift.comp (Algebra.FormallySmooth.lift (RingHom.ker f.kerSqu …
    -/
    erw [AlgEquiv.apply_symm_apply] at this
    /-
      case mp.H
      R : Type u
      inst✝⁵ : CommRing R
      P A : Type u
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      f : AlgHom R P A
      hf : Function.Surjective ⇑f
      inst✝ : Algebra.FormallySmooth R P
      a✝ : Algebra.FormallySmooth R A
      surj : Function.Surjective ⇑f.kerSquareLift
      sqz : Eq (HPow.hPow (RingHom.ker f.kerSquareLift.toRingHom) 2) 0
      x : A
      this : Eq (↑(Ideal.quotientKerAlgEquivOfSurjective surj) ((Ideal.Quotient.mk ( …
      ⊢ Eq ((f.kerSquareLift.comp (Algebra.FormallySmooth.lift (RingHom.ker f.kerSqu …
    -/
    conv_rhs => rw [← this, AlgHom.id_apply]
    /-
      case mp.H
      R : Type u
      inst✝⁵ : CommRing R
      P A : Type u
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      f : AlgHom R P A
      hf : Function.Surjective ⇑f
      inst✝ : Algebra.FormallySmooth R P
      a✝ : Algebra.FormallySmooth R A
      surj : Function.Surjective ⇑f.kerSquareLift
      sqz : Eq (HPow.hPow (RingHom.ker f.kerSquareLift.toRingHom) 2) 0
      x : A
      this : Eq (↑(Ideal.quotientKerAlgEquivOfSurjective surj) ((Ideal.Quotient.mk ( …
      ⊢ Eq ((f.kerSquareLift.comp (Algebra.FormallySmooth.lift (RingHom.ker f.kerSqu …
    -/
    rfl
    /-
      🎉 no goals
    -/
    -- Porting note: lean3 was not finished here:
    -- obtain ⟨y, e⟩ :=
    --   Ideal.Quotient.mk_surjective
    --     (FormallySmooth.lift _ ⟨2, sqz⟩
    --       (Ideal.quotientKerAlgEquivOfSurjective surj).symm.toAlgHom
    --       x)
    -- dsimp at e ⊢
    -- rw [← e]
    -- rfl
    /-
      case mpr
      R : Type u
      inst✝⁵ : CommRing R
      P A : Type u
      inst✝⁴ : CommRing A
      inst✝³ : Algebra R A
      inst✝² : CommRing P
      inst✝¹ : Algebra R P
      f : AlgHom R P A
      hf : Function.Surjective ⇑f
      inst✝ : Algebra.FormallySmooth R P
      ⊢ (Exists fun g => Eq (f.kerSquareLift.comp g) (AlgHom.id R A)) → Algebra.Form …
    -/
  · rintro ⟨g, hg⟩; exact FormallySmooth.of_split f g hg
                    /-
                      🎉 no goals
                    -/


instance base_change [FormallySmooth R A] : FormallySmooth B (B ⊗[R] A) := by
  /-
    R : Type u
    inst✝⁵ : CommSemiring R
    A : Type u
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    B : Type u
    inst✝² : CommSemiring B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallySmooth R A
    ⊢ Algebra.FormallySmooth B (TensorProduct R B A)
  -/
  constructor
  /-
    case comp_surjective
    R : Type u
    inst✝⁵ : CommSemiring R
    A : Type u
    inst✝⁴ : Semiring A
    inst✝³ : Algebra R A
    B : Type u
    inst✝² : CommSemiring B
    inst✝¹ : Algebra R B
    inst✝ : Algebra.FormallySmooth R A
    ⊢ ∀ ⦃B_1 : Type u⦄ [inst : CommRing B_1] [inst_1 : Algebra B B_1] (I : Ideal B …
  -/
  intro C _ _ I hI f
  /-
    case comp_surjective
    R : Type u
    inst✝⁷ : CommSemiring R
    A : Type u
    inst✝⁶ : Semiring A
    inst✝⁵ : Algebra R A
    B : Type u
    inst✝⁴ : CommSemiring B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallySmooth R A
    C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra B C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom B (TensorProduct R B A) (HasQuotient.Quotient C I)
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ B I).comp a) f
  -/
  letI := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
  /-
    case comp_surjective
    R : Type u
    inst✝⁷ : CommSemiring R
    A : Type u
    inst✝⁶ : Semiring A
    inst✝⁵ : Algebra R A
    B : Type u
    inst✝⁴ : CommSemiring B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallySmooth R A
    C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra B C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom B (TensorProduct R B A) (HasQuotient.Quotient C I)
    this : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ B I).comp a) f
  -/
  haveI : IsScalarTower R B C := IsScalarTower.of_algebraMap_eq' rfl
  /-
    case comp_surjective
    R : Type u
    inst✝⁷ : CommSemiring R
    A : Type u
    inst✝⁶ : Semiring A
    inst✝⁵ : Algebra R A
    B : Type u
    inst✝⁴ : CommSemiring B
    inst✝³ : Algebra R B
    inst✝² : Algebra.FormallySmooth R A
    C : Type u
    inst✝¹ : CommRing C
    inst✝ : Algebra B C
    I : Ideal C
    hI : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom B (TensorProduct R B A) (HasQuotient.Quotient C I)
    this✝ : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
    this : IsScalarTower R B C
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ B I).comp a) f
  -/
  refine ⟨TensorProduct.productLeftAlgHom (Algebra.ofId B C) ?_, ?_⟩
    /-
      case comp_surjective.refine_1
      R : Type u
      inst✝⁷ : CommSemiring R
      A : Type u
      inst✝⁶ : Semiring A
      inst✝⁵ : Algebra R A
      B : Type u
      inst✝⁴ : CommSemiring B
      inst✝³ : Algebra R B
      inst✝² : Algebra.FormallySmooth R A
      C : Type u
      inst✝¹ : CommRing C
      inst✝ : Algebra B C
      I : Ideal C
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom B (TensorProduct R B A) (HasQuotient.Quotient C I)
      this✝ : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
      this : IsScalarTower R B C
      ⊢ AlgHom R A C
    -/
  · exact FormallySmooth.lift I ⟨2, hI⟩ ((f.restrictScalars R).comp TensorProduct.includeRight)
    /-
      🎉 no goals
    -/
    /-
      case comp_surjective.refine_2
      R : Type u
      inst✝⁷ : CommSemiring R
      A : Type u
      inst✝⁶ : Semiring A
      inst✝⁵ : Algebra R A
      B : Type u
      inst✝⁴ : CommSemiring B
      inst✝³ : Algebra R B
      inst✝² : Algebra.FormallySmooth R A
      C : Type u
      inst✝¹ : CommRing C
      inst✝ : Algebra B C
      I : Ideal C
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom B (TensorProduct R B A) (HasQuotient.Quotient C I)
      this✝ : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
      this : IsScalarTower R B C
      ⊢ Eq ((Ideal.Quotient.mkₐ B I).comp (Algebra.TensorProduct.productLeftAlgHom ( …
    -/
  · apply AlgHom.restrictScalars_injective R
    /-
      case comp_surjective.refine_2.a
      R : Type u
      inst✝⁷ : CommSemiring R
      A : Type u
      inst✝⁶ : Semiring A
      inst✝⁵ : Algebra R A
      B : Type u
      inst✝⁴ : CommSemiring B
      inst✝³ : Algebra R B
      inst✝² : Algebra.FormallySmooth R A
      C : Type u
      inst✝¹ : CommRing C
      inst✝ : Algebra B C
      I : Ideal C
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom B (TensorProduct R B A) (HasQuotient.Quotient C I)
      this✝ : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
      this : IsScalarTower R B C
      ⊢ Eq (AlgHom.restrictScalars R ((Ideal.Quotient.mkₐ B I).comp (Algebra.TensorP …
    -/
    apply TensorProduct.ext'
    /-
      case comp_surjective.refine_2.a.H
      R : Type u
      inst✝⁷ : CommSemiring R
      A : Type u
      inst✝⁶ : Semiring A
      inst✝⁵ : Algebra R A
      B : Type u
      inst✝⁴ : CommSemiring B
      inst✝³ : Algebra R B
      inst✝² : Algebra.FormallySmooth R A
      C : Type u
      inst✝¹ : CommRing C
      inst✝ : Algebra B C
      I : Ideal C
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom B (TensorProduct R B A) (HasQuotient.Quotient C I)
      this✝ : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
      this : IsScalarTower R B C
      ⊢ ∀ (a : B) (b : A), Eq ((AlgHom.restrictScalars R ((Ideal.Quotient.mkₐ B I).c …
    -/
    intro b a
    /-
      case comp_surjective.refine_2.a.H
      R : Type u
      inst✝⁷ : CommSemiring R
      A : Type u
      inst✝⁶ : Semiring A
      inst✝⁵ : Algebra R A
      B : Type u
      inst✝⁴ : CommSemiring B
      inst✝³ : Algebra R B
      inst✝² : Algebra.FormallySmooth R A
      C : Type u
      inst✝¹ : CommRing C
      inst✝ : Algebra B C
      I : Ideal C
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom B (TensorProduct R B A) (HasQuotient.Quotient C I)
      this✝ : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
      this : IsScalarTower R B C
      b : B
      a : A
      ⊢ Eq ((AlgHom.restrictScalars R ((Ideal.Quotient.mkₐ B I).comp (Algebra.Tensor …
    -/
    suffices algebraMap B _ b * f (1 ⊗ₜ[R] a) = f (b ⊗ₜ[R] a) by simpa [Algebra.ofId_apply]
    /-
      case comp_surjective.refine_2.a.H
      R : Type u
      inst✝⁷ : CommSemiring R
      A : Type u
      inst✝⁶ : Semiring A
      inst✝⁵ : Algebra R A
      B : Type u
      inst✝⁴ : CommSemiring B
      inst✝³ : Algebra R B
      inst✝² : Algebra.FormallySmooth R A
      C : Type u
      inst✝¹ : CommRing C
      inst✝ : Algebra B C
      I : Ideal C
      hI : Eq (HPow.hPow I 2) Bot.bot
      f : AlgHom B (TensorProduct R B A) (HasQuotient.Quotient C I)
      this✝ : Algebra R C := ((algebraMap B C).comp (algebraMap R B)).toAlgebra
      this : IsScalarTower R B C
      b : B
      a : A
      ⊢ Eq (HMul.hMul ((algebraMap B (HasQuotient.Quotient C I)) b) (f (TensorProduc …
    -/
    rw [← Algebra.smul_def, ← map_smul, TensorProduct.smul_tmul', smul_eq_mul, mul_one]
    /-
      🎉 no goals
    -/


theorem of_isLocalization : FormallySmooth R Rₘ := by
  /-
    R Rₘ : Type u
    inst✝³ : CommRing R
    inst✝² : CommRing Rₘ
    M : Submonoid R
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    ⊢ Algebra.FormallySmooth R Rₘ
  -/
  constructor
  /-
    case comp_surjective
    R Rₘ : Type u
    inst✝³ : CommRing R
    inst✝² : CommRing Rₘ
    M : Submonoid R
    inst✝¹ : Algebra R Rₘ
    inst✝ : IsLocalization M Rₘ
    ⊢ ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra R B] (I : Ideal B), Eq  …
  -/
  intro Q _ _ I e f
  have : ∀ x : M, IsUnit (algebraMap R Q x) := by
    intro x
    apply (IsNilpotent.isUnit_quotient_mk_iff ⟨2, e⟩).mp
    convert (IsLocalization.map_units Rₘ x).map f
    simp only [Ideal.Quotient.mk_algebraMap, AlgHom.commutes]
  let this : Rₘ →ₐ[R] Q :=
    { IsLocalization.lift this with commutes' := IsLocalization.lift_eq this }
  /-
    case comp_surjective
    R Rₘ : Type u
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing Rₘ
    M : Submonoid R
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    Q : Type u
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R Rₘ (HasQuotient.Quotient Q I)
    this✝ : ∀ (x : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap R Q)  …
    this : AlgHom R Rₘ Q :=
      let __src := IsLocalization.lift this✝;
      { toRingHom := __src, commutes' := ⋯ }
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ R I).comp a) f
  -/
  use this
  /-
    case h
    R Rₘ : Type u
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing Rₘ
    M : Submonoid R
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    Q : Type u
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R Rₘ (HasQuotient.Quotient Q I)
    this✝ : ∀ (x : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap R Q)  …
    this : AlgHom R Rₘ Q :=
      let __src := IsLocalization.lift this✝;
      { toRingHom := __src, commutes' := ⋯ }
    ⊢ Eq ((Ideal.Quotient.mkₐ R I).comp this) f
  -/
  apply AlgHom.coe_ringHom_injective
  /-
    case h.a
    R Rₘ : Type u
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing Rₘ
    M : Submonoid R
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    Q : Type u
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R Rₘ (HasQuotient.Quotient Q I)
    this✝ : ∀ (x : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap R Q)  …
    this : AlgHom R Rₘ Q :=
      let __src := IsLocalization.lift this✝;
      { toRingHom := __src, commutes' := ⋯ }
    ⊢ Eq ↑((Ideal.Quotient.mkₐ R I).comp this) ↑f
  -/
  refine IsLocalization.ringHom_ext M ?_
  /-
    case h.a
    R Rₘ : Type u
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing Rₘ
    M : Submonoid R
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    Q : Type u
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R Rₘ (HasQuotient.Quotient Q I)
    this✝ : ∀ (x : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap R Q)  …
    this : AlgHom R Rₘ Q :=
      let __src := IsLocalization.lift this✝;
      { toRingHom := __src, commutes' := ⋯ }
    ⊢ Eq ((↑((Ideal.Quotient.mkₐ R I).comp this)).comp (algebraMap R Rₘ)) ((↑f).co …
  -/
  ext
  /-
    case h.a.a
    R Rₘ : Type u
    inst✝⁵ : CommRing R
    inst✝⁴ : CommRing Rₘ
    M : Submonoid R
    inst✝³ : Algebra R Rₘ
    inst✝² : IsLocalization M Rₘ
    Q : Type u
    inst✝¹ : CommRing Q
    inst✝ : Algebra R Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom R Rₘ (HasQuotient.Quotient Q I)
    this✝ : ∀ (x : Subtype fun x => Membership.mem M x), IsUnit ((algebraMap R Q)  …
    this : AlgHom R Rₘ Q :=
      let __src := IsLocalization.lift this✝;
      { toRingHom := __src, commutes' := ⋯ }
    x✝ : R
    ⊢ Eq (((↑((Ideal.Quotient.mkₐ R I).comp this)).comp (algebraMap R Rₘ)) x✝) ((( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem localization_base [FormallySmooth R Sₘ] : FormallySmooth Rₘ Sₘ := by
  /-
    R Rₘ Sₘ : Type u
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing Rₘ
    inst✝⁶ : CommRing Sₘ
    M : Submonoid R
    inst✝⁵ : Algebra R Sₘ
    inst✝⁴ : Algebra R Rₘ
    inst✝³ : Algebra Rₘ Sₘ
    inst✝² : IsScalarTower R Rₘ Sₘ
    inst✝¹ : IsLocalization M Rₘ
    inst✝ : Algebra.FormallySmooth R Sₘ
    ⊢ Algebra.FormallySmooth Rₘ Sₘ
  -/
  constructor
  /-
    case comp_surjective
    R Rₘ Sₘ : Type u
    inst✝⁸ : CommRing R
    inst✝⁷ : CommRing Rₘ
    inst✝⁶ : CommRing Sₘ
    M : Submonoid R
    inst✝⁵ : Algebra R Sₘ
    inst✝⁴ : Algebra R Rₘ
    inst✝³ : Algebra Rₘ Sₘ
    inst✝² : IsScalarTower R Rₘ Sₘ
    inst✝¹ : IsLocalization M Rₘ
    inst✝ : Algebra.FormallySmooth R Sₘ
    ⊢ ∀ ⦃B : Type u⦄ [inst : CommRing B] [inst_1 : Algebra Rₘ B] (I : Ideal B), Eq …
  -/
  intro Q _ _ I e f
  /-
    case comp_surjective
    R Rₘ Sₘ : Type u
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing Rₘ
    inst✝⁸ : CommRing Sₘ
    M : Submonoid R
    inst✝⁷ : Algebra R Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsLocalization M Rₘ
    inst✝² : Algebra.FormallySmooth R Sₘ
    Q : Type u
    inst✝¹ : CommRing Q
    inst✝ : Algebra Rₘ Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom Rₘ Sₘ (HasQuotient.Quotient Q I)
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ Rₘ I).comp a) f
  -/
  letI := ((algebraMap Rₘ Q).comp (algebraMap R Rₘ)).toAlgebra
  /-
    case comp_surjective
    R Rₘ Sₘ : Type u
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing Rₘ
    inst✝⁸ : CommRing Sₘ
    M : Submonoid R
    inst✝⁷ : Algebra R Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsLocalization M Rₘ
    inst✝² : Algebra.FormallySmooth R Sₘ
    Q : Type u
    inst✝¹ : CommRing Q
    inst✝ : Algebra Rₘ Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f : AlgHom Rₘ Sₘ (HasQuotient.Quotient Q I)
    this : Algebra R Q := ((algebraMap Rₘ Q).comp (algebraMap R Rₘ)).toAlgebra
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ Rₘ I).comp a) f
  -/
  letI : IsScalarTower R Rₘ Q := IsScalarTower.of_algebraMap_eq' rfl
  let f : Sₘ →ₐ[Rₘ] Q := by
    refine { FormallySmooth.lift I ⟨2, e⟩ (f.restrictScalars R) with commutes' := ?_ }
    intro r
    change
      (RingHom.comp (FormallySmooth.lift I ⟨2, e⟩ (f.restrictScalars R) : Sₘ →+* Q)
            (algebraMap _ _))
          r =
        algebraMap _ _ r
    congr 1
    refine IsLocalization.ringHom_ext M ?_
    rw [RingHom.comp_assoc, ← IsScalarTower.algebraMap_eq, ← IsScalarTower.algebraMap_eq,
      AlgHom.comp_algebraMap]
  /-
    case comp_surjective
    R Rₘ Sₘ : Type u
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing Rₘ
    inst✝⁸ : CommRing Sₘ
    M : Submonoid R
    inst✝⁷ : Algebra R Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsLocalization M Rₘ
    inst✝² : Algebra.FormallySmooth R Sₘ
    Q : Type u
    inst✝¹ : CommRing Q
    inst✝ : Algebra Rₘ Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f✝ : AlgHom Rₘ Sₘ (HasQuotient.Quotient Q I)
    this✝ : Algebra R Q := ((algebraMap Rₘ Q).comp (algebraMap R Rₘ)).toAlgebra
    this : IsScalarTower R Rₘ Q := IsScalarTower.of_algebraMap_eq' rfl
    f : AlgHom Rₘ Sₘ Q :=
      let __src := Algebra.FormallySmooth.lift I ⋯ (AlgHom.restrictScalars R f✝);
      { toRingHom := __src.toRingHom, commutes' := ⋯ }
    ⊢ Exists fun a => Eq ((Ideal.Quotient.mkₐ Rₘ I).comp a) f✝
  -/
  use f
  /-
    case h
    R Rₘ Sₘ : Type u
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing Rₘ
    inst✝⁸ : CommRing Sₘ
    M : Submonoid R
    inst✝⁷ : Algebra R Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsLocalization M Rₘ
    inst✝² : Algebra.FormallySmooth R Sₘ
    Q : Type u
    inst✝¹ : CommRing Q
    inst✝ : Algebra Rₘ Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f✝ : AlgHom Rₘ Sₘ (HasQuotient.Quotient Q I)
    this✝ : Algebra R Q := ((algebraMap Rₘ Q).comp (algebraMap R Rₘ)).toAlgebra
    this : IsScalarTower R Rₘ Q := IsScalarTower.of_algebraMap_eq' rfl
    f : AlgHom Rₘ Sₘ Q :=
      let __src := Algebra.FormallySmooth.lift I ⋯ (AlgHom.restrictScalars R f✝);
      { toRingHom := __src.toRingHom, commutes' := ⋯ }
    ⊢ Eq ((Ideal.Quotient.mkₐ Rₘ I).comp f) f✝
  -/
  ext
  /-
    case h.H
    R Rₘ Sₘ : Type u
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing Rₘ
    inst✝⁸ : CommRing Sₘ
    M : Submonoid R
    inst✝⁷ : Algebra R Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsLocalization M Rₘ
    inst✝² : Algebra.FormallySmooth R Sₘ
    Q : Type u
    inst✝¹ : CommRing Q
    inst✝ : Algebra Rₘ Q
    I : Ideal Q
    e : Eq (HPow.hPow I 2) Bot.bot
    f✝ : AlgHom Rₘ Sₘ (HasQuotient.Quotient Q I)
    this✝ : Algebra R Q := ((algebraMap Rₘ Q).comp (algebraMap R Rₘ)).toAlgebra
    this : IsScalarTower R Rₘ Q := IsScalarTower.of_algebraMap_eq' rfl
    f : AlgHom Rₘ Sₘ Q :=
      let __src := Algebra.FormallySmooth.lift I ⋯ (AlgHom.restrictScalars R f✝);
      { toRingHom := __src.toRingHom, commutes' := ⋯ }
    x✝ : Sₘ
    ⊢ Eq (((Ideal.Quotient.mkₐ Rₘ I).comp f) x✝) (f✝ x✝)
  -/
  simp [f]
  /-
    🎉 no goals
  -/


theorem localization_map [FormallySmooth R S] : FormallySmooth Rₘ Sₘ := by
  /-
    R S Rₘ Sₘ : Type u
    inst✝¹³ : CommRing R
    inst✝¹² : CommRing S
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : CommRing Sₘ
    M : Submonoid R
    inst✝⁹ : Algebra R S
    inst✝⁸ : Algebra R Sₘ
    inst✝⁷ : Algebra S Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsScalarTower R S Sₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : IsLocalization (Submonoid.map (algebraMap R S) M) Sₘ
    inst✝ : Algebra.FormallySmooth R S
    ⊢ Algebra.FormallySmooth Rₘ Sₘ
  -/
  haveI : FormallySmooth S Sₘ := FormallySmooth.of_isLocalization (M.map (algebraMap R S))
  /-
    R S Rₘ Sₘ : Type u
    inst✝¹³ : CommRing R
    inst✝¹² : CommRing S
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : CommRing Sₘ
    M : Submonoid R
    inst✝⁹ : Algebra R S
    inst✝⁸ : Algebra R Sₘ
    inst✝⁷ : Algebra S Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsScalarTower R S Sₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : IsLocalization (Submonoid.map (algebraMap R S) M) Sₘ
    inst✝ : Algebra.FormallySmooth R S
    this : Algebra.FormallySmooth S Sₘ
    ⊢ Algebra.FormallySmooth Rₘ Sₘ
  -/
  haveI : FormallySmooth R Sₘ := FormallySmooth.comp R S Sₘ
  /-
    R S Rₘ Sₘ : Type u
    inst✝¹³ : CommRing R
    inst✝¹² : CommRing S
    inst✝¹¹ : CommRing Rₘ
    inst✝¹⁰ : CommRing Sₘ
    M : Submonoid R
    inst✝⁹ : Algebra R S
    inst✝⁸ : Algebra R Sₘ
    inst✝⁷ : Algebra S Sₘ
    inst✝⁶ : Algebra R Rₘ
    inst✝⁵ : Algebra Rₘ Sₘ
    inst✝⁴ : IsScalarTower R Rₘ Sₘ
    inst✝³ : IsScalarTower R S Sₘ
    inst✝² : IsLocalization M Rₘ
    inst✝¹ : IsLocalization (Submonoid.map (algebraMap R S) M) Sₘ
    inst✝ : Algebra.FormallySmooth R S
    this✝ : Algebra.FormallySmooth S Sₘ
    this : Algebra.FormallySmooth R Sₘ
    ⊢ Algebra.FormallySmooth Rₘ Sₘ
  -/
  exact FormallySmooth.localization_base M
  /-
    🎉 no goals
  -/


/-- An `R` algebra `A` is smooth if it is formally smooth and of finite presentation.

In the stacks project, the definition of smooth is completely different, and tag
<https://stacks.math.columbia.edu/tag/00TN> proves that their definition is equivalent
to this.
-/
class Smooth [CommSemiring R] (A : Type u) [Semiring A] [Algebra R A] : Prop where
  formallySmooth : FormallySmooth R A := by infer_instance
  finitePresentation : FinitePresentation R A := by infer_instance


/-- Being smooth is transported via algebra isomorphisms. -/
theorem of_equiv [Smooth R A] (e : A ≃ₐ[R] B) : Smooth R B where
  formallySmooth := FormallySmooth.of_equiv e
  finitePresentation := FinitePresentation.equiv e


/-- Localization at an element is smooth. -/
theorem of_isLocalization_Away (r : R) [IsLocalization.Away r A] : Smooth R A where
  formallySmooth := Algebra.FormallySmooth.of_isLocalization (Submonoid.powers r)
  finitePresentation := IsLocalization.Away.finitePresentation r


/-- Smooth is stable under composition. -/
theorem comp [Algebra A B] [IsScalarTower R A B] [Smooth R A] [Smooth A B] : Smooth R B where
  formallySmooth := FormallySmooth.comp R A B
  finitePresentation := FinitePresentation.trans R A B


/-- Smooth is stable under base change. -/
instance baseChange [Smooth R A] : Smooth B (B ⊗[R] A) where


