lemma le_comap_range_lTensor (q : Q) :
    LinearMap.range g ≤ (LinearMap.range (lTensor Q g)).comap (TensorProduct.mk R Q P q) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    q : Q
    ⊢ LE.le (LinearMap.range g) (Submodule.comap ((TensorProduct.mk R Q P) q) (Lin …
  -/
  rintro x ⟨n, rfl⟩
  /-
    case intro
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    q : Q
    n : N
    ⊢ Membership.mem (Submodule.comap ((TensorProduct.mk R Q P) q) (LinearMap.rang …
  -/
  exact ⟨q ⊗ₜ[R] n, rfl⟩
  /-
    🎉 no goals
  -/


lemma le_comap_range_rTensor (q : Q) :
    LinearMap.range g ≤ (LinearMap.range (rTensor Q g)).comap
      ((TensorProduct.mk R P Q).flip q) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    q : Q
    ⊢ LE.le (LinearMap.range g) (Submodule.comap ((TensorProduct.mk R P Q).flip q) …
  -/
  rintro x ⟨n, rfl⟩
  /-
    case intro
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    q : Q
    n : N
    ⊢ Membership.mem (Submodule.comap ((TensorProduct.mk R P Q).flip q) (LinearMap …
  -/
  exact ⟨n ⊗ₜ[R] q, rfl⟩
  /-
    🎉 no goals
  -/


/-- If `g` is surjective, then `lTensor Q g` is surjective -/
theorem LinearMap.lTensor_surjective (hg : Function.Surjective g) :
    Function.Surjective (lTensor Q g) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    hg : Function.Surjective ⇑g
    ⊢ Function.Surjective ⇑(LinearMap.lTensor Q g)
  -/
  intro z
  induction z with
  | zero => exact ⟨0, map_zero _⟩
  | tmul q p =>
    obtain ⟨n, rfl⟩ := hg p
    exact ⟨q ⊗ₜ[R] n, rfl⟩
  | add x y hx hy =>
    obtain ⟨x, rfl⟩ := hx
    obtain ⟨y, rfl⟩ := hy
    exact ⟨x + y, map_add _ _ _⟩


theorem LinearMap.lTensor_range :
    range (lTensor Q g) =
      range (lTensor Q (Submodule.subtype (range g))) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    ⊢ Eq (LinearMap.range (LinearMap.lTensor Q g)) (LinearMap.range (LinearMap.lTe …
  -/
  have : g = (Submodule.subtype _).comp g.rangeRestrict := rfl
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Eq (LinearMap.range (LinearMap.lTensor Q g)) (LinearMap.range (LinearMap.lTe …
  -/
  nth_rewrite 1 [this]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Eq (LinearMap.range (LinearMap.lTensor Q ((LinearMap.range g).subtype.comp g …
  -/
  rw [lTensor_comp]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Eq (LinearMap.range ((LinearMap.lTensor Q (LinearMap.range g).subtype).comp  …
  -/
  apply range_comp_of_range_eq_top
  /-
    case hf
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Eq (LinearMap.range (LinearMap.lTensor Q g.rangeRestrict)) Top.top
  -/
  rw [range_eq_top]
  /-
    case hf
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Function.Surjective ⇑(LinearMap.lTensor Q g.rangeRestrict)
  -/
  apply lTensor_surjective
  /-
    case hf.hg
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Function.Surjective ⇑g.rangeRestrict
  -/
  rw [← range_eq_top, range_rangeRestrict]
  /-
    🎉 no goals
  -/


/-- If `g` is surjective, then `rTensor Q g` is surjective -/
theorem LinearMap.rTensor_surjective (hg : Function.Surjective g) :
    Function.Surjective (rTensor Q g) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    hg : Function.Surjective ⇑g
    ⊢ Function.Surjective ⇑(LinearMap.rTensor Q g)
  -/
  intro z
  induction z with
  | zero => exact ⟨0, map_zero _⟩
  | tmul p q =>
    obtain ⟨n, rfl⟩ := hg p
    exact ⟨n ⊗ₜ[R] q, rfl⟩
  | add x y hx hy =>
    obtain ⟨x, rfl⟩ := hx
    obtain ⟨y, rfl⟩ := hy
    exact ⟨x + y, map_add _ _ _⟩


theorem LinearMap.rTensor_range :
    range (rTensor Q g) =
      range (rTensor Q (Submodule.subtype (range g))) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    ⊢ Eq (LinearMap.range (LinearMap.rTensor Q g)) (LinearMap.range (LinearMap.rTe …
  -/
  have : g = (Submodule.subtype _).comp g.rangeRestrict := rfl
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Eq (LinearMap.range (LinearMap.rTensor Q g)) (LinearMap.range (LinearMap.rTe …
  -/
  nth_rewrite 1 [this]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Eq (LinearMap.range (LinearMap.rTensor Q ((LinearMap.range g).subtype.comp g …
  -/
  rw [rTensor_comp]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Eq (LinearMap.range ((LinearMap.rTensor Q (LinearMap.range g).subtype).comp  …
  -/
  apply range_comp_of_range_eq_top
  /-
    case hf
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Eq (LinearMap.range (LinearMap.rTensor Q g.rangeRestrict)) Top.top
  -/
  rw [range_eq_top]
  /-
    case hf
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Function.Surjective ⇑(LinearMap.rTensor Q g.rangeRestrict)
  -/
  apply rTensor_surjective
  /-
    case hf.hg
    R : Type u_1
    inst✝⁶ : CommSemiring R
    N : Type u_3
    P : Type u_4
    Q : Type u_5
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : AddCommMonoid P
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R N
    inst✝¹ : Module R P
    inst✝ : Module R Q
    g : LinearMap (RingHom.id R) N P
    this : Eq g ((LinearMap.range g).subtype.comp g.rangeRestrict)
    ⊢ Function.Surjective ⇑g.rangeRestrict
  -/
  rw [← range_eq_top, range_rangeRestrict]
  /-
    🎉 no goals
  -/


lemma LinearMap.rTensor_exact_iff_lTensor_exact :
    Function.Exact (f.rTensor Q) (g.rTensor Q) ↔
    Function.Exact (f.lTensor Q) (g.lTensor Q) :=
  Function.Exact.iff_of_ladder_linearEquiv (e₁ := TensorProduct.comm _ _ _)
    (e₂ := TensorProduct.comm _ _ _) (e₃ := TensorProduct.comm _ _ _)
        /-
          R : Type u_1
          inst✝⁸ : CommSemiring R
          M : Type u_2
          N : Type u_3
          P : Type u_4
          Q : Type u_5
          inst✝⁷ : AddCommMonoid M
          inst✝⁶ : AddCommMonoid N
          inst✝⁵ : AddCommMonoid P
          inst✝⁴ : AddCommMonoid Q
          inst✝³ : Module R M
          inst✝² : Module R N
          inst✝¹ : Module R P
          inst✝ : Module R Q
          f : LinearMap (RingHom.id R) M N
          g : LinearMap (RingHom.id R) N P
          ⊢ Eq ((LinearMap.rTensor Q f).comp ↑(TensorProduct.comm R Q M)) ((↑(TensorProd …
        -/
             /-
               🎉 no goals
             -/
    (by ext; simp) (by ext; simp)
                            /-
                              🎉 no goals
                            -/


include hg hg' in
theorem TensorProduct.map_surjective : Function.Surjective (TensorProduct.map g g') := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    N : Type u_3
    P : Type u_4
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : AddCommMonoid P
    inst✝⁵ : Module R N
    inst✝⁴ : Module R P
    g : LinearMap (RingHom.id R) N P
    hg : Function.Surjective ⇑g
    N' : Type u_6
    P' : Type u_7
    inst✝³ : AddCommMonoid N'
    inst✝² : AddCommMonoid P'
    inst✝¹ : Module R N'
    inst✝ : Module R P'
    g' : LinearMap (RingHom.id R) N' P'
    hg' : Function.Surjective ⇑g'
    ⊢ Function.Surjective ⇑(TensorProduct.map g g')
  -/
  rw [← lTensor_comp_rTensor, coe_comp]
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    N : Type u_3
    P : Type u_4
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : AddCommMonoid P
    inst✝⁵ : Module R N
    inst✝⁴ : Module R P
    g : LinearMap (RingHom.id R) N P
    hg : Function.Surjective ⇑g
    N' : Type u_6
    P' : Type u_7
    inst✝³ : AddCommMonoid N'
    inst✝² : AddCommMonoid P'
    inst✝¹ : Module R N'
    inst✝ : Module R P'
    g' : LinearMap (RingHom.id R) N' P'
    hg' : Function.Surjective ⇑g'
    ⊢ Function.Surjective (Function.comp ⇑(LinearMap.lTensor P g') ⇑(LinearMap.rTe …
  -/
  exact Function.Surjective.comp (lTensor_surjective _ hg') (rTensor_surjective _ hg)
  /-
    🎉 no goals
  -/


variable (M R) in
theorem TensorProduct.mk_surjective (S) [Semiring S] [Algebra R S]
    (h : Function.Surjective (algebraMap R S)) :
    Function.Surjective (TensorProduct.mk R S M 1) := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_8
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    h : Function.Surjective ⇑(algebraMap R S)
    ⊢ Function.Surjective ⇑((TensorProduct.mk R S M) 1)
  -/
  rw [← LinearMap.range_eq_top, ← top_le_iff, ← span_tmul_eq_top, Submodule.span_le]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_8
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    h : Function.Surjective ⇑(algebraMap R S)
    ⊢ HasSubset.Subset (setOf fun t => Exists fun m => Exists fun n => Eq (TensorP …
  -/
  rintro _ ⟨x, y, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_8
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    h : Function.Surjective ⇑(algebraMap R S)
    x : S
    y : M
    ⊢ Membership.mem (↑(LinearMap.range ((TensorProduct.mk R S M) 1))) (TensorProd …
  -/
  obtain ⟨x, rfl⟩ := h x
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_8
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    h : Function.Surjective ⇑(algebraMap R S)
    y : M
    x : R
    ⊢ Membership.mem (↑(LinearMap.range ((TensorProduct.mk R S M) 1))) (TensorProd …
  -/
  rw [Algebra.algebraMap_eq_smul_one, smul_tmul]
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_2
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    S : Type u_8
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    h : Function.Surjective ⇑(algebraMap R S)
    y : M
    x : R
    ⊢ Membership.mem (↑(LinearMap.range ((TensorProduct.mk R S M) 1))) (TensorProd …
  -/
  exact ⟨x • y, rfl⟩
  /-
    🎉 no goals
  -/


/-- The direct map in `lTensor.equiv` -/
noncomputable def lTensor.toFun (hfg : Exact f g) :
    Q ⊗[R] N ⧸ LinearMap.range (lTensor Q f) →ₗ[R] Q ⊗[R] P :=
  Submodule.liftQ _ (lTensor Q g) <| by
    rw [LinearMap.range_le_iff_comap, ← LinearMap.ker_comp,
      ← lTensor_comp, hfg.linearMap_comp_eq_zero, lTensor_zero, ker_zero]


/-- The inverse map in `lTensor.equiv_of_rightInverse` (computably, given a right inverse)-/
noncomputable def lTensor.inverse_of_rightInverse {h : P → N} (hfg : Exact f g)
    (hgh : Function.RightInverse h g) :
    Q ⊗[R] P →ₗ[R] Q ⊗[R] N ⧸ LinearMap.range (lTensor Q f) :=
  TensorProduct.lift <| LinearMap.flip <| {
    toFun := fun p ↦ Submodule.mkQ _ ∘ₗ ((TensorProduct.mk R _ _).flip (h p))
    map_add' := fun p p' => LinearMap.ext fun q => (Submodule.Quotient.eq _).mpr <| by
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        p p' : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.lTensor Q f)) (HSub.hSub (((Tenso …
      -/
      change q ⊗ₜ[R] (h (p + p')) - (q ⊗ₜ[R] (h p) + q ⊗ₜ[R] (h p')) ∈ range (lTensor Q f)
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        p p' : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.lTensor Q f)) (HSub.hSub (TensorP …
      -/
      rw [← TensorProduct.tmul_add, ← TensorProduct.tmul_sub]
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        p p' : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.lTensor Q f)) (TensorProduct.tmul …
      -/
      apply le_comap_range_lTensor f
      /-
        case a
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        p p' : P
        q : Q
        ⊢ Membership.mem (LinearMap.range f) (HSub.hSub (h (HAdd.hAdd p p')) (HAdd.hAd …
      -/
      rw [exact_iff] at hfg
      /-
        case a
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Eq (LinearMap.ker g) (LinearMap.range f)
        hgh : Function.RightInverse h ⇑g
        p p' : P
        q : Q
        ⊢ Membership.mem (LinearMap.range f) (HSub.hSub (h (HAdd.hAdd p p')) (HAdd.hAd …
      -/
      simp only [← hfg, mem_ker, map_sub, map_add, hgh _, sub_self]
      /-
        🎉 no goals
      -/
    map_smul' := fun r p => LinearMap.ext fun q => (Submodule.Quotient.eq _).mpr <| by
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        r : R
        p : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.lTensor Q f)) (HSub.hSub (((Tenso …
      -/
      change q ⊗ₜ[R] (h (r • p)) - r • q ⊗ₜ[R] (h p) ∈ range (lTensor Q f)
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        r : R
        p : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.lTensor Q f)) (HSub.hSub (TensorP …
      -/
      rw [← TensorProduct.tmul_smul, ← TensorProduct.tmul_sub]
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        r : R
        p : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.lTensor Q f)) (TensorProduct.tmul …
      -/
      apply le_comap_range_lTensor f
      /-
        case a
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        r : R
        p : P
        q : Q
        ⊢ Membership.mem (LinearMap.range f) (HSub.hSub (h (HSMul.hSMul r p)) (HSMul.h …
      -/
      rw [exact_iff] at hfg
      /-
        case a
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Eq (LinearMap.ker g) (LinearMap.range f)
        hgh : Function.RightInverse h ⇑g
        r : R
        p : P
        q : Q
        ⊢ Membership.mem (LinearMap.range f) (HSub.hSub (h (HSMul.hSMul r p)) (HSMul.h …
      -/
      simp only [← hfg, mem_ker, map_sub, map_smul, hgh _, sub_self] }
      /-
        🎉 no goals
      -/


lemma lTensor.inverse_of_rightInverse_apply
    {h : P → N} (hgh : Function.RightInverse h g) (y : Q ⊗[R] N) :
    (lTensor.inverse_of_rightInverse Q hfg hgh) ((lTensor Q g) y) =
      Submodule.Quotient.mk (p := (LinearMap.range (lTensor Q f))) y := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R Q N
    ⊢ Eq ((lTensor.inverse_of_rightInverse Q hfg hgh) ((LinearMap.lTensor Q g) y)) …
  -/
  simp only [← LinearMap.comp_apply, ← Submodule.mkQ_apply]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R Q N
    ⊢ Eq (((lTensor.inverse_of_rightInverse Q hfg hgh).comp (LinearMap.lTensor Q g …
  -/
  rw [exact_iff] at hfg
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R Q N
    ⊢ Eq (((lTensor.inverse_of_rightInverse Q hfg✝ hgh).comp (LinearMap.lTensor Q  …
  -/
  apply LinearMap.congr_fun
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R Q N
    ⊢ Eq ((lTensor.inverse_of_rightInverse Q hfg✝ hgh).comp (LinearMap.lTensor Q g …
  -/
  apply TensorProduct.ext'
  /-
    case h.H
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R Q N
    ⊢ ∀ (x : Q) (y : N), Eq (((lTensor.inverse_of_rightInverse Q hfg✝ hgh).comp (L …
  -/
  intro n q
  simp? [lTensor.inverse_of_rightInverse] says
    simp only [inverse_of_rightInverse, coe_comp, Function.comp_apply, lTensor_tmul,
      lift.tmul, flip_apply, coe_mk, AddHom.coe_mk, mk_apply, Submodule.mkQ_apply]
  /-
    case h.H
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R Q N
    n : Q
    q : N
    ⊢ Eq (Submodule.Quotient.mk (TensorProduct.tmul R n (h (g q)))) (Submodule.Quo …
  -/
  rw [Submodule.Quotient.eq, ← TensorProduct.tmul_sub]
  /-
    case h.H
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R Q N
    n : Q
    q : N
    ⊢ Membership.mem (LinearMap.range (LinearMap.lTensor Q f)) (TensorProduct.tmul …
  -/
  apply le_comap_range_lTensor f n
  /-
    case h.H.a
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R Q N
    n : Q
    q : N
    ⊢ Membership.mem (LinearMap.range f) (HSub.hSub (h (g q)) q)
  -/
  rw [← hfg, mem_ker, map_sub, sub_eq_zero, hgh]
  /-
    🎉 no goals
  -/


lemma lTensor.inverse_of_rightInverse_comp_lTensor
    {h : P → N} (hgh : Function.RightInverse h g) :
    (lTensor.inverse_of_rightInverse Q hfg hgh).comp (lTensor Q g) =
      Submodule.mkQ (p := LinearMap.range (lTensor Q f)) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    h : P → N
    hgh : Function.RightInverse h ⇑g
    ⊢ Eq ((lTensor.inverse_of_rightInverse Q hfg hgh).comp (LinearMap.lTensor Q g) …
  -/
  rw [LinearMap.ext_iff]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    h : P → N
    hgh : Function.RightInverse h ⇑g
    ⊢ ∀ (x : TensorProduct R Q N), Eq (((lTensor.inverse_of_rightInverse Q hfg hgh …
  -/
  intro y
  simp only [coe_comp, Function.comp_apply, Submodule.mkQ_apply,
    lTensor.inverse_of_rightInverse_apply]


/-- The inverse map in `lTensor.equiv` -/
noncomputable
def lTensor.inverse :
    Q ⊗[R] P →ₗ[R] Q ⊗[R] N ⧸ LinearMap.range (lTensor Q f) :=
  lTensor.inverse_of_rightInverse Q hfg (Function.rightInverse_surjInv hg)


lemma lTensor.inverse_apply (y : Q ⊗[R] N) :
    (lTensor.inverse Q hfg hg) ((lTensor Q g) y) =
      Submodule.Quotient.mk (p := (LinearMap.range (lTensor Q f))) y := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    y : TensorProduct R Q N
    ⊢ Eq ((lTensor.inverse Q hfg hg) ((LinearMap.lTensor Q g) y)) (Submodule.Quoti …
  -/
  rw [lTensor.inverse, lTensor.inverse_of_rightInverse_apply]
  /-
    🎉 no goals
  -/


lemma lTensor.inverse_comp_lTensor :
    (lTensor.inverse Q hfg hg).comp (lTensor Q g) =
      Submodule.mkQ (p := LinearMap.range (lTensor Q f)) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    ⊢ Eq ((lTensor.inverse Q hfg hg).comp (LinearMap.lTensor Q g)) (LinearMap.rang …
  -/
  rw [lTensor.inverse, lTensor.inverse_of_rightInverse_comp_lTensor]
  /-
    🎉 no goals
  -/


/-- For a surjective `f : N →ₗ[R] P`,
  the natural equivalence between `Q ⊗ N ⧸ (image of ker f)` to `Q ⊗ P`
  (computably, given a right inverse) -/
noncomputable
def lTensor.linearEquiv_of_rightInverse {h : P → N} (hgh : Function.RightInverse h g) :
    ((Q ⊗[R] N) ⧸ (LinearMap.range (lTensor Q f))) ≃ₗ[R] (Q ⊗[R] P) := {
  toLinearMap := lTensor.toFun Q hfg
  invFun    := lTensor.inverse_of_rightInverse Q hfg hgh
  left_inv  := fun y ↦ by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      y : HasQuotient.Quotient (TensorProduct R Q N) (LinearMap.range (LinearMap.lTe …
      ⊢ Eq ((lTensor.inverse_of_rightInverse Q hfg hgh) ((lTensor.toFun Q hfg).toFun …
    -/
    simp only [lTensor.toFun, AddHom.toFun_eq_coe, coe_toAddHom]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      y : HasQuotient.Quotient (TensorProduct R Q N) (LinearMap.range (LinearMap.lTe …
      ⊢ Eq ((lTensor.inverse_of_rightInverse Q hfg hgh) (((LinearMap.range (LinearMa …
    -/
    obtain ⟨y, rfl⟩ := Submodule.mkQ_surjective _ y
    /-
      case intro
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      y : TensorProduct R Q N
      ⊢ Eq ((lTensor.inverse_of_rightInverse Q hfg hgh) (((LinearMap.range (LinearMa …
    -/
    simp only [Submodule.mkQ_apply, Submodule.liftQ_apply, lTensor.inverse_of_rightInverse_apply]
    /-
      🎉 no goals
    -/
  right_inv := fun z ↦ by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      z : TensorProduct R Q P
      ⊢ Eq ((lTensor.toFun Q hfg).toFun ((lTensor.inverse_of_rightInverse Q hfg hgh) …
    -/
    simp only [AddHom.toFun_eq_coe, coe_toAddHom]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      z : TensorProduct R Q P
      ⊢ Eq ((lTensor.toFun Q hfg) ((lTensor.inverse_of_rightInverse Q hfg hgh) z)) z
    -/
    obtain ⟨y, rfl⟩ := lTensor_surjective Q (hgh.surjective) z
    /-
      case intro
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      y : TensorProduct R Q N
      ⊢ Eq ((lTensor.toFun Q hfg) ((lTensor.inverse_of_rightInverse Q hfg hgh) ((Lin …
    -/
    rw [lTensor.inverse_of_rightInverse_apply]
    /-
      case intro
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      y : TensorProduct R Q N
      ⊢ Eq ((lTensor.toFun Q hfg) (Submodule.Quotient.mk y)) ((LinearMap.lTensor Q g …
    -/
    simp only [lTensor.toFun, Submodule.liftQ_apply] }
    /-
      🎉 no goals
    -/


/-- For a surjective `f : N →ₗ[R] P`,
  the natural equivalence between `Q ⊗ N ⧸ (image of ker f)` to `Q ⊗ P` -/
noncomputable def lTensor.equiv :
    ((Q ⊗[R] N) ⧸ (LinearMap.range (lTensor Q f))) ≃ₗ[R] (Q ⊗[R] P) :=
  lTensor.linearEquiv_of_rightInverse Q hfg (Function.rightInverse_surjInv hg)


include hfg hg in
/-- Tensoring an exact pair on the left gives an exact pair -/
theorem lTensor_exact : Exact (lTensor Q f) (lTensor Q g) := by
  rw [exact_iff, ← Submodule.ker_mkQ (p := range (lTensor Q f)),
    ← lTensor.inverse_comp_lTensor Q hfg hg]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    ⊢ Eq (LinearMap.ker (LinearMap.lTensor Q g)) (LinearMap.ker ((lTensor.inverse  …
  -/
  apply symm
  /-
    case a
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    ⊢ Eq (LinearMap.ker ((lTensor.inverse Q hfg hg).comp (LinearMap.lTensor Q g))) …
  -/
  apply LinearMap.ker_comp_of_ker_eq_bot
  /-
    case a.hg
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    ⊢ Eq (LinearMap.ker (lTensor.inverse Q hfg hg)) Bot.bot
  -/
  rw [LinearMap.ker_eq_bot]
  /-
    case a.hg
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    ⊢ Function.Injective ⇑(lTensor.inverse Q hfg hg)
  -/
  exact (lTensor.equiv Q hfg hg).symm.injective
  /-
    🎉 no goals
  -/


/-- Right-exactness of tensor product -/
lemma lTensor_mkQ (N : Submodule R M) :
    ker (lTensor Q (N.mkQ)) = range (lTensor Q N.subtype) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    N : Submodule R M
    ⊢ Eq (LinearMap.ker (LinearMap.lTensor Q N.mkQ)) (LinearMap.range (LinearMap.l …
  -/
  rw [← exact_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    N : Submodule R M
    ⊢ Function.Exact ⇑(LinearMap.lTensor Q N.subtype) ⇑(LinearMap.lTensor Q N.mkQ)
  -/
  exact lTensor_exact Q (LinearMap.exact_subtype_mkQ N) (Submodule.mkQ_surjective N)
  /-
    🎉 no goals
  -/


/-- The direct map in `rTensor.equiv` -/
noncomputable def rTensor.toFun (hfg : Exact f g) :
    N ⊗[R] Q ⧸ range (rTensor Q f) →ₗ[R] P ⊗[R] Q :=
  Submodule.liftQ _ (rTensor Q g) <| by
    rw [range_le_iff_comap, ← ker_comp, ← rTensor_comp,
      hfg.linearMap_comp_eq_zero, rTensor_zero, ker_zero]


/-- The inverse map in `rTensor.equiv_of_rightInverse` (computably, given a right inverse) -/
noncomputable def rTensor.inverse_of_rightInverse {h : P → N} (hfg : Exact f g)
    (hgh : Function.RightInverse h g) :
    P ⊗[R] Q →ₗ[R] N ⊗[R] Q ⧸ LinearMap.range (rTensor Q f) :=
  TensorProduct.lift  {
    toFun := fun p ↦ Submodule.mkQ _ ∘ₗ TensorProduct.mk R _ _ (h p)
    map_add' := fun p p' => LinearMap.ext fun q => (Submodule.Quotient.eq _).mpr <| by
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        p p' : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.rTensor Q f)) (HSub.hSub (((Tenso …
      -/
      change h (p + p') ⊗ₜ[R] q - (h p ⊗ₜ[R] q + h p' ⊗ₜ[R] q) ∈ range (rTensor Q f)
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        p p' : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.rTensor Q f)) (HSub.hSub (TensorP …
      -/
      rw [← TensorProduct.add_tmul, ← TensorProduct.sub_tmul]
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        p p' : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.rTensor Q f)) (TensorProduct.tmul …
      -/
      apply le_comap_range_rTensor f
      /-
        case a
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        p p' : P
        q : Q
        ⊢ Membership.mem (LinearMap.range f) (HSub.hSub (h (HAdd.hAdd p p')) (HAdd.hAd …
      -/
      rw [exact_iff] at hfg
      /-
        case a
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Eq (LinearMap.ker g) (LinearMap.range f)
        hgh : Function.RightInverse h ⇑g
        p p' : P
        q : Q
        ⊢ Membership.mem (LinearMap.range f) (HSub.hSub (h (HAdd.hAdd p p')) (HAdd.hAd …
      -/
      simp only [← hfg, mem_ker, map_sub, map_add, hgh _, sub_self]
      /-
        🎉 no goals
      -/
    map_smul' := fun r p => LinearMap.ext fun q => (Submodule.Quotient.eq _).mpr <| by
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        r : R
        p : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.rTensor Q f)) (HSub.hSub (((Tenso …
      -/
      change h (r • p) ⊗ₜ[R] q - r • h p ⊗ₜ[R] q ∈ range (rTensor Q f)
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        r : R
        p : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.rTensor Q f)) (HSub.hSub (TensorP …
      -/
      rw [TensorProduct.smul_tmul', ← TensorProduct.sub_tmul]
      /-
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        r : R
        p : P
        q : Q
        ⊢ Membership.mem (LinearMap.range (LinearMap.rTensor Q f)) (TensorProduct.tmul …
      -/
      apply le_comap_range_rTensor f
      /-
        case a
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Function.Exact ⇑f ⇑g
        hgh : Function.RightInverse h ⇑g
        r : R
        p : P
        q : Q
        ⊢ Membership.mem (LinearMap.range f) (HSub.hSub (h (HSMul.hSMul r p)) (HSMul.h …
      -/
      rw [exact_iff] at hfg
      /-
        case a
        R : Type u_1
        M : Type u_2
        N : Type u_3
        P : Type u_4
        inst✝⁸ : CommRing R
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : AddCommGroup N
        inst✝⁵ : AddCommGroup P
        inst✝⁴ : Module R M
        inst✝³ : Module R N
        inst✝² : Module R P
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        Q : Type u_5
        inst✝¹ : AddCommGroup Q
        inst✝ : Module R Q
        hfg✝ : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        h : P → N
        hfg : Eq (LinearMap.ker g) (LinearMap.range f)
        hgh : Function.RightInverse h ⇑g
        r : R
        p : P
        q : Q
        ⊢ Membership.mem (LinearMap.range f) (HSub.hSub (h (HSMul.hSMul r p)) (HSMul.h …
      -/
      simp only [← hfg, mem_ker, map_sub, map_smul, hgh _, sub_self] }
      /-
        🎉 no goals
      -/


lemma rTensor.inverse_of_rightInverse_apply
    {h : P → N} (hgh : Function.RightInverse h g) (y : N ⊗[R] Q) :
    (rTensor.inverse_of_rightInverse Q hfg hgh) ((rTensor Q g) y) =
      Submodule.Quotient.mk (p := LinearMap.range (rTensor Q f)) y := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R N Q
    ⊢ Eq ((rTensor.inverse_of_rightInverse Q hfg hgh) ((LinearMap.rTensor Q g) y)) …
  -/
  simp only [← LinearMap.comp_apply, ← Submodule.mkQ_apply]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R N Q
    ⊢ Eq (((rTensor.inverse_of_rightInverse Q hfg hgh).comp (LinearMap.rTensor Q g …
  -/
  rw [exact_iff] at hfg
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R N Q
    ⊢ Eq (((rTensor.inverse_of_rightInverse Q hfg✝ hgh).comp (LinearMap.rTensor Q  …
  -/
  apply LinearMap.congr_fun
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R N Q
    ⊢ Eq ((rTensor.inverse_of_rightInverse Q hfg✝ hgh).comp (LinearMap.rTensor Q g …
  -/
  apply TensorProduct.ext'
  /-
    case h.H
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R N Q
    ⊢ ∀ (x : N) (y : Q), Eq (((rTensor.inverse_of_rightInverse Q hfg✝ hgh).comp (L …
  -/
  intro n q
  simp? [rTensor.inverse_of_rightInverse] says
    simp only [inverse_of_rightInverse, coe_comp, Function.comp_apply, rTensor_tmul,
      lift.tmul, coe_mk, AddHom.coe_mk, mk_apply, Submodule.mkQ_apply]
  /-
    case h.H
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R N Q
    n : N
    q : Q
    ⊢ Eq (Submodule.Quotient.mk (TensorProduct.tmul R (h (g n)) q)) (Submodule.Quo …
  -/
  rw [Submodule.Quotient.eq, ← TensorProduct.sub_tmul]
  /-
    case h.H
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R N Q
    n : N
    q : Q
    ⊢ Membership.mem (LinearMap.range (LinearMap.rTensor Q f)) (TensorProduct.tmul …
  -/
  apply le_comap_range_rTensor f
  /-
    case h.H.a
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg✝ : Function.Exact ⇑f ⇑g
    hfg : Eq (LinearMap.ker g) (LinearMap.range f)
    h : P → N
    hgh : Function.RightInverse h ⇑g
    y : TensorProduct R N Q
    n : N
    q : Q
    ⊢ Membership.mem (LinearMap.range f) (HSub.hSub (h (g n)) n)
  -/
  rw [← hfg, mem_ker, map_sub, sub_eq_zero, hgh]
  /-
    🎉 no goals
  -/


lemma rTensor.inverse_of_rightInverse_comp_rTensor
    {h : P → N} (hgh : Function.RightInverse h g) :
    (rTensor.inverse_of_rightInverse Q hfg hgh).comp (rTensor Q g) =
      Submodule.mkQ (p := LinearMap.range (rTensor Q f)) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    h : P → N
    hgh : Function.RightInverse h ⇑g
    ⊢ Eq ((rTensor.inverse_of_rightInverse Q hfg hgh).comp (LinearMap.rTensor Q g) …
  -/
  rw [LinearMap.ext_iff]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    h : P → N
    hgh : Function.RightInverse h ⇑g
    ⊢ ∀ (x : TensorProduct R N Q), Eq (((rTensor.inverse_of_rightInverse Q hfg hgh …
  -/
  intro y
  simp only [coe_comp, Function.comp_apply, Submodule.mkQ_apply,
    rTensor.inverse_of_rightInverse_apply]


/-- The inverse map in `rTensor.equiv` -/
noncomputable
def rTensor.inverse :
    P ⊗[R] Q →ₗ[R] N ⊗[R] Q ⧸ LinearMap.range (rTensor Q f) :=
  rTensor.inverse_of_rightInverse Q hfg (Function.rightInverse_surjInv hg)


lemma rTensor.inverse_apply (y : N ⊗[R] Q) :
    (rTensor.inverse Q hfg hg) ((rTensor Q g) y) =
      Submodule.Quotient.mk (p := LinearMap.range (rTensor Q f)) y := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    y : TensorProduct R N Q
    ⊢ Eq ((rTensor.inverse Q hfg hg) ((LinearMap.rTensor Q g) y)) (Submodule.Quoti …
  -/
  rw [rTensor.inverse, rTensor.inverse_of_rightInverse_apply]
  /-
    🎉 no goals
  -/


lemma rTensor.inverse_comp_rTensor :
    (rTensor.inverse Q hfg hg).comp (rTensor Q g) =
      Submodule.mkQ (p := LinearMap.range (rTensor Q f)) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    ⊢ Eq ((rTensor.inverse Q hfg hg).comp (LinearMap.rTensor Q g)) (LinearMap.rang …
  -/
  rw [rTensor.inverse, rTensor.inverse_of_rightInverse_comp_rTensor]
  /-
    🎉 no goals
  -/


/-- For a surjective `f : N →ₗ[R] P`,
  the natural equivalence between `N ⊗[R] Q ⧸ (range (rTensor Q f))` and `P ⊗[R] Q`
  (computably, given a right inverse) -/
noncomputable
def rTensor.linearEquiv_of_rightInverse {h : P → N} (hgh : Function.RightInverse h g) :
    ((N ⊗[R] Q) ⧸ (range (rTensor Q f))) ≃ₗ[R] (P ⊗[R] Q) := {
  toLinearMap := rTensor.toFun Q hfg
  invFun      := rTensor.inverse_of_rightInverse Q hfg hgh
  left_inv    := fun y ↦ by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      y : HasQuotient.Quotient (TensorProduct R N Q) (LinearMap.range (LinearMap.rTe …
      ⊢ Eq ((rTensor.inverse_of_rightInverse Q hfg hgh) ((rTensor.toFun Q hfg).toFun …
    -/
    simp only [rTensor.toFun, AddHom.toFun_eq_coe, coe_toAddHom]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      y : HasQuotient.Quotient (TensorProduct R N Q) (LinearMap.range (LinearMap.rTe …
      ⊢ Eq ((rTensor.inverse_of_rightInverse Q hfg hgh) (((LinearMap.range (LinearMa …
    -/
    obtain ⟨y, rfl⟩ := Submodule.mkQ_surjective _ y
    /-
      case intro
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      y : TensorProduct R N Q
      ⊢ Eq ((rTensor.inverse_of_rightInverse Q hfg hgh) (((LinearMap.range (LinearMa …
    -/
    simp only [Submodule.mkQ_apply, Submodule.liftQ_apply, rTensor.inverse_of_rightInverse_apply]
    /-
      🎉 no goals
    -/
  right_inv   := fun z ↦ by
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      z : TensorProduct R P Q
      ⊢ Eq ((rTensor.toFun Q hfg).toFun ((rTensor.inverse_of_rightInverse Q hfg hgh) …
    -/
    simp only [AddHom.toFun_eq_coe, coe_toAddHom]
    /-
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      z : TensorProduct R P Q
      ⊢ Eq ((rTensor.toFun Q hfg) ((rTensor.inverse_of_rightInverse Q hfg hgh) z)) z
    -/
    obtain ⟨y, rfl⟩ := rTensor_surjective Q hgh.surjective z
    /-
      case intro
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      y : TensorProduct R N Q
      ⊢ Eq ((rTensor.toFun Q hfg) ((rTensor.inverse_of_rightInverse Q hfg hgh) ((Lin …
    -/
    rw [rTensor.inverse_of_rightInverse_apply]
    /-
      case intro
      R : Type u_1
      M : Type u_2
      N : Type u_3
      P : Type u_4
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : AddCommGroup P
      inst✝⁴ : Module R M
      inst✝³ : Module R N
      inst✝² : Module R P
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      Q : Type u_5
      inst✝¹ : AddCommGroup Q
      inst✝ : Module R Q
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      h : P → N
      hgh : Function.RightInverse h ⇑g
      y : TensorProduct R N Q
      ⊢ Eq ((rTensor.toFun Q hfg) (Submodule.Quotient.mk y)) ((LinearMap.rTensor Q g …
    -/
    simp only [rTensor.toFun, Submodule.liftQ_apply] }
    /-
      🎉 no goals
    -/


/-- For a surjective `f : N →ₗ[R] P`,
  the natural equivalence between `N ⊗[R] Q ⧸ (range (rTensor Q f))` and `P ⊗[R] Q` -/
noncomputable def rTensor.equiv :
    ((N ⊗[R] Q) ⧸ (LinearMap.range (rTensor Q f))) ≃ₗ[R] (P ⊗[R] Q) :=
  rTensor.linearEquiv_of_rightInverse Q hfg (Function.rightInverse_surjInv hg)


include hfg hg in
/-- Tensoring an exact pair on the right gives an exact pair -/
theorem rTensor_exact : Exact (rTensor Q f) (rTensor Q g) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    ⊢ Function.Exact ⇑(LinearMap.rTensor Q f) ⇑(LinearMap.rTensor Q g)
  -/
  rw [rTensor_exact_iff_lTensor_exact]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R M
    inst✝³ : Module R N
    inst✝² : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    ⊢ Function.Exact ⇑(LinearMap.lTensor Q f) ⇑(LinearMap.lTensor Q g)
  -/
  exact lTensor_exact Q hfg hg
  /-
    🎉 no goals
  -/


/-- Right-exactness of tensor product (`rTensor`) -/
lemma rTensor_mkQ (N : Submodule R M) :
    ker (rTensor Q (N.mkQ)) = range (rTensor Q N.subtype) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    N : Submodule R M
    ⊢ Eq (LinearMap.ker (LinearMap.rTensor Q N.mkQ)) (LinearMap.range (LinearMap.r …
  -/
  rw [← exact_iff]
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    Q : Type u_5
    inst✝¹ : AddCommGroup Q
    inst✝ : Module R Q
    N : Submodule R M
    ⊢ Function.Exact ⇑(LinearMap.rTensor Q N.subtype) ⇑(LinearMap.rTensor Q N.mkQ)
  -/
  exact rTensor_exact Q (LinearMap.exact_subtype_mkQ N) (Submodule.mkQ_surjective N)
  /-
    🎉 no goals
  -/


include hg hg' hfg hfg' in
/-- Kernel of a product map (right-exactness of tensor product) -/
theorem TensorProduct.map_ker :
    ker (TensorProduct.map g g') = range (lTensor N f') ⊔ range (rTensor N' f) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : AddCommGroup N
    inst✝⁹ : AddCommGroup P
    inst✝⁸ : Module R M
    inst✝⁷ : Module R N
    inst✝⁶ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    M' : Type u_6
    N' : Type u_7
    P' : Type u_8
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : AddCommGroup N'
    inst✝³ : AddCommGroup P'
    inst✝² : Module R M'
    inst✝¹ : Module R N'
    inst✝ : Module R P'
    f' : LinearMap (RingHom.id R) M' N'
    g' : LinearMap (RingHom.id R) N' P'
    hfg' : Function.Exact ⇑f' ⇑g'
    hg' : Function.Surjective ⇑g'
    ⊢ Eq (LinearMap.ker (TensorProduct.map g g')) (Max.max (LinearMap.range (Linea …
  -/
  rw [← lTensor_comp_rTensor]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : AddCommGroup N
    inst✝⁹ : AddCommGroup P
    inst✝⁸ : Module R M
    inst✝⁷ : Module R N
    inst✝⁶ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    M' : Type u_6
    N' : Type u_7
    P' : Type u_8
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : AddCommGroup N'
    inst✝³ : AddCommGroup P'
    inst✝² : Module R M'
    inst✝¹ : Module R N'
    inst✝ : Module R P'
    f' : LinearMap (RingHom.id R) M' N'
    g' : LinearMap (RingHom.id R) N' P'
    hfg' : Function.Exact ⇑f' ⇑g'
    hg' : Function.Surjective ⇑g'
    ⊢ Eq (LinearMap.ker ((LinearMap.lTensor P g').comp (LinearMap.rTensor N' g)))  …
  -/
  rw [ker_comp]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : AddCommGroup N
    inst✝⁹ : AddCommGroup P
    inst✝⁸ : Module R M
    inst✝⁷ : Module R N
    inst✝⁶ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    M' : Type u_6
    N' : Type u_7
    P' : Type u_8
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : AddCommGroup N'
    inst✝³ : AddCommGroup P'
    inst✝² : Module R M'
    inst✝¹ : Module R N'
    inst✝ : Module R P'
    f' : LinearMap (RingHom.id R) M' N'
    g' : LinearMap (RingHom.id R) N' P'
    hfg' : Function.Exact ⇑f' ⇑g'
    hg' : Function.Surjective ⇑g'
    ⊢ Eq (Submodule.comap (LinearMap.rTensor N' g) (LinearMap.ker (LinearMap.lTens …
  -/
  rw [← Exact.linearMap_ker_eq (rTensor_exact N' hfg hg)]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : AddCommGroup N
    inst✝⁹ : AddCommGroup P
    inst✝⁸ : Module R M
    inst✝⁷ : Module R N
    inst✝⁶ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    M' : Type u_6
    N' : Type u_7
    P' : Type u_8
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : AddCommGroup N'
    inst✝³ : AddCommGroup P'
    inst✝² : Module R M'
    inst✝¹ : Module R N'
    inst✝ : Module R P'
    f' : LinearMap (RingHom.id R) M' N'
    g' : LinearMap (RingHom.id R) N' P'
    hfg' : Function.Exact ⇑f' ⇑g'
    hg' : Function.Surjective ⇑g'
    ⊢ Eq (Submodule.comap (LinearMap.rTensor N' g) (LinearMap.ker (LinearMap.lTens …
  -/
  rw [← Submodule.comap_map_eq]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : AddCommGroup N
    inst✝⁹ : AddCommGroup P
    inst✝⁸ : Module R M
    inst✝⁷ : Module R N
    inst✝⁶ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    M' : Type u_6
    N' : Type u_7
    P' : Type u_8
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : AddCommGroup N'
    inst✝³ : AddCommGroup P'
    inst✝² : Module R M'
    inst✝¹ : Module R N'
    inst✝ : Module R P'
    f' : LinearMap (RingHom.id R) M' N'
    g' : LinearMap (RingHom.id R) N' P'
    hfg' : Function.Exact ⇑f' ⇑g'
    hg' : Function.Surjective ⇑g'
    ⊢ Eq (Submodule.comap (LinearMap.rTensor N' g) (LinearMap.ker (LinearMap.lTens …
  -/
  apply congr_arg₂ _ rfl
  rw [range_eq_map, ← Submodule.map_comp, rTensor_comp_lTensor,
    Submodule.map_top]
  rw [← lTensor_comp_rTensor, range_eq_map, Submodule.map_comp,
    Submodule.map_top]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : AddCommGroup N
    inst✝⁹ : AddCommGroup P
    inst✝⁸ : Module R M
    inst✝⁷ : Module R N
    inst✝⁶ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    M' : Type u_6
    N' : Type u_7
    P' : Type u_8
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : AddCommGroup N'
    inst✝³ : AddCommGroup P'
    inst✝² : Module R M'
    inst✝¹ : Module R N'
    inst✝ : Module R P'
    f' : LinearMap (RingHom.id R) M' N'
    g' : LinearMap (RingHom.id R) N' P'
    hfg' : Function.Exact ⇑f' ⇑g'
    hg' : Function.Surjective ⇑g'
    ⊢ Eq (LinearMap.ker (LinearMap.lTensor P g')) (Submodule.map (LinearMap.lTenso …
  -/
  rw [range_eq_top.mpr (rTensor_surjective M' hg), Submodule.map_top]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    P : Type u_4
    inst✝¹² : CommRing R
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : AddCommGroup N
    inst✝⁹ : AddCommGroup P
    inst✝⁸ : Module R M
    inst✝⁷ : Module R N
    inst✝⁶ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    M' : Type u_6
    N' : Type u_7
    P' : Type u_8
    inst✝⁵ : AddCommGroup M'
    inst✝⁴ : AddCommGroup N'
    inst✝³ : AddCommGroup P'
    inst✝² : Module R M'
    inst✝¹ : Module R N'
    inst✝ : Module R P'
    f' : LinearMap (RingHom.id R) M' N'
    g' : LinearMap (RingHom.id R) N' P'
    hfg' : Function.Exact ⇑f' ⇑g'
    hg' : Function.Surjective ⇑g'
    ⊢ Eq (LinearMap.ker (LinearMap.lTensor P g')) (LinearMap.range (LinearMap.lTen …
  -/
  rw [Exact.linearMap_ker_eq (lTensor_exact P hfg' hg')]
  /-
    🎉 no goals
  -/


/-- The ideal of `A ⊗[R] B` generated by `I` is the image of `I ⊗[R] B` -/
lemma Ideal.map_includeLeft_eq (I : Ideal A) :
    (I.map (Algebra.TensorProduct.includeLeft : A →ₐ[R] A ⊗[R] B)).restrictScalars R
      = LinearMap.range (LinearMap.rTensor B (Submodule.subtype (I.restrictScalars R))) := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_2
    B : Type u_3
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    I : Ideal A
    ⊢ Eq (Submodule.restrictScalars R (Ideal.map Algebra.TensorProduct.includeLeft …
  -/
  rw [← Submodule.carrier_inj]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_2
    B : Type u_3
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    I : Ideal A
    ⊢ Eq (Submodule.restrictScalars R (Ideal.map Algebra.TensorProduct.includeLeft …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_2
      B : Type u_3
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal A
      ⊢ LE.le (Submodule.restrictScalars R (Ideal.map Algebra.TensorProduct.includeL …
    -/
  · intro x
    simp only [AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
      Submodule.mem_toAddSubmonoid, Submodule.restrictScalars_mem, LinearMap.mem_range]
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_2
      B : Type u_3
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal A
      x : TensorProduct R A B
      ⊢ Membership.mem (Ideal.map Algebra.TensorProduct.includeLeft I) x → Exists fu …
    -/
    intro hx
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_2
      B : Type u_3
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal A
      x : TensorProduct R A B
      hx : Membership.mem (Ideal.map Algebra.TensorProduct.includeLeft I) x
      ⊢ Exists fun y => Eq ((LinearMap.rTensor B (Submodule.restrictScalars R I).sub …
    -/
    rw [Ideal.map, ← submodule_span_eq] at hx
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_2
      B : Type u_3
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal A
      x : TensorProduct R A B
      hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
      ⊢ Exists fun y => Eq ((LinearMap.rTensor B (Submodule.restrictScalars R I).sub …
    -/
    refine Submodule.span_induction ?_ ?_ ?_ ?_ hx
      /-
        case a.refine_1
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal A
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        ⊢ ∀ (x : TensorProduct R A B), Membership.mem (Set.image ⇑Algebra.TensorProduc …
      -/
    · intro x
      /-
        case a.refine_1
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal A
        x✝ : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        x : TensorProduct R A B
        ⊢ Membership.mem (Set.image ⇑Algebra.TensorProduct.includeLeft ↑I) x → Exists  …
      -/
      simp only [includeLeft_apply, Set.mem_image, SetLike.mem_coe]
      /-
        case a.refine_1
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal A
        x✝ : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        x : TensorProduct R A B
        ⊢ (Exists fun x_1 => And (Membership.mem I x_1) (Eq (TensorProduct.tmul R x_1  …
      -/
      rintro ⟨y, hy, rfl⟩
      /-
        case a.refine_1.intro.intro
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal A
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        y : A
        hy : Membership.mem I y
        ⊢ Exists fun y_1 => Eq ((LinearMap.rTensor B (Submodule.restrictScalars R I).s …
      -/
      use ⟨y, hy⟩ ⊗ₜ[R] 1
      /-
        case h
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal A
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        y : A
        hy : Membership.mem I y
        ⊢ Eq ((LinearMap.rTensor B (Submodule.restrictScalars R I).subtype) (TensorPro …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case a.refine_2
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal A
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        ⊢ Exists fun y => Eq ((LinearMap.rTensor B (Submodule.restrictScalars R I).sub …
      -/
    · use 0
      /-
        case h
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal A
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        ⊢ Eq ((LinearMap.rTensor B (Submodule.restrictScalars R I).subtype) 0) 0
      -/
      simp only [map_zero]
      /-
        🎉 no goals
      -/
      /-
        case a.refine_3
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal A
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        ⊢ ∀ (x y : TensorProduct R A B), Membership.mem (Submodule.span (TensorProduct …
      -/
    · rintro x y - - ⟨x, hx, rfl⟩ ⟨y, hy, rfl⟩
      /-
        case a.refine_3.intro.refl.intro.refl
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal A
        x✝ : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        x y : TensorProduct R (Subtype fun x => Membership.mem (Submodule.restrictScal …
        ⊢ Exists fun y_1 => Eq ((LinearMap.rTensor B (Submodule.restrictScalars R I).s …
      -/
      use x + y
      /-
        case h
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal A
        x✝ : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        x y : TensorProduct R (Subtype fun x => Membership.mem (Submodule.restrictScal …
        ⊢ Eq ((LinearMap.rTensor B (Submodule.restrictScalars R I).subtype) (HAdd.hAdd …
      -/
      simp only [map_add]
      /-
        🎉 no goals
      -/
      /-
        case a.refine_4
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal A
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        ⊢ ∀ (a x : TensorProduct R A B), Membership.mem (Submodule.span (TensorProduct …
      -/
    · rintro a x - ⟨x, hx, rfl⟩
      induction a with
      | zero =>
        use 0
        simp only [map_zero, smul_eq_mul, zero_mul]
      | tmul a b =>
        induction x with
        | zero =>
          use 0
          simp only [map_zero, smul_eq_mul, mul_zero]
        | tmul x y =>
          use (a • x) ⊗ₜ[R] (b * y)
          simp only [LinearMap.lTensor_tmul, Submodule.coe_subtype, smul_eq_mul, tmul_mul_tmul]
          with_unfolding_all rfl
        | add x y hx hy =>
          obtain ⟨x', hx'⟩ := hx
          obtain ⟨y', hy'⟩ := hy
          use x' + y'
          simp only [map_add, hx', smul_add, hy']
      | add a b ha hb =>
        obtain ⟨x', ha'⟩ := ha
        obtain ⟨y', hb'⟩ := hb
        use x' + y'
        simp only [map_add, ha', add_smul, hb']

    /-
      case a
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_2
      B : Type u_3
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal A
      ⊢ LE.le (LinearMap.range (LinearMap.rTensor B (Submodule.restrictScalars R I). …
    -/
  · rintro x ⟨y, rfl⟩
    induction y with
    | zero =>
        rw [map_zero]
        apply zero_mem
    | tmul a b =>
        simp only [LinearMap.rTensor_tmul, Submodule.coe_subtype]
        suffices (a : A) ⊗ₜ[R] b = ((1 : A) ⊗ₜ[R] b) * ((a : A) ⊗ₜ[R] (1 : B)) by
          simp only [AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
            Submodule.mem_toAddSubmonoid, Submodule.restrictScalars_mem]
          rw [this]
          apply Ideal.mul_mem_left
          -- Note: adding `includeLeft` as a hint fixes a timeout https://github.com/leanprover-community/mathlib4/pull/8386
          apply Ideal.mem_map_of_mem includeLeft
          exact Submodule.coe_mem a
        simp only [Submodule.coe_restrictScalars, Algebra.TensorProduct.tmul_mul_tmul,
          mul_one, one_mul]
    | add x y hx hy =>
        rw [map_add]
        apply Submodule.add_mem _ hx hy


/-- The ideal of `A ⊗[R] B` generated by `I` is the image of `A ⊗[R] I` -/
lemma Ideal.map_includeRight_eq (I : Ideal B) :
    (I.map (Algebra.TensorProduct.includeRight : B →ₐ[R] A ⊗[R] B)).restrictScalars R
      = LinearMap.range (LinearMap.lTensor A (Submodule.subtype (I.restrictScalars R))) := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_2
    B : Type u_3
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    I : Ideal B
    ⊢ Eq (Submodule.restrictScalars R (Ideal.map Algebra.TensorProduct.includeRigh …
  -/
  rw [← Submodule.carrier_inj]
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    A : Type u_2
    B : Type u_3
    inst✝³ : Semiring A
    inst✝² : Semiring B
    inst✝¹ : Algebra R A
    inst✝ : Algebra R B
    I : Ideal B
    ⊢ Eq (Submodule.restrictScalars R (Ideal.map Algebra.TensorProduct.includeRigh …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_2
      B : Type u_3
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal B
      ⊢ LE.le (Submodule.restrictScalars R (Ideal.map Algebra.TensorProduct.includeR …
    -/
  · intro x
    simp only [AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
      Submodule.mem_toAddSubmonoid, Submodule.restrictScalars_mem, LinearMap.mem_range]
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_2
      B : Type u_3
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal B
      x : TensorProduct R A B
      ⊢ Membership.mem (Ideal.map Algebra.TensorProduct.includeRight I) x → Exists f …
    -/
    intro hx
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_2
      B : Type u_3
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal B
      x : TensorProduct R A B
      hx : Membership.mem (Ideal.map Algebra.TensorProduct.includeRight I) x
      ⊢ Exists fun y => Eq ((LinearMap.lTensor A (Submodule.restrictScalars R I).sub …
    -/
    rw [Ideal.map, ← submodule_span_eq] at hx
    /-
      case a
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_2
      B : Type u_3
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal B
      x : TensorProduct R A B
      hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
      ⊢ Exists fun y => Eq ((LinearMap.lTensor A (Submodule.restrictScalars R I).sub …
    -/
    refine Submodule.span_induction ?_ ?_ ?_ ?_ hx
      /-
        case a.refine_1
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal B
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        ⊢ ∀ (x : TensorProduct R A B), Membership.mem (Set.image ⇑Algebra.TensorProduc …
      -/
    · intro x
      /-
        case a.refine_1
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal B
        x✝ : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        x : TensorProduct R A B
        ⊢ Membership.mem (Set.image ⇑Algebra.TensorProduct.includeRight ↑I) x → Exists …
      -/
      simp only [includeRight_apply, Set.mem_image, SetLike.mem_coe]
      /-
        case a.refine_1
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal B
        x✝ : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        x : TensorProduct R A B
        ⊢ (Exists fun x_1 => And (Membership.mem I x_1) (Eq (TensorProduct.tmul R 1 x_ …
      -/
      rintro ⟨y, hy, rfl⟩
      /-
        case a.refine_1.intro.intro
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal B
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        y : B
        hy : Membership.mem I y
        ⊢ Exists fun y_1 => Eq ((LinearMap.lTensor A (Submodule.restrictScalars R I).s …
      -/
      use 1 ⊗ₜ[R] ⟨y, hy⟩
      /-
        case h
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal B
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        y : B
        hy : Membership.mem I y
        ⊢ Eq ((LinearMap.lTensor A (Submodule.restrictScalars R I).subtype) (TensorPro …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case a.refine_2
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal B
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        ⊢ Exists fun y => Eq ((LinearMap.lTensor A (Submodule.restrictScalars R I).sub …
      -/
    · use 0
      /-
        case h
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal B
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        ⊢ Eq ((LinearMap.lTensor A (Submodule.restrictScalars R I).subtype) 0) 0
      -/
      simp only [map_zero]
      /-
        🎉 no goals
      -/
      /-
        case a.refine_3
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal B
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        ⊢ ∀ (x y : TensorProduct R A B), Membership.mem (Submodule.span (TensorProduct …
      -/
    · rintro x y - - ⟨x, hx, rfl⟩ ⟨y, hy, rfl⟩
      /-
        case a.refine_3.intro.refl.intro.refl
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal B
        x✝ : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        x y : TensorProduct R A (Subtype fun x => Membership.mem (Submodule.restrictSc …
        ⊢ Exists fun y_1 => Eq ((LinearMap.lTensor A (Submodule.restrictScalars R I).s …
      -/
      use x + y
      /-
        case h
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal B
        x✝ : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        x y : TensorProduct R A (Subtype fun x => Membership.mem (Submodule.restrictSc …
        ⊢ Eq ((LinearMap.lTensor A (Submodule.restrictScalars R I).subtype) (HAdd.hAdd …
      -/
      simp only [map_add]
      /-
        🎉 no goals
      -/
      /-
        case a.refine_4
        R : Type u_1
        inst✝⁴ : CommSemiring R
        A : Type u_2
        B : Type u_3
        inst✝³ : Semiring A
        inst✝² : Semiring B
        inst✝¹ : Algebra R A
        inst✝ : Algebra R B
        I : Ideal B
        x : TensorProduct R A B
        hx : Membership.mem (Submodule.span (TensorProduct R A B) (Set.image ⇑Algebra. …
        ⊢ ∀ (a x : TensorProduct R A B), Membership.mem (Submodule.span (TensorProduct …
      -/
    · rintro a x - ⟨x, hx, rfl⟩
      induction a with
      | zero =>
        use 0
        simp only [map_zero, smul_eq_mul, zero_mul]
      | tmul a b =>
        induction x with
        | zero =>
          use 0
          simp only [map_zero, smul_eq_mul, mul_zero]
        | tmul x y =>
          use (a * x) ⊗ₜ[R] (b •y)
          simp only [LinearMap.lTensor_tmul, Submodule.coe_subtype, smul_eq_mul, tmul_mul_tmul]
          rfl
        | add x y hx hy =>
          obtain ⟨x', hx'⟩ := hx
          obtain ⟨y', hy'⟩ := hy
          use x' + y'
          simp only [map_add, hx', smul_add, hy']
      | add a b ha hb =>
        obtain ⟨x', ha'⟩ := ha
        obtain ⟨y', hb'⟩ := hb
        use x' + y'
        simp only [map_add, ha', add_smul, hb']

    /-
      case a
      R : Type u_1
      inst✝⁴ : CommSemiring R
      A : Type u_2
      B : Type u_3
      inst✝³ : Semiring A
      inst✝² : Semiring B
      inst✝¹ : Algebra R A
      inst✝ : Algebra R B
      I : Ideal B
      ⊢ LE.le (LinearMap.range (LinearMap.lTensor A (Submodule.restrictScalars R I). …
    -/
  · rintro x ⟨y, rfl⟩
    induction y with
    | zero =>
        rw [map_zero]
        apply zero_mem
    | tmul a b =>
        simp only [LinearMap.lTensor_tmul, Submodule.coe_subtype]
        suffices a ⊗ₜ[R] (b : B) = (a ⊗ₜ[R] (1 : B)) * ((1 : A) ⊗ₜ[R] (b : B)) by
          rw [this]
          simp only [AddSubsemigroup.mem_carrier, AddSubmonoid.mem_toSubsemigroup,
            Submodule.mem_toAddSubmonoid, Submodule.restrictScalars_mem]
          apply Ideal.mul_mem_left
          -- Note: adding `includeRight` as a hint fixes a timeout https://github.com/leanprover-community/mathlib4/pull/8386
          apply Ideal.mem_map_of_mem includeRight
          exact Submodule.coe_mem b
        simp only [Submodule.coe_restrictScalars, Algebra.TensorProduct.tmul_mul_tmul,
          mul_one, one_mul]
    | add x y hx hy =>
        rw [map_add]
        apply Submodule.add_mem _ hx hy

-- Now, we can prove the right exactness properties of the tensor product,
-- in its versions for algebras


/-- If `g` is surjective, then the kernel of `(id A) ⊗ g` is generated by the kernel of `g` -/
lemma Algebra.TensorProduct.lTensor_ker (hg : Function.Surjective g) :
    RingHom.ker (map (AlgHom.id R A) g) =
      (RingHom.ker g).map (Algebra.TensorProduct.includeRight : C →ₐ[R] A ⊗[R] C) := by
  /-
    R : Type u_4
    inst✝⁶ : CommRing R
    A : Type u_5
    C : Type u_7
    D : Type u_8
    inst✝⁵ : Ring A
    inst✝⁴ : Ring C
    inst✝³ : Ring D
    inst✝² : Algebra R A
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    g : AlgHom R C D
    hg : Function.Surjective ⇑g
    ⊢ Eq (RingHom.ker (Algebra.TensorProduct.map (AlgHom.id R A) g)) (Ideal.map Al …
  -/
  rw [← Submodule.restrictScalars_inj R]
  have : (RingHom.ker (map (AlgHom.id R A) g)).restrictScalars R =
    LinearMap.ker (LinearMap.lTensor A (AlgHom.toLinearMap g)) := rfl
  /-
    R : Type u_4
    inst✝⁶ : CommRing R
    A : Type u_5
    C : Type u_7
    D : Type u_8
    inst✝⁵ : Ring A
    inst✝⁴ : Ring C
    inst✝³ : Ring D
    inst✝² : Algebra R A
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    g : AlgHom R C D
    hg : Function.Surjective ⇑g
    this : Eq (Submodule.restrictScalars R (RingHom.ker (Algebra.TensorProduct.map …
    ⊢ Eq (Submodule.restrictScalars R (RingHom.ker (Algebra.TensorProduct.map (Alg …
  -/
  rw [this, Ideal.map_includeRight_eq]
  /-
    R : Type u_4
    inst✝⁶ : CommRing R
    A : Type u_5
    C : Type u_7
    D : Type u_8
    inst✝⁵ : Ring A
    inst✝⁴ : Ring C
    inst✝³ : Ring D
    inst✝² : Algebra R A
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    g : AlgHom R C D
    hg : Function.Surjective ⇑g
    this : Eq (Submodule.restrictScalars R (RingHom.ker (Algebra.TensorProduct.map …
    ⊢ Eq (LinearMap.ker (LinearMap.lTensor A g.toLinearMap)) (LinearMap.range (Lin …
  -/
  rw [(lTensor_exact A g.toLinearMap.exact_subtype_ker_map hg).linearMap_ker_eq]
  /-
    R : Type u_4
    inst✝⁶ : CommRing R
    A : Type u_5
    C : Type u_7
    D : Type u_8
    inst✝⁵ : Ring A
    inst✝⁴ : Ring C
    inst✝³ : Ring D
    inst✝² : Algebra R A
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    g : AlgHom R C D
    hg : Function.Surjective ⇑g
    this : Eq (Submodule.restrictScalars R (RingHom.ker (Algebra.TensorProduct.map …
    ⊢ Eq (LinearMap.range (LinearMap.lTensor A (LinearMap.ker g.toLinearMap).subty …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `f` is surjective, then the kernel of `f ⊗ (id B)` is generated by the kernel of `f` -/
lemma Algebra.TensorProduct.rTensor_ker (hf : Function.Surjective f) :
    RingHom.ker (map f (AlgHom.id R C)) =
      (RingHom.ker f).map (Algebra.TensorProduct.includeLeft : A →ₐ[R] A ⊗[R] C) := by
  /-
    R : Type u_4
    inst✝⁶ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    inst✝⁵ : Ring A
    inst✝⁴ : Ring B
    inst✝³ : Ring C
    inst✝² : Algebra R A
    inst✝¹ : Algebra R B
    inst✝ : Algebra R C
    f : AlgHom R A B
    hf : Function.Surjective ⇑f
    ⊢ Eq (RingHom.ker (Algebra.TensorProduct.map f (AlgHom.id R C))) (Ideal.map Al …
  -/
  rw [← Submodule.restrictScalars_inj R]
  have : (RingHom.ker (map f (AlgHom.id R C))).restrictScalars R =
    LinearMap.ker (LinearMap.rTensor C (AlgHom.toLinearMap f)) := rfl
  /-
    R : Type u_4
    inst✝⁶ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    inst✝⁵ : Ring A
    inst✝⁴ : Ring B
    inst✝³ : Ring C
    inst✝² : Algebra R A
    inst✝¹ : Algebra R B
    inst✝ : Algebra R C
    f : AlgHom R A B
    hf : Function.Surjective ⇑f
    this : Eq (Submodule.restrictScalars R (RingHom.ker (Algebra.TensorProduct.map …
    ⊢ Eq (Submodule.restrictScalars R (RingHom.ker (Algebra.TensorProduct.map f (A …
  -/
  rw [this, Ideal.map_includeLeft_eq]
  /-
    R : Type u_4
    inst✝⁶ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    inst✝⁵ : Ring A
    inst✝⁴ : Ring B
    inst✝³ : Ring C
    inst✝² : Algebra R A
    inst✝¹ : Algebra R B
    inst✝ : Algebra R C
    f : AlgHom R A B
    hf : Function.Surjective ⇑f
    this : Eq (Submodule.restrictScalars R (RingHom.ker (Algebra.TensorProduct.map …
    ⊢ Eq (LinearMap.ker (LinearMap.rTensor C f.toLinearMap)) (LinearMap.range (Lin …
  -/
  rw [(rTensor_exact C f.toLinearMap.exact_subtype_ker_map hf).linearMap_ker_eq]
  /-
    R : Type u_4
    inst✝⁶ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    inst✝⁵ : Ring A
    inst✝⁴ : Ring B
    inst✝³ : Ring C
    inst✝² : Algebra R A
    inst✝¹ : Algebra R B
    inst✝ : Algebra R C
    f : AlgHom R A B
    hf : Function.Surjective ⇑f
    this : Eq (Submodule.restrictScalars R (RingHom.ker (Algebra.TensorProduct.map …
    ⊢ Eq (LinearMap.range (LinearMap.rTensor C (LinearMap.ker f.toLinearMap).subty …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `f` and `g` are surjective morphisms of algebras, then
  the kernel of `Algebra.TensorProduct.map f g` is generated by the kernels of `f` and `g` -/
theorem Algebra.TensorProduct.map_ker (hf : Function.Surjective f) (hg : Function.Surjective g) :
    RingHom.ker (map f g) =
      (RingHom.ker f).map (Algebra.TensorProduct.includeLeft : A →ₐ[R] A ⊗[R] C) ⊔
        (RingHom.ker g).map (Algebra.TensorProduct.includeRight : C →ₐ[R] A ⊗[R] C) := by
  -- rewrite map f g as the composition of two maps
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    ⊢ Eq (RingHom.ker (Algebra.TensorProduct.map f g)) (Max.max (Ideal.map Algebra …
  -/
  have : map f g = (map f (AlgHom.id R D)).comp (map (AlgHom.id R A) g) := ext rfl rfl
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    this : Eq (Algebra.TensorProduct.map f g) ((Algebra.TensorProduct.map f (AlgHo …
    ⊢ Eq (RingHom.ker (Algebra.TensorProduct.map f g)) (Max.max (Ideal.map Algebra …
  -/
  rw [this]
  -- this needs some rewriting to RingHom
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    this : Eq (Algebra.TensorProduct.map f g) ((Algebra.TensorProduct.map f (AlgHo …
    ⊢ Eq (RingHom.ker ((Algebra.TensorProduct.map f (AlgHom.id R D)).comp (Algebra …
  -/
  simp only [AlgHom.coe_ker, AlgHom.comp_toRingHom]
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    this : Eq (Algebra.TensorProduct.map f g) ((Algebra.TensorProduct.map f (AlgHo …
    ⊢ Eq (RingHom.ker ((↑(Algebra.TensorProduct.map f (AlgHom.id R D))).comp ↑(Alg …
  -/
  rw [← RingHom.comap_ker]
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    this : Eq (Algebra.TensorProduct.map f g) ((Algebra.TensorProduct.map f (AlgHo …
    ⊢ Eq (Ideal.comap (↑(Algebra.TensorProduct.map (AlgHom.id R A) g)) (RingHom.ke …
  -/
  simp only [← AlgHom.coe_ker]
  -- apply one step of exactness
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    this : Eq (Algebra.TensorProduct.map f g) ((Algebra.TensorProduct.map f (AlgHo …
    ⊢ Eq (Ideal.comap (↑(Algebra.TensorProduct.map (AlgHom.id R A) g)) (RingHom.ke …
  -/
  rw [← Algebra.TensorProduct.lTensor_ker _ hg, RingHom.ker_eq_comap_bot (map (AlgHom.id R A) g)]
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    this : Eq (Algebra.TensorProduct.map f g) ((Algebra.TensorProduct.map f (AlgHo …
    ⊢ Eq (Ideal.comap (↑(Algebra.TensorProduct.map (AlgHom.id R A) g)) (RingHom.ke …
  -/
  rw [← Ideal.comap_map_of_surjective (map (AlgHom.id R A) g) (LinearMap.lTensor_surjective A hg)]
  -- apply the other step of exactness
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    this : Eq (Algebra.TensorProduct.map f g) ((Algebra.TensorProduct.map f (AlgHo …
    ⊢ Eq (Ideal.comap (↑(Algebra.TensorProduct.map (AlgHom.id R A) g)) (RingHom.ke …
  -/
  rw [Algebra.TensorProduct.rTensor_ker _ hf]
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    this : Eq (Algebra.TensorProduct.map f g) ((Algebra.TensorProduct.map f (AlgHo …
    ⊢ Eq (Ideal.comap (↑(Algebra.TensorProduct.map (AlgHom.id R A) g)) (Ideal.map  …
  -/
  apply congr_arg₂ _ rfl
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    this : Eq (Algebra.TensorProduct.map f g) ((Algebra.TensorProduct.map f (AlgHo …
    ⊢ Eq (Ideal.map Algebra.TensorProduct.includeLeft (RingHom.ker f)) (Ideal.map  …
  -/
  simp only [AlgHom.coe_ideal_map, Ideal.map_map]
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    this : Eq (Algebra.TensorProduct.map f g) ((Algebra.TensorProduct.map f (AlgHo …
    ⊢ Eq (Ideal.map (↑Algebra.TensorProduct.includeLeft) (RingHom.ker f)) (Ideal.m …
  -/
  rw [← AlgHom.comp_toRingHom, Algebra.TensorProduct.map_comp_includeLeft]
  /-
    R : Type u_4
    inst✝⁸ : CommRing R
    A : Type u_5
    B : Type u_6
    C : Type u_7
    D : Type u_8
    inst✝⁷ : Ring A
    inst✝⁶ : Ring B
    inst✝⁵ : Ring C
    inst✝⁴ : Ring D
    inst✝³ : Algebra R A
    inst✝² : Algebra R B
    inst✝¹ : Algebra R C
    inst✝ : Algebra R D
    f : AlgHom R A B
    g : AlgHom R C D
    hf : Function.Surjective ⇑f
    hg : Function.Surjective ⇑g
    this : Eq (Algebra.TensorProduct.map f g) ((Algebra.TensorProduct.map f (AlgHo …
    ⊢ Eq (Ideal.map (↑Algebra.TensorProduct.includeLeft) (RingHom.ker f)) (Ideal.m …
  -/
  rfl
  /-
    🎉 no goals
  -/


