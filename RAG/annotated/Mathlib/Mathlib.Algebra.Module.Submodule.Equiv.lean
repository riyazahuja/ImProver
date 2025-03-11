/-- Linear equivalence between two equal submodules. -/
def ofEq (h : p = q) : p ≃ₗ[R] q :=
  { Equiv.Set.ofEq (congr_arg _ h) with
    map_smul' := fun _ _ => rfl
    map_add' := fun _ _ => rfl }


@[simp]
theorem coe_ofEq_apply (h : p = q) (x : p) : (ofEq p q h x : M) = x :=
  rfl


@[simp]
theorem ofEq_symm (h : p = q) : (ofEq p q h).symm = ofEq q p h.symm :=
  rfl


@[simp]
                                                             /-
                                                               R : Type u_1
                                                               M : Type u_5
                                                               inst✝¹ : Semiring R
                                                               inst✝ : AddCommMonoid M
                                                               module_M : Module R M
                                                               p : Submodule R M
                                                               ⊢ Eq (LinearEquiv.ofEq p p ⋯) (LinearEquiv.refl R (Subtype fun x => Membership …
                                                             -/
theorem ofEq_rfl : ofEq p p rfl = LinearEquiv.refl R p := by ext; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- A linear equivalence which maps a submodule of one module onto another, restricts to a linear
equivalence of the two submodules. -/
def ofSubmodules (p : Submodule R M) (q : Submodule R₂ M₂) (h : p.map (e : M →ₛₗ[σ₁₂] M₂) = q) :
    p ≃ₛₗ[σ₁₂] q :=
  (e.submoduleMap p).trans (LinearEquiv.ofEq _ _ h)


@[simp]
theorem ofSubmodules_apply {p : Submodule R M} {q : Submodule R₂ M₂} (h : p.map ↑e = q) (x : p) :
    ↑(e.ofSubmodules p q h x) = e x :=
  rfl


@[simp]
theorem ofSubmodules_symm_apply {p : Submodule R M} {q : Submodule R₂ M₂} (h : p.map ↑e = q)
    (x : q) : ↑((e.ofSubmodules p q h).symm x) = e.symm x :=
  rfl


/-- A linear equivalence of two modules restricts to a linear equivalence from the preimage of any
submodule to that submodule.

This is `LinearEquiv.ofSubmodule` but with `comap` on the left instead of `map` on the right. -/
def ofSubmodule' [Module R M] [Module R₂ M₂] (f : M ≃ₛₗ[σ₁₂] M₂) (U : Submodule R₂ M₂) :
    U.comap (f : M →ₛₗ[σ₁₂] M₂) ≃ₛₗ[σ₁₂] U :=
  (f.symm.ofSubmodules _ _ f.symm.map_eq_comap).symm


theorem ofSubmodule'_toLinearMap [Module R M] [Module R₂ M₂] (f : M ≃ₛₗ[σ₁₂] M₂)
    (U : Submodule R₂ M₂) :
    (f.ofSubmodule' U).toLinearMap = (f.toLinearMap.domRestrict _).codRestrict _ Subtype.prop := by
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    f : LinearEquiv σ₁₂ M M₂
    U : Submodule R₂ M₂
    ⊢ Eq (↑(f.ofSubmodule' U)) (LinearMap.codRestrict U ((↑f).domRestrict (Submodu …
  -/
  ext
  /-
    case h.a
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    inst✝¹ : Module R M
    inst✝ : Module R₂ M₂
    f : LinearEquiv σ₁₂ M M₂
    U : Submodule R₂ M₂
    x✝ : Subtype fun x => Membership.mem (Submodule.comap (↑f) U) x
    ⊢ Eq ↑(↑(f.ofSubmodule' U) x✝) ↑((LinearMap.codRestrict U ((↑f).domRestrict (S …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ofSubmodule'_apply [Module R M] [Module R₂ M₂] (f : M ≃ₛₗ[σ₁₂] M₂) (U : Submodule R₂ M₂)
    (x : U.comap (f : M →ₛₗ[σ₁₂] M₂)) : (f.ofSubmodule' U x : M₂) = f (x : M) :=
  rfl


@[simp]
theorem ofSubmodule'_symm_apply [Module R M] [Module R₂ M₂] (f : M ≃ₛₗ[σ₁₂] M₂)
    (U : Submodule R₂ M₂) (x : U) : ((f.ofSubmodule' U).symm x : M) = f.symm (x : M₂) :=
  rfl


/-- The top submodule of `M` is linearly equivalent to `M`. -/
def ofTop (h : p = ⊤) : p ≃ₗ[R] M :=
  { p.subtype with
    invFun := fun x => ⟨x, h.symm ▸ trivial⟩
    left_inv := fun _ => rfl
    right_inv := fun _ => rfl }


@[simp]
theorem ofTop_apply {h} (x : p) : ofTop p h x = x :=
  rfl


@[simp]
theorem coe_ofTop_symm_apply {h} (x : M) : ((ofTop p h).symm x : M) = x :=
  rfl


theorem ofTop_symm_apply {h} (x : M) : (ofTop p h).symm x = ⟨x, h.symm ▸ trivial⟩ :=
  rfl


@[simp]
protected theorem range : LinearMap.range (e : M →ₛₗ[σ₁₂] M₂) = ⊤ :=
  LinearMap.range_eq_top.2 e.toEquiv.surjective


@[simp]
protected theorem _root_.LinearEquivClass.range [Module R M] [Module R₂ M₂] {F : Type*}
    [EquivLike F M M₂] [SemilinearEquivClass F σ₁₂ M M₂] (e : F) : LinearMap.range e = ⊤ :=
  LinearMap.range_eq_top.2 (EquivLike.surjective e)


theorem eq_bot_of_equiv [Module R₂ M₂] (e : p ≃ₛₗ[σ₁₂] (⊥ : Submodule R₂ M₂)) : p = ⊥ := by
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁴ : Semiring R
    inst✝³ : Semiring R₂
    inst✝² : AddCommMonoid M
    inst✝¹ : AddCommMonoid M₂
    module_M : Module R M
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    p : Submodule R M
    inst✝ : Module R₂ M₂
    e : LinearEquiv σ₁₂ (Subtype fun x => Membership.mem p x) (Subtype fun x => Me …
    ⊢ Eq p Bot.bot
  -/
  refine bot_unique (SetLike.le_def.2 fun b hb => (Submodule.mem_bot R).2 ?_)
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁴ : Semiring R
    inst✝³ : Semiring R₂
    inst✝² : AddCommMonoid M
    inst✝¹ : AddCommMonoid M₂
    module_M : Module R M
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    p : Submodule R M
    inst✝ : Module R₂ M₂
    e : LinearEquiv σ₁₂ (Subtype fun x => Membership.mem p x) (Subtype fun x => Me …
    b : M
    hb : Membership.mem p b
    ⊢ Eq b 0
  -/
  rw [← p.mk_eq_zero hb, ← e.map_eq_zero_iff]
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁴ : Semiring R
    inst✝³ : Semiring R₂
    inst✝² : AddCommMonoid M
    inst✝¹ : AddCommMonoid M₂
    module_M : Module R M
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    re₁₂ : RingHomInvPair σ₁₂ σ₂₁
    re₂₁ : RingHomInvPair σ₂₁ σ₁₂
    p : Submodule R M
    inst✝ : Module R₂ M₂
    e : LinearEquiv σ₁₂ (Subtype fun x => Membership.mem p x) (Subtype fun x => Me …
    b : M
    hb : Membership.mem p b
    ⊢ Eq (e ⟨b, hb⟩) 0
  -/
  apply Submodule.eq_zero_of_bot_submodule
  /-
    🎉 no goals
  -/

-- Porting note: `RingHomSurjective σ₁₂` is an unused argument.

@[simp]
theorem range_comp [RingHomSurjective σ₂₃] [RingHomSurjective σ₁₃] :
    LinearMap.range (h.comp (e : M →ₛₗ[σ₁₂] M₂) : M →ₛₗ[σ₁₃] M₃) = LinearMap.range h :=
  LinearMap.range_comp_of_range_eq_top _ e.range


/-- A linear map `f : M →ₗ[R] M₂` with a left-inverse `g : M₂ →ₗ[R] M` defines a linear
equivalence between `M` and `f.range`.

This is a computable alternative to `LinearEquiv.ofInjective`, and a bidirectional version of
`LinearMap.rangeRestrict`. -/
def ofLeftInverse [RingHomInvPair σ₁₂ σ₂₁] [RingHomInvPair σ₂₁ σ₁₂] {g : M₂ → M}
    (h : Function.LeftInverse g f) : M ≃ₛₗ[σ₁₂] (LinearMap.range f) :=
  { LinearMap.rangeRestrict f with
    toFun := LinearMap.rangeRestrict f
    invFun := g ∘ (LinearMap.range f).subtype
    left_inv := h
    right_inv := fun x =>
      Subtype.ext <|
        let ⟨x', hx'⟩ := LinearMap.mem_range.mp x.prop
                            /-
                              R : Type u_1
                              R₁ : Type u_2
                              R₂ : Type u_3
                              R₃ : Type u_4
                              M : Type u_5
                              M₁ : Type u_6
                              M₂ : Type u_7
                              M₃ : Type u_8
                              N : Type u_9
                              inst✝⁸ : Semiring R
                              inst✝⁷ : Semiring R₂
                              inst✝⁶ : Semiring R₃
                              inst✝⁵ : AddCommMonoid M
                              inst✝⁴ : AddCommMonoid M₂
                              inst✝³ : AddCommMonoid M₃
                              module_M : Module R M
                              module_M₂ : Module R₂ M₂
                              module_M₃ : Module R₃ M₃
                              σ₁₂ : RingHom R R₂
                              σ₂₁ : RingHom R₂ R
                              σ₂₃ : RingHom R₂ R₃
                              σ₁₃ : RingHom R R₃
                              inst✝² : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                              σ₃₂ : RingHom R₃ R₂
                              re₁₂ : RingHomInvPair σ₁₂ σ₂₁
                              re₂₁ : RingHomInvPair σ₂₁ σ₁₂
                              re₂₃ : RingHomInvPair σ₂₃ σ₃₂
                              re₃₂ : RingHomInvPair σ₃₂ σ₂₃
                              f : LinearMap σ₁₂ M M₂
                              g✝ : LinearMap σ₂₁ M₂ M
                              e : LinearEquiv σ₁₂ M M₂
                              h✝ : LinearMap σ₂₃ M₂ M₃
                              p q : Submodule R M
                              inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
                              inst✝ : RingHomInvPair σ₂₁ σ₁₂
                              g : M₂ → M
                              h : Function.LeftInverse g ⇑f
                              x : Subtype fun x => Membership.mem (LinearMap.range f) x
                              x' : M
                              hx' : Eq (f x') ↑x
                              ⊢ Eq (f (g ↑x)) ↑x
                            -/
        show f (g x) = x by rw [← hx', h x'] }
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem ofLeftInverse_apply [RingHomInvPair σ₁₂ σ₂₁] [RingHomInvPair σ₂₁ σ₁₂]
    (h : Function.LeftInverse g f) (x : M) : ↑(ofLeftInverse h x) = f x :=
  rfl


@[simp]
theorem ofLeftInverse_symm_apply [RingHomInvPair σ₁₂ σ₂₁] [RingHomInvPair σ₂₁ σ₁₂]
    (h : Function.LeftInverse g f) (x : LinearMap.range f) : (ofLeftInverse h).symm x = g x :=
  rfl


/-- An `Injective` linear map `f : M →ₗ[R] M₂` defines a linear equivalence
between `M` and `f.range`. See also `LinearMap.ofLeftInverse`. -/
noncomputable def ofInjective [RingHomInvPair σ₁₂ σ₂₁] [RingHomInvPair σ₂₁ σ₁₂] (h : Injective f) :
    M ≃ₛₗ[σ₁₂] LinearMap.range f :=
  ofLeftInverse <| Classical.choose_spec h.hasLeftInverse


@[simp]
theorem ofInjective_apply [RingHomInvPair σ₁₂ σ₂₁] [RingHomInvPair σ₂₁ σ₁₂] {h : Injective f}
    (x : M) : ↑(ofInjective f h x) = f x :=
  rfl


@[simp]
lemma ofInjective_symm_apply [RingHomInvPair σ₁₂ σ₂₁] [RingHomInvPair σ₂₁ σ₁₂] {h : Injective f}
    (x : LinearMap.range f) :
    f ((ofInjective f h).symm x) = x := by
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    module_M : Module R M
    module_M₂ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    f : LinearMap σ₁₂ M M₂
    inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
    inst✝ : RingHomInvPair σ₂₁ σ₁₂
    h : Function.Injective ⇑f
    x : Subtype fun x => Membership.mem (LinearMap.range f) x
    ⊢ Eq (f ((LinearEquiv.ofInjective f h).symm x)) ↑x
  -/
  obtain ⟨-, ⟨y, rfl⟩⟩ := x
  /-
    case mk.intro
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    module_M : Module R M
    module_M₂ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    f : LinearMap σ₁₂ M M₂
    inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
    inst✝ : RingHomInvPair σ₂₁ σ₁₂
    h : Function.Injective ⇑f
    y : M
    ⊢ Eq (f ((LinearEquiv.ofInjective f h).symm ⟨f y, ⋯⟩)) ↑⟨f y, ⋯⟩
  -/
  have : ⟨f y, LinearMap.mem_range_self f y⟩ = LinearEquiv.ofInjective f h y := rfl
  /-
    case mk.intro
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    module_M : Module R M
    module_M₂ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    f : LinearMap σ₁₂ M M₂
    inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
    inst✝ : RingHomInvPair σ₂₁ σ₁₂
    h : Function.Injective ⇑f
    y : M
    this : Eq ⟨f y, ⋯⟩ ((LinearEquiv.ofInjective f h) y)
    ⊢ Eq (f ((LinearEquiv.ofInjective f h).symm ⟨f y, ⋯⟩)) ↑⟨f y, ⋯⟩
  -/
  simp [this]
  /-
    🎉 no goals
  -/


/-- A bijective linear map is a linear equivalence. -/
noncomputable def ofBijective [RingHomInvPair σ₁₂ σ₂₁] [RingHomInvPair σ₂₁ σ₁₂] (hf : Bijective f) :
    M ≃ₛₗ[σ₁₂] M₂ :=
  (ofInjective f hf.injective).trans <| ofTop _ <|
    LinearMap.range_eq_top.2 hf.surjective


@[simp]
theorem ofBijective_apply [RingHomInvPair σ₁₂ σ₂₁] [RingHomInvPair σ₂₁ σ₁₂] {hf} (x : M) :
    ofBijective f hf x = f x :=
  rfl


@[simp]
theorem ofBijective_symm_apply_apply [RingHomInvPair σ₁₂ σ₂₁] [RingHomInvPair σ₂₁ σ₁₂] {h} (x : M) :
    (ofBijective f h).symm (f x) = x := by
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    module_M : Module R M
    module_M₂ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    f : LinearMap σ₁₂ M M₂
    inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
    inst✝ : RingHomInvPair σ₂₁ σ₁₂
    h : Function.Bijective ⇑f
    x : M
    ⊢ Eq ((LinearEquiv.ofBijective f h).symm (f x)) x
  -/
  simp [LinearEquiv.symm_apply_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem apply_ofBijective_symm_apply [RingHomInvPair σ₁₂ σ₂₁] [RingHomInvPair σ₂₁ σ₁₂] {h}
    (x : M₂) : f ((ofBijective f h).symm x) = x := by
  /-
    R : Type u_1
    R₂ : Type u_3
    M : Type u_5
    M₂ : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : Semiring R₂
    inst✝³ : AddCommMonoid M
    inst✝² : AddCommMonoid M₂
    module_M : Module R M
    module_M₂ : Module R₂ M₂
    σ₁₂ : RingHom R R₂
    σ₂₁ : RingHom R₂ R
    f : LinearMap σ₁₂ M M₂
    inst✝¹ : RingHomInvPair σ₁₂ σ₂₁
    inst✝ : RingHomInvPair σ₂₁ σ₁₂
    h : Function.Bijective ⇑f
    x : M₂
    ⊢ Eq (f ((LinearEquiv.ofBijective f h).symm x)) x
  -/
  rw [← ofBijective_apply f ((ofBijective f h).symm x), apply_symm_apply]
  /-
    🎉 no goals
  -/


/-- Given `p` a submodule of the module `M` and `q` a submodule of `p`, `p.equivSubtypeMap q`
is the natural `LinearEquiv` between `q` and `q.map p.subtype`. -/
def equivSubtypeMap (p : Submodule R M) (q : Submodule R p) : q ≃ₗ[R] q.map p.subtype :=
                                                /-
                                                  R : Type u_1
                                                  R₁ : Type u_2
                                                  R₂ : Type u_3
                                                  R₃ : Type u_4
                                                  M : Type u_5
                                                  M₁ : Type u_6
                                                  M₂ : Type u_7
                                                  M₃ : Type u_8
                                                  N : Type u_9
                                                  inst✝⁴ : Semiring R
                                                  inst✝³ : AddCommMonoid M
                                                  inst✝² : Module R M
                                                  inst✝¹ : AddCommMonoid N
                                                  inst✝ : Module R N
                                                  p : Submodule R M
                                                  q : Submodule R (Subtype fun x => Membership.mem p x)
                                                  ⊢ ∀ (c : Subtype fun x => Membership.mem q x), Membership.mem (Submodule.map p …
                                                -/
  { (p.subtype.domRestrict q).codRestrict _ (by rintro ⟨x, hx⟩; exact ⟨x, hx, rfl⟩) with
                                                                /-
                                                                  🎉 no goals
                                                                -/
    invFun := by
      /-
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        N : Type u_9
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        inst✝¹ : AddCommMonoid N
        inst✝ : Module R N
        p : Submodule R M
        q : Submodule R (Subtype fun x => Membership.mem p x)
        ⊢ (Subtype fun x => Membership.mem (Submodule.map p.subtype q) x) → Subtype fu …
      -/
      rintro ⟨x, hx⟩
      /-
        case mk
        R : Type u_1
        R₁ : Type u_2
        R₂ : Type u_3
        R₃ : Type u_4
        M : Type u_5
        M₁ : Type u_6
        M₂ : Type u_7
        M₃ : Type u_8
        N : Type u_9
        inst✝⁴ : Semiring R
        inst✝³ : AddCommMonoid M
        inst✝² : Module R M
        inst✝¹ : AddCommMonoid N
        inst✝ : Module R N
        p : Submodule R M
        q : Submodule R (Subtype fun x => Membership.mem p x)
        x : M
        hx : Membership.mem (Submodule.map p.subtype q) x
        ⊢ Subtype fun x => Membership.mem q x
      -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
      refine ⟨⟨x, ?_⟩, ?_⟩ <;> rcases hx with ⟨⟨_, h⟩, _, rfl⟩ <;> assumption
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    left_inv := fun ⟨⟨_, _⟩, _⟩ => rfl
                                               /-
                                                 R : Type u_1
                                                 R₁ : Type u_2
                                                 R₂ : Type u_3
                                                 R₃ : Type u_4
                                                 M : Type u_5
                                                 M₁ : Type u_6
                                                 M₂ : Type u_7
                                                 M₃ : Type u_8
                                                 N : Type u_9
                                                 inst✝⁴ : Semiring R
                                                 inst✝³ : AddCommMonoid M
                                                 inst✝² : Module R M
                                                 inst✝¹ : AddCommMonoid N
                                                 inst✝ : Module R N
                                                 p : Submodule R M
                                                 q : Submodule R (Subtype fun x => Membership.mem p x)
                                                 x✝ : Subtype fun x => Membership.mem (Submodule.map p.subtype q) x
                                                 x : M
                                                 h : Membership.mem p x
                                                 left✝ : Membership.mem ↑q ⟨x, h⟩
                                                 ⊢ Eq (__src✝.toFun (Subtype.casesOn ⟨x, ⋯⟩ fun x hx => ⟨⟨x, ⋯⟩, ⋯⟩)) ⟨x, ⋯⟩
                                               -/
    right_inv := fun ⟨x, ⟨_, h⟩, _, rfl⟩ => by ext; rfl }
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp]
theorem equivSubtypeMap_apply {p : Submodule R M} {q : Submodule R p} (x : q) :
    (p.equivSubtypeMap q x : M) = p.subtype.domRestrict q x :=
  rfl


@[simp]
theorem equivSubtypeMap_symm_apply {p : Submodule R M} {q : Submodule R p} (x : q.map p.subtype) :
    ((p.equivSubtypeMap q).symm x : M) = x := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    p : Submodule R M
    q : Submodule R (Subtype fun x => Membership.mem p x)
    x : Subtype fun x => Membership.mem (Submodule.map p.subtype q) x
    ⊢ Eq ↑↑((p.equivSubtypeMap q).symm x) ↑x
  -/
  cases x
  /-
    case mk
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    p : Submodule R M
    q : Submodule R (Subtype fun x => Membership.mem p x)
    val✝ : M
    property✝ : Membership.mem (Submodule.map p.subtype q) val✝
    ⊢ Eq ↑↑((p.equivSubtypeMap q).symm ⟨val✝, property✝⟩) ↑⟨val✝, property✝⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A linear injection `M ↪ N` restricts to an equivalence `f⁻¹ p ≃ p` for any submodule `p`
contained in its range. -/
@[simps! apply]
noncomputable def comap_equiv_self_of_inj_of_le {f : M →ₗ[R] N} {p : Submodule R N}
    (hf : Injective f) (h : p ≤ LinearMap.range f) :
    p.comap f ≃ₗ[R] p :=
  LinearEquiv.ofBijective
  ((f ∘ₗ (p.comap f).subtype).codRestrict p <| fun ⟨_, hx⟩ ↦ mem_comap.mp hx)
                     /-
                       R : Type u_1
                       R₁ : Type u_2
                       R₂ : Type u_3
                       R₃ : Type u_4
                       M : Type u_5
                       M₁ : Type u_6
                       M₂ : Type u_7
                       M₃ : Type u_8
                       N : Type u_9
                       inst✝⁴ : Semiring R
                       inst✝³ : AddCommMonoid M
                       inst✝² : Module R M
                       inst✝¹ : AddCommMonoid N
                       inst✝ : Module R N
                       f : LinearMap (RingHom.id R) M N
                       p : Submodule R N
                       hf : Function.Injective ⇑f
                       h : LE.le p (LinearMap.range f)
                       x y : Subtype fun x => Membership.mem (Submodule.comap f p) x
                       hxy : Eq ((LinearMap.codRestrict p (f.comp (Submodule.comap f p).subtype) ⋯) x …
                       ⊢ Eq x y
                     -/
  (⟨fun x y hxy ↦ by simpa using hf (Subtype.ext_iff.mp hxy),
                     /-
                       🎉 no goals
                     -/
                     /-
                       R : Type u_1
                       R₁ : Type u_2
                       R₂ : Type u_3
                       R₃ : Type u_4
                       M : Type u_5
                       M₁ : Type u_6
                       M₂ : Type u_7
                       M₃ : Type u_8
                       N : Type u_9
                       inst✝⁴ : Semiring R
                       inst✝³ : AddCommMonoid M
                       inst✝² : Module R M
                       inst✝¹ : AddCommMonoid N
                       inst✝ : Module R N
                       f : LinearMap (RingHom.id R) M N
                       p : Submodule R N
                       hf : Function.Injective ⇑f
                       h : LE.le p (LinearMap.range f)
                       x✝ : Subtype fun x => Membership.mem p x
                       x : N
                       hx : Membership.mem p x
                       ⊢ Exists fun a => Eq ((LinearMap.codRestrict p (f.comp (Submodule.comap f p).s …
                     -/
    fun ⟨x, hx⟩ ↦ by obtain ⟨y, rfl⟩ := h hx; exact ⟨⟨y, hx⟩, by simp [Subtype.ext_iff]⟩⟩)
                                              /-
                                                🎉 no goals
                                              -/


/-- The restriction of a linear map on the target to a submodule of the target given by
an inclusion. -/
noncomputable def codRestrictOfInjective : M₁ →ₗ[R] M₃ :=
  (LinearEquiv.ofInjective i hi).symm ∘ₗ f.codRestrict (LinearMap.range i) hf


@[simp]
lemma codRestrictOfInjective_comp_apply (x : M₁) :
    i (LinearMap.codRestrictOfInjective f i hi hf x) = f x := by
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₃ : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M₁ M₂
    i : LinearMap (RingHom.id R) M₃ M₂
    hi : Function.Injective ⇑i
    hf : ∀ (x : M₁), Membership.mem (LinearMap.range i) (f x)
    x : M₁
    ⊢ Eq (i ((f.codRestrictOfInjective i hi hf) x)) (f x)
  -/
  simp [LinearMap.codRestrictOfInjective]
  /-
    🎉 no goals
  -/


@[simp]
lemma codRestrictOfInjective_comp :
    i ∘ₗ LinearMap.codRestrictOfInjective f i hi hf = f := by
  /-
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₃ : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M₁ M₂
    i : LinearMap (RingHom.id R) M₃ M₂
    hi : Function.Injective ⇑i
    hf : ∀ (x : M₁), Membership.mem (LinearMap.range i) (f x)
    ⊢ Eq (i.comp (f.codRestrictOfInjective i hi hf)) f
  -/
  ext
  /-
    case h
    R : Type u_1
    M₁ : Type u_6
    M₂ : Type u_7
    M₃ : Type u_8
    inst✝⁶ : CommSemiring R
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M₃
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M₁ M₂
    i : LinearMap (RingHom.id R) M₃ M₂
    hi : Function.Injective ⇑i
    hf : ∀ (x : M₁), Membership.mem (LinearMap.range i) (f x)
    x✝ : M₁
    ⊢ Eq ((i.comp (f.codRestrictOfInjective i hi hf)) x✝) (f x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The restriction of a bilinear map to a submodule in which it takes values. -/
noncomputable def codRestrict₂ :
    M₁ →ₗ[R] M₂ →ₗ[R] M₃ :=
  let e : LinearMap.range i ≃ₗ[R] M₃ := (LinearEquiv.ofInjective i hi).symm
  { toFun := fun x ↦ e.comp <| (f x).codRestrict (p := LinearMap.range i) (hf x)
                   /-
                     R : Type u_1
                     R₁ : Type u_2
                     R₂ : Type u_3
                     R₃ : Type u_4
                     M : Type u_5
                     M₁ : Type u_6
                     M₂ : Type u_7
                     M₃ : Type u_8
                     N : Type u_9
                     inst✝⁸ : CommSemiring R
                     inst✝⁷ : AddCommMonoid M
                     inst✝⁶ : AddCommMonoid M₁
                     inst✝⁵ : AddCommMonoid M₂
                     inst✝⁴ : AddCommMonoid M₃
                     inst✝³ : Module R M
                     inst✝² : Module R M₁
                     inst✝¹ : Module R M₂
                     inst✝ : Module R M₃
                     f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
                     i : LinearMap (RingHom.id R) M₃ M
                     hi : Function.Injective ⇑i
                     hf : ∀ (x : M₁) (y : M₂), Membership.mem (LinearMap.range i) ((f x) y)
                     e : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (LinearMap.ran …
                     ⊢ ∀ (x y : M₁), Eq ((fun x => (↑e).comp (LinearMap.codRestrict (LinearMap.rang …
                   -/
    map_add' := by intro x₁ x₂; ext y; simp [f.map_add, ← e.map_add, codRestrict]
                                       /-
                                         🎉 no goals
                                       -/
                    /-
                      R : Type u_1
                      R₁ : Type u_2
                      R₂ : Type u_3
                      R₃ : Type u_4
                      M : Type u_5
                      M₁ : Type u_6
                      M₂ : Type u_7
                      M₃ : Type u_8
                      N : Type u_9
                      inst✝⁸ : CommSemiring R
                      inst✝⁷ : AddCommMonoid M
                      inst✝⁶ : AddCommMonoid M₁
                      inst✝⁵ : AddCommMonoid M₂
                      inst✝⁴ : AddCommMonoid M₃
                      inst✝³ : Module R M
                      inst✝² : Module R M₁
                      inst✝¹ : Module R M₂
                      inst✝ : Module R M₃
                      f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
                      i : LinearMap (RingHom.id R) M₃ M
                      hi : Function.Injective ⇑i
                      hf : ∀ (x : M₁) (y : M₂), Membership.mem (LinearMap.range i) ((f x) y)
                      e : LinearEquiv (RingHom.id R) (Subtype fun x => Membership.mem (LinearMap.ran …
                      ⊢ ∀ (m : R) (x : M₁), Eq ({ toFun := fun x => (↑e).comp (LinearMap.codRestrict …
                    -/
    map_smul' := by intro t x; ext y; simp [f.map_smul, ← e.map_smul, codRestrict] }
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
lemma codRestrict₂_apply (x : M₁) (y : M₂) :
    i (codRestrict₂ f i hi hf x y) = f x y := by
  /-
    R : Type u_1
    M : Type u_5
    M₁ : Type u_6
    M₂ : Type u_7
    M₃ : Type u_8
    inst✝⁸ : CommSemiring R
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : AddCommMonoid M₁
    inst✝⁵ : AddCommMonoid M₂
    inst✝⁴ : AddCommMonoid M₃
    inst✝³ : Module R M
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M₃
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    i : LinearMap (RingHom.id R) M₃ M
    hi : Function.Injective ⇑i
    hf : ∀ (x : M₁) (y : M₂), Membership.mem (LinearMap.range i) ((f x) y)
    x : M₁
    y : M₂
    ⊢ Eq (i (((f.codRestrict₂ i hi hf) x) y)) ((f x) y)
  -/
  simp [codRestrict₂]
  /-
    🎉 no goals
  -/


